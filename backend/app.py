from pathlib import Path
import mimetypes
import shutil
from string import hexdigits
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from .dsp.audio_utils import AudioLoadError, extract_audio_info
    from .dsp.audio_visualization import generate_separation_visualizations
    from .dsp.source_separation import separate_audio_file
except ImportError:
    # Allows both `python app.py` and package-style imports during tests.
    from dsp.audio_utils import AudioLoadError, extract_audio_info
    from dsp.audio_visualization import generate_separation_visualizations
    from dsp.source_separation import separate_audio_file

try:
    from .separation.openunmix_dsp_separator import (
        separate_audio_file as separate_music_audio_file,
    )
except ImportError:
    from separation.openunmix_dsp_separator import (
        separate_audio_file as separate_music_audio_file,
    )


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
ALLOWED_OUTPUT_FILES = {
    "vocals.wav",
    "instrumental.wav",
    "original_waveform.png",
    "original_spectrogram.png",
    "vocals_waveform.png",
    "vocals_spectrogram.png",
    "instrumental_waveform.png",
    "instrumental_spectrogram.png",
}
DEFAULT_UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
DEFAULT_OUTPUT_FOLDER = Path(__file__).resolve().parent / "outputs"


ALLOWED_ORIGINAL_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS


class UploadValidationError(ValueError):
    """Raised when a multipart upload does not contain supported audio."""


def _store_uploaded_audio(uploaded_file, upload_folder: str | Path) -> tuple[Path, str]:
    if uploaded_file is None:
        raise UploadValidationError("No audio file was provided.")

    original_filename = uploaded_file.filename or ""
    safe_filename = secure_filename(original_filename)
    display_filename = Path(original_filename.replace("\\", "/")).name

    if not safe_filename:
        raise UploadValidationError("No audio file was selected.")

    extension = Path(safe_filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise UploadValidationError(
            "Unsupported file type. Please upload WAV, MP3, or FLAC audio."
        )

    destination = Path(upload_folder)
    destination.mkdir(parents=True, exist_ok=True)
    stored_path = destination / f"{uuid4().hex}{extension}"
    uploaded_file.save(stored_path)
    return stored_path, display_filename


def _is_job_identifier(value: str) -> bool:
    return len(value) == 32 and all(character in hexdigits for character in value)


def _is_allowed_output_filename(filename: str) -> bool:
    if filename in ALLOWED_OUTPUT_FILES:
        return True
    return (
        filename.startswith('original.')
        and Path(filename).name == filename
        and Path(filename).suffix.lower() in ALLOWED_ORIGINAL_EXTENSIONS
    )


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        UPLOAD_FOLDER=DEFAULT_UPLOAD_FOLDER,
        OUTPUT_FOLDER=DEFAULT_OUTPUT_FOLDER,
    )

    if test_config:
        app.config.update(test_config)

    CORS(app)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.post("/api/audio-info")
    def audio_info():
        try:
            stored_path, display_filename = _store_uploaded_audio(
                request.files.get("file"),
                app.config["UPLOAD_FOLDER"],
            )
        except UploadValidationError as error:
            return jsonify(success=False, error=str(error)), 400

        try:
            metadata = extract_audio_info(stored_path, filename=display_filename)
        except AudioLoadError as error:
            stored_path.unlink(missing_ok=True)
            return jsonify(success=False, error=str(error)), 400

        return jsonify(success=True, **metadata)

    @app.post("/api/separate")
    def separate_audio():
        try:
            stored_path, display_filename = _store_uploaded_audio(
                request.files.get("file"),
                app.config["UPLOAD_FOLDER"],
            )
        except UploadValidationError as error:
            return jsonify(success=False, error=str(error)), 400

        job_identifier = uuid4().hex
        output_directory = Path(app.config["OUTPUT_FOLDER"]) / job_identifier

        try:
            original_info = extract_audio_info(stored_path, filename=display_filename)
            result = separate_music_audio_file(
                stored_path,
                output_directory=output_directory,
                device='auto',
            )
            output_paths = result.get('paths', {})
            for output_name in ('vocals', 'instrumental'):
                output_path = Path(output_paths.get(output_name, ''))
                if output_path.resolve().parent != output_directory.resolve() or not output_path.is_file():
                    raise RuntimeError(f'missing separator output: {output_name}')
            original_name = f'original{stored_path.suffix.lower()}'
            original_path = output_directory / original_name
            shutil.copyfile(stored_path, original_path)
            generate_separation_visualizations(
                original_path,
                output_paths['vocals'],
                output_paths['instrumental'],
                output_directory,
            )
        except (AudioLoadError, ValueError) as error:
            stored_path.unlink(missing_ok=True)
            return jsonify(success=False, error=str(error)), 400
        except Exception:
            stored_path.unlink(missing_ok=True)
            app.logger.exception('music separation failed')
            return jsonify(success=False, error='Music separation failed.'), 500

        stored_path.unlink(missing_ok=True)

        vocal_url = url_for(
            "serve_separation_output",
            job_identifier=job_identifier,
            filename="vocals.wav",
        )
        instrumental_url = url_for(
            "serve_separation_output",
            job_identifier=job_identifier,
            filename="instrumental.wav",
        )
        original_url = url_for(
            "serve_separation_output",
            job_identifier=job_identifier,
            filename=original_name,
        )
        original_info["audio_url"] = original_url

        def visualization_url(source_name: str, visualization_name: str) -> str:
            return url_for(
                "serve_separation_output",
                job_identifier=job_identifier,
                filename=f"{source_name}_{visualization_name}.png",
            )

        return jsonify(
            success=True,
            original={
                **original_info,
                "waveform_url": visualization_url("original", "waveform"),
                "spectrogram_url": visualization_url("original", "spectrogram"),
            },
            vocals={
                "audio_url": vocal_url,
                "download_url": f"{vocal_url}?download=1",
                "waveform_url": visualization_url("vocals", "waveform"),
                "spectrogram_url": visualization_url("vocals", "spectrogram"),
            },
            instrumental={
                "audio_url": instrumental_url,
                "download_url": f"{instrumental_url}?download=1",
                "waveform_url": visualization_url("instrumental", "waveform"),
                "spectrogram_url": visualization_url("instrumental", "spectrogram"),
            },
            # Keep the legacy fields for existing clients.
            vocal_audio_url=vocal_url,
            instrumental_audio_url=instrumental_url,
        )

    @app.get("/api/outputs/<job_identifier>/<filename>")
    def serve_separation_output(job_identifier: str, filename: str):
        if not _is_job_identifier(job_identifier) or not _is_allowed_output_filename(filename):
            return jsonify(success=False, error="Audio output was not found."), 404

        output_directory = Path(app.config["OUTPUT_FOLDER"]) / job_identifier
        output_path = output_directory / filename
        if not output_path.is_file():
            return jsonify(success=False, error="Audio output was not found."), 404

        mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        return send_from_directory(
            output_directory,
            filename,
            mimetype=mime_type,
            as_attachment=request.args.get("download") == "1",
            download_name=filename,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
