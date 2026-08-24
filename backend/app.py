from pathlib import Path
from string import hexdigits
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from .dsp.audio_utils import AudioLoadError, extract_audio_info
    from .dsp.source_separation import separate_audio_file
except ImportError:
    # Allows both `python app.py` and package-style imports during tests.
    from dsp.audio_utils import AudioLoadError, extract_audio_info
    from dsp.source_separation import separate_audio_file


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
ALLOWED_OUTPUT_FILES = {"vocals.wav", "instrumental.wav"}
DEFAULT_UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
DEFAULT_OUTPUT_FOLDER = Path(__file__).resolve().parent / "outputs"


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
            result = separate_audio_file(stored_path, output_directory=output_directory)
        except (AudioLoadError, ValueError) as error:
            stored_path.unlink(missing_ok=True)
            return jsonify(success=False, error=str(error)), 400

        original_info = {
            "filename": display_filename,
            "sample_rate": result["sample_rate"],
            "duration": result["samples"] / result["sample_rate"],
            "channels": result["channels"],
            "samples": result["samples"],
        }

        return jsonify(
            success=True,
            original=original_info,
            vocal_audio_url=url_for(
                "serve_separation_output",
                job_identifier=job_identifier,
                filename="vocals.wav",
            ),
            instrumental_audio_url=url_for(
                "serve_separation_output",
                job_identifier=job_identifier,
                filename="instrumental.wav",
            ),
        )

    @app.get("/api/outputs/<job_identifier>/<filename>")
    def serve_separation_output(job_identifier: str, filename: str):
        if not _is_job_identifier(job_identifier) or filename not in ALLOWED_OUTPUT_FILES:
            return jsonify(success=False, error="Audio output was not found."), 404

        output_directory = Path(app.config["OUTPUT_FOLDER"]) / job_identifier
        output_path = output_directory / filename
        if not output_path.is_file():
            return jsonify(success=False, error="Audio output was not found."), 404

        return send_from_directory(
            output_directory,
            filename,
            mimetype="audio/wav",
            as_attachment=request.args.get("download") == "1",
            download_name=filename,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
