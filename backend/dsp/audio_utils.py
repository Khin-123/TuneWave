"""Audio loading and metadata helpers shared by TuneWave DSP modules."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class AudioLoadError(ValueError):
    """Raised when an uploaded file cannot be decoded as audio."""


def load_audio(
    file_path: str | Path,
    preserve_channels: bool = True,
) -> tuple[np.ndarray, int]:
    """Load audio without resampling.

    Mono audio is returned as ``(samples,)``. When ``preserve_channels`` is
    true, multichannel audio is returned as ``(channels, samples)`` so stereo
    left and right channels remain separate for future DSP operations.
    """

    path = Path(file_path)

    try:
        # soundfile reads audio as (samples, channels). always_2d makes the
        # channel conversion below consistent for both mono and stereo files.
        samples_by_channel, sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=True,
        )
        channel_first = np.ascontiguousarray(samples_by_channel.T)
    except (sf.LibsndfileError, RuntimeError, OSError) as soundfile_error:
        # librosa provides a fallback decoder for formats that the installed
        # libsndfile build cannot read directly.
        try:
            channel_first, sample_rate = librosa.load(
                path,
                sr=None,
                mono=False,
                dtype=np.float32,
            )
            channel_first = np.asarray(channel_first, dtype=np.float32)
            if channel_first.ndim == 1:
                channel_first = channel_first[np.newaxis, :]
        except Exception as librosa_error:
            raise AudioLoadError("The file could not be decoded as audio.") from librosa_error

        if channel_first.size == 0:
            raise AudioLoadError("The audio file contains no samples.") from soundfile_error

    if channel_first.size == 0 or channel_first.shape[-1] == 0:
        raise AudioLoadError("The audio file contains no samples.")

    if sample_rate <= 0:
        raise AudioLoadError("The audio file has an invalid sample rate.")

    if not np.isfinite(channel_first).all():
        raise AudioLoadError("The audio file contains invalid sample values.")

    if preserve_channels:
        audio = channel_first[0] if channel_first.shape[0] == 1 else channel_first
    else:
        audio = librosa.to_mono(channel_first)

    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def extract_audio_info(file_path: str | Path, filename: str | None = None) -> dict:
    """Load an audio file and return JSON-ready basic metadata."""

    audio, sample_rate = load_audio(file_path, preserve_channels=True)
    channels = 1 if audio.ndim == 1 else int(audio.shape[0])
    sample_count = int(audio.shape[-1])

    return {
        "filename": filename or Path(file_path).name,
        "sample_rate": sample_rate,
        # Duration is exactly the samples-per-channel divided by sample rate.
        # Computing it directly avoids unnecessary feature-analysis overhead.
        "duration": sample_count / sample_rate,
        "channels": channels,
        "samples": sample_count,
    }
