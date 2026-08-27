"""Server-side waveform and spectrogram generation for separated audio."""

from pathlib import Path

import matplotlib

# Flask may call this module without a display server. The backend must be set
# before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .audio_utils import load_audio
from .stft_utils import DEFAULT_HOP_LENGTH, DEFAULT_N_FFT, magnitude, stft


DEFAULT_MAX_WAVEFORM_POINTS = 4_000
DEFAULT_MIN_DB = -80.0
VISUALIZATION_NAMES = ("original", "vocals", "instrumental")


def _as_visualization_mono(audio: np.ndarray) -> np.ndarray:
    """Return a mono plotting signal without changing any audio output file.

    Multichannel audio is averaged across channels solely to make one clean,
    comparable plot per source. The separator's stereo files remain untouched.
    """

    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.ndim == 1:
        return audio_array
    if audio_array.ndim == 2:
        return np.mean(audio_array, axis=0, dtype=np.float32)
    raise ValueError("audio must be mono or channel-first multichannel data")


def _waveform_envelope(
    signal: np.ndarray,
    sample_rate: int,
    max_points: int = DEFAULT_MAX_WAVEFORM_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a long waveform to a min/max envelope for efficient plotting."""

    if max_points < 2:
        raise ValueError("max_points must be at least 2")

    sample_count = signal.size
    if sample_count <= max_points:
        return np.arange(sample_count) / sample_rate, signal

    target_block_count = max_points // 2
    block_size = int(np.ceil(sample_count / target_block_count))
    block_count = int(np.ceil(sample_count / block_size))
    padded_size = block_count * block_size
    padded = np.pad(
        signal,
        (0, padded_size - sample_count),
        mode="constant",
        constant_values=np.nan,
    ).reshape(block_count, block_size)

    minimum = np.nanmin(padded, axis=1)
    maximum = np.nanmax(padded, axis=1)
    centers = np.minimum(
        (np.arange(block_count) * block_size) + (block_size - 1) / 2,
        sample_count - 1,
    ) / sample_rate

    times = np.repeat(centers, 2)
    amplitudes = np.column_stack((minimum, maximum)).reshape(-1)
    return times, amplitudes


def _save_waveform(
    signal: np.ndarray,
    sample_rate: int,
    output_path: Path,
    max_points: int,
) -> None:
    times, amplitudes = _waveform_envelope(signal, sample_rate, max_points)
    figure, axis = plt.subplots(figsize=(10, 3), constrained_layout=True)
    try:
        axis.plot(times, amplitudes, color="#2563eb", linewidth=0.8)
        axis.axhline(0, color="#94a3b8", linewidth=0.6)
        axis.set_xlim(0, signal.size / sample_rate)
        axis.set_ylim(-1.05, 1.05)
        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Amplitude")
        axis.grid(True, color="#e2e8f0", linewidth=0.5, alpha=0.8)
        figure.savefig(output_path, dpi=120, format="png", facecolor="white")
    finally:
        plt.close(figure)


def _save_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    output_path: Path,
    n_fft: int,
    hop_length: int,
    min_db: float,
) -> None:
    spectrum = stft(
        signal,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    spectrogram_magnitude = magnitude(spectrum)
    reference = max(float(np.max(spectrogram_magnitude)), np.finfo(np.float64).eps)
    magnitude_db = 20.0 * np.log10(
        np.maximum(spectrogram_magnitude, np.finfo(np.float64).eps) / reference
    )
    magnitude_db = np.maximum(magnitude_db, min_db)

    duration = signal.size / sample_rate
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    try:
        image = axis.imshow(
            magnitude_db,
            origin="lower",
            aspect="auto",
            extent=(0, duration, 0, sample_rate / 2),
            vmin=min_db,
            vmax=0,
            cmap="magma",
            interpolation="nearest",
        )
        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Frequency (Hz)")
        colorbar = figure.colorbar(image, ax=axis, pad=0.015)
        colorbar.set_label("Magnitude (dB)")
        figure.savefig(output_path, dpi=120, format="png", facecolor="white")
    finally:
        plt.close(figure)


def generate_audio_visualizations(
    audio_path: str | Path,
    output_directory: str | Path,
    name: str,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    max_waveform_points: int = DEFAULT_MAX_WAVEFORM_POINTS,
    min_db: float = DEFAULT_MIN_DB,
) -> dict[str, Path]:
    """Generate waveform and real Hann-STFT spectrogram PNGs for one file."""

    if name not in VISUALIZATION_NAMES:
        raise ValueError(f"unsupported visualization name: {name}")

    audio, sample_rate = load_audio(audio_path, preserve_channels=True)
    plotting_signal = _as_visualization_mono(audio)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    waveform_path = destination / f"{name}_waveform.png"
    spectrogram_path = destination / f"{name}_spectrogram.png"
    _save_waveform(
        plotting_signal,
        sample_rate,
        waveform_path,
        max_waveform_points,
    )
    _save_spectrogram(
        plotting_signal,
        sample_rate,
        spectrogram_path,
        n_fft,
        hop_length,
        min_db,
    )
    return {"waveform": waveform_path, "spectrogram": spectrogram_path}


def generate_separation_visualizations(
    original_path: str | Path,
    vocals_path: str | Path,
    instrumental_path: str | Path,
    output_directory: str | Path,
) -> dict[str, dict[str, Path]]:
    """Generate all six visualization files for a completed separation job."""

    audio_paths = {
        "original": original_path,
        "vocals": vocals_path,
        "instrumental": instrumental_path,
    }
    return {
        name: generate_audio_visualizations(path, output_directory, name)
        for name, path in audio_paths.items()
    }
