"""Classical frequency-domain Wiener enhancement for noisy speech.

Mono speech is processed directly. Stereo speech retains separate channel
spectra and uses a shared set of low-energy time frames for noise estimation,
preserving channel count, spatial differences, sample rate, and duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import soundfile as sf

try:
    from ..dsp.audio_utils import AudioLoadError, load_audio
    from ..dsp.stft_utils import istft, magnitude, stft
except ImportError:
    from dsp.audio_utils import AudioLoadError, load_audio
    from dsp.stft_utils import istft, magnitude, stft


@dataclass(frozen=True)
class SpeechEnhancementConfig:
    """Configuration for explainable short-time Wiener filtering."""

    frame_duration_ms: float = 25.0
    overlap_ratio: float = 0.75
    noise_estimation: str = "auto"
    noise_frame_percent: float = 0.15
    minimum_noise_frames: int = 8
    noise_duration_seconds: float = 0.5
    noise_percentile: float = 50.0
    min_gain: float = 0.10
    frequency_smoothing_bins: int = 3
    temporal_smoothing_frames: int = 3
    epsilon: float = 1e-10
    peak_limit: float = 0.99

    def __post_init__(self) -> None:
        if not np.isfinite(self.frame_duration_ms) or not 5 <= self.frame_duration_ms <= 100:
            raise ValueError("frame_duration_ms must be between 5 and 100 ms")
        if not np.isfinite(self.overlap_ratio) or not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in the interval [0, 1)")
        if self.noise_estimation not in {"auto", "initial"}:
            raise ValueError("noise_estimation must be 'auto' or 'initial'")
        if not np.isfinite(self.noise_frame_percent) or not 0 < self.noise_frame_percent <= 1:
            raise ValueError("noise_frame_percent must be in the interval (0, 1]")
        if not isinstance(self.minimum_noise_frames, (int, np.integer)) or self.minimum_noise_frames < 1:
            raise ValueError("minimum_noise_frames must be a positive integer")
        if not np.isfinite(self.noise_duration_seconds) or self.noise_duration_seconds <= 0:
            raise ValueError("noise_duration_seconds must be positive")
        if not np.isfinite(self.noise_percentile) or not 0 <= self.noise_percentile <= 100:
            raise ValueError("noise_percentile must be between 0 and 100")
        if not np.isfinite(self.min_gain) or not 0 <= self.min_gain <= 1:
            raise ValueError("min_gain must be in the interval [0, 1]")
        for name, value in (
            ("frequency_smoothing_bins", self.frequency_smoothing_bins),
            ("temporal_smoothing_frames", self.temporal_smoothing_frames),
        ):
            if not isinstance(value, (int, np.integer)) or value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not np.isfinite(self.peak_limit) or not 0 < self.peak_limit <= 1:
            raise ValueError("peak_limit must be in the interval (0, 1]")


def speech_stft_parameters(
    sample_rate: int,
    config: SpeechEnhancementConfig,
) -> tuple[int, int]:
    """Return sample-rate-aware frame and hop lengths.

    The default is a 25 ms frame with 75% overlap. Unlike the music model's
    fixed 4096-point transform, this keeps speech time resolution comparable
    at common 8, 16, 44.1, and 48 kHz sample rates.
    """

    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    frame_length = max(2, int(round(sample_rate * config.frame_duration_ms / 1000.0)))
    hop_length = max(1, int(round(frame_length * (1.0 - config.overlap_ratio))))
    return frame_length, min(hop_length, frame_length)


def _validate_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, bool]:
    values = np.asarray(audio)
    if values.ndim == 1:
        channel_first = values[np.newaxis, :]
        was_mono_vector = True
    elif values.ndim == 2 and values.shape[0] in (1, 2):
        channel_first = values
        was_mono_vector = False
    else:
        raise ValueError("audio must be mono or channel-first stereo")
    if channel_first.shape[-1] == 0:
        raise ValueError("audio must contain at least one sample")
    if np.iscomplexobj(channel_first) or not np.isfinite(channel_first).all():
        raise ValueError("audio must contain finite real samples")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return np.ascontiguousarray(channel_first, dtype=np.float64), was_mono_vector


def estimate_noise_power(
    complex_spectrum: np.ndarray,
    start_frame: int,
    stop_frame: int,
    percentile: float = 50.0,
) -> np.ndarray:
    """Estimate noise power per frequency using an early-frame percentile.

    A low percentile is less influenced by brief speech energy than a simple
    mean, but continuous speech or changing noise during the initial interval
    can still make the estimate inaccurate.
    """

    spectrum = np.asarray(complex_spectrum)
    if spectrum.ndim != 2 or spectrum.shape[1] == 0:
        raise ValueError("complex_spectrum must be a non-empty 2-D STFT")
    if not np.isfinite(spectrum).all():
        raise ValueError("complex_spectrum must contain finite values")
    if not isinstance(start_frame, (int, np.integer)) or not isinstance(
        stop_frame, (int, np.integer)
    ):
        raise ValueError("noise frame bounds must be integers")
    if start_frame < 0 or stop_frame <= start_frame or stop_frame > spectrum.shape[1]:
        raise ValueError("noise frame bounds are outside the STFT")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    initial_power = magnitude(spectrum[:, start_frame:stop_frame]) ** 2
    noise_power = np.percentile(initial_power, percentile, axis=1, keepdims=True)
    return np.maximum(np.asarray(noise_power, dtype=np.float64), 0.0)


def estimate_noise_power_from_frames(
    complex_spectrum: np.ndarray,
    frame_indices: np.ndarray,
    percentile: float = 50.0,
) -> np.ndarray:
    """Estimate per-frequency noise power from selected STFT frames."""

    spectrum = np.asarray(complex_spectrum)
    indices = np.asarray(frame_indices)
    if spectrum.ndim != 2 or spectrum.shape[1] == 0:
        raise ValueError("complex_spectrum must be a non-empty 2-D STFT")
    if not np.isfinite(spectrum).all():
        raise ValueError("complex_spectrum must contain finite values")
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("frame_indices must be a non-empty integer vector")
    if np.any(indices < 0) or np.any(indices >= spectrum.shape[1]):
        raise ValueError("frame_indices are outside the STFT")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    selected_power = magnitude(spectrum[:, indices]) ** 2
    noise_power = np.percentile(selected_power, percentile, axis=1, keepdims=True)
    return np.maximum(np.asarray(noise_power, dtype=np.float64), 0.0)


def select_low_energy_noise_frames(
    frame_energy: np.ndarray,
    frame_percent: float = 0.15,
    minimum_frames: int = 8,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return the lowest-energy candidate frame indices in time order."""

    energy = np.asarray(frame_energy, dtype=np.float64)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("frame_energy must be a non-empty vector")
    if not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("frame_energy must contain finite non-negative values")
    if not np.isfinite(frame_percent) or not 0 < frame_percent <= 1:
        raise ValueError("frame_percent must be in the interval (0, 1]")
    if not isinstance(minimum_frames, (int, np.integer)) or minimum_frames < 1:
        raise ValueError("minimum_frames must be a positive integer")

    if candidate_indices is None:
        candidates = np.arange(energy.size, dtype=np.int64)
    else:
        candidates = np.asarray(candidate_indices)
        if candidates.ndim != 1 or candidates.size == 0 or not np.issubdtype(candidates.dtype, np.integer):
            raise ValueError("candidate_indices must be a non-empty integer vector")
        if np.any(candidates < 0) or np.any(candidates >= energy.size):
            raise ValueError("candidate_indices are outside frame_energy")
        candidates = candidates.astype(np.int64, copy=False)

    selected_count = min(
        candidates.size,
        max(int(minimum_frames), int(np.ceil(candidates.size * frame_percent))),
    )
    candidate_energy = energy[candidates]
    local_indices = np.argpartition(candidate_energy, selected_count - 1)[:selected_count]
    return np.sort(candidates[local_indices])


def compute_wiener_gain(
    noisy_power: np.ndarray,
    noise_power: np.ndarray,
    *,
    min_gain: float = 0.10,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Return stable Wiener gains from noisy and estimated-noise power.

    Estimated clean power is ``max(P_y - P_n, 0)`` and the gain is
    ``P_clean / (P_clean + P_n + epsilon)``. A small gain floor reduces hard
    spectral holes and musical-noise artifacts; gains can never exceed one.
    """

    noisy = np.asarray(noisy_power, dtype=np.float64)
    noise = np.asarray(noise_power, dtype=np.float64)
    if noisy.ndim != 2 or noise.ndim not in (1, 2):
        raise ValueError("power spectra must be two-dimensional or broadcastable")
    if np.any(noisy < 0) or np.any(noise < 0):
        raise ValueError("power spectra must be non-negative")
    if not np.isfinite(noisy).all() or not np.isfinite(noise).all():
        raise ValueError("power spectra must contain finite values")
    if noise.ndim == 1:
        if noise.shape[0] != noisy.shape[0]:
            raise ValueError("one-dimensional noise_power must match frequency bins")
        noise = noise[:, np.newaxis]
    if not np.isfinite(min_gain) or not 0 <= min_gain <= 1:
        raise ValueError("min_gain must be in the interval [0, 1]")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")

    try:
        clean_power = np.maximum(noisy - noise, 0.0)
        gain = clean_power / (clean_power + noise + epsilon)
    except ValueError as error:
        raise ValueError("noise_power is not broadcastable to noisy_power") from error
    gain = np.nan_to_num(gain, nan=min_gain, posinf=1.0, neginf=min_gain)
    return np.clip(gain, min_gain, 1.0)


def smooth_wiener_gain(
    gain: np.ndarray,
    *,
    frequency_bins: int = 3,
    temporal_frames: int = 3,
    min_gain: float = 0.10,
) -> np.ndarray:
    """Apply small triangular smoothers across frequency and time."""

    values = np.asarray(gain, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("gain must be a finite two-dimensional matrix")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("gain values must be in the interval [0, 1]")
    if not np.isfinite(min_gain) or not 0 <= min_gain <= 1:
        raise ValueError("min_gain must be in the interval [0, 1]")

    smoothed = values
    for axis, width, name in (
        (0, frequency_bins, "frequency_bins"),
        (1, temporal_frames, "temporal_frames"),
    ):
        if not isinstance(width, (int, np.integer)) or width < 1 or width % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")
        if width == 1:
            continue
        radius = width // 2
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (radius, radius)
        padded = np.pad(smoothed, pad_width, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=axis)
        weights = np.bartlett(width + 2)[1:-1]
        weights /= np.sum(weights)
        smoothed = np.tensordot(windows, weights, axes=([-1], [0]))

    return np.clip(smoothed, min_gain, 1.0)


def _noise_frame_bounds(
    frame_count: int,
    sample_count: int,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    requested_seconds: float,
) -> tuple[int, int, float]:
    used_seconds = min(requested_seconds, sample_count / sample_rate)
    stop_frame = min(
        frame_count,
        max(1, int(np.floor(used_seconds * sample_rate / hop_length)) + 1),
    )
    # Skip centered boundary frames when enough early audio exists, avoiding a
    # downward noise bias from zero padding at the start of the recording.
    first_complete_frame = int(np.ceil((frame_length / 2) / hop_length))
    start_frame = first_complete_frame if stop_frame > first_complete_frame else 0
    if stop_frame <= start_frame:
        stop_frame = min(frame_count, start_frame + 1)
    return start_frame, stop_frame, used_seconds


def enhance_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: SpeechEnhancementConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Enhance mono or channel-first stereo speech with classical Wiener DSP."""

    used_config = config or SpeechEnhancementConfig()
    channels, was_mono_vector = _validate_audio(audio, sample_rate)
    frame_length, hop_length = speech_stft_parameters(sample_rate, used_config)
    started = perf_counter()
    spectra = [
        stft(
            channel,
            sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
        )
        for channel in channels
    ]
    noisy_powers = [magnitude(spectrum) ** 2 for spectrum in spectra]
    frame_count = int(spectra[0].shape[1])

    noise_duration_used: float | None = None
    if used_config.noise_estimation == "auto":
        # The mean spectral power is proportional to the Hann-windowed frame
        # energy. Stereo channels are combined only for frame ranking; their
        # spectra and per-frequency noise estimates remain separate.
        combined_frame_energy = np.mean(
            np.stack(noisy_powers, axis=0),
            axis=(0, 1),
        )
        boundary_frames = int(np.ceil((frame_length / 2) / hop_length))
        candidate_start = min(boundary_frames, frame_count)
        candidate_stop = max(candidate_start, frame_count - boundary_frames)
        complete_frames = np.arange(candidate_start, candidate_stop, dtype=np.int64)
        if complete_frames.size == 0:
            complete_frames = np.arange(frame_count, dtype=np.int64)
        noise_frame_indices = select_low_energy_noise_frames(
            combined_frame_energy,
            frame_percent=used_config.noise_frame_percent,
            minimum_frames=used_config.minimum_noise_frames,
            candidate_indices=complete_frames,
        )
    else:
        start_frame, stop_frame, noise_duration_used = _noise_frame_bounds(
            frame_count,
            channels.shape[1],
            sample_rate,
            frame_length,
            hop_length,
            used_config.noise_duration_seconds,
        )
        noise_frame_indices = np.arange(start_frame, stop_frame, dtype=np.int64)

    reconstructed_channels: list[np.ndarray] = []
    gain_minimum = 1.0
    gain_maximum = 0.0
    gain_sum = 0.0
    gain_value_count = 0
    attenuated_over_3_db = 0
    attenuated_over_6_db = 0

    for complex_spectrum, noisy_power in zip(spectra, noisy_powers):
        noise_power = estimate_noise_power_from_frames(
            complex_spectrum,
            noise_frame_indices,
            percentile=used_config.noise_percentile,
        )
        gain = compute_wiener_gain(
            noisy_power,
            noise_power,
            min_gain=used_config.min_gain,
            epsilon=used_config.epsilon,
        )
        gain = smooth_wiener_gain(
            gain,
            frequency_bins=used_config.frequency_smoothing_bins,
            temporal_frames=used_config.temporal_smoothing_frames,
            min_gain=used_config.min_gain,
        )
        gain_minimum = min(gain_minimum, float(np.min(gain)))
        gain_maximum = max(gain_maximum, float(np.max(gain)))
        gain_sum += float(np.sum(gain))
        gain_value_count += int(gain.size)
        attenuated_over_3_db += int(np.count_nonzero(gain < 10 ** (-3.0 / 20.0)))
        attenuated_over_6_db += int(np.count_nonzero(gain < 10 ** (-6.0 / 20.0)))
        enhanced_spectrum = gain * complex_spectrum
        reconstructed = istft(
            enhanced_spectrum,
            sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            length=channels.shape[1],
        )
        reconstructed_channels.append(reconstructed)

    enhanced = np.stack(reconstructed_channels, axis=0)
    input_peak = float(np.max(np.abs(channels)))
    pre_limit_output_peak = float(np.max(np.abs(enhanced)))
    shared_gain = 1.0
    if pre_limit_output_peak > used_config.peak_limit:
        shared_gain = used_config.peak_limit / pre_limit_output_peak
        enhanced *= shared_gain
    output_peak = float(np.max(np.abs(enhanced)))
    if not np.isfinite(enhanced).all():
        raise RuntimeError("speech enhancement produced non-finite samples")

    noise_frame_count = int(noise_frame_indices.size)
    mean_wiener_gain = gain_sum / gain_value_count
    input_rms = float(np.sqrt(np.mean(channels**2)))
    output_rms = float(np.sqrt(np.mean(enhanced**2)))

    diagnostics: dict[str, Any] = {
        "sample_rate": int(sample_rate),
        "channels": int(channels.shape[0]),
        "samples": int(channels.shape[1]),
        "duration_seconds": channels.shape[1] / sample_rate,
        "stft_size": int(frame_length),
        "hop_length": int(hop_length),
        "overlap_ratio": float(used_config.overlap_ratio),
        "frames": frame_count,
        "noise_estimation": used_config.noise_estimation,
        "noise_frames": noise_frame_count,
        "noise_frame_percent": float(used_config.noise_frame_percent),
        "noise_frames_percent": 100.0 * noise_frame_count / frame_count,
        "noise_estimation_duration_seconds": noise_duration_used,
        "noise_percentile": float(used_config.noise_percentile),
        "wiener_gain_min": gain_minimum,
        "wiener_gain_max": gain_maximum,
        "mean_wiener_gain": mean_wiener_gain,
        "bins_attenuated_over_3_db_percent": 100.0 * attenuated_over_3_db / gain_value_count,
        "bins_attenuated_over_6_db_percent": 100.0 * attenuated_over_6_db / gain_value_count,
        "frequency_smoothing_bins": int(used_config.frequency_smoothing_bins),
        "temporal_smoothing_frames": int(used_config.temporal_smoothing_frames),
        "input_rms": input_rms,
        "output_rms": output_rms,
        "input_peak": input_peak,
        "pre_limit_output_peak": pre_limit_output_peak,
        "output_peak": output_peak,
        "shared_output_gain": shared_gain,
        "processing_seconds": perf_counter() - started,
    }
    result = np.asarray(enhanced, dtype=np.float32)
    if was_mono_vector:
        result = result[0]
    return result, diagnostics


def enhance_audio_file(
    input_path: str | Path,
    output_directory: str | Path,
    config: SpeechEnhancementConfig | None = None,
) -> dict[str, Any]:
    """Enhance one WAV and write original_noisy.wav and enhanced_speech.wav."""

    source = Path(input_path)
    if source.suffix.lower() != ".wav":
        raise AudioLoadError("Speech enhancement currently supports WAV input only.")
    audio, sample_rate = load_audio(source, preserve_channels=True)
    enhanced, diagnostics = enhance_audio(audio, sample_rate, config)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    original_path = destination / "original_noisy.wav"
    enhanced_path = destination / "enhanced_speech.wav"

    original_for_write = audio if audio.ndim == 1 else audio.T
    enhanced_for_write = enhanced if enhanced.ndim == 1 else enhanced.T
    sf.write(original_path, original_for_write, sample_rate, subtype="PCM_16")
    sf.write(enhanced_path, enhanced_for_write, sample_rate, subtype="PCM_16")
    return {
        **diagnostics,
        "input_path": source,
        "paths": {
            "original_noisy": original_path,
            "enhanced_speech": enhanced_path,
        },
    }
