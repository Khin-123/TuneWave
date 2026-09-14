"""Classical short-time spectral-subtraction enhancement for noisy speech.

The production speech path is deliberately model-free:

    framing -> Hann window -> FFT -> magnitude/phase -> low-energy noise
    estimate -> spectral subtraction -> spectral floor -> mild Wiener
    post-filter -> IFFT -> overlap-add -> level compensation

Audio is processed one channel at a time. The shared STFT helpers provide the
explicit FFT/IFFT and normalized overlap-add operations, while this module
contains the speech-specific noise model and spectral processing.
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
    from ..dsp.stft_utils import istft, magnitude, phase, stft
except ImportError:
    from dsp.audio_utils import AudioLoadError, load_audio
    from dsp.stft_utils import istft, magnitude, phase, stft


@dataclass(frozen=True)
class SpeechEnhancementConfig:
    """Configuration for the classical speech enhancement pipeline.

    ``noise_estimation='auto'`` is the final/default mode: frames are ranked
    over the whole recording and the lowest-energy frames form the noise
    model. ``'initial'`` remains accepted as a compatibility option for old
    callers, but is not used by the default workflow.

    ``min_gain``, ``noise_duration_seconds``, and the smoothing settings are
    retained as compatibility fields for existing callers. The production
    Wiener post-filter uses the explicitly named mild-Wiener settings below.
    """

    frame_duration_ms: float = 25.0
    overlap_ratio: float = 0.75
    beta: float = 1.5
    spectral_floor: float = 0.08
    noise_estimation: str = "auto"
    noise_frame_percent: float = 0.15
    minimum_noise_frames: int = 8
    noise_duration_seconds: float = 0.5
    noise_percentile: float = 50.0
    wiener_strength: float = 0.25
    wiener_min_gain: float = 0.65
    # Legacy public option retained for callers of the previous enhancer.
    min_gain: float = 0.10
    frequency_smoothing_bins: int = 3
    temporal_smoothing_frames: int = 3
    epsilon: float = 1e-10
    peak_limit: float = 0.99
    max_compensation_gain: float = 1.5
    # Adaptive processing controls appended for positional compatibility.
    adaptive_beta: bool = True
    beta_min: float = 0.8
    beta_max: float = 1.8
    beta_adaptation_strength: float = 0.5
    beta_low_snr_db: float = 0.0
    beta_high_snr_db: float = 15.0
    noise_tracking_alpha: float = 0.04
    noise_tracking_snr_db: float = 3.0
    noise_tracking_energy_percentile: float = 40.0
    speech_activity_margin_db: float = 6.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.frame_duration_ms) or not 5 <= self.frame_duration_ms <= 100:
            raise ValueError("frame_duration_ms must be between 5 and 100 ms")
        if not np.isfinite(self.overlap_ratio) or not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in the interval [0, 1)")
        if self.noise_estimation not in {"auto", "initial"}:
            raise ValueError("noise_estimation must be 'auto' or 'initial'")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be a finite non-negative number")
        if not np.isfinite(self.spectral_floor) or not 0 <= self.spectral_floor <= 1:
            raise ValueError("spectral_floor must be in the interval [0, 1]")
        if not np.isfinite(self.noise_frame_percent) or not 0 < self.noise_frame_percent <= 1:
            raise ValueError("noise_frame_percent must be in the interval (0, 1]")
        if not isinstance(self.minimum_noise_frames, (int, np.integer)) or self.minimum_noise_frames < 1:
            raise ValueError("minimum_noise_frames must be a positive integer")
        if not np.isfinite(self.noise_duration_seconds) or self.noise_duration_seconds <= 0:
            raise ValueError("noise_duration_seconds must be positive")
        if not np.isfinite(self.noise_percentile) or not 0 <= self.noise_percentile <= 100:
            raise ValueError("noise_percentile must be between 0 and 100")
        if not np.isfinite(self.wiener_strength) or not 0 <= self.wiener_strength <= 1:
            raise ValueError("wiener_strength must be in the interval [0, 1]")
        if not np.isfinite(self.wiener_min_gain) or not 0 <= self.wiener_min_gain <= 1:
            raise ValueError("wiener_min_gain must be in the interval [0, 1]")
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
        if not np.isfinite(self.max_compensation_gain) or self.max_compensation_gain < 1:
            raise ValueError("max_compensation_gain must be at least 1")
        if not isinstance(self.adaptive_beta, (bool, np.bool_)):
            raise ValueError("adaptive_beta must be boolean")
        if not np.isfinite(self.beta_min) or not 0.0 <= self.beta_min <= 1.8:
            raise ValueError("beta_min must be in the interval [0, 1.8]")
        if not np.isfinite(self.beta_max) or not self.beta_min <= self.beta_max <= 1.8:
            raise ValueError("beta_max must be >= beta_min and at most 1.8")
        if not np.isfinite(self.beta) or not self.beta_min <= self.beta <= self.beta_max:
            raise ValueError("beta must remain within beta_min and beta_max")
        if not np.isfinite(self.beta_adaptation_strength) or not 0 <= self.beta_adaptation_strength <= 1:
            raise ValueError("beta_adaptation_strength must be in the interval [0, 1]")
        if not np.isfinite(self.beta_low_snr_db) or not np.isfinite(self.beta_high_snr_db) or self.beta_high_snr_db <= self.beta_low_snr_db:
            raise ValueError("beta SNR bounds must be finite and ordered")
        if not np.isfinite(self.noise_tracking_alpha) or not 0 <= self.noise_tracking_alpha <= 1:
            raise ValueError("noise_tracking_alpha must be in the interval [0, 1]")
        if not np.isfinite(self.noise_tracking_snr_db):
            raise ValueError("noise_tracking_snr_db must be finite")
        if not np.isfinite(self.noise_tracking_energy_percentile) or not 0 < self.noise_tracking_energy_percentile <= 100:
            raise ValueError("noise_tracking_energy_percentile must be in the interval (0, 100]")
        if not np.isfinite(self.speech_activity_margin_db) or self.speech_activity_margin_db < 0:
            raise ValueError("speech_activity_margin_db must be non-negative")


def speech_stft_parameters(
    sample_rate: int,
    config: SpeechEnhancementConfig,
) -> tuple[int, int]:
    """Return sample-rate-aware frame and hop lengths.

    The default is a 25 ms frame with 75% overlap (a 5 ms hop). This keeps
    analysis time resolution appropriate for speech at common sample rates.
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


def _validate_spectrum(complex_spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.asarray(complex_spectrum)
    if spectrum.ndim != 2 or spectrum.shape[1] == 0:
        raise ValueError("complex_spectrum must be a non-empty 2-D STFT")
    if not np.isfinite(spectrum).all():
        raise ValueError("complex_spectrum must contain finite values")
    return spectrum


def _validate_frame_indices(
    complex_spectrum: np.ndarray,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spectrum = _validate_spectrum(complex_spectrum)
    indices = np.asarray(frame_indices)
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("frame_indices must be a non-empty integer vector")
    if np.any(indices < 0) or np.any(indices >= spectrum.shape[1]):
        raise ValueError("frame_indices are outside the STFT")
    return spectrum, indices.astype(np.int64, copy=False)


def estimate_noise_power(
    complex_spectrum: np.ndarray,
    start_frame: int,
    stop_frame: int,
    percentile: float = 50.0,
) -> np.ndarray:
    """Estimate per-frequency noise power from a selected frame interval.

    This legacy interval helper remains available for API compatibility. The
    enhancer itself uses selected low-energy frames from the full recording.
    """

    spectrum = _validate_spectrum(complex_spectrum)
    if not isinstance(start_frame, (int, np.integer)) or not isinstance(
        stop_frame, (int, np.integer)
    ):
        raise ValueError("noise frame bounds must be integers")
    if start_frame < 0 or stop_frame <= start_frame or stop_frame > spectrum.shape[1]:
        raise ValueError("noise frame bounds are outside the STFT")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    selected_magnitude = magnitude(spectrum[:, start_frame:stop_frame])
    noise_power = np.percentile(selected_magnitude**2, percentile, axis=1, keepdims=True)
    return np.maximum(np.asarray(noise_power, dtype=np.float64), 0.0)


def estimate_noise_magnitude_from_frames(
    complex_spectrum: np.ndarray,
    frame_indices: np.ndarray,
    percentile: float = 50.0,
) -> np.ndarray:
    """Estimate noise magnitude per frequency from selected STFT frames."""

    spectrum, indices = _validate_frame_indices(complex_spectrum, frame_indices)
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    selected_magnitude = magnitude(spectrum[:, indices])
    noise_magnitude = np.percentile(
        selected_magnitude,
        percentile,
        axis=1,
        keepdims=True,
    )
    return np.maximum(np.asarray(noise_magnitude, dtype=np.float64), 0.0)


def estimate_noise_power_from_frames(
    complex_spectrum: np.ndarray,
    frame_indices: np.ndarray,
    percentile: float = 50.0,
) -> np.ndarray:
    """Estimate noise power per frequency from selected STFT frames."""

    noise_magnitude = estimate_noise_magnitude_from_frames(
        complex_spectrum,
        frame_indices,
        percentile=percentile,
    )
    return np.maximum(noise_magnitude**2, 0.0)


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


def adaptive_subtraction_factor(
    local_snr_db: np.ndarray,
    *,
    beta: float = 1.3,
    beta_min: float = 0.8,
    beta_max: float = 1.8,
    adaptation_strength: float = 0.5,
    low_snr_db: float = 0.0,
    high_snr_db: float = 15.0,
) -> np.ndarray:
    """Map local SNR to bounded beta: stronger in noise, weaker in speech."""

    snr = np.asarray(local_snr_db, dtype=np.float64)
    if snr.ndim == 0 or not np.isfinite(snr).all():
        raise ValueError("local_snr_db must be a finite non-scalar array")
    if not np.isfinite(beta_min) or not np.isfinite(beta_max) or not beta_min <= beta_max:
        raise ValueError("beta bounds must be finite and ordered")
    if not np.isfinite(beta) or not beta_min <= beta <= beta_max:
        raise ValueError("beta must be within beta_min and beta_max")
    if not np.isfinite(adaptation_strength) or not 0 <= adaptation_strength <= 1:
        raise ValueError("adaptation_strength must be in the interval [0, 1]")
    if not np.isfinite(low_snr_db) or not np.isfinite(high_snr_db) or high_snr_db <= low_snr_db:
        raise ValueError("SNR bounds must be finite and ordered")

    speech_weight = np.clip(
        (snr - low_snr_db) / (high_snr_db - low_snr_db),
        0.0,
        1.0,
    )
    low_snr_beta = np.clip(beta + adaptation_strength, beta_min, beta_max)
    high_snr_beta = np.clip(beta - adaptation_strength, beta_min, beta_max)
    result = low_snr_beta * (1.0 - speech_weight) + high_snr_beta * speech_weight
    return np.clip(np.nan_to_num(result), beta_min, beta_max)


def compute_snr_db(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    epsilon: float = 1e-10,
) -> float:
    """Return finite global SNR in dB for a reference and estimate."""

    reference_values = np.asarray(reference, dtype=np.float64)
    estimate_values = np.asarray(estimate, dtype=np.float64)
    if reference_values.shape != estimate_values.shape:
        raise ValueError("reference and estimate must have the same shape")
    if reference_values.size == 0 or not np.isfinite(reference_values).all() or not np.isfinite(estimate_values).all():
        raise ValueError("reference and estimate must be finite and non-empty")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")

    reference_power = float(np.mean(reference_values**2))
    error_power = float(np.mean((estimate_values - reference_values) ** 2))
    return float(10.0 * np.log10(max(reference_power, epsilon) / max(error_power, epsilon)))

def spectral_subtraction(
    noisy_magnitude: np.ndarray,
    estimated_noise_magnitude: np.ndarray,
    *,
    beta: float | np.ndarray = 1.5,
    spectral_floor: float = 0.08,
) -> np.ndarray:
    """Apply spectral subtraction with a relative magnitude floor.

    The core rule is exactly::

        max(noisy_magnitude - beta * estimated_noise_magnitude,
            spectral_floor * noisy_magnitude)

    ``estimated_noise_magnitude`` may be one column and will broadcast across
    time frames.
    """

    noisy = np.asarray(noisy_magnitude, dtype=np.float64)
    noise = np.asarray(estimated_noise_magnitude, dtype=np.float64)
    beta_values = np.asarray(beta, dtype=np.float64)
    if noisy.ndim != 2 or noisy.shape[1] == 0:
        raise ValueError("noisy_magnitude must be a non-empty two-dimensional matrix")
    if noise.ndim not in (1, 2):
        raise ValueError("estimated_noise_magnitude must be one- or two-dimensional")
    if noise.ndim == 1:
        if noise.shape[0] != noisy.shape[0]:
            raise ValueError("one-dimensional noise magnitude must match frequency bins")
        noise = noise[:, np.newaxis]
    if beta_values.ndim > 2:
        raise ValueError("beta must be scalar, one-dimensional, or two-dimensional")
    if not np.isfinite(noisy).all() or not np.isfinite(noise).all() or not np.isfinite(beta_values).all():
        raise ValueError("magnitude and beta inputs must be finite")
    if np.any(noisy < 0) or np.any(noise < 0) or np.any(beta_values < 0):
        raise ValueError("magnitude and beta inputs must be non-negative")
    if not np.isfinite(spectral_floor) or not 0 <= spectral_floor <= 1:
        raise ValueError("spectral_floor must be in the interval [0, 1]")

    try:
        subtracted = np.maximum(noisy - beta_values * noise, spectral_floor * noisy)
    except ValueError as error:
        raise ValueError("estimated_noise_magnitude is not broadcastable") from error
    return np.nan_to_num(subtracted, nan=0.0, posinf=0.0, neginf=0.0)


def spectral_subtract(
    noisy_magnitude: np.ndarray,
    estimated_noise_magnitude: np.ndarray,
    *,
    beta: float | np.ndarray = 1.5,
    spectral_floor: float = 0.08,
) -> np.ndarray:
    """Compatibility alias for :func:`spectral_subtraction`."""

    return spectral_subtraction(
        noisy_magnitude,
        estimated_noise_magnitude,
        beta=beta,
        spectral_floor=spectral_floor,
    )


def compute_wiener_gain(
    noisy_power: np.ndarray,
    noise_power: np.ndarray,
    *,
    min_gain: float = 0.10,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Return stable bounded Wiener gains.

    This helper remains part of the previous public module interface. The
    production pipeline uses it only for the mild post-filter after spectral
    subtraction has already done the main denoising.
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


def apply_mild_wiener_postfilter(
    enhanced_spectrum: np.ndarray,
    estimated_noise_magnitude: np.ndarray,
    *,
    strength: float = 0.25,
    min_gain: float = 0.65,
    frequency_bins: int = 3,
    temporal_frames: int = 3,
    epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a deliberately mild Wiener residual-noise post-filter.

    The Wiener estimate is calculated from the already-subtracted spectrum,
    smoothed, floored, and blended only ``strength`` of the way from unity.
    Thus spectral subtraction remains the main denoising operation and the
    post-filter cannot create hard spectral holes.
    """

    spectrum = _validate_spectrum(enhanced_spectrum)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in the interval [0, 1]")
    if not np.isfinite(min_gain) or not 0 <= min_gain <= 1:
        raise ValueError("min_gain must be in the interval [0, 1]")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")

    noise_magnitude = np.asarray(estimated_noise_magnitude, dtype=np.float64)
    if noise_magnitude.ndim == 1:
        noise_magnitude = noise_magnitude[:, np.newaxis]
    if noise_magnitude.ndim != 2 or noise_magnitude.shape[0] != spectrum.shape[0]:
        raise ValueError("estimated_noise_magnitude has incompatible frequency bins")
    if not np.isfinite(noise_magnitude).all() or np.any(noise_magnitude < 0):
        raise ValueError("estimated_noise_magnitude must be finite and non-negative")

    residual_wiener_gain = compute_wiener_gain(
        magnitude(spectrum) ** 2,
        noise_magnitude**2,
        min_gain=0.0,
        epsilon=epsilon,
    )
    residual_wiener_gain = smooth_wiener_gain(
        residual_wiener_gain,
        frequency_bins=frequency_bins,
        temporal_frames=temporal_frames,
        min_gain=min_gain,
    )
    post_gain = 1.0 - strength * (1.0 - residual_wiener_gain)
    filtered_magnitude = magnitude(spectrum) * post_gain
    filtered_spectrum = filtered_magnitude * np.exp(1j * phase(spectrum))
    return np.nan_to_num(filtered_spectrum), post_gain


def _noise_frame_bounds(
    frame_count: int,
    sample_count: int,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    requested_seconds: float,
) -> tuple[int, int, float]:
    """Return legacy initial-interval bounds for compatibility callers."""

    used_seconds = min(requested_seconds, sample_count / sample_rate)
    stop_frame = min(
        frame_count,
        max(1, int(np.floor(used_seconds * sample_rate / hop_length)) + 1),
    )
    first_complete_frame = int(np.ceil((frame_length / 2) / hop_length))
    start_frame = first_complete_frame if stop_frame > first_complete_frame else 0
    if stop_frame <= start_frame:
        stop_frame = min(frame_count, start_frame + 1)
    return start_frame, stop_frame, used_seconds


def _select_noise_frames(
    spectra: list[np.ndarray],
    noisy_powers: list[np.ndarray],
    sample_count: int,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    config: SpeechEnhancementConfig,
) -> tuple[np.ndarray, float | None]:
    """Select low-energy frames, ranking the complete recording by default."""

    frame_count = int(spectra[0].shape[1])
    if config.noise_estimation == "initial":
        start_frame, stop_frame, duration_used = _noise_frame_bounds(
            frame_count,
            sample_count,
            sample_rate,
            frame_length,
            hop_length,
            config.noise_duration_seconds,
        )
        return np.arange(start_frame, stop_frame, dtype=np.int64), duration_used

    # The mean spectral power is proportional to windowed frame energy.
    # Channels are combined only for ranking; each channel gets its own noise
    # spectrum below so stereo spatial differences are preserved.
    combined_frame_energy = np.mean(np.stack(noisy_powers, axis=0), axis=(0, 1))
    boundary_frames = int(np.ceil((frame_length / 2) / hop_length))
    candidate_start = min(boundary_frames, frame_count)
    candidate_stop = max(candidate_start, frame_count - boundary_frames)
    complete_frames = np.arange(candidate_start, candidate_stop, dtype=np.int64)
    if complete_frames.size == 0:
        complete_frames = np.arange(frame_count, dtype=np.int64)
    selected = select_low_energy_noise_frames(
        combined_frame_energy,
        frame_percent=config.noise_frame_percent,
        minimum_frames=config.minimum_noise_frames,
        candidate_indices=complete_frames,
    )
    return selected, None


def _track_noise_powers(
    noisy_powers: list[np.ndarray],
    initial_noise_powers: list[np.ndarray],
    config: SpeechEnhancementConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Track noise only on low-energy frames likely dominated by noise."""

    observed = np.stack(noisy_powers, axis=0)
    _, _, frame_count = observed.shape
    tracked = np.stack(
        [np.asarray(noise_power[:, 0], dtype=np.float64) for noise_power in initial_noise_powers],
        axis=0,
    )
    tracked = np.maximum(tracked, 0.0)
    tracks = np.empty_like(observed, dtype=np.float64)
    local_snr_db = np.empty_like(observed, dtype=np.float64)
    posterior_snr_db = np.empty_like(observed, dtype=np.float64)
    frame_energy = np.mean(observed, axis=(0, 1))
    energy_limit = float(np.percentile(frame_energy, config.noise_tracking_energy_percentile))
    initial_noise_energy = max(
        float(np.mean(tracked)),
        config.epsilon,
    )
    speech_energy_limit = initial_noise_energy * 10.0 ** (
        config.speech_activity_margin_db / 10.0
    )
    noise_dominant_frames = np.zeros(frame_count, dtype=bool)

    for frame_index in range(frame_count):
        current_noise = np.maximum(tracked, 0.0)
        observed_frame = observed[:, :, frame_index]
        denominator = current_noise + config.epsilon
        gamma = observed_frame / denominator
        posterior_snr = np.maximum(observed_frame - current_noise, 0.0) / denominator
        local_snr_db[:, :, frame_index] = 10.0 * np.log10(np.maximum(gamma, config.epsilon))
        posterior_snr_db[:, :, frame_index] = 10.0 * np.log10(
            np.maximum(posterior_snr, config.epsilon)
        )
        tracks[:, :, frame_index] = current_noise
        frame_snr_db = 10.0 * np.log10(
            max(float(np.mean(observed_frame)), config.epsilon)
            / max(float(np.mean(current_noise)), config.epsilon)
        )
        is_noise_dominant = (
            frame_snr_db <= config.noise_tracking_snr_db
            and frame_energy[frame_index] <= energy_limit
            and frame_energy[frame_index] <= speech_energy_limit
        )
        noise_dominant_frames[frame_index] = is_noise_dominant
        if is_noise_dominant and config.noise_tracking_alpha > 0:
            tracked = (
                (1.0 - config.noise_tracking_alpha) * current_noise
                + config.noise_tracking_alpha * observed_frame
            )

    speech_active_frames = frame_energy > speech_energy_limit
    return tracks, local_snr_db, posterior_snr_db, noise_dominant_frames, speech_active_frames

def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def enhance_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: SpeechEnhancementConfig | None = None,
    reference_audio: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Enhance mono or channel-first stereo speech with classical DSP."""

    used_config = config or SpeechEnhancementConfig()
    channels, was_mono_vector = _validate_audio(audio, sample_rate)
    reference_channels: np.ndarray | None = None
    if reference_audio is not None:
        reference_channels, _ = _validate_audio(reference_audio, sample_rate)
        if reference_channels.shape != channels.shape:
            raise ValueError("reference_audio must match audio dimensions")
    frame_length, hop_length = speech_stft_parameters(sample_rate, used_config)
    started = perf_counter()

    # stft performs framing, Hann analysis-windowing, and an explicit rFFT.
    spectra = [
        stft(
            channel,
            sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
        )
        for channel in channels
    ]
    noisy_magnitudes = [magnitude(spectrum) for spectrum in spectra]
    noisy_powers = [spectrum_magnitude**2 for spectrum_magnitude in noisy_magnitudes]
    frame_count = int(spectra[0].shape[1])
    noise_frame_indices, noise_duration_used = _select_noise_frames(
        spectra,
        noisy_powers,
        channels.shape[1],
        sample_rate,
        frame_length,
        hop_length,
        used_config,
    )

    initial_noise_powers = [
        estimate_noise_power_from_frames(
            spectrum,
            noise_frame_indices,
            percentile=used_config.noise_percentile,
        )
        for spectrum in spectra
    ]
    (
        noise_power_tracks,
        local_snr_db,
        posterior_snr_db,
        noise_dominant_frames,
        speech_active_frames,
    ) = _track_noise_powers(
        noisy_powers,
        initial_noise_powers,
        used_config,
    )
    if used_config.adaptive_beta:
        adaptive_betas = adaptive_subtraction_factor(
            local_snr_db,
            beta=used_config.beta,
            beta_min=used_config.beta_min,
            beta_max=used_config.beta_max,
            adaptation_strength=used_config.beta_adaptation_strength,
            low_snr_db=used_config.beta_low_snr_db,
            high_snr_db=used_config.beta_high_snr_db,
        )
    else:
        adaptive_betas = np.full_like(local_snr_db, used_config.beta, dtype=np.float64)
    reconstructed_channels: list[np.ndarray] = []
    post_gain_minimum = 1.0
    post_gain_maximum = 0.0
    post_gain_sum = 0.0
    post_gain_value_count = 0
    post_gain_attenuated_over_3_db = 0
    post_gain_attenuated_over_6_db = 0
    subtraction_floor_bins = 0
    subtraction_attenuated_bins = 0

    for channel_index, (complex_spectrum, noisy_magnitude) in enumerate(zip(spectra, noisy_magnitudes)):
        # Use the conservative per-frame track, not one static noise spectrum.
        noise_magnitude = np.sqrt(np.maximum(noise_power_tracks[channel_index], 0.0))

        # Spectral subtraction is the main denoising operation. Noisy phase is
        # retained explicitly when rebuilding the complex spectrum.
        subtracted_magnitude = spectral_subtraction(
            noisy_magnitude,
            noise_magnitude,
            beta=adaptive_betas[channel_index],
            spectral_floor=used_config.spectral_floor,
        )
        subtraction_floor_bins += int(
            np.count_nonzero(
                subtracted_magnitude
                <= used_config.spectral_floor * noisy_magnitude + used_config.epsilon
            )
        )
        subtraction_attenuated_bins += int(
            np.count_nonzero(subtracted_magnitude < noisy_magnitude)
        )
        subtracted_spectrum = subtracted_magnitude * np.exp(1j * phase(complex_spectrum))

        # The Wiener stage is intentionally mild and occurs only after
        # subtraction. It retains the same noisy phase as the spectrum above.
        filtered_spectrum, post_gain = apply_mild_wiener_postfilter(
            subtracted_spectrum,
            noise_magnitude,
            strength=used_config.wiener_strength,
            min_gain=used_config.wiener_min_gain,
            frequency_bins=used_config.frequency_smoothing_bins,
            temporal_frames=used_config.temporal_smoothing_frames,
            epsilon=used_config.epsilon,
        )
        post_gain_minimum = min(post_gain_minimum, float(np.min(post_gain)))
        post_gain_maximum = max(post_gain_maximum, float(np.max(post_gain)))
        post_gain_sum += float(np.sum(post_gain))
        post_gain_value_count += int(post_gain.size)
        post_gain_attenuated_over_3_db += int(np.count_nonzero(post_gain < 10 ** (-3.0 / 20.0)))
        post_gain_attenuated_over_6_db += int(np.count_nonzero(post_gain < 10 ** (-6.0 / 20.0)))

        # istft performs the explicit IFFT and Hann-windowed overlap-add,
        # trimming centered analysis padding to the original duration.
        reconstructed_channels.append(
            istft(
                filtered_spectrum,
                sample_rate,
                n_fft=frame_length,
                hop_length=hop_length,
                length=channels.shape[1],
            )
        )

    enhanced = np.stack(reconstructed_channels, axis=0)
    input_rms = _rms(channels)
    output_rms_before_compensation = _rms(enhanced)

    # Raise a quieter enhanced signal toward the input level, but cap the
    # boost so a poor estimate cannot produce an unexpectedly loud result.
    compensation_gain = 1.0
    if input_rms > used_config.epsilon and output_rms_before_compensation > used_config.epsilon:
        if output_rms_before_compensation < input_rms:
            compensation_gain = min(
                input_rms / output_rms_before_compensation,
                used_config.max_compensation_gain,
            )
    enhanced *= compensation_gain

    # Shared scaling preserves stereo balance and prevents clipping.
    pre_limit_output_peak = float(np.max(np.abs(enhanced)))
    shared_gain = 1.0
    if pre_limit_output_peak > used_config.peak_limit:
        shared_gain = used_config.peak_limit / pre_limit_output_peak
        enhanced *= shared_gain
    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
    enhanced = np.clip(enhanced, -used_config.peak_limit, used_config.peak_limit)
    output_rms_after_compensation = _rms(enhanced)
    output_peak = float(np.max(np.abs(enhanced)))
    if not np.isfinite(enhanced).all():
        raise RuntimeError("speech enhancement produced non-finite samples")

    synthetic_input_snr: float | None = None
    synthetic_output_snr: float | None = None
    snr_improvement: float | None = None
    if reference_channels is not None:
        synthetic_input_snr = compute_snr_db(reference_channels, channels, epsilon=used_config.epsilon)
        synthetic_output_snr = compute_snr_db(reference_channels, enhanced, epsilon=used_config.epsilon)
        snr_improvement = synthetic_output_snr - synthetic_input_snr
    noise_frame_count = int(noise_frame_indices.size)
    mean_post_gain = post_gain_sum / max(post_gain_value_count, 1)
    total_bins = max(post_gain_value_count, 1)
    processing_time = perf_counter() - started
    diagnostics: dict[str, Any] = {
        # Required diagnostics.
        "sample_rate": int(sample_rate),
        "duration": channels.shape[1] / sample_rate,
        "frame_size": int(frame_length),
        "hop_size": int(hop_length),
        "number_of_frames": frame_count,
        "noise_frames_selected": noise_frame_count,
        "mean_local_snr": float(np.mean(local_snr_db)),
        "mean_local_snr_db": float(np.mean(local_snr_db)),
        "mean_posterior_snr_db": float(np.mean(posterior_snr_db)),
        "beta_min": float(np.min(adaptive_betas)),
        "beta_mean": float(np.mean(adaptive_betas)),
        "beta_max": float(np.max(adaptive_betas)),
        "number_of_noise_dominant_frames": int(np.count_nonzero(noise_dominant_frames)),
        "number_of_speech_active_frames": int(np.count_nonzero(speech_active_frames)),
        "speech_activity_margin_db": float(used_config.speech_activity_margin_db),
        "beta": float(used_config.beta),
        "spectral_floor": float(used_config.spectral_floor),
        "input_rms": input_rms,
        "output_rms_before_compensation": output_rms_before_compensation,
        "output_rms_after_compensation": output_rms_after_compensation,
        "synthetic_input_snr": synthetic_input_snr,
        "synthetic_output_snr": synthetic_output_snr,
        "snr_improvement": snr_improvement,
        "synthetic_input_snr_db": synthetic_input_snr,
        "synthetic_output_snr_db": synthetic_output_snr,
        "snr_improvement_db": snr_improvement,
        "processing_time": processing_time,
        # Compatibility and useful detail diagnostics.
        "channels": int(channels.shape[0]),
        "samples": int(channels.shape[1]),
        "duration_seconds": channels.shape[1] / sample_rate,
        "noise_estimation": used_config.noise_estimation,
        "noise_estimation_method": (
            "low_energy_frames"
            if used_config.noise_estimation == "auto"
            else "legacy_initial_interval_with_tracking"
        ),
        "noise_dominant_frames": int(np.count_nonzero(noise_dominant_frames)),
        "noise_tracking_alpha": float(used_config.noise_tracking_alpha),
        "noise_tracking_snr_db": float(used_config.noise_tracking_snr_db),
        "noise_tracking_energy_percentile": float(used_config.noise_tracking_energy_percentile),
        "adaptive_beta": bool(used_config.adaptive_beta),
        "beta_adaptation_strength": float(used_config.beta_adaptation_strength),
        "beta_low_snr_db": float(used_config.beta_low_snr_db),
        "beta_high_snr_db": float(used_config.beta_high_snr_db),
        "noise_frames": noise_frame_count,
        "noise_frame_percent": float(used_config.noise_frame_percent),
        "noise_frames_percent": 100.0 * noise_frame_count / max(frame_count, 1),
        "noise_estimation_duration_seconds": noise_duration_used,
        "noise_percentile": float(used_config.noise_percentile),
        "wiener_strength": float(used_config.wiener_strength),
        "wiener_min_gain": float(used_config.wiener_min_gain),
        "wiener_gain_min": post_gain_minimum,
        "wiener_gain_max": post_gain_maximum,
        "mean_wiener_gain": mean_post_gain,
        "subtraction_floor_bins_percent": 100.0 * subtraction_floor_bins / total_bins,
        "subtraction_attenuated_bins_percent": 100.0 * subtraction_attenuated_bins / total_bins,
        "bins_attenuated_over_3_db_percent": 100.0 * post_gain_attenuated_over_3_db / total_bins,
        "bins_attenuated_over_6_db_percent": 100.0 * post_gain_attenuated_over_6_db / total_bins,
        "frequency_smoothing_bins": int(used_config.frequency_smoothing_bins),
        "temporal_smoothing_frames": int(used_config.temporal_smoothing_frames),
        "output_rms": output_rms_after_compensation,
        "input_peak": float(np.max(np.abs(channels))),
        "pre_limit_output_peak": pre_limit_output_peak,
        "output_peak": output_peak,
        "compensation_gain": compensation_gain,
        "output_level_compensation_gain": compensation_gain,
        "shared_output_gain": shared_gain,
        "frame_duration_ms": float(used_config.frame_duration_ms),
        "overlap_ratio": float(used_config.overlap_ratio),
        "stft_size": int(frame_length),
        "hop_length": int(hop_length),
        "frames": frame_count,
        "processing_seconds": processing_time,
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
    """Enhance decoded audio and write WAV result files."""

    source = Path(input_path)
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