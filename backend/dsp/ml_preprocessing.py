"""DSP preprocessing shared by the future stereo mask model.

This module deliberately stops before machine-learning inference.  It keeps
the complex Left/Right STFT separate from the normalized magnitudes so that a
later model can consume only the normalized features while reconstruction
still uses the original mixture phase and magnitude scale.
"""

from dataclasses import dataclass
from math import gcd
from typing import Iterator

import numpy as np
from scipy.signal import resample_poly

from .stft_utils import DEFAULT_HOP_LENGTH, DEFAULT_N_FFT, istft, magnitude, stft


DEFAULT_MODEL_SAMPLE_RATE = 44_100
DEFAULT_CHUNK_SECONDS = 6.0
DEFAULT_CHUNK_HOP_SECONDS = 3.0
DEFAULT_NORMALIZATION_EPSILON = 1e-8


@dataclass(frozen=True)
class PreprocessingConfig:
    """Numerical contract shared by dataset preparation and inference."""

    sample_rate: int = DEFAULT_MODEL_SAMPLE_RATE
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    chunk_hop_seconds: float = DEFAULT_CHUNK_HOP_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.sample_rate, (int, np.integer)) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if not isinstance(self.n_fft, (int, np.integer)) or self.n_fft <= 0:
            raise ValueError("n_fft must be a positive integer.")
        if not isinstance(self.hop_length, (int, np.integer)) or self.hop_length <= 0:
            raise ValueError("hop_length must be a positive integer.")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must not be greater than n_fft.")
        if not np.isfinite(self.chunk_seconds) or self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive and finite.")
        if not np.isfinite(self.chunk_hop_seconds) or self.chunk_hop_seconds <= 0:
            raise ValueError("chunk_hop_seconds must be positive and finite.")
        if self.chunk_hop_seconds > self.chunk_seconds:
            raise ValueError("chunk_hop_seconds must not exceed chunk_seconds.")
        if self.chunk_samples < self.n_fft:
            raise ValueError("chunk_seconds must provide at least one FFT window.")

    @property
    def chunk_samples(self) -> int:
        """Number of waveform samples in one model chunk."""

        return max(1, int(round(self.sample_rate * self.chunk_seconds)))

    @property
    def chunk_hop_samples(self) -> int:
        """Number of samples between adjacent chunk starts."""

        return max(1, int(round(self.sample_rate * self.chunk_hop_seconds)))


@dataclass(frozen=True)
class MagnitudeNormalizationStats:
    """Mean and standard deviation for log-magnitude normalization.

    The arrays have shape ``(channels, 1, 1)`` and broadcast over frequency
    and time.  Stage 3 can aggregate these statistics over training chunks and
    freeze them in a model checkpoint.
    """

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class StereoChunk:
    """A fixed-size stereo waveform chunk and its valid unpadded length."""

    audio: np.ndarray
    start_sample: int
    valid_samples: int


@dataclass(frozen=True)
class StereoFeatures:
    """Complex spectra and model features for one stereo waveform segment."""

    complex_stft: np.ndarray
    magnitude: np.ndarray
    normalized_magnitude: np.ndarray
    normalization_stats: MagnitudeNormalizationStats


def _validate_stereo_audio(stereo_audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(stereo_audio)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("stereo_audio must have shape (2, samples).")
    if audio.shape[1] == 0:
        raise ValueError("stereo_audio must contain at least one sample.")
    if np.iscomplexobj(audio):
        raise ValueError("stereo_audio must contain real-valued samples.")
    if not np.isfinite(audio).all():
        raise ValueError("stereo_audio contains non-finite sample values.")
    return np.ascontiguousarray(audio, dtype=np.float32)


def prepare_model_audio(
    stereo_audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = DEFAULT_MODEL_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Validate stereo audio and resample it to the model sample rate.

    Resampling is performed before chunking and STFT analysis.  The expected
    output length is enforced so later chunk stitching can restore a precise
    sample count at the model rate.
    """

    audio = _validate_stereo_audio(stereo_audio)
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if not isinstance(target_sample_rate, (int, np.integer)) or target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be a positive integer.")

    source_rate = int(sample_rate)
    target_rate = int(target_sample_rate)
    if source_rate == target_rate:
        return audio.copy(), target_rate

    divisor = gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    resampled_channels = [
        resample_poly(channel, up=up, down=down).astype(np.float32, copy=False)
        for channel in audio
    ]
    resampled = np.vstack(resampled_channels)

    expected_samples = max(1, int(round(audio.shape[1] * target_rate / source_rate)))
    if resampled.shape[1] < expected_samples:
        resampled = np.pad(
            resampled,
            ((0, 0), (0, expected_samples - resampled.shape[1])),
        )
    elif resampled.shape[1] > expected_samples:
        resampled = resampled[:, :expected_samples]

    if not np.isfinite(resampled).all():
        raise ValueError("resampling produced non-finite sample values.")
    return np.ascontiguousarray(resampled, dtype=np.float32), target_rate


def iter_stereo_chunks(
    stereo_audio: np.ndarray,
    config: PreprocessingConfig,
) -> Iterator[StereoChunk]:
    """Yield fixed-size, overlapping chunks with zero padding at the end."""

    audio = _validate_stereo_audio(stereo_audio)
    chunk_samples = config.chunk_samples
    hop_samples = config.chunk_hop_samples
    final_start = max(0, audio.shape[1] - chunk_samples)

    starts = list(range(0, final_start + 1, hop_samples))
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    for start_sample in starts:
        valid_samples = min(chunk_samples, audio.shape[1] - start_sample)
        chunk = np.zeros((2, chunk_samples), dtype=np.float32)
        chunk[:, :valid_samples] = audio[:, start_sample : start_sample + valid_samples]
        yield StereoChunk(
            audio=chunk,
            start_sample=int(start_sample),
            valid_samples=int(valid_samples),
        )


def stereo_stft(
    stereo_audio: np.ndarray,
    sample_rate: int,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Compute complex STFTs for Left and Right channels.

    The output shape is ``(2, frequency_bins, time_frames)``.  The input must
    already be at ``config.sample_rate`` so training and inference cannot
    silently use different physical frequency/time resolutions.
    """

    audio = _validate_stereo_audio(stereo_audio)
    if int(sample_rate) != config.sample_rate:
        raise ValueError(
            "sample_rate must equal config.sample_rate; call prepare_model_audio first."
        )

    spectra = [
        stft(channel, sample_rate, n_fft=config.n_fft, hop_length=config.hop_length)
        for channel in audio
    ]
    return np.ascontiguousarray(np.stack(spectra, axis=0))


def stereo_istft(
    complex_stft: np.ndarray,
    sample_rate: int,
    config: PreprocessingConfig,
    length: int,
) -> np.ndarray:
    """Reconstruct a channel-first stereo waveform from complex STFTs."""

    spectra = np.asarray(complex_stft)
    if spectra.ndim != 3 or spectra.shape[0] != 2:
        raise ValueError(
            "complex_stft must have shape (2, frequency_bins, time_frames)."
        )
    if int(sample_rate) != config.sample_rate:
        raise ValueError(
            "sample_rate must equal config.sample_rate; call prepare_model_audio first."
        )
    if not isinstance(length, (int, np.integer)) or length < 0:
        raise ValueError("length must be a non-negative integer.")

    reconstructed = [
        istft(
            channel_spectrum,
            sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            length=int(length),
        )
        for channel_spectrum in spectra
    ]
    return np.ascontiguousarray(np.stack(reconstructed, axis=0), dtype=np.float32)


def _validate_magnitude(stereo_magnitude: np.ndarray) -> np.ndarray:
    values = np.asarray(stereo_magnitude)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError(
            "stereo_magnitude must have shape (2, frequency_bins, time_frames)."
        )
    if values.size == 0:
        raise ValueError("stereo_magnitude must not be empty.")
    if np.iscomplexobj(values) or np.any(values < 0):
        raise ValueError("stereo_magnitude must contain non-negative real values.")
    if not np.isfinite(values).all():
        raise ValueError("stereo_magnitude contains non-finite values.")
    return np.asarray(values, dtype=np.float32)


def fit_magnitude_normalization(
    stereo_magnitude: np.ndarray,
    epsilon: float = DEFAULT_NORMALIZATION_EPSILON,
) -> MagnitudeNormalizationStats:
    """Fit per-channel statistics on log-compressed magnitudes."""

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite.")
    magnitude_values = _validate_magnitude(stereo_magnitude)
    log_values = np.log1p(magnitude_values)
    mean = np.mean(log_values, axis=(-2, -1), keepdims=True)
    std = np.maximum(np.std(log_values, axis=(-2, -1), keepdims=True), epsilon)
    return MagnitudeNormalizationStats(
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
    )


def normalize_magnitude(
    stereo_magnitude: np.ndarray,
    stats: MagnitudeNormalizationStats | None = None,
) -> tuple[np.ndarray, MagnitudeNormalizationStats]:
    """Return normalized log magnitudes and the statistics used.

    The returned normalized values are model features only.  They must never
    be used for complex-STFT reconstruction; callers should retain the
    unnormalized complex STFT from :func:`stereo_stft`.
    """

    magnitude_values = _validate_magnitude(stereo_magnitude)
    log_values = np.log1p(magnitude_values)
    used_stats = stats or fit_magnitude_normalization(magnitude_values)
    mean = np.asarray(used_stats.mean, dtype=np.float32)
    std = np.asarray(used_stats.std, dtype=np.float32)
    if mean.shape != (2, 1, 1) or std.shape != (2, 1, 1):
        raise ValueError("normalization statistics must have shape (2, 1, 1).")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization statistics must be finite with positive std.")

    normalized = (log_values - mean) / std
    return np.asarray(normalized, dtype=np.float32), MagnitudeNormalizationStats(
        mean=mean,
        std=std,
    )


def extract_stereo_features(
    stereo_audio: np.ndarray,
    sample_rate: int,
    config: PreprocessingConfig,
    normalization_stats: MagnitudeNormalizationStats | None = None,
) -> StereoFeatures:
    """Compute complex spectra, magnitudes, and normalized model features."""

    complex_spectra = stereo_stft(stereo_audio, sample_rate, config)
    magnitude_values = np.asarray(magnitude(complex_spectra), dtype=np.float32)
    normalized, used_stats = normalize_magnitude(magnitude_values, normalization_stats)
    return StereoFeatures(
        complex_stft=complex_spectra,
        magnitude=magnitude_values,
        normalized_magnitude=normalized,
        normalization_stats=used_stats,
    )


def preprocess_stereo_chunks(
    stereo_audio: np.ndarray,
    sample_rate: int,
    config: PreprocessingConfig,
    normalization_stats: MagnitudeNormalizationStats | None = None,
) -> tuple[int, tuple[StereoChunk, ...], tuple[StereoFeatures, ...]]:
    """Prepare resampled chunks and their stereo spectrogram features."""

    prepared_audio, prepared_rate = prepare_model_audio(
        stereo_audio,
        sample_rate,
        target_sample_rate=config.sample_rate,
    )
    chunks = tuple(iter_stereo_chunks(prepared_audio, config))
    features = tuple(
        extract_stereo_features(
            chunk.audio,
            prepared_rate,
            config,
            normalization_stats=normalization_stats,
        )
        for chunk in chunks
    )
    return prepared_rate, chunks, features
