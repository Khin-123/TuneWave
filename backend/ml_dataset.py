"""Lazy compressed MUSDB18 loading for stereo vocal-mask training.

Stage 3 deliberately stops at NumPy preprocessing. The official ``musdb``
reader opens ``.stem.mp4`` containers through ``stempeg``/FFmpeg and each
``Track`` is configured with a small chunk before its audio is read. No WAV
copies, complete-track arrays, or spectrogram caches are created here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np

from dsp.ml_preprocessing import (
    MagnitudeNormalizationStats,
    PreprocessingConfig,
    extract_stereo_features,
    stereo_stft,
)
from dsp.stft_utils import magnitude


DEFAULT_MUSDB_ROOT = Path("datasets") / "musdb18"
MUSDB_SAMPLE_RATE = 44_100
MUSDB_STEM_ORDER = ("mixture", "drums", "bass", "other", "vocals")
DEFAULT_TARGET_MASK_POWER = 2.0
DEFAULT_TARGET_MASK_EPSILON = 1e-8


class DatasetError(ValueError):
    """Raised when compressed MUSDB18 cannot be opened or is invalid."""


@dataclass(frozen=True)
class ChunkIndex:
    """Metadata for a chunk; audio is decoded only when it is requested."""

    track_index: int
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class CompressedStemChunk:
    """Aligned channel-first stereo arrays for one decoded chunk."""

    mixture: np.ndarray
    vocals: np.ndarray
    instrumental: np.ndarray
    start_seconds: float
    duration_seconds: float


class TrainingExample(NamedTuple):
    """One normalized L/R feature tensor and its stereo vocal-mask target."""

    input_tensor: np.ndarray
    target_mask: np.ndarray


def _require_stereo(audio: np.ndarray, description: str) -> np.ndarray:
    """Convert musdb's ``(samples, channels)`` audio to ``(2, samples)``."""

    values = np.asarray(audio)
    if values.ndim != 2:
        raise DatasetError(f"{description} must be a two-dimensional stereo array.")
    if values.shape[1] == 2:
        channel_first = values.T
    elif values.shape[0] == 2:
        channel_first = values
    else:
        raise DatasetError(f"{description} must contain exactly two channels.")
    if channel_first.shape[1] == 0:
        raise DatasetError(f"{description} contains no samples.")
    if np.iscomplexobj(channel_first) or not np.isfinite(channel_first).all():
        raise DatasetError(f"{description} must contain finite real samples.")
    return np.ascontiguousarray(channel_first, dtype=np.float32)


def _load_tracks(dataset_root: str | Path, split: str, sample_rate: int) -> tuple[object, ...]:
    """Open compressed tracks through the official musdb API."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise DatasetError(f"MUSDB18 root does not exist: {root}")
    if int(sample_rate) != MUSDB_SAMPLE_RATE:
        raise DatasetError(
            "Compressed MUSDB18 is kept at 44.1 kHz; use a config with "
            f"sample_rate={MUSDB_SAMPLE_RATE}."
        )
    try:
        import musdb  # type: ignore[import-not-found]
        import stempeg  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as error:
        raise DatasetError(
            "Compressed MUSDB18 requires the 'musdb' and 'stempeg' Python "
            "packages plus an FFmpeg executable on PATH."
        ) from error

    if split == "train":
        subsets, musdb_split = "train", "train"
    elif split == "valid":
        subsets, musdb_split = "train", "valid"
    elif split == "all_train":
        subsets, musdb_split = "train", None
    elif split == "test":
        subsets, musdb_split = "test", None
    else:
        raise DatasetError("split must be 'train', 'valid', 'all_train', or 'test'.")

    try:
        database = musdb.DB(
            root=str(root),
            subsets=subsets,
            split=musdb_split,
            is_wav=False,
            sample_rate=int(sample_rate),
        )
        tracks = tuple(database.tracks)
    except Exception as error:
        raise DatasetError(f"Could not open compressed MUSDB18 at {root}.") from error
    if not tracks:
        raise DatasetError(f"No tracks found for MUSDB18 split '{split}'.")
    return tracks


def _chunk_starts(duration_seconds: float, chunk_seconds: float, hop_seconds: float) -> Iterator[tuple[float, float]]:
    """Yield metadata-only chunk locations, including an aligned final chunk."""

    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise DatasetError("Track duration must be positive and finite.")
    if duration_seconds <= chunk_seconds:
        yield 0.0, float(duration_seconds)
        return

    final_start = duration_seconds - chunk_seconds
    start = 0.0
    while start < final_start:
        yield float(start), float(chunk_seconds)
        start += hop_seconds
    yield float(final_start), float(chunk_seconds)


def _set_chunk(track: object, start_seconds: float, duration_seconds: float) -> None:
    """Set the official musdb lazy-read window on a Track instance."""

    track.chunk_start = float(start_seconds)
    track.chunk_duration = float(duration_seconds)


def _read_track_chunk(track: object, start_seconds: float, duration_seconds: float) -> CompressedStemChunk:
    """Decode mixture and four source stems for one window on demand."""

    _set_chunk(track, start_seconds, duration_seconds)
    try:
        mixture = _require_stereo(track.audio, "mixture")
        sources = track.sources
        vocals = _require_stereo(sources["vocals"].audio, "vocals")
        drums = _require_stereo(sources["drums"].audio, "drums")
        bass = _require_stereo(sources["bass"].audio, "bass")
        other = _require_stereo(sources["other"].audio, "other")
    except DatasetError:
        raise
    except Exception as error:
        name = getattr(track, "name", "unknown")
        raise DatasetError(f"Could not decode chunk from MUSDB18 track {name}.") from error

    sample_count = min(mixture.shape[1], vocals.shape[1], drums.shape[1], bass.shape[1], other.shape[1])
    if sample_count <= 0:
        raise DatasetError("Decoded MUSDB18 chunk contains no aligned samples.")
    instrumental = drums[:, :sample_count] + bass[:, :sample_count] + other[:, :sample_count]
    return CompressedStemChunk(
        mixture=np.ascontiguousarray(mixture[:, :sample_count]),
        vocals=np.ascontiguousarray(vocals[:, :sample_count]),
        instrumental=np.ascontiguousarray(instrumental, dtype=np.float32),
        start_seconds=float(start_seconds),
        duration_seconds=float(duration_seconds),
    )


def create_vocal_mask(vocal_stft: np.ndarray, instrumental_stft: np.ndarray, power: float = DEFAULT_TARGET_MASK_POWER, epsilon: float = DEFAULT_TARGET_MASK_EPSILON) -> np.ndarray:
    """Create a bounded ideal ratio (Wiener-style) stereo vocal mask."""

    vocals = np.asarray(vocal_stft)
    instrumental = np.asarray(instrumental_stft)
    if vocals.ndim != 3 or instrumental.ndim != 3:
        raise DatasetError("Target STFTs must be three-dimensional.")
    if vocals.shape != instrumental.shape or vocals.shape[0] != 2:
        raise DatasetError("Target STFTs must have matching stereo shapes.")
    if vocals.size == 0 or not np.isfinite(vocals).all() or not np.isfinite(instrumental).all():
        raise DatasetError("Target STFTs must be non-empty and finite.")
    if not np.isfinite(power) or power <= 0:
        raise DatasetError("power must be positive and finite.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise DatasetError("epsilon must be positive and finite.")

    vocal_weight = magnitude(vocals) ** power
    instrumental_weight = magnitude(instrumental) ** power
    target = vocal_weight / (vocal_weight + instrumental_weight + epsilon)
    return np.asarray(np.clip(target, 0.0, 1.0), dtype=np.float32)


def make_training_example(chunk: CompressedStemChunk, config: PreprocessingConfig, normalization_stats: MagnitudeNormalizationStats | None = None, target_mask_power: float = DEFAULT_TARGET_MASK_POWER) -> TrainingExample:
    """Convert one decoded chunk to normalized input and target mask tensors."""

    mixture_features = extract_stereo_features(chunk.mixture, config.sample_rate, config, normalization_stats=normalization_stats)
    vocal_stft = stereo_stft(chunk.vocals, config.sample_rate, config)
    instrumental_stft = stereo_stft(chunk.instrumental, config.sample_rate, config)
    target_mask = create_vocal_mask(vocal_stft, instrumental_stft, power=target_mask_power)
    return TrainingExample(
        input_tensor=np.asarray(mixture_features.normalized_magnitude, dtype=np.float32),
        target_mask=target_mask,
    )


def fit_training_normalization(dataset: "MusicSeparationDataset") -> MagnitudeNormalizationStats:
    """Fit global per-channel log-magnitude statistics from training chunks."""

    if dataset.split not in {"train", "all_train"}:
        raise DatasetError(
            "Normalization statistics must be fit on a training split only; "
            "validation and test tracks remain held out."
        )
    total = np.zeros(2, dtype=np.float64)
    total_squared = np.zeros(2, dtype=np.float64)
    count = 0
    for index in range(len(dataset)):
        chunk = dataset.load_chunk(index)
        spectra = stereo_stft(chunk.mixture, dataset.config.sample_rate, dataset.config)
        log_values = np.log1p(magnitude(spectra))
        total += np.sum(log_values, axis=(-2, -1), dtype=np.float64)
        total_squared += np.sum(log_values ** 2, axis=(-2, -1), dtype=np.float64)
        count += log_values.shape[-2] * log_values.shape[-1]

    if count <= 0:
        raise DatasetError("Cannot fit normalization statistics without training chunks.")
    mean = total / count
    variance = np.maximum(total_squared / count - mean ** 2, DEFAULT_TARGET_MASK_EPSILON ** 2)
    return MagnitudeNormalizationStats(
        mean=np.asarray(mean[:, None, None], dtype=np.float32),
        std=np.asarray(np.sqrt(variance)[:, None, None], dtype=np.float32),
    )


class MusicSeparationDataset:
    """Lazy compressed MUSDB18 dataset returning ``(input, target)`` pairs."""

    def __init__(self, dataset_root: str | Path = DEFAULT_MUSDB_ROOT, split: str = "train", config: PreprocessingConfig | None = None, normalization_stats: MagnitudeNormalizationStats | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.tracks = _load_tracks(self.dataset_root, split, self.config.sample_rate)
        self.normalization_stats = normalization_stats
        self._index: list[ChunkIndex] = []

        # Only metadata is read here. Track audio is not decoded until __getitem__.
        for track_index, track in enumerate(self.tracks):
            duration = float(getattr(track, "duration"))
            for start, chunk_duration in _chunk_starts(duration, self.config.chunk_seconds, self.config.chunk_hop_seconds):
                self._index.append(ChunkIndex(track_index, start, chunk_duration))

    def __len__(self) -> int:
        return len(self._index)

    def _locate(self, index: int) -> ChunkIndex:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer.")
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError("dataset index out of range.")
        return self._index[normalized]

    def load_chunk(self, index: int) -> CompressedStemChunk:
        """Decode one mixture/source chunk from its compressed stem container."""

        record = self._locate(index)
        return _read_track_chunk(self.tracks[record.track_index], record.start_seconds, record.duration_seconds)

    def __getitem__(self, index: int) -> TrainingExample:
        return make_training_example(self.load_chunk(index), self.config, normalization_stats=self.normalization_stats)

    def set_normalization_stats(self, stats: MagnitudeNormalizationStats) -> None:
        """Apply frozen training statistics to subsequent model inputs."""

        self.normalization_stats = stats
