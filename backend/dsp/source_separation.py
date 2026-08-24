"""Classical DSP vocal and instrumental separation using mid-side masking.

This module intentionally uses no machine learning. It estimates centered vocal
content with mid-side processing, STFT analysis, and soft spectral masks.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_utils import load_audio
from .mid_side import decode_mid_side, encode_mid_side
from .stft_utils import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    istft,
    magnitude,
    stft,
)


DEFAULT_MASK_POWER = 2.0
DEFAULT_ALPHA = 1.0
DEFAULT_EPSILON = 1e-10
DEFAULT_PEAK_LIMIT = 0.99
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs"


def create_spectral_masks(
    mid_stft: np.ndarray,
    side_stft: np.ndarray,
    p: float = DEFAULT_MASK_POWER,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Create complementary soft masks for mid vocal and instrumental content.

    Center-panned sources occur similarly in left and right channels and are
    therefore strong in the mid signal. A time-frequency region dominated by
    mid magnitude is consequently treated as more likely to contain vocals:

        VocalMask = |M|^p / (|M|^p + alpha * |S|^p + epsilon)

    This is only a classical DSP assumption. Centered instruments such as bass,
    kick, or snare can also be mid-dominant and may leak into the vocal result.
    Likewise, off-center vocals may leak into the instrumental result.
    """

    mid_spectrum = np.asarray(mid_stft)
    side_spectrum = np.asarray(side_stft)

    if mid_spectrum.ndim != 2 or side_spectrum.ndim != 2:
        raise ValueError("mid_stft and side_stft must be two-dimensional matrices.")
    if mid_spectrum.shape != side_spectrum.shape:
        raise ValueError("mid_stft and side_stft must have identical shapes.")
    if mid_spectrum.size == 0:
        raise ValueError("STFT matrices must not be empty.")
    if not np.isfinite(mid_spectrum).all() or not np.isfinite(side_spectrum).all():
        raise ValueError("STFT matrices must contain only finite values.")
    if not np.isfinite(p) or p <= 0:
        raise ValueError("p must be a positive finite value.")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a non-negative finite value.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite value.")

    mid_weight = magnitude(mid_spectrum) ** p
    side_weight = magnitude(side_spectrum) ** p
    denominator = mid_weight + alpha * side_weight + epsilon
    vocal_mask = mid_weight / denominator

    # A complementary soft mask assigns the remaining mid content to the
    # instrumental stem without hard binary time-frequency decisions.
    instrumental_mask = 1.0 - vocal_mask
    return vocal_mask, instrumental_mask


def _apply_shared_peak_limit(
    vocals: np.ndarray,
    instrumental: np.ndarray,
    peak_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one shared gain when either stem would clip.

    A shared scale preserves the level relationship between the two stems. It
    is preferable to independently normalizing them, which would change the
    balance established by the complementary masks.
    """

    if not np.isfinite(peak_limit) or peak_limit <= 0 or peak_limit > 1:
        raise ValueError("peak_limit must be in the interval (0, 1].")

    peak = max(
        float(np.max(np.abs(vocals))),
        float(np.max(np.abs(instrumental))),
    )
    if peak <= peak_limit:
        return vocals, instrumental

    gain = peak_limit / peak
    return vocals * gain, instrumental * gain


def separate_stereo_audio(
    stereo_audio: np.ndarray,
    sample_rate: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    p: float = DEFAULT_MASK_POWER,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    peak_limit: float = DEFAULT_PEAK_LIMIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate channel-first stereo audio into vocal and instrumental stems."""

    mid, side = encode_mid_side(stereo_audio)

    mid_spectrum = stft(mid, sample_rate, n_fft=n_fft, hop_length=hop_length)
    side_spectrum = stft(side, sample_rate, n_fft=n_fft, hop_length=hop_length)
    vocal_mask, instrumental_mask = create_spectral_masks(
        mid_spectrum,
        side_spectrum,
        p=p,
        alpha=alpha,
        epsilon=epsilon,
    )

    # Multiplication by real-valued soft masks changes magnitude while keeping
    # the original complex STFT phase for reconstruction.
    vocal_mid_spectrum = vocal_mask * mid_spectrum
    instrumental_mid_spectrum = instrumental_mask * mid_spectrum

    sample_count = np.asarray(stereo_audio).shape[-1]
    vocal_mid = istft(
        vocal_mid_spectrum,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        length=sample_count,
    )
    instrumental_mid = istft(
        instrumental_mid_spectrum,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        length=sample_count,
    )
    instrumental_side = istft(
        side_spectrum,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        length=sample_count,
    )

    # The vocal estimate is centered (zero side). The instrumental stem keeps
    # all side information along with the complementary portion of the mid.
    vocals = decode_mid_side(vocal_mid, np.zeros_like(vocal_mid))
    instrumental = decode_mid_side(instrumental_mid, instrumental_side)
    return _apply_shared_peak_limit(vocals, instrumental, peak_limit)


def separate_audio_file(
    input_path: str | Path,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    p: float = DEFAULT_MASK_POWER,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    peak_limit: float = DEFAULT_PEAK_LIMIT,
) -> dict:
    """Separate an audio file and write ``vocals.wav`` and ``instrumental.wav``."""

    stereo_audio, sample_rate = load_audio(input_path, preserve_channels=True)
    vocals, instrumental = separate_stereo_audio(
        stereo_audio,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        p=p,
        alpha=alpha,
        epsilon=epsilon,
        peak_limit=peak_limit,
    )

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    vocals_path = destination / "vocals.wav"
    instrumental_path = destination / "instrumental.wav"

    # soundfile expects (samples, channels), so transpose channel-first stems.
    sf.write(vocals_path, vocals.T, sample_rate, subtype="PCM_16")
    sf.write(instrumental_path, instrumental.T, sample_rate, subtype="PCM_16")

    return {
        "sample_rate": sample_rate,
        "channels": int(stereo_audio.shape[0]),
        "samples": int(stereo_audio.shape[-1]),
        "vocals_path": vocals_path,
        "instrumental_path": instrumental_path,
    }
