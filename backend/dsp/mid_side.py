"""Classical mid-side encoding and decoding for stereo audio."""

import numpy as np


def _validate_stereo(stereo_audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(stereo_audio)

    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(
            "stereo_audio must have shape (2, samples) with separate left and "
            "right channels; mono audio is not supported."
        )
    if audio.shape[1] == 0:
        raise ValueError("stereo_audio must contain at least one sample per channel.")
    if np.iscomplexobj(audio):
        raise ValueError("stereo_audio must contain real-valued samples.")

    audio = audio.astype(np.float64, copy=False)
    if not np.isfinite(audio).all():
        raise ValueError("stereo_audio contains non-finite sample values.")

    return audio


def _validate_mid_side_channel(channel: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(channel)

    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional signal.")
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real-valued samples.")

    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite sample values.")

    return values


def encode_mid_side(stereo_audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert channel-first stereo audio into mid and side signals.

    For left channel L[n] and right channel R[n]:

        M[n] = (L[n] + R[n]) / 2
        S[n] = (L[n] - R[n]) / 2

    Mid represents content common to both channels, while side represents the
    difference between the channels.
    """

    audio = _validate_stereo(stereo_audio)
    left = audio[0]
    right = audio[1]

    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    return mid, side


def decode_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Reconstruct channel-first stereo audio from mid and side signals.

    The inverse equations are:

        L[n] = M[n] + S[n]
        R[n] = M[n] - S[n]
    """

    mid_signal = _validate_mid_side_channel(mid, "mid")
    side_signal = _validate_mid_side_channel(side, "side")

    if mid_signal.shape != side_signal.shape:
        raise ValueError("mid and side must contain the same number of samples.")

    left = mid_signal + side_signal
    right = mid_signal - side_signal
    return np.vstack((left, right))
