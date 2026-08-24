"""Reusable framing, Hann window, STFT, and ISTFT utilities."""

import numpy as np


DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


def hann_window(window_length: int) -> np.ndarray:
    """Return a standard symmetric Hann window.

    The window tap at index n is:

        w[n] = 0.5 - 0.5 cos(2πn/(N-1))

    A Hann window tapers each frame toward zero at its boundaries, reducing
    spectral leakage when the frame is transformed with the FFT.
    """

    if not isinstance(window_length, (int, np.integer)) or window_length <= 0:
        raise ValueError("window_length must be a positive integer.")

    if window_length == 1:
        return np.ones(1, dtype=np.float64)

    sample_indices = np.arange(window_length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(
        2.0 * np.pi * sample_indices / (window_length - 1)
    )


def _validate_settings(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> None:
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if not isinstance(n_fft, (int, np.integer)) or n_fft <= 0:
        raise ValueError("n_fft must be a positive integer.")
    if not isinstance(hop_length, (int, np.integer)) or hop_length <= 0:
        raise ValueError("hop_length must be a positive integer.")
    if hop_length > n_fft:
        raise ValueError("hop_length must not be greater than n_fft.")


def _as_mono_signal(signal: np.ndarray) -> np.ndarray:
    signal_array = np.asarray(signal)

    if signal_array.ndim != 1:
        raise ValueError(
            "signal must be one-dimensional; process each audio channel separately."
        )
    if signal_array.size == 0:
        raise ValueError("signal must contain at least one sample.")
    if np.iscomplexobj(signal_array):
        raise ValueError("signal must contain real-valued audio samples.")

    signal_array = signal_array.astype(np.float64, copy=False)
    if not np.isfinite(signal_array).all():
        raise ValueError("signal contains non-finite sample values.")

    return signal_array


def _frame_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """Split a signal into overlapping frames.

    A frame contains ``frame_length`` consecutive samples. Each new frame
    begins ``hop_length`` samples after the previous frame, so a hop smaller
    than the frame length creates the overlap needed for smooth reconstruction.
    """

    # Centering pads both boundaries so every original sample is covered by
    # non-zero portions of overlapping Hann windows.
    center_padding = frame_length // 2
    padded_signal = np.pad(signal, (center_padding, center_padding))

    # Pad the final partial frame instead of discarding audio at the end.
    frame_count = max(
        1,
        int(np.ceil((padded_signal.size - frame_length) / hop_length)) + 1,
    )
    framed_length = frame_length + (frame_count - 1) * hop_length
    if padded_signal.size < framed_length:
        padded_signal = np.pad(padded_signal, (0, framed_length - padded_signal.size))

    sliding_frames = np.lib.stride_tricks.sliding_window_view(
        padded_signal,
        frame_length,
    )
    return np.ascontiguousarray(sliding_frames[::hop_length])


def stft(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Compute the complex short-time Fourier transform of a mono signal.

    STFT applies the FFT to overlapping, Hann-windowed frames. Rows in the
    returned matrix are non-negative frequency bins and columns are time
    frames. The complex values retain both magnitude and phase.
    """

    _validate_settings(sample_rate, n_fft, hop_length)
    signal_array = _as_mono_signal(signal)
    frames = _frame_signal(signal_array, n_fft, hop_length)
    windowed_frames = frames * hann_window(n_fft)

    # rFFT is sufficient for real audio and returns n_fft // 2 + 1 bins.
    return np.fft.rfft(windowed_frames, n=n_fft, axis=1).T


def magnitude(complex_stft: np.ndarray) -> np.ndarray:
    """Return STFT magnitude |X(k,m)| for frequency bin k and frame m."""

    return np.abs(np.asarray(complex_stft))


def phase(complex_stft: np.ndarray) -> np.ndarray:
    """Return STFT phase angles, in radians, for every time-frequency bin."""

    return np.angle(np.asarray(complex_stft))


def istft(
    complex_stft: np.ndarray,
    sample_rate: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    length: int | None = None,
) -> np.ndarray:
    """Reconstruct a time-domain signal using inverse FFT and overlap-add.

    Each inverse-FFT frame is multiplied by the same Hann window used during
    analysis. Overlapping frames are added, then normalized by the sum of
    squared window values. Pass the original sample count as ``length`` to
    remove analysis padding and reproduce the exact original length.
    """

    _validate_settings(sample_rate, n_fft, hop_length)
    stft_matrix = np.asarray(complex_stft)

    if stft_matrix.ndim != 2:
        raise ValueError("complex_stft must be a two-dimensional matrix.")
    if stft_matrix.shape[0] != n_fft // 2 + 1:
        raise ValueError(
            f"complex_stft must have {n_fft // 2 + 1} frequency bins for "
            f"n_fft={n_fft}."
        )
    if stft_matrix.shape[1] == 0:
        raise ValueError("complex_stft must contain at least one time frame.")
    if not np.isfinite(stft_matrix).all():
        raise ValueError("complex_stft contains non-finite values.")
    if length is not None and (
        not isinstance(length, (int, np.integer)) or length < 0
    ):
        raise ValueError("length must be a non-negative integer or None.")

    # ISTFT restores each time-domain frame while preserving the phase stored
    # in the complex STFT coefficients.
    time_frames = np.fft.irfft(stft_matrix.T, n=n_fft, axis=1)
    window = hann_window(n_fft)
    window_power = window**2
    output_length = n_fft + (time_frames.shape[0] - 1) * hop_length
    reconstructed = np.zeros(output_length, dtype=np.float64)
    normalization = np.zeros(output_length, dtype=np.float64)

    for frame_index, frame in enumerate(time_frames):
        start = frame_index * hop_length
        stop = start + n_fft
        reconstructed[start:stop] += frame * window
        normalization[start:stop] += window_power

    usable_samples = normalization > np.finfo(np.float64).eps
    reconstructed[usable_samples] /= normalization[usable_samples]

    center_padding = n_fft // 2
    if length is None:
        end = reconstructed.size - center_padding
        return reconstructed[center_padding:end]

    required_length = center_padding + int(length)
    if reconstructed.size < required_length:
        reconstructed = np.pad(
            reconstructed,
            (0, required_length - reconstructed.size),
        )

    return reconstructed[center_padding:required_length]
