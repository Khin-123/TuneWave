'''Explicit DSP plus pretrained Open-Unmix music separation.

Unlike the proof-of-concept wrapper, this module exposes the signal path used
by the final music separator: our stereo STFT and Hann window feed the
Open-Unmix spectrogram models, then Open-Unmix's multichannel Wiener routine
refines complex source spectra before DSP ISTFT reconstruction.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any
from functools import lru_cache
import logging
import gc
from time import perf_counter

import numpy as np
import soundfile as sf
import torch

try:
    from ..dsp.audio_utils import AudioLoadError, load_audio
except ImportError:
    from dsp.audio_utils import AudioLoadError, load_audio

from .openunmix_separator import (
    DEFAULT_MODEL as LEGACY_DEFAULT_MODEL,
    DEFAULT_PEAK_LIMIT,
    OPENUNMIX_SAMPLE_RATE,
    OPENUNMIX_TARGETS,
    _apply_shared_peak_limit,
    resolve_device,
)


DEFAULT_N_FFT = 4096
DEFAULT_HOP_LENGTH = 1024
DEFAULT_WIENER_ITERATIONS = 1
DEFAULT_WIENER_WIN_LENGTH = 300
PRODUCTION_DEFAULT_MODEL = 'umxl'
PRODUCTION_DEFAULT_WIENER_ITERATIONS = 2
DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_OVERLAP_SECONDS = 2.0

logger = logging.getLogger(__name__)


def _load_openunmix_core():
    try:
        from openunmix import filtering, utils
    except (ImportError, OSError) as error:
        raise RuntimeError(
            'Open-Unmix core dependencies could not be imported. '
            'Install backend/requirements.txt and a compatible torchaudio build.'
        ) from error
    return filtering, utils


@lru_cache(maxsize=8)
def _load_target_models(
    model: str,
    device_name: str,
    targets: tuple[str, ...] = OPENUNMIX_TARGETS,
):
    '''Load pretrained target models once per model/device pair.'''

    _, utils = _load_openunmix_core()
    return utils.load_target_models(
        targets=list(targets),
        model_str_or_path=model,
        device=device_name,
        pretrained=True,
    )


def _stereo_stft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    if sample_rate != OPENUNMIX_SAMPLE_RATE:
        raise AudioLoadError(
            f'Open-Unmix expects {OPENUNMIX_SAMPLE_RATE} Hz input; '
            f'got {sample_rate} Hz.'
        )
    window = torch.hann_window(n_fft, dtype=torch.float32)
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
    channels, samples = waveform.shape[1], waveform.shape[2]
    transformed = torch.stft(
        waveform.reshape(channels, samples),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        pad_mode='reflect',
        return_complex=True,
    )
    return transformed.reshape(1, channels, transformed.shape[-2], transformed.shape[-1])


def _estimate_source_magnitudes(
    mixture_magnitude: torch.Tensor,
    model: str,
    device: torch.device,
) -> torch.Tensor:
    target_models = _load_target_models(model, str(device))
    estimates = []
    with torch.no_grad():
        for target in OPENUNMIX_TARGETS:
            target_model = target_models[target].to(device).eval()
            estimate = target_model(mixture_magnitude.to(device))
            if estimate.shape != mixture_magnitude.shape:
                raise RuntimeError(
                    f'Open-Unmix {target} estimate shape {tuple(estimate.shape)} '
                    f'does not match mixture shape {tuple(mixture_magnitude.shape)}'
                )
            estimates.append(estimate)
    return torch.stack(estimates, dim=-1)


def _estimate_vocal_magnitude(
    mixture_magnitude: torch.Tensor,
    vocal_model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    '''Estimate only vocals for the production two-stem path.'''

    with torch.inference_mode():
        estimate = vocal_model(mixture_magnitude.to(device))
    if estimate.shape != mixture_magnitude.shape:
        raise RuntimeError(
            f'Open-Unmix vocals estimate shape {tuple(estimate.shape)} '
            f'does not match mixture shape {tuple(mixture_magnitude.shape)}'
        )
    return estimate


def _wiener_source_stft(
    mixture_stft: torch.Tensor,
    source_magnitudes: torch.Tensor,
    filtering: Any,
    iterations: int,
    wiener_win_length: int,
    residual: bool = False,
) -> torch.Tensor:
    '''Apply Open-Unmix multichannel Wiener filtering in bounded frame windows.'''

    if iterations < 0 or wiener_win_length <= 0:
        raise ValueError('Wiener iterations must be non-negative and window positive')
    # Open-Unmix filtering expects (frames, bins, channels, complex) and
    # (frames, bins, channels, sources) magnitude estimates.
    mixture_for_filter = torch.view_as_real(mixture_stft).permute(0, 3, 2, 1, 4)
    magnitudes_for_filter = source_magnitudes.permute(0, 3, 2, 1, 4)
    filtered_batches = []
    output_source_count = source_magnitudes.shape[-1] + (1 if residual else 0)
    for batch_index in range(mixture_for_filter.shape[0]):
        frame_count = mixture_for_filter.shape[1]
        filtered = torch.zeros(
            frame_count,
            mixture_for_filter.shape[2],
            mixture_for_filter.shape[3],
            2,
            output_source_count,
            dtype=torch.float64,
            device=mixture_stft.device,
        )
        mixture_batch = mixture_for_filter[batch_index].to(torch.float64)
        magnitude_batch = magnitudes_for_filter[batch_index].to(torch.float64)
        for start in range(0, frame_count, wiener_win_length):
            stop = min(frame_count, start + wiener_win_length)
            filtered[start:stop] = filtering.wiener(
                magnitude_batch[start:stop],
                mixture_batch[start:stop],
                iterations=iterations,
                softmask=False,
                residual=residual,
            )
        filtered_batches.append(filtered)
    # Return (batch, sources, channels, bins, frames, complex).
    return torch.stack(filtered_batches, dim=0).permute(0, 5, 3, 2, 1, 4).contiguous()


def _chunk_starts(
    sample_count: int,
    chunk_samples: int,
    overlap_samples: int,
) -> list[int]:
    if chunk_samples <= 0:
        raise ValueError('chunk length must be positive')
    if overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError('overlap must be non-negative and shorter than chunk length')
    if sample_count <= chunk_samples:
        return [0]
    step = chunk_samples - overlap_samples
    starts = list(range(0, sample_count - chunk_samples + 1, step))
    final_start = sample_count - chunk_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _chunk_crossfade_weights(
    index: int,
    starts: list[int],
    chunk_samples: int,
    valid_samples: int,
) -> np.ndarray:
    '''Return a linear crossfade window for one chunk.'''

    weights = np.ones(valid_samples, dtype=np.float32)
    if index > 0:
        overlap = min(
            valid_samples,
            max(0, starts[index - 1] + chunk_samples - starts[index]),
        )
        if overlap:
            weights[:overlap] = np.linspace(
                0.0, 1.0, overlap, endpoint=False, dtype=np.float32
            )
    if index + 1 < len(starts):
        overlap = min(
            valid_samples,
            max(0, starts[index] + chunk_samples - starts[index + 1]),
        )
        if overlap:
            weights[-overlap:] = np.linspace(
                1.0, 0.0, overlap, endpoint=False, dtype=np.float32
            )
    return weights


def _separate_two_stem_chunk(
    chunk_audio: np.ndarray,
    sample_rate: int,
    device: torch.device,
    vocal_model: torch.nn.Module,
    filtering: Any,
    iterations: int,
    wiener_win_length: int,
    n_fft: int,
    hop_length: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    tuple[int, ...],
    float,
    float,
]:
    '''Separate one chunk and return channel-first vocal/instrumental audio.'''

    mixture_stft = _stereo_stft(chunk_audio, sample_rate, n_fft, hop_length)
    mixture_magnitude = torch.abs(mixture_stft)
    stft_shape = tuple(int(value) for value in mixture_stft.shape)
    magnitude_shape = tuple(int(value) for value in mixture_magnitude.shape)
    vocal_magnitude = _estimate_vocal_magnitude(
        mixture_magnitude,
        vocal_model,
        device,
    )
    source_magnitudes = vocal_magnitude.unsqueeze(-1)
    filtered = _wiener_source_stft(
        mixture_stft.to(device),
        source_magnitudes,
        filtering,
        iterations,
        wiener_win_length,
        residual=True,
    )
    # Open-Unmix residual Wiener output has [vocals, residual]. Recompute the
    # residual from the original complex mixture to guarantee mixture
    # consistency in every time-frequency bin.
    source_complex = torch.view_as_complex(filtered)
    vocal_complex = source_complex[0, 0]
    instrumental_complex = mixture_stft.to(device)[0] - vocal_complex
    complex_consistency_error = float(
        torch.max(
            torch.abs(vocal_complex + instrumental_complex - mixture_stft.to(device)[0])
        )
        .detach()
        .cpu()
    )
    reconstruction = torch.stack((vocal_complex, instrumental_complex), dim=0)
    source_audio = _stereo_istft(
        reconstruction.to('cpu'),
        chunk_audio.shape[1],
        n_fft,
        hop_length,
    )
    reconstructed_mixture = _stereo_istft(
        mixture_stft[0].unsqueeze(0),
        chunk_audio.shape[1],
        n_fft,
        hop_length,
    )[0]
    waveform_consistency_error = float(
        np.max(np.abs(source_audio[0] + source_audio[1] - reconstructed_mixture))
    )
    del (
        mixture_stft,
        mixture_magnitude,
        vocal_magnitude,
        source_magnitudes,
        filtered,
        source_complex,
        vocal_complex,
        instrumental_complex,
        reconstruction,
    )
    return (
        source_audio[0],
        source_audio[1],
        stft_shape,
        magnitude_shape,
        complex_consistency_error,
        waveform_consistency_error,
    )


def _stereo_istft(
    source_stft: torch.Tensor,
    length: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    window = torch.hann_window(n_fft, dtype=source_stft.real.dtype)
    reconstructed = []
    for source_index in range(source_stft.shape[0]):
        channel_audio = []
        for channel_index in range(source_stft.shape[1]):
            channel_audio.append(
                torch.istft(
                    source_stft[source_index, channel_index],
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    center=True,
                    normalized=False,
                    onesided=True,
                    length=length,
                )
            )
        reconstructed.append(torch.stack(channel_audio, dim=0))
    return torch.stack(reconstructed, dim=0).detach().cpu().numpy().astype(np.float32)


def separate_audio_file_four_stem(
    input_path: str | Path,
    output_directory: str | Path,
    model: str = LEGACY_DEFAULT_MODEL,
    device: str = 'auto',
    niter: int = DEFAULT_WIENER_ITERATIONS,
    wiener_win_length: int = DEFAULT_WIENER_WIN_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    peak_limit: float = DEFAULT_PEAK_LIMIT,
) -> dict[str, Any]:
    '''Separate one 44.1 kHz stereo file through explicit DSP and Open-Unmix.'''

    source = Path(input_path)
    audio, sample_rate = load_audio(source, preserve_channels=True)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise AudioLoadError('The explicit Open-Unmix pipeline requires stereo input.')
    processing_device = resolve_device(device)
    filtering, _ = _load_openunmix_core()
    mixture_stft = _stereo_stft(audio, sample_rate, n_fft, hop_length)
    mixture_magnitude = torch.abs(mixture_stft)
    source_magnitudes = _estimate_source_magnitudes(
        mixture_magnitude,
        model,
        processing_device,
    )
    source_stft = _wiener_source_stft(
        mixture_stft.to(processing_device),
        source_magnitudes,
        filtering,
        niter,
        wiener_win_length,
    )
    source_stft_complex = torch.view_as_complex(source_stft).to('cpu')
    source_audio = _stereo_istft(
        source_stft_complex[0],
        audio.shape[1],
        n_fft,
        hop_length,
    )
    stems = {
        target: source_audio[index]
        for index, target in enumerate(OPENUNMIX_TARGETS)
    }
    stems['instrumental'] = stems['drums'] + stems['bass'] + stems['other']
    stems = _apply_shared_peak_limit(stems, peak_limit)

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, stem in stems.items():
        path = destination / f'{name}.wav'
        sf.write(path, stem.T, OPENUNMIX_SAMPLE_RATE, subtype='PCM_16')
        paths[name] = path
    return {
        'input_path': source,
        'input_sample_rate': int(sample_rate),
        'sample_rate': OPENUNMIX_SAMPLE_RATE,
        'channels': 2,
        'samples': int(audio.shape[1]),
        'device': str(processing_device),
        'model': model,
        'paths': paths,
        'stft_shape': tuple(int(value) for value in mixture_stft.shape),
        'magnitude_shape': tuple(int(value) for value in mixture_magnitude.shape),
    }


def separate_audio_file(
    input_path: str | Path,
    output_directory: str | Path,
    model: str = PRODUCTION_DEFAULT_MODEL,
    device: str = 'auto',
    niter: int = PRODUCTION_DEFAULT_WIENER_ITERATIONS,
    wiener_win_length: int = DEFAULT_WIENER_WIN_LENGTH,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    peak_limit: float = DEFAULT_PEAK_LIMIT,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> dict[str, Any]:
    '''Production vocal/instrumental separation using bounded chunks.

    Only the pretrained vocals target is loaded. Open-Unmix's residual Wiener
    mode supplies a two-source initialization, while the final instrumental
    complex spectrum is explicitly the mixture minus vocals.
    '''

    if niter < 0:
        raise ValueError('niter must be non-negative')
    if not np.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError('chunk_seconds must be positive')
    if not np.isfinite(overlap_seconds) or overlap_seconds < 0:
        raise ValueError('overlap_seconds must be non-negative')
    if overlap_seconds >= chunk_seconds:
        raise ValueError('overlap_seconds must be shorter than chunk_seconds')

    source = Path(input_path)
    audio, sample_rate = load_audio(source, preserve_channels=True)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise AudioLoadError('The explicit Open-Unmix pipeline requires stereo input.')

    processing_device = resolve_device(device)
    chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
    overlap_samples = int(round(overlap_seconds * sample_rate))
    starts = _chunk_starts(audio.shape[1], chunk_samples, overlap_samples)
    logger.info(
        'music separation input=%.2fs chunks=%d device=%s chunk=%.2fs overlap=%.2fs',
        audio.shape[1] / sample_rate,
        len(starts),
        processing_device,
        chunk_seconds,
        overlap_seconds,
    )

    model_load_started = perf_counter()
    filtering, _ = _load_openunmix_core()
    target_models = _load_target_models(
        model,
        str(processing_device),
        targets=('vocals',),
    )
    vocal_model = target_models['vocals'].to(processing_device).eval()
    model_load_elapsed = perf_counter() - model_load_started
    logger.info(
        'Open-Unmix model=%s targets=vocals model_load_seconds=%.3f',
        model,
        model_load_elapsed,
    )

    vocal_accumulator = np.zeros_like(audio, dtype=np.float64)
    instrumental_accumulator = np.zeros_like(audio, dtype=np.float64)
    mixture_accumulator = np.zeros_like(audio, dtype=np.float64)
    weight_accumulator = np.zeros(audio.shape[1], dtype=np.float64)
    first_stft_shape: tuple[int, ...] | None = None
    first_magnitude_shape: tuple[int, ...] | None = None
    chunk_complex_errors: list[float] = []
    chunk_waveform_errors: list[float] = []
    total_started = perf_counter()

    with torch.inference_mode():
        for chunk_index, start in enumerate(starts):
            valid_samples = min(chunk_samples, audio.shape[1] - start)
            chunk = np.zeros((2, chunk_samples), dtype=np.float32)
            chunk[:, :valid_samples] = audio[:, start:start + valid_samples]
            chunk_started = perf_counter()
            (
                chunk_vocals,
                chunk_instrumental,
                stft_shape,
                magnitude_shape,
                chunk_complex_error,
                chunk_waveform_error,
            ) = _separate_two_stem_chunk(
                chunk,
                sample_rate,
                processing_device,
                vocal_model,
                filtering,
                niter,
                wiener_win_length,
                n_fft,
                hop_length,
            )
            if first_stft_shape is None:
                first_stft_shape = stft_shape
                first_magnitude_shape = magnitude_shape
            end = start + valid_samples
            weights = _chunk_crossfade_weights(
                chunk_index,
                starts,
                chunk_samples,
                valid_samples,
            )
            vocal_accumulator[:, start:end] += (
                chunk_vocals[:, :valid_samples] * weights[None, :]
            )
            instrumental_accumulator[:, start:end] += (
                chunk_instrumental[:, :valid_samples] * weights[None, :]
            )
            mixture_accumulator[:, start:end] += (
                chunk[:, :valid_samples] * weights[None, :]
            )
            weight_accumulator[start:end] += weights
            chunk_complex_errors.append(chunk_complex_error)
            chunk_waveform_errors.append(chunk_waveform_error)
            elapsed = perf_counter() - chunk_started
            logger.info(
                'music separation chunk=%d/%d start=%.2fs duration=%.2fs elapsed=%.3fs '
                'complex_error=%.3g waveform_error=%.3g',
                chunk_index + 1,
                len(starts),
                start / sample_rate,
                valid_samples / sample_rate,
                elapsed,
                chunk_complex_error,
                chunk_waveform_error,
            )
            del chunk, chunk_vocals, chunk_instrumental, weights
            gc.collect()
            if processing_device.type == 'cuda':
                torch.cuda.empty_cache()

    if np.any(weight_accumulator <= 0):
        raise RuntimeError('Chunk overlap-add produced uncovered output samples.')
    vocals = (vocal_accumulator / weight_accumulator[None, :]).astype(np.float32)
    overlap_add_mixture = (
        mixture_accumulator / weight_accumulator[None, :]
    ).astype(np.float32)
    instrumental_ola = (
        instrumental_accumulator / weight_accumulator[None, :]
    ).astype(np.float32)
    overlap_add_mixture_error = float(
        np.max(np.abs(overlap_add_mixture - audio))
    )
    overlap_add_source_error = float(
        np.max(np.abs(vocals + instrumental_ola - overlap_add_mixture))
    )
    # Enforce the invariant after identical overlap-add weighting. This also
    # removes any tiny independent ISTFT/OLA round-off from the residual stem.
    instrumental = overlap_add_mixture - vocals
    pre_limit_reconstruction_error = float(
        np.max(np.abs(vocals + instrumental - overlap_add_mixture))
    )
    boundary_reconstruction_errors = tuple(
        float(
            np.max(
                np.abs(
                    (vocals + instrumental)[:, max(0, start - 1):min(audio.shape[1], start + 1)]
                    - overlap_add_mixture[:, max(0, start - 1):min(audio.shape[1], start + 1)]
                )
            )
        )
        for start in starts[1:]
    )
    logger.info(
        'music separation overlap_add_mixture_error=%.3g '
        'overlap_add_source_error=%.3g pre_limit_reconstruction_error=%.3g',
        overlap_add_mixture_error,
        overlap_add_source_error,
        pre_limit_reconstruction_error,
    )
    peak = max(float(np.max(np.abs(vocals))), float(np.max(np.abs(instrumental))))
    shared_gain = 1.0
    if peak > peak_limit:
        shared_gain = peak_limit / peak
    stems = {
        'vocals': vocals * shared_gain,
        'instrumental': instrumental * shared_gain,
    }
    post_limit_reconstruction_error = float(
        np.max(np.abs(stems['vocals'] + stems['instrumental'] - audio * shared_gain))
    )
    logger.info(
        'music separation shared_gain=%.7f post_limit_reconstruction_error=%.3g',
        shared_gain,
        post_limit_reconstruction_error,
    )

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, stem in stems.items():
        path = destination / f'{name}.wav'
        sf.write(path, stem.T, OPENUNMIX_SAMPLE_RATE, subtype='PCM_16')
        paths[name] = path

    total_elapsed = perf_counter() - total_started
    logger.info(
        'music separation complete duration=%.2fs chunks=%d elapsed=%.3fs',
        audio.shape[1] / sample_rate,
        len(starts),
        total_elapsed,
    )
    return {
        'input_path': source,
        'input_sample_rate': int(sample_rate),
        'sample_rate': OPENUNMIX_SAMPLE_RATE,
        'channels': 2,
        'samples': int(audio.shape[1]),
        'device': str(processing_device),
        'model': model,
        'targets': ('vocals',),
        'chunks': len(starts),
        'chunk_seconds': float(chunk_seconds),
        'overlap_seconds': float(overlap_seconds),
        'paths': paths,
        'stft_shape': first_stft_shape,
        'magnitude_shape': first_magnitude_shape,
        'model_load_seconds': model_load_elapsed,
        'processing_seconds': total_elapsed,
        'chunk_starts': tuple(int(start) for start in starts),
        'chunk_complex_errors': tuple(chunk_complex_errors),
        'chunk_waveform_errors': tuple(chunk_waveform_errors),
        'overlap_add_mixture_error': overlap_add_mixture_error,
        'overlap_add_source_error': overlap_add_source_error,
        'pre_limit_reconstruction_error': pre_limit_reconstruction_error,
        'boundary_reconstruction_errors': boundary_reconstruction_errors,
        'shared_gain': shared_gain,
        'post_limit_reconstruction_error': post_limit_reconstruction_error,
    }
