'''Proof-of-concept separation using the pretrained Open-Unmix API.

This stage intentionally delegates model loading, STFT processing, Wiener
post-processing, and reconstruction to Open-Unmix.  The project-specific DSP
STFT/Wiener integration is reserved for a later stage.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

try:
    from ..dsp.audio_utils import AudioLoadError, load_audio
except ImportError:
    from dsp.audio_utils import AudioLoadError, load_audio


OPENUNMIX_TARGETS = ('vocals', 'drums', 'bass', 'other')
OPENUNMIX_SAMPLE_RATE = 44_100
DEFAULT_MODEL = 'umxhq'
DEFAULT_NITER = 1
DEFAULT_PEAK_LIMIT = 0.99


def resolve_device(requested: str = 'auto') -> torch.device:
    '''Resolve inference device, requiring CUDA only when explicitly requested.'''

    requested = requested.lower()
    if requested not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('device must be auto, cpu, or cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but no CUDA device is available')
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


def _load_predict_module():
    try:
        from openunmix import predict
    except ImportError as error:
        raise RuntimeError(
            'Open-Unmix is not installed. Install backend/requirements.txt first.'
        ) from error
    return predict


def _to_stereo_array(value: Any, target: str) -> np.ndarray:
    '''Convert an Open-Unmix estimate to finite channel-first stereo audio.'''

    if isinstance(value, torch.Tensor):
        value = value.detach().to('cpu').numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f'Open-Unmix returned an unexpected {target} batch shape')
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f'Open-Unmix returned an unexpected {target} shape')
    if array.shape[0] != 2 and array.shape[1] == 2:
        array = array.T
    if array.shape[0] != 2 or array.shape[1] == 0:
        raise ValueError(f'Open-Unmix returned a non-stereo {target} estimate')
    if not np.isfinite(array).all():
        raise ValueError(f'Open-Unmix returned non-finite {target} samples')
    return np.ascontiguousarray(array)


def _pad_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[1] == length:
        return audio
    padded = np.zeros((2, length), dtype=np.float32)
    padded[:, : min(length, audio.shape[1])] = audio[:, :length]
    return padded


def _apply_shared_peak_limit(
    stems: dict[str, np.ndarray],
    peak_limit: float,
) -> dict[str, np.ndarray]:
    if not np.isfinite(peak_limit) or peak_limit <= 0 or peak_limit > 1:
        raise ValueError('peak_limit must be in the interval (0, 1]')
    peak = max(float(np.max(np.abs(audio))) for audio in stems.values())
    if peak <= peak_limit:
        return stems
    gain = peak_limit / peak
    return {name: audio * gain for name, audio in stems.items()}


def separate_audio_file(
    input_path: str | Path,
    output_directory: str | Path,
    model: str = DEFAULT_MODEL,
    device: str = 'auto',
    niter: int = DEFAULT_NITER,
    peak_limit: float = DEFAULT_PEAK_LIMIT,
) -> dict[str, Any]:
    '''Separate one stereo file and write four stems plus instrumental.'''

    if niter < 0:
        raise ValueError('niter must be non-negative')
    source = Path(input_path)
    audio, input_sample_rate = load_audio(source, preserve_channels=True)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise AudioLoadError('Open-Unmix proof of concept requires stereo input.')
    processing_device = resolve_device(device)
    predict = _load_predict_module()
    mixture = torch.from_numpy(np.ascontiguousarray(audio)).float()

    estimates = predict.separate(
        audio=mixture,
        rate=int(input_sample_rate),
        model_str_or_path=model,
        targets=list(OPENUNMIX_TARGETS),
        niter=niter,
        device=str(processing_device),
        filterbank='torch',
    )
    stems = {
        target: _to_stereo_array(estimates[target], target)
        for target in OPENUNMIX_TARGETS
    }
    output_length = max(audio.shape[1], *(stem.shape[1] for stem in stems.values()))
    stems = {name: _pad_to_length(stem, output_length) for name, stem in stems.items()}
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
        'input_sample_rate': int(input_sample_rate),
        'sample_rate': OPENUNMIX_SAMPLE_RATE,
        'channels': 2,
        'samples': output_length,
        'device': str(processing_device),
        'model': model,
        'paths': paths,
    }
