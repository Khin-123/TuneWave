'''CLI verification for the explicit DSP plus Open-Unmix pipeline.'''

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.dsp.audio_utils import load_audio
from backend.separation.openunmix_dsp_separator import separate_audio_file_four_stem


EXPECTED_OUTPUTS = ('vocals', 'drums', 'bass', 'other', 'instrumental')


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Test explicit DSP plus pretrained Open-Unmix separation'
    )
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--poc-dir', type=Path)
    parser.add_argument('--model', default='umxhq')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--niter', type=int, default=1)
    parser.add_argument('--wiener-win-length', type=int, default=300)
    return parser


def _verify_outputs(
    directory: Path,
    input_duration: float,
    expected_sample_rate: int,
) -> None:
    for name in EXPECTED_OUTPUTS:
        path = directory / f'{name}.wav'
        if not path.is_file():
            raise RuntimeError(f'missing output file: {path}')
        audio, sample_rate = load_audio(path, preserve_channels=True)
        if sample_rate != expected_sample_rate:
            raise RuntimeError(f'unexpected sample rate for {path}: {sample_rate}')
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise RuntimeError(f'output is not stereo: {path}')
        if not np.isfinite(audio).all():
            raise RuntimeError(f'output contains non-finite values: {path}')
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            raise RuntimeError(f'output clips: {path} (peak {peak})')
        duration = audio.shape[1] / sample_rate
        if abs(duration - input_duration) > 0.05:
            raise RuntimeError(f'output duration differs from input: {path}')
        print(f'{name}: {path} ({sample_rate} Hz, stereo, {duration:.3f}s, peak={peak:.4f})')


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_audio, input_rate = load_audio(args.input, preserve_channels=True)
    if input_audio.ndim != 2 or input_audio.shape[0] != 2:
        raise ValueError('The input file must contain two audio channels.')
    input_duration = input_audio.shape[1] / input_rate
    result = separate_audio_file_four_stem(
        args.input,
        args.output_dir,
        model=args.model,
        device=args.device,
        niter=args.niter,
        wiener_win_length=args.wiener_win_length,
    )

    print(f'input filename: {args.input.name}')
    print(f'input sample rate: {input_rate} Hz')
    print(f'input channels: {input_audio.shape[0]}')
    processing_device = result['device']
    model_used = result['model']
    stft_shape = result['stft_shape']
    magnitude_shape = result['magnitude_shape']
    print(f'processing device: {processing_device}')
    print(f'model used: {model_used}')
    print(f'complex STFT shape: {stft_shape}')
    print(f'magnitude shape: {magnitude_shape}')
    print('explicit DSP plus Open-Unmix outputs:')
    _verify_outputs(args.output_dir, input_duration, result['sample_rate'])

    if args.poc_dir is not None:
        print('high-level Open-Unmix POC outputs:')
        _verify_outputs(args.poc_dir, input_duration, result['sample_rate'])
        print('POC comparison: both output sets passed format and duration checks')
    print('explicit DSP plus Open-Unmix verification: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
