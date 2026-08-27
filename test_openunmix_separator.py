'''CLI proof-of-concept test for the pretrained Open-Unmix separator.'''

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.dsp.audio_utils import load_audio
from backend.separation.openunmix_separator import separate_audio_file


EXPECTED_OUTPUTS = ('vocals', 'drums', 'bass', 'other', 'instrumental')


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Test pretrained Open-Unmix separation')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='umxhq')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--niter', type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_audio, input_rate = load_audio(args.input, preserve_channels=True)
    if input_audio.ndim != 2 or input_audio.shape[0] != 2:
        raise ValueError('The input file must contain two audio channels.')
    input_duration = input_audio.shape[1] / input_rate
    result = separate_audio_file(
        args.input,
        args.output_dir,
        model=args.model,
        device=args.device,
        niter=args.niter,
    )

    print(f'input filename: {args.input.name}')
    output_rate = result['sample_rate']
    processing_device = result['device']
    model_used = result['model']
    print(f'sample rate: {input_rate} Hz input, {output_rate} Hz output')
    print(f'input channels: {input_audio.shape[0]}')
    print(f'processing device: {processing_device}')
    print(f'model used: {model_used}')

    for name in EXPECTED_OUTPUTS:
        path = result['paths'][name]
        if not path.is_file():
            raise RuntimeError(f'missing output file: {path}')
        audio, sample_rate = load_audio(path, preserve_channels=True)
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise RuntimeError(f'output is not stereo: {path}')
        if not np.isfinite(audio).all():
            raise RuntimeError(f'output contains non-finite values: {path}')
        duration = audio.shape[1] / sample_rate
        if abs(duration - input_duration) > 0.5:
            raise RuntimeError(f'output duration differs from input: {path}')
        print(f'{name}: {path} ({sample_rate} Hz, {audio.shape[0]} channels, {duration:.3f}s)')

    print('Open-Unmix proof-of-concept verification: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
