'''CLI verification for the production two-stem chunked separator.'''

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.dsp.audio_utils import load_audio
from backend.separation.openunmix_dsp_separator import separate_audio_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Test chunked DSP plus pretrained Open-Unmix vocals/instrumental separation'
    )
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--chunk-seconds', type=float, default=30.0)
    parser.add_argument('--overlap-seconds', type=float, default=2.0)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--model', default='umxl')
    parser.add_argument('--niter', type=int, default=2)
    args = parser.parse_args()

    mixture, sample_rate = load_audio(args.input, preserve_channels=True)
    if mixture.ndim != 2 or mixture.shape[0] != 2:
        raise ValueError('The input file must contain stereo audio.')
    result = separate_audio_file(
        args.input,
        args.output_dir,
        model=args.model,
        device=args.device,
        niter=args.niter,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    print(f'input: {args.input}')
    print(f'input duration: {mixture.shape[1] / sample_rate:.3f}s')
    print(f'device: {result["device"]}')
    print(f'model targets: {result["targets"]}')
    print(f'chunks: {result["chunks"]}')
    print(f'model load seconds: {result["model_load_seconds"]:.3f}')
    print(f'processing seconds: {result["processing_seconds"]:.3f}')
    print(
        'chunk complex consistency max error: '
        f'{max(result["chunk_complex_errors"]):.6g}'
    )
    print(
        'chunk ISTFT consistency max error: '
        f'{max(result["chunk_waveform_errors"]):.6g}'
    )
    print(
        'overlap-add mixture max error: '
        f'{result["overlap_add_mixture_error"]:.6g}'
    )
    print(
        'pre-limiter reconstruction max error: '
        f'{result["pre_limit_reconstruction_error"]:.6g}'
    )
    print(
        'raw overlap-add source max error before residual correction: '
        f'{result["overlap_add_source_error"]:.6g}'
    )
    print(f'shared output gain: {result["shared_gain"]:.9f}')
    print(
        'chunk-boundary pre-limiter errors: '
        f'{result["boundary_reconstruction_errors"]}'
    )

    vocal, vocal_rate = sf.read(
        result['paths']['vocals'], always_2d=True, dtype='float32'
    )
    instrumental, instrumental_rate = sf.read(
        result['paths']['instrumental'], always_2d=True, dtype='float32'
    )
    for name, stem, stem_rate in (
        ('vocals', vocal, vocal_rate),
        ('instrumental', instrumental, instrumental_rate),
    ):
        if stem_rate != sample_rate or stem.shape != (mixture.shape[1], 2):
            raise RuntimeError(f'{name} has unexpected rate, channels, or duration')
        if not np.isfinite(stem).all() or np.max(np.abs(stem)) > 1.0:
            raise RuntimeError(f'{name} contains invalid or clipping samples')
    reconstruction_error = float(
        np.max(
            np.abs(
                vocal + instrumental
                - mixture.T * result['shared_gain']
            )
        )
    )
    print(
        'post-limiter reconstruction max error against gain-adjusted mixture: '
        f'{reconstruction_error:.6f}'
    )
    if reconstruction_error > 3 / 32767:
        raise RuntimeError(
            'vocal + instrumental does not reconstruct the gain-adjusted mixture'
        )
    print('chunked two-stem verification: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
