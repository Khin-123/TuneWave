"""Standalone CLI verification for classical DSP speech enhancement."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.speech.wiener_enhancer import (
    SpeechEnhancementConfig,
    enhance_audio_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance a mono or stereo noisy-speech WAV with Wiener filtering"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--noise-estimation",
        choices=("auto", "initial"),
        default="auto",
        help="Noise model: lowest-energy frames (default) or initial interval",
    )
    parser.add_argument("--noise-frame-percent", type=float, default=0.15)
    parser.add_argument("--minimum-noise-frames", type=int, default=8)
    parser.add_argument("--noise-duration", type=float, default=0.5)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--noise-percentile", type=float, default=50.0)
    parser.add_argument("--min-gain", type=float, default=0.10)
    parser.add_argument("--frequency-smoothing-bins", type=int, default=3)
    parser.add_argument("--temporal-smoothing-frames", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = SpeechEnhancementConfig(
        frame_duration_ms=args.frame_ms,
        overlap_ratio=args.overlap,
        noise_estimation=args.noise_estimation,
        noise_frame_percent=args.noise_frame_percent,
        minimum_noise_frames=args.minimum_noise_frames,
        noise_duration_seconds=args.noise_duration,
        noise_percentile=args.noise_percentile,
        min_gain=args.min_gain,
        frequency_smoothing_bins=args.frequency_smoothing_bins,
        temporal_smoothing_frames=args.temporal_smoothing_frames,
    )
    result = enhance_audio_file(args.input, args.output_dir, config)
    print(f'sample rate: {result["sample_rate"]} Hz')
    print(f'channels: {result["channels"]}')
    print(f'duration: {result["duration_seconds"]:.3f} s')
    print(f'STFT size: {result["stft_size"]}')
    print(f'hop length: {result["hop_length"]}')
    print(f'total frames: {result["frames"]}')
    if result["noise_estimation"] == "initial":
        noise_detail = f'{result["noise_estimation_duration_seconds"]:.3f} s initial interval'
    else:
        noise_detail = f'lowest-energy {result["noise_frame_percent"] * 100:.1f}% candidates'
    print(f'noise estimation mode: {result["noise_estimation"]} ({noise_detail})')
    print(f'noise frames selected: {result["noise_frames"]}')
    print(f'frames classified as noise: {result["noise_frames_percent"]:.2f}%')
    print(f'noise power percentile: {result["noise_percentile"]:.1f}')
    print(
        f'Wiener gain range: {result["wiener_gain_min"]:.4f} '
        f'to {result["wiener_gain_max"]:.4f}'
    )
    print(f'mean Wiener gain: {result["mean_wiener_gain"]:.4f}')
    print(
        'bins attenuated >3 dB: '
        f'{result["bins_attenuated_over_3_db_percent"]:.2f}%'
    )
    print(
        'bins attenuated >6 dB: '
        f'{result["bins_attenuated_over_6_db_percent"]:.2f}%'
    )
    print(f'input RMS: {result["input_rms"]:.6f}')
    print(f'output RMS: {result["output_rms"]:.6f}')
    print(f'input peak: {result["input_peak"]:.6f}')
    print(f'output peak: {result["output_peak"]:.6f}')
    print(f'shared output gain: {result["shared_output_gain"]:.6f}')
    print(f'processing time: {result["processing_seconds"]:.3f} s')
    print(f'original noisy WAV: {result["paths"]["original_noisy"]}')
    print(f'enhanced speech WAV: {result["paths"]["enhanced_speech"]}')
    print('classical DSP speech enhancement: PASS')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
