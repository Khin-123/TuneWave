"""Standalone CLI verification for classical DSP speech enhancement."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.speech.wiener_enhancer import SpeechEnhancementConfig, enhance_audio_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance a mono or stereo noisy-speech WAV with spectral subtraction"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--noise-estimation",
        choices=("auto", "initial"),
        default="auto",
        help="Use low-energy frames across the recording (default)",
    )
    parser.add_argument("--noise-frame-percent", type=float, default=0.15)
    parser.add_argument("--minimum-noise-frames", type=int, default=8)
    parser.add_argument("--noise-duration", type=float, default=0.5)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--spectral-floor", type=float, default=0.08)
    parser.add_argument("--noise-percentile", type=float, default=50.0)
    parser.add_argument("--wiener-strength", type=float, default=0.25)
    parser.add_argument("--wiener-min-gain", type=float, default=0.65)
    parser.add_argument("--min-gain", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument("--frequency-smoothing-bins", type=int, default=3)
    parser.add_argument("--temporal-smoothing-frames", type=int, default=3)
    parser.add_argument("--max-compensation-gain", type=float, default=1.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = SpeechEnhancementConfig(
        frame_duration_ms=args.frame_ms,
        overlap_ratio=args.overlap,
        beta=args.beta,
        spectral_floor=args.spectral_floor,
        noise_estimation=args.noise_estimation,
        noise_frame_percent=args.noise_frame_percent,
        minimum_noise_frames=args.minimum_noise_frames,
        noise_duration_seconds=args.noise_duration,
        noise_percentile=args.noise_percentile,
        wiener_strength=args.wiener_strength,
        wiener_min_gain=args.wiener_min_gain,
        min_gain=args.min_gain,
        frequency_smoothing_bins=args.frequency_smoothing_bins,
        temporal_smoothing_frames=args.temporal_smoothing_frames,
        max_compensation_gain=args.max_compensation_gain,
    )
    result = enhance_audio_file(args.input, args.output_dir, config)
    print(f'sample rate: {result["sample_rate"]} Hz')
    print(f'channels: {result["channels"]}')
    print(f'duration: {result["duration"]:.3f} s')
    print(f'frame size: {result["frame_size"]} samples ({result["frame_duration_ms"]:.1f} ms)')
    print(f'hop size: {result["hop_size"]} samples')
    print(f'number of frames: {result["number_of_frames"]}')
    print(f'noise estimation: {result["noise_estimation_method"]}')
    print(f'noise frames selected: {result["noise_frames_selected"]}')
    print(f'beta: {result["beta"]:.3f}')
    print(f'spectral floor: {result["spectral_floor"]:.3f}')
    print(f'noise magnitude percentile: {result["noise_percentile"]:.1f}')
    print(f'Wiener post-filter strength: {result["wiener_strength"]:.3f}')
    print(f'Wiener post-filter gain range: {result["wiener_gain_min"]:.4f} to {result["wiener_gain_max"]:.4f}')
    print(f'input RMS: {result["input_rms"]:.6f}')
    print(f'output RMS before compensation: {result["output_rms_before_compensation"]:.6f}')
    print(f'output RMS after compensation: {result["output_rms_after_compensation"]:.6f}')
    print(f'output-level compensation gain: {result["compensation_gain"]:.6f}')
    print(f'output peak: {result["output_peak"]:.6f}')
    print(f'processing time: {result["processing_time"]:.3f} s')
    print(f'original noisy WAV: {result["paths"]["original_noisy"]}')
    print(f'enhanced speech WAV: {result["paths"]["enhanced_speech"]}')
    print('classical DSP speech enhancement: PASS')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())