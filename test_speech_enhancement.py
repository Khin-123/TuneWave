"""Standalone CLI verification for classical adaptive speech enhancement."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from backend.dsp.audio_utils import load_audio
from backend.speech.wiener_enhancer import (
    SpeechEnhancementConfig,
    compute_snr_db,
    enhance_audio_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance a mono or stereo noisy-speech WAV with adaptive spectral subtraction"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-reference", type=Path, default=None)
    parser.add_argument("--compare-fixed-beta", action="store_true")
    parser.add_argument("--fixed-beta", action="store_true", help="Run only the fixed-beta reference")
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
    parser.add_argument("--beta", type=float, default=1.3)
    parser.add_argument("--spectral-floor", type=float, default=0.08)
    parser.add_argument("--noise-percentile", type=float, default=50.0)
    parser.add_argument("--wiener-strength", type=float, default=0.25)
    parser.add_argument("--wiener-min-gain", type=float, default=0.65)
    parser.add_argument("--min-gain", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument("--frequency-smoothing-bins", type=int, default=3)
    parser.add_argument("--temporal-smoothing-frames", type=int, default=3)
    parser.add_argument("--max-compensation-gain", type=float, default=1.5)
    parser.add_argument("--noise-tracking-alpha", type=float, default=0.04)
    parser.add_argument("--noise-tracking-snr-db", type=float, default=3.0)
    parser.add_argument("--noise-tracking-energy-percentile", type=float, default=40.0)
    parser.add_argument("--speech-activity-margin-db", type=float, default=6.0)
    return parser


def _attach_reference_snr(result: dict, input_path: Path, reference_path: Path | None) -> None:
    if reference_path is None:
        return
    noisy, sample_rate = load_audio(input_path, preserve_channels=True)
    reference, reference_rate = load_audio(reference_path, preserve_channels=True)
    enhanced, enhanced_rate = load_audio(result["paths"]["enhanced_speech"], preserve_channels=True)
    if sample_rate != reference_rate or sample_rate != enhanced_rate:
        raise ValueError("clean reference and input must have the same sample rate")
    result["synthetic_input_snr"] = compute_snr_db(reference, noisy)
    result["synthetic_output_snr"] = compute_snr_db(reference, enhanced)
    result["snr_improvement"] = result["synthetic_output_snr"] - result["synthetic_input_snr"]


def _print_result(result: dict, label: str = "adaptive") -> None:
    def number(value: object, digits: int = 3) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    print(f"{label} sample rate: {result['sample_rate']} Hz")
    print(f"{label} channels: {result['channels']}")
    print(f"{label} duration: {result['duration']:.3f} s")
    print(f"{label} frame size: {result['frame_size']} samples ({result['frame_duration_ms']:.1f} ms)")
    print(f"{label} hop size: {result['hop_size']} samples")
    print(f"{label} number of frames: {result['number_of_frames']}")
    print(f"{label} noise estimation: {result['noise_estimation_method']}")
    print(f"{label} noise frames selected: {result['noise_frames_selected']}")
    print(f"{label} noise-dominant frames: {result['number_of_noise_dominant_frames']}")
    print(f"{label} speech-active frames: {result['number_of_speech_active_frames']}")
    print(f"{label} speech-activity margin: {result['speech_activity_margin_db']:.1f} dB")
    print(f"{label} mean local SNR: {result['mean_local_snr_db']:.3f} dB")
    print(f"{label} beta min/mean/max: {result['beta_min']:.3f}/{result['beta_mean']:.3f}/{result['beta_max']:.3f}")
    print(f"{label} spectral floor: {result['spectral_floor']:.3f}")
    print(f"{label} Wiener strength: {result['wiener_strength']:.3f}")
    print(f"{label} input SNR: {number(result.get('synthetic_input_snr'))} dB")
    print(f"{label} output SNR: {number(result.get('synthetic_output_snr'))} dB")
    print(f"{label} SNR improvement: {number(result.get('snr_improvement'))} dB")
    print(f"{label} input RMS: {result['input_rms']:.6f}")
    print(f"{label} output RMS before compensation: {result['output_rms_before_compensation']:.6f}")
    print(f"{label} output RMS after compensation: {result['output_rms_after_compensation']:.6f}")
    print(f"{label} compensation gain: {result['compensation_gain']:.6f}")
    print(f"{label} output peak: {result['output_peak']:.6f}")
    print(f"{label} processing time: {result['processing_time']:.3f} s")
    print(f"{label} enhanced WAV: {result['paths']['enhanced_speech']}")


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
        adaptive_beta=not args.fixed_beta,
        noise_tracking_alpha=args.noise_tracking_alpha,
        noise_tracking_snr_db=args.noise_tracking_snr_db,
        noise_tracking_energy_percentile=args.noise_tracking_energy_percentile,
        speech_activity_margin_db=args.speech_activity_margin_db,
    )
    result = enhance_audio_file(args.input, args.output_dir, config)
    _attach_reference_snr(result, args.input, args.clean_reference)
    _print_result(result, "fixed-beta" if args.fixed_beta else "adaptive")

    if args.compare_fixed_beta:
        fixed_config = replace(
            config,
            beta=1.5,
            adaptive_beta=False,
            noise_tracking_alpha=0.0,
        )
        fixed_result = enhance_audio_file(
            args.input,
            args.output_dir / "fixed_beta_reference",
            fixed_config,
        )
        _attach_reference_snr(fixed_result, args.input, args.clean_reference)
        _print_result(fixed_result, "fixed-beta reference")
        if result.get("synthetic_output_snr") is not None and fixed_result.get("synthetic_output_snr") is not None:
            delta = result["synthetic_output_snr"] - fixed_result["synthetic_output_snr"]
            print(f"adaptive minus fixed-beta SNR: {delta:.3f} dB")
        print(f"fixed-beta reference WAV: {fixed_result['paths']['enhanced_speech']}")

    print("classical adaptive DSP speech enhancement: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())