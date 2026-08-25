"""Stage 6 DSP -> U-Net -> DSP reconstruction for one audio chunk.

The U-Net predicts only a stereo mask. This module keeps the original complex
mixture STFT, applies the predicted and complementary masks, and reconstructs
both outputs with the existing DSP ISTFT implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dsp.ml_preprocessing import (
    PreprocessingConfig,
    extract_stereo_features,
    stereo_istft,
)
from ml_dataset import CompressedStemChunk, MusicSeparationDataset
from models.vocal_unet import StereoUNet
from train_vocal_unet import resolve_device


DEFAULT_DATASET_ROOT = Path("dataset") / "musdb"
DEFAULT_CHECKPOINT = Path("models") / "stage5_smoke" / "smoke_best_model.pth"
DEFAULT_OUTPUT_DIR = Path("outputs") / "stage6_reconstruction"


@dataclass(frozen=True)
class ReconstructionResult:
    """In-memory result of one chunk reconstruction."""

    mixture: np.ndarray
    ground_truth_vocals: np.ndarray
    ground_truth_instrumental: np.ndarray
    predicted_vocals: np.ndarray
    predicted_instrumental: np.ndarray
    vocal_mask: np.ndarray


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device_name: str = "auto",
) -> tuple[StereoUNet, torch.device]:
    """Load a Stage 5 checkpoint once and return an evaluation-mode model."""

    device = resolve_device(device_name)
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain a model_state_dict")

    model = StereoUNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def _validate_stereo(audio: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(audio)
    if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must have shape (2, samples)")
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite real samples")
    return np.ascontiguousarray(values, dtype=np.float32)


def reconstruct_chunk(
    chunk: CompressedStemChunk,
    model: StereoUNet,
    config: PreprocessingConfig,
    device: torch.device | None = None,
) -> ReconstructionResult:
    """Predict a mask and reconstruct stereo vocals/instrumental audio."""

    mixture = _validate_stereo(chunk.mixture, "mixture")
    vocals = _validate_stereo(chunk.vocals, "ground-truth vocals")
    instrumental = _validate_stereo(chunk.instrumental, "ground-truth instrumental")
    if not (mixture.shape == vocals.shape == instrumental.shape):
        raise ValueError("mixture and reference stems must have matching stereo shapes")

    features = extract_stereo_features(mixture, config.sample_rate, config)
    model_device = device or next(model.parameters()).device
    model_input = torch.from_numpy(features.normalized_magnitude[None]).to(model_device)
    with torch.no_grad():
        predicted_mask_tensor = model(model_input)
    if predicted_mask_tensor.shape != model_input.shape:
        raise RuntimeError("U-Net output shape does not match its input shape")
    if not torch.isfinite(predicted_mask_tensor).all():
        raise RuntimeError("predicted vocal mask contains NaN or Inf")
    predicted_mask_tensor = predicted_mask_tensor.clamp(0.0, 1.0)
    vocal_mask = predicted_mask_tensor[0].cpu().numpy().astype(np.float32, copy=False)
    instrument_mask = 1.0 - vocal_mask

    # Apply masks to the original complex mixture STFT. The normalized ML
    # feature tensor is never used for reconstruction.
    mixture_stft = features.complex_stft
    predicted_vocal_stft = vocal_mask * mixture_stft
    predicted_instrument_stft = instrument_mask * mixture_stft
    predicted_vocals = stereo_istft(
        predicted_vocal_stft,
        config.sample_rate,
        config,
        length=mixture.shape[1],
    )
    predicted_instrumental = stereo_istft(
        predicted_instrument_stft,
        config.sample_rate,
        config,
        length=mixture.shape[1],
    )
    return ReconstructionResult(
        mixture=mixture,
        ground_truth_vocals=vocals,
        ground_truth_instrumental=instrumental,
        predicted_vocals=_validate_stereo(predicted_vocals, "predicted vocals"),
        predicted_instrumental=_validate_stereo(
            predicted_instrumental, "predicted instrumental"
        ),
        vocal_mask=vocal_mask,
    )


def _prevent_clipping(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    values = _validate_stereo(audio, "audio")
    if not np.isfinite(peak) or peak <= 0 or peak > 1:
        raise ValueError("peak must be in the interval (0, 1]")
    maximum = float(np.max(np.abs(values)))
    if maximum > peak:
        values = values * (peak / maximum)
    return np.ascontiguousarray(values, dtype=np.float32)


def write_stereo_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
) -> Path:
    """Write a finite stereo WAV, scaling only when necessary to avoid clipping."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_audio = _prevent_clipping(audio)
    sf.write(output_path, safe_audio.T, int(sample_rate), subtype="PCM_16")
    info = sf.info(output_path)
    if info.channels != 2 or info.samplerate != int(sample_rate):
        raise RuntimeError(f"WAV verification failed for {output_path}")
    return output_path


def reconstruct_dataset_chunk(
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: PreprocessingConfig | None = None,
    split: str = "valid",
    chunk_index: int = 0,
    device_name: str = "auto",
) -> dict[str, Path]:
    """Decode one lazy dataset chunk and write predicted/reference WAV files."""

    used_config = config or PreprocessingConfig()
    model, device = load_checkpoint_model(checkpoint_path, device_name)
    dataset = MusicSeparationDataset(dataset_root, split=split, config=used_config)
    chunk = dataset.load_chunk(chunk_index)
    result = reconstruct_chunk(chunk, model, used_config, device)
    destination = Path(output_dir)
    outputs = {
        "mixture": destination / "mixture.wav",
        "ground_truth_vocals": destination / "ground_truth_vocals.wav",
        "ground_truth_instrumental": destination / "ground_truth_instrumental.wav",
        "predicted_vocals": destination / "predicted_vocals.wav",
        "predicted_instrumental": destination / "predicted_instrumental.wav",
    }
    write_stereo_wav(outputs["mixture"], result.mixture, used_config.sample_rate)
    write_stereo_wav(
        outputs["ground_truth_vocals"], result.ground_truth_vocals, used_config.sample_rate
    )
    write_stereo_wav(
        outputs["ground_truth_instrumental"],
        result.ground_truth_instrumental,
        used_config.sample_rate,
    )
    write_stereo_wav(outputs["predicted_vocals"], result.predicted_vocals, used_config.sample_rate)
    write_stereo_wav(
        outputs["predicted_instrumental"],
        result.predicted_instrumental,
        used_config.sample_rate,
    )
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct one TuneWave U-Net audio chunk")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    outputs = reconstruct_dataset_chunk(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        chunk_index=args.chunk_index,
        device_name=args.device,
    )
    for name, path in outputs.items():
        info = sf.info(path)
        print(f"{name}: {path} ({info.samplerate} Hz, {info.channels} channels, {info.duration:.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
