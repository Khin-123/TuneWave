"""Small, configurable Stage 5 training smoke test for the stereo U-Net.

This module intentionally performs only a short training run. It reuses the
lazy compressed MUSDB18 dataset from :mod:`ml_dataset` and the Stage 4 model;
it does not implement audio reconstruction, Flask integration, or production
training configuration.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from dsp.ml_preprocessing import PreprocessingConfig
from ml_dataset import MusicSeparationDataset
from models.vocal_unet import StereoUNet


DEFAULT_DATASET_ROOT = Path("dataset") / "musdb"
DEFAULT_CHECKPOINT_DIR = Path("models")


@dataclass(frozen=True)
class SmokeTrainingConfig:
    """All smoke-run settings in one place rather than scattered constants."""

    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-3
    max_train_chunks: int = 2
    max_valid_chunks: int = 2
    num_workers: int = 0
    seed: int = 1337
    device: str = "auto"
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    best_checkpoint_name: str = "smoke_best_model.pth"
    last_checkpoint_name: str = "smoke_last_model.pth"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.max_train_chunks <= 0 or self.max_valid_chunks <= 0:
            raise ValueError("chunk limits must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must not be negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for a repeatable smoke run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Select CUDA when requested/available, otherwise use CPU."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def limit_chunks(dataset: Dataset[Any], maximum: int) -> Dataset[Any]:
    """Return only the first few lazy chunks without decoding them at startup."""

    if len(dataset) == 0:
        raise ValueError("dataset contains no chunks")
    return Subset(dataset, range(min(len(dataset), maximum)))


def _collate_examples(batch: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Stage 3 NumPy examples to a float32 PyTorch batch."""

    inputs = np.stack([np.asarray(item.input_tensor) for item in batch], axis=0)
    targets = np.stack([np.asarray(item.target_mask) for item in batch], axis=0)
    return (
        torch.from_numpy(inputs).to(dtype=torch.float32),
        torch.from_numpy(targets).to(dtype=torch.float32),
    )


def _make_loader(dataset: Dataset[Any], config: SmokeTrainingConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=_collate_examples,
        pin_memory=False,
    )


def _validate_batch_shapes(inputs: torch.Tensor, targets: torch.Tensor) -> None:
    if inputs.ndim != 4 or targets.ndim != 4:
        raise RuntimeError("batches must have shape (B, 2, F, T)")
    if inputs.shape != targets.shape:
        raise RuntimeError(
            f"input/target shape mismatch: {tuple(inputs.shape)} vs {tuple(targets.shape)}"
        )
    if inputs.shape[1] != 2:
        raise RuntimeError("Stage 5 expects stereo input and target channels")
    if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise RuntimeError("dataset batch contains NaN or Inf")


def _run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for inputs, targets in loader:
        _validate_batch_shapes(inputs, targets)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        if predictions.shape != targets.shape:
            raise RuntimeError("model output shape does not match target shape")
        loss = criterion(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became NaN or Inf")
        loss.backward()
        optimizer.step()
        batch_size = inputs.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("training loader produced no batches")
    return total_loss / sample_count


@torch.no_grad()
def _run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    minimum = float("inf")
    maximum = float("-inf")
    for inputs, targets in loader:
        _validate_batch_shapes(inputs, targets)
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        if predictions.shape != targets.shape:
            raise RuntimeError("model output shape does not match target shape")
        loss = criterion(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("validation loss became NaN or Inf")
        minimum = min(minimum, float(predictions.min().cpu()))
        maximum = max(maximum, float(predictions.max().cpu()))
        batch_size = inputs.shape[0]
        total_loss += float(loss.cpu()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("validation loader produced no batches")
    return total_loss / sample_count, minimum, maximum


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    result: EpochResult,
    config: SmokeTrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": result.train_loss,
            "validation_loss": result.validation_loss,
            "learning_rate": result.learning_rate,
            "training_config": asdict(config),
        },
        path,
    )


def train_model(
    train_dataset: Dataset[Any],
    validation_dataset: Dataset[Any],
    config: SmokeTrainingConfig,
    model: nn.Module | None = None,
) -> list[EpochResult]:
    """Run the short train/validation loop and save best/last checkpoints."""

    set_reproducible_seed(config.seed)
    device = resolve_device(config.device)
    train_loader = _make_loader(train_dataset, config, shuffle=True)
    validation_loader = _make_loader(validation_dataset, config, shuffle=False)
    model = model or StereoUNet()
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = float("inf")
    results: list[EpochResult] = []
    best_path = config.checkpoint_dir / config.best_checkpoint_name
    last_path = config.checkpoint_dir / config.last_checkpoint_name

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_training_epoch(model, train_loader, optimizer, criterion, device)
        validation_loss, mask_minimum, mask_maximum = _run_validation_epoch(
            model, validation_loader, criterion, device
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        result = EpochResult(epoch, train_loss, validation_loss, learning_rate)
        results.append(result)
        _save_checkpoint(last_path, model, optimizer, epoch, result, config)
        if validation_loss < best_loss:
            best_loss = validation_loss
            _save_checkpoint(best_path, model, optimizer, epoch, result, config)
        print(
            f"epoch {epoch}/{config.epochs} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"lr={learning_rate:.2e} "
            f"mask_range=[{mask_minimum:.4f}, {mask_maximum:.4f}]"
        )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("training completed without writing checkpoints")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny TuneWave U-Net training smoke test")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-train-chunks", type=int, default=2)
    parser.add_argument("--max-valid-chunks", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = SmokeTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_chunks=args.max_train_chunks,
        max_valid_chunks=args.max_valid_chunks,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    preprocessing = PreprocessingConfig()
    train_dataset = MusicSeparationDataset(args.dataset_root, split="train", config=preprocessing)
    validation_dataset = MusicSeparationDataset(args.dataset_root, split="valid", config=preprocessing)
    train_dataset = limit_chunks(train_dataset, config.max_train_chunks)
    validation_dataset = limit_chunks(validation_dataset, config.max_valid_chunks)
    print(f"device: {resolve_device(config.device)}")
    print(f"train chunks: {len(train_dataset)}; validation chunks: {len(validation_dataset)}")
    train_model(train_dataset, validation_dataset, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
