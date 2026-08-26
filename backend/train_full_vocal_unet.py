"""Stage 7 full-training configuration and short dry-run harness.

This module prepares a resource-aware training setup without launching a
long MUSDB18 job automatically. It opens every compressed track in ``train/``
as metadata, creates a deterministic track-level split, and decodes only the
chunks requested by a DataLoader.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from dsp.ml_preprocessing import PreprocessingConfig
from ml_dataset import MusicSeparationDataset
from models.vocal_unet import StereoUNet
from train_vocal_unet import resolve_device, set_reproducible_seed


DEFAULT_DATASET_ROOT = Path("dataset") / "musdb"
DEFAULT_CHECKPOINT_DIR = Path("models")


@dataclass(frozen=True)
class FullTrainingConfig:
    """Central configuration for eventual full training and dry runs."""

    dataset_root: Path = DEFAULT_DATASET_ROOT
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    sample_rate: int = 44_100
    n_fft: int = 2048
    hop_length: int = 512
    chunk_seconds: float = 6.0
    chunk_hop_seconds: float = 3.0
    validation_fraction: float = 0.2
    batch_size: int = 1
    learning_rate: float = 1e-3
    epochs: int = 50
    optimizer: str = "adam"
    loss: str = "l1"
    early_stopping_patience: int | None = 8
    checkpoint_interval: int = 1
    seed: int = 1337
    num_workers: int = 0
    device: str = "auto"
    mixed_precision: bool = True
    resume_from: Path | None = None
    max_train_tracks: int | None = None
    max_validation_tracks: int | None = None
    max_train_chunks: int | None = None
    max_valid_chunks: int | None = None
    best_checkpoint_name: str = "best_model.pth"
    last_checkpoint_name: str = "last_model.pth"

    def __post_init__(self) -> None:
        if self.sample_rate != 44_100:
            raise ValueError("MUSDB18 training must remain at 44.1 kHz")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.optimizer.lower() not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if self.loss.lower() not in {"l1", "mae", "mse"}:
            raise ValueError("loss must be 'l1', 'mae', or 'mse'")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative or None")
        if self.checkpoint_interval <= 0 or self.num_workers < 0:
            raise ValueError("checkpoint_interval must be positive and num_workers non-negative")
        for limit in (self.max_train_chunks, self.max_valid_chunks):
            if limit is not None and limit <= 0:
                raise ValueError("chunk limits must be positive when provided")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        for limit in (self.max_train_tracks, self.max_validation_tracks):
            if limit is not None and limit <= 0:
                raise ValueError("track limits must be positive when provided")

    def preprocessing(self) -> PreprocessingConfig:
        return PreprocessingConfig(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            chunk_seconds=self.chunk_seconds,
            chunk_hop_seconds=self.chunk_hop_seconds,
        )


@dataclass(frozen=True)
class TrackSplit:
    train_names: tuple[str, ...]
    validation_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedTrainingData:
    all_train_dataset: MusicSeparationDataset
    train_dataset: Dataset[Any]
    validation_dataset: Dataset[Any]
    split: TrackSplit


@dataclass(frozen=True)
class FullEpochResult:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


class TrackChunkDataset(Dataset[Any]):
    """View of Stage 3 chunks belonging only to selected track indices."""

    def __init__(self, base: MusicSeparationDataset, track_indices: Iterable[int]):
        self.base = base
        self.track_indices = tuple(int(index) for index in track_indices)
        allowed = set(self.track_indices)
        self._chunk_indices = tuple(
            chunk_index
            for chunk_index, record in enumerate(base._index)
            if record.track_index in allowed
        )
        if not self._chunk_indices:
            raise ValueError("selected tracks contain no chunks")

    @property
    def track_names(self) -> tuple[str, ...]:
        return tuple(str(self.base.tracks[index].name) for index in self.track_indices)

    def __len__(self) -> int:
        return len(self._chunk_indices)

    def __getitem__(self, index: int):
        return self.base[self._chunk_indices[index]]


def deterministic_track_split(
    track_names: Iterable[str],
    validation_fraction: float,
    seed: int,
) -> TrackSplit:
    """Partition unique train-track names reproducibly, never by chunk."""

    names = sorted({str(name) for name in track_names})
    if len(names) < 2:
        raise ValueError("at least two train tracks are required for a split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    shuffled = names[:]
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation = tuple(sorted(shuffled[:validation_count]))
    train = tuple(sorted(shuffled[validation_count:]))
    if set(train) & set(validation):
        raise RuntimeError("track split overlap detected")
    return TrackSplit(train_names=train, validation_names=validation)


def _limit_chunks(dataset: Dataset[Any], maximum: int | None) -> Dataset[Any]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))


def prepare_training_data(config: FullTrainingConfig) -> PreparedTrainingData:
    """Open all compressed train metadata, then select disjoint track views."""

    base = MusicSeparationDataset(
        config.dataset_root,
        split="all_train",
        config=config.preprocessing(),
    )
    names = [str(track.name) for track in base.tracks]
    split = deterministic_track_split(names, config.validation_fraction, config.seed)
    name_to_index = {str(track.name): index for index, track in enumerate(base.tracks)}
    selected_train_names = split.train_names
    selected_validation_names = split.validation_names
    if config.max_train_tracks is not None:
        selected_train_names = selected_train_names[: config.max_train_tracks]
    if config.max_validation_tracks is not None:
        selected_validation_names = selected_validation_names[: config.max_validation_tracks]
    selected_split = TrackSplit(selected_train_names, selected_validation_names)
    train_indices = [name_to_index[name] for name in selected_split.train_names]
    validation_indices = [name_to_index[name] for name in selected_split.validation_names]
    train_dataset: Dataset[Any] = TrackChunkDataset(base, train_indices)
    validation_dataset: Dataset[Any] = TrackChunkDataset(base, validation_indices)
    return PreparedTrainingData(
        all_train_dataset=base,
        train_dataset=_limit_chunks(train_dataset, config.max_train_chunks),
        validation_dataset=_limit_chunks(validation_dataset, config.max_valid_chunks),
        split=selected_split,
    )


def _collate_examples(batch: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = np.stack([np.asarray(item.input_tensor) for item in batch], axis=0)
    targets = np.stack([np.asarray(item.target_mask) for item in batch], axis=0)
    return torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()


def _loader(dataset: Dataset[Any], config: FullTrainingConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=_collate_examples,
        pin_memory=config.device in {"auto", "cuda"},
    )


def _criterion(name: str) -> nn.Module:
    return {"l1": nn.L1Loss(), "mae": nn.L1Loss(), "mse": nn.MSELoss()}[name.lower()]


def _optimizer(name: str, model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    if name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def _validate_batch(inputs: torch.Tensor, targets: torch.Tensor) -> None:
    if inputs.ndim != 4 or targets.ndim != 4 or inputs.shape != targets.shape:
        raise RuntimeError("training batches must have matching shape (B, 2, F, T)")
    if inputs.shape[1] != 2 or not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise RuntimeError("invalid stereo training batch")


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    minimum = float("inf")
    maximum = float("-inf")
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            _validate_batch(inputs, targets)
            inputs = inputs.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predictions = model(inputs)
                if predictions.shape != targets.shape or not torch.isfinite(predictions).all():
                    raise RuntimeError("invalid U-Net output")
                loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("loss became NaN or Inf")
            if training:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            batch_size = inputs.shape[0]
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
            minimum = min(minimum, float(predictions.min().detach().cpu()))
            maximum = max(maximum, float(predictions.max().detach().cpu()))
    if count == 0:
        raise RuntimeError("loader produced no batches")
    return total / count, minimum, maximum


def _load_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint:
        raise ValueError("resume checkpoint lacks model or optimizer state")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state and scaler.is_enabled():
        scaler.load_state_dict(scaler_state)
    return (
        int(checkpoint.get("epoch", 0)) + 1,
        float(checkpoint.get("best_validation_loss", checkpoint.get("validation_loss", float("inf")))),
        int(checkpoint.get("stale_epochs", 0)),
    )


def _save(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    result: FullEpochResult,
    best_loss: float,
    stale_epochs: int,
    config: FullTrainingConfig,
    split: TrackSplit | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "train_loss": result.train_loss,
            "validation_loss": result.validation_loss,
            "best_validation_loss": best_loss,
            "stale_epochs": stale_epochs,
            "mixed_precision_enabled": scaler.is_enabled(),
            "learning_rate": result.learning_rate,
            "training_config": asdict(config),
            "train_track_names": split.train_names if split else None,
            "validation_track_names": split.validation_names if split else None,
        },
        path,
    )


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f'training history must be a JSON list of records: {path}')
    return list(data)


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)
        handle.write('\n')


def _record_history(path: Path, record: dict[str, Any]) -> None:
    history = _load_history(path)
    epoch = int(record['epoch'])
    history = [item for item in history if int(item.get('epoch', -1)) != epoch]
    history.append(record)
    history.sort(key=lambda item: int(item.get('epoch', -1)))
    _write_history(path, history)


def _append_training_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with path.open('a', encoding='utf-8') as handle:
        handle.write(f'[{timestamp}] {message}\n')


def _write_training_config(path: Path, config: FullTrainingConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(asdict(config), handle, indent=2, default=str)
        handle.write('\n')


def run_full_training(
    train_dataset: Dataset[Any],
    validation_dataset: Dataset[Any],
    config: FullTrainingConfig,
    split: TrackSplit | None = None,
    model: nn.Module | None = None,
) -> list[FullEpochResult]:
    """Run configured training, with resume, best/last checkpoints, and stopping."""

    set_reproducible_seed(config.seed)
    device = resolve_device(config.device)
    model = model or StereoUNet()
    model.to(device)
    optimizer = _optimizer(config.optimizer, model, config.learning_rate)
    criterion = _criterion(config.loss)
    amp_enabled = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"training device: {device}; mixed precision: {'enabled' if amp_enabled else 'disabled'}")
    start_epoch = 1
    best_loss = float("inf")
    stale_epochs = 0
    if config.resume_from is not None:
        start_epoch, best_loss, stale_epochs = _load_resume(
            config.resume_from,
            model,
            optimizer,
            scaler,
            device,
        )
    train_loader = _loader(train_dataset, config, shuffle=True)
    validation_loader = _loader(validation_dataset, config, shuffle=False)
    results: list[FullEpochResult] = []
    best_path = config.checkpoint_dir / config.best_checkpoint_name
    last_path = config.checkpoint_dir / config.last_checkpoint_name
    history_path = config.checkpoint_dir / 'training_history.json'
    log_path = config.checkpoint_dir / 'training.log'
    _write_training_config(config.checkpoint_dir / 'training_config.json', config)
    _append_training_log(
        log_path,
        f'training start epoch={start_epoch} final_epoch={config.epochs} '
        f'device={device} mixed_precision={amp_enabled} stale_epochs={stale_epochs}',
    )

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, _, _ = _epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            amp_enabled,
        )
        validation_loss, mask_minimum, mask_maximum = _epoch(
            model,
            validation_loader,
            criterion,
            device,
            None,
            scaler,
            amp_enabled,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        result = FullEpochResult(epoch, train_loss, validation_loss, learning_rate)
        results.append(result)
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            stale_epochs = 0
            _save(
                best_path,
                model,
                optimizer,
                scaler,
                epoch,
                result,
                best_loss,
                stale_epochs,
                config,
                split,
            )
        else:
            stale_epochs += 1
        _save(
            last_path,
            model,
            optimizer,
            scaler,
            epoch,
            result,
            best_loss,
            stale_epochs,
            config,
            split,
        )
        if epoch % config.checkpoint_interval == 0:
            periodic = config.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            _save(
                periodic,
                model,
                optimizer,
                scaler,
                epoch,
                result,
                best_loss,
                stale_epochs,
                config,
                split,
            )
        print(
            f"epoch {epoch}/{config.epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} lr={learning_rate:.2e} "
            f"mask_range=[{mask_minimum:.4f}, {mask_maximum:.4f}]"
        )
        elapsed_epoch_seconds = time.perf_counter() - epoch_started
        _record_history(
            history_path,
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'learning_rate': learning_rate,
                'is_best': bool(improved),
                'stale_epochs': stale_epochs,
                'elapsed_epoch_seconds': elapsed_epoch_seconds,
            },
        )
        _append_training_log(
            log_path,
            f'epoch={epoch} train_loss={train_loss:.8f} '
            f'validation_loss={validation_loss:.8f} lr={learning_rate:.8g} '
            f'is_best={bool(improved)} stale_epochs={stale_epochs} '
            f'elapsed_epoch_seconds={elapsed_epoch_seconds:.4f}',
        )
        if (
            config.early_stopping_patience is not None
            and stale_epochs >= config.early_stopping_patience
        ):
            print(f"early stopping at epoch {epoch}")
            break
    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("training did not produce best_model.pth and last_model.pth")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare/run TuneWave full-training configuration")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--chunk-seconds", type=float, default=6.0)
    parser.add_argument("--chunk-hop-seconds", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-train-tracks", type=int)
    parser.add_argument("--max-validation-tracks", type=int)
    parser.add_argument("--max-train-chunks", type=int)
    parser.add_argument("--max-valid-chunks", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--loss", choices=("l1", "mae", "mse"), default="l1")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", default=True)
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--run", action="store_true", help="explicitly start the requested training run")
    parser.add_argument(
        "--cloud-smoke",
        action="store_true",
        help="run 3 epochs on 8 train tracks and 2 validation tracks",
    )
    parser.add_argument("--dry-run", action="store_true", help="run one epoch on two chunks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dry_run = args.dry_run
    cloud_smoke = args.cloud_smoke
    if dry_run and cloud_smoke:
        raise ValueError("choose either --dry-run or --cloud-smoke")
    config = FullTrainingConfig(
        dataset_root=args.dataset_root,
        checkpoint_dir=args.checkpoint_dir,
        chunk_seconds=args.chunk_seconds,
        chunk_hop_seconds=args.chunk_hop_seconds,
        validation_fraction=args.validation_fraction,
        batch_size=2 if cloud_smoke and args.batch_size == 1 else args.batch_size,
        learning_rate=args.learning_rate,
        epochs=1 if dry_run else (3 if cloud_smoke else args.epochs),
        optimizer=args.optimizer,
        loss=args.loss,
        early_stopping_patience=1 if dry_run else args.early_stopping_patience,
        checkpoint_interval=1 if dry_run or cloud_smoke else args.checkpoint_interval,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        mixed_precision=args.mixed_precision,
        resume_from=args.resume_from,
        max_train_tracks=8 if cloud_smoke else args.max_train_tracks,
        max_validation_tracks=2 if cloud_smoke else args.max_validation_tracks,
        max_train_chunks=2 if dry_run else args.max_train_chunks,
        max_valid_chunks=2 if dry_run else args.max_valid_chunks,
    )
    prepared = prepare_training_data(config)
    print(f"train tracks: {len(prepared.split.train_names)}")
    print(f"validation tracks: {len(prepared.split.validation_names)}")
    print(f"train chunks: {len(prepared.train_dataset)}; validation chunks: {len(prepared.validation_dataset)}")
    print(f"device: {resolve_device(config.device)}")
    if not dry_run and not cloud_smoke and not args.run:
        print("Configuration prepared. Add --run to start training; no training was launched.")
        return 0
    run_full_training(
        prepared.train_dataset,
        prepared.validation_dataset,
        config,
        split=prepared.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
