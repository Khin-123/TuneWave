import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ml_dataset import ChunkIndex
from train_full_vocal_unet import (
    FullTrainingConfig,
    TrackChunkDataset,
    deterministic_track_split,
    run_full_training,
)


class _Track:
    def __init__(self, name):
        self.name = name


class _Example:
    def __init__(self, seed):
        generator = np.random.default_rng(seed)
        self.input_tensor = generator.normal(size=(2, 32, 32)).astype(np.float32)
        self.target_mask = generator.uniform(0.0, 1.0, size=(2, 32, 32)).astype(np.float32)


class _BaseDataset:
    def __init__(self):
        self.tracks = [_Track("track-c"), _Track("track-a"), _Track("track-b"), _Track("track-d")]
        self._index = [ChunkIndex(index, 0.0, 0.1) for index in range(len(self.tracks))]
        self.examples = [_Example(index) for index in range(len(self.tracks))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class FullTrainingConfigTestCase(unittest.TestCase):
    def test_track_split_is_deterministic_and_disjoint(self):
        names = ["track-c", "track-a", "track-b", "track-d"]
        first = deterministic_track_split(names, validation_fraction=0.25, seed=1337)
        second = deterministic_track_split(names, validation_fraction=0.25, seed=1337)
        self.assertEqual(first, second)
        self.assertFalse(set(first.train_names) & set(first.validation_names))
        self.assertEqual(set(first.train_names) | set(first.validation_names), set(names))

    def test_short_dry_run_saves_best_last_and_supports_resume(self):
        base = _BaseDataset()
        split = deterministic_track_split([track.name for track in base.tracks], 0.25, 1337)
        by_name = {track.name: index for index, track in enumerate(base.tracks)}
        train = TrackChunkDataset(base, [by_name[name] for name in split.train_names])
        validation = TrackChunkDataset(base, [by_name[name] for name in split.validation_names])

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config = FullTrainingConfig(
                checkpoint_dir=directory,
                epochs=1,
                batch_size=1,
                device="cpu",
                early_stopping_patience=None,
            )
            first_results = run_full_training(train, validation, config, split=split)
            self.assertEqual(first_results[0].epoch, 1)
            self.assertTrue((directory / "best_model.pth").is_file())
            self.assertTrue((directory / "last_model.pth").is_file())
            checkpoint = torch.load(directory / "last_model.pth", map_location="cpu", weights_only=False)
            self.assertFalse(checkpoint["mixed_precision_enabled"])
            self.assertIsNone(checkpoint["scaler_state_dict"])

            resumed_config = FullTrainingConfig(
                checkpoint_dir=directory,
                epochs=2,
                batch_size=1,
                device="cpu",
                early_stopping_patience=None,
                resume_from=directory / "last_model.pth",
            )
            resumed_results = run_full_training(train, validation, resumed_config, split=split)
            self.assertEqual(resumed_results[0].epoch, 2)
            self.assertTrue(np.isfinite(resumed_results[0].validation_loss))
            print("best checkpoint:", directory / "best_model.pth")
            print("last checkpoint:", directory / "last_model.pth")


if __name__ == "__main__":
    unittest.main()
