import json
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
from torch import nn


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


class _TinyMaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, inputs):
        return torch.sigmoid(self.projection(inputs))


class FullTrainingConfigTestCase(unittest.TestCase):
    def test_track_split_is_deterministic_and_disjoint(self):
        names = ["track-c", "track-a", "track-b", "track-d"]
        first = deterministic_track_split(names, validation_fraction=0.25, seed=1337)
        second = deterministic_track_split(names, validation_fraction=0.25, seed=1337)
        self.assertEqual(first, second)
        self.assertFalse(set(first.train_names) & set(first.validation_names))
        self.assertEqual(set(first.train_names) | set(first.validation_names), set(names))

    def test_fresh_and_resume_preserve_stale_epochs_and_total_epoch_limit(self):
        base = _BaseDataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fresh_dir = root / 'fresh'
            fresh_config = FullTrainingConfig(
                checkpoint_dir=fresh_dir,
                epochs=1,
                batch_size=2,
                device='cpu',
                early_stopping_patience=None,
            )
            run_full_training(base, base, fresh_config, model=_TinyMaskModel())
            fresh_checkpoint = torch.load(
                fresh_dir / 'last_model.pth', map_location='cpu', weights_only=False
            )
            self.assertEqual(fresh_checkpoint['stale_epochs'], 0)
            fresh_history = json.loads((fresh_dir / 'training_history.json').read_text())
            self.assertEqual(fresh_history[0]['stale_epochs'], 0)
            self.assertTrue((fresh_dir / 'training.log').is_file())
            self.assertTrue((fresh_dir / 'training_config.json').is_file())

            resume_dir = root / 'resume'
            resume_dir.mkdir()
            resume_model = _TinyMaskModel()
            resume_optimizer = torch.optim.Adam(resume_model.parameters(), lr=1e-3)
            resume_checkpoint = {
                'epoch': 5,
                'model_state_dict': resume_model.state_dict(),
                'optimizer_state_dict': resume_optimizer.state_dict(),
                'scaler_state_dict': None,
                'best_validation_loss': -1.0,
                'stale_epochs': 3,
            }
            resume_path = resume_dir / 'last_model.pth'
            torch.save(resume_checkpoint, resume_path)
            torch.save(resume_checkpoint, resume_dir / 'best_model.pth')
            history = [
                {'epoch': epoch, 'stale_epochs': epoch - 1}
                for epoch in range(1, 6)
            ]
            (resume_dir / 'training_history.json').write_text(json.dumps(history))

            resume_config = FullTrainingConfig(
                checkpoint_dir=resume_dir,
                epochs=20,
                batch_size=2,
                device='cpu',
                early_stopping_patience=None,
                resume_from=resume_path,
            )
            results = run_full_training(
                base, base, resume_config, model=_TinyMaskModel()
            )
            self.assertEqual(results[0].epoch, 6)
            self.assertEqual(results[-1].epoch, 20)
            self.assertEqual(len(results), 15)
            final_checkpoint = torch.load(
                resume_dir / 'last_model.pth', map_location='cpu', weights_only=False
            )
            self.assertEqual(final_checkpoint['stale_epochs'], 18)
            self.assertEqual(final_checkpoint['epoch'], 20)
            loadable_model = _TinyMaskModel()
            loadable_model.load_state_dict(final_checkpoint['model_state_dict'])
            loadable_optimizer = torch.optim.Adam(loadable_model.parameters(), lr=1e-3)
            loadable_optimizer.load_state_dict(final_checkpoint['optimizer_state_dict'])
            self.assertIn('scaler_state_dict', final_checkpoint)
            self.assertIsNone(final_checkpoint['scaler_state_dict'])
            amp_scaler = torch.amp.GradScaler('cpu', enabled=True)
            amp_checkpoint = dict(final_checkpoint)
            amp_checkpoint['scaler_state_dict'] = amp_scaler.state_dict()
            amp_path = resume_dir / 'amp_checkpoint.pth'
            torch.save(amp_checkpoint, amp_path)
            loaded_amp_checkpoint = torch.load(
                amp_path, map_location='cpu', weights_only=False
            )
            restored_amp_scaler = torch.amp.GradScaler('cpu', enabled=True)
            restored_amp_scaler.load_state_dict(loaded_amp_checkpoint['scaler_state_dict'])
            final_history = json.loads((resume_dir / 'training_history.json').read_text())
            self.assertEqual([item['epoch'] for item in final_history], list(range(1, 21)))
            self.assertEqual(final_history[5]['stale_epochs'], 4)
            self.assertEqual(len((resume_dir / 'training.log').read_text().splitlines()), 16)

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
            self.assertEqual(
                torch.load(directory / 'last_model.pth', map_location='cpu', weights_only=False)['stale_epochs'],
                0,
            )
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
