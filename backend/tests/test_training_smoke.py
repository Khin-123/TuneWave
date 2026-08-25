import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_vocal_unet import SmokeTrainingConfig, limit_chunks, train_model


class TinyMaskDataset:
    """Small in-memory contract test; production uses MusicSeparationDataset."""

    def __init__(self, count: int, seed: int):
        generator = np.random.default_rng(seed)
        self.examples = []
        for _ in range(count):
            inputs = generator.normal(size=(2, 32, 32)).astype(np.float32)
            targets = generator.uniform(0.0, 1.0, size=(2, 32, 32)).astype(np.float32)
            self.examples.append(type("Example", (), {"input_tensor": inputs, "target_mask": targets})())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class TrainingSmokeTestCase(unittest.TestCase):
    def test_dataset_model_loss_backprop_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = SmokeTrainingConfig(
                epochs=1,
                batch_size=1,
                max_train_chunks=2,
                max_valid_chunks=2,
                device="cpu",
                checkpoint_dir=Path(temporary_directory),
            )
            train_dataset = limit_chunks(TinyMaskDataset(3, seed=1), config.max_train_chunks)
            validation_dataset = limit_chunks(TinyMaskDataset(3, seed=2), config.max_valid_chunks)
            results = train_model(train_dataset, validation_dataset, config)

            self.assertEqual(len(results), 1)
            self.assertTrue(np.isfinite(results[0].train_loss))
            self.assertTrue(np.isfinite(results[0].validation_loss))
            self.assertTrue((Path(temporary_directory) / "smoke_best_model.pth").is_file())
            self.assertTrue((Path(temporary_directory) / "smoke_last_model.pth").is_file())
            print("training loss:", results[0].train_loss)
            print("validation loss:", results[0].validation_loss)
            print("best checkpoint:", Path(temporary_directory) / "smoke_best_model.pth")


if __name__ == "__main__":
    unittest.main()
