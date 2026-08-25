import os
import unittest
from pathlib import Path

from dsp.ml_preprocessing import PreprocessingConfig
from ml_dataset import MusicSeparationDataset


class CompressedMusdbSmokeTest(unittest.TestCase):
    """Read one real compressed track when MUSDB18 and its tools are installed."""

    def test_load_one_compressed_chunk(self):
        repository_root = Path(__file__).resolve().parents[2]
        configured_root = os.environ.get("MUSDB18_ROOT")
        candidates = [
            Path(configured_root) if configured_root else None,
            repository_root / "datasets" / "musdb18",
            repository_root / "dataset" / "musdb18",
        ]
        root = next((candidate for candidate in candidates if candidate and candidate.is_dir()), None)
        if root is None:
            self.skipTest("Set MUSDB18_ROOT to a compressed MUSDB18 root to run this smoke test.")
        try:
            import musdb  # noqa: F401
            import stempeg  # noqa: F401
        except ImportError:
            self.skipTest("Install musdb and stempeg before running the compressed smoke test.")

        config = PreprocessingConfig(chunk_seconds=6.0, chunk_hop_seconds=3.0)
        dataset = MusicSeparationDataset(root, split="train", config=config)
        input_tensor, target_mask = dataset[0]
        print("compressed MUSDB18 input shape:", input_tensor.shape)
        print("compressed MUSDB18 target shape:", target_mask.shape)
        self.assertEqual(input_tensor.shape[0], 2)
        self.assertEqual(target_mask.shape, input_tensor.shape)


if __name__ == "__main__":
    unittest.main()
