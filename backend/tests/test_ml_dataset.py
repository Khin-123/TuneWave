import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dsp.ml_preprocessing import MagnitudeNormalizationStats, PreprocessingConfig
from ml_dataset import DatasetError, MusicSeparationDataset, create_vocal_mask, fit_training_normalization


class FakeSource:
    def __init__(self, track, name, audio):
        self.track = track
        self.name = name
        self.full_audio = audio

    @property
    def audio(self):
        self.track.reads += 1
        start = int(round(self.track.chunk_start * self.track.rate))
        count = int(round(self.track.chunk_duration * self.track.rate))
        return self.full_audio[start : start + count]


class FakeTrack:
    def __init__(self, name="compressed-demo", duration=0.5, rate=44_100):
        self.name = name
        self.duration = duration
        self.rate = rate
        self.chunk_start = 0.0
        self.chunk_duration = duration
        sample_count = int(round(duration * rate))
        time = np.arange(sample_count, dtype=np.float32) / rate
        vocals = np.stack((0.20 * np.sin(2 * np.pi * 440 * time),) * 2, axis=1)
        drums = np.stack((0.08 * np.sin(2 * np.pi * 880 * time),) * 2, axis=1)
        bass = np.stack((0.10 * np.sin(2 * np.pi * 110 * time),) * 2, axis=1)
        other = np.stack((0.05 * np.sin(2 * np.pi * 660 * time),) * 2, axis=1)
        mixture = vocals + drums + bass + other
        self.full_sources = {
            "mixture": mixture,
            "vocals": vocals,
            "drums": drums,
            "bass": bass,
            "other": other,
        }
        self.sources = {
            name: FakeSource(self, name, audio)
            for name, audio in self.full_sources.items()
            if name != "mixture"
        }
        self.reads = 0

    @property
    def audio(self):
        self.reads += 1
        start = int(round(self.chunk_start * self.rate))
        count = int(round(self.chunk_duration * self.rate))
        return self.full_sources["mixture"][start : start + count]


class FakeDB:
    calls = []

    def __init__(self, **kwargs):
        FakeDB.calls.append(kwargs)
        self.tracks = [FakeTrack()]


class CompressedDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.config = PreprocessingConfig(
            sample_rate=44_100,
            n_fft=512,
            hop_length=128,
            chunk_seconds=0.1,
            chunk_hop_seconds=0.05,
        )
        FakeDB.calls.clear()
        self.fake_musdb = types.SimpleNamespace(DB=FakeDB)

    def make_dataset(self, split="train"):
        modules = {"musdb": self.fake_musdb, "stempeg": types.SimpleNamespace()}
        temporary_directory = tempfile.TemporaryDirectory()
        root = Path(temporary_directory.name)
        dataset = MusicSeparationDataset(root, split=split, config=self.config)
        return temporary_directory, dataset

    def test_compressed_chunk_is_lazy_stereo_and_prints_shapes(self):
        with patch.dict(sys.modules, {"musdb": self.fake_musdb, "stempeg": types.SimpleNamespace()}):
            temporary_directory, dataset = self.make_dataset()
            try:
                self.assertEqual(FakeDB.calls[0]["is_wav"], False)
                self.assertEqual(FakeDB.calls[0]["subsets"], "train")
                self.assertEqual(FakeDB.calls[0]["split"], "train")
                track = dataset.tracks[0]
                self.assertEqual(track.reads, 0)
                example = dataset[0]
                print("compressed chunk input shape:", example.input_tensor.shape)
                print("compressed chunk target shape:", example.target_mask.shape)
                self.assertEqual(example.input_tensor.shape[0], 2)
                self.assertEqual(example.target_mask.shape, example.input_tensor.shape)
                self.assertLess(track.chunk_duration, track.duration)
                self.assertGreater(track.reads, 0)
                self.assertTrue(np.isfinite(example.input_tensor).all())
                self.assertTrue(np.all((example.target_mask >= 0) & (example.target_mask <= 1)))
            finally:
                temporary_directory.cleanup()

    def test_train_and_valid_use_train_subset_but_test_is_separate(self):
        with patch.dict(sys.modules, {"musdb": self.fake_musdb, "stempeg": types.SimpleNamespace()}):
            train_dir, _ = self.make_dataset("train")
            train_dir.cleanup()
            valid_dir, _ = self.make_dataset("valid")
            valid_dir.cleanup()
            test_dir, test_dataset = self.make_dataset("test")
            test_dir.cleanup()
            self.assertEqual(FakeDB.calls[0]["split"], "train")
            self.assertEqual(FakeDB.calls[1]["split"], "valid")
            self.assertEqual(FakeDB.calls[1]["subsets"], "train")
            self.assertIsNone(FakeDB.calls[2]["split"])
            self.assertEqual(FakeDB.calls[2]["subsets"], "test")
            with self.assertRaises(DatasetError):
                fit_training_normalization(test_dataset)

    def test_instrumental_is_drums_plus_bass_plus_other_and_normalization_fits(self):
        with patch.dict(sys.modules, {"musdb": self.fake_musdb, "stempeg": types.SimpleNamespace()}):
            temporary_directory, dataset = self.make_dataset()
            try:
                chunk = dataset.load_chunk(0)
                expected = (
                    dataset.tracks[0].full_sources["drums"]
                    + dataset.tracks[0].full_sources["bass"]
                    + dataset.tracks[0].full_sources["other"]
                )[: chunk.instrumental.shape[1]].T
                np.testing.assert_allclose(chunk.instrumental, expected, atol=1e-6)
                stats = fit_training_normalization(dataset)
                self.assertIsInstance(stats, MagnitudeNormalizationStats)
                self.assertEqual(stats.mean.shape, (2, 1, 1))
                self.assertEqual(stats.std.shape, (2, 1, 1))
            finally:
                temporary_directory.cleanup()

    def test_ratio_mask_is_bounded(self):
        vocal = np.zeros((2, 4, 3), dtype=np.complex128)
        instrumental = np.ones((2, 4, 3), dtype=np.complex128)
        mask = create_vocal_mask(vocal, instrumental)
        self.assertEqual(mask.shape, vocal.shape)
        np.testing.assert_allclose(mask, 0.0)


if __name__ == "__main__":
    unittest.main()
