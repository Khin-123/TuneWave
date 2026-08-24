import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dsp.source_separation import (
    create_spectral_masks,
    separate_audio_file,
    separate_stereo_audio,
)


class SourceSeparationTestCase(unittest.TestCase):
    @staticmethod
    def make_stereo_mixture(
        sample_rate: int = 8_000,
        sample_count: int = 4_173,
    ) -> np.ndarray:
        time = np.arange(sample_count) / sample_rate
        centered_source = 0.25 * np.sin(2 * np.pi * 440 * time)
        side_source = 0.15 * np.sin(2 * np.pi * 880 * time + 0.2)
        left = centered_source + side_source
        right = centered_source - side_source
        return np.vstack((left, right))

    def test_masks_are_soft_and_complementary(self):
        mid_spectrum = np.array([[4 + 0j, 1 + 0j, 0 + 0j]])
        side_spectrum = np.array([[1 + 0j, 1 + 0j, 4 + 0j]])

        vocal_mask, instrumental_mask = create_spectral_masks(
            mid_spectrum,
            side_spectrum,
            p=2,
            alpha=1.0,
            epsilon=1e-10,
        )

        self.assertTrue(np.all((vocal_mask >= 0) & (vocal_mask <= 1)))
        self.assertGreater(vocal_mask[0, 0], vocal_mask[0, 1])
        self.assertGreater(vocal_mask[0, 1], vocal_mask[0, 2])
        np.testing.assert_allclose(vocal_mask + instrumental_mask, 1.0)

    def test_stems_reconstruct_original_mixture(self):
        original = self.make_stereo_mixture()

        vocals, instrumental = separate_stereo_audio(
            original,
            sample_rate=8_000,
            n_fft=512,
            hop_length=128,
        )

        self.assertEqual(vocals.shape, original.shape)
        self.assertEqual(instrumental.shape, original.shape)
        np.testing.assert_allclose(vocals[0], vocals[1], atol=1e-12)
        np.testing.assert_allclose(vocals + instrumental, original, atol=1e-10)
        self.assertLessEqual(float(np.max(np.abs(vocals))), 0.99)
        self.assertLessEqual(float(np.max(np.abs(instrumental))), 0.99)

    def test_separation_rejects_mono_audio(self):
        with self.assertRaisesRegex(ValueError, "mono audio is not supported"):
            separate_stereo_audio(np.zeros(2_000), sample_rate=8_000)

    def test_shared_peak_limit_prevents_clipping(self):
        loud_mixture = 5.0 * self.make_stereo_mixture()

        vocals, instrumental = separate_stereo_audio(
            loud_mixture,
            sample_rate=8_000,
            n_fft=512,
            hop_length=128,
            peak_limit=0.99,
        )

        self.assertLessEqual(float(np.max(np.abs(vocals))), 0.99 + 1e-12)
        self.assertLessEqual(float(np.max(np.abs(instrumental))), 0.99 + 1e-12)

    def test_file_separation_writes_stereo_wav_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "mixture.wav"
            output_directory = directory / "outputs"
            sf.write(input_path, self.make_stereo_mixture().T, 8_000)

            result = separate_audio_file(
                input_path,
                output_directory=output_directory,
                n_fft=512,
                hop_length=128,
            )

            self.assertEqual(result["vocals_path"].name, "vocals.wav")
            self.assertEqual(result["channels"], 2)
            self.assertEqual(result["samples"], 4_173)
            self.assertEqual(
                result["instrumental_path"].name,
                "instrumental.wav",
            )
            self.assertTrue(result["vocals_path"].is_file())
            self.assertTrue(result["instrumental_path"].is_file())
            self.assertEqual(sf.info(result["vocals_path"]).channels, 2)
            self.assertEqual(sf.info(result["instrumental_path"]).channels, 2)
            self.assertEqual(sf.info(result["vocals_path"]).samplerate, 8_000)


if __name__ == "__main__":
    unittest.main()
