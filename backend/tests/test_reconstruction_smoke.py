import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dsp.ml_preprocessing import PreprocessingConfig
from ml_dataset import CompressedStemChunk
from models.vocal_unet import StereoUNet
from reconstruct_vocal_unet import load_checkpoint_model, reconstruct_chunk, write_stereo_wav


class ReconstructionSmokeTestCase(unittest.TestCase):
    def test_checkpoint_mask_original_complex_stft_istft_and_wav_outputs(self):
        config = PreprocessingConfig(
            sample_rate=44_100,
            n_fft=256,
            hop_length=64,
            chunk_seconds=2048 / 44_100,
            chunk_hop_seconds=2048 / 44_100,
        )
        time = np.arange(config.chunk_samples, dtype=np.float32) / config.sample_rate
        vocals = np.stack(
            (0.2 * np.sin(2 * np.pi * 440 * time), 0.18 * np.sin(2 * np.pi * 440 * time)),
            axis=0,
        ).astype(np.float32)
        instrumental = np.stack(
            (0.1 * np.sin(2 * np.pi * 110 * time), 0.09 * np.sin(2 * np.pi * 110 * time)),
            axis=0,
        ).astype(np.float32)
        chunk = CompressedStemChunk(
            mixture=vocals + instrumental,
            vocals=vocals,
            instrumental=instrumental,
            start_seconds=0.0,
            duration_seconds=config.chunk_seconds,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "smoke_best_model.pth"
            original_model = StereoUNet()
            torch.save({"model_state_dict": original_model.state_dict()}, checkpoint_path)
            model, device = load_checkpoint_model(checkpoint_path, "cpu")
            result = reconstruct_chunk(chunk, model, config, device)

            self.assertEqual(result.predicted_vocals.shape, chunk.mixture.shape)
            self.assertEqual(result.predicted_instrumental.shape, chunk.mixture.shape)
            self.assertEqual(result.vocal_mask.shape, (2, config.n_fft // 2 + 1, result.vocal_mask.shape[-1]))
            self.assertTrue(np.isfinite(result.predicted_vocals).all())
            self.assertTrue(np.isfinite(result.predicted_instrumental).all())
            self.assertGreaterEqual(float(result.vocal_mask.min()), 0.0)
            self.assertLessEqual(float(result.vocal_mask.max()), 1.0)

            predicted_vocals_path = write_stereo_wav(
                Path(temporary_directory) / "predicted_vocals.wav",
                result.predicted_vocals,
                config.sample_rate,
            )
            predicted_instrumental_path = write_stereo_wav(
                Path(temporary_directory) / "predicted_instrumental.wav",
                result.predicted_instrumental,
                config.sample_rate,
            )
            for path in (predicted_vocals_path, predicted_instrumental_path):
                info = sf.info(path)
                self.assertEqual(info.channels, 2)
                self.assertEqual(info.samplerate, config.sample_rate)
                self.assertEqual(info.frames, config.chunk_samples)
                decoded, rate = sf.read(path, always_2d=True)
                self.assertEqual(rate, config.sample_rate)
                self.assertEqual(decoded.shape, (config.chunk_samples, 2))
                self.assertTrue(np.isfinite(decoded).all())
            print("reconstructed predicted vocals:", predicted_vocals_path)
            print("reconstructed predicted instrumental:", predicted_instrumental_path)


if __name__ == "__main__":
    unittest.main()
