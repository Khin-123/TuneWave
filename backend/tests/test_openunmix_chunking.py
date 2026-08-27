import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import separation.openunmix_dsp_separator as separator


class _FakeVocalModel(torch.nn.Module):
    def forward(self, mixture_magnitude):
        return mixture_magnitude * 0.5


class _FakeFiltering:
    @staticmethod
    def wiener(targets, mixture, iterations=0, softmask=False, residual=False):
        vocal = torch.stack(
            (targets[..., 0], torch.zeros_like(targets[..., 0])),
            dim=-1,
        ).unsqueeze(-1)
        if not residual:
            return vocal
        return torch.cat((vocal, mixture.unsqueeze(-1) - vocal), dim=-1)


class OpenUnmixChunkingTestCase(unittest.TestCase):
    def test_chunk_starts_cover_long_input_with_configured_overlap(self):
        self.assertEqual(
            separator._chunk_starts(180, chunk_samples=100, overlap_samples=20),
            [0, 80],
        )
        self.assertEqual(
            separator._chunk_starts(250, chunk_samples=100, overlap_samples=20),
            [0, 80, 150],
        )

    def test_production_path_is_two_stem_and_reconstructs_mixture(self):
        sample_rate = separator.OPENUNMIX_SAMPLE_RATE
        samples = int(sample_rate * 1.1)
        time = np.arange(samples, dtype=np.float32) / sample_rate
        audio = np.stack(
            (
                0.2 * np.sin(2 * np.pi * 220 * time),
                0.2 * np.sin(2 * np.pi * 330 * time),
            )
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with (
                patch.object(separator, 'load_audio', return_value=(audio, sample_rate)),
                patch.object(
                    separator,
                    '_load_openunmix_core',
                    return_value=(_FakeFiltering, None),
                ),
                patch.object(
                    separator,
                    '_load_target_models',
                    return_value={'vocals': _FakeVocalModel()},
                ) as model_loader,
            ):
                result = separator.separate_audio_file(
                    'mixture.wav',
                    output_directory,
                    device='cpu',
                    niter=0,
                    chunk_seconds=0.4,
                    overlap_seconds=0.1,
                    n_fft=512,
                    hop_length=128,
                )

            self.assertEqual(result['targets'], ('vocals',))
            model_loader.assert_called_once_with(
                'umxl',
                'cpu',
                targets=('vocals',),
            )
            self.assertLess(result['overlap_add_mixture_error'], 1e-6)
            self.assertLess(result['overlap_add_source_error'], 1e-6)
            self.assertLess(result['pre_limit_reconstruction_error'], 1e-6)
            self.assertEqual(result['shared_gain'], 1.0)
            self.assertLess(result['post_limit_reconstruction_error'], 1e-6)
            self.assertTrue(
                all(error < 1e-6 for error in result['boundary_reconstruction_errors'])
            )
            self.assertTrue(
                all(error < 1e-6 for error in result['chunk_complex_errors'])
            )
            self.assertTrue(
                all(error < 1e-6 for error in result['chunk_waveform_errors'])
            )
            self.assertGreater(result['chunks'], 1)
            self.assertEqual(set(result['paths']), {'vocals', 'instrumental'})
            vocals, vocals_rate = sf.read(
                result['paths']['vocals'], always_2d=True, dtype='float32'
            )
            instrumental, instrumental_rate = sf.read(
                result['paths']['instrumental'], always_2d=True, dtype='float32'
            )
            self.assertEqual(vocals_rate, sample_rate)
            self.assertEqual(instrumental_rate, sample_rate)
            self.assertEqual(vocals.shape, (samples, 2))
            self.assertEqual(instrumental.shape, (samples, 2))
            self.assertTrue(np.isfinite(vocals).all())
            self.assertTrue(np.isfinite(instrumental).all())
            np.testing.assert_allclose(
                vocals + instrumental,
                audio.T,
                atol=3 / 32767,
            )
            for boundary in result['chunk_starts'][1:]:
                start = max(0, boundary - 4)
                stop = min(samples, boundary + 4)
                np.testing.assert_allclose(
                    (vocals + instrumental)[start:stop],
                    audio.T[start:stop],
                    atol=3 / 32767,
                )


if __name__ == '__main__':
    unittest.main()
