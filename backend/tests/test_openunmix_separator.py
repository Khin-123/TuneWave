import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import separation.openunmix_separator as separator_module


class _FakePredict:
    @staticmethod
    def separate(**kwargs):
        audio = kwargs['audio']
        samples = audio.shape[-1]
        estimate = torch.full((1, 2, samples), 0.1, dtype=torch.float32)
        return {name: estimate.clone() for name in separator_module.OPENUNMIX_TARGETS}


class OpenUnmixSeparatorTestCase(unittest.TestCase):
    def test_stereo_outputs_and_instrumental_are_written(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / 'mixture.wav'
            samples = np.zeros((4410, 2), dtype=np.float32)
            sf.write(input_path, samples, 44_100)
            with patch.object(separator_module, '_load_predict_module', return_value=_FakePredict):
                result = separator_module.separate_audio_file(
                    input_path,
                    root / 'outputs',
                    device='cpu',
                    niter=0,
                )
            self.assertEqual(result['sample_rate'], 44_100)
            self.assertEqual(result['channels'], 2)
            for name, path in result['paths'].items():
                self.assertTrue(path.is_file(), name)
                audio, sample_rate = sf.read(path, always_2d=True, dtype='float32')
                self.assertEqual(sample_rate, 44_100)
                self.assertEqual(audio.shape[1], 2)
                self.assertTrue(np.isfinite(audio).all())
                self.assertLessEqual(float(np.max(np.abs(audio))), 0.99)

    def test_auto_device_is_cpu_without_cuda(self):
        device = separator_module.resolve_device('auto')
        expected = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(str(device), expected)


if __name__ == '__main__':
    unittest.main()
