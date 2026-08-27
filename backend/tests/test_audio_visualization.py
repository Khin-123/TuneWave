import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import soundfile as sf

from dsp.audio_visualization import _waveform_envelope, generate_audio_visualizations


class AudioVisualizationTestCase(unittest.TestCase):
    def test_waveform_envelope_is_finite_and_bounded(self):
        signal = np.sin(np.linspace(0, 80 * np.pi, 8_001)).astype(np.float32)

        times, amplitudes = _waveform_envelope(signal, 8_000, max_points=4_000)

        self.assertLessEqual(times.size, 4_000)
        self.assertEqual(times.shape, amplitudes.shape)
        self.assertTrue(np.isfinite(times).all())
        self.assertTrue(np.isfinite(amplitudes).all())

    def test_generates_valid_waveform_and_spectrogram_pngs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            sample_rate = 8_000
            time = np.arange(2_400) / sample_rate
            audio = np.column_stack((
                0.25 * np.sin(2 * np.pi * 440 * time),
                0.20 * np.sin(2 * np.pi * 880 * time),
            ))
            audio_path = directory / 'stereo.wav'
            sf.write(audio_path, audio, sample_rate, subtype='PCM_16')

            paths = generate_audio_visualizations(
                audio_path,
                directory,
                'original',
            )

            self.assertEqual(set(paths), {'waveform', 'spectrogram'})
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
                image = mpimg.imread(path)
                self.assertEqual(image.ndim, 3)
                self.assertGreater(image.shape[0], 0)
                self.assertGreater(image.shape[1], 0)


if __name__ == '__main__':
    unittest.main()
