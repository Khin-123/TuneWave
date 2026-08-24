import unittest

import numpy as np

from dsp.stft_utils import hann_window, istft, magnitude, phase, stft


class StftUtilitiesTestCase(unittest.TestCase):
    def test_hann_window_matches_standard_equation(self):
        window_length = 16
        indices = np.arange(window_length)
        expected = 0.5 - 0.5 * np.cos(
            2 * np.pi * indices / (window_length - 1)
        )

        np.testing.assert_allclose(hann_window(window_length), expected)
        self.assertEqual(hann_window(window_length)[0], 0.0)
        self.assertEqual(hann_window(window_length)[-1], 0.0)

    def test_magnitude_and_phase_represent_complex_stft(self):
        complex_values = np.array(
            [[1 + 1j, -2j], [-3 + 4j, 2 - 2j]],
            dtype=np.complex128,
        )

        reconstructed = magnitude(complex_values) * np.exp(
            1j * phase(complex_values)
        )

        np.testing.assert_allclose(reconstructed, complex_values, atol=1e-12)

    def test_sine_wave_stft_istft_reconstruction(self):
        sample_rate = 16_000
        sample_count = sample_rate + 137
        time = np.arange(sample_count) / sample_rate
        original = 0.7 * np.sin(2 * np.pi * 440 * time)

        transformed = stft(
            original,
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
        )
        reconstructed = istft(
            transformed,
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            length=original.size,
        )

        self.assertEqual(reconstructed.shape, original.shape)
        np.testing.assert_allclose(reconstructed, original, atol=1e-10)

    def test_signal_shorter_than_fft_size_reconstructs(self):
        sample_rate = 8_000
        original = np.linspace(-0.5, 0.5, 127)

        transformed = stft(
            original,
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=128,
        )
        reconstructed = istft(
            transformed,
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=128,
            length=original.size,
        )

        np.testing.assert_allclose(reconstructed, original, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
