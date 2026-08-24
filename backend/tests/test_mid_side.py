import unittest

import numpy as np

from dsp.mid_side import decode_mid_side, encode_mid_side


class MidSideTestCase(unittest.TestCase):
    def test_encode_uses_mid_side_equations(self):
        stereo = np.array(
            [
                [1.0, 0.5, -0.5, -1.0],
                [0.5, -0.5, 0.5, -0.5],
            ]
        )

        mid, side = encode_mid_side(stereo)

        np.testing.assert_allclose(mid, [0.75, 0.0, 0.0, -0.75])
        np.testing.assert_allclose(side, [0.25, 0.5, -0.5, -0.25])

    def test_stereo_mid_side_round_trip(self):
        sample_rate = 16_000
        time = np.arange(sample_rate + 37) / sample_rate
        left = 0.6 * np.sin(2 * np.pi * 440 * time)
        right = 0.4 * np.sin(2 * np.pi * 660 * time + 0.3)
        original = np.vstack((left, right))

        mid, side = encode_mid_side(original)
        reconstructed = decode_mid_side(mid, side)

        self.assertEqual(reconstructed.shape, original.shape)
        np.testing.assert_allclose(reconstructed, original, atol=1e-15)

    def test_encode_rejects_mono_input(self):
        mono = np.zeros(1_000)

        with self.assertRaisesRegex(ValueError, "mono audio is not supported"):
            encode_mid_side(mono)

    def test_decode_rejects_different_channel_lengths(self):
        with self.assertRaisesRegex(ValueError, "same number of samples"):
            decode_mid_side(np.zeros(100), np.zeros(99))


if __name__ == "__main__":
    unittest.main()
