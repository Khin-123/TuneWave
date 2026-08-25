import unittest

import torch

from models.vocal_unet import StereoUNet


class StereoUNetTestCase(unittest.TestCase):
    def test_stereo_mask_shape_range_and_parameter_count(self):
        # Stage 3 defaults: n_fft=2048 gives 1025 frequency bins and the
        # current centered STFT produces 518 frames for a 6-second chunk.
        inputs = torch.randn(1, 2, 1025, 518)
        model = StereoUNet()
        model.eval()

        with torch.no_grad():
            masks = model(inputs)

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        minimum = float(masks.min().item())
        maximum = float(masks.max().item())
        print("input shape:", tuple(inputs.shape))
        print("output shape:", tuple(masks.shape))
        print("parameter count:", parameter_count)
        print("minimum mask value:", minimum)
        print("maximum mask value:", maximum)

        self.assertEqual(masks.shape, inputs.shape)
        self.assertGreaterEqual(minimum, 0.0)
        self.assertLessEqual(maximum, 1.0)
        self.assertTrue(torch.isfinite(masks).all().item())


if __name__ == "__main__":
    unittest.main()
