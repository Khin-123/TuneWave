"""Small stereo 2-D U-Net for vocal-mask prediction.

The model consumes the two normalized magnitude channels produced by the
Stage 2/3 DSP preprocessing pipeline and returns one independent soft mask
channel for Left and Right.  It predicts masks only; waveform reconstruction
remains a later DSP responsibility.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as torch_functional


class DoubleConv(nn.Module):
    """Two spatial convolutions with ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DownBlock(nn.Module):
    """One encoder level: halve frequency/time resolution, then convolve."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(inputs))


class UpBlock(nn.Module):
    """One decoder level with learned upsampling and a skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(inputs)
        # Odd F/T dimensions (for example F=1025) cannot always be doubled
        # exactly. Interpolate only the learned decoder feature to the skip
        # tensor's size so the final model output exactly matches its input.
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = torch_functional.interpolate(
                upsampled,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(torch.cat((skip, upsampled), dim=1))


class StereoUNet(nn.Module):
    """Understandable four-level stereo spectrogram U-Net.

    Parameters
    ----------
    in_channels:
        Number of spectrogram input channels. Stage 3 uses ``2`` for L/R.
    out_channels:
        Number of mask channels. The fixed stereo separator uses ``2``.
    base_channels:
        Width of the first encoder level. ``16`` keeps the model practical on
        a university-project CPU while retaining four encoder/decoder levels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("channel counts must be positive integers")

        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.encoder1 = DoubleConv(in_channels, channels[0])
        self.encoder2 = DownBlock(channels[0], channels[1])
        self.encoder3 = DownBlock(channels[1], channels[2])
        self.encoder4 = DownBlock(channels[2], channels[3])
        self.bottleneck = DownBlock(channels[3], channels[3] * 2)

        self.decoder4 = UpBlock(channels[3] * 2, channels[3], channels[3])
        self.decoder3 = UpBlock(channels[3], channels[2], channels[2])
        self.decoder2 = UpBlock(channels[2], channels[1], channels[1])
        self.decoder1 = UpBlock(channels[1], channels[0], channels[0])
        self.mask_head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (batch, channels, frequency, time)")
        if inputs.shape[1] != self.encoder1.layers[0].in_channels:
            raise ValueError(
                "inputs has the wrong channel count: "
                f"expected {self.encoder1.layers[0].in_channels}, got {inputs.shape[1]}"
            )

        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        bottleneck = self.bottleneck(skip4)

        decoded = self.decoder4(bottleneck, skip4)
        decoded = self.decoder3(decoded, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder1(decoded, skip1)
        return torch.sigmoid(self.mask_head(decoded))


__all__ = ["DoubleConv", "DownBlock", "UpBlock", "StereoUNet"]
