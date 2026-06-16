"""
A compact CNN for UrbanSound8K log-mel classification.

Design rationale
----------------
This is intentionally a *small* network in the lineage of Piczak (2015,
"Environmental sound classification with convolutional neural networks")
and Salamon & Bello's SB-CNN (2017). Those works show that a few
convolutional blocks over a log-mel input reach the ~73-76% accuracy /
macro-F1 band on UrbanSound8K's 10-fold protocol *without* a large
network -- which is exactly the operating point this dissertation wants as
a reference before introducing the privacy bottleneck.

Architecture (≈0.5M params):
    4 x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2 (first 3 blocks)]
    -> AdaptiveAvgPool2d(1) -> Dropout -> Linear -> logits

The global adaptive pool makes the head invariant to the exact number of
time frames, so variable clip lengths and feature configs work unchanged.
Batch norm + a single dropout before the classifier provide the modest
regularisation these baselines rely on.
"""
from __future__ import annotations

from torch import Tensor, nn


def _conv_block(in_ch: int, out_ch: int, *, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*layers)


class SmallAudioCNN(nn.Module):
    """Compact 4-block CNN over a single-channel log-mel spectrogram.

    Args:
        num_classes: number of output classes (10 for UrbanSound8K).
        in_channels: input channels (1 for a mono log-mel).
        channels: per-block channel widths.
        dropout: dropout probability before the linear classifier.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        channels: tuple[int, int, int, int] = (32, 64, 128, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            _conv_block(in_channels, c1, pool=True),
            _conv_block(c1, c2, pool=True),
            _conv_block(c2, c3, pool=True),
            _conv_block(c3, c4, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, in_channels, n_mels, n_frames) -> logits (batch, num_classes)."""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
