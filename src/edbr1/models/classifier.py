"""
Downstream classifier C on the (possibly quantised) latent grid.

Deliberately minimal -- a global average pool over the latent grid, a single
dropout, and a linear head -- mirroring the head of :class:`SmallAudioCNN` so
that the no-bottleneck control is architecturally faithful to the baseline. All
representational capacity lives in the encoder; the classifier just reads out the
latent, so the utility-vs-bitrate curve reflects what the bottleneck preserves
rather than extra classifier modelling.
"""
from __future__ import annotations

from torch import Tensor, nn


class LatentClassifier(nn.Module):
    """Global-average-pool + dropout + linear over a ``(B, D, F', T')`` latent."""

    def __init__(self, latent_dim: int, num_classes: int, *, dropout: float = 0.3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, latent: Tensor) -> Tensor:
        """latent: (B, D, F', T') -> logits (B, num_classes)."""
        pooled = self.pool(latent).flatten(1)
        return self.classifier(pooled)
