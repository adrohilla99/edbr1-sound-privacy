"""Gradient-reversal adversary for speech-suppression training.

The training-time adversary tries to predict a speech attribute (speaker id over
a closed set, or "no speech") from the code; a **gradient reversal layer** (Ganin
& Lempitsky, 2015) between the code and the adversary makes the encoder fight it,
so the encoder is pushed to drop speech-predictive information while the
classifier keeps the urban-sound-class information.

This is deliberately a *modest* adversary -- the training-time opponent, not the
independent, stronger evaluation probes of Phase 4. Its accuracy is only an
internal sanity signal, never a privacy claim.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from edbr1.config import BottleneckConfig, EncoderConfig
from edbr1.models.bottleneck import BottleneckOutput
from edbr1.models.encoder_classifier import EncoderClassifier


class _GradientReversal(torch.autograd.Function):
    """Identity forward; negated, ``lambda``-scaled gradient backward."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        (grad_output,) = grad_outputs
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: Tensor, lambda_: float) -> Tensor:
    """Reverse-and-scale the gradient flowing back through ``x`` by ``lambda_``."""
    return _GradientReversal.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    """Module wrapper around :func:`gradient_reversal` with a settable ``lambda_``.

    ``lambda_`` is a plain attribute so a warmup schedule can update it per epoch
    (e.g. ``model.grl.lambda_ = ...``). It is the *reversal strength* only; the
    adversary head itself always learns at full rate (it sits after the layer).
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return gradient_reversal(x, self.lambda_)


class AdversaryHead(nn.Module):
    """Small MLP on the global-average-pooled code -> speech-attribute logits."""

    def __init__(
        self, latent_dim: int, num_classes: int, *, hidden_dim: int = 128, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, latent: Tensor) -> Tensor:
        """latent: (B, D, F', T') -> (B, num_classes) speech-attribute logits."""
        return self.net(self.pool(latent).flatten(1))


class AdversarialEncoderClassifier(EncoderClassifier):
    """``EncoderClassifier`` with a gradient-reversal speech adversary on the code.

    ``forward`` returns ``(class_logits, bottleneck_output, adversary_logits)``.
    The adversary reads the quantised latent through the GRL, so its gradient
    reaches the encoder reversed (via the straight-through estimator) -- the
    encoder learns to make the code un-predictive of the speech attribute.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        bottleneck_config: BottleneckConfig,
        num_classes: int,
        adversary_classes: int,
        *,
        adversary_hidden: int = 128,
        in_channels: int = 1,
        n_mels: int = 64,
        nominal_frames: int = 401,
    ) -> None:
        super().__init__(
            encoder_config, bottleneck_config, num_classes,
            in_channels=in_channels, n_mels=n_mels, nominal_frames=nominal_frames,
        )
        self.grl = GradientReversalLayer(0.0)
        self.adversary = AdversaryHead(
            encoder_config.latent_dim, adversary_classes,
            hidden_dim=adversary_hidden, dropout=encoder_config.dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, BottleneckOutput, Tensor]:  # type: ignore[override]
        z = self.encoder(x)
        bottleneck = self.bottleneck(z)
        logits = self.classifier(bottleneck.latent)
        adv_logits = self.adversary(self.grl(bottleneck.latent))
        return logits, bottleneck, adv_logits
