"""
Encoder -> bottleneck -> classifier model (the refactored baseline).

``EncoderClassifier`` composes the three pieces of the privacy pipeline:

    log-mel --E--> latent grid --B--> (quantised) latent --C--> class logits

With ``BottleneckConfig(type='none')`` the bottleneck is the identity and the
network is an ordinary encoder->classifier model -- the *control* that must
reproduce the canonical baseline. With ``type='vq'`` a VQ-VAE bottleneck emits a
discrete, low-bitrate latent, and ``forward`` returns the auxiliary VQ loss and
codebook-usage stats alongside the logits so the trainer can add the loss and log
the bitrate/usage.
"""
from __future__ import annotations

from torch import Tensor, nn

from edbr1.config import BottleneckConfig, EncoderConfig, TrainConfig
from edbr1.models.bottleneck import BottleneckOutput, build_bottleneck
from edbr1.models.classifier import LatentClassifier
from edbr1.models.encoder import AudioEncoder


class EncoderClassifier(nn.Module):
    """Compose encoder E, bottleneck B and classifier C into one model."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        bottleneck_config: BottleneckConfig,
        num_classes: int,
        *,
        in_channels: int = 1,
        n_mels: int = 64,
        nominal_frames: int = 401,
    ) -> None:
        super().__init__()
        self.encoder = AudioEncoder(
            encoder_config,
            in_channels=in_channels,
            n_mels=n_mels,
            nominal_frames=nominal_frames,
        )
        self.bottleneck = build_bottleneck(bottleneck_config, encoder_config.latent_dim)
        self.classifier = LatentClassifier(
            encoder_config.latent_dim,
            num_classes,
            dropout=encoder_config.dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, BottleneckOutput]:
        """x: (B, 1, n_mels, n_frames) -> (logits, bottleneck output)."""
        z = self.encoder(x)
        bottleneck = self.bottleneck(z)
        logits = self.classifier(bottleneck.latent)
        return logits, bottleneck

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encoder_parameters(self) -> int:
        """Parameter count of the on-device encoder E (target < 500K)."""
        return self.encoder.num_parameters()


def nominal_frames_for(config: TrainConfig) -> int:
    """Expected time-frame count of a fixed-length clip under ``config``.

    Matches torchaudio's centred ``MelSpectrogram`` framing: ``L // hop + 1`` for
    a signal of ``clip_seconds * sample_rate`` samples. Used only to pick the
    encoder's time-downsampling depth (the exact frame count is guaranteed by the
    encoder's final adaptive pool).
    """
    samples = int(round(config.clip_seconds * config.features.sample_rate))
    return samples // config.features.hop_length + 1


def build_model(config: TrainConfig, num_classes: int) -> nn.Module:
    """Build the model named by ``config.model``.

    ``'cnn'`` returns the original :class:`SmallAudioCNN` (legacy baseline path,
    unchanged). ``'encoder_classifier'`` returns the refactored
    :class:`EncoderClassifier` wired from the encoder/bottleneck configs.
    """
    from edbr1.models.cnn import SmallAudioCNN

    if config.model == "cnn":
        return SmallAudioCNN(num_classes=num_classes)
    return EncoderClassifier(
        config.encoder,
        config.bottleneck,
        num_classes,
        n_mels=config.features.n_mels,
        nominal_frames=nominal_frames_for(config),
    )
