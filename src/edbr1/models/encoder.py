"""
On-device audio encoder E: a compact depthwise-separable conv stack.

The encoder maps a single-channel log-mel spectrogram to a latent grid
``(latent_dim, latent_freq, latent_frames)`` -- the sequence of tokens that the
downstream bottleneck quantises. It is intentionally a *small*, MobileNet-style
network (depthwise 3x3 + pointwise 1x1 blocks) so it is a plausible on-device
front end and stays well under 500K parameters (the target for E).

Downsampling to the target grid is honest: the number of frequency/time
halvings is derived from the target grid so the pre-pool feature map is never
smaller than the target, and a final ``AdaptiveAvgPool2d`` snaps to the exact
``(latent_freq, latent_frames)``. Every emitted token therefore corresponds to
a genuine strided-conv receptive field -- the token count is never inflated by
upsampling. This matters because token count sets the bitrate.

The trunk depth adapts to the operating point: it uses as many blocks as the
downsampling needs, but never fewer than ``min_depth`` (a capacity floor so the
no-bottleneck control still reaches the baseline). Channels ramp from
``base_channels`` towards ``latent_dim``.
"""
from __future__ import annotations

import math

from torch import Tensor, nn

from edbr1.config import EncoderConfig


def _depthwise_separable_block(
    in_ch: int, out_ch: int, *, pool: tuple[int, int]
) -> nn.Sequential:
    """A MobileNet-style block: depthwise 3x3 -> pointwise 1x1, each BN+ReLU.

    ``pool`` gives the (freq, time) max-pool factors; a factor of 1 on an axis
    leaves it unchanged. Pooling is skipped entirely when both factors are 1.
    """
    layers: list[nn.Module] = [
        # Depthwise: one 3x3 filter per input channel (groups=in_ch).
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        # Pointwise: 1x1 mixes channels and changes width.
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool != (1, 1):
        layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool))
    return nn.Sequential(*layers)


def _halvings(numerator: int, target: int) -> int:
    """How many /2 stages fit before reaching ``target`` (floor, clamped >= 0).

    ``floor(log2(numerator / target))`` so that ``numerator >> k >= target``:
    the strided trunk downsamples *towards but not past* the target grid, and
    the final adaptive pool covers the small remainder.
    """
    if numerator <= target:
        return 0
    return int(math.floor(math.log2(numerator / target)))


class AudioEncoder(nn.Module):
    """Depthwise-separable encoder producing a ``(B, D, F', T')`` latent grid.

    Args:
        config: encoder architecture + target latent grid.
        in_channels: input channels (1 for a mono log-mel).
        n_mels: mel bands of the input (frequency axis to downsample from).
        nominal_frames: expected number of time frames of the input (time axis
            to downsample from). Only used to choose the number of time-halving
            stages; the final adaptive pool guarantees the exact frame count.
    """

    def __init__(
        self,
        config: EncoderConfig,
        *,
        in_channels: int = 1,
        n_mels: int = 64,
        nominal_frames: int = 401,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_mels = n_mels
        self.nominal_frames = nominal_frames

        freq_halvings = _halvings(n_mels, config.latent_freq)
        time_halvings = _halvings(nominal_frames, config.latent_frames)
        n_blocks = max(freq_halvings, time_halvings, config.min_depth)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, config.base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        in_ch = config.base_channels
        for i in range(n_blocks):
            out_ch = min(config.latent_dim, config.base_channels * 2 ** (i + 1))
            pool = (
                2 if i < freq_halvings else 1,
                2 if i < time_halvings else 1,
            )
            blocks.append(_depthwise_separable_block(in_ch, out_ch, pool=pool))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Project to the codebook vector dimension and snap to the exact grid.
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, config.latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.latent_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((config.latent_freq, config.latent_frames))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, in_channels, n_mels, n_frames) -> latent (B, D, F', T')."""
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return self.pool(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
