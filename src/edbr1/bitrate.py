"""
Honest bitrate accounting for the discrete (VQ) bottleneck.

A VQ operating point emits a fixed grid of discrete tokens per clip. Each token
is one of ``codebook_size`` symbols, so it carries ``log2(codebook_size)`` bits.
The emitted bitrate is therefore

    bits_per_second = tokens_per_second * log2(codebook_size)

which is the single number the utility-vs-bitrate curve is plotted against. This
module keeps that arithmetic in one pure-Python place (no torch) so it can be
unit-tested cheaply and reused by the trainer, the sweep runner and the configs.

The token count is deliberately taken from the *declared* latent grid
(``latent_freq * latent_frames``), not inferred from a tensor shape at runtime:
the encoder pools to exactly that grid, so the declared count is the true count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def tokens_per_second(tokens_per_clip: int, clip_seconds: float) -> float:
    """Token emission rate for a clip that yields ``tokens_per_clip`` tokens.

    ``tokens_per_clip`` is ``latent_freq * latent_frames`` for the 2-D latent
    grid. ``clip_seconds`` is the fixed per-sample clip length used for batching.
    """
    if tokens_per_clip < 0:
        raise ValueError(f"tokens_per_clip must be >= 0, got {tokens_per_clip}")
    if clip_seconds <= 0:
        raise ValueError(f"clip_seconds must be > 0, got {clip_seconds}")
    return tokens_per_clip / clip_seconds


def bits_per_token(codebook_size: int) -> float:
    """Bits carried by one token drawn from a codebook of ``codebook_size``.

    ``log2(codebook_size)``. A degenerate codebook of size 1 carries 0 bits.
    """
    if codebook_size < 1:
        raise ValueError(f"codebook_size must be >= 1, got {codebook_size}")
    return math.log2(codebook_size)


def bits_per_second(tokens_per_second: float, codebook_size: int) -> float:
    """Emitted bitrate ``tokens_per_second * log2(codebook_size)``."""
    if tokens_per_second < 0:
        raise ValueError(f"tokens_per_second must be >= 0, got {tokens_per_second}")
    return tokens_per_second * bits_per_token(codebook_size)


@dataclass(frozen=True)
class OperatingPoint:
    """A resolved VQ operating point and its honest bitrate accounting."""

    latent_freq: int
    latent_frames: int
    codebook_size: int
    clip_seconds: float

    @property
    def tokens_per_clip(self) -> int:
        return self.latent_freq * self.latent_frames

    @property
    def tokens_per_second(self) -> float:
        return tokens_per_second(self.tokens_per_clip, self.clip_seconds)

    @property
    def bits_per_token(self) -> float:
        return bits_per_token(self.codebook_size)

    @property
    def bits_per_second(self) -> float:
        return bits_per_second(self.tokens_per_second, self.codebook_size)

    def as_dict(self) -> dict[str, float | int]:
        """Flat, JSON/CSV-friendly summary for the sweep artifacts."""
        return {
            "latent_freq": self.latent_freq,
            "latent_frames": self.latent_frames,
            "codebook_size": self.codebook_size,
            "clip_seconds": self.clip_seconds,
            "tokens_per_clip": self.tokens_per_clip,
            "tokens_per_second": self.tokens_per_second,
            "bits_per_token": self.bits_per_token,
            "bits_per_second": self.bits_per_second,
        }
