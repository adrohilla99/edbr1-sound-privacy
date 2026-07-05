"""Probe networks: speaker-ID, CTC ASR, and a learned mel inverter.

All read the discrete codes through their own learned embedding (so they are at
least as strong as, and independent of, the Phase-3 adversary that read the
codebook latent directly). The speaker probe is deliberately larger than that
adversary head -- the whole point is a *stronger* attacker.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SpeakerProbe(nn.Module):
    """Embedding + 2-layer conv over the code grid + MLP -> N-way speaker logits."""

    def __init__(
        self, codebook_size: int, num_speakers: int, *,
        embed_dim: int = 64, channels: int = 128, hidden: int = 256, dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(codebook_size, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden, num_speakers),
        )

    def forward(self, indices: Tensor) -> Tensor:
        """(B, F', T') code indices -> (B, num_speakers) logits."""
        e = self.embed(indices).permute(0, 3, 1, 2)  # (B, embed, F', T')
        return self.head(self.pool(self.conv(e)).flatten(1))


class CharVocab:
    """Upper-case character vocabulary for CTC (index 0 is the CTC blank)."""

    def __init__(self) -> None:
        self.chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")
        self.blank = 0
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.size = len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text.upper() if c in self.char_to_idx]

    def greedy_decode(self, logprobs: Tensor) -> str:
        """(T, vocab) log-probs -> string (collapse repeats, drop blanks)."""
        ids = logprobs.argmax(dim=-1).tolist()
        out: list[str] = []
        prev: int | None = None
        for i in ids:
            if i != prev and i != self.blank:
                out.append(self.idx_to_char.get(i, ""))
            prev = i
        return "".join(out)


class CTCProbe(nn.Module):
    """Embedding + BiLSTM over the code time axis -> per-frame char log-probs."""

    def __init__(
        self, codebook_size: int, vocab_size: int, latent_freq: int, *,
        embed_dim: int = 64, hidden: int = 256, layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_freq = latent_freq
        self.embed = nn.Embedding(codebook_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim * latent_freq, hidden, num_layers=layers,
            batch_first=True, bidirectional=True, dropout=0.2 if layers > 1 else 0.0,
        )
        self.out = nn.Linear(2 * hidden, vocab_size)

    def forward(self, indices: Tensor) -> Tensor:
        """(B, F', T') -> (B, T', vocab) log-probs (CTC time axis is T')."""
        b, f, t = indices.shape
        e = self.embed(indices).permute(0, 2, 1, 3).reshape(b, t, f * self.embed_dim)
        h, _ = self.lstm(e)
        return self.out(h).log_softmax(dim=-1)


class InverterProbe(nn.Module):
    """Embedding + conv + resize -> reconstructed ``(1, n_mels, frames)`` mel."""

    def __init__(
        self, codebook_size: int, *, embed_dim: int = 64,
        out_mels: int = 64, out_frames: int = 401,
    ) -> None:
        super().__init__()
        self.out_mels, self.out_frames = out_mels, out_frames
        self.embed = nn.Embedding(codebook_size, embed_dim)
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.final = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, indices: Tensor) -> Tensor:
        e = self.embed(indices).permute(0, 3, 1, 2)  # (B, embed, F', T')
        h = self.net(e)
        h = F.interpolate(h, size=(self.out_mels, self.out_frames), mode="bilinear",
                          align_corners=False)
        return self.final(h)


def num_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
