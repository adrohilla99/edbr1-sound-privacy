"""
Training-time audio augmentation (training folds only).

Two families of transform, both driven by :class:`edbr1.config.AugmentConfig`:

* **Waveform level** -- a fast, zero-filled random time shift (pure torch) and,
  behind off-by-default flags, mild pitch shift / time stretch (via librosa,
  slower). Applied before the fixed-length crop so a length-changing stretch is
  re-normalised by the dataset's length fix.
* **Spectrogram level** -- SpecAugment (Park et al., 2019): a few random
  frequency and time masks over the log-mel spectrogram. This is the cheapest,
  highest-value addition and is on by default when augmentation is enabled.

None of these should ever touch the held-out test fold; the dataset only calls
them when built with ``train=True`` and an enabled config.
"""
from __future__ import annotations

import torch
from torch import Tensor

from edbr1.config import AugmentConfig


def _time_shift(wav: Tensor, max_shift: int) -> Tensor:
    """Randomly shift a mono waveform by up to ``max_shift`` samples (zero-fill)."""
    if max_shift <= 0:
        return wav
    k = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    if k == 0:
        return wav
    out = torch.roll(wav, k)
    if k > 0:
        out[:k] = 0.0
    else:
        out[k:] = 0.0
    return out


def _pitch_shift(wav: Tensor, sr: int, n_steps: float) -> Tensor:
    import librosa

    shifted = librosa.effects.pitch_shift(wav.numpy(), sr=sr, n_steps=float(n_steps))
    return torch.from_numpy(shifted)


def _time_stretch(wav: Tensor, rate: float) -> Tensor:
    import librosa

    stretched = librosa.effects.time_stretch(wav.numpy(), rate=float(rate))
    return torch.from_numpy(stretched)


def augment_waveform(wav: Tensor, sr: int, cfg: AugmentConfig) -> Tensor:
    """Apply the enabled waveform-level transforms to a mono ``wav``.

    Each transform fires independently with probability ``cfg.waveform_prob``.
    Pitch shift / time stretch are off by default (they are slower).
    """
    if cfg.pitch_shift and torch.rand(1).item() < cfg.waveform_prob:
        n_steps = float(
            torch.empty(1).uniform_(-cfg.pitch_shift_steps, cfg.pitch_shift_steps).item()
        )
        wav = _pitch_shift(wav, sr, n_steps)
    if cfg.time_stretch and torch.rand(1).item() < cfg.waveform_prob:
        rate = float(
            torch.empty(1).uniform_(cfg.time_stretch_min, cfg.time_stretch_max).item()
        )
        wav = _time_stretch(wav, rate)
    if cfg.time_shift and torch.rand(1).item() < cfg.waveform_prob:
        max_shift = int(round(cfg.time_shift_ms / 1000.0 * sr))
        wav = _time_shift(wav, max_shift)
    return wav.contiguous()


def spec_augment(spec: Tensor, cfg: AugmentConfig) -> Tensor:
    """Apply SpecAugment frequency/time masks to a ``(n_mels, n_frames)`` log-mel.

    Masked regions are set to the spectrogram's own mean (the SpecAugment
    recommendation), which is well defined before per-band normalisation. The
    input tensor is modified and returned.
    """
    if not cfg.spec_augment or torch.rand(1).item() >= cfg.spec_augment_prob:
        return spec
    n_mels, n_frames = spec.shape
    fill = float(spec.mean())

    for _ in range(cfg.freq_masks):
        f = int(torch.randint(0, cfg.freq_mask_param + 1, (1,)).item())
        if f > 0 and n_mels - f > 0:
            f0 = int(torch.randint(0, n_mels - f + 1, (1,)).item())
            spec[f0 : f0 + f, :] = fill

    for _ in range(cfg.time_masks):
        t = int(torch.randint(0, cfg.time_mask_param + 1, (1,)).item())
        if t > 0 and n_frames - t > 0:
            t0 = int(torch.randint(0, n_frames - t + 1, (1,)).item())
            spec[:, t0 : t0 + t] = fill

    return spec
