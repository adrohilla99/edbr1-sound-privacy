"""Speech-over-scene overlay for the adversarial training stream.

Mixes a LibriSpeech speech segment into an UrbanSound8K scene at a controlled
signal-to-noise ratio (speech = signal, urban scene = noise) -- the "loud
argument in the street" condition. The overlay is **train-only**: it is attached
to the training dataset alone, never to the val/test passes, so the held-out
UrbanSound8K fold stays clean and no LibriSpeech speaker crosses the boundary.

Each call returns the mixed waveform and an integer speech attribute:
``0`` (no speech) or ``1..N`` (the closed-set speaker id), which is what the
gradient-reversal adversary is trained to predict from the code.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from edbr1.data.librispeech import SpeechPool

_EPS = 1e-8


def mix_at_snr(scene: Tensor, speech: Tensor, snr_db: float) -> Tensor:
    """Mix ``speech`` into ``scene`` at ``snr_db`` dB (speech = signal).

    Scales the speech by RMS so that ``10*log10(P_speech / P_scene) == snr_db``,
    then adds it to the scene. Silent scene/speech (near-zero RMS) is returned
    unchanged. Both inputs are 1-D and the same length.
    """
    scene_rms = scene.pow(2).mean().sqrt()
    speech_rms = speech.pow(2).mean().sqrt()
    # Guard silent scene/speech first (a clamp here would defeat this check).
    if float(speech_rms) <= _EPS or float(scene_rms) <= _EPS:
        return scene
    gain = (scene_rms / speech_rms) * (10.0 ** (snr_db / 20.0))
    return scene + gain * speech


class SpeechOverlay:
    """Sample a speech segment and mix it into a scene at a random SNR.

    With probability ``overlay_prob`` a segment from ``pool`` is mixed in at an
    SNR drawn uniformly from ``snr_choices`` (dB); otherwise the scene is left
    clean and the label is ``0`` (no speech). Randomness uses the ambient torch
    RNG so it follows the dataset's per-worker seeding, exactly like augmentation.
    """

    def __init__(
        self,
        pool: SpeechPool,
        *,
        overlay_prob: float = 0.5,
        snr_choices: Sequence[float] = (0.0, 5.0, 10.0),
    ) -> None:
        if not 0.0 <= overlay_prob <= 1.0:
            raise ValueError(f"overlay_prob must be in [0, 1], got {overlay_prob}")
        if len(snr_choices) == 0:
            raise ValueError("snr_choices must be non-empty")
        self.pool = pool
        self.overlay_prob = overlay_prob
        self.snr_choices = tuple(float(s) for s in snr_choices)

    @property
    def num_classes(self) -> int:
        return self.pool.num_classes

    def apply(self, scene: Tensor) -> tuple[Tensor, int]:
        """Return ``(mixed_or_clean_scene, speech_label)`` for one scene."""
        if float(torch.rand(())) >= self.overlay_prob:
            return scene, 0  # no speech present
        speech, label = self.pool.sample()
        snr = self.snr_choices[int(torch.randint(0, len(self.snr_choices), (1,)))]
        return mix_at_snr(scene, speech, snr), label
