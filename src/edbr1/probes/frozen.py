"""Frozen encoder + batched code extraction for the probes.

Loads a trained encoder/bottleneck checkpoint, **freezes it** (eval mode, no
grad), and emits the discrete codes a probe attacks. A probe therefore never
updates the encoder -- it only sees the transmitted indices. To match the
deployment / training distribution, probe speech is overlaid on held-out
UrbanSound8K scenes at the configured SNRs before encoding.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from edbr1.config import train_config_from_dict
from edbr1.data.overlay import mix_at_snr
from edbr1.features.melspec import LogMelExtractor
from edbr1.models import EncoderClassifier, nominal_frames_for
from edbr1.probes.splits import ProbeItem


class FrozenEncoder:
    """A trained encoder+bottleneck, frozen, exposing only its emitted codes."""

    def __init__(self, checkpoint_path: str | Path, device: torch.device) -> None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = train_config_from_dict(ckpt["config"])
        model = EncoderClassifier(
            self.config.encoder, self.config.bottleneck,
            num_classes=len(ckpt["class_names"]),
            n_mels=self.config.features.n_mels,
            nominal_frames=nominal_frames_for(self.config),
        )
        # Load encoder/bottleneck/classifier weights; ignore any adversary/grl keys.
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.encoder = model.encoder
        self.bottleneck = model.bottleneck
        # Mel extraction stays on CPU: overlay/mix happens on CPU waveforms and
        # emit_codes moves the resulting mels to the device. (Keeping it on the
        # device would break on the CPU waveforms fed by extract_codes.)
        self.extractor = LogMelExtractor(self.config.features)
        self.mean = ckpt["norm_mean"].to(device)
        self.std = ckpt["norm_std"].to(device)
        self.device = device
        self.codebook_size = self.config.bottleneck.codebook_size
        self.latent_freq = self.config.encoder.latent_freq
        self.latent_frames = self.config.encoder.latent_frames
        self.target_len = int(round(self.config.clip_seconds * self.config.features.sample_rate))

    def parameter_fingerprint(self) -> float:
        """Sum of all encoder+bottleneck weights AND buffers (the EMA codebook is
        a buffer), to assert nothing in the frozen path changes during probing."""
        total = 0.0
        for module in (self.encoder, self.bottleneck):
            for p in module.parameters():
                total += float(p.detach().double().sum())
            for buf in module.buffers():
                if buf.dtype.is_floating_point:
                    total += float(buf.detach().double().sum())
        return total

    @torch.no_grad()
    def emit_codes(self, mels: Tensor) -> Tensor:
        """(B, 1, n_mels, frames) log-mels -> (B, F', T') long code indices."""
        x = (mels.to(self.device) - self.mean) / self.std
        out = self.bottleneck(self.encoder(x))
        assert out.indices is not None, "frozen encoder has no discrete bottleneck"
        return out.indices


class ScenePool:
    """A bank of held-out UrbanSound8K scene waveforms (4 s, mono, target SR)."""

    def __init__(
        self, metadata: object, *, num_scenes: int, target_len: int,
        sample_rate: int, seed: int = 1337,
    ) -> None:
        import numpy as np
        import pandas as pd
        import soundfile as sf
        import torchaudio.functional as af

        assert isinstance(metadata, pd.DataFrame)
        rng = np.random.default_rng(seed)
        rows = metadata.sample(n=min(num_scenes, len(metadata)), random_state=seed)
        scenes: list[Tensor] = []
        for path in rows["path"]:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = torch.from_numpy(data).mean(dim=1)
            if sr != sample_rate:
                mono = af.resample(mono, sr, sample_rate)
            n = mono.shape[-1]
            if n >= target_len:
                start = int(rng.integers(0, n - target_len + 1))
                mono = mono[start:start + target_len]
            else:
                mono = torch.nn.functional.pad(mono, (0, target_len - n))
            scenes.append(mono.contiguous())
        self.scenes = torch.stack(scenes)

    def sample(self, generator: torch.Generator) -> Tensor:
        i = int(torch.randint(0, self.scenes.shape[0], (1,), generator=generator))
        return self.scenes[i]


@dataclass
class ProbeCodes:
    """Extracted, cached probe data: codes + all targets a probe might use."""

    indices: Tensor            # (N, F', T') long
    speaker_labels: Tensor     # (N,) long, closed-set index
    transcripts: list[str]
    clean_mels: Tensor | None  # (N, 1, n_mels, frames) clean-speech mel, or None


@torch.no_grad()
def extract_codes(
    frozen: FrozenEncoder,
    items: Sequence[ProbeItem],
    scene_pool: ScenePool,
    *,
    snr_choices: Sequence[float] = (0.0, 5.0, 10.0),
    seed: int = 1337,
    batch_size: int = 64,
    want_clean_mel: bool = False,
) -> ProbeCodes:
    """Overlay each utterance on a random scene, encode, and cache the codes.

    Deterministic given ``seed`` (scene choice + SNR). ``clean_mels`` (the mel of
    the *clean* speech, no scene) is the inversion target when ``want_clean_mel``.
    """
    import soundfile as sf

    gen = torch.Generator().manual_seed(seed)
    sr = frozen.config.features.sample_rate
    tlen = frozen.target_len
    idx_chunks: list[Tensor] = []
    clean_chunks: list[Tensor] = []
    mel_batch: list[Tensor] = []
    clean_batch: list[Tensor] = []

    def flush() -> None:
        if not mel_batch:
            return
        idx_chunks.append(frozen.emit_codes(torch.stack(mel_batch)).cpu())
        if want_clean_mel:
            clean_chunks.append(torch.stack(clean_batch))
        mel_batch.clear()
        clean_batch.clear()

    for it in items:
        data, file_sr = sf.read(str(it.path), dtype="float32", always_2d=True)
        speech = torch.from_numpy(data).mean(dim=1)
        if file_sr != sr:
            import torchaudio.functional as af
            speech = af.resample(speech, file_sr, sr)
        speech = (speech[:tlen] if speech.shape[-1] >= tlen
                  else torch.nn.functional.pad(speech, (0, tlen - speech.shape[-1])))
        scene = scene_pool.sample(gen)
        snr = snr_choices[int(torch.randint(0, len(snr_choices), (1,), generator=gen))]
        mixed = mix_at_snr(scene, speech, float(snr))
        mel_batch.append(frozen.extractor(mixed, sr).unsqueeze(0).cpu())
        if want_clean_mel:
            clean_batch.append(frozen.extractor(speech, sr).unsqueeze(0).cpu())
        if len(mel_batch) >= batch_size:
            flush()
    flush()

    return ProbeCodes(
        indices=torch.cat(idx_chunks),
        speaker_labels=torch.tensor([it.speaker_label for it in items], dtype=torch.long),
        transcripts=[it.transcript for it in items],
        clean_mels=torch.cat(clean_chunks) if want_clean_mel else None,
    )
