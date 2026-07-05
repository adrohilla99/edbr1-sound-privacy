"""Train + evaluate the three leakage probes on extracted codes.

Shared by ``scripts/run_probes.py`` (the Phase-4a grid) and
``scripts/run_overlay_robustness.py`` (the Phase-4b SNR sweep). Each helper takes
already-extracted :class:`~edbr1.probes.frozen.ProbeCodes` (so the frozen encoder
is never re-run here) and returns a metrics dict.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from edbr1.probes.frozen import ProbeCodes
from edbr1.probes.metrics import (
    character_error_rate,
    log_spectral_distance,
    top1_accuracy,
    word_error_rate,
)
from edbr1.probes.models import CharVocab, CTCProbe, InverterProbe, SpeakerProbe, num_parameters


def _batches(n: int, bs: int, device: torch.device) -> list[torch.Tensor]:
    perm = torch.randperm(n, device=device)
    return [perm[i:i + bs] for i in range(0, n, bs)]


def train_speaker_probe(
    codebook_size: int, tr: ProbeCodes, te: ProbeCodes, num_speakers: int,
    device: torch.device, *, epochs: int, seed: int,
) -> dict[str, Any]:
    """Closed-set speaker-ID probe -> top-1 vs chance (1/N)."""
    torch.manual_seed(seed)
    probe = SpeakerProbe(codebook_size, num_speakers).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tr, y_tr = tr.indices.to(device), tr.speaker_labels.to(device)
    x_te, y_te = te.indices.to(device), te.speaker_labels.to(device)
    for _ in range(epochs):
        probe.train()
        for idx in _batches(len(x_tr), 128, device):
            opt.zero_grad()
            F.cross_entropy(probe(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        preds = probe(x_te).argmax(dim=1)
    return {
        "top1": top1_accuracy(preds.cpu(), y_te.cpu()),
        "chance": 1.0 / num_speakers,
        "n_test": int(len(y_te)),
        "probe_params": num_parameters(probe),
    }


def train_asr_probe(
    codebook_size: int, latent_freq: int, tr: ProbeCodes, te: ProbeCodes,
    device: torch.device, *, epochs: int, seed: int,
) -> dict[str, Any]:
    """From-scratch CTC ASR probe on codes -> WER/CER vs the ~1.0 ceiling."""
    torch.manual_seed(seed)
    random.seed(seed)
    vocab = CharVocab()
    frames = tr.indices.shape[2]

    def prep(codes: ProbeCodes) -> list[tuple[torch.Tensor, list[int], str]]:
        out = []
        for i, txt in enumerate(codes.transcripts):
            enc = vocab.encode(txt)
            if 0 < len(enc) < frames:  # CTC needs input_len > target_len
                out.append((codes.indices[i], enc, txt))
        return out

    train_items, test_items = prep(tr), prep(te)
    if len(train_items) < 16 or len(test_items) < 8:
        return {"skipped": f"too few utts fit T'={frames} (tr={len(train_items)}, "
                f"te={len(test_items)})", "input_frames": frames}

    probe = CTCProbe(codebook_size, vocab.size, latent_freq).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    ctc = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
    for _ in range(epochs):
        probe.train()
        random.shuffle(train_items)
        for i in range(0, len(train_items), 32):
            batch = train_items[i:i + 32]
            x = torch.stack([b[0] for b in batch]).to(device)
            logp = probe(x).permute(1, 0, 2)  # (T', B, vocab) for CTC
            targets = torch.tensor([c for b in batch for c in b[1]], device=device)
            tgt_len = torch.tensor([len(b[1]) for b in batch])
            in_len = torch.full((len(batch),), frames)
            opt.zero_grad()
            ctc(logp, targets, in_len, tgt_len).backward()
            opt.step()
    probe.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for idx, _enc, txt in test_items:
            logp = probe(idx.unsqueeze(0).to(device))[0]
            hyps.append(vocab.greedy_decode(logp))
            refs.append(txt)
    return {
        "wer": word_error_rate(refs, hyps),
        "cer": character_error_rate(refs, hyps),
        "ceiling_wer": 1.0,
        "n_test": len(test_items),
        "input_frames": frames,
        "probe_params": num_parameters(probe),
    }


def train_inverter_probe(
    codebook_size: int, tr: ProbeCodes, te: ProbeCodes, silence_mel: torch.Tensor,
    device: torch.device, *, epochs: int, seed: int, fig_path: Path | None = None,
) -> dict[str, Any]:
    """Learned code->mel inverter -> LSD/MSE vs clean speech, vs a silence floor."""
    assert tr.clean_mels is not None and te.clean_mels is not None
    torch.manual_seed(seed)
    n_mels, frames = tr.clean_mels.shape[2], tr.clean_mels.shape[3]
    probe = InverterProbe(codebook_size, out_mels=n_mels, out_frames=frames).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    x_tr, y_tr = tr.indices.to(device), tr.clean_mels.to(device)
    x_te, y_te = te.indices.to(device), te.clean_mels.to(device)
    for _ in range(epochs):
        probe.train()
        for idx in _batches(len(x_tr), 64, device):
            opt.zero_grad()
            F.mse_loss(probe(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        recon = probe(x_te)
    silence = silence_mel.to(y_te).expand_as(y_te)
    if fig_path is not None:
        save_spectrogram_examples(y_te[:3].cpu(), recon[:3].cpu(), fig_path)
    return {
        "lsd": log_spectral_distance(recon.cpu(), y_te.cpu()),
        "mse": float(F.mse_loss(recon, y_te)),
        "lsd_silence_floor": log_spectral_distance(silence.cpu(), y_te.cpu()),
        "n_test": int(len(x_te)),
        "probe_params": num_parameters(probe),
        "note": "PESQ/STOI omitted (packages absent; would need waveform reconstruction)",
    }


def save_spectrogram_examples(clean: torch.Tensor, recon: torch.Tensor, path: Path) -> None:
    """Save paired clean-vs-reconstructed mel spectrograms (no audio)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = clean.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 4))
    for j in range(n):
        axes[0, j].imshow(clean[j, 0], origin="lower", aspect="auto")
        axes[0, j].set_title("clean speech" if j == 0 else "")
        axes[1, j].imshow(recon[j, 0], origin="lower", aspect="auto")
        axes[1, j].set_title("inverter recon" if j == 0 else "")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
