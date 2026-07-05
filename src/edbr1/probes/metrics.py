"""Probe metrics implemented in-repo (no jiwer/pesq/pystoi dependency).

* WER / CER via Levenshtein edit distance (Graves et al. 2006 use the same),
* log-spectral distance (LSD) on dB log-mels for the inverter,
* top-1 accuracy for the speaker probe.

PESQ/STOI need extra packages + a waveform reconstruction; they are attempted
best-effort by the runner and clearly reported as present-or-omitted.
"""
from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor


def edit_distance(ref: Sequence[object], hyp: Sequence[object]) -> int:
    """Levenshtein distance between two token sequences (DP, O(len*len))."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    prev_row = list(range(n + 1))
    for i in range(1, m + 1):
        cur_row = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur_row[j] = min(prev_row[j] + 1, cur_row[j - 1] + 1, prev_row[j - 1] + cost)
        prev_row = cur_row
    return prev_row[n]


def word_error_rate(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Aggregate WER = total word edits / total reference words (1.0 == chance-bad)."""
    errors = total = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        ref_words = ref.split()
        errors += edit_distance(ref_words, hyp.split())
        total += len(ref_words)
    return errors / max(total, 1)


def character_error_rate(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Aggregate CER = total char edits / total reference chars."""
    errors = total = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        errors += edit_distance(list(ref), list(hyp))
        total += len(ref)
    return errors / max(total, 1)


def log_spectral_distance(a: Tensor, b: Tensor) -> float:
    """LSD (dB) between two dB log-mels ``(..., n_mels, frames)``.

    Per frame, the RMS over mel bands of the dB difference; averaged over frames
    (and any batch dims). Lower is closer; silence-vs-speech gives the floor.
    """
    diff = a - b
    per_frame = diff.pow(2).mean(dim=-2).sqrt()  # RMS over mel bands
    return float(per_frame.mean())


def top1_accuracy(preds: Tensor, targets: Tensor) -> float:
    """Fraction of exact matches between predicted and target class indices."""
    if targets.numel() == 0:
        return 0.0
    return float((preds == targets).float().mean())
