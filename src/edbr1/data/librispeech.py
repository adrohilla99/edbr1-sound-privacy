"""LibriSpeech speaker index and a preloaded speech-segment pool for overlays.

Phase 3 mixes LibriSpeech speech into UrbanSound8K scenes so a training-time
adversary has a speech target to predict from the code. This module provides:

* :func:`speaker_utterances` -- index a permitted LibriSpeech subset into
  ``speaker_id -> [flac paths]`` straight from the on-disk directory tree.
* :class:`SpeechPool` -- a deterministic, **train-only** closed set of speakers
  with a fixed bank of fixed-length mono segments preloaded (and disk-cached) for
  fast per-item sampling.

LibriSpeech audio is natively 16 kHz mono, so building a segment only needs a
crop/pad. Only the three permitted, checksum-verified subsets exist on disk
(see ``scripts/download_librispeech.py``); ``train-clean-100`` is the default
source and the closed speaker set is drawn from it alone, so speakers never
cross into the (clean) UrbanSound8K test fold.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# The three subsets the downloader permits; the closed speaker set for the
# adversary is drawn from a *training* subset only.
PERMITTED_SUBSETS = ("train-clean-100", "dev-clean", "test-clean")


def speaker_utterances(subset_dir: Path) -> dict[int, list[Path]]:
    """Map ``speaker_id -> sorted list of .flac utterances`` for a subset dir.

    ``subset_dir`` is e.g. ``data/raw/librispeech/LibriSpeech/train-clean-100``,
    whose immediate children are per-speaker directories named by speaker id.
    """
    index: dict[int, list[Path]] = {}
    for spk_dir in sorted(subset_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        try:
            speaker = int(spk_dir.name)
        except ValueError:
            continue
        flacs = sorted(spk_dir.rglob("*.flac"))
        if flacs:
            index[speaker] = flacs
    return index


class SpeechPool:
    """A fixed, train-only bank of mono speech segments over a closed speaker set.

    The closed set is the ``num_speakers`` speakers (of the chosen subset) with
    the most utterances, tie-broken by id, so the selection is deterministic.
    Each speaker contributes ``segments_per_speaker`` fixed-length segments,
    yielding labels ``1..num_speakers`` (label ``0`` is reserved by the overlay
    for "no speech"). The assembled bank is cached to disk so it is decoded once.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        subset: str = "train-clean-100",
        num_speakers: int = 20,
        segments_per_speaker: int = 50,
        segment_seconds: float = 4.0,
        sample_rate: int = 16_000,
        seed: int = 1337,
        cache_dir: str | Path | None = None,
    ) -> None:
        if subset not in PERMITTED_SUBSETS:
            raise ValueError(f"subset must be one of {PERMITTED_SUBSETS}, got {subset!r}")
        self.subset = subset
        self.num_speakers = num_speakers
        self.segment_len = int(round(segment_seconds * sample_rate))
        self.sample_rate = sample_rate
        self.seed = seed
        self._subset_dir = Path(root) / subset

        cached = self._cache_path(cache_dir)
        if cached is not None and cached.exists():
            blob = torch.load(cached)
            self.segments = blob["segments"]
            self.labels = blob["labels"]
            self.speaker_ids = list(blob["speaker_ids"])
        else:
            self.segments, self.labels, self.speaker_ids = self._build(segments_per_speaker)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"segments": self.segments, "labels": self.labels,
                     "speaker_ids": self.speaker_ids},
                    cached,
                )
        # The segment bank can be a few hundred MB; put it in shared memory so
        # spawned DataLoader workers reference one copy instead of pickling their own.
        self.segments = self.segments.contiguous()
        self.segments.share_memory_()
        self.labels.share_memory_()

    @property
    def num_classes(self) -> int:
        """Adversary output classes: no-speech (0) + one per closed-set speaker."""
        return self.num_speakers + 1

    def _cache_path(self, cache_dir: str | Path | None) -> Path | None:
        if cache_dir is None:
            return None
        key = (
            f"{self.subset}|N={self.num_speakers}|L={self.segment_len}"
            f"|sr={self.sample_rate}|seed={self.seed}"
        )
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        return Path(cache_dir) / f"speechpool_{self.subset}_{self.num_speakers}spk_{digest}.pt"

    def _closed_set(self, index: dict[int, list[Path]], need: int) -> list[int]:
        """Pick the ``num_speakers`` speakers with the most utterances (>= need)."""
        eligible = [s for s, u in index.items() if len(u) >= need]
        if len(eligible) < self.num_speakers:
            raise ValueError(
                f"Only {len(eligible)} {self.subset} speakers have >= {need} "
                f"utterances; need {self.num_speakers}."
            )
        # Most utterances first, id as a deterministic tie-break.
        eligible.sort(key=lambda s: (-len(index[s]), s))
        return sorted(eligible[: self.num_speakers])

    def _build(self, segments_per_speaker: int) -> tuple[Tensor, Tensor, list[int]]:
        """Decode a deterministic bank of segments for the closed speaker set."""
        import soundfile as sf  # lazy: only when actually decoding

        index = speaker_utterances(self._subset_dir)
        speaker_ids = self._closed_set(index, segments_per_speaker)

        segments: list[Tensor] = []
        labels: list[int] = []
        for label, speaker in enumerate(speaker_ids, start=1):
            rng = np.random.default_rng(self.seed + speaker)
            utts = index[speaker]
            chosen = rng.permutation(len(utts))[:segments_per_speaker]
            for j in chosen:
                data, sr = sf.read(str(utts[j]), dtype="float32", always_2d=True)
                if sr != self.sample_rate:
                    raise ValueError(
                        f"Expected {self.sample_rate} Hz LibriSpeech, got {sr} at {utts[j]}"
                    )
                wav = torch.from_numpy(data).mean(dim=1)  # (samples,) mono
                segments.append(self._fixed_window(wav, rng))
                labels.append(label)

        return torch.stack(segments), torch.tensor(labels, dtype=torch.long), speaker_ids

    def _fixed_window(self, wav: Tensor, rng: np.random.Generator) -> Tensor:
        """A ``segment_len`` window: random start if long, zero-pad if short."""
        n = wav.shape[-1]
        if n >= self.segment_len:
            start = int(rng.integers(0, n - self.segment_len + 1))
            return wav[start : start + self.segment_len].contiguous()
        return torch.nn.functional.pad(wav, (0, self.segment_len - n))

    def sample(self, generator: torch.Generator | None = None) -> tuple[Tensor, int]:
        """Return a random ``(segment, speaker_label)`` (label in ``1..N``)."""
        i = int(torch.randint(0, self.segments.shape[0], (1,), generator=generator))
        return self.segments[i], int(self.labels[i])
