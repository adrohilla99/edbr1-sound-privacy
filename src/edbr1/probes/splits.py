"""Leak-guarded probe-train / probe-test splits over LibriSpeech.

The whole privacy claim rests on the probe never seeing, during its own
training, anything that leaks the test answer. This module builds the splits and
**verifies the disjointness invariants at construction** (raising on any leak):

* ``mode='speaker_id'`` -- a closed set of speakers appears in *both* train and
  test (that is what closed-set identification means), but their **utterances are
  disjoint**. Top-1 accuracy is then measured on utterances the probe never saw.
* ``mode='generalization'`` -- **speakers are disjoint** between train and test
  (harder): ASR / inversion is scored on entirely unseen speakers.

In both modes the probe speakers are drawn from a subset (default ``dev-clean``)
and must be **disjoint from the encoder's training/overlay speakers**
(``exclude_speakers``), so any success is representational leakage rather than
memorisation of speakers the encoder saw. All selection is seeded/deterministic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

Mode = str  # "speaker_id" | "generalization"
_MODES = ("speaker_id", "generalization")


@dataclass(frozen=True)
class ProbeItem:
    """One probe example: an utterance path + its closed-set label + transcript."""

    path: Path
    speaker: int          # raw LibriSpeech speaker id
    speaker_label: int    # 0..N-1 closed-set index
    transcript: str       # upper-case words, or "" if transcripts not loaded


def load_transcripts(subset_dir: Path) -> dict[str, str]:
    """Map ``utterance_stem -> transcript`` from every ``*.trans.txt`` in a subset."""
    transcripts: dict[str, str] = {}
    for trans in subset_dir.rglob("*.trans.txt"):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, _, text = line.partition(" ")
            transcripts[utt_id] = text.strip()
    return transcripts


def speaker_utterances_with_transcripts(
    subset_dir: Path,
) -> tuple[dict[int, list[Path]], dict[str, str]]:
    """Index a subset into ``(speaker -> [flac paths], utt_stem -> transcript)``."""
    from edbr1.data.librispeech import speaker_utterances

    return speaker_utterances(subset_dir), load_transcripts(subset_dir)


def _item(path: Path, speaker: int, label: int, transcripts: dict[str, str] | None) -> ProbeItem:
    text = transcripts.get(path.stem, "") if transcripts else ""
    return ProbeItem(path=path, speaker=speaker, speaker_label=label, transcript=text)


@dataclass
class ProbeSplitResult:
    """A verified probe split; construction guarantees the leak guards hold."""

    mode: Mode
    speakers: list[int]           # the closed set (raw ids), sorted
    train: list[ProbeItem]
    test: list[ProbeItem]

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / len(self.speakers)

    def verify(self, exclude_speakers: frozenset[int]) -> None:
        """Raise if any disjointness invariant is violated (called at build time)."""
        if not self.train or not self.test:
            raise ValueError("probe split has an empty train or test partition")

        train_spk = {it.speaker for it in self.train}
        test_spk = {it.speaker for it in self.test}
        leaked = (train_spk | test_spk) & exclude_speakers
        if leaked:
            raise AssertionError(
                f"LEAK: probe speakers overlap the encoder's speakers: {sorted(leaked)}"
            )

        if self.mode == "speaker_id":
            if train_spk != test_spk:
                raise AssertionError(
                    "closed-set ID requires identical train/test speakers; "
                    f"train-only={sorted(train_spk - test_spk)} "
                    f"test-only={sorted(test_spk - train_spk)}"
                )
            utt_overlap = {it.path for it in self.train} & {it.path for it in self.test}
            if utt_overlap:
                raise AssertionError(
                    f"LEAK: {len(utt_overlap)} utterances in both train and test"
                )
        elif self.mode == "generalization":
            spk_overlap = train_spk & test_spk
            if spk_overlap:
                raise AssertionError(
                    f"LEAK: {len(spk_overlap)} speakers in both train and test (must be disjoint)"
                )
        else:  # pragma: no cover - guarded earlier
            raise ValueError(f"unknown mode {self.mode!r}")


def build_probe_split(
    speaker_utts: dict[int, list[Path]],
    *,
    num_speakers: int,
    mode: Mode = "speaker_id",
    test_fraction: float = 0.3,
    seed: int = 1337,
    exclude_speakers: frozenset[int] = frozenset(),
    transcripts: dict[str, str] | None = None,
    max_utts_per_speaker: int | None = None,
) -> ProbeSplitResult:
    """Build and verify a leak-guarded probe split (pure; no audio decoded)."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    rng = random.Random(seed)
    # Enough utterances to split, and never a speaker the encoder trained on.
    eligible = [
        s for s, utts in speaker_utts.items()
        if s not in exclude_speakers and len(utts) >= 2
    ]
    if len(eligible) < num_speakers:
        raise ValueError(
            f"only {len(eligible)} eligible speakers (need {num_speakers}); "
            "excluded the encoder's speakers and any with <2 utterances"
        )
    # Deterministic: most utterances first, speaker id as tie-break.
    eligible.sort(key=lambda s: (-len(speaker_utts[s]), s))
    speakers = sorted(eligible[:num_speakers])
    label = {s: i for i, s in enumerate(speakers)}

    def utts_for(s: int) -> list[Path]:
        u = sorted(speaker_utts[s])
        rng.shuffle(u)
        return u[:max_utts_per_speaker] if max_utts_per_speaker else u

    train: list[ProbeItem] = []
    test: list[ProbeItem] = []
    if mode == "speaker_id":
        for s in speakers:
            u = utts_for(s)
            n_test = max(1, int(round(len(u) * test_fraction)))
            for p in u[n_test:]:
                train.append(_item(p, s, label[s], transcripts))
            for p in u[:n_test]:
                test.append(_item(p, s, label[s], transcripts))
    else:  # generalization: split speakers, not utterances
        shuffled = speakers.copy()
        rng.shuffle(shuffled)
        n_test_spk = max(1, int(round(num_speakers * test_fraction)))
        test_speakers = set(shuffled[:n_test_spk])
        for s in speakers:
            bucket = test if s in test_speakers else train
            for p in utts_for(s):
                bucket.append(_item(p, s, label[s], transcripts))

    result = ProbeSplitResult(mode=mode, speakers=speakers, train=train, test=test)
    result.verify(exclude_speakers)  # hard guard: raises on any leak
    return result
