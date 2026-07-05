"""Leak-guard tests for probe splits -- the load-bearing privacy discipline."""
from __future__ import annotations

from pathlib import Path

import pytest

from edbr1.probes.splits import ProbeItem, ProbeSplitResult, build_probe_split


def _synth(n_speakers: int = 6, utts: int = 10) -> dict[int, list[Path]]:
    """A synthetic speaker->utterances index (no audio, no data dependency)."""
    return {
        1000 + s: [Path(f"/x/{1000 + s}/{1000 + s}-0-{u}.flac") for u in range(utts)]
        for s in range(n_speakers)
    }


def test_speaker_id_split_same_speakers_disjoint_utterances():
    r = build_probe_split(_synth(), num_speakers=4, mode="speaker_id", seed=1)
    train_spk = {it.speaker for it in r.train}
    test_spk = {it.speaker for it in r.test}
    assert train_spk == test_spk == set(r.speakers)  # closed set on both sides
    assert {it.path for it in r.train}.isdisjoint({it.path for it in r.test})  # no utt leak
    assert r.chance_accuracy == pytest.approx(0.25)


def test_generalization_split_speakers_are_disjoint():
    r = build_probe_split(_synth(n_speakers=8), num_speakers=8, mode="generalization", seed=2)
    train_spk = {it.speaker for it in r.train}
    test_spk = {it.speaker for it in r.test}
    assert train_spk.isdisjoint(test_spk)  # unseen speakers at test
    assert train_spk and test_spk


def test_split_excludes_encoder_speakers():
    # Speakers the encoder trained on must never appear among probe speakers.
    exclude = frozenset({1000, 1001, 1002})
    r = build_probe_split(
        _synth(n_speakers=8), num_speakers=4, mode="speaker_id",
        exclude_speakers=exclude, seed=3,
    )
    assert set(r.speakers).isdisjoint(exclude)


def test_split_is_deterministic():
    a = build_probe_split(_synth(), num_speakers=4, seed=7)
    b = build_probe_split(_synth(), num_speakers=4, seed=7)
    assert [it.path for it in a.test] == [it.path for it in b.test]


def test_split_rejects_insufficient_speakers():
    with pytest.raises(ValueError, match="eligible speakers"):
        build_probe_split(_synth(n_speakers=3), num_speakers=10)


def test_verify_catches_utterance_leak():
    shared = Path("/x/1-0-0.flac")
    leaky = ProbeSplitResult(
        mode="speaker_id", speakers=[1, 2],
        train=[ProbeItem(shared, 1, 0, ""), ProbeItem(Path("/x/2-0-0.flac"), 2, 1, "")],
        test=[ProbeItem(shared, 1, 0, ""), ProbeItem(Path("/x/2-0-1.flac"), 2, 1, "")],
    )
    with pytest.raises(AssertionError, match="both train and test"):
        leaky.verify(frozenset())


def test_verify_catches_encoder_speaker_leak():
    leaky = ProbeSplitResult(
        mode="generalization", speakers=[1, 2],
        train=[ProbeItem(Path("/x/1-0-0.flac"), 1, 0, "")],
        test=[ProbeItem(Path("/x/2-0-0.flac"), 2, 1, "")],
    )
    with pytest.raises(AssertionError, match="encoder's speakers"):
        leaky.verify(frozenset({1}))  # speaker 1 was an encoder speaker


def test_verify_catches_generalization_speaker_leak():
    leaky = ProbeSplitResult(
        mode="generalization", speakers=[1, 2],
        train=[ProbeItem(Path("/x/1-0-0.flac"), 1, 0, "")],
        test=[ProbeItem(Path("/x/1-0-1.flac"), 1, 0, "")],  # same speaker both sides
    )
    with pytest.raises(AssertionError, match="must be disjoint"):
        leaky.verify(frozenset())


_LIBRI = Path("data/raw/librispeech/LibriSpeech")


@pytest.mark.skipif(not (_LIBRI / "dev-clean").is_dir(), reason="LibriSpeech not present")
def test_real_dev_clean_split_disjoint_from_train_clean_100():
    from edbr1.data.librispeech import speaker_utterances

    dev = speaker_utterances(_LIBRI / "dev-clean")
    train_speakers = frozenset(speaker_utterances(_LIBRI / "train-clean-100"))
    r = build_probe_split(
        dev, num_speakers=20, mode="speaker_id",
        exclude_speakers=train_speakers, seed=1337, max_utts_per_speaker=20,
    )
    # Real closed set of 20 dev-clean speakers, none seen by the encoder.
    assert r.num_speakers == 20
    assert set(r.speakers).isdisjoint(train_speakers)
    assert {it.path for it in r.train}.isdisjoint({it.path for it in r.test})
