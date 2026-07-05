"""Probe-metric correctness on toy inputs (no torch needed for the text metrics)."""
from __future__ import annotations

import pytest

from edbr1.probes.metrics import (
    character_error_rate,
    edit_distance,
    word_error_rate,
)


def test_edit_distance_known_cases():
    assert edit_distance("kitten", "sitting") == 3  # classic Levenshtein
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("", "abc") == 3
    assert edit_distance(["a", "b"], ["a", "c"]) == 1  # token sequences too


def test_word_error_rate():
    # 1 substitution out of 4 reference words -> 0.25.
    assert word_error_rate(["the cat sat down"], ["the dog sat down"]) == pytest.approx(0.25)
    assert word_error_rate(["a b c"], ["a b c"]) == 0.0
    # Total-word normalisation across the corpus.
    wer = word_error_rate(["one two", "three"], ["one two", "four"])
    assert wer == pytest.approx(1 / 3)


def test_character_error_rate():
    assert character_error_rate(["abcd"], ["abxd"]) == pytest.approx(0.25)
    assert character_error_rate(["hello"], ["hello"]) == 0.0


def test_log_spectral_distance_and_top1():
    torch = pytest.importorskip("torch")
    from edbr1.probes.metrics import log_spectral_distance, top1_accuracy

    a = torch.randn(2, 1, 64, 40)
    assert log_spectral_distance(a, a) == pytest.approx(0.0, abs=1e-6)  # identical -> 0
    b = a + 3.0  # constant 3 dB offset everywhere -> LSD == 3
    assert log_spectral_distance(a, b) == pytest.approx(3.0, abs=1e-5)

    preds = torch.tensor([1, 2, 3, 4])
    assert top1_accuracy(preds, torch.tensor([1, 2, 0, 0])) == pytest.approx(0.5)
