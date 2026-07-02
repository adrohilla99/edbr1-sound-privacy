"""Bitrate accounting tests (pure Python, no ML deps required)."""
from __future__ import annotations

import math

import pytest

from edbr1.bitrate import (
    OperatingPoint,
    bits_per_second,
    bits_per_token,
    tokens_per_second,
)


def test_tokens_per_second_is_grid_over_clip():
    # 8 tokens/frame-grid * ... : a 4x50 grid over a 4 s clip -> 200 tokens / 4 s.
    assert tokens_per_second(4 * 50, 4.0) == 50.0
    assert tokens_per_second(32, 4.0) == 8.0


def test_bits_per_token_is_log2_codebook():
    assert bits_per_token(1024) == 10.0
    assert bits_per_token(256) == 8.0
    # A degenerate one-code codebook carries zero information.
    assert bits_per_token(1) == 0.0


def test_bits_per_second_matches_declared_formula():
    # bits_per_second = tokens_per_second * log2(codebook_size)
    tps = tokens_per_second(1 * 32, 4.0)  # 8 tokens/s
    assert bits_per_second(tps, 1024) == pytest.approx(80.0)
    tps_hi = tokens_per_second(32 * 200, 4.0)  # 1600 tokens/s
    assert bits_per_second(tps_hi, 1024) == pytest.approx(16000.0)


def test_operating_point_spans_target_range():
    # The six swept operating points must span ~100 bits/s to ~16 kbits/s.
    points = [
        (1, 32, 1024, 80.0),
        (2, 50, 1024, 250.0),
        (4, 100, 1024, 1000.0),
        (8, 100, 1024, 2000.0),
        (8, 200, 1024, 4000.0),
        (32, 200, 1024, 16000.0),
    ]
    for f, t, k, expected_bps in points:
        op = OperatingPoint(latent_freq=f, latent_frames=t, codebook_size=k, clip_seconds=4.0)
        assert op.tokens_per_clip == f * t
        assert op.bits_per_second == pytest.approx(expected_bps)
        # Round-trip through the flat summary used for the sweep CSV/JSON.
        d = op.as_dict()
        assert d["bits_per_second"] == pytest.approx(expected_bps)
        assert d["bits_per_token"] == pytest.approx(math.log2(k))


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        tokens_per_second(-1, 4.0)
    with pytest.raises(ValueError):
        tokens_per_second(10, 0.0)
    with pytest.raises(ValueError):
        bits_per_token(0)
    with pytest.raises(ValueError):
        bits_per_second(-1.0, 256)
