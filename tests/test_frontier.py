"""Frontier-assembly correctness: the committed figure data pulls the right numbers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
pytest.importorskip("matplotlib")

from make_figures import _leakage_points  # noqa: E402

_DATA = Path("docs/figures/sweep_data.json")


def test_leakage_points_flatten_the_right_fields():
    rec = [{
        "bits_per_second": 1000.0, "grl_lambda": 0.0, "utility_macro_f1": 0.8,
        "speaker_id": {"top1": 0.11, "chance": 0.05}, "asr": {"wer": 1.1},
        "inverter": {"lsd": 16.0, "lsd_silence_floor": 75.0},
    }]
    pts = _leakage_points(rec)
    assert pts[0]["speaker_top1"] == 0.11
    assert pts[0]["asr_wer"] == 1.1
    assert pts[0]["inverter_lsd"] == 16.0


@pytest.mark.skipif(not _DATA.is_file(), reason="committed frontier data absent")
def test_committed_frontier_data_has_all_phases():
    data = json.loads(_DATA.read_text(encoding="utf-8"))
    for key in ("anticollapse", "lambda_sweep", "leakage", "robustness", "esc50"):
        assert key in data, f"frontier data missing {key}"
    # Correctness sanity: the right numbers were pulled -- at 1000 b/s the
    # adversarial (lambda=2) speaker leakage is below the non-adversarial one.
    knee = {(p["bits_per_second"], p["grl_lambda"]): p for p in data["leakage"]}
    assert knee[(1000.0, 2.0)]["speaker_top1"] < knee[(1000.0, 0.0)]["speaker_top1"]
