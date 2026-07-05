"""ESC-50 loader + fold-disjointness leak-guard tests."""
from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from edbr1.data.esc50 import esc50_fold_split, load_esc50_metadata  # noqa: E402


def _synth() -> pd.DataFrame:
    rows = [{"filename": f"{f}-{i}.wav", "fold": f, "target": i, "category": "x", "path": "/x"}
            for f in range(1, 6) for i in range(4)]
    return pd.DataFrame(rows)


def test_esc50_fold_split_is_disjoint_and_partitions():
    df = _synth()
    for test_fold in range(1, 6):
        train, test = esc50_fold_split(df, test_fold)
        assert set(train["fold"]).isdisjoint({test_fold})   # no leak
        assert set(test["fold"]) == {test_fold}
        assert len(train) + len(test) == len(df)            # partition


def test_esc50_fold_split_rejects_out_of_range_fold():
    with pytest.raises(ValueError, match="test_fold must be"):
        esc50_fold_split(_synth(), 6)


_ESC50 = Path("data/raw/esc50/ESC-50-master")


@pytest.mark.skipif(not (_ESC50 / "meta" / "esc50.csv").is_file(), reason="ESC-50 not present")
def test_real_esc50_metadata_shape():
    df = load_esc50_metadata(_ESC50)
    assert len(df) == 2000
    assert df["target"].nunique() == 50
    assert set(df["fold"].unique()) == {1, 2, 3, 4, 5}
    assert df["path"].iloc[0].endswith(".wav")
