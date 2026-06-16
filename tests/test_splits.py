"""Fold-split integrity tests for UrbanSound8K (pandas only, no ML deps)."""
from __future__ import annotations

import pandas as pd
import pytest

from edbr1.data.urbansound8k import NUM_FOLDS, train_test_fold_split


def _synthetic_metadata(per_fold: int = 5) -> pd.DataFrame:
    rows = []
    for fold in range(1, NUM_FOLDS + 1):
        for i in range(per_fold):
            rows.append(
                {
                    "slice_file_name": f"f{fold}_{i}.wav",
                    "fold": fold,
                    "classID": i % 10,
                    "class": "x",
                }
            )
    return pd.DataFrame(rows)


def test_no_fold_overlap_between_train_and_test():
    meta = _synthetic_metadata()
    for test_fold in range(1, NUM_FOLDS + 1):
        train_df, test_df = train_test_fold_split(meta, test_fold)
        train_folds = set(train_df["fold"])
        test_folds = set(test_df["fold"])
        # The leak guard: held-out fold only in test, never in train.
        assert test_folds == {test_fold}
        assert test_fold not in train_folds
        assert train_folds & test_folds == set()


def test_split_is_exhaustive_and_disjoint_by_row():
    meta = _synthetic_metadata()
    train_df, test_df = train_test_fold_split(meta, test_fold=3)
    assert len(train_df) + len(test_df) == len(meta)
    train_files = set(train_df["slice_file_name"])
    test_files = set(test_df["slice_file_name"])
    assert train_files.isdisjoint(test_files)


def test_invalid_fold_rejected():
    meta = _synthetic_metadata()
    with pytest.raises(ValueError, match="test_fold must be"):
        train_test_fold_split(meta, test_fold=0)
    with pytest.raises(ValueError, match="test_fold must be"):
        train_test_fold_split(meta, test_fold=11)
