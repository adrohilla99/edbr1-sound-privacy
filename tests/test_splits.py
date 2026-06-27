"""Fold-split integrity tests for UrbanSound8K (pandas only, no ML deps)."""
from __future__ import annotations

import pandas as pd
import pytest

from edbr1.data.urbansound8k import (
    NUM_FOLDS,
    carve_validation_fold,
    train_test_fold_split,
)


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


def test_carve_validation_fold_is_leak_free():
    meta = _synthetic_metadata()
    train_df, test_df = train_test_fold_split(meta, test_fold=1)
    inner_df, val_df = carve_validation_fold(train_df, val_fold=10)

    inner_folds = set(inner_df["fold"])
    val_folds = set(val_df["fold"])
    # Validation holds exactly its fold; the test fold is in neither side.
    assert val_folds == {10}
    assert 10 not in inner_folds
    assert 1 not in inner_folds and 1 not in val_folds
    assert inner_folds & val_folds == set()
    # The carve is exhaustive over the training frame.
    assert len(inner_df) + len(val_df) == len(train_df)


def test_carve_validation_rejects_non_training_fold():
    meta = _synthetic_metadata()
    train_df, _ = train_test_fold_split(meta, test_fold=1)
    # Fold 1 is the held-out test fold -- it must not be usable as validation.
    with pytest.raises(ValueError, match="not among the training folds"):
        carve_validation_fold(train_df, val_fold=1)
