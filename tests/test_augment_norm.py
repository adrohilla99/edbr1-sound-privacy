"""Augmentation-gating and normalisation-stat tests (skipped without torch)."""
from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("sklearn")

from edbr1.config import AugmentConfig, FeatureConfig, TrainConfig  # noqa: E402
from edbr1.data.urbansound8k import UrbanSound8KDataset  # noqa: E402
from edbr1.train import _make_loader, compute_norm_stats  # noqa: E402
from edbr1.utils import seed_worker  # noqa: E402


def _one_row_metadata() -> pd.DataFrame:
    # __init__ never reads audio, so a single placeholder row is enough to
    # check the augmentation gating flag.
    return pd.DataFrame(
        [{"slice_file_name": "a.wav", "fold": 1, "classID": 0, "class": "x", "path": "a.wav"}]
    )


def test_augmentation_off_for_test_on_for_train():
    meta = _one_row_metadata()
    aug = AugmentConfig(enabled=True)

    train_ds = UrbanSound8KDataset(meta, FeatureConfig(), train=True, augment=aug)
    test_ds = UrbanSound8KDataset(meta, FeatureConfig(), train=False, augment=aug)

    assert train_ds.augment_on is True
    assert test_ds.augment_on is False


def test_augmentation_off_when_config_disabled_or_absent():
    meta = _one_row_metadata()
    # Train dataset but augmentation disabled / not supplied -> still off.
    assert UrbanSound8KDataset(meta, train=True, augment=None).augment_on is False
    disabled = AugmentConfig(enabled=False)
    assert UrbanSound8KDataset(meta, train=True, augment=disabled).augment_on is False


def _fake_loader(n_mels: int, frames: int = 7, batches: int = 3, batch: int = 4):
    return [(torch.randn(batch, 1, n_mels, frames), torch.zeros(batch)) for _ in range(batches)]


def test_per_band_norm_stats_have_shape_n_mels():
    n_mels = 64
    mean, std = compute_norm_stats(_fake_loader(n_mels), per_band=True)
    # Broadcastable (1, 1, n_mels, 1) with exactly n_mels distinct band stats.
    assert mean.shape == (1, 1, n_mels, 1)
    assert std.shape == (1, 1, n_mels, 1)
    assert mean.flatten().shape[0] == n_mels
    assert torch.all(std > 0)


def test_global_norm_stats_are_scalar():
    mean, std = compute_norm_stats(_fake_loader(64), per_band=False)
    assert mean.numel() == 1
    assert std.numel() == 1


def test_seed_worker_is_deterministic_and_seeds_all_rngs():
    import random

    import numpy as np

    # torch.initial_seed() reflects the main-process torch seed, so a fixed
    # torch seed must reproduce the same numpy/random stream after seed_worker.
    torch.manual_seed(4242)
    seed_worker(0)
    first = (np.random.rand(), random.random())
    torch.manual_seed(4242)
    seed_worker(0)
    second = (np.random.rand(), random.random())
    assert first == second
    # A different base seed must change the stream (workers stay distinct).
    torch.manual_seed(9999)
    seed_worker(0)
    assert (np.random.rand(), random.random()) != first


def test_loader_wires_worker_seeding_when_workers_enabled():
    meta = _one_row_metadata()
    ds = UrbanSound8KDataset(meta, FeatureConfig(), train=True)
    device = torch.device("cpu")

    cfg_workers = TrainConfig(num_workers=2)
    loader = _make_loader(ds, shuffle=True, config=cfg_workers, device=device)
    assert loader.worker_init_fn is seed_worker

    # With a single-process loader there are no workers to seed.
    single = _make_loader(ds, shuffle=False, config=cfg_workers, device=device, num_workers=0)
    assert single.worker_init_fn is None
