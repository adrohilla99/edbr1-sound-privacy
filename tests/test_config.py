"""Config loading and derived-parameter tests (no ML deps required)."""
from __future__ import annotations

from pathlib import Path

import pytest

from edbr1.config import (
    FeatureConfig,
    TrainConfig,
    config_to_dict,
    load_feature_config,
    load_train_config,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_feature_config_derived_values():
    cfg = FeatureConfig()  # 16 kHz, 64 mel, 25 ms win, 10 ms hop
    assert cfg.win_length == 400  # 25 ms * 16 kHz
    assert cfg.hop_length == 160  # 10 ms * 16 kHz
    assert cfg.resolved_n_fft == 512  # next power of two >= 400
    assert cfg.resolved_f_max == 8000.0  # Nyquist


def test_load_feature_config_yaml():
    cfg = load_feature_config(CONFIGS / "features.yaml")
    assert cfg.sample_rate == 16_000
    assert cfg.n_mels == 64


def test_load_train_config_yaml_nested_features():
    cfg = load_train_config(CONFIGS / "baseline.yaml")
    assert isinstance(cfg, TrainConfig)
    assert cfg.test_folds == tuple(range(1, 11))
    assert cfg.features.n_mels == 64


def test_config_to_dict_is_yaml_safe():
    import yaml

    cfg = load_train_config(CONFIGS / "baseline.yaml")
    dumped = yaml.safe_dump(config_to_dict(cfg))  # must not raise on tuples
    assert "features" in dumped


def test_unknown_key_rejected():
    from edbr1.config import feature_config_from_dict

    with pytest.raises(ValueError, match="Unknown config keys"):
        feature_config_from_dict({"not_a_real_key": 1})


def test_baseline_defaults_keep_regularisers_off():
    # The plain baseline must remain the reproducible reference: no augment,
    # global norm, no scheduler, no early stopping.
    cfg = load_train_config(CONFIGS / "baseline.yaml")
    assert cfg.norm == "global"
    assert cfg.augment.enabled is False
    assert cfg.schedule.scheduler == "none"
    assert cfg.schedule.early_stopping is False


def test_canonical_config_enables_regularisers_and_round_trips():
    import yaml

    # baseline_final.yaml is the canonical regularised baseline (16 kHz).
    cfg = load_train_config(CONFIGS / "baseline_final.yaml")
    assert cfg.norm == "per_band"
    assert cfg.augment.enabled is True
    assert cfg.augment.spec_augment is True
    assert cfg.schedule.scheduler == "cosine"
    assert cfg.schedule.early_stopping is True
    assert cfg.features.sample_rate == 16_000
    # Nested augment/schedule must survive serialisation for the run log.
    dumped = yaml.safe_dump(config_to_dict(cfg))
    assert "augment" in dumped and "schedule" in dumped


def test_22k_experiment_config_differs_only_by_sample_rate():
    # The rejected-lever evidence config: same recipe, 22.05 kHz.
    canonical = load_train_config(CONFIGS / "baseline_final.yaml")
    exp = load_train_config(CONFIGS / "improved_22k.yaml")
    assert exp.features.sample_rate == 22_050
    assert canonical.features.sample_rate == 16_000
    # Everything else about the feature front end matches (single-lever A/B).
    assert exp.augment == canonical.augment
    assert exp.schedule == canonical.schedule
    assert exp.norm == canonical.norm
    assert exp.features.n_mels == canonical.features.n_mels


def test_invalid_norm_and_scheduler_rejected():
    with pytest.raises(ValueError, match="norm must be"):
        TrainConfig(norm="bogus")
    from edbr1.config import ScheduleConfig

    with pytest.raises(ValueError, match="scheduler must be"):
        ScheduleConfig(scheduler="bogus")
