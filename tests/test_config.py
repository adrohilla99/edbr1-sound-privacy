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
