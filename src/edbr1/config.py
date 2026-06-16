"""
Configuration objects for the EDBR.1 feature front end and baseline trainer.

Parameters live here as typed dataclasses with sensible defaults, and can
be overridden from a small YAML file. Nothing in the feature/model code
should hard-code a window length or mel-band count -- it should read from
these objects so every run is described by one serialisable config.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatureConfig:
    """Log-mel spectrogram front-end parameters.

    Defaults follow common UrbanSound8K small-CNN recipes: 16 kHz mono,
    64 mel bands, a 25 ms analysis window and a 10 ms hop.
    """

    sample_rate: int = 16_000
    n_mels: int = 64
    window_ms: float = 25.0
    hop_ms: float = 10.0
    f_min: float = 0.0
    f_max: float | None = None  # None -> Nyquist (sample_rate / 2)
    n_fft: int | None = None  # None -> next power of two >= win_length
    power: float = 2.0  # 2.0 -> power spectrogram, then converted to dB
    top_db: float = 80.0  # dynamic-range floor for the dB conversion

    @property
    def win_length(self) -> int:
        """Analysis window length in samples."""
        return int(round(self.sample_rate * self.window_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        """Hop length in samples."""
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def resolved_n_fft(self) -> int:
        """FFT size, defaulting to the next power of two >= win_length."""
        if self.n_fft is not None:
            return self.n_fft
        n = 1
        while n < self.win_length:
            n <<= 1
        return n

    @property
    def resolved_f_max(self) -> float:
        return self.f_max if self.f_max is not None else self.sample_rate / 2.0


@dataclass(frozen=True)
class TrainConfig:
    """Baseline training hyper-parameters and bookkeeping."""

    seed: int = 1337
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    # 10-fold CV: which folds to evaluate as the held-out test fold. The
    # default runs the full official protocol (folds 1..10).
    test_folds: tuple[int, ...] = tuple(range(1, 11))
    # Per-sample fixed clip length in seconds for batching (UrbanSound8K
    # clips are <= 4 s); shorter clips are zero-padded, longer ones cropped.
    clip_seconds: float = 4.0
    features: FeatureConfig = field(default_factory=FeatureConfig)


def _filter_known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that correspond to fields of dataclass ``cls``."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return {k: v for k, v in data.items() if k in known}


def feature_config_from_dict(data: dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(**_filter_known(FeatureConfig, data))


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    data = dict(data)
    feats = data.pop("features", {})
    if "test_folds" in data and data["test_folds"] is not None:
        data["test_folds"] = tuple(data["test_folds"])
    kwargs = _filter_known(TrainConfig, data)
    return TrainConfig(features=feature_config_from_dict(feats), **kwargs)


def load_train_config(path: str | Path) -> TrainConfig:
    """Load a :class:`TrainConfig` (with nested features) from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root in {path} must be a mapping, got {type(data)}")
    return train_config_from_dict(data)


def load_feature_config(path: str | Path) -> FeatureConfig:
    """Load a standalone :class:`FeatureConfig` from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root in {path} must be a mapping, got {type(data)}")
    # Allow either a bare feature mapping or one nested under "features".
    if "features" in data and isinstance(data["features"], dict):
        data = data["features"]
    return feature_config_from_dict(data)


def _yaml_safe(value: Any) -> Any:
    """Recursively convert tuples to lists so PyYAML's safe dumper accepts them."""
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return value


def config_to_dict(config: FeatureConfig | TrainConfig) -> dict[str, Any]:
    """Serialise a config (and any nested config) to a YAML-safe plain dict."""
    return _yaml_safe(dataclasses.asdict(config))
