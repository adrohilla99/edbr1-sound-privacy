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

    # 16 kHz is the default. Published UrbanSound8K baselines often use
    # 22_050 Hz, which preserves more high-frequency detail (helpful for
    # classes like siren and car_horn) at the cost of ~1.4x more compute per
    # clip. Set sample_rate: 22050 in the YAML config to try it.
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
class AugmentConfig:
    """Training-time data augmentation parameters.

    Augmentation is applied to the **training folds only** (never the held-out
    test fold). ``enabled`` is the master switch and defaults to ``False`` so
    the plain baseline stays reproducible; the improved config turns it on.

    SpecAugment (Park et al., 2019) masks bands of the log-mel spectrogram and
    is the cheapest, highest-value addition. The waveform-level transforms are
    lighter: a random time shift is fast (pure torch); pitch shift and time
    stretch are slower (they use librosa) and are off by default.
    """

    enabled: bool = False

    # --- SpecAugment (applied on the (n_mels, n_frames) log-mel) ---
    spec_augment: bool = True
    time_masks: int = 2          # number of time masks per clip
    time_mask_param: int = 24    # max width of each time mask (frames)
    freq_masks: int = 2          # number of frequency masks per clip
    freq_mask_param: int = 8     # max width of each freq mask (mel bands)
    spec_augment_prob: float = 1.0  # prob. of applying SpecAugment to a clip

    # --- waveform-level augmentation (applied before the length fix) ---
    time_shift: bool = True
    time_shift_ms: float = 100.0    # max absolute random shift (zero-filled)
    pitch_shift: bool = False       # mild pitch shift; slower (librosa)
    pitch_shift_steps: float = 2.0  # +/- semitone range
    time_stretch: bool = False      # mild time stretch; slower (librosa)
    time_stretch_min: float = 0.9
    time_stretch_max: float = 1.1
    waveform_prob: float = 0.5      # prob. of applying each waveform transform


@dataclass(frozen=True)
class ScheduleConfig:
    """Learning-rate scheduling and early-stopping parameters.

    When ``early_stopping`` is on (or a plateau scheduler is used) one of the
    training folds is held out as a validation set -- carved from the training
    folds only, never the test fold -- and the best-by-validation checkpoint is
    restored before the held-out test fold is scored. All defaults are off so
    the plain baseline trains for a fixed number of epochs as before.
    """

    scheduler: str = "none"     # "none" | "cosine" | "plateau"
    early_stopping: bool = False
    patience: int = 10          # epochs of no val-F1 improvement before stop
    min_delta: float = 0.0      # minimum val-F1 gain counted as improvement
    val_fold: int | None = None  # training fold held out for validation;
    #                              None -> highest-numbered training fold
    plateau_factor: float = 0.5  # ReduceLROnPlateau: LR multiplier on plateau
    plateau_patience: int = 5    # ReduceLROnPlateau: epochs before reducing

    def __post_init__(self) -> None:
        if self.scheduler not in ("none", "cosine", "plateau"):
            raise ValueError(
                f"scheduler must be 'none', 'cosine' or 'plateau', got {self.scheduler!r}"
            )


@dataclass(frozen=True)
class EncoderConfig:
    """On-device encoder E: a depthwise-separable conv stack -> latent grid.

    The encoder maps a ``(1, n_mels, n_frames)`` log-mel spectrogram to a latent
    grid of shape ``(latent_dim, latent_freq, latent_frames)``. That grid is the
    sequence of tokens the bottleneck quantises: there are
    ``latent_freq * latent_frames`` token positions per clip, so together with
    ``TrainConfig.clip_seconds`` they fix the token rate (see
    :mod:`edbr1.bitrate`).

    Downsampling to the target grid is done with real strided pooling inside the
    conv trunk (the number of freq/time halvings is derived from the target grid
    so the pre-pool map is never smaller than the target), followed by an exact
    adaptive average pool. The token count therefore corresponds to genuine
    encoder outputs -- it is never inflated by upsampling.

    Defaults are sized to reproduce the canonical baseline when the bottleneck is
    disabled: a MobileNet-style depthwise-separable trunk well under 500K
    parameters. ``channels`` ramp from ``base_channels`` towards ``latent_dim``
    over however many blocks the downsampling schedule needs (at least
    ``min_depth`` for capacity).
    """

    base_channels: int = 32
    latent_dim: int = 128
    # Latent token grid. latent_freq * latent_frames token positions per clip.
    latent_freq: int = 8
    latent_frames: int = 50
    # Minimum number of depthwise-separable blocks (capacity floor). More blocks
    # are added automatically if the target grid needs more downsampling stages.
    min_depth: int = 3
    # Single dropout before the linear classifier head (mirrors SmallAudioCNN).
    dropout: float = 0.3

    def __post_init__(self) -> None:
        for name, value in (
            ("base_channels", self.base_channels),
            ("latent_dim", self.latent_dim),
            ("latent_freq", self.latent_freq),
            ("latent_frames", self.latent_frames),
            ("min_depth", self.min_depth),
        ):
            if value < 1:
                raise ValueError(f"EncoderConfig.{name} must be >= 1, got {value}")

    def tokens_per_clip(self) -> int:
        """Number of latent token positions emitted per clip."""
        return self.latent_freq * self.latent_frames


@dataclass(frozen=True)
class BottleneckConfig:
    """Discrete (VQ-VAE) bottleneck B inserted between encoder and classifier.

    ``type='none'`` is the control: the continuous latent passes straight through
    (no quantisation, no auxiliary loss), reproducing an ordinary encoder ->
    classifier network. ``type='vq'`` enables a van den Oord et al. (2017) vector
    quantiser: each latent token is snapped to its nearest of ``codebook_size``
    entries, a straight-through estimator carries gradients back to the encoder,
    and a codebook + ``commitment_beta``-weighted commitment loss trains the
    codebook (Reference: "Neural Discrete Representation Learning", VQ-VAE).

    The bitrate of a ``vq`` operating point is
    ``tokens_per_second * log2(codebook_size)`` (see :mod:`edbr1.bitrate`); the
    codebook vector dimension is :attr:`EncoderConfig.latent_dim`.

    ``ema`` switches the codebook from loss-based updates to exponential-moving-
    average updates (van den Oord et al., Appendix A). EMA is more collapse-
    resistant; it is off by default because plain loss-based VQ collapse at low
    bitrate is itself a finding we want to observe honestly.
    """

    type: str = "none"  # "none" | "vq"
    codebook_size: int = 512
    commitment_beta: float = 0.25
    ema: bool = False
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.type not in ("none", "vq"):
            raise ValueError(f"bottleneck type must be 'none' or 'vq', got {self.type!r}")
        if self.codebook_size < 1:
            raise ValueError(f"codebook_size must be >= 1, got {self.codebook_size}")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")


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
    # Feature normalisation: "global" (single scalar mean/std, the original
    # baseline) or "per_band" (a mean/std per mel band). Either way the stats
    # are estimated on the TRAINING folds only -- never the test fold.
    norm: str = "global"
    # Which model to build: "cnn" is the original SmallAudioCNN (the legacy
    # baseline path, left byte-for-byte unchanged); "encoder_classifier" is the
    # refactored encoder -> bottleneck -> classifier used for the bitrate sweep.
    model: str = "cnn"
    features: FeatureConfig = field(default_factory=FeatureConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)

    def __post_init__(self) -> None:
        if self.norm not in ("global", "per_band"):
            raise ValueError(f"norm must be 'global' or 'per_band', got {self.norm!r}")
        if self.model not in ("cnn", "encoder_classifier"):
            raise ValueError(
                f"model must be 'cnn' or 'encoder_classifier', got {self.model!r}"
            )


def _filter_known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that correspond to fields of dataclass ``cls``."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return {k: v for k, v in data.items() if k in known}


def feature_config_from_dict(data: dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(**_filter_known(FeatureConfig, data))


def augment_config_from_dict(data: dict[str, Any]) -> AugmentConfig:
    return AugmentConfig(**_filter_known(AugmentConfig, data))


def schedule_config_from_dict(data: dict[str, Any]) -> ScheduleConfig:
    return ScheduleConfig(**_filter_known(ScheduleConfig, data))


def encoder_config_from_dict(data: dict[str, Any]) -> EncoderConfig:
    return EncoderConfig(**_filter_known(EncoderConfig, data))


def bottleneck_config_from_dict(data: dict[str, Any]) -> BottleneckConfig:
    return BottleneckConfig(**_filter_known(BottleneckConfig, data))


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    data = dict(data)
    feats = data.pop("features", {}) or {}
    aug = data.pop("augment", {}) or {}
    sched = data.pop("schedule", {}) or {}
    enc = data.pop("encoder", {}) or {}
    bottleneck = data.pop("bottleneck", {}) or {}
    if "test_folds" in data and data["test_folds"] is not None:
        data["test_folds"] = tuple(data["test_folds"])
    kwargs = _filter_known(TrainConfig, data)
    return TrainConfig(
        features=feature_config_from_dict(feats),
        augment=augment_config_from_dict(aug),
        schedule=schedule_config_from_dict(sched),
        encoder=encoder_config_from_dict(enc),
        bottleneck=bottleneck_config_from_dict(bottleneck),
        **kwargs,
    )


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
