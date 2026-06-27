"""
UrbanSound8K dataset loader.

UrbanSound8K ships with an **official 10-fold cross-validation split** in
its metadata CSV (the ``fold`` column). That split is curated so that
slices from the same source recording never straddle a fold boundary;
re-rolling our own random split would leak near-duplicate clips between
train and test and inflate scores. So this module **only** ever partitions
by the ``fold`` column and includes an explicit leak guard.

Metadata CSV columns (UrbanSound8K.csv):
    slice_file_name, fsID, start, end, salience, fold, classID, class
Audio lives at:  <root>/audio/fold<fold>/<slice_file_name>
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from edbr1.config import AugmentConfig, FeatureConfig
from edbr1.data.augment import augment_waveform, spec_augment
from edbr1.features.melspec import LogMelExtractor

# Class names indexed by classID (0..9), per the official dataset taxonomy.
URBANSOUND8K_CLASSES: tuple[str, ...] = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)
NUM_FOLDS = 10
REQUIRED_COLUMNS = {"slice_file_name", "fold", "classID", "class"}


def load_metadata(root: str | Path) -> pd.DataFrame:
    """Load and validate the UrbanSound8K metadata CSV.

    ``root`` is the extracted ``UrbanSound8K/`` directory (the one that
    contains ``metadata/`` and ``audio/``). Returns the dataframe with an
    added absolute ``path`` column.
    """
    root = Path(root)
    csv_path = root / "metadata" / "UrbanSound8K.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"UrbanSound8K metadata not found at {csv_path}. Run "
            "scripts/download_urbansound8k.py first."
        )
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing expected columns: {sorted(missing)}")

    folds = set(df["fold"].unique())
    if not folds <= set(range(1, NUM_FOLDS + 1)):
        raise ValueError(f"Unexpected fold values in metadata: {sorted(folds)}")

    df = df.copy()
    df["path"] = df.apply(
        lambda r: str(root / "audio" / f"fold{r['fold']}" / r["slice_file_name"]),
        axis=1,
    )
    return df


def train_test_fold_split(
    metadata: pd.DataFrame, test_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``metadata`` into (train, test) by the official ``fold`` column.

    ``test_fold`` is the held-out fold (1..10); every other fold is training.
    Includes a leak guard: it asserts the two partitions share no rows and
    that no fold appears on both sides.
    """
    if test_fold not in range(1, NUM_FOLDS + 1):
        raise ValueError(f"test_fold must be in 1..{NUM_FOLDS}, got {test_fold}")

    test_df = metadata[metadata["fold"] == test_fold]
    train_df = metadata[metadata["fold"] != test_fold]

    # Leak guard -- never let a fold leak between train and test.
    train_folds = set(train_df["fold"].unique())
    test_folds = set(test_df["fold"].unique())
    overlap = train_folds & test_folds
    if overlap:
        raise AssertionError(f"Fold leak detected between train and test: {overlap}")
    if test_folds != {test_fold}:
        raise AssertionError(f"Test partition contains unexpected folds: {test_folds}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def carve_validation_fold(
    train_df: pd.DataFrame, val_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set out of an already train-only frame, by ``fold``.

    Used for early stopping: ``train_df`` must already exclude the held-out
    test fold (it is the output of :func:`train_test_fold_split`). One of its
    remaining folds becomes the validation set; the rest stay for training.

    Includes a leak guard mirroring :func:`train_test_fold_split`: the two
    partitions share no fold, and the validation partition holds exactly
    ``val_fold``. Estimating normalisation and stopping on this fold therefore
    never touches the test fold.
    """
    folds = set(train_df["fold"].unique())
    if val_fold not in folds:
        raise ValueError(
            f"val_fold {val_fold} is not among the training folds {sorted(folds)} "
            "(it must not be the held-out test fold)"
        )

    val_df = train_df[train_df["fold"] == val_fold]
    inner_df = train_df[train_df["fold"] != val_fold]

    inner_folds = set(inner_df["fold"].unique())
    if val_fold in inner_folds:
        raise AssertionError(f"Validation fold {val_fold} leaked into inner-train")

    return inner_df.reset_index(drop=True), val_df.reset_index(drop=True)


@dataclass
class UrbanSound8K:
    """Handle to an extracted UrbanSound8K tree plus its metadata."""

    root: Path
    metadata: pd.DataFrame

    @classmethod
    def from_root(cls, root: str | Path) -> UrbanSound8K:
        root = Path(root)
        return cls(root=root, metadata=load_metadata(root))

    def fold(self, fold: int) -> pd.DataFrame:
        """Return only the rows for a single fold (1..10)."""
        if fold not in range(1, NUM_FOLDS + 1):
            raise ValueError(f"fold must be in 1..{NUM_FOLDS}, got {fold}")
        return self.metadata[self.metadata["fold"] == fold].reset_index(drop=True)


class UrbanSound8KDataset(Dataset[tuple[Tensor, int]]):
    """Torch dataset yielding ``(log_mel, label)`` pairs.

    Each item is a ``(1, n_mels, n_frames)`` float tensor (single channel
    for the CNN) and an integer ``classID``. Waveforms are downmixed to
    mono, resampled to the configured rate, and fixed to
    ``clip_seconds`` (pad short, centre-independent crop long) so a batch
    has uniform shape.

    Augmentation is gated by ``train``: it is applied only when the dataset is
    built with ``train=True`` *and* an enabled :class:`AugmentConfig`. The
    held-out test fold must always be built with ``train=False`` so its clips
    are never augmented.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        feature_config: FeatureConfig | None = None,
        clip_seconds: float = 4.0,
        *,
        train: bool = False,
        augment: AugmentConfig | None = None,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.config = feature_config or FeatureConfig()
        self.extractor = LogMelExtractor(self.config)
        self.target_len = int(round(clip_seconds * self.config.sample_rate))
        self.train = train
        self.augment = augment
        # Augmentation is on only for training data with an enabled config.
        self.augment_on = bool(train and augment is not None and augment.enabled)

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_waveform(self, path: str) -> tuple[Tensor, int]:
        """Load an audio file as a (channels, samples) float32 tensor + sr."""
        import soundfile as sf  # lazy: only needed when actually reading audio

        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
        wav = torch.from_numpy(data).transpose(0, 1).contiguous()  # (channels, samples)
        return wav, int(sr)

    def _fix_length(self, wav: Tensor) -> Tensor:
        """Pad (with zeros) or crop a mono waveform to ``target_len`` samples."""
        n = wav.shape[-1]
        if n < self.target_len:
            return torch.nn.functional.pad(wav, (0, self.target_len - n))
        return wav[..., : self.target_len]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.metadata.iloc[index]
        wav, sr = self._load_waveform(row["path"])

        # Downmix + resample at the waveform level so the fixed-length crop
        # is applied at the target rate (uniform frame count across a batch).
        import torchaudio.functional as AF  # lazy import

        mono = wav.mean(dim=0)
        if sr != self.config.sample_rate:
            mono = AF.resample(mono, sr, self.config.sample_rate)

        # Waveform-level augmentation runs before the length fix so a
        # length-changing time stretch is re-normalised to target_len.
        if self.augment_on:
            assert self.augment is not None
            mono = augment_waveform(mono, self.config.sample_rate, self.augment)

        mono = self._fix_length(mono)

        log_mel = self.extractor(mono, self.config.sample_rate)  # (n_mels, frames)
        if self.augment_on:
            assert self.augment is not None
            log_mel = spec_augment(log_mel, self.augment)
        return log_mel.unsqueeze(0), int(row["classID"])
