"""ESC-50 dataset loader (cross-dataset generalisation test, Phase 4b).

ESC-50: 2000 environmental-sound clips, 50 classes, an **official 5-fold** split
in the metadata CSV (the ``fold`` column). Like UrbanSound8K we only ever
partition by that column, with an explicit leak guard -- the folds are curated so
clips from the same source recording never straddle a boundary.

Metadata (``meta/esc50.csv``): ``filename, fold, target, category, esc10,
src_file, take``. Audio lives at ``<root>/audio/<filename>`` (5 s, 44.1 kHz).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from edbr1.config import FeatureConfig
from edbr1.features.melspec import LogMelExtractor

ESC50_NUM_FOLDS = 5
ESC50_NUM_CLASSES = 50
_REQUIRED = {"filename", "fold", "target", "category"}


def load_esc50_metadata(root: str | Path) -> pd.DataFrame:
    """Load + validate the ESC-50 metadata CSV, adding an absolute ``path`` column.

    ``root`` is the extracted ``ESC-50-master/`` directory (contains ``meta/`` and
    ``audio/``).
    """
    root = Path(root)
    csv_path = root / "meta" / "esc50.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"ESC-50 metadata not found at {csv_path}. Run scripts/download_esc50.py."
        )
    df = pd.read_csv(csv_path)
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"ESC-50 metadata missing columns: {sorted(missing)}")
    df["path"] = df["filename"].map(lambda f: str(root / "audio" / f))
    return df


def esc50_fold_split(df: pd.DataFrame, test_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by the official ESC-50 fold with an explicit leak guard (1..5)."""
    if test_fold not in range(1, ESC50_NUM_FOLDS + 1):
        raise ValueError(f"test_fold must be in 1..{ESC50_NUM_FOLDS}, got {test_fold}")
    train_df = df[df["fold"] != test_fold].reset_index(drop=True)
    test_df = df[df["fold"] == test_fold].reset_index(drop=True)
    if test_fold in set(train_df["fold"].unique()):
        raise AssertionError(f"LEAK: test fold {test_fold} present in the training split")
    return train_df, test_df


class ESC50Dataset(Dataset[tuple[Tensor, int]]):
    """Yields ``(log_mel, target)`` for ESC-50 clips at the model's feature config.

    Clips are down-mixed to mono, resampled to the configured rate and fixed to
    ``clip_seconds`` -- so the mels are drop-in for a UrbanSound8K-trained encoder.
    No augmentation (this is a frozen-encoder transfer eval).
    """

    def __init__(
        self, metadata: pd.DataFrame, feature_config: FeatureConfig | None = None,
        clip_seconds: float = 4.0,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.config = feature_config or FeatureConfig()
        self.extractor = LogMelExtractor(self.config)
        self.target_len = int(round(clip_seconds * self.config.sample_rate))

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        import soundfile as sf
        import torchaudio.functional as af

        row = self.metadata.iloc[index]
        data, sr = sf.read(row["path"], dtype="float32", always_2d=True)
        mono = torch.from_numpy(data).mean(dim=1)
        if sr != self.config.sample_rate:
            mono = af.resample(mono, sr, self.config.sample_rate)
        n = mono.shape[-1]
        mono = (mono[: self.target_len] if n >= self.target_len
                else torch.nn.functional.pad(mono, (0, self.target_len - n)))
        mel = self.extractor(mono, self.config.sample_rate)
        return mel.unsqueeze(0), int(row["target"])
