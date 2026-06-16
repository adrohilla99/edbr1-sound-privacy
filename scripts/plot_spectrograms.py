"""
Sanity-check the log-mel front end on UrbanSound8K **fold 1 only**.

Loads a handful of clips from fold 1, runs them through the configured
LogMelExtractor, and saves a grid of spectrograms so you can eyeball that
the pipeline is wired correctly (sensible dynamic range, time on the x
axis, mel bands on the y axis, one clip per class where possible).

Deliberately restricted to fold 1 so this script never touches the
held-out evaluation folds. Outputs a PNG under results/ (gitignored).

Usage:
    python scripts/plot_spectrograms.py \
        --root data/raw/urbansound8k/UrbanSound8K \
        --num 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from edbr1.config import FeatureConfig, load_feature_config  # noqa: E402
from edbr1.data.urbansound8k import (  # noqa: E402
    URBANSOUND8K_CLASSES,
    UrbanSound8K,
    UrbanSound8KDataset,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "urbansound8k" / "UrbanSound8K",
        help="Extracted UrbanSound8K directory (contains metadata/ and audio/).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "features.yaml",
        help="Feature config YAML (falls back to defaults if missing).",
    )
    parser.add_argument("--num", type=int, default=8, help="How many clips to plot.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "fold1_spectrograms.png",
        help="Output PNG path (results/ is gitignored).",
    )
    args = parser.parse_args()

    config = (
        load_feature_config(args.config)
        if args.config.is_file()
        else FeatureConfig()
    )

    dataset_handle = UrbanSound8K.from_root(args.root)
    fold1 = dataset_handle.fold(1)
    # One example per class where available, then top up to --num.
    examples = fold1.groupby("classID", group_keys=False).head(1)
    if len(examples) < args.num:
        extra = fold1.drop(examples.index).head(args.num - len(examples))
        examples = pd.concat([examples, extra])
    examples = examples.head(args.num).reset_index(drop=True)

    ds = UrbanSound8KDataset(examples, feature_config=config)

    cols = 4
    rows = (len(ds) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= len(ds):
            ax.axis("off")
            continue
        log_mel, label = ds[idx]
        spec = log_mel.squeeze(0).numpy()
        im = ax.imshow(spec, origin="lower", aspect="auto")
        ax.set_title(URBANSOUND8K_CLASSES[label], fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("mel band")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"UrbanSound8K fold 1 log-mel sanity check "
        f"({config.n_mels} mels, {config.sample_rate} Hz)",
        fontsize=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"[done] wrote {len(ds)} spectrograms to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
