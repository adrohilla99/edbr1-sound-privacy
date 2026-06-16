"""
UrbanSound8K small-CNN baseline trainer.

Runs the **official 10-fold cross-validation** protocol: for each held-out
test fold, train on the remaining nine and evaluate. Reports macro-F1 per
fold and averaged across folds, writes per-fold confusion matrices and a
machine-readable results JSON, and logs the exact config used.

Reproducibility: a single seed (offset per fold so folds differ but the
whole run is deterministic), the resolved config, and all metrics are
written under a timestamped directory in ``results/`` (gitignored).

Honest-reporting note: this script reports whatever macro-F1 the model
achieves. If it lands below the published ~73-76% band, that is surfaced
in the summary rather than silently tuned away -- the likely culprits to
investigate first are feature normalisation, clip length/cropping, and
making sure fold handling has not leaked.

Usage:
    python -m edbr1.train \
        --root data/raw/urbansound8k/UrbanSound8K \
        --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from edbr1.config import TrainConfig, config_to_dict, load_train_config
from edbr1.data.urbansound8k import (
    URBANSOUND8K_CLASSES,
    UrbanSound8KDataset,
    load_metadata,
    train_test_fold_split,
)
from edbr1.evaluate import classification_metrics, save_confusion_matrix
from edbr1.models import SmallAudioCNN
from edbr1.utils import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_norm_stats(loader: DataLoader[tuple[Tensor, int]]) -> tuple[float, float]:
    """Global mean/std of the log-mel features over a (training) loader.

    Standardising features with statistics estimated on the *training*
    folds only (never the test fold) is important for reaching the
    published baseline and avoids leaking test statistics.
    """
    total = 0.0
    total_sq = 0.0
    count = 0
    for x, _ in tqdm(loader, desc="  norm-stats", leave=False):
        total += float(x.sum())
        total_sq += float((x * x).sum())
        count += x.numel()
    mean = total / count
    var = max(total_sq / count - mean * mean, 1e-12)
    return mean, var**0.5


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    device: torch.device,
    mean: float,
    std: float,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, list[int], list[int]]:
    """One pass. Train if ``optimizer`` is given, else evaluate. Returns
    (mean_loss, y_true, y_pred)."""
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = (x.to(device) - mean) / std
            y = y.to(device)
            if training:
                assert optimizer is not None
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                assert optimizer is not None
                optimizer.step()
            running_loss += float(loss.detach()) * y.size(0)
            seen += y.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    return running_loss / max(seen, 1), y_true, y_pred


def train_one_fold(
    metadata: object,
    test_fold: int,
    config: TrainConfig,
    device: torch.device,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Train on all folds except ``test_fold`` and evaluate on it."""
    import pandas as pd

    assert isinstance(metadata, pd.DataFrame)
    seed_everything(config.seed + test_fold)

    train_df, test_df = train_test_fold_split(metadata, test_fold)
    train_ds = UrbanSound8KDataset(train_df, config.features, config.clip_seconds)
    test_ds = UrbanSound8KDataset(test_df, config.features, config.clip_seconds)

    train_loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    test_loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    mean, std = compute_norm_stats(train_loader)

    model = SmallAudioCNN(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(1, config.epochs + 1):
        loss, _, _ = _run_epoch(
            model, train_loader, device, mean, std, optimizer=optimizer
        )
        print(f"    fold {test_fold} epoch {epoch:>3}/{config.epochs}  loss={loss:.4f}")

    _, y_true, y_pred = _run_epoch(
        model, test_loader, device, mean, std, optimizer=None
    )
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["test_fold"] = test_fold
    metrics["norm_mean"] = mean
    metrics["norm_std"] = std
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
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
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
        help="Training config YAML.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Root for run outputs (gitignored).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument(
        "--test-folds",
        type=int,
        nargs="+",
        default=None,
        help="Subset of folds to evaluate (default: all from config).",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    args = parser.parse_args(argv)

    config = load_train_config(args.config)
    if args.epochs is not None:
        config = dataclasses.replace(config, epochs=args.epochs)
    test_folds = tuple(args.test_folds) if args.test_folds else config.test_folds

    device = _pick_device(args.device)
    metadata = load_metadata(args.root)
    class_names = URBANSOUND8K_CLASSES

    run_dir = args.results_dir / time.strftime("us8k_baseline_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(config), fh, sort_keys=False)

    print(f"Device: {device}")
    print(f"Model params: {SmallAudioCNN(num_classes=len(class_names)).num_parameters():,}")
    print(f"Evaluating folds: {test_folds}")

    fold_metrics: list[dict[str, Any]] = []
    for test_fold in test_folds:
        print(f"\n=== Fold {test_fold} ===")
        metrics = train_one_fold(metadata, test_fold, config, device, class_names)
        fold_metrics.append(metrics)
        print(f"  fold {test_fold} macro-F1 = {metrics['macro_f1']:.4f}")
        save_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            run_dir / f"confusion_fold{test_fold}.png",
            title=f"UrbanSound8K fold {test_fold}",
        )

    macro_f1s = [float(m["macro_f1"]) for m in fold_metrics]
    mean_f1 = sum(macro_f1s) / len(macro_f1s)
    std_f1 = (sum((f - mean_f1) ** 2 for f in macro_f1s) / len(macro_f1s)) ** 0.5

    summary = {
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "per_fold_macro_f1": {
            int(m["test_fold"]): float(m["macro_f1"]) for m in fold_metrics
        },
        "folds": fold_metrics,
        "config": config_to_dict(config),
    }
    with (run_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"Mean macro-F1 over {len(macro_f1s)} fold(s): {mean_f1:.4f} (+/- {std_f1:.4f})")
    print("Published small-CNN reference band: ~0.73-0.76 macro-F1")
    if mean_f1 < 0.73:
        print(
            "NOTE: below the published band. Before tuning, check: feature "
            "normalisation, clip length/cropping, and fold-split integrity."
        )
    print(f"Results written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
