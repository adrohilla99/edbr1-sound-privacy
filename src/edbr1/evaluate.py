"""
Evaluation metrics for the UrbanSound8K baseline.

Reports macro-F1 (the headline number for UrbanSound8K's class-imbalanced
10-fold protocol), per-class F1, and a confusion matrix. Macro-F1 is used
rather than plain accuracy because the classes are unevenly sized and the
published baselines report macro-averaged scores.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute macro-F1, per-class F1 and the confusion matrix.

    Labels are evaluated over the full label set ``range(len(class_names))``
    so that a class absent from a particular fold still appears (with F1 0).
    """
    labels = list(range(len(class_names)))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": macro_f1,
        "per_class_f1": {
            name: float(v) for name, v in zip(class_names, per_class, strict=True)
        },
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_true)),
    }


def save_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: Sequence[str],
    out_path: str | Path,
    *,
    title: str = "Confusion matrix",
    normalize: bool = True,
) -> Path:
    """Render a confusion matrix heatmap to ``out_path`` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    mat = np.asarray(cm, dtype=float)
    fmt = "d"
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
        fmt = ".2f"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        mat,
        annot=True,
        fmt=fmt,
        cmap="viridis",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
