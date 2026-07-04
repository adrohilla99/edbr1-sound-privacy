"""Adversarial GRL-lambda sweep at the utility knee (Phase 3, Task 3).

Fixes the bitrate at the Phase-2 knee (1000 bits/s, anti-collapse codebook) and
sweeps the gradient-reversal strength ``adversary.grl_lambda`` over a small grid
under the full official 10-fold CV. For each lambda it records classification
macro-F1 (mean +/- std), the training-adversary's own speech-attribute accuracy
(an internal sanity signal, NOT a privacy result), codebook perplexity, and the
nominal + perplexity-effective bitrate. Writes sweep.json / sweep.csv and a
lambda-vs-macro-F1 figure incrementally; per-point results.json are preserved by
run_training.

Usage:
    python -u scripts/run_lambda_sweep.py --wav-cache data/processed/wavcache
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edbr1.config import load_train_config  # noqa: E402
from edbr1.train import _pick_device, run_training  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "adv" / "adv_lambda_base.yaml"
DEFAULT_LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0]


def _effective_bps(record: dict[str, Any]) -> float | None:
    perp = record.get("codebook_perplexity_mean")
    tps = record.get("tokens_per_second")
    if not perp or not tps or perp <= 1:
        return None
    return tps * math.log2(perp)


def _record(config: Any, summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten one lambda run into a sweep row."""
    b = summary.get("bottleneck") or {}
    a = summary.get("adversary") or {}
    row: dict[str, Any] = {
        "grl_lambda": config.adversary.grl_lambda,
        "macro_f1_mean": summary["mean_macro_f1"],
        "macro_f1_std": summary["std_macro_f1"],
        "adversary_train_acc_mean": a.get("adversary_train_acc_mean"),
        "adversary_train_acc_std": a.get("adversary_train_acc_std"),
        "adversary_classes": a.get("adversary_classes"),
        "codebook_perplexity_mean": b.get("codebook_perplexity_mean"),
        "codebook_fraction_used_mean": b.get("codebook_fraction_used_mean"),
        "tokens_per_second": b.get("tokens_per_second"),
        "bits_per_second": b.get("bits_per_second"),
        "n_folds": len(summary["folds"]),
        "run_dir": summary.get("run_dir"),
    }
    row["effective_bits_per_second"] = _effective_bps(row)
    return row


def _write_aggregates(sweep_dir: Path, records: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    with (sweep_dir / "sweep.json").open("w", encoding="utf-8") as fh:
        json.dump({**meta, "points": records}, fh, indent=2)
    fields = [
        "grl_lambda", "macro_f1_mean", "macro_f1_std",
        "adversary_train_acc_mean", "adversary_train_acc_std", "adversary_classes",
        "codebook_perplexity_mean", "codebook_fraction_used_mean",
        "tokens_per_second", "bits_per_second", "effective_bits_per_second",
        "n_folds", "run_dir",
    ]
    with (sweep_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def plot_lambda(records: list[dict[str, Any]], out_path: Path) -> None:
    """Macro-F1 (left) and training-adversary accuracy (right) vs GRL lambda."""
    pts = sorted(records, key=lambda r: r["grl_lambda"])
    lam = [p["grl_lambda"] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        lam, [p["macro_f1_mean"] for p in pts], yerr=[p["macro_f1_std"] for p in pts],
        marker="o", color="C0", capsize=3, label="classification macro-F1",
    )
    ax.set_xlabel("GRL reversal strength lambda")
    ax.set_ylabel("Macro-F1 (10-fold CV)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.grid(True, ls=":", alpha=0.4)

    adv_pts = [
        (p["grl_lambda"], p["adversary_train_acc_mean"])
        for p in pts if p.get("adversary_train_acc_mean") is not None
    ]
    if adv_pts:
        ax2 = ax.twinx()
        ax2.plot(
            [a for a, _ in adv_pts], [b for _, b in adv_pts],
            marker="s", color="C3", ls="--",
            label="train-adversary acc (SANITY, not privacy)",
        )
        ax2.set_ylabel("Train-adversary accuracy", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
    fig.suptitle("Utility & training-adversary vs GRL lambda (1000 bits/s knee)")
    fig.legend(loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--root", type=Path,
                        default=PROJECT_ROOT / "data" / "raw" / "urbansound8k" / "UrbanSound8K")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wav-cache", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (debug only).")
    parser.add_argument("--test-folds", type=int, nargs="+", default=None,
                        help="Subset of folds (DEBUG only; results need the full 10).")
    args = parser.parse_args(argv)

    device = _pick_device(args.device)
    base = load_train_config(args.config)
    sweep_dir = args.results_dir / time.strftime("us8k_adv_lambda_%Y%m%d_%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep dir: {sweep_dir}\nDevice: {device}\nLambdas: {args.lambdas}")

    meta: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "UrbanSound8K official 10-fold CV, macro-F1; adversarial GRL sweep",
        "bitrate_note": "fixed at 1000 bits/s knee, anti-collapse codebook",
        "adversary_note": "train-adversary accuracy is a sanity signal, NOT a privacy result",
        "device": str(device),
    }
    records: list[dict[str, Any]] = []
    for i, lam in enumerate(args.lambdas, start=1):
        print(f"\n{'#' * 70}\n# [{i}/{len(args.lambdas)}] lambda = {lam}\n{'#' * 70}")
        adv = dataclasses.replace(base.adversary, grl_lambda=lam)
        config = dataclasses.replace(base, adversary=adv)
        if args.epochs is not None:
            config = dataclasses.replace(config, epochs=args.epochs)
        try:
            summary = run_training(
                config, root=args.root, results_dir=args.results_dir, device=device,
                test_folds=args.test_folds, cache_dir=args.wav_cache,
            )
        except Exception as exc:  # keep the sweep alive; record the failure
            print(f"!! lambda={lam} FAILED: {exc!r}")
            records.append({"grl_lambda": lam, "error": repr(exc)})
            _write_aggregates(sweep_dir, records, meta)
            continue
        records.append(_record(config, summary))
        _write_aggregates(sweep_dir, records, meta)
        try:
            plot_lambda(
                [r for r in records if "error" not in r],
                sweep_dir / "lambda_vs_utility.png",
            )
        except Exception as exc:  # plotting must never kill the sweep
            print(f"(plot skipped: {exc!r})")

    print(f"\nSweep complete. Aggregates + figure in: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
