"""
Run the VQ utility-vs-bitrate sweep and produce the dissertation trade-off curve.

For each operating-point config this trains the full **official 10-fold CV**
(via :func:`edbr1.train.run_training`, the exact same protocol and per-run
artifacts as ``python -m edbr1.train``) and records the honest bitrate,
mean +/- std macro-F1 across folds, and codebook usage. The no-bottleneck
control config is run too and drawn as the utility ceiling (it has no finite
bitrate -- its latent is continuous).

Outputs land in a single timestamped sweep directory under ``results/``:

    sweep.json          per-point records + control reference + provenance
    sweep.csv           (config, bitrate, macro-F1 mean/std, codebook usage, run_dir)
    bitrate_curve.png    macro-F1 vs bits/s (log x), control ceiling dashed

The aggregate files are rewritten after *every* completed point, so a crash
part-way through still leaves a valid partial sweep (and every point also keeps
its own ``results.json`` in its own run dir). Conclusions must come from the full
10-fold numbers only -- do not read anything into a partial-fold run.

Usage:
    python scripts/run_bitrate_sweep.py \
        --root data/raw/urbansound8k/UrbanSound8K
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from edbr1.config import load_train_config
from edbr1.train import _pick_device, run_training

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = PROJECT_ROOT / "configs"

# The six swept operating points, in increasing bitrate (see configs/vq/).
DEFAULT_SWEEP = [
    CONFIGS / "vq" / "vq_00080bps.yaml",
    CONFIGS / "vq" / "vq_00250bps.yaml",
    CONFIGS / "vq" / "vq_01000bps.yaml",
    CONFIGS / "vq" / "vq_02000bps.yaml",
    CONFIGS / "vq" / "vq_04000bps.yaml",
    CONFIGS / "vq" / "vq_16000bps.yaml",
]
DEFAULT_CONTROL = CONFIGS / "encoder_nobottleneck.yaml"


def _record_from_summary(config_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten a run summary into one sweep row (bitrate + utility + usage)."""
    bottleneck = summary.get("bottleneck") or {}
    return {
        "config": config_path.name,
        "model": summary["config"]["model"],
        "bottleneck_type": summary["config"]["bottleneck"]["type"],
        "codebook_size": bottleneck.get("codebook_size"),
        "latent_freq": summary["config"]["encoder"]["latent_freq"],
        "latent_frames": summary["config"]["encoder"]["latent_frames"],
        "tokens_per_second": bottleneck.get("tokens_per_second"),
        "bits_per_token": bottleneck.get("bits_per_token"),
        "bits_per_second": bottleneck.get("bits_per_second"),
        "macro_f1_mean": summary["mean_macro_f1"],
        "macro_f1_std": summary["std_macro_f1"],
        "codebook_perplexity_mean": bottleneck.get("codebook_perplexity_mean"),
        "codebook_perplexity_std": bottleneck.get("codebook_perplexity_std"),
        "codebook_fraction_used_mean": bottleneck.get("codebook_fraction_used_mean"),
        "n_folds": len(summary["folds"]),
        "run_dir": summary.get("run_dir"),
    }


def _write_aggregates(sweep_dir: Path, records: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    """(Re)write sweep.json and sweep.csv from the records gathered so far."""
    with (sweep_dir / "sweep.json").open("w", encoding="utf-8") as fh:
        json.dump({**meta, "points": records}, fh, indent=2)

    fields = [
        "config", "bottleneck_type", "codebook_size", "latent_freq", "latent_frames",
        "tokens_per_second", "bits_per_token", "bits_per_second",
        "macro_f1_mean", "macro_f1_std",
        "codebook_perplexity_mean", "codebook_fraction_used_mean",
        "n_folds", "run_dir",
    ]
    with (sweep_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def plot_curve(
    records: list[dict[str, Any]], control: dict[str, Any] | None, out_path: Path
) -> None:
    """Plot macro-F1 vs bitrate (log x) with the control drawn as the ceiling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vq = sorted(
        [r for r in records if r.get("bits_per_second")],
        key=lambda r: r["bits_per_second"],
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    if vq:
        xs = [r["bits_per_second"] for r in vq]
        ys = [r["macro_f1_mean"] for r in vq]
        es = [r["macro_f1_std"] for r in vq]
        ax.errorbar(
            xs, ys, yerr=es, marker="o", capsize=4, linewidth=1.8,
            color="#1f77b4", label="VQ bottleneck (10-fold mean +/- std)",
        )
        ax.set_xscale("log")
        for r in vq:
            ax.annotate(
                f"{r['macro_f1_mean']:.2f}",
                (r["bits_per_second"], r["macro_f1_mean"]),
                textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center",
            )

    if control is not None:
        cy = control["macro_f1_mean"]
        ax.axhline(cy, linestyle="--", color="#555555", linewidth=1.4)
        ax.axhspan(
            cy - control["macro_f1_std"], cy + control["macro_f1_std"],
            color="#999999", alpha=0.15,
        )
        ax.text(
            0.01, cy, f"  no-bottleneck control {cy:.3f}",
            transform=ax.get_yaxis_transform(), va="bottom", fontsize=9, color="#333333",
        )

    ax.set_xlabel("Bitrate (bits/second, log scale)")
    ax.set_ylabel("Macro-F1 (UrbanSound8K official 10-fold CV)")
    ax.set_title("Utility vs bitrate -- VQ bottleneck on UrbanSound8K")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "urbansound8k" / "UrbanSound8K",
        help="Extracted UrbanSound8K directory.",
    )
    parser.add_argument(
        "--configs", type=Path, nargs="+", default=DEFAULT_SWEEP,
        help="Operating-point config YAMLs, in any order (default: the six VQ points).",
    )
    parser.add_argument(
        "--control", type=Path, default=DEFAULT_CONTROL,
        help="No-bottleneck control config drawn as the utility ceiling "
        "(pass 'none' to skip).",
    )
    parser.add_argument(
        "--control-results", type=Path, default=None,
        help="Reuse an EXISTING control run (its dir or results.json) as the "
        "ceiling instead of training a fresh control. Overrides --control.",
    )
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs (debug only; not for results)."
    )
    parser.add_argument(
        "--test-folds", type=int, nargs="+", default=None,
        help="Subset of folds (DEBUG plumbing only; results require the full 10-fold set).",
    )
    parser.add_argument(
        "--wav-cache", type=Path, default=None,
        help="Directory for the resampled-waveform cache (decode once, reuse across "
        "epochs/folds/points; results bit-identical). Strongly speeds the sweep.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override dataloader workers for every point (e.g. 18 to use idle cores). "
        "Recorded in each run's saved config.",
    )
    args = parser.parse_args(argv)

    device = _pick_device(args.device)
    sweep_dir = args.results_dir / time.strftime("us8k_vq_sweep_%Y%m%d_%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep dir: {sweep_dir}\nDevice: {device}")

    records: list[dict[str, Any]] = []
    control_record: dict[str, Any] | None = None
    meta: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "UrbanSound8K official 10-fold CV, macro-F1",
        "device": str(device),
    }

    # Reuse an already-trained control as the ceiling, if pointed at one.
    if args.control_results is not None:
        results_json = args.control_results
        if results_json.is_dir():
            results_json = results_json / "results.json"
        control_summary = json.loads(results_json.read_text(encoding="utf-8"))
        control_record = _record_from_summary(Path("encoder_nobottleneck.yaml"), control_summary)
        control_record["run_dir"] = str(results_json.parent)
        meta["control"] = control_record
        meta["control_config"] = "encoder_nobottleneck.yaml (reused)"
        run_control = False
    else:
        run_control = str(args.control).lower() != "none"
        meta["control_config"] = args.control.name if run_control else None
    queue = ([args.control] if run_control else []) + list(args.configs)
    for i, cfg_path in enumerate(queue, start=1):
        print(f"\n{'#' * 70}\n# [{i}/{len(queue)}] {cfg_path.name}\n{'#' * 70}")
        config = load_train_config(cfg_path)
        overrides: dict[str, Any] = {}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        if args.num_workers is not None:
            overrides["num_workers"] = args.num_workers
        if overrides:
            import dataclasses

            config = dataclasses.replace(config, **overrides)
        try:
            summary = run_training(
                config, root=args.root, results_dir=args.results_dir, device=device,
                test_folds=args.test_folds, cache_dir=args.wav_cache,
            )
        except Exception as exc:  # keep the sweep alive; record the failure
            print(f"!! {cfg_path.name} FAILED: {exc!r}")
            records.append({"config": cfg_path.name, "error": repr(exc)})
            _write_aggregates(sweep_dir, records, meta)
            continue

        record = _record_from_summary(cfg_path, summary)
        is_control = run_control and cfg_path == args.control
        if is_control:
            control_record = record
            meta["control"] = record
        else:
            records.append(record)

        _write_aggregates(sweep_dir, records, meta)
        # Redraw the curve after each point so progress is visible mid-sweep.
        try:
            plot_curve(records, control_record, sweep_dir / "bitrate_curve.png")
        except Exception as exc:  # plotting must never kill the sweep
            print(f"(plot skipped: {exc!r})")

        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()

    print(f"\nSweep complete. Aggregates + curve in: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
