"""Regenerate the key dissertation figures from committed sweep data.

The heavy sweep artifacts live under the gitignored ``results/``. So that the
figures stay auditable and reproducible from the repo alone, the small numbers
behind them are extracted once into a committed JSON
(``docs/figures/sweep_data.json``); this script plots from that file with
matplotlib only. Pass ``--refresh`` to rebuild the JSON from the live
``results/`` sweeps (e.g. after re-running a sweep).

Figures written to ``docs/figures/``:
  * utility_vs_bitrate.png     -- anti-collapse utility-vs-bitrate curve
  * collapsed_vs_fixed.png     -- collapsed vs anti-collapse (the flattening)
  * codebook_usage.png         -- codes used vs bitrate, collapsed vs fixed

Usage:
    python scripts/make_figures.py            # plot from committed sweep_data.json
    python scripts/make_figures.py --refresh  # rebuild sweep_data.json from results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never a display
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
DATA_JSON = FIG_DIR / "sweep_data.json"

# Live (gitignored) sweeps the committed data is extracted from.
DEFAULT_COLLAPSED = PROJECT_ROOT / "results" / "us8k_vq_sweep_20260702_233340" / "sweep.json"
DEFAULT_ANTICOLLAPSE = PROJECT_ROOT / "results" / "us8k_vq_sweep_20260703_171719" / "sweep.json"
DEFAULT_LAMBDA = PROJECT_ROOT / "results" / "us8k_adv_lambda_20260704_112525" / "sweep.json"

_POINT_KEYS = (
    "bits_per_second", "tokens_per_second", "macro_f1_mean", "macro_f1_std",
    "codebook_perplexity_mean", "codebook_fraction_used_mean",
)
_LAMBDA_KEYS = ("grl_lambda", "macro_f1_mean", "macro_f1_std", "adversary_train_acc_mean")


def _points(sweep: dict[str, Any]) -> list[dict[str, float]]:
    """Extract only the plotted fields (no absolute paths) from a sweep dict."""
    rows = [{k: p.get(k) for k in _POINT_KEYS} for p in sweep["points"]]
    return sorted(rows, key=lambda r: r["bits_per_second"] or 0.0)


def _lambda_points(sweep: dict[str, Any]) -> list[dict[str, float]]:
    rows = [{k: p.get(k) for k in _LAMBDA_KEYS} for p in sweep["points"] if "error" not in p]
    return sorted(rows, key=lambda r: r["grl_lambda"])


def build_data(
    collapsed_json: Path, anticollapse_json: Path, lambda_json: Path | None = None
) -> dict[str, Any]:
    """Extract the committed, path-free figure data from the live sweep.json files."""
    collapsed = json.loads(collapsed_json.read_text(encoding="utf-8"))
    anti = json.loads(anticollapse_json.read_text(encoding="utf-8"))
    control = anti.get("control") or collapsed.get("control") or {}
    data: dict[str, Any] = {
        "note": "Extracted from gitignored results/ sweeps; source of the committed figures.",
        "control": {
            "macro_f1_mean": control.get("macro_f1_mean"),
            "macro_f1_std": control.get("macro_f1_std"),
        },
        "collapsed": _points(collapsed),
        "anticollapse": _points(anti),
    }
    if lambda_json is not None and lambda_json.exists():
        data["lambda_sweep"] = _lambda_points(json.loads(lambda_json.read_text(encoding="utf-8")))
    return data


def _errbar(ax: Any, pts: list[dict[str, float]], **kw: Any) -> None:
    ax.errorbar(
        [p["bits_per_second"] for p in pts],
        [p["macro_f1_mean"] for p in pts],
        yerr=[p["macro_f1_std"] for p in pts],
        marker="o", capsize=3, **kw,
    )


def _control_band(ax: Any, control: dict[str, float]) -> None:
    m, s = control["macro_f1_mean"], control["macro_f1_std"]
    ax.axhline(m, ls="--", color="0.4", lw=1, label=f"no-bottleneck control {m:.3f}")
    ax.axhspan(m - s, m + s, color="0.85", zorder=0)


def plot_utility_vs_bitrate(data: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _control_band(ax, data["control"])
    _errbar(ax, data["anticollapse"], color="C0", label="VQ bottleneck (anti-collapse)")
    ax.set_xscale("log")
    ax.set_xlabel("Bitrate (bits/second, log scale)")
    ax.set_ylabel("Macro-F1 (UrbanSound8K official 10-fold CV)")
    ax.set_title("Utility vs bitrate -- VQ bottleneck (codebook fixed)")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_collapsed_vs_fixed(data: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _control_band(ax, data["control"])
    _errbar(ax, data["collapsed"], color="C3", label="collapsed (loss-VQ)")
    _errbar(ax, data["anticollapse"], color="C0", label="anti-collapse (EMA+kmeans+revival)")
    ax.set_xscale("log")
    ax.set_xlabel("Bitrate (bits/second, log scale)")
    ax.set_ylabel("Macro-F1 (10-fold CV)")
    ax.set_title("Fixing codebook collapse flattens the utility-bitrate curve")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_codebook_usage(data: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, color, label in (
        ("collapsed", "C3", "collapsed (loss-VQ)"),
        ("anticollapse", "C0", "anti-collapse"),
    ):
        pts = data[key]
        ax.plot(
            [p["bits_per_second"] for p in pts],
            [p["codebook_fraction_used_mean"] * 100 for p in pts],
            marker="o", color=color, label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Bitrate (bits/second, log scale)")
    ax.set_ylabel("Codebook used (% of 1024 codes)")
    ax.set_title("Codebook utilisation: collapsed vs fixed")
    ax.legend(loc="center right")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_lambda_utility(data: dict[str, Any], out: Path) -> None:
    """Macro-F1 (left) and training-adversary accuracy (right) vs GRL lambda.

    The adversary axis is fixed to [0.4, 1.0] with the no-speech-majority floor
    (0.5) drawn in, so the adversary sitting *at* the floor is shown honestly
    rather than exaggerated by auto-scaling.
    """
    pts = data.get("lambda_sweep")
    if not pts:
        return
    lam = [p["grl_lambda"] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _control_band(ax, data["control"])
    ax.errorbar(
        lam, [p["macro_f1_mean"] for p in pts], yerr=[p["macro_f1_std"] for p in pts],
        marker="o", color="C0", capsize=3, label="classification macro-F1",
    )
    ax.set_xlabel("GRL reversal strength lambda")
    ax.set_ylabel("Macro-F1 (10-fold CV)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_ylim(0.4, 0.82)
    ax.grid(True, ls=":", alpha=0.4)

    ax2 = ax.twinx()
    ax2.plot(
        lam, [p["adversary_train_acc_mean"] for p in pts],
        marker="s", color="C3", ls="--", label="train-adversary acc (SANITY, not privacy)",
    )
    ax2.axhline(0.5, color="C3", ls=":", lw=1, alpha=0.6)  # no-speech-majority floor
    ax2.set_ylabel("Train-adversary accuracy", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(0.4, 1.0)
    ax.set_title("Adversarial GRL sweep: utility flat, adversary at floor (1000 bits/s)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="Rebuild docs/figures/sweep_data.json from the live results/ sweeps.",
    )
    parser.add_argument("--collapsed", type=Path, default=DEFAULT_COLLAPSED)
    parser.add_argument("--anticollapse", type=Path, default=DEFAULT_ANTICOLLAPSE)
    parser.add_argument("--lambda-sweep", type=Path, default=DEFAULT_LAMBDA, dest="lambda_json")
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh or not DATA_JSON.exists():
        data = build_data(args.collapsed, args.anticollapse, args.lambda_json)
        DATA_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {DATA_JSON}")
    else:
        data = json.loads(DATA_JSON.read_text(encoding="utf-8"))

    plot_utility_vs_bitrate(data, FIG_DIR / "utility_vs_bitrate.png")
    plot_collapsed_vs_fixed(data, FIG_DIR / "collapsed_vs_fixed.png")
    plot_codebook_usage(data, FIG_DIR / "codebook_usage.png")
    plot_lambda_utility(data, FIG_DIR / "lambda_vs_utility.png")
    print(f"figures written to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
