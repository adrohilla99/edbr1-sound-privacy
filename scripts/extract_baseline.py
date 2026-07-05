"""Extract the baseline / sample-rate / control numbers from the run artifacts.

Reads the (gitignored) ``results/*/results.json`` for the Chapter-4 baseline runs
and emits ``docs/figures/baseline_ablation.json`` **programmatically** -- no
hand-typed values -- so the committed snapshot is source-derived. The run
directories exist on the training machine only; if one is missing this stops
rather than fabricating.

Usage:
    python scripts/extract_baseline.py            # write the snapshot
    python scripts/extract_baseline.py --check    # print + reconcile, no write
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "docs" / "figures" / "baseline_ablation.json"

# name, run_dir, config, description, protocol
RUNS = [
    ("plain", "us8k_baseline_20260625_171148", "baseline.yaml",
     "plain CNN, no regularisation", "10-fold"),
    ("regularised_3fold_16k", "us8k_baseline_20260626_161434", "regularised 16 kHz (partial)",
     "+augmentation +per-band norm +cosine LR +early stopping", "3-fold (1-3)"),
    ("regularised_3fold_22k", "us8k_baseline_20260626_175658", "regularised 22.05 kHz (partial)",
     "same recipe, 22.05 kHz", "3-fold (1-3)"),
    ("canonical_16k", "us8k_baseline_20260627_132115", "baseline_final.yaml",
     "CANONICAL regularised recipe, 16 kHz", "10-fold"),
    ("sr_22k", "us8k_baseline_20260626_215429", "improved_22k.yaml",
     "same recipe, 22.05 kHz (tested, no gain)", "10-fold"),
    ("control", "us8k_encoder_20260702_181348", "encoder_nobottleneck.yaml",
     "no-bottleneck encoder->classifier control", "10-fold"),
]


def _read(run_dir: str) -> dict[str, Any]:
    path = RESULTS / run_dir / "results.json"
    if not path.is_file():
        raise SystemExit(f"MISSING run artifact: {path}\n"
                         "Run directory not on this machine; stopping (not fabricating).")
    d = json.loads(path.read_text(encoding="utf-8"))
    per_fold = d["per_fold_macro_f1"]
    ordered = [round(per_fold[k], 4) for k in sorted(per_fold, key=int)]
    return {"mean": round(d["mean_macro_f1"], 4), "std": round(d["std_macro_f1"], 4),
            "per_fold": ordered, "n_folds": len(ordered)}


def build() -> dict[str, Any]:
    reads = {name: _read(run_dir) for name, run_dir, *_ in RUNS}
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
    ablation = []
    for name, run_dir, config, desc, protocol in RUNS:
        if name == "control":
            continue
        r = reads[name]
        ablation.append({
            "config": config, "desc": desc, "protocol": protocol,
            "macro_f1_mean": r["mean"], "macro_f1_std": r["std"],
            "per_fold": r["per_fold"], "artifact": f"results/{run_dir}/",
        })
    c16, c22 = reads["canonical_16k"], reads["sr_22k"]
    ctrl = reads["control"]
    return {
        "note": f"Source-derived from results/*/results.json by scripts/extract_baseline.py "
                f"at commit {head}; NOT hand-typed. The run artifacts are gitignored.",
        "provenance": {name: f"results/{run_dir}/results.json" for name, run_dir, *_ in RUNS},
        "ablation": ablation,
        "sample_rate_ab_per_fold": {
            "fold": list(range(1, len(c16["per_fold"]) + 1)),
            "macro_f1_16khz": c16["per_fold"], "macro_f1_22050hz": c22["per_fold"],
            "mean_16khz": c16["mean"], "std_16khz": c16["std"],
            "mean_22050hz": c22["mean"], "std_22050hz": c22["std"],
            "delta_mean": round(c22["mean"] - c16["mean"], 4),
            "verdict": "22.05 kHz statistically indistinguishable at ~1.4x compute; "
                       "16 kHz retained as canonical",
        },
        "control": {
            "config": "encoder_nobottleneck.yaml", "macro_f1_mean": ctrl["mean"],
            "macro_f1_std": ctrl["std"], "per_fold": ctrl["per_fold"],
            "artifact": "results/us8k_encoder_20260702_181348/",
        },
    }


def _reconcile(new: dict[str, Any]) -> None:
    """Diff freshly extracted numbers against the current committed snapshot."""
    if not OUT.is_file():
        print("(no existing snapshot to reconcile against)")
        return
    old = json.loads(OUT.read_text(encoding="utf-8"))
    old_map = {a["config"]: a["macro_f1_mean"] for a in old.get("ablation", [])}
    print("\nReconcile vs current committed snapshot (tol 1e-3):")
    all_match = True
    for a in new["ablation"]:
        ov = old_map.get(a["config"])
        nv = a["macro_f1_mean"]
        match = ov is not None and abs(ov - nv) < 1e-3
        all_match = all_match and match
        print(f"  {a['config']:<28} old={ov} new={nv}  {'MATCH' if match else 'DIFFER'}")
    print("  ALL MATCH" if all_match else "  >>> DIFFERENCES (extracted is authoritative) <<<")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="print + reconcile, do not write")
    args = parser.parse_args(argv)
    data = build()
    if args.check:
        print(json.dumps(data, indent=2))
        _reconcile(data)
        return 0
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
