"""Train the frozen encoders the Phase-4a probes attack.

Trains one encoder per operating point (single held-out fold, so it is one
encoder not a 10-fold set), all anti-collapse + overlay, saving a frozen-encoder
checkpoint and a manifest the probe runner reads. Operating points cover the
Phase-2 bitrate range and the Phase-3 lambda contrast:

    250 bits/s  (lambda 0),  1000 bits/s (lambda 0 and 2),  16000 bits/s (lambda 0)

Each encoder trains on UrbanSound8K folds 1-9 (fold 10 held out); the probes use
disjoint dev-clean speakers, so the encoder never saw them.

Usage:
    python -u scripts/train_probe_encoders.py --wav-cache data/processed/wavcache
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edbr1.config import load_train_config  # noqa: E402
from edbr1.train import _pick_device, run_training  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "adv" / "adv_lambda_base.yaml"
HELD_OUT_FOLD = 10  # encoder trains on folds 1-9; fold 10 held out

# (name, latent_freq, latent_frames, grl_lambda). tokens/s = freq*frames/4.
POINTS = [
    ("bps00250_l0", 2, 50, 0.0),
    ("bps01000_l0", 4, 100, 0.0),
    ("bps01000_l2", 4, 100, 2.0),
    ("bps16000_l0", 32, 200, 0.0),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path,
                        default=PROJECT_ROOT / "data" / "raw" / "urbansound8k" / "UrbanSound8K")
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results" / "probe_encoders",
    )
    parser.add_argument("--wav-cache", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (debug only).")
    args = parser.parse_args(argv)

    device = _pick_device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    base = load_train_config(args.config)
    manifest: dict[str, dict] = {}

    for i, (name, freq, frames, lam) in enumerate(POINTS, start=1):
        bar = "#" * 70
        print(f"\n{bar}\n# [{i}/{len(POINTS)}] {name}  grid {freq}x{frames}  lambda {lam}\n{bar}")
        encoder = dataclasses.replace(base.encoder, latent_freq=freq, latent_frames=frames)
        adversary = dataclasses.replace(base.adversary, grl_lambda=lam)
        config = dataclasses.replace(base, encoder=encoder, adversary=adversary)
        if args.epochs is not None:
            config = dataclasses.replace(config, epochs=args.epochs)
        summary = run_training(
            config, root=args.root, results_dir=args.results_dir, device=device,
            test_folds=[HELD_OUT_FOLD], cache_dir=args.wav_cache, save_checkpoints=True,
        )
        checkpoint = Path(summary["run_dir"]) / f"encoder_fold{HELD_OUT_FOLD}.pt"
        b = summary.get("bottleneck") or {}
        manifest[name] = {
            "checkpoint": str(checkpoint),
            "bits_per_second": b.get("bits_per_second"),
            "grl_lambda": lam,
            "macro_f1": summary["mean_macro_f1"],
            "codebook_perplexity": b.get("codebook_perplexity_mean"),
            "latent_freq": freq,
            "latent_frames": frames,
        }
        (args.results_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"  -> checkpoint {checkpoint}  (macro-F1 {summary['mean_macro_f1']:.3f})")

    print(f"\nEncoders trained. Manifest: {args.results_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
