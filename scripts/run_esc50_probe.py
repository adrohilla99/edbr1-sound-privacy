"""Phase-4b Task B: ESC-50 cross-dataset generalisation on the frozen code.

Freeze a UrbanSound8K-trained encoder (the 1000 bits/s knee, lambda 0 and 2) and
ask whether its low-bitrate code still carries transferable *scene* information on
a dataset it never saw. The encoder is frozen; only a light MLP head is trained on
the pooled quantised latent, under ESC-50's official 5-fold CV (leak-guarded).
Reports 50-way macro-F1 vs chance (2%).

Usage:
    python -u scripts/run_esc50_probe.py --device auto
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edbr1.data.esc50 import (  # noqa: E402
    ESC50_NUM_CLASSES,
    ESC50Dataset,
    load_esc50_metadata,
)
from edbr1.evaluate import classification_metrics  # noqa: E402
from edbr1.probes.frozen import FrozenEncoder  # noqa: E402

ESC50 = PROJECT_ROOT / "data" / "raw" / "esc50" / "ESC-50-master"
KNEE = ("bps01000_l0", "bps01000_l2")


def extract_latents(frozen: FrozenEncoder, dataset: ESC50Dataset, device: torch.device):
    """Frozen pooled latents + targets for every ESC-50 clip (order preserved)."""
    loader: DataLoader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    latents, targets = [], []
    for mels, tgt in loader:
        latents.append(frozen.emit_pooled_latent(mels).cpu())
        targets.append(tgt)
    return torch.cat(latents), torch.cat(targets)


def train_head(x_tr, y_tr, x_te, latent_dim: int, class_names: list[str],
               device: torch.device, *, epochs: int, seed: int) -> list[int]:
    """Light MLP head on the frozen latent -> test-set predictions."""
    torch.manual_seed(seed)
    head = nn.Sequential(
        nn.Linear(latent_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(256, ESC50_NUM_CLASSES),
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tr, y_tr, x_te = x_tr.to(device), y_tr.to(device), x_te.to(device)
    for _ in range(epochs):
        head.train()
        for i in torch.randperm(len(x_tr), device=device).split(128):
            opt.zero_grad()
            nn.functional.cross_entropy(head(x_tr[i]), y_tr[i]).backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        preds = head(x_te).argmax(dim=1).cpu().tolist()
    return preds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "results" / "probe_encoders" / "manifest.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    out_dir = args.results_dir / time.strftime("esc50_transfer_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ESC-50 transfer: {out_dir}\nDevice: {device}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    meta = load_esc50_metadata(ESC50)
    class_names = [c for _, c in sorted(set(zip(meta["target"], meta["category"], strict=True)))]
    folds = torch.tensor(meta["fold"].to_numpy())

    records: list[dict[str, Any]] = []
    for name in KNEE:
        info = manifest[name]
        frozen = FrozenEncoder(info["checkpoint"], device)
        dataset = ESC50Dataset(meta, frozen.config.features, frozen.config.clip_seconds)
        latents, targets = extract_latents(frozen, dataset, device)
        latent_dim = latents.shape[1]

        fold_f1: list[float] = []
        for test_fold in range(1, 6):
            te = folds == test_fold
            tr = ~te
            preds = train_head(latents[tr], targets[tr], latents[te], latent_dim,
                               class_names, device, epochs=args.epochs, seed=args.seed)
            f1 = float(classification_metrics(targets[te].tolist(), preds, class_names)["macro_f1"])
            fold_f1.append(f1)
        mean = sum(fold_f1) / len(fold_f1)
        std = (sum((f - mean) ** 2 for f in fold_f1) / len(fold_f1)) ** 0.5
        rec = {
            "name": name, "bits_per_second": info["bits_per_second"],
            "grl_lambda": info["grl_lambda"], "esc50_macro_f1_mean": mean,
            "esc50_macro_f1_std": std, "per_fold": fold_f1,
            "chance": 1.0 / ESC50_NUM_CLASSES, "us8k_utility_macro_f1": info["macro_f1"],
        }
        records.append(rec)
        (out_dir / "esc50.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"  {name}: ESC-50 50-way macro-F1 {mean:.3f} +/- {std:.3f} "
              f"(chance {1 / ESC50_NUM_CLASSES:.3f})")

    print(f"\nESC-50 transfer complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
