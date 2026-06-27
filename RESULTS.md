# Results log — EDBR.1

Living record of headline metrics and the run artifacts that produced them.
The progression below is the raw material for the dissertation's baseline
**ablation table** (what each regularisation component contributes), so each
row keeps its config, protocol, number and artifact path.

## UrbanSound8K — small-CNN baseline (official 10-fold CV, macro-F1)

| # | Configuration | Protocol | Mean macro-F1 | Artifact |
|---|---------------|----------|---------------|----------|
| 1 | Plain CNN, no regularisation — `configs/baseline.yaml` | 10-fold | **0.626 ± 0.127** | `results/us8k_baseline_20260625_171148/` |
| 2 | + augmentation + per-band norm + cosine LR + early stopping, 16 kHz | 3-fold (1–3) | ~0.698 | `results/us8k_baseline_20260626_161434/` |
| 3 | same recipe, 22.05 kHz | 3-fold (1–3) | 0.711 | `results/us8k_baseline_20260626_175658/` |
| 4 | **same recipe, 16 kHz — CANONICAL** — `configs/baseline_final.yaml` | 10-fold | **0.746 ± 0.050** | `results/us8k_baseline_20260627_132115/` |
| 5 | same recipe, 22.05 kHz (tested, no gain) — `configs/improved_22k.yaml` | 10-fold | 0.739 ± 0.050 | `results/us8k_baseline_20260626_215429/` |

Published small-CNN reference band: **~0.73–0.76** macro-F1. The canonical
16 kHz run sits inside that band.

### Headline ablation progression (for the dissertation table)

```
plain CNN                                              0.626 ± 0.127  (10-fold)
  + augmentation + per-band norm + cosine LR + early stop (16 kHz)
                                                        0.746 ± 0.050  (10-fold)
  + 22.05 kHz sample rate                               0.739 ± 0.050  (10-fold)  -> no gain
```

The regularisation recipe is what closes the gap (0.626 → 0.746). Sample rate
was tested as a further lever and **rejected**: it does not help (see A/B below).

### Sample-rate A/B — 16 kHz vs 22.05 kHz, full 10-fold (only sample rate differs)

| fold | 16 kHz (canonical) | 22.05 kHz | Δ (22k − 16k) |
|------|--------------------|-----------|---------------|
| 1 | 0.7263 | 0.6922 | −0.0341 |
| 2 | 0.7700 | 0.7891 | +0.0191 |
| 3 | 0.6347 | 0.7063 | +0.0716 |
| 4 | 0.7251 | 0.7332 | +0.0081 |
| 5 | 0.8375 | 0.8666 | +0.0291 |
| 6 | 0.7321 | 0.7163 | −0.0158 |
| 7 | 0.7874 | 0.7508 | −0.0366 |
| 8 | 0.7378 | 0.7309 | −0.0069 |
| 9 | 0.7436 | 0.7130 | −0.0306 |
| 10 | 0.7645 | 0.6948 | −0.0697 |
| **mean** | **0.7459 ± 0.0495** | **0.7393 ± 0.0504** | **−0.0066** |

**Verdict:** 22.05 kHz wins 4 folds, loses 6, net −0.0066 (~0.13σ) — statistically
indistinguishable, and it costs ~1.4× the compute. The +0.013 seen on the 3-fold
check was noise. **16 kHz is retained as canonical.** A genuine rejected-lever
result, not a tuning failure — `configs/improved_22k.yaml` is kept as evidence.

### Reproduce the canonical run

```
python -u -m edbr1.train --root data/raw/urbansound8k/UrbanSound8K --config configs/baseline_final.yaml
```

Method invariants held across every row: official fold split with leak guard,
normalisation/early-stopping statistics estimated on training folds only
(never the test fold), seed 1337 (+ deterministic per-worker seeding). The
exact resolved config for each run is saved as `config.yaml` inside its
artifact directory.

### Per-class follow-up (for later)

The per-fold confusion matrices for both full runs are saved alongside their
`results.json` (16 kHz: `..._20260627_132115/`, 22.05 kHz: `..._20260626_215429/`).
These show which classes are hard — expected to be the acoustically overlapping
ones (drilling vs jackhammer, air_conditioner vs engine_idling) — which becomes
useful when explaining how the privacy bottleneck affects different sound types.
