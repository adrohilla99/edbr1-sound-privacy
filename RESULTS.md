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

## UrbanSound8K — utility vs bitrate (VQ discrete bottleneck, official 10-fold CV)

The project's first trade-off curve. The baseline classifier is split into an
on-device encoder **E** → a VQ discrete bottleneck **B** → a classifier **C**
(VQ-VAE style: codebook, straight-through estimator, commitment loss). Codebook
size is held at **K = 1024** (10 bits/token); the **token rate** is swept to set
the bitrate, reported honestly as
`bits_per_second = tokens_per_second · log₂(codebook_size)`. There is **no
adversarial objective yet** — this is the utility-only reference the later
privacy results are compared against.

Each operating point is a full official **10-fold CV** run under the same
invariants as the baseline (leak-guarded fold split, train-only
normalisation/early-stopping stats, seed 1337 + deterministic per-worker
seeding). The no-bottleneck encoder→classifier **control** — 0.748 ± 0.058,
statistically identical to the 0.746 canonical CNN — is the utility ceiling.

| bits/s | tokens/s × log₂K | Macro-F1 (10-fold) | perplexity | codes used | run dir (under `results/`) |
|--------|------------------|--------------------|-----------|------------|----------------------------|
| ∞ (control) | — (no bottleneck) | **0.748 ± 0.058** | — | — | `us8k_encoder_20260702_181348/` |
| 80 | 8 × 10 | 0.506 ± 0.106 | 4.2 | 0.5% | `us8k_vq_20260702_233341/` |
| 250 | 25 × 10 | 0.639 ± 0.072 | 6.4 | 0.7% | `us8k_vq_20260703_015506/` |
| 1000 | 100 × 10 | 0.690 ± 0.027 | 12.3 | 1.6% | `us8k_vq_20260703_040308/` |
| 2000 | 200 × 10 | 0.687 ± 0.054 | 17.2 | 2.5% | `us8k_vq_20260703_063440/` |
| 4000 | 400 × 10 | 0.697 ± 0.038 | 22.7 | 3.8% | `us8k_vq_20260703_091718/` |
| 16000 | 1600 × 10 | 0.724 ± 0.040 | 25.3 | 5.4% | `us8k_vq_20260703_120335/` |

Sweep wall time 59,563 s (~16.5 h, one RTX 5060). Aggregate artifact
`results/us8k_vq_sweep_20260702_233340/` (`sweep.json`, `sweep.csv`,
`bitrate_curve.png`).

### Shape of the curve

```
control (no bottleneck)   0.748 ± 0.058
    80 bits/s             0.506 ± 0.106
   250 bits/s             0.639 ± 0.072
  1000 bits/s             0.690 ± 0.027
  2000 bits/s             0.687 ± 0.054
  4000 bits/s             0.697 ± 0.038
 16000 bits/s             0.724 ± 0.040
```

- Steep rise 80 → 1000 bits/s (0.506 → 0.690), then a **plateau at ~0.69 through
  4000 bits/s**, and only a partial recovery to **0.724 at 16 kbits/s** — still
  ~0.024 under the control, i.e. just inside the control's ±0.058 fold noise. The
  VQ bottleneck never fully reaches the unquantised ceiling in this range.
- **80 bits/s is both low and unstable** (±0.106, per-fold spread 0.37): at
  extreme compression some folds essentially fail. This is the low-bitrate
  fall-off the curve is meant to expose — reported as-is, not smoothed.

### Codebook collapse (reported plainly)

The codebook is **badly underused at every operating point.** Perplexity never
exceeds ~25 of 1024, and the fraction of codes ever used peaks at 5.4% (the
80 bits/s point uses ~0.5%, i.e. roughly four codes). So the *nominal* bitrate
overstates the true information rate: an effective rate of
`tokens/s · log₂(perplexity)` is only ~20–47% of nominal.

| nominal bits/s | perplexity (≈ effective codes) | effective bits/s ≈ tok/s·log₂(perplexity) | % of nominal |
|----------------|-------------------------------|-------------------------------------------|--------------|
| 80 | 4.2 | ~17 | 21% |
| 250 | 6.4 | ~67 | 27% |
| 1000 | 12.3 | ~362 | 36% |
| 2000 | 17.2 | ~821 | 41% |
| 4000 | 22.7 | ~1800 | 45% |
| 16000 | 25.3 | ~7500 | 47% |

With only classification + commitment loss to shape it (no adversary, no entropy
pressure), the quantiser settles on a handful of prototypes rather than spreading
mass across the codebook. This is itself a result to carry forward: it is the
codebook-usage baseline the privacy bottleneck's behaviour will be read against,
and it means later comparisons should track the **effective** rate, not just the
nominal capacity.

### Reproduce

```
python -u scripts/run_bitrate_sweep.py \
  --control-results results/us8k_encoder_20260702_181348 \
  --wav-cache data/processed/wavcache
```

`configs/encoder_nobottleneck.yaml` is the control; `configs/vq/vq_*bps.yaml` are
the six operating points. `--wav-cache` decodes+resamples each clip once to a
gitignored on-disk cache (bit-identical to recomputing); `--num-workers` can
override dataloader workers. The exact resolved config is saved as `config.yaml`
inside every run dir.
