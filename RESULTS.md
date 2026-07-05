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

## UrbanSound8K — fixing codebook collapse, then re-sweep (Phase 2b)

The sweep above is honest but **throttled by accident**: its codebook collapsed
at every point (≤5.4% of 1024 codes used, perplexity ≤25.3), so the *nominal*
bitrate overstated the true information rate 2–5×. Before any adversarial-privacy
work is built on top, the bottleneck must actually use its codebook — otherwise
later leakage numbers would confound "our method hides speech" with "the
bottleneck was throttled." This section diagnoses the collapse and fixes it.

### Diagnosis

Three standard causes, all present in the collapsed run:

1. **Data-independent init far from the latent distribution (primary).** The
   codebook was initialised `uniform(±1/K)` — a tiny blob at the origin — while
   the encoder head ends in BatchNorm+ReLU, so latents are non-negative and
   O(0.1–1). Most codes are nowhere near the data and are dead from step 0.
2. **No EMA** (`ema: false`): loss-based updates only move *selected* codes, so
   dead codes never migrate toward the data.
3. **No dead-code revival**: nothing ever resets unused codes.

Per-epoch perplexity (now logged) confirms it: the collapsed config sits at
perplexity ~12 throughout; the fixed config climbs to ~880 within three epochs
and holds (1000 bits/s, fold 1: ep1 135 → ep3 723 → ep4+ ~850–890).

### Fix (config-gated, all off by default so the collapsed run stays reproducible)

* **k-means init** — data-dependent codebook init from the first training
  batch's encoder outputs (a few Lloyd iterations).
* **EMA codebook** — van den Oord et al. (2017) Appendix A.
* **dead-code revival** — every `restart_interval` steps, codes whose usage EMA
  falls below `dead_code_threshold` are re-seeded to random batch vectors
  (Dhariwal et al. 2020).

Enabled together via `run_bitrate_sweep.py --anti-collapse`. Reference verify at
1000 bits/s (folds 1–3): perplexity **12.6 → 703**, codes used **1.6% → 99.5%**,
macro-F1 0.672 → 0.680.

### Re-sweep — before (collapsed) → after (anti-collapse), full 10-fold

| bits/s | macro-F1 collapsed → fixed | perplexity | codes used | effective bits/s (tok/s·log₂ ppl) |
|--------|---------------------------|------------|------------|-----------------------------------|
| 80 | 0.506 → **0.743** (+0.236) | 4.2 → 513.4 | 0.5% → 90.9% | 17 → 72 |
| 250 | 0.639 → **0.754** (+0.115) | 6.4 → 626.7 | 0.7% → 98.8% | 67 → 232 |
| 1000 | 0.690 → **0.723** (+0.033) | 12.3 → 691.9 | 1.6% → 99.7% | 362 → 943 |
| 2000 | 0.687 → **0.734** (+0.047) | 17.2 → 691.2 | 2.5% → 99.9% | 821 → 1887 |
| 4000 | 0.697 → **0.738** (+0.041) | 22.7 → 625.9 | 3.8% → 100.0% | 1802 → 3716 |
| 16000 | 0.724 → **0.751** (+0.028) | 25.3 → 483.6 | 5.4% → 100.0% | 7455 → 14268 |
| ∞ control | 0.748 ± 0.058 | — | — | — |

Anti-collapse sweep wall time 45,743 s (~12.7 h, one RTX 5060). Artifact
`results/us8k_vq_sweep_20260703_171719/` (`sweep.json`, `sweep.csv`,
`bitrate_curve.png`); per-point run dirs listed in `sweep.json`.

### Did fixing collapse change the curve? Yes — plainly.

1. **Collapse is fixed.** Codes used rise from ≤5.4% to **90.9–100%**; the
   effective rate is now **~89–94% of nominal** (was 21–47%). The reported
   bitrates are now honest.
2. **The curve flattens onto the control.** Every operating point lands at
   0.72–0.75 macro-F1 — all within fold-noise of the 0.748 unbottlenecked
   control. The collapsed curve's steep low-bitrate rise (0.506 → 0.724) is gone:

```
collapsed:  80→0.506  250→0.639  1000→0.690  2000→0.687  4000→0.697  16000→0.724
anti-collapse: 80→0.743 250→0.754 1000→0.723 2000→0.734 4000→0.738 16000→0.751
control 0.748
```

3. **The utility gain is concentrated at low bitrate** (+0.236 at 80 bits/s,
   +0.115 at 250, then within noise): that is exactly where the collapsed
   codebook (~4–6 live codes) was too small to separate 10 classes. Once the
   codebook is used, **≈72 honest bits/s already matches the unbottlenecked
   model** — this 10-way task needs very little information when it is spent
   efficiently.

Consequence for the privacy phase: on utility grounds the bottleneck can be
pushed very low almost for free, so the interesting trade-off will be about *what
else* those bits carry (speaker/speech leakage), not classification accuracy.

### Reproduce

```
python -u scripts/run_bitrate_sweep.py \
  --control-results results/us8k_encoder_20260702_181348 \
  --wav-cache data/processed/wavcache --anti-collapse
```

The anti-collapse levers are `BottleneckConfig` fields
(`kmeans_init`, `ema`, `restart_dead_codes`, `restart_interval`,
`dead_code_threshold`), off by default; `--anti-collapse` sets the first three
and the resolved values are saved in each run's `config.yaml`. The original
collapsed sweep (`results/us8k_vq_sweep_20260702_233340/`) remains reproducible
by omitting the flag — it stays a documented finding (the nominal-vs-effective
bitrate discrepancy).

## UrbanSound8K — adversarial speech-suppression at the knee (Phase 3)

Phase 2 showed classification utility is nearly free to compress, so the
question is whether an adversarial objective can suppress *speech* leakage
without spending that (cheap) utility. Setup: the training fold overlays
LibriSpeech speech into each UrbanSound8K scene (prob 0.5, RMS SNR ∈ {0,5,10} dB),
and a gradient-reversal adversary predicts a speech attribute from the code —
**(N+1)-way: 0 = no speech, 1..20 = closed-set `train-clean-100` speaker**. The
bitrate is fixed at the knee (1000 bits/s, anti-collapse codebook) and the GRL
strength `lambda` is swept. **Overlay + labels are train-only; the val/test folds
are clean UrbanSound8K**, so utility stays comparable and no speaker/fold crosses
the boundary. `lambda=0` is the non-adversarial baseline *on the overlay stream*
(not the Phase-2 clean 0.748).

| lambda | macro-F1 (10-fold) | Δ vs λ=0 | train-adv acc (21-way) | perplexity | eff. bits/s |
|--------|--------------------|----------|------------------------|------------|-------------|
| 0.0 | 0.729 ± 0.059 | — | 0.501 | 674 | 940 |
| 0.1 | 0.732 ± 0.063 | +0.003 | 0.504 | 657 | 936 |
| 0.5 | 0.733 ± 0.065 | +0.004 | 0.501 | 608 | 925 |
| 1.0 | 0.726 ± 0.060 | −0.003 | 0.501 | 580 | 918 |
| 2.0 | 0.724 ± 0.050 | −0.004 | 0.502 | 584 | 919 |

Full sweep 35,460 s (~9.85 h, one RTX 5060), artifact
`results/us8k_adv_lambda_20260704_112525/`
(`sweep.json`, `sweep.csv`, `lambda_vs_utility.png`); figure also regenerated to
`docs/figures/lambda_vs_utility.png`.

### Findings (plain, and carefully scoped)

1. **GRL training is stable and costs no classification utility.** Across a 20×
   range of `lambda` (0 → 2), macro-F1 stays flat at 0.724–0.733 — all within
   one fold-σ of each other and of the `lambda=0` baseline (worst case −0.004 at
   `lambda=2`). GRL training is notoriously unstable; here it was not. So the
   adversarial objective is essentially *free on the utility axis* at this knee.
2. **The training-time adversary never beats the trivial floor.** With
   `overlay_prob=0.5`, half the training examples are no-speech, so predicting
   "no speech" scores ~0.50. The 21-way adversary sits at **0.501–0.504 at every
   lambda, including `lambda=0`** (no reversal) — i.e. the modest MLP cannot
   extract speaker identity from the 1000 bits/s code even when *not* being
   fought. So there is little speaker signal for the GRL to suppress, which is
   consistent with utility being unmoved by `lambda`.
3. **The overlay itself costs ~0.02 utility.** The `lambda=0` overlay baseline
   (0.729) sits ~0.02 below the Phase-2 clean control (0.748): training on
   50%-speech-contaminated scenes but testing clean is slightly harder, as
   expected. This is the correct baseline for isolating the adversary's marginal
   cost (≈ 0).

### Important caveat — this is NOT the privacy result

The adversary accuracy above is a **training-time sanity signal only**. It shows
the adversary is weak (by design) and found little to suppress; it does **not**
establish that the code is private. That is measured in **Phase 4** by separate,
*stronger* probes (speaker-ID / ASR / inverter) trained against the frozen
representation, plus a held-out speech-overlay evaluation. The honest Phase-3
conclusion is narrow: *at the knee, the adversarial objective is stable and
utility-neutral, but the training-time adversary was too weak to exert pressure —
so a stronger probe (Phase 4) is needed to say anything about leakage.*

### Reproduce

```
python -u scripts/run_lambda_sweep.py --wav-cache data/processed/wavcache
```

Base config `configs/adv/adv_lambda_base.yaml` (1000 bits/s, anti-collapse,
overlay + adversary on); the runner sweeps `adversary.grl_lambda` over
{0, 0.1, 0.5, 1.0, 2.0}, each a full 10-fold run, and the resolved `grl_lambda`
is saved in every run's `config.yaml`. The training-adversary/overlay code is
config-gated and off by default, so the non-adversarial paths are unchanged.

## UrbanSound8K — evaluation-time speech-leakage probes (Phase 4a)

Phase 3 left an ambiguity: the training adversary never beat the no-speech floor,
so it was unknown whether the low-bitrate code destroys speech or the adversary
was just too weak. Phase 4a resolves it with **three independent, stronger probes
that attack the frozen encoder's codes** — the actual privacy measurement.
Discipline: the encoder is frozen (weights fingerprinted, verified unchanged);
probe speech is **held-out dev-clean speakers, disjoint from the encoder's
train-clean-100 overlay speakers** (so any success is representational leakage,
not memorisation); splits are utterance-disjoint (speaker-ID) or speaker-disjoint
(ASR/inversion) and **verified at construction**. Every number is an **empirical
LOWER bound** — a stronger future probe can only do better. The speaker probe is
~325K params (≈30× the ~11K Phase-3 adversary head).

| operating point | utility F1 | speaker top-1 (chance 0.05) | ASR WER / CER (ceiling ~1.0) | inverter LSD dB (silence floor 75.5) |
|-----------------|-----------|-----------------------------|------------------------------|--------------------------------------|
| 250 b/s, λ=0    | 0.772 | 0.125 (2.5× chance) | 1.12 / 0.80 | 17.2 |
| 1000 b/s, λ=0   | 0.804 | 0.112 (2.2× chance) | 1.12 / 0.77 | 16.1 |
| 1000 b/s, λ=2   | 0.807 | **0.062 (1.2× chance)** | 1.12 / 0.77 | 16.4 |
| 16000 b/s, λ=0  | 0.783 | 0.087 (1.7× chance) | 1.22 / 0.80 | 15.2 |

(speaker n_test=80, ASR n_test 23–86, inverter n_test=90. Frozen encoders are
single-fold retrains of the Phase-2/3 operating points; artifact
`results/us8k_probes_20260705_011207/`, encoders in `results/probe_encoders/`.)

### Findings — leakage is real but content-specific

1. **Raw acoustic content leaks heavily.** The learned inverter reconstructs the
   clean-speech mel far better than the silence floor (LSD ~15–17 dB vs 75.5).
   The reconstructions (`docs/figures/probe_inversion_1000bps.png`) recover the
   **coarse speech envelope / syllable rhythm**, but not fine spectral detail.
2. **Linguistic content does not leak** — the from-scratch CTC ASR probe is at or
   above the ~1.0 WER ceiling at every point (it recovers no words). Either the
   code genuinely destroys phonetic detail or a *pretrained* recogniser (a
   stronger, un-run probe) would do better; the from-scratch attacker found none.
3. **Speaker identity leaks modestly** — top-1 is 1.7–2.5× chance at λ=0
   (0.087–0.125 vs 0.05). Small but consistently above chance, and above the
   Phase-3 adversary's floor. (n_test=80, so ±~0.03; treat the ordering across
   bitrates as within noise.)

### Answering the Phase-3 ambiguity, and RQ3

- **The training adversary was too weak, not "bitrate destroyed speech".** The
  stronger probes extract what the training adversary could not: the speech
  envelope (inverter) and modest speaker identity — from the same 1000 b/s code
  where the Phase-3 adversary sat at the floor. But the picture is **mixed by
  content type**: envelope + coarse identity survive; words do not (by the CTC
  probe). It is genuinely both interpretations, split by content.
- **RQ3 — does adversarial λ help beyond bitrate?** At 1000 b/s, λ=2 roughly
  **halves speaker top-1 (0.112 → 0.062, toward chance)** at no utility cost
  (0.804 → 0.807) — so the adversary reduces *identity* leakage beyond what
  bitrate alone does. It does **not** reduce acoustic-envelope leakage (inverter
  LSD 16.1 → 16.4, unchanged) or ASR (already failed). So the λ objective buys
  some speaker privacy specifically, essentially for free, but leaves the coarse
  acoustic envelope intact.

### Caveats

These are lower bounds from *these* probes. ASR is a from-scratch CTC (a
pretrained-recogniser attacker is the stronger, un-run probe); PESQ/STOI were not
computed (packages absent + they need waveform reconstruction), so the inverter
is reported by LSD/MSE vs a silence floor; speaker n_test is small. None of this
inflates privacy — a stronger probe can only raise the leakage estimates.

### Reproduce

```
python -u scripts/train_probe_encoders.py --wav-cache data/processed/wavcache
python -u scripts/run_probes.py
```

The first trains the four frozen encoders (250/λ0, 1000/λ0, 1000/λ2, 16000/λ0,
single held-out fold) and writes a manifest; the second freezes each, builds the
leak-guarded dev-clean splits, extracts codes, and trains the three probes. The
speech-overlay robustness curve, ESC-50 and the utility-vs-leakage Pareto
synthesis are Phase 4b.

## UrbanSound8K — robustness, cross-dataset transfer, and the frontier (Phase 4b)

### A. Test-time SNR robustness (the "loud argument" condition)

At the 1000 bits/s knee (λ=0 and λ=2), sweeping the test-time speech-to-scene SNR
(same leak guards as 4a: dev-clean probe speakers, held-out fold-10 scenes).
Figure `docs/figures/robustness_vs_snr.png`.

| SNR dB | util F1 (λ0/λ2) | speaker top-1 (λ0/λ2) | ASR WER | inverter LSD (λ0/λ2) |
|--------|-----------------|-----------------------|---------|----------------------|
| −10 | 0.816 / 0.817 | 0.100 / 0.087 | ~1.1 | 17.1 / 17.4 |
| −5  | 0.827 / 0.828 | 0.138 / 0.087 | ~1.1 | 17.0 / 16.9 |
| 0   | 0.812 / 0.821 | 0.112 / 0.087 | ~1.1 | 16.6 / 16.6 |
| +5  | 0.806 / 0.816 | 0.125 / 0.062 | ~1.1 | 15.8 / 16.0 |
| +10 | 0.790 / 0.796 | 0.087 / 0.112 | ~1.1 | 15.4 / 15.7 |

- **Utility is robust to loud speech** — macro-F1 stays 0.79–0.83 across a 20 dB
  SNR range, dipping only slightly at +10 dB (speech as loud as +10 dB over the
  scene). Scene detection survives the loud-argument condition.
- **Louder speech leaks more acoustic content** — inverter LSD falls monotonically
  as SNR rises (17.1→15.4 dB at λ=0): the louder the speech, the better the
  envelope reconstructs. ASR fails at every SNR (WER ~1.1).
- **Speaker leakage is noisy but λ=2 ≤ λ=0** at almost every SNR — the adversary's
  identity-suppression holds across the SNR sweep, not just at one operating point.

### B. ESC-50 cross-dataset generalisation (frozen transfer)

The frozen 1000 bits/s encoder (US8K-trained, never exposed to ESC-50) with only a
light MLP head trained on the pooled code, under ESC-50's official 5-fold CV:

| encoder | ESC-50 50-way macro-F1 | vs chance (0.02) |
|---------|------------------------|------------------|
| 1000 b/s, λ=0 | **0.361 ± 0.013** | ~18× |
| 1000 b/s, λ=2 | 0.343 ± 0.025 | ~17× |

**The privacy-oriented low-bitrate representation still carries strongly
transferable scene information** — a frozen code trained for 10 US8K classes at
1000 bits/s reaches ~0.36 macro-F1 on 50 unseen ESC-50 classes (18× chance), and
adversarial λ barely dents it (0.361 → 0.343, within noise). Consistent with the
4a picture: broad acoustic-scene structure is preserved (and generalises) while
speech-specific fine detail is not.

### C. Privacy–utility–compute frontier (the headline exhibit)

Combining Phases 2–4a. Leakage is reported **per channel** (speaker / ASR /
inverter) — the result is content-specific, so collapsing it to one scalar would
mislead. Figures: `utility_vs_speaker_leakage.png` (the RQ3 exhibit),
`leakage_vs_bitrate.png` (per channel), regenerated by `make_figures.py` from the
committed `docs/figures/sweep_data.json` (the frontier assembly is folded into
`make_figures`, not a separate script).

**Recommended operating-point table** (compute: effective bits/s from Phase 2b;
encoder < 500K params, ~66K at the knee; MACs/s deferred — no profiler dep):

| operating point | utility F1 | speaker (chance .05) | ASR | acoustic (inverter LSD vs 75.5) | eff. bits/s |
|-----------------|-----------|----------------------|-----|---------------------------------|-------------|
| 250 b/s, λ=0    | 0.772 | 0.125 (2.5×) | no words | 17.2 | ~232 |
| 1000 b/s, λ=0   | 0.804 | 0.112 (2.2×) | no words | 16.1 | ~943 |
| **1000 b/s, λ=2** | **0.807** | **0.062 (1.2×)** | **no words** | 16.4 | ~918 |
| 16000 b/s, λ=0  | 0.783 | 0.087 (1.7×) | no words | 15.2 | ~7460 |

**Recommended: 1000 bits/s, λ=2** — full utility (0.807, at the flat top of the
Phase-2 curve), speaker identity pushed to ~1.2× chance (halved vs λ=0) for free,
no linguistic content, with the caveat that the coarse acoustic **envelope still
leaks** (inverter ≫ silence floor) — an honest, content-specific recommendation,
not a "private" absolute. All leakage figures are empirical **lower bounds**.

### Reproduce (Phase 4b)

```
python -u scripts/run_overlay_robustness.py --wav-cache data/processed/wavcache
python -u scripts/run_esc50_probe.py
python scripts/make_figures.py --refresh   # regenerate every figure from committed data
```
