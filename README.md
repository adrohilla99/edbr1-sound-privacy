# EDBR.1 — Detecting Sounds with Privacy

Privacy–utility–compute trade-offs of low-bitrate on-device audio encoders for
environmental sound classification, evaluated against independent, evaluation-time
speech-leakage probes.

Status: **Complete.** An end-to-end study: baseline → VQ bitrate sweep
(+ codebook-collapse fix) → adversarial gradient-reversal speech-suppression →
evaluation-time leakage probes (speaker-ID / ASR / inverter) → test-time SNR
robustness, ESC-50 cross-dataset transfer, and the privacy–utility–compute
frontier. Full numbers in [RESULTS.md](RESULTS.md); figures reproduce from
committed data via `scripts/make_figures.py`. Ethics approval: APPROVED.

> Download scripts are gated: they require the explicit `--i-have-ethics-approval`
> flag and verify dataset integrity (checksums) before use. Datasets and run
> outputs are gitignored; use is **non-commercial academic research** only.

## Headline results

- **Utility is nearly free to compress.** Once the VQ codebook is actually used
  (k-means init + EMA + dead-code revival fix a collapse that wasted ≤5.4% of
  codes), macro-F1 is flat within fold-noise of the 0.748 no-bottleneck control
  from ~250 bits/s up.
- **Adversarial λ buys privacy for free, but only for identity.** At the 1000 bits/s
  knee, λ=2 halves speaker-ID probe accuracy toward chance (0.112 → 0.062) at no
  utility cost (0.804 → 0.807).
- **Leakage is content-specific (empirical lower bounds).** The coarse acoustic
  envelope leaks heavily (inverter LSD ~16 vs 75.5 dB silence floor); linguistic
  content does not (from-scratch CTC ASR at the ~1.0 WER ceiling).
- **Recommended operating point: 1000 bits/s, λ=2.**

See [RESULTS.md](RESULTS.md) and [`docs/figures/`](docs/figures/).

## Quickstart

```bash
# 1. Clone
git clone <repo-url> detecting-sounds-with-privacy
cd detecting-sounds-with-privacy

# 2. Virtual environment (uv preferred, plain venv otherwise), Python 3.11
uv venv --python 3.11        # or: python -m venv .venv

# 3. Activate — Windows PowerShell:  .\.venv\Scripts\Activate.ps1
#              POSIX shells:         source .venv/bin/activate

# 4. Install (editable). Lightweight (scaffolding/config/tests) vs full ML stack:
pip install -e ".[dev]"
pip install -e ".[dev,ml]"

# 5. Sanity-check the environment, then run the tests
python scripts/verify_environment.py
pytest -q
```

**Testing note:** `89 passed` with the datasets present locally; on a dataset-free
clone (ml deps installed) it is `85 passed / 4 skipped` — the 4 skips are the only
real-data tests, each `skipif`-guarded. The synthetic leak-guard logic tests run
either way.

## Reproducing the figures (no datasets needed)

All figures regenerate from committed JSON — no gitignored `results/` required:

```bash
python scripts/make_figures.py        # -> docs/figures/*.png from docs/figures/sweep_data.json
# or run notebooks/figures.ipynb (a thin driver around the same script)
```

## Running the full pipeline (download → train → sweep → probe)

Requires the `ml` extra and the approved ethics waiver. Datasets/outputs are gitignored.

```bash
# --- Data (gated; UrbanSound8K MD5 from Zenodo, LibriSpeech OpenSLR MD5, ESC-50 structural) ---
python scripts/download_urbansound8k.py --i-have-ethics-approval
python scripts/download_librispeech.py  --i-have-ethics-approval   # test/dev/train-clean-100 only
python scripts/download_esc50.py        --i-have-ethics-approval

# --- Phase 2: baseline + VQ utility-vs-bitrate sweep ---
python -m edbr1.train --root data/raw/urbansound8k/UrbanSound8K --config configs/baseline_final.yaml
python scripts/run_bitrate_sweep.py --wav-cache data/processed/wavcache                  # collapsed
python scripts/run_bitrate_sweep.py --wav-cache data/processed/wavcache --anti-collapse  # fixed (Phase 2b)

# --- Phase 3: adversarial gradient-reversal lambda sweep at the 1000 b/s knee ---
python scripts/run_lambda_sweep.py --wav-cache data/processed/wavcache

# --- Phase 4a: freeze encoders, then attack the code with the three probes ---
python scripts/train_probe_encoders.py --wav-cache data/processed/wavcache   # writes results/probe_encoders/manifest.json
python scripts/run_probes.py                                                 # speaker-ID / ASR / inverter leakage table

# --- Phase 4b: robustness + cross-dataset transfer, then refresh committed figure data ---
python scripts/run_overlay_robustness.py --wav-cache data/processed/wavcache # test-time SNR sweep
python scripts/run_esc50_probe.py                                            # ESC-50 frozen transfer
python scripts/make_figures.py --refresh                                     # rebuild docs/figures/ from results/
```

## Architecture

The privacy pipeline (`model: encoder_classifier`) is **E → B → C**:

- **Encoder E** — a compact depthwise-separable (MobileNet-style) conv stack
  (< 500K params; ~48K for the control) mapping a log-mel spectrogram to a latent
  token grid — a plausible always-on front end.
- **Bottleneck B** — a VQ-VAE vector quantiser (codebook 1024, straight-through
  estimator, commitment loss), with a config-gated **anti-collapse** recipe
  (k-means data-dependent init, EMA updates, dead-code revival). Bitrate is logged
  honestly as `tokens/s · log2(codebook)`, with the perplexity-effective rate
  `tokens/s · log2(perplexity)` and codebook usage reported per fold.
- **Classifier C** — a minimal global-pool + linear head.
- **Adversary (train-time)** — a gradient-reversal layer (Ganin & Lempitsky 2015)
  + speech head that penalises speaker recoverability from the code, using a
  LibriSpeech-into-UrbanSound8K overlay stream (train-only, leak-guarded).
- **Probes (eval-time)** — independent, *stronger* speaker-ID / CTC-ASR / mel-inverter
  probes that attack the **frozen** code, with dev-clean speakers disjoint from the
  encoder's, reporting empirical **lower bounds** on leakage.

Feature parameters (16 kHz, 64 mel bands, 25 ms window, 10 ms hop) and all
hyper-parameters are config-driven — see [`src/edbr1/config.py`](src/edbr1/config.py).

## Project layout

```
.
├── README.md · RESULTS.md · TODO.md · pyproject.toml · .gitignore · .python-version
├── configs/
│   ├── features.yaml · baseline.yaml · baseline_final.yaml (canonical 0.746)
│   ├── improved_22k.yaml (rejected lever) · encoder_nobottleneck.yaml (control)
│   ├── vq/            # 6 VQ bitrate operating points (~80 b/s .. ~16 kb/s)
│   └── adv/           # adv_lambda_base.yaml (adversarial knee, lambda swept)
├── docs/
│   ├── 00_project_brief · 01_research_questions · 02_datasets
│   ├── 03_methodology · 04_evaluation_plan
│   └── figures/       # committed PNGs + sweep_data.json + baseline_ablation.json
├── scripts/
│   ├── _download_common · download_{urbansound8k,librispeech,esc50} · verify_environment
│   ├── plot_spectrograms · make_figures · extract_baseline
│   ├── run_bitrate_sweep · run_lambda_sweep                 # Phase 2 / 3 sweeps
│   ├── train_probe_encoders · run_probes                    # Phase 4a probes
│   └── run_overlay_robustness · run_esc50_probe             # Phase 4b
├── src/edbr1/
│   ├── config.py · bitrate.py · evaluate.py · train.py
│   ├── data/         # urbansound8k, esc50, librispeech, overlay, augment loaders
│   ├── features/     # log-mel extractor
│   ├── models/       # cnn, encoder, bottleneck (VQ+anti-collapse), classifier,
│   │                 #   encoder_classifier, adversary (GRL + head)
│   ├── probes/       # splits (leak guards), frozen, models, metrics, train
│   └── utils/        # seeding
├── tests/            # 14 files incl. leak-guard, frozen-invariance, metric tests
├── notebooks/        # figures.ipynb (reproduces figures from committed data)
├── data/ · results/  # gitignored (raw/processed data; run outputs)
```

## Documentation

- [Results log](RESULTS.md)
- [Project brief](docs/00_project_brief.md) · [Research questions](docs/01_research_questions.md)
- [Datasets & licences](docs/02_datasets.md) · [Methodology](docs/03_methodology.md) ·
  [Evaluation plan](docs/04_evaluation_plan.md)

## Licence & data use

Academic, **non-commercial** research only. Dataset licences (see
[docs/02_datasets.md](docs/02_datasets.md)): UrbanSound8K and ESC-50 are
**CC BY-NC 4.0** (non-commercial); LibriSpeech is CC BY 4.0. The datasets are not
redistributed here (gitignored); download them via the gated scripts and cite the
original authors.
