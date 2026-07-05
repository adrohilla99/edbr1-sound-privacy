# EDBR.1 — Detecting Sounds with Privacy

Privacy–utility–compute trade-offs of low-bitrate on-device audio encoders for environmental sound classification, evaluated against adversarial speech-leakage probes.

Status: **Feature-complete (Phases 2–4b).** Baseline → VQ bitrate sweep (+
codebook-collapse fix) → adversarial GRL speech-suppression → evaluation-time
leakage probes (speaker/ASR/inverter) → SNR robustness, ESC-50 transfer, and the
privacy–utility–compute frontier. Full numbers + figures in
[RESULTS.md](RESULTS.md); reproducible figures from committed data via
`scripts/make_figures.py`. Ethics approval: APPROVED.

> Download scripts are enabled but still gated: they require the explicit
> `--i-have-ethics-approval` flag and verify dataset integrity before use.

## Quickstart

```bash
# 1. Clone
git clone <repo-url> edbr1-dissertation
cd edbr1-dissertation

# 2. Create a virtual environment (uv preferred, plain venv otherwise)
#    uv:
uv venv --python 3.11
#    or plain venv:
python -m venv .venv

# 3. Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Activate (POSIX shells)
# source .venv/bin/activate

# 4. Install in editable mode.
#    Lightweight (scaffolding, downloaders, config/split logic, tests):
pip install -e ".[dev]"
#    Full ML/audio stack (features, model, training):
pip install -e ".[dev,ml]"

# 5. Sanity check the environment
python scripts/verify_environment.py

# 6. Run the smoke test
pytest -q
```

## Running the pipeline (download → preprocess → train)

Requires the `ml` extra installed (`pip install -e ".[dev,ml]"`) and the
approved ethics waiver. Datasets and run outputs are gitignored.

```bash
# 1. Download UrbanSound8K (gated; verifies MD5 from Zenodo before use)
python scripts/download_urbansound8k.py --i-have-ethics-approval
#    LibriSpeech (only test-clean/dev-clean/train-clean-100) and ESC-50:
python scripts/download_librispeech.py --i-have-ethics-approval
python scripts/download_esc50.py --i-have-ethics-approval

# 2. Preprocess sanity check: log-mel spectrograms on fold 1 only
#    (writes a PNG under results/)
python scripts/plot_spectrograms.py \
    --root data/raw/urbansound8k/UrbanSound8K

# 3. Train the baseline under the official 10-fold CV and report macro-F1
#    (per-fold confusion matrices + results.json land in results/<run>/)
python -m edbr1.train \
    --root data/raw/urbansound8k/UrbanSound8K \
    --config configs/baseline.yaml
#    Quick smoke (2 folds, 1 epoch):
python -m edbr1.train --epochs 1 --test-folds 1 2

# 4. VQ utility-vs-bitrate sweep: the encoder->classifier control plus six
#    discrete-bottleneck operating points (~80 bits/s .. ~16 kbits/s), each
#    under the full 10-fold CV. Writes sweep.{json,csv} + bitrate_curve.png.
python scripts/run_bitrate_sweep.py \
    --root data/raw/urbansound8k/UrbanSound8K
```

The encoder->classifier model (`model: encoder_classifier`) splits the network
into an on-device depthwise-separable encoder E (< 500K params) that emits a
latent token grid, an optional VQ-VAE bottleneck B (codebook + straight-through
estimator + commitment loss), and a downstream classifier C. Bitrate is logged
honestly as `tokens_per_second * log2(codebook_size)`; codebook perplexity and
the fraction of codes used are reported per fold so codebook collapse at low
bitrate is surfaced rather than hidden.

Feature parameters (16 kHz, 64 mel bands, 25 ms window, 10 ms hop) and
training hyper-parameters are config-driven via `configs/` — see
[`src/edbr1/config.py`](src/edbr1/config.py).

## Project layout

```
.
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version
├── TODO.md
├── configs/                  # feature + training configs (YAML)
│   ├── features.yaml
│   ├── baseline.yaml         # plain CNN baseline
│   ├── baseline_final.yaml   # canonical regularised CNN baseline (0.746)
│   ├── improved_22k.yaml     # rejected 22.05 kHz lever (kept as evidence)
│   ├── encoder_nobottleneck.yaml  # encoder->classifier control (no VQ)
│   └── vq/                   # VQ bitrate operating points (one per point)
├── docs/
│   ├── 00_project_brief.md
│   ├── 01_research_questions.md
│   ├── 02_datasets.md
│   ├── 03_methodology.md
│   └── 04_evaluation_plan.md
├── data/
│   ├── raw/         # gitignored
│   ├── processed/   # gitignored
│   └── README.md
├── results/         # run outputs (gitignored)
├── scripts/
│   ├── _download_common.py   # shared download helpers (stdlib + tqdm)
│   ├── download_urbansound8k.py
│   ├── download_librispeech.py
│   ├── download_esc50.py
│   ├── plot_spectrograms.py  # fold-1 sanity check
│   ├── run_bitrate_sweep.py  # VQ utility-vs-bitrate sweep + trade-off curve
│   └── verify_environment.py
├── src/
│   └── edbr1/
│       ├── __init__.py
│       ├── config.py         # Feature/Train/Encoder/Bottleneck configs
│       ├── bitrate.py        # honest bits/second accounting for the VQ bottleneck
│       ├── data/             # UrbanSound8K loader + fold split
│       ├── features/         # log-mel extractor
│       ├── models/           # CNN baseline + encoder / VQ bottleneck / classifier
│       ├── utils/            # seeding
│       ├── evaluate.py       # macro-F1, per-class F1, confusion matrix
│       └── train.py          # 10-fold CV entry point (run_training)
├── tests/
│   ├── test_smoke.py
│   ├── test_config.py
│   ├── test_splits.py
│   ├── test_features.py
│   ├── test_augment_norm.py
│   ├── test_bitrate.py
│   └── test_bottleneck.py
└── notebooks/
```

## Documentation

- [Project brief](docs/00_project_brief.md)
- [Research questions](docs/01_research_questions.md)
- [Datasets](docs/02_datasets.md)
- [Methodology](docs/03_methodology.md)
- [Evaluation plan](docs/04_evaluation_plan.md)

## License

TBD — academic use only pending dissertation submission.
