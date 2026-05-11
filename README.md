# EDBR.1 — Detecting Sounds with Privacy

Privacy–utility–compute trade-offs of low-bitrate on-device audio encoders for environmental sound classification, evaluated against adversarial speech-leakage probes.

Status: Day 1 scaffolding. Ethics approval: PENDING.

> **Do not run data download scripts until ethics waiver is approved.**

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

# 4. Install in editable mode with dev extras
pip install -e ".[dev]"

# 5. Sanity check the environment
python scripts/verify_environment.py

# 6. Run the smoke test
pytest -q
```

## Project layout

```
.
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version
├── TODO.md
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
├── scripts/
│   ├── download_urbansound8k.py
│   ├── download_librispeech.py
│   ├── download_esc50.py
│   └── verify_environment.py
├── src/
│   └── edbr1/
│       └── __init__.py
├── tests/
│   └── test_smoke.py
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
