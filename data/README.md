# data/

This directory is the on-disk landing zone for datasets used by the
EDBR.1 dissertation. **Its contents are gitignored** — nothing under
`data/raw/` or `data/processed/` is tracked in version control. Only this
README is committed.

## Layout

```
data/
├── raw/         # untouched downloads, one subdirectory per dataset
│   ├── urbansound8k/
│   ├── librispeech/
│   └── esc50/
├── processed/   # preprocessed artefacts (mel-spectrograms, splits, etc.)
└── README.md    # this file
```

## Status

Ethics approval is **PENDING**. No datasets may be downloaded yet. The
scripts under `scripts/` are placeholders and will refuse to run without
an explicit `--i-have-ethics-approval` flag, and even with the flag they
currently raise `NotImplementedError` until Day 2.

## On-disk size (approximate, plan accordingly)

- UrbanSound8K: ~6 GB
- LibriSpeech (test-clean + dev-clean + train-clean-100 only): ~30 GB
- ESC-50: ~600 MB

## Storage

For the local workstation and Azure ML, plan to mount this directory
from external storage; do not place it on the OS drive.
