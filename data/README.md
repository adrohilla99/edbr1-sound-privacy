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

Ethics approval is **APPROVED**. The download scripts under `scripts/` are
implemented and may be run, but remain gated: they refuse to run without an
explicit `--i-have-ethics-approval` flag and verify dataset integrity
(checksums where the source publishes them) before extraction. Contents of
`data/raw/` and `data/processed/` remain gitignored under all circumstances.

## On-disk size (approximate, plan accordingly)

- UrbanSound8K: ~6 GB
- LibriSpeech (test-clean + dev-clean + train-clean-100 only): ~30 GB
- ESC-50: ~600 MB

## Storage

For the local workstation and Azure ML, plan to mount this directory
from external storage; do not place it on the OS drive.
