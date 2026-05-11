"""
Download script for UrbanSound8K.

Dataset:    UrbanSound8K (10 urban sound classes, 8732 labelled excerpts,
            official 10-fold cross-validation split)
License:    Creative Commons Attribution Noncommercial 4.0 (CC BY-NC 4.0)
Source:     https://urbansounddataset.weebly.com/urbansound8k.html
Mirror:     https://zenodo.org/records/1203745
Size:       ~6 GB on disk

============================================================
WARNING: DO NOT RUN UNTIL ETHICS WAIVER IS APPROVED -- see README.md
============================================================
This script will refuse to download anything without the explicit
--i-have-ethics-approval flag, and even with the flag the download
function is not yet implemented (will raise NotImplementedError).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WARNING = (
    "WARNING: DO NOT RUN UNTIL ETHICS WAIVER IS APPROVED.\n"
    "See README.md. Re-run with --i-have-ethics-approval to proceed."
)

DATASET_NAME = "urbansound8k"


def download(target_dir: Path) -> None:
    raise NotImplementedError(
        "Implement after ethics approval -- see TODO.md"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download UrbanSound8K into data/raw/urbansound8k/"
    )
    parser.add_argument(
        "--i-have-ethics-approval",
        action="store_true",
        help="Required to acknowledge ethics waiver before any download.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Root directory under which <dataset>/ will be created.",
    )
    args = parser.parse_args()

    if not args.i_have_ethics_approval:
        print(WARNING, file=sys.stderr)
        return 2

    target_dir = args.data_root / DATASET_NAME
    target_dir.mkdir(parents=True, exist_ok=True)
    download(target_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
