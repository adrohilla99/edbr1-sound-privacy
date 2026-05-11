"""
Download script for LibriSpeech.

Dataset:    LibriSpeech ASR corpus (read English speech, derived from
            LibriVox audiobooks). For this project ONLY the following
            subsets are in scope:
              - test-clean
              - dev-clean
              - train-clean-100
            Do NOT download train-other-500 or train-clean-360.
License:    Creative Commons Attribution 4.0 (CC BY 4.0)
Source:     https://www.openslr.org/12
Size:       ~30 GB on disk for the three permitted subsets combined.

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

DATASET_NAME = "librispeech"
ALLOWED_SUBSETS = ("test-clean", "dev-clean", "train-clean-100")


def download(target_dir: Path, subsets: tuple[str, ...]) -> None:
    raise NotImplementedError(
        "Implement after ethics approval -- see TODO.md"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download permitted LibriSpeech subsets into "
        "data/raw/librispeech/",
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
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(ALLOWED_SUBSETS),
        choices=list(ALLOWED_SUBSETS),
        help="Which LibriSpeech subsets to fetch. Restricted by policy.",
    )
    args = parser.parse_args()

    if not args.i_have_ethics_approval:
        print(WARNING, file=sys.stderr)
        return 2

    target_dir = args.data_root / DATASET_NAME
    target_dir.mkdir(parents=True, exist_ok=True)
    download(target_dir, tuple(args.subsets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
