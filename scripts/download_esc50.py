"""
Download script for ESC-50.

Dataset:    ESC-50 (2000 environmental audio clips of 5s each, 50 classes,
            5-fold cross-validation)
License:    Creative Commons Attribution Non-Commercial 4.0 (CC BY-NC 4.0)
            for the audio, with some clips under different attributions;
            see the upstream LICENSE file before publishing results.
Source:     https://github.com/karolpiczak/ESC-50
Size:       ~600 MB on disk

============================================================
ETHICS: waiver APPROVED. This script still refuses to download anything
without the explicit --i-have-ethics-approval flag.
============================================================

ASSUMPTION: the upstream GitHub repository does not publish a stable
checksum for its archive (a GitHub-generated zip is not byte-stable over
time), so integrity is verified structurally instead -- after extraction
we assert the expected 2000 wav clips and the metadata CSV are present. If
the project later adopts the Zenodo/Harvard mirror, swap in an MD5 check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _download_common import count_files, download_file, extract_archive, require_present

WARNING = (
    "WARNING: download gated behind ethics acknowledgement.\n"
    "Re-run with --i-have-ethics-approval to proceed."
)

DATASET_NAME = "esc50"

# Pinned to the master branch archive of the upstream repository.
ARCHIVE_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
ARCHIVE_NAME = "ESC-50-master.zip"
EXPECTED_CLIPS = 2000


def download(target_dir: Path) -> None:
    """Download and extract ESC-50 into ``target_dir``."""
    archive = target_dir / ARCHIVE_NAME
    # No source-published checksum (see module docstring); structural check below.
    download_file(ARCHIVE_URL, archive, expected_hash=None)

    extracted_root = target_dir / "ESC-50-master"
    meta_csv = extracted_root / "meta" / "esc50.csv"
    audio_dir = extracted_root / "audio"
    extract_archive(archive, target_dir, marker=meta_csv)

    require_present([meta_csv, audio_dir])
    n_clips = count_files(audio_dir, ".wav")
    if n_clips != EXPECTED_CLIPS:
        raise RuntimeError(
            f"ESC-50 integrity check failed: expected {EXPECTED_CLIPS} wav clips, "
            f"found {n_clips} under {audio_dir}."
        )
    print(f"[done] ESC-50 ready under {extracted_root} ({n_clips} clips verified)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ESC-50 into data/raw/esc50/"
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
