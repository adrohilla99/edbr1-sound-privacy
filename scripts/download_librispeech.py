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
ETHICS: waiver APPROVED. This script still refuses to download anything
without the explicit --i-have-ethics-approval flag, and it hard-refuses
any LibriSpeech subset outside the three permitted above.
============================================================

Files come from OpenSLR resource 12. OpenSLR publishes an MD5 for every
archive; those constants are embedded below and every download is verified
against them. The operation is idempotent (verified archives and already
extracted subsets are skipped).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _download_common import download_file, extract_archive, require_present

WARNING = (
    "WARNING: download gated behind ethics acknowledgement.\n"
    "Re-run with --i-have-ethics-approval to proceed."
)

DATASET_NAME = "librispeech"
ALLOWED_SUBSETS = ("test-clean", "dev-clean", "train-clean-100")

OPENSLR_BASE = "https://www.openslr.org/resources/12"

# MD5 checksums as published on https://www.openslr.org/12/ . Keyed by
# subset; only the three permitted subsets are listed, by design -- there
# is deliberately no entry for train-clean-360 or train-other-500.
SUBSET_MD5 = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
}


def download(target_dir: Path, subsets: tuple[str, ...]) -> None:
    """Download and extract the permitted LibriSpeech ``subsets``."""
    # Defence in depth: argparse already restricts choices, but a direct
    # call to download() must not be able to smuggle in a forbidden subset.
    forbidden = [s for s in subsets if s not in ALLOWED_SUBSETS]
    if forbidden:
        raise ValueError(
            "Refusing to download non-permitted LibriSpeech subset(s): "
            f"{', '.join(forbidden)}. Only {', '.join(ALLOWED_SUBSETS)} are in scope."
        )

    for subset in subsets:
        md5 = SUBSET_MD5[subset]
        archive_name = f"{subset}.tar.gz"
        url = f"{OPENSLR_BASE}/{archive_name}"
        print(f"== {subset} ==")
        archive = target_dir / archive_name
        download_file(url, archive, expected_hash=md5, algorithm="md5")

        # Archives expand to LibriSpeech/<subset>/... Use the subset
        # directory as the idempotency marker.
        subset_dir = target_dir / "LibriSpeech" / subset
        extract_archive(archive, target_dir, marker=subset_dir)
        require_present([subset_dir])

    print(f"[done] LibriSpeech subsets ready: {', '.join(subsets)}")


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
