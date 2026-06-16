"""
Download script for UrbanSound8K.

Dataset:    UrbanSound8K (10 urban sound classes, 8732 labelled excerpts,
            official 10-fold cross-validation split)
License:    Creative Commons Attribution Noncommercial 4.0 (CC BY-NC 4.0)
Source:     https://urbansounddataset.weebly.com/urbansound8k.html
Mirror:     https://zenodo.org/records/1203745
Size:       ~6 GB on disk

============================================================
ETHICS: waiver APPROVED. This script still refuses to download anything
without the explicit --i-have-ethics-approval flag.
============================================================

The download is fetched from the Zenodo mirror (record 1203745). Zenodo
publishes per-file MD5 checksums via its REST API, so the archive is
verified against the source-of-truth hash before extraction. The whole
operation is idempotent: an already-downloaded-and-verified archive is not
re-fetched, and an already-extracted tree is not re-extracted.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from _download_common import download_file, extract_archive, require_present

WARNING = (
    "WARNING: download gated behind ethics acknowledgement.\n"
    "Re-run with --i-have-ethics-approval to proceed."
)

DATASET_NAME = "urbansound8k"

# Zenodo mirror of UrbanSound8K. The API record lists the downloadable
# files together with their MD5 checksums, which we use for verification.
ZENODO_RECORD_ID = "1203745"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ARCHIVE_NAME = "UrbanSound8K.tar.gz"
_USER_AGENT = "edbr1-dissertation-downloader/0.1 (+academic research)"


def _fetch_zenodo_file_record(archive_name: str) -> tuple[str, str]:
    """Return ``(download_url, md5_hex)`` for ``archive_name`` from Zenodo.

    Querying the API rather than hard-coding the URL means we always pull
    the file that Zenodo currently serves and verify it against the
    checksum Zenodo itself reports.
    """
    req = urllib.request.Request(ZENODO_API, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted https
        record = json.load(resp)

    for entry in record.get("files", []):
        if entry.get("key") == archive_name:
            url = entry["links"]["self"]
            checksum = entry["checksum"]  # e.g. "md5:9aa69802bbf37fb986694eb4..."
            algorithm, _, digest = checksum.partition(":")
            if algorithm != "md5":
                raise RuntimeError(f"Unexpected checksum algorithm: {algorithm}")
            return url, digest

    available = ", ".join(e.get("key", "?") for e in record.get("files", []))
    raise RuntimeError(
        f"{archive_name} not found in Zenodo record {ZENODO_RECORD_ID}. "
        f"Available files: {available}"
    )


def download(target_dir: Path) -> None:
    """Download and extract UrbanSound8K into ``target_dir``."""
    print(f"Resolving {ARCHIVE_NAME} from Zenodo record {ZENODO_RECORD_ID} ...")
    url, md5 = _fetch_zenodo_file_record(ARCHIVE_NAME)

    archive = target_dir / ARCHIVE_NAME
    download_file(url, archive, expected_hash=md5, algorithm="md5")

    # The tarball expands to UrbanSound8K/ with audio/fold{1..10}/ and
    # metadata/UrbanSound8K.csv. Use the metadata CSV as the extraction marker.
    extracted_root = target_dir / "UrbanSound8K"
    metadata_csv = extracted_root / "metadata" / "UrbanSound8K.csv"
    extract_archive(archive, target_dir, marker=metadata_csv)

    require_present(
        [metadata_csv, *(extracted_root / "audio" / f"fold{i}" for i in range(1, 11))]
    )
    print(f"[done] UrbanSound8K ready under {extracted_root}")


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
