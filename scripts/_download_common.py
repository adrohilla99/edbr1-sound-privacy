"""
Shared helpers for the EDBR.1 dataset download scripts.

Standard library only (plus tqdm for progress bars), to match the
convention that download scripts pull in no heavy dependencies. This
module is imported by the sibling ``download_*.py`` scripts; because each
script is launched as ``python scripts/download_xxx.py``, the ``scripts/``
directory is on ``sys.path`` and a plain ``import _download_common`` works.

Design goals:
  - Idempotent: a file that already exists and passes its checksum is not
    re-downloaded; an archive whose extracted marker already exists is not
    re-extracted.
  - Verifiable: every download is checked against a hash where the source
    publishes one.
  - Resumable-friendly: a partial download lands in ``<name>.part`` and is
    only renamed into place once the full byte stream has been written, so
    an interrupted run never leaves a truncated file masquerading as good.
"""
from __future__ import annotations

import hashlib
import tarfile
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

from tqdm import tqdm

# Be a polite, identifiable client. Some mirrors reject the default
# urllib User-Agent.
_USER_AGENT = "edbr1-dissertation-downloader/0.1 (+academic research)"
_CHUNK = 1 << 16  # 64 KiB


def _opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", _USER_AGENT)]
    return opener


def hash_file(path: Path, algorithm: str = "md5") -> str:
    """Return the hex digest of ``path`` using ``algorithm`` (md5/sha256/...)."""
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, expected_hash: str, algorithm: str = "md5") -> bool:
    """True iff ``path`` exists and its digest matches ``expected_hash``."""
    if not path.is_file():
        return False
    actual = hash_file(path, algorithm)
    return actual.lower() == expected_hash.lower()


def download_file(
    url: str,
    dest: Path,
    *,
    expected_hash: str | None = None,
    algorithm: str = "md5",
) -> Path:
    """Stream ``url`` to ``dest`` with a tqdm progress bar.

    Idempotent: if ``dest`` already exists and (when given) matches
    ``expected_hash``, the download is skipped. On a fresh download the
    bytes are written to ``dest.with_suffix(dest.suffix + ".part")`` and
    renamed into place only after the stream completes and verifies.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():
        if expected_hash is None:
            print(f"  [skip] {dest.name} already present (no checksum to verify)")
            return dest
        if verify_file(dest, expected_hash, algorithm):
            print(f"  [skip] {dest.name} already present and verified")
            return dest
        print(f"  [warn] {dest.name} present but checksum mismatch; re-downloading")
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()

    opener = _opener()
    with opener.open(url) as response:  # noqa: S310 - trusted https dataset URLs
        total = int(response.headers.get("Content-Length", 0)) or None
        with (
            part.open("wb") as out,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {dest.name}",
            ) as bar,
        ):
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                out.write(chunk)
                bar.update(len(chunk))

    if expected_hash is not None and not verify_file(part, expected_hash, algorithm):
        actual = hash_file(part, algorithm)
        part.unlink()
        raise RuntimeError(
            f"Checksum mismatch for {dest.name}: expected {algorithm}="
            f"{expected_hash}, got {actual}. Download discarded."
        )

    part.replace(dest)
    return dest


def extract_archive(archive: Path, target_dir: Path, *, marker: Path) -> None:
    """Extract ``archive`` into ``target_dir`` unless ``marker`` already exists.

    ``marker`` is a path that should exist after a successful extraction
    (e.g. the dataset's metadata CSV); its presence makes extraction
    idempotent. Supports ``.tar.gz``/``.tgz``/``.tar`` and ``.zip``.
    """
    if marker.exists():
        print(f"  [skip] already extracted ({marker.relative_to(target_dir.parent)})")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    print(f"  extracting {archive.name} -> {target_dir} ...")
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            _safe_extract_tar(tf, target_dir)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, target_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive.name}")

    if not marker.exists():
        raise RuntimeError(
            f"Extraction of {archive.name} finished but expected marker "
            f"{marker} is missing. The archive layout may have changed."
        )
    print(f"  [ok] extracted ({marker.name} present)")


def _is_within(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    try:
        target.resolve().relative_to(directory)
    except ValueError:
        return False
    return True


def _safe_extract_tar(tf: tarfile.TarFile, target_dir: Path) -> None:
    """Extract guarding against path-traversal (CVE-2007-4559) entries."""
    members = tf.getmembers()
    for member in members:
        member_path = target_dir / member.name
        if not _is_within(target_dir, member_path):
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    for member in tqdm(members, desc="  members", unit="file"):
        tf.extract(member, target_dir)


def _safe_extract_zip(zf: zipfile.ZipFile, target_dir: Path) -> None:
    names = zf.namelist()
    for name in names:
        member_path = target_dir / name
        if not _is_within(target_dir, member_path):
            raise RuntimeError(f"Unsafe path in archive: {name}")
    for name in tqdm(names, desc="  members", unit="file"):
        zf.extract(name, target_dir)


def count_files(root: Path, suffix: str) -> int:
    """Count files under ``root`` (recursively) ending with ``suffix``."""
    suffix = suffix.lower()
    return sum(1 for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(suffix))


def require_present(paths: Iterable[Path]) -> None:
    """Raise if any of ``paths`` is missing (post-extraction structural check)."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError("Expected files missing after download:\n  " + "\n  ".join(missing))
