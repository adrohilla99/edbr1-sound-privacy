"""
Safe environment sanity check for the EDBR.1 project.

Prints Python version, OS, approximate available RAM, whether any CUDA
devices are exposed via the CUDA_VISIBLE_DEVICES environment variable,
and a depth-2 tree of the project directory.

Standard library only. Does NOT import torch or any ML dependency.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def available_ram_bytes() -> int | None:
    """Best-effort, stdlib-only available RAM lookup. Returns None on failure."""
    system = platform.system()
    try:
        if system == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys)
        if system == "Linux":
            meminfo = Path("/proc/meminfo").read_text()
            for line in meminfo.splitlines():
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
            return None
        if system == "Darwin":
            # vm_stat is the simplest stdlib-free path; skip rather than parse.
            return None
    except Exception:
        return None
    return None


def gib(n: int | None) -> str:
    if n is None:
        return "unknown"
    return f"{n / (1024 ** 3):.2f} GiB"


def print_tree(root: Path, max_depth: int = 2) -> None:
    """Print a directory tree up to max_depth, skipping common noise."""
    skip = {"__pycache__", ".git", ".venv", ".uv", ".pytest_cache",
            ".ipynb_checkpoints", "node_modules"}

    def walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(
                [p for p in path.iterdir() if p.name not in skip],
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            return
        for idx, entry in enumerate(entries):
            last = idx == len(entries) - 1
            connector = "`-- " if last else "|-- "
            print(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if last else "|   "
                walk(entry, prefix + extension, depth + 1)

    print(f"{root.name}/")
    walk(root, "", 1)


def main() -> int:
    print("=" * 60)
    print("EDBR.1 environment verification")
    print("=" * 60)
    print(f"Python version : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform       : {platform.system()} {platform.release()} "
          f"({platform.machine()})")
    print(f"Available RAM  : {gib(available_ram_bytes())}")

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda is None:
        cuda_repr = "<unset>"
    elif cuda == "":
        cuda_repr = "<empty -- CUDA disabled by env>"
    else:
        cuda_repr = cuda
    print(f"CUDA_VISIBLE_DEVICES : {cuda_repr}")
    print(f"Project root   : {PROJECT_ROOT}")
    print()
    print("Project tree (depth 2):")
    print_tree(PROJECT_ROOT, max_depth=2)
    print()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
