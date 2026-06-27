"""
Reproducibility helpers.

``seed_everything`` fixes the Python, NumPy and PyTorch RNGs and, when
asked, switches cuDNN into deterministic mode. Fixed seeds plus a logged
config are what make the baseline runs reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    With ``deterministic=True`` cuDNN is asked to behave deterministically
    (at a small speed cost), which keeps fold results reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: make per-worker augmentation reproducible.

    With ``num_workers > 0`` each batch is produced in a separate process, so
    augmentation no longer draws from the main-process RNG. PyTorch already
    assigns each worker a *distinct* torch seed (``base_seed + worker_id``,
    where ``base_seed`` is drawn from the main RNG when the loader's iterator is
    created), so the torch-based augmentation here is already deterministic for
    a fixed run. This hook mirrors that per-worker seed into NumPy and Python's
    ``random`` as well, so any non-torch augmentation (e.g. the optional librosa
    pitch shift / time stretch) is also reproducible and not duplicated across
    workers. ``worker_id`` is unused -- ``torch.initial_seed()`` already encodes
    it -- but kept to match the ``worker_init_fn`` signature.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
