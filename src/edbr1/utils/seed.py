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
