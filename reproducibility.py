"""Fixed global RNG and deterministic backends for repeatable runs."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU + CUDA) and disable non-deterministic cuDNN heuristics.

    Call once at process start, before constructing ``nn.Module`` or ``DataLoader`` with shuffle.
    For CUDA, also set env ``CUBLAS_WORKSPACE_CONFIG`` (see PyTorch deterministic ops docs).
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # Helps some CUDA ops pick a deterministic algorithm when available (PyTorch 1.8+).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
