"""Centralized seeding for reproducible runs"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int | None, *, deterministic_cuda: bool = False) -> None:
    """Seed Python, NumPy, and torch (CPU + all CUDA devices).

    Args:
        seed: integer seed. If None, no-op (caller is opting out of reproducibility).
        deterministic_cuda: if True, force cuDNN/CUBLAS into deterministic mode.
            This makes runs bitwise-reproducible across launches on the same hardware
            but can cost 10-30% throughput and disables some kernels. Leave False
            for normal runs; turn on only when chasing a non-determinism bug.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # transformers.set_seed does the same three above, but we call it too so that
    # any downstream HF code that re-reads its own RNG state stays consistent.
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass

    # Make the hash seed deterministic for any code that uses set/dict iteration
    # order as a tiebreaker. Must be set before the interpreter reads it, so this
    # only takes effect for subprocesses; setting it here is belt-and-braces.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    if deterministic_cuda:
        # Documented torch knobs for full determinism. CUBLAS workspace must be
        # fixed before any CUDA op runs, so set it as an env var.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn that re-seeds NumPy and Python `random`.

    PyTorch already sets a unique torch seed per worker derived from the base
    generator. NumPy and `random` are NOT auto-seeded, so without this every
    worker gets the same RNG state — anything in the dataset's __getitem__ that
    uses np.random or random will produce identical draws across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)