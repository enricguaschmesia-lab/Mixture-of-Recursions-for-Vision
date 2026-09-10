"""Raw TerraMesh array -> tensor ready for tokenizer.encode().

The whole preprocessing contract lives here and in contract.py. Anything that
tokenizes TerraMesh must go through prepare() -- never re-implement these steps.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import contract as C


def center_crop(a: np.ndarray) -> np.ndarray:
    """(..., 264, 264) -> (..., 256, 256). Symmetric, 4 px per side."""
    o, c = C.CROP_OFF, C.CROP
    return a[..., o:o + c, o:o + c]


def fill_nan(x: np.ndarray, modality: str, stats_mean) -> tuple[np.ndarray, int]:
    """Apply the NaN policy. Returns (filled, n_nan)."""
    mask = np.isnan(x)
    n = int(mask.sum())
    if n == 0:
        return x, 0
    if C.NAN_POLICY == "mean":
        # per-channel modality mean -> becomes exactly 0 after standardization
        fill = np.asarray(stats_mean, dtype=x.dtype).reshape(-1, 1, 1)
        x = np.where(mask, np.broadcast_to(fill, x.shape), x)
    else:
        raise ValueError(f"unknown NAN_POLICY {C.NAN_POLICY}")
    return x, n


def prepare(arr: np.ndarray, modality: str, *, stats: str = "v1",
            device: str = "cuda:0", crop: int | None = None,
            reverse_bands: bool = False, standardize: bool = True,
            ) -> tuple[torch.Tensor, dict]:
    """Raw on-disk array -> (1, C, crop, crop) float32 tensor on device.

    arr:  (1, C, 264, 264) as stored (leading time dim).
    stats: 'v1' (production, D3.2) or 'yaml' (A/B control only).
    The reverse_bands / standardize / crop overrides exist ONLY to build the
    deliberately-broken controls of Step 3.6.
    """
    info = {}
    a = arr
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]                                   # drop time dim
    assert a.ndim == 3, f"expected (C,H,W), got {a.shape}"
    assert a.shape[0] == (1 if modality == "LULC" else C.N_CHANNELS[modality]), \
        f"{modality}: unexpected channel count {a.shape[0]}"

    # crop
    if crop is None or crop == C.CROP:
        a = center_crop(a)
    else:
        o = (C.NATIVE - crop) // 2
        a = a[..., o:o + crop, o:o + crop]
    info["crop"] = a.shape[-1]

    if modality == "LULC":
        # one-hot over 10 classes; NO standardization (tok_lulc mean 0 std 1)
        cls = torch.from_numpy(a.astype(np.int64))          # (1, H, W)
        assert int(cls.max()) < C.LULC_N_CLASSES, \
            f"LULC class {int(cls.max())} >= {C.LULC_N_CLASSES}"
        info["classes"] = sorted(np.unique(a).tolist())
        info["n_nan"] = 0
        x = F.one_hot(cls[0], C.LULC_N_CLASSES).permute(2, 0, 1).float()
        return x.unsqueeze(0).to(device), info

    mean = (C.V1_TOK_MEAN if stats == "v1" else C.YAML_MEAN)[modality]
    std = (C.V1_TOK_STD if stats == "v1" else C.YAML_STD)[modality]

    a = a.astype(np.float32)
    a, n_nan = fill_nan(a, modality, mean)
    info["n_nan"] = n_nan

    if reverse_bands:
        a = a[::-1].copy()

    x = torch.from_numpy(a)
    if standardize:
        m = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        s = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        x = (x - m) / s
    info["stats"] = stats if standardize else "none"
    return x.unsqueeze(0).to(device), info


def destandardize(x: torch.Tensor, modality: str, stats: str = "v1"
                  ) -> torch.Tensor:
    """Inverse of the standardize step, so errors are in physical units."""
    if modality == "LULC":
        return x
    mean = (C.V1_TOK_MEAN if stats == "v1" else C.YAML_MEAN)[modality]
    std = (C.V1_TOK_STD if stats == "v1" else C.YAML_STD)[modality]
    m = torch.tensor(mean, dtype=torch.float32, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=x.device).view(-1, 1, 1)
    return x * s + m


def flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """(B, H_q, W_q) -> (B, H_q*W_q) row-major. See contract.FLATTEN_ORDER."""
    assert tokens.ndim == 3, f"expected (B,Hq,Wq), got {tuple(tokens.shape)}"
    return tokens.reshape(tokens.shape[0], -1)
