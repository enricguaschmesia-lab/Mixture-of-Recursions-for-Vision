"""Packed BigEarthNet rasters -> encoder-ready (B, C, 224, 224) tensors (`mor` env).

The BEN half of the tokenizer contract (PHASE6_PLAN Step 1.1/1.2, §4.2, §4.3). Every
constant comes from `eo/terramesh_tok/contract.py`; nothing is restated here.

  S2 -> the S2L2A tokenizer. 12 bands, each resampled from its NATIVE size (120 / 60 / 20 px)
        straight to 224 x 224 in ONE interpolation (never 20 m -> 10 m -> 224), reassembled
        in `contract.BAND_ORDER["S2L2A"]`, then standardized with `V1_TOK_MEAN/STD`. BEN v1
        S2 carries no BOA offset (worklog 2026-10-11), so no offset is added or removed.
  S1 -> the S1GRD tokenizer (or S1RTC, for the 1.3 comparison). VV, VH in dB, interpolated
        in dB (the space the tokenizer standardizes). Non-finite values follow the contract's
        NaN policy ('mean': filled BEFORE interpolation, so they cannot spread).

Why resampling at all: BEN patches are 120 px at 10 m and the tokenizers accept nothing below
224 (120 = 7.5 x 16, so padding does not fit the token grid). 120 -> 224 changes the GSD to
~5.4 m, a distribution the tokenizers never saw. Step 1.3 measures what that costs.

The `band_perm`, `stats` and `standardize` overrides exist ONLY for the broken controls.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from eo.terramesh_tok import contract as C
from . import layout

# band name -> (packed group, index within it)
BAND_LOC: Dict[str, Tuple[str, int]] = {
    b: (g, j) for g, (bands, _) in layout.S2_GROUPS.items() for j, b in enumerate(bands)
}
assert sorted(BAND_LOC) == sorted(C.BAND_ORDER["S2L2A"]), "S2 bands disagree with the contract"
S1_ORDER = [b.upper() for b in C.BAND_ORDER["S1GRD"]]          # ['VV', 'VH']
assert S1_ORDER == layout.S1_BANDS[0], "S1 band order disagrees with the contract"
assert C.BAND_ORDER["S1GRD"] == C.BAND_ORDER["S1RTC"]


def _resize(x: torch.Tensor, mode: str) -> torch.Tensor:
    if x.shape[-1] == C.CROP:
        return x
    kw = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    return F.interpolate(x, size=(C.CROP, C.CROP), mode=mode, **kw)


def _standardize(x: torch.Tensor, modality: str, stats: str) -> torch.Tensor:
    mean = (C.V1_TOK_MEAN if stats == "v1" else C.YAML_MEAN)[modality]
    std = (C.V1_TOK_STD if stats == "v1" else C.YAML_STD)[modality]
    m = torch.tensor(mean, dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    return (x - m) / s


def s2_physical(p10, p20, p60, device, mode: str = "bilinear") -> torch.Tensor:
    """(B,12,224,224) float32 DN, contract band order, NOT standardized."""
    groups = {"S2_10m": p10, "S2_20m": p20, "S2_60m": p60}
    up = {g: _resize(torch.as_tensor(np.asarray(a), dtype=torch.float32).to(device), mode)
          for g, a in groups.items()}
    return torch.stack([up[g][:, j] for g, j in (BAND_LOC[b] for b in C.BAND_ORDER["S2L2A"])], 1)


def s2_batch(p10, p20, p60, device, *, mode: str = "bilinear", stats: str = "v1",
             standardize: bool = True, band_perm: Optional[Sequence[int]] = None) -> torch.Tensor:
    x = s2_physical(p10, p20, p60, device, mode)
    if band_perm is not None:                       # broken control: wrong band order
        x = x[:, list(band_perm)]
    return _standardize(x, "S2L2A", stats) if standardize else x


def s1_physical(s1, device, tokenizer: str = "S1GRD", mode: str = "bilinear"
                ) -> Tuple[torch.Tensor, int]:
    """(B,2,224,224) float32 dB, non-finite filled with the tokenizer's mean first."""
    x = torch.as_tensor(np.asarray(s1), dtype=torch.float32).to(device)
    bad = ~torch.isfinite(x)
    n_bad = int(bad.sum())
    if n_bad:
        assert C.NAN_POLICY == "mean"
        fill = torch.tensor(C.V1_TOK_MEAN[tokenizer], device=device).view(1, -1, 1, 1).expand_as(x)
        x = torch.where(bad, fill, x)
    return _resize(x, mode), n_bad


def s1_batch(s1, device, *, tokenizer: str = "S1GRD", mode: str = "bilinear",
             stats: str = "v1", standardize: bool = True, reverse_bands: bool = False
             ) -> Tuple[torch.Tensor, int]:
    x, n_bad = s1_physical(s1, device, tokenizer, mode)
    if reverse_bands:                               # broken control
        x = x.flip(1)
    return (_standardize(x, tokenizer, stats) if standardize else x), n_bad


def destandardize(z: torch.Tensor, modality: str, stats: str = "v1") -> torch.Tensor:
    mean = (C.V1_TOK_MEAN if stats == "v1" else C.YAML_MEAN)[modality]
    std = (C.V1_TOK_STD if stats == "v1" else C.YAML_STD)[modality]
    m = torch.tensor(mean, dtype=torch.float32, device=z.device).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=z.device).view(1, -1, 1, 1)
    return z * s + m
