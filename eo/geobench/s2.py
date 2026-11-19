"""The S2 input contract for GEO-Bench-2's segmentation tasks (A2 Step 2.3-2.4).

Raw S2 (DN, GEO-Bench-2's band order) -> the S2L2A tokenizer's input (B, 12, crop, crop), float32:

  1. bands to `contract.BAND_ORDER["S2L2A"]`. PASTIS ships 10 bands (no B01, B09): those two are
     imputed from the other ten (IMPUTE below, chosen in Step 2.3 on BEN and TerraMesh against the
     true bands). CloudSen12 ships the 12 bands in contract order, plus AOT and WVP, which are dropped.
  2. no-data: CloudSen12's tiles have a zero border (every band 0); those pixels are filled with the
     contract mean, i.e. 0 after standardization (contract.NAN_POLICY "mean").
  3. one bilinear interpolation to the crop (PASTIS 128 -> 224, ~5.7 m GSD; align_corners False), as
     BEN does (eo/ben/preprocess.py); CloudSen12 is cut at 224 from its 512-px tile, no resampling.
  4. standardization with V1_TOK_MEAN/STD.

The dihedral variants (k = 0..7: k % 4 quarter turns counter-clockwise, then a left-right flip if
k >= 4) are applied to the native raster BEFORE any of this, and to the label map identically, so a
variant is simply another sample.

The overrides `band_perm`, `standardize` and `scale` exist only for the broken controls of Step 2.4
(wrong band order, raw DN, x10^4).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from eo.terramesh_tok import contract as C

ORDER = C.BAND_ORDER["S2L2A"]
PASTIS_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")   # GEO-Bench-2 / PASTIS-R
CLOUDSEN12_BANDS = tuple(ORDER) + ("AOT", "WVP")     # the tortilla's 14 bands (A2 Step 2.4: L2A, not L1C)
MISSING = ("B01", "B09")
_MEAN = np.asarray(C.V1_TOK_MEAN["S2L2A"], np.float32)
_STD = np.asarray(C.V1_TOK_STD["S2L2A"], np.float32)

# ---- B01 / B09 imputation (Step 2.3) ------------------------------------------------------------
# "mean":   the contract mean, i.e. exactly 0 after standardization.
# "copy":   the nearest real band in wavelength: B01 <- B02, B09 <- B8A.
# "linear": an affine map of the ten real bands in DN, least squares on TerraMesh val S2L2A pixels
#           (cleaned; rows disjoint from every evaluation row), fitted by
#           eo/experiments/segmentation/step2_impute.py --fit. Coefficients are in LINEAR below, in
#           PASTIS_BANDS order, then the intercept.
IMPUTE_METHODS = ("mean", "copy", "linear")
# Fitted 2026-11-19 on 600,000 pixels of 300 rows: R^2 B01 0.9686, B09 0.9283; residual
# 0.167 and 0.252 in z.
LINEAR: dict[str, list[float]] | None = {
    "B01": [1.3893479962578665, -0.35326981704678106, -0.5231865638175578, 0.4831380920759904, 0.004696305027166031, 0.15692866925891577, -0.2043447654400392, -0.01948402846998007, -0.06624462867505612, 0.06041727645385238, 54.90962200299186],
    "B09": [0.2709409469929685, -0.13436361302233157, -0.0993081453739239, 0.009795655658364447, -0.01809703094495272, 0.12598697062125844, 0.04641430909330099, 0.7600937846171931, 0.12653508447249057, -0.0452597544789273, 78.37826392390734],
}
LINEAR_SOURCE = {"path": "/data/enric/reports/a2_seg/step2/impute/linear_fit.json", "sha256": "b1617a82afd1fb5c940df2aab02c48dedc52961790c7d453ce9473ce061ef720"}
# Step 2.3's choice (2026-11-19). On BEN and TerraMesh against the true bands, linear and copy tie on the
# functional test (b10_lora S2 -> BEN micro-mAP on ft_valsel: true 0.7900, linear 0.7825, copy 0.7836,
# mean 0.4800) and linear has the lower imputed-band error and 10-band round trip on both sets and the
# higher token agreement on TerraMesh (0.477 against 0.440; BEN 0.441 against 0.450). Mean filling loses
# 0.31 mAP. Evidence: /data/enric/reports/a2_seg/step2/impute/ (eval.json, ben_eval/).
CHOSEN_IMPUTATION = "linear"


def dihedral(x: torch.Tensor | np.ndarray, k: int):
    """The k-th of the 8 dihedral variants of the last two axes."""
    if not 0 <= k < 8:
        raise ValueError(k)
    if isinstance(x, np.ndarray):
        y = np.rot90(x, k % 4, axes=(-2, -1))
        return np.ascontiguousarray(y[..., ::-1] if k >= 4 else y)
    y = torch.rot90(x, k % 4, dims=(-2, -1))
    return y.flip(-1) if k >= 4 else y


def impute(x10: torch.Tensor, method: str, linear: Optional[dict] = None) -> torch.Tensor:
    """x10 (B, 10, H, W) DN in PASTIS_BANDS order -> (B, 2, H, W) DN for B01, B09."""
    if method == "mean":
        m = torch.tensor(_MEAN[[ORDER.index(b) for b in MISSING]], device=x10.device)
        return m.view(1, 2, 1, 1).expand(x10.shape[0], 2, *x10.shape[-2:]).clone()
    if method == "copy":
        return torch.stack([x10[:, PASTIS_BANDS.index("B02")], x10[:, PASTIS_BANDS.index("B8A")]], 1)
    if method == "linear":
        coef = linear if linear is not None else LINEAR
        if coef is None:
            raise RuntimeError("no linear imputation fitted yet (step2_impute.py --fit)")
        out = []
        for b in MISSING:
            w = torch.tensor(coef[b], dtype=torch.float32, device=x10.device)
            out.append(torch.einsum("bchw,c->bhw", x10.float(), w[:10]) + w[10])
        return torch.stack(out, 1)
    raise ValueError(method)


def assemble_pastis(x10: torch.Tensor, method: str, linear: Optional[dict] = None) -> torch.Tensor:
    """(B, 10, H, W) DN in PASTIS_BANDS order -> (B, 12, H, W) DN in contract order, B01/B09 imputed."""
    assert x10.shape[1] == 10, x10.shape
    x10 = x10.float()
    imp = impute(x10, method, linear)
    chans = []
    for b in ORDER:
        chans.append(imp[:, MISSING.index(b)] if b in MISSING else x10[:, PASTIS_BANDS.index(b)])
    return torch.stack(chans, 1)


def resample(x: torch.Tensor, size: int = C.CROP) -> torch.Tensor:
    """One bilinear interpolation to size x size (align_corners False); a no-op at that size."""
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def standardize(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(_MEAN, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(_STD, device=x.device).view(1, -1, 1, 1)
    return (x - m) / s


def destandardize(z: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(_MEAN, device=z.device).view(1, -1, 1, 1)
    s = torch.tensor(_STD, device=z.device).view(1, -1, 1, 1)
    return z * s + m


def fill_nodata(x12: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pixels where every band is 0 (CloudSen12's border) -> the contract mean. Returns (x, n_pixels)."""
    bad = (x12 == 0).all(1, keepdim=True)
    n = int(bad.sum())
    if n:
        m = torch.tensor(_MEAN, device=x12.device).view(1, -1, 1, 1).expand_as(x12)
        x12 = torch.where(bad.expand_as(x12), m, x12)
    return x12, n


def _finish(x12: torch.Tensor, standardize_: bool, band_perm, scale: float) -> torch.Tensor:
    if scale != 1.0:                                   # broken control: wrong scale
        x12 = x12 * scale
    if band_perm is not None:                          # broken control: wrong band order
        x12 = x12[:, list(band_perm)]
    return standardize(x12) if standardize_ else x12


def pastis_input(x10_dn, method: str, device: str = "cuda", *, linear: Optional[dict] = None,
                 standardize_: bool = True, band_perm: Optional[Sequence[int]] = None,
                 scale: float = 1.0) -> torch.Tensor:
    """PASTIS frames (B, 10, 128, 128) DN -> (B, 12, 224, 224) encoder input."""
    x = torch.as_tensor(np.asarray(x10_dn), dtype=torch.float32, device=device)
    x12 = assemble_pastis(x, method, linear)          # impute at native resolution, then one resample
    return _finish(resample(x12), standardize_, band_perm, scale)


def pastis_physical(x10_dn, method: str, device: str = "cuda", linear: Optional[dict] = None) -> torch.Tensor:
    """The same, in DN, not standardized: the round trip's reference."""
    x = torch.as_tensor(np.asarray(x10_dn), dtype=torch.float32, device=device)
    return resample(assemble_pastis(x, method, linear))


def cloudsen12_physical(x14_dn, device: str = "cuda") -> tuple[torch.Tensor, int]:
    """CloudSen12 crops (B, 14, S, S) DN -> (B, 12, S, S) DN in contract order, border filled."""
    x = torch.as_tensor(np.asarray(x14_dn), dtype=torch.float32, device=device)
    assert x.shape[1] == len(CLOUDSEN12_BANDS), x.shape
    return fill_nodata(x[:, :12])


def cloudsen12_input(x14_dn, device: str = "cuda", *, standardize_: bool = True,
                     band_perm: Optional[Sequence[int]] = None, scale: float = 1.0) -> tuple[torch.Tensor, int]:
    """CloudSen12 crops (B, 14, 224, 224) DN -> (B, 12, 224, 224) encoder input, and the no-data count."""
    x12, n = cloudsen12_physical(x14_dn, device)
    assert x12.shape[-1] == C.CROP, x12.shape
    return _finish(x12, standardize_, band_perm, scale), n
