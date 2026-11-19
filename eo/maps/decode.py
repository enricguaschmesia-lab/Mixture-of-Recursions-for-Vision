"""Decoded maps -> class scores -> class maps: the decoding rules of A2 Step 1.

Class scores are computed at the tokenizer's resolution and pooled to the native grid by `to_native`:
each native pixel takes the mean over exactly the tokenizer-resolution pixels that the nearest-neighbour
render (`palette.render`) copied it to, the adjoint of the render. Step 0 area-resampled instead
(`F.interpolate(mode="area")`, adaptive pooling over whole-pixel bins): where the native grid is not a
whole multiple of the tokenizer's (PASTIS at 224 = x1.75; LULC and CloudSen12 crops at 256 = x8/7) those
bins straddle the render's pixels and mix in a neighbour's colour, so even a perfect tokenizer read 0.813
on PASTIS fold 4 at 224 (and 0.948 on LULC at 256). At whole multiples the two poolings are identical, so
every Step 0 reading at 128 -> 256 (PASTIS) and 224 -> 224 (LULC) stands. `to_native_area` keeps Step 0's.
The scores:
  - RGB output: the negative squared RGB distance to each class's colour;
  - palette logits (32 universal colours): log-softmax over the 32, read at each class's colour.
The ignored class is never predicted.

Rules (a rule is a dict, e.g. {"rule": "region", "tau": 0.15, "radius": 4}):
  nearest   argmax of the pooled class scores (the notebook's snap; for logits, the plain argmax). For
            RGB, the argmax of pooled -d^2 is the nearest colour to the mean RGB of the native pixel's
            rendered pixels.
  vote      every pixel snaps to its nearest class at the tokenizer's resolution first, and a native
            pixel takes the class most of its rendered pixels snapped to (ties broken by the pooled
            scores).
  region    Step 0's `snap_region`, unchanged: a pixel whose nearest colour is within tau (RGB
            distance) is confident; any other pixel may only take a class that a confident pixel holds
            within `radius` native pixels; with none near, it keeps its nearest class. RGB only. It
            uses no labels.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

NEG = -1e9


def rgb_scores(rgb: torch.Tensor, class_rgb: np.ndarray) -> torch.Tensor:
    """(B, 3, S, S) in [0, 1] -> (B, K, S, S): -||rgb - colour_k||^2 per class."""
    pal = torch.as_tensor(np.asarray(class_rgb), dtype=torch.float32, device=rgb.device)
    return -((rgb.float()[:, None] - pal[None, :, :, None, None]) ** 2).sum(2)


def logit_scores(logits: torch.Tensor, assign) -> torch.Tensor:
    """(B, 32, S, S) palette logits -> (B, K, S, S): log-probability of each class's colour."""
    lp = F.log_softmax(logits.float(), dim=1)
    return lp[:, torch.as_tensor(list(assign), device=logits.device)]


@lru_cache(maxsize=None)
def _pool_matrix(native: int, size: int, device: str) -> torch.Tensor:
    """(native, size): row i averages the render's pixels whose nearest-neighbour source is i. The
    source index is read off F.interpolate itself, so it is exactly palette.render's mapping."""
    ramp = torch.arange(native * native, dtype=torch.float32).view(1, 1, native, native)
    src = F.interpolate(ramp, size=(size, size), mode="nearest")[0, 0]
    rows = (src[:, 0] // native).long()
    cols = (src[0, :] % native).long()
    assert torch.equal(rows, cols), "the render's mapping is not the same on both axes"
    A = torch.zeros(native, size)
    A[rows, torch.arange(size)] = 1.0
    cnt = A.sum(1, keepdim=True)
    assert bool((cnt > 0).all()), "a native pixel has no rendered pixel (size < native)"
    return (A / cnt).to(device)


def to_native(sc: torch.Tensor, native: int) -> torch.Tensor:
    """(B, K, S, S) at the tokenizer's resolution -> (B, K, native, native): the mean over each native
    pixel's rendered pixels (the adjoint of the nearest-neighbour render)."""
    if sc.shape[-1] == native and sc.shape[-2] == native:
        return sc
    A = _pool_matrix(native, sc.shape[-1], str(sc.device))
    return torch.matmul(torch.matmul(A, sc), A.t())


def to_native_area(sc: torch.Tensor, native: int) -> torch.Tensor:
    """Step 0's pooling (adaptive average pooling), kept for the record and the gate's comparison."""
    if sc.shape[-1] == native and sc.shape[-2] == native:
        return sc
    return F.interpolate(sc, size=(native, native), mode="area")


def mask_ignored(sc: torch.Tensor, ignore: int | None) -> torch.Tensor:
    if ignore is None:
        return sc
    sc = sc.clone()
    sc[:, ignore] = NEG
    return sc


def snap_nearest(sc: torch.Tensor, ignore: int | None) -> torch.Tensor:
    return mask_ignored(sc, ignore).argmax(1)


def snap_region(sc: torch.Tensor, ignore: int | None, tau: float, radius: int) -> torch.Tensor:
    """Step 0's region rule (`rgb_map_codes.snap_region`), on any device. sc holds -dist^2."""
    sc = mask_ignored(sc, ignore)
    best, k1 = sc.max(1)
    conf = (-best) <= tau ** 2
    K = sc.shape[1]
    onehot = F.one_hot(k1, K).permute(0, 3, 1, 2).float() * conf[:, None].float()
    near = F.max_pool2d(onehot, 2 * radius + 1, stride=1, padding=radius) > 0
    allowed = near | conf[:, None]
    has_any = near.any(1, keepdim=True)
    allowed = torch.where(has_any, allowed, torch.ones_like(allowed))
    sc2 = torch.where(allowed, sc, torch.full_like(sc, NEG))
    out = sc2.argmax(1)
    return torch.where(conf, k1, out)


def votes_at_s(sc: torch.Tensor, ignore: int | None) -> torch.Tensor:
    """Scores at the tokenizer's resolution -> one-hot of each pixel's nearest class (B, K, S, S)."""
    k = mask_ignored(sc, ignore).argmax(1)
    return F.one_hot(k, sc.shape[1]).permute(0, 3, 1, 2).to(sc.dtype)


def apply_rule(sc: torch.Tensor, ignore: int | None, rule: dict, output: str,
               votes: torch.Tensor | None = None) -> torch.Tensor:
    """sc: the area-averaged scores on the native grid; votes: the area-averaged one-hot snaps (vote)."""
    name = rule["rule"]
    if name == "nearest":
        return snap_nearest(sc, ignore)
    if name == "vote":
        if votes is None:
            raise ValueError("the vote rule needs the votes")
        s = sc.clamp(min=-3.0)
        s = s - s.amax(1, keepdim=True)                        # in [-3, 0]: a tie-break only
        return mask_ignored(votes + 1e-4 * s, ignore).argmax(1)
    if name == "region":
        if output != "rgb":
            raise ValueError("the region rule reads RGB distances; this tokenizer outputs logits")
        return snap_region(sc, ignore, float(rule["tau"]), int(rule["radius"]))
    raise ValueError(f"unknown rule {name}")


def rule_key(rule: dict) -> str:
    if rule["rule"] == "region":
        return f"region_tau{float(rule['tau']):g}_r{int(rule['radius'])}"
    return rule["rule"]


def region_grid(taus=(0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3),
                radii=(1, 2, 3, 4, 6)) -> list[dict]:
    """The sweep the rule's parameters are chosen from (wider than Step 0's 0.10-0.25 x 2-4: the
    universal palette's colours are closer than Step 0's per-dataset ones)."""
    return [{"rule": "region", "tau": t, "radius": r} for t in taus for r in radii]
