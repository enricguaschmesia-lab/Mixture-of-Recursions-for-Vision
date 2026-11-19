"""How a map is cut into tokenizer inputs, and how the decoded class scores are put back (A2 Step 1).

PASTIS (128 px) and LULC (224 px) maps are one crop each, nearest-upsampled to the tokenizer's input
size. CloudSen12's 512-px tiles are read as GEO-Bench-2 tests them: 3 x 3 overlapping 224-px crops at
stride 144 (offsets 0, 144, 288), each the crop a generated map would cover, since the input side
(TerraMind's S2L2A tokenizer) takes 224-px crops. A 256-px map tokenizer sees each 224-px crop
upsampled to 256, as it would in the pipeline. Class scores are averaged where crops overlap and the
rule is applied on the stitched 512-px scores.

`crop_mode="native256"` is the alternative read for a 256-px tokenizer: 3 x 3 native 256-px crops at
stride 128 (stride 144 cannot hold three 256-px crops in 512 px). It is reported beside the main read,
never instead of it.
"""
from __future__ import annotations

import torch

TILE_CROP = {"crop224": 224, "native256": 256}


def offsets(tile: int, crop: int, n: int = 3) -> list[int]:
    if crop >= tile:
        return [0]
    stride = (tile - crop) // (n - 1)
    out = [i * stride for i in range(n)]
    assert out[-1] + crop == tile, (tile, crop, out)
    return out


def plan(dataset: str, native: int, crop_mode: str = "crop224") -> list[tuple[int, int, int]]:
    """[(y0, x0, size)] of the crops that cover one map."""
    if dataset != "cloudsen12":
        return [(0, 0, native)]
    c = TILE_CROP[crop_mode]
    return [(y, x, c) for y in offsets(native, c) for x in offsets(native, c)]


def cut(labels: torch.Tensor, crops: list[tuple[int, int, int]]) -> torch.Tensor:
    """(N, H, W) -> (N * len(crops), c, c), map-major."""
    return torch.stack([labels[:, y:y + c, x:x + c] for (y, x, c) in crops], 1).flatten(0, 1)


def stitch(scores: torch.Tensor, crops: list[tuple[int, int, int]], native: int) -> torch.Tensor:
    """(N * len(crops), K, c, c) class scores at crop resolution -> (N, K, native, native), averaged
    where crops overlap."""
    n_c = len(crops)
    if n_c == 1:
        return scores
    N = scores.shape[0] // n_c
    K = scores.shape[1]
    s = scores.view(N, n_c, K, *scores.shape[-2:])
    out = torch.zeros(N, K, native, native, dtype=scores.dtype, device=scores.device)
    cnt = torch.zeros(1, 1, native, native, dtype=scores.dtype, device=scores.device)
    for j, (y, x, c) in enumerate(crops):
        out[:, :, y:y + c, x:x + c] += s[:, j]
        cnt[:, :, y:y + c, x:x + c] += 1
    return out / cnt
