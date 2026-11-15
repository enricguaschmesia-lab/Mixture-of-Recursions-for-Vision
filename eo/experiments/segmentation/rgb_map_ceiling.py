"""Tokenizer ceilings for label maps rendered as RGB (Paper A, A2 Step 0).

Perfect-token round trip: label map -> palette RGB -> image tokenizer encode -> decode -> snap
to the nearest palette colour -> class map, scored against the label map on its native grid.
This is the best mIoU any model writing these tokens can reach, before it makes a single error.

Also: the ceiling of the Phase-6 route (one class token per grid cell, majority class), and the
tokens themselves, saved for the code-usage and learned-decoder analyses.

Datasets
  pastis  PASTIS(-R) semantic targets, 128 px, classes 0-18 scored, 19 = void (ignored, and never
          a prediction: its colour is excluded from the snap).
  lulc    TerraMesh val LULC maps (224 px centre crops, extracted beforehand in `mor`; see
          --lulc-npy), classes 1-9 scored, 0 = no data (ignored, as void).

Runs in `.venv` (cosmos-tokenizer 1.0.0). One process per configuration, so that a TITAN V stall
costs one configuration (CLAUDE.md). Pick the card by name: CUDA_DEVICE_ORDER=PCI_BUS_ID and
--expect-gpu.

Example:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -X faulthandler \
    eo/experiments/segmentation/rgb_map_ceiling.py --tokenizer c01_di16 --size 224 \
    --palette distinct --dataset pastis --folds 5 --expect-gpu "TITAN V" \
    --out /data/enric/reports/a2_seg/tokenizer/c01_di16_224_distinct_pastis_f5.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PASTIS = Path("/data/enric/data/PASTIS/PASTIS-R")
HF = Path("/data/enric/hf/hub")
TOKENIZERS = {
    "c01_di16": ("models--nvidia--Cosmos-0.1-Tokenizer-DI16x16", 16),
    "c01_di8": ("models--nvidia--Cosmos-0.1-Tokenizer-DI8x8", 8),
    "t1_di16": ("models--nvidia--Cosmos-Tokenize1-DI16x16-360p", 16),
    "t1_di8": ("models--nvidia--Cosmos-Tokenize1-DI8x8-360p", 8),
}


# ---------------------------------------------------------------- palettes

def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0,1] (..., 3) -> CIELAB (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def _greedy(n: int, levels: int, space: str) -> np.ndarray:
    """Farthest-point colours on a levels^3 RGB grid, from black and white."""
    lv = np.linspace(0, 1, levels)
    grid = np.stack(np.meshgrid(lv, lv, lv, indexing="ij"), -1).reshape(-1, 3)
    emb = _srgb_to_lab(grid) if space == "lab" else grid
    chosen = [0, len(grid) - 1]                       # black, white
    while len(chosen) < n:
        d = ((emb[:, None] - emb[chosen][None]) ** 2).sum(-1).min(1)
        chosen.append(int(d.argmax()))
    return grid[chosen]                                # [black, white, ...]


def palette(name: str, n_classes: int, ignore: int) -> np.ndarray:
    """(n_classes, 3) in [0,1]. The ignored class (void / no data) is white, class 0 or the
    first scored class is black, as in Gianfranco's notebook."""
    if name == "pastis":                               # PASTIS's own convention (tab20)
        import matplotlib.pyplot as plt
        assert n_classes == 20
        return np.array([(0, 0, 0), *plt.get_cmap("tab20").colors[1:19], (1, 1, 1)])
    if name in ("distinct", "distinct_lab"):
        levels, space = (6, "rgb") if name == "distinct" else (11, "lab")
        g = _greedy(n_classes, levels, space)
        black, white, rest = g[0], g[1], list(g[2:])
        out, k = np.zeros((n_classes, 3)), 0
        first_scored = 0 if ignore != 0 else 1
        for c in range(n_classes):
            if c == ignore:
                out[c] = white
            elif c == first_scored:
                out[c] = black
            else:
                out[c] = rest[k]
                k += 1
        return out
    raise ValueError(name)


# ---------------------------------------------------------------- data

def pastis_ids(folds: list[int]) -> list[int]:
    meta = json.loads((PASTIS / "metadata.geojson").read_text())
    return sorted(int(f["properties"]["ID_PATCH"]) for f in meta["features"]
                  if f["properties"]["Fold"] in folds)


def load_pastis(ids: list[int]) -> np.ndarray:
    return np.stack([np.load(PASTIS / "ANNOTATIONS" / f"TARGET_{i}.npy")[0] for i in ids]
                    ).astype(np.int64)


# ---------------------------------------------------------------- metrics

def boundary(labels: np.ndarray) -> np.ndarray:
    e = np.zeros(labels.shape, bool)
    d = labels[..., 1:, :] != labels[..., :-1, :]
    e[..., 1:, :] |= d
    e[..., :-1, :] |= d
    d = labels[..., :, 1:] != labels[..., :, :-1]
    e[..., :, 1:] |= d
    e[..., :, :-1] |= d
    return e


def scores(gt: np.ndarray, pred: np.ndarray, n: int, ignore: int) -> dict:
    valid = gt != ignore
    conf = np.bincount(gt[valid] * n + pred[valid], minlength=n * n).reshape(n, n)
    tp = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
    support = conf.sum(1) > 0
    support[ignore] = False
    b = boundary(gt)
    ok = pred == gt
    return {
        "mIoU": float(np.nanmean(iou[support])),
        "pixel_acc": float(ok[valid].mean()),
        "boundary_acc": float(ok[valid & b].mean()),
        "interior_acc": float(ok[valid & ~b].mean()),
        "boundary_share": float((valid & b).sum() / valid.sum()),
        "iou_per_class": [None if np.isnan(v) else float(v) for v in iou],
        "gt_share": (conf.sum(1) / conf.sum()).tolist(),
    }


def cell_ceiling(gt: np.ndarray, g: int, n: int, ignore: int) -> np.ndarray:
    """Phase-6 route: one class per cell of a g x g grid (majority of non-ignored pixels)."""
    N, H, W = gt.shape
    ri = (np.arange(H) * g) // H
    ci = (np.arange(W) * g) // W
    cell = (ri[:, None] * g + ci[None, :])                       # (H, W)
    pred = np.empty_like(gt)
    for k in range(N):
        lab = gt[k].ravel()
        cid = cell.ravel()
        v = lab != ignore
        cnt = np.zeros((g * g, n), np.int64)
        np.add.at(cnt, (cid[v], lab[v]), 1)
        maj = cnt.argmax(1)
        maj[cnt.sum(1) == 0] = 0 if ignore != 0 else 1
        pred[k] = maj[cell]
    return pred


# ---------------------------------------------------------------- tokenizer

def load_tokenizer(name: str, dtype: str, device: str):
    from cosmos_tokenizer.image_lib import ImageTokenizer
    repo, factor = TOKENIZERS[name]
    snap_dir = next((HF / repo / "snapshots").iterdir())
    tok = ImageTokenizer(checkpoint_enc=str(snap_dir / "encoder.jit"),
                         checkpoint_dec=str(snap_dir / "decoder.jit"),
                         device=device, dtype=dtype)
    return tok, factor


def roundtrip(tok, labels: np.ndarray, pal: np.ndarray, size: int, dtype, device, batch: int):
    """labels (N,H,W) -> (indices (N,h,w) int32, decoded RGB at `size` (N,3,size,size) float16)."""
    idx_all, rec_all = [], []
    pal_t = torch.tensor(pal, dtype=torch.float32)
    for s in range(0, len(labels), batch):
        lab = torch.from_numpy(labels[s:s + batch])
        x = pal_t[lab].permute(0, 3, 1, 2)                       # (B,3,H,W) in [0,1]
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="nearest")
        x = (x * 2 - 1).to(device, dtype)
        indices, _ = tok.encode(x)
        y = tok.decode(indices).float().clamp(-1, 1)
        idx_all.append(indices.int().cpu().numpy())
        rec_all.append(((y + 1) / 2).half().cpu().numpy())
    return np.concatenate(idx_all), np.concatenate(rec_all)


def snap(rec: np.ndarray, pal: np.ndarray, native: int, ignore: int | None,
         avgpool_rgb: bool = False) -> np.ndarray:
    """Decoded RGB (N,3,S,S) -> class map on the native grid. Per-class scores (negative squared
    distance to each palette colour) are area-resampled to the native grid, then argmax; the
    ignored class is never predicted. avgpool_rgb reproduces the notebook (RGB pooled, then snap,
    void allowed)."""
    out = []
    pal_t = torch.tensor(pal, dtype=torch.float32)
    for s in range(0, len(rec), 64):
        y = torch.from_numpy(rec[s:s + 64].astype(np.float32))
        if avgpool_rgb:
            if y.shape[-1] != native:
                y = F.interpolate(y, size=(native, native), mode="area")
            d = -((y[:, None] - pal_t[None, :, :, None, None]) ** 2).sum(2)
        else:
            d = -((y[:, None] - pal_t[None, :, :, None, None]) ** 2).sum(2)   # (B,K,S,S)
            if d.shape[-1] != native:
                d = F.interpolate(d, size=(native, native), mode="area")
            if ignore is not None:
                d[:, ignore] = -1e9
        out.append(d.argmax(1).numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", choices=list(TOKENIZERS) + ["none"], required=True)
    ap.add_argument("--size", type=int, default=224, help="tokenizer input size (px)")
    ap.add_argument("--palette", default="distinct")
    ap.add_argument("--dataset", choices=["pastis", "lulc"], default="pastis")
    ap.add_argument("--folds", default="5")
    ap.add_argument("--lulc-npy", default="/data/enric/reports/a2_seg/data/lulc_val600.npy")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--repro", action="store_true",
                    help="Gianfranco's notebook: 300 random patches (seed 0), RGB avg-pooled, "
                         "snap over all 20 colours, mIoU over every class with support (void too)")
    ap.add_argument("--cells", default="14,16,28,32", help="Phase-6 grid ceilings (tokenizer none)")
    ap.add_argument("--save-tokens", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.dataset == "pastis":
        n_classes, ignore, native = 20, 19, 128
        if a.repro:
            ids = random.Random(0).sample(pastis_ids([1, 2, 3, 4, 5]), 300)
        else:
            ids = pastis_ids([int(f) for f in a.folds.split(",")])
        labels = load_pastis(ids)
    else:
        n_classes, ignore, native = 10, 0, 224
        labels = np.load(a.lulc_npy).astype(np.int64)
        ids = list(range(len(labels)))
    res = {"args": vars(a), "n_maps": len(labels)}
    t0 = time.time()

    if a.tokenizer == "none":                                   # Phase-6 class-cell ceilings
        for g in [int(c) for c in a.cells.split(",")]:
            pred = cell_ceiling(labels, g, n_classes, ignore)
            r = scores(labels, pred, n_classes, ignore)
            res[f"cells_{g}"] = {k: r[k] for k in ("mIoU", "pixel_acc", "boundary_acc",
                                                   "interior_acc")}
            res[f"cells_{g}"]["tokens"] = g * g
            print(f"cells {g}x{g} ({g * g} tokens): mIoU {r['mIoU']:.4f} acc {r['pixel_acc']:.4f}"
                  f" boundary {r['boundary_acc']:.4f}", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        return

    device = "cuda"
    name = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in name:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {name}")
    pal = palette(a.palette, n_classes, ignore)
    dists = np.sqrt(((pal[:, None] - pal[None]) ** 2).sum(-1))
    keep = [c for c in range(n_classes) if c != ignore]
    res["palette"] = pal.tolist()
    res["closest_pair_rgb"] = float(dists[np.ix_(keep, keep)][np.triu_indices(len(keep), 1)].min())

    tok, factor = load_tokenizer(a.tokenizer, a.dtype, device)
    dtype = getattr(torch, a.dtype)
    idx, rec = roundtrip(tok, labels, pal, a.size, dtype, device, a.batch)
    res["grid"] = list(idx.shape[1:])
    res["tokens_per_map"] = int(np.prod(idx.shape[1:]))
    res["distinct_codes"] = int(len(np.unique(idx)))
    res["seconds"] = time.time() - t0

    # PSNR of the decoded RGB against the rendered map at the tokenizer's size
    pal_t = torch.tensor(pal, dtype=torch.float32)
    ref = F.interpolate(pal_t[torch.from_numpy(labels)].permute(0, 3, 1, 2),
                        size=(a.size, a.size), mode="nearest").numpy()
    mse = ((rec.astype(np.float32) - ref) ** 2).mean(axis=(1, 2, 3))
    res["psnr_db"] = float((10 * np.log10(1 / np.maximum(mse, 1e-10))).mean())

    if a.repro:
        pred = snap(rec, pal, native, None, avgpool_rgb=True)
        valid_all = np.full(n_classes, True)
        conf = np.bincount(labels.ravel() * n_classes + pred.ravel(),
                           minlength=n_classes ** 2).reshape(n_classes, n_classes)
        tp = np.diag(conf)
        union = conf.sum(0) + conf.sum(1) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(union > 0, tp / union, np.nan)
        b = boundary(labels)
        ok = pred == labels
        res["repro"] = {"mIoU": float(np.nanmean(iou[conf.sum(1) > 0])),
                        "pixel_acc": float(ok.mean()), "boundary_acc": float(ok[b].mean()),
                        "interior_acc": float(ok[~b].mean())}
        print(f"REPRO {a.tokenizer} {a.size} {a.palette} {a.dtype}: {res['repro']}", flush=True)
    pred = snap(rec, pal, native, ignore)
    res["scores"] = scores(labels, pred, n_classes, ignore)
    s = res["scores"]
    print(f"{a.tokenizer} size {a.size} grid {res['grid']} palette {a.palette} {a.dtype} "
          f"{a.dataset}: mIoU {s['mIoU']:.4f} acc {s['pixel_acc']:.4f} boundary "
          f"{s['boundary_acc']:.4f} interior {s['interior_acc']:.4f} PSNR {res['psnr_db']:.2f} "
          f"codes {res['distinct_codes']} ({res['seconds']:.0f} s)", flush=True)
    if a.save_tokens:
        np.savez_compressed(a.save_tokens, indices=idx, ids=np.array(ids))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
