"""Decoding rules and code statistics for label maps rendered as RGB (Paper A, A2 Step 0).

Reads the tokens that rgb_map_ceiling.py saved, decodes them again with the same tokenizer, and
reports, for one configuration:
  1. the ceiling under two decoding rules:
     - nearest: every pixel takes the nearest palette colour (as rgb_map_ceiling.py);
     - region: pixels far from every palette colour (edge blends) may only take a class that a
       confidently decoded pixel holds within `radius` native pixels. This removes the spurious
       third class that the blend of two colours lands on. It uses no labels.
  2. code statistics against a second token set (the training folds, or TerraMesh LULC renders):
     distinct codes, the share of test tokens whose code never occurs in the reference set
     ("unseen"), and the share of the test code mass the reference covers.

Runs in `.venv`, one configuration per process (TITAN V: see rgb_map_ceiling.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "rgb_map_ceiling", Path(__file__).with_name("rgb_map_ceiling.py"))
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)


def decode(tok, idx: np.ndarray, device, batch: int = 16) -> np.ndarray:
    out = []
    for s in range(0, len(idx), batch):
        x = torch.from_numpy(idx[s:s + batch]).to(device).long()
        y = tok.decode(x).float().clamp(-1, 1)
        out.append(((y + 1) / 2).half().cpu().numpy())
    return np.concatenate(out)


def class_scores(rec: np.ndarray, pal: np.ndarray, native: int) -> torch.Tensor:
    """(N,3,S,S) -> per-class negative squared distance on the native grid, (N,K,n,n)."""
    pal_t = torch.tensor(pal, dtype=torch.float32)
    outs = []
    for s in range(0, len(rec), 64):
        y = torch.from_numpy(rec[s:s + 64].astype(np.float32))
        d = -((y[:, None] - pal_t[None, :, :, None, None]) ** 2).sum(2)
        if d.shape[-1] != native:
            d = F.interpolate(d, size=(native, native), mode="area")
        outs.append(d)
    return torch.cat(outs)


def snap_nearest(sc: torch.Tensor, ignore: int) -> np.ndarray:
    sc = sc.clone()
    sc[:, ignore] = -1e9
    return sc.argmax(1).numpy()


def snap_region(sc: torch.Tensor, ignore: int, tau: float, radius: int) -> np.ndarray:
    """A pixel whose nearest palette colour is within tau (RGB distance) is confident. Any other
    pixel may only take a class held by a confident pixel within `radius`; if none is near, it
    keeps its nearest colour."""
    sc = sc.clone()
    sc[:, ignore] = -1e9
    best, k1 = sc.max(1)                                   # best = -dist^2
    conf = (-best) <= tau ** 2
    K = sc.shape[1]
    onehot = F.one_hot(k1, K).permute(0, 3, 1, 2).float() * conf[:, None].float()
    near = F.max_pool2d(onehot, 2 * radius + 1, stride=1, padding=radius) > 0   # (N,K,n,n)
    allowed = near | conf[:, None]
    has_any = near.any(1, keepdim=True)
    allowed = torch.where(has_any, allowed, torch.ones_like(allowed))
    sc2 = torch.where(allowed, sc, torch.full_like(sc, -1e9))
    out = sc2.argmax(1)
    out = torch.where(conf, k1, out)
    return out.numpy()


def code_stats(test_idx: np.ndarray, ref_idx: np.ndarray) -> dict:
    t = test_idx.ravel()
    ref = np.unique(ref_idx.ravel())
    seen = np.isin(t, ref)
    ut, ct = np.unique(t, return_counts=True)
    return {
        "test_tokens": int(t.size),
        "test_distinct": int(len(ut)),
        "ref_distinct": int(len(ref)),
        "unseen_token_share": float(1 - seen.mean()),
        "unseen_code_share": float(1 - np.isin(ut, ref).mean()),
        "top100_code_mass": float(np.sort(ct)[::-1][:100].sum() / t.size),
        "top1000_code_mass": float(np.sort(ct)[::-1][:1000].sum() / t.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--palette", default="distinct")
    ap.add_argument("--dataset", choices=["pastis", "lulc"], default="pastis")
    ap.add_argument("--test-tokens", required=True, help="npz from rgb_map_ceiling.py")
    ap.add_argument("--ref-tokens", default="", help="comma-separated npz files: reference sets")
    ap.add_argument("--taus", default="0.10,0.15,0.20,0.25")
    ap.add_argument("--radii", default="2,3,4")
    ap.add_argument("--lulc-npy", default="/data/enric/reports/a2_seg/data/lulc_val600.npy")
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.test_tokens)
    idx, ids = z["indices"], z["ids"]
    if a.dataset == "pastis":
        n_classes, ignore, native = 20, 19, 128
        labels = rmc.load_pastis(list(ids))
    else:
        n_classes, ignore, native = 10, 0, 224
        labels = np.load(a.lulc_npy).astype(np.int64)[ids]
    pal = rmc.palette(a.palette, n_classes, ignore)
    res = {"args": vars(a)}

    if a.ref_tokens:
        for f in a.ref_tokens.split(","):
            ref = np.load(f)["indices"]
            res[f"codes_vs_{Path(f).stem}"] = code_stats(idx, ref)
            print(Path(f).stem, res[f"codes_vs_{Path(f).stem}"], flush=True)

    name = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in name:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {name}")
    t0 = time.time()
    tok, _ = rmc.load_tokenizer(a.tokenizer, "bfloat16", "cuda")
    rec = decode(tok, idx, "cuda")
    sc = class_scores(rec, pal, native)
    pred = snap_nearest(sc, ignore)
    r = rmc.scores(labels, pred, n_classes, ignore)
    res["nearest"] = {k: r[k] for k in ("mIoU", "pixel_acc", "boundary_acc", "interior_acc",
                                        "iou_per_class")}
    print(f"nearest: mIoU {r['mIoU']:.4f} acc {r['pixel_acc']:.4f} boundary "
          f"{r['boundary_acc']:.4f}", flush=True)
    best = None
    for tau in [float(t) for t in a.taus.split(",")]:
        for rad in [int(x) for x in a.radii.split(",")]:
            pred = snap_region(sc, ignore, tau, rad)
            r = rmc.scores(labels, pred, n_classes, ignore)
            key = f"region_tau{tau}_r{rad}"
            res[key] = {k: r[k] for k in ("mIoU", "pixel_acc", "boundary_acc", "interior_acc",
                                          "iou_per_class")}
            print(f"{key}: mIoU {r['mIoU']:.4f} acc {r['pixel_acc']:.4f} boundary "
                  f"{r['boundary_acc']:.4f}", flush=True)
            if best is None or r["mIoU"] > best[1]:
                best = (key, r["mIoU"])
    res["best_region"] = best
    res["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
