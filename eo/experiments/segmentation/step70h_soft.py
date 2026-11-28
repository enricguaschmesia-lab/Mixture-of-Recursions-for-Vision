#!/usr/bin/env python
"""A2 Step 7.0h: per-sample class probabilities on the native grid, summed per patch, for soft voting
(docs/plans/a2_segmentation.md, Step 7, "7.0h"; `a2seg` env, the GTX Titan X). Decoding only.

    python eo/experiments/segmentation/step70h_soft.py --gen <gen dir> --out <dir> [--root <seg root>] [--votes <dir>]

Each sample is decoded by `harness.class_scores`' own path (the frozen arm's decoder, palette logits read at the task's
class colours as log-softmax over the 32 universal colours) and pooled to the native grid by `decode.to_native`, the
render's exact reverse, in two ways:
  log   the mean of the log-probabilities over the native pixel's 1, 2 or 4 canvas pixels (today's), then exp;
  prob  the mean of the probabilities.
Per pixel the 19 scored classes are renormalized to sum to 1 (void zeroed); a TTA view's map is brought back to k = 0
(the inverse of eo/geobench/s2.dihedral) before it is summed into its patch. Written per patch, float32:
  <out>/<subset>_<pool>.npy   (patches, 20, H, W): the SUM over the subset's samples (k8: the k = 0 view's samples;
                              tta64: every view's samples, TTA runs only)
  <out>/soft.json             counts, seconds, and the validity check: each sample's argmax of the log-pooled scores
                              (void masked, in the view's frame) against its stored class map in --votes (7.0e's decode)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def to_k0_t(y, k: int):
    """Torch twin of step70e_votes.to_k0 on the last two axes."""
    import torch
    z = torch.flip(y, dims=(-1,)) if k >= 4 else y
    return torch.rot90(z, -(k % 4), dims=(-2, -1))


def renorm(p, ignore: int):
    """(.., C, H, W) non-negative scores -> per-pixel distributions over the scored classes (the ignored one zeroed)."""
    p = p.clone()
    p[:, ignore] = 0.0
    return p / p.sum(1, keepdim=True).clamp_min(1e-30)


def pooled_pair(tok, idx, task: str, arm: str):
    """codes (N, side, side) -> (log-pooled, prob-pooled) per-sample class distributions (N, C, H, W), and the
    log-pooled scores themselves (for the argmax check), by harness.class_scores' own steps."""
    import torch
    from eo.maps import decode as D, harness as H, palette as P, tiling
    from eo.maps.tokenizers import run_fixed
    t = P.TASKS[task]
    _, assign, _ = H.task_palette(task, P.arm_palette(arm))
    crops = tiling.plan(task, t.native, "crop224")
    out = run_fixed(tok.decode, idx, tok.batch)
    if tok.output != "logits":
        raise ValueError(f"arm {arm} outputs {tok.output}; 7.0h reads palette logits")
    sc_s = D.logit_scores(out, assign)                                              # (N, C, S, S) log-probs
    lg = tiling.stitch(D.to_native(sc_s, crops[0][2]), crops, t.native)             # today's class scores
    pr = tiling.stitch(D.to_native(sc_s.exp(), crops[0][2]), crops, t.native)
    return renorm(lg.exp(), t.ignore), renorm(pr, t.ignore), lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--votes", default=None, help="7.0e's decode of the same codes (classmaps.npy) for the check")
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    import torch
    from eo.data import map_vocab as MV
    from eo.eval import seg as SEG
    from eo.maps import decode as D, frozen, palette as P
    import step70e_votes as VT
    dev = "cuda"
    if a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    out = Path(a.out)
    if (out / "soft.json").exists():
        print(f"skip {out}: done already", flush=True)
        return
    g = Path(a.gen)
    rec = json.loads((g / "record.json").read_text())
    if rec["split"] != "val" or rec["task"] != "pastis":
        raise PermissionError(f"{g}: {rec['task']} {rec['split']}")
    views = VT._views_for(rec, a.root)
    codes = np.load(g / "codes.npy")
    V, K, G = codes.shape
    arm = rec["arm"]
    side = MV.spec(arm)[2]
    t = P.TASKS["pastis"]
    ks = views.k.to_numpy()
    tta = bool((ks != 0).any())
    if tta and not (ks.reshape(-1, 8) == np.arange(8)).all():
        raise RuntimeError("TTA views are not patch-major k = 0..7")
    per_patch = 8 if tta else 1
    n_p = V // per_patch
    stored = np.load(Path(a.votes) / "classmaps.npy", mmap_mode="r") if a.votes else None
    if stored is not None and stored.shape[:2] != (V, K):
        raise RuntimeError(f"stored class maps {stored.shape} do not match codes {codes.shape}")
    tok = frozen.load(arm, dev)
    out.mkdir(parents=True, exist_ok=True)
    subsets = ["k8", "tta64"] if tta else ["k8"]
    acc = {(s, p): np.lib.format.open_memmap(out / f"{s}_{p}.npy", mode="w+", dtype=np.float32,
                                             shape=(n_p, t.n_classes, t.native, t.native))
           for s in subsets for p in ("log", "prob")}
    mism, checked = 0, 0
    t0 = time.time()
    chunk = 8 if tta else 64                                                     # patches a chunk
    with torch.no_grad():
        for p0 in range(0, n_p, chunk):
            p1 = min(p0 + chunk, n_p)
            sums = {key: torch.zeros(p1 - p0, t.n_classes, t.native, t.native, device=dev) for key in acc}
            v0, v1 = p0 * per_patch, p1 * per_patch
            for j in range(K):
                idx = torch.as_tensor(codes[v0:v1, j].astype(np.int64), device=dev).view(-1, side, side)
                lp, pp, lg = pooled_pair(tok, idx, "pastis", arm)
                if stored is not None:
                    am = D.snap_nearest(lg, t.ignore).to(torch.uint8).cpu().numpy()
                    mism += int((am != np.asarray(stored[v0:v1, j])).sum())
                    checked += am.size
                for n in range(v1 - v0):
                    k = int(ks[v0 + n])
                    pi = n // per_patch
                    l_k0, p_k0 = to_k0_t(lp[n], k), to_k0_t(pp[n], k)
                    if k == 0:
                        sums[("k8", "log")][pi] += l_k0
                        sums[("k8", "prob")][pi] += p_k0
                    if tta:
                        sums[("tta64", "log")][pi] += l_k0
                        sums[("tta64", "prob")][pi] += p_k0
            for key, arr in acc.items():
                arr[p0:p1] = sums[key].cpu().numpy()
            if p0 % (chunk * 10) == 0:
                print(f"  patches {p1}/{n_p}, {time.time() - t0:.0f} s", flush=True)
    for arr in acc.values():
        arr.flush()
    meta = {"gen": str(g), "root": a.root, "arm": arm, "views": int(V), "K": int(K), "patches": int(n_p),
            "tta": tta, "subsets": subsets, "samples_per_patch": {"k8": int(K), "tta64": int(K * 8) if tta else None},
            "argmax_check": {"pixels": checked, "mismatches": mism} if stored is not None else None,
            "seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0)}
    (out / "soft.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
