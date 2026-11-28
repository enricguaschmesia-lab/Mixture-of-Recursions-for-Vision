#!/usr/bin/env python
"""A2 Step 7.0i, CloudSen12's corrected vote (docs/plans/a2_segmentation.md, Step 7, "7.0i"): per-tile vote counts of a
CloudSen12 K-sample run (`a2seg`, the GTX Titan X). Decoding only.

    python eo/experiments/segmentation/step70i_c12_votes.py --gen <run>/full_k40 --out <dir> [--tiles N]
                                                           [--drop-first] [--expect-gpu "GTX TITAN X"]

Each sample is decoded by the scorer's own path (eo/eval/seg.score: harness.class_scores over each tile's 3 x 3 crops,
stitched, then the arm's rule) and its class map counted per pixel and class:
  <out>/counts_k<K>.npy   (tiles, 4, 512, 512) uint8, the votes over all K samples
  <out>/counts_k8.npy     the same over the first 8 samples
  <out>/votes.json        the tiles' ids in order, and the validity: the plain vote of each count array (argmax, ties to
                          the lower class, as eo/eval/seg.vote) gives per-tile matrices equal to the scorer's (the run's
                          per_patch_cm.npy, and k8/per_patch_cm.npy written by step45_kcurve.py)
--tiles N decodes the first N tiles only (the gate); --drop-first leaves sample 0 out of the K count (the gate's control).
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


def plain_from_counts(counts: np.ndarray) -> np.ndarray:
    """(T, C, H, W) votes -> (T, H, W) the per-pixel majority, ties to the lower class (eo/eval/seg.vote)."""
    return np.argmax(counts, axis=1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiles", type=int, default=None)
    ap.add_argument("--drop-first", action="store_true")
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    import torch
    from eo.data import map_vocab as MV
    from eo.data import seg_dataset as SD
    from eo.eval import seg as SEG
    from eo.maps import decode as D, frozen, harness as H, palette as P
    from step4_score import per_patch_cm
    dev = "cuda"
    if a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    g, out = Path(a.gen), Path(a.out)
    if (out / "votes.json").exists():
        print(f"skip {out}: done already", flush=True)
        return
    rec = json.loads((g / "record.json").read_text())
    if rec["split"] != "val" or rec["task"] != "cloudsen12":
        raise PermissionError(f"{g}: {rec['task']} {rec['split']}")
    task, arm = rec["task"], rec["arm"]
    t = P.TASKS[task]
    side = MV.spec(arm)[2]
    views = SD.SegDataset(SD.SegSpec(task, arm, rec["frames"], rec["split"])).views.set_index("view").loc[rec["views"]].reset_index()
    problems = SEG.crop_order_problems(task, views)
    if problems:
        raise RuntimeError("; ".join(problems[:3]))
    codes = np.load(g / "codes.npy", mmap_mode="r")
    V, K, G = codes.shape
    if G != side * side or V != len(views):
        raise ValueError(f"codes {codes.shape} for {len(views)} views of arm {arm}")
    labels, _ = SEG.gt_for_views(task, views)
    n_crops = V // len(labels)
    tiles = list(dict.fromkeys(views.sample_id.tolist()))
    T = min(a.tiles or len(labels), len(labels))
    tok = frozen.load(arm, dev)
    rule = frozen.record(arm)["rule"]
    out.mkdir(parents=True, exist_ok=True)
    shape = (T, t.n_classes, t.native, t.native)
    cK = np.lib.format.open_memmap(out / f"counts_k{K}.npy", mode="w+", dtype=np.uint8, shape=shape)
    c8 = np.lib.format.open_memmap(out / "counts_k8.npy", mode="w+", dtype=np.uint8, shape=shape)
    per = 4
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, T, per):
            e = min(s + per, T)
            accK = torch.zeros((e - s, t.n_classes, t.native, t.native), dtype=torch.uint8, device=dev)
            acc8 = torch.zeros_like(accK)
            for j in range(K):
                ib = torch.as_tensor(np.asarray(codes[s * n_crops:e * n_crops, j]).astype(np.int64), device=dev)
                sc = H.class_scores(tok, ib.view(-1, side, side), task, P.arm_palette(arm), "crop224")
                pr = D.apply_rule(sc, t.ignore, rule, tok.output).long()                       # (n, H, W)
                oh = torch.nn.functional.one_hot(pr, t.n_classes).permute(0, 3, 1, 2).to(torch.uint8)
                if not (a.drop_first and j == 0):
                    accK += oh
                if j < 8:
                    acc8 += oh
            cK[s:e], c8[s:e] = accK.cpu().numpy(), acc8.cpu().numpy()
            if s % 40 == 0:
                print(f"  tiles {e}/{T}, {time.time() - t0:.0f} s", flush=True)
    cK.flush()
    c8.flush()
    lab = np.asarray(labels[:T])
    check = {}
    for name, arr, ref in ((f"k{K}", cK, g / "per_patch_cm.npy"), ("k8", c8, g / "k8" / "per_patch_cm.npy")):
        cms = per_patch_cm(plain_from_counts(np.asarray(arr)), lab, t.n_classes, t.ignore)
        stored = np.load(ref)[:T] if ref.exists() else None
        check[name] = {"equal_to_scorer": bool(stored is not None and np.array_equal(cms, stored)),
                       "scorer_matrices": str(ref), "tiles": int(T)}
    meta = {"gen": str(g), "task": task, "arm": arm, "rule": rule, "K": int(K), "tiles": tiles[:T],
            "drop_first": bool(a.drop_first), "check": check, "seconds": time.time() - t0,
            "gpu": torch.cuda.get_device_name(0)}
    (out / "votes.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "tiles"}), flush=True)


if __name__ == "__main__":
    main()
