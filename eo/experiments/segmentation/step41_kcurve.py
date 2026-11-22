#!/usr/bin/env python
"""A2 Step 4.1, item 11 (described, not deciding): the vote over the first k of K = 16 samples, k = 1, 2, 4, 8, 16
(`a2seg`, one card; the Titan X under A1's flock through /data/enric/reports/a2_seg/queues/step41_titanx.sh).

eval_seg.py generate seeds sample j as seed * 1,000,003 + j * 7,919 + batch start, so at one batch size the first k
samples of a K = 16 run are the draws a K = k run would make. Each subset is scored by step4_score.score_codes (the
frozen nearest rule, the per-pixel majority, GEO-Bench-2's mIoU at native 128 px) into <dir>/k<k>/ (score.json,
per_patch_cm.npy). Also written: <dir>/k8_equals_full_k8.json, whether the first 8 samples equal the run's own K = 8
codes (same checkpoint, seed and batch) sample for sample.

    python eo/experiments/segmentation/step41_kcurve.py --dirs <gen dir>/full_k16 [...] [--expect-gpu "GTX TITAN X"]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eo.data import seg_dataset as SD        # noqa: E402
from eo.maps import frozen                    # noqa: E402
from step4_score import score_codes           # noqa: E402

KS = (1, 2, 4, 8, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    toks = {}
    for d in map(Path, a.dirs):
        rec = json.loads((d / "record.json").read_text())
        if rec["split"] != "val" or int(rec["k"]) != 16 or rec["greedy"]:
            raise SystemExit(f"{d}: not a sampled K = 16 val run ({rec['split']}, K {rec['k']}, greedy {rec['greedy']})")
        codes = np.load(d / "codes.npy")
        ds = SD.SegDataset(SD.SegSpec(rec["task"], rec["arm"], rec["frames"], rec["split"]))
        views = ds.views.set_index("view").loc[rec["views"]].reset_index()
        tok = toks.setdefault(rec["arm"], frozen.load(rec["arm"], dev))
        for k in KS:
            score_codes(rec["task"], rec["arm"], codes[:, :k], views, tok, dev,
                        dict(rec, k=k, note=f"the first {k} samples of the K = 16 run"), d / f"k{k}")
        k8 = d.parent / "full_k8" / "codes.npy"
        if k8.exists():
            c8 = np.load(k8)
            same = [bool(np.array_equal(c8[:, j], codes[:, j])) for j in range(8)]
            (d / "k8_equals_full_k8.json").write_text(json.dumps(
                {"full_k8": str(k8), "samples_equal": same, "all_equal": all(same),
                 "token_agreement": float((c8 == codes[:, :8]).mean())}, indent=1) + "\n")
            print(f"{d}: first 8 samples equal the K = 8 run's: {same} (token agreement "
                  f"{float((c8 == codes[:, :8]).mean()):.6f})", flush=True)


if __name__ == "__main__":
    main()
