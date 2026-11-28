#!/usr/bin/env python
"""A2 Step 4.5 (plan Step 4, "4.5, the grid decision", items 3 and 8): score Route A at the generation-matched K_m and
its vote curve (`a2seg`, one card: the GTX Titan X under /data/enric/reports/a2_seg/queues/step43_titanx.sh).

For a sampled val run at K_m (eo/scripts/eval_seg.py generate, seed 0, batch 256) and its Route A K = 8 run (same
checkpoint, seed and batch): first whether the K_m run's first 8 samples equal the K = 8 run's codes sample for sample
(generation seeds sample j as seed * 1,000,003 + j * 7,919 + batch start, so they must; written to
<dir>/k8_equals_full_k8.json, and the read stops on a mismatch). Then the per-pixel vote over the first k samples,
k = 8, 16, ..., K_m, each scored by step52_score.score_codes (decode with the frozen arm, CloudSen12's stitching, the
arm's rule, GEO-Bench-2's mIoU, one confusion matrix per patch or tile) into <dir>/k<k>/; k = K_m is also written to
<dir> itself (score.json, per_patch_cm.npy), the matched reference the read uses.

    python eo/experiments/segmentation/step45_kcurve.py --dir <gen>/full_k40 --ref <route A gen>/full_k8 \
        [--expect-gpu "GTX TITAN X"]
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

from eo.data import seg_dataset as SD      # noqa: E402
from eo.maps import frozen                  # noqa: E402
from step52_score import score_codes        # noqa: E402


def same_run_diffs(rec: dict, rr: dict, same_weights=None) -> dict:
    """The fields in which two generate records differ, of those that define the run. With `same_weights` (two SHA-256
    listing files, e.g. a checkpoint pushed from one RCP site to another: A2 Step 7.0i), the checkpoint paths may differ
    in the site root alone (the part before `/mor-eo-rcp/`), and only if the two listings are identical and non-empty."""
    diffs = {k: (rec[k], rr[k]) for k in ("checkpoint", "seed", "temperature", "precision", "task", "arm", "variant")
             if rec[k] != rr[k]}
    if "checkpoint" in diffs and same_weights:
        rel = lambda p: str(p).split("/mor-eo-rcp/", 1)[-1]                                   # noqa: E731
        ls = [sorted(l for l in Path(f).read_text().splitlines() if l.strip()) for f in same_weights]
        if rel(rec["checkpoint"]) == rel(rr["checkpoint"]) and len(ls) == 2 and ls[0] and ls[0] == ls[1]:
            diffs.pop("checkpoint")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True, help="the same checkpoint's K = 8 run (full_k8)")
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--same-weights", nargs=2, default=None, metavar=("A.sha256", "B.sha256"),
                    help="two identical SHA-256 listings of the checkpoint's files at the two sites; then the records' "
                         "checkpoint paths may differ in the site root alone (the first-8 check still decides)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    d, ref = Path(a.dir), Path(a.ref)
    rec, rr = json.loads((d / "record.json").read_text()), json.loads((ref / "record.json").read_text())
    K = int(rec["k"])
    if rec["split"] != "val" or rec["greedy"] or K < 16 or K % 8:
        raise SystemExit(f"{d}: not a sampled val run at a multiple of 8 >= 16 ({rec['split']}, K {K})")
    same_run = same_run_diffs(rec, rr, a.same_weights)
    if same_run or rec["views"] != rr["views"] or int(rr["k"]) != 8:
        raise SystemExit(f"{d} and {ref} are not the same checkpoint's runs: {same_run}")
    codes, c8 = np.load(d / "codes.npy"), np.load(ref / "codes.npy")
    eq = [bool(np.array_equal(c8[:, j], codes[:, j])) for j in range(8)]
    (d / "k8_equals_full_k8.json").write_text(json.dumps(
        {"full_k8": str(ref), "samples_equal": eq, "all_equal": all(eq),
         "token_agreement": float((c8 == codes[:, :8]).mean())}, indent=1) + "\n")
    print(f"first 8 samples equal the K = 8 run's: {eq}", flush=True)
    if not all(eq):
        raise SystemExit("the K_m run's first 8 samples are not the K = 8 run's: stop (plan item 3)")
    ds = SD.SegDataset(SD.SegSpec(rec["task"], rec["arm"], rec["frames"], rec["split"]))
    views = ds.views.set_index("view").loc[rec["views"]].reset_index()
    tok = frozen.load(rec["arm"], dev)
    for k in range(8, K + 1, 8):
        score_codes(rec["task"], rec["arm"], codes[:, :k], views, tok, dev,
                    dict(rec, k=k, note=f"the first {k} samples of the K = {K} run"), d / f"k{k}")
    score_codes(rec["task"], rec["arm"], codes, views, tok, dev, rec, d)


if __name__ == "__main__":
    main()
