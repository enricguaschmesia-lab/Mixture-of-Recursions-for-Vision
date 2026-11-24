#!/usr/bin/env python
"""A2 Step 5.2: score generated val maps of either task, PASTIS-1 or CloudSen12 (`a2seg`, one card: the GTX Titan X
under /data/enric/reports/a2_seg/queues/step52_titanx.sh).

Step 4.0's step4_score.py assumes one view per sample (PASTIS). A CloudSen12 tile is 9 views, GEO-Bench-2's 3 x 3
overlapping 224-px crops: eo/eval/seg.score decodes each crop's codes with the frozen arm, averages the class scores
where crops overlap (eo/maps/tiling.stitch), applies the arm's rule at 512 px, votes the K samples per pixel and reads
GEO-Bench-2's macro Jaccard (nothing ignored). Here the stitched predictions also give one confusion matrix per SAMPLE
(tile or patch), so the paired bootstrap of Step 5's read resamples tiles. Writes beside the codes:
  score.json            eval/seg.score's record (mIoU, pixel accuracy, predicted-class shares) plus the per-sample sum
  per_patch_cm.npy      (samples, C, C) int64, one matrix per patch (PASTIS) or tile (CloudSen12), in the views'
                        sample order; summed it is the scorer's matrix exactly (checked here)
A K > 1 directory also gets k1/ from its first sample (generation seeds sample j as seed * 1,000,003 + j * 7,919 +
batch start, so sample 0 is a K = 1 run) and k1b/ from its second, Z9's functional control (the own input, the same
path, another sample). A directory holding both files is skipped (resumable).
`--ceiling ARM --task T --out DIR` scores the stored TRUE map tokens of T's val views into DIR/ceiling_<T>_<ARM>/ the
same way (gate Z17 reads it against Step 1's frozen ceiling). Never a test split (eo/eval/seg refuses it).

    python eo/experiments/segmentation/step52_score.py --dirs <gen dir> [...] [--expect-gpu "GTX TITAN X"]
    python eo/experiments/segmentation/step52_score.py --ceiling aligned --task cloudsen12 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eo.data import seg_dataset as SD      # noqa: E402
from eo.eval import seg as SEG              # noqa: E402
from eo.maps import frozen, palette as P    # noqa: E402
from step4_score import miou, per_patch_cm  # noqa: E402


def sample_index_problems(s_idx: np.ndarray, n_samples: int, n_views: int) -> list:
    """Each sample's views must be consecutive and in first-seen order: n_crops views per sample (1 for PASTIS, 9 for
    CloudSen12), so the scorer's stitched prediction i is sample i."""
    n_crops = n_views // max(n_samples, 1)
    if n_crops * n_samples != n_views:
        return [f"{n_views} views are not a whole number of crops for {n_samples} samples"]
    want = np.repeat(np.arange(n_samples), n_crops)
    return [] if np.array_equal(s_idx, want) else ["views are not grouped by sample in order"]


def score_codes(task: str, arm: str, codes: np.ndarray, views, tok, dev: str, rec: dict, out: Path) -> dict:
    if (out / "score.json").exists() and (out / "per_patch_cm.npy").exists():
        print(f"skip {out}: scored already", flush=True)
        return json.loads((out / "score.json").read_text())
    t0 = time.time()
    res = SEG.score(task, arm, codes, views, device=dev, tok=tok, return_pred=True)
    pred = res.pop("pred")
    labels, s_idx = SEG.gt_for_views(task, views)
    problems = sample_index_problems(s_idx, len(labels), len(views))
    if problems or len(pred) != len(labels):
        raise RuntimeError(f"{task}: {problems or 'predictions and labels differ in number'}")
    t = P.TASKS[task]
    cms = per_patch_cm(pred, labels, t.n_classes, t.ignore)
    whole = miou(cms.sum(0), t.ignore)
    if abs(whole - res["result"]["mIoU"]) > 1e-12:
        raise RuntimeError(f"per-sample matrices sum to mIoU {whole}, the scorer read {res['result']['mIoU']}")
    res.update({"source": rec, "seconds": time.time() - t0, "per_patch_sum_mIoU": whole,
                "per_patch_unit": "tile" if task == "cloudsen12" else "patch",
                "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"})
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "per_patch_cm.npy", cms)
    (out / "score.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    r = res["result"]
    print(f"{task} {arm} {rec.get('variant', 'ceiling')} K {codes.shape[1] if codes.ndim == 3 else 1}: mIoU "
          f"{r['mIoU']:.4f} (per-sample sum {whole:.4f}, {len(labels)} samples), pixel {r['pixel_acc']:.4f}, top class "
          f"share {res['pred_top_share']:.3f}, {time.time() - t0:.0f} s -> {out}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--ceiling", nargs="*", default=[])
    ap.add_argument("--task", default=None, help="with --ceiling")
    ap.add_argument("--out", default=None, help="where --ceiling writes")
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    toks = {}
    for d in map(Path, a.dirs):
        rec = json.loads((d / "record.json").read_text())
        if rec["split"] != "val":
            raise PermissionError(f"{d}: split {rec['split']}")
        codes = np.load(d / "codes.npy")
        ds = SD.SegDataset(SD.SegSpec(rec["task"], rec["arm"], rec["frames"], rec["split"]))
        views = ds.views.set_index("view").loc[rec["views"]].reset_index()
        tok = toks.setdefault(rec["arm"], frozen.load(rec["arm"], dev))
        score_codes(rec["task"], rec["arm"], codes, views, tok, dev, rec, d)
        if codes.shape[1] > 1:
            score_codes(rec["task"], rec["arm"], codes[:, :1], views, tok, dev,
                        dict(rec, k=1, note="sample 0 of the K run"), d / "k1")
            score_codes(rec["task"], rec["arm"], codes[:, 1:2], views, tok, dev,
                        dict(rec, k=1, note="sample 1 of the K run (Z9's functional control)"), d / "k1b")
    for arm in a.ceiling:
        if a.task not in SD.TASKS or not a.out:
            raise SystemExit("--ceiling needs --task (pastis or cloudsen12) and --out")
        frames = "p1" if a.task == "pastis" else "single"
        ds = SD.SegDataset(SD.SegSpec(a.task, arm, frames, "val"))
        views = ds.views
        codes = np.asarray(ds.maps[views.view.values])[:, None]
        tok = toks.setdefault(arm, frozen.load(arm, dev))
        score_codes(a.task, arm, codes, views, tok, dev,
                    {"ceiling": True, "task": a.task, "arm": arm, "frames": frames, "split": "val",
                     "frozen": frozen.ALL[arm]}, Path(a.out) / f"ceiling_{a.task}_{arm}")


if __name__ == "__main__":
    main()
