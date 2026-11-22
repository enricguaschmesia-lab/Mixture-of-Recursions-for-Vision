#!/usr/bin/env python
"""A2 Step 4.0: score generated fold-4 maps (protocol item 6; `a2seg`, one card: the TITAN V only under
/data/enric/reports/a2_seg/queues/step4_titanv_unit.sh).

For each directory written by `eval_seg.py generate` (record.json + codes.npy), through eo/eval/seg.score: the frozen
arm decodes each code grid (SHA256SUMS re-hashed by eo/maps/frozen), the arm's nearest rule picks a class per pixel at
native 128 px, K samples vote per pixel, and GEO-Bench-2's macro Jaccard is read (void ignored). Writes beside the
codes:
  score.json            eval_seg.py score's record (mIoU, pixel and boundary accuracy, predicted-class shares)
  per_patch_cm.npy      (patches, C, C) int64: each patch's confusion matrix (target x prediction, void excluded),
                        in the views' order; summed over patches it is the scorer's matrix, exactly (checked here)
A K = 8 directory also gets `k1/` (score.json, per_patch_cm.npy) from its FIRST sample: generation seeds sample j
as seed * 1,000,003 + j * 7,919 + batch start, so sample 0 is a K = 1 run (the plan's "How the protocol is read");
and `k1b/` from its second sample, Z9's functional control (the own input, the same path, another sample).
`--ceiling ARM` scores the stored TRUE map tokens of the fold-4 views into <out>/ceiling_<arm>/ the same way.
A directory that already holds both files is skipped (resumable). Never a test split (eo/eval/seg refuses it).

    python eo/experiments/segmentation/step4_score.py --dirs <gen dir> [<gen dir> ...] [--ceiling aligned short long
        --out /data/enric/reports/a2_seg/step4/ceiling]
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

from eo.data import seg_dataset as SD      # noqa: E402
from eo.eval import seg as SEG              # noqa: E402
from eo.maps import frozen, palette as P    # noqa: E402


def per_patch_cm(pred: np.ndarray, labels: np.ndarray, n: int, ignore) -> np.ndarray:
    out = np.zeros((len(pred), n, n), np.int64)
    for i in range(len(pred)):
        g, p = labels[i].astype(np.int64).ravel(), pred[i].astype(np.int64).ravel()
        v = g != ignore if ignore is not None else np.ones_like(g, bool)
        out[i] = np.bincount(g[v] * n + p[v], minlength=n * n).reshape(n, n)
    return out


def miou(cm: np.ndarray, ignore) -> float:
    """eo/maps/scoring.Confusion.result's mIoU: macro over the classes with any target or prediction, void out."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), 0.0)
    w = np.ones(len(cm))
    if ignore is not None:
        w[ignore] = 0
    w[(cm.sum(0) + cm.sum(1)) == 0] = 0
    return float((w * iou).sum() / w.sum())


def score_codes(task: str, arm: str, codes: np.ndarray, views, tok, dev: str, rec: dict, out: Path) -> dict:
    if (out / "score.json").exists() and (out / "per_patch_cm.npy").exists():
        print(f"skip {out}: scored already", flush=True)
        return json.loads((out / "score.json").read_text())
    t0 = time.time()
    res = SEG.score(task, arm, codes, views, device=dev, tok=tok, return_pred=True)
    pred = res.pop("pred")
    labels, s_idx = SEG.gt_for_views(task, views)
    if not np.array_equal(s_idx, np.arange(len(labels))):
        raise RuntimeError("PASTIS views are not one per patch in order")
    t = P.TASKS[task]
    cms = per_patch_cm(pred, labels, t.n_classes, t.ignore)
    whole = miou(cms.sum(0), t.ignore)
    if abs(whole - res["result"]["mIoU"]) > 1e-12:
        raise RuntimeError(f"per-patch matrices sum to mIoU {whole}, the scorer read {res['result']['mIoU']}")
    res.update({"source": rec, "seconds": time.time() - t0, "per_patch_sum_mIoU": whole,
                "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"})
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "per_patch_cm.npy", cms)
    (out / "score.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    r = res["result"]
    print(f"{task} {arm} {rec.get('variant', 'ceiling')} K {codes.shape[1] if codes.ndim == 3 else 1}: mIoU "
          f"{r['mIoU']:.4f} (per-patch sum {whole:.4f}), pixel {r['pixel_acc']:.4f}, top class share "
          f"{res['pred_top_share']:.3f}, {time.time() - t0:.0f} s -> {out}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--ceiling", nargs="*", default=[])
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
        ds = SD.SegDataset(SD.SegSpec("pastis", arm, "p1", "val"))
        views = ds.views
        codes = np.asarray(ds.maps[views.view.values])[:, None]
        tok = toks.setdefault(arm, frozen.load(arm, dev))
        score_codes("pastis", arm, codes, views, tok, dev,
                    {"ceiling": True, "task": "pastis", "arm": arm, "frames": "p1", "split": "val",
                     "frozen": frozen.ALL[arm]}, Path(a.out) / f"ceiling_{arm}")


if __name__ == "__main__":
    main()
