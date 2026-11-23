"""A2 Step 4.2: the scorers of Routes B and C (docs/plans/a2_segmentation.md, Step 4.2, items 2-3).

Route A's path (eo/eval/seg.score) with the tokenizer's decode replaced, so the three routes share the native-grid pooling,
the stitching of CloudSen12's crops, the nearest rule, the vote and the metric:
  Route B   generated cell codes (V, K, 784) in raster order -> eo/maps/cells.cell_scores (each cell's one-hot over its
            8 x 8 canvas pixels, pooled to the native grid by the render's adjoint) -> CloudSen12's 3 x 3 crops stitched
            (eo/maps/tiling.stitch, averaged where they overlap) -> the nearest rule (eo/maps/decode.apply_rule: argmax,
            void masked) per sample -> the K-sample vote (eo/eval/seg.vote, ties to the lower class) -> scoring.Confusion.
  Route C   a prediction per sample, made by the head's own evaluation (eo/finetune/seg_head.predict_views: pooled
            logits -> log-softmax -> stitched -> argmax with void masked) -> scoring.Confusion here.
Both write one confusion matrix per sample (patch or tile; target x prediction, void excluded), whose sum is the scorer's
matrix exactly (checked), so the paired bootstrap of Step 5's read can resample samples.

⚠ Val only: the ground truth is eo/maps/sets' val set (eo.eval.seg.gt_for_views refuses anything else).
Runs in `.venv` and `a2seg`: numpy and torch, no tokenizer.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def per_sample_cm(pred: np.ndarray, labels: np.ndarray, n: int, ignore: Optional[int]) -> np.ndarray:
    """(S, n, n) int64: each sample's confusion matrix, target x prediction, the ignored target excluded
    (eo/experiments/segmentation/step4_score.per_patch_cm's definition)."""
    out = np.zeros((len(pred), n, n), np.int64)
    for i in range(len(pred)):
        g, p = labels[i].astype(np.int64).ravel(), pred[i].astype(np.int64).ravel()
        v = g != ignore if ignore is not None else np.ones_like(g, bool)
        out[i] = np.bincount(g[v] * n + p[v], minlength=n * n).reshape(n, n)
    return out


def miou_of(cm: np.ndarray, ignore: Optional[int]) -> float:
    """scoring.Confusion.result's mIoU from a summed matrix: macro over classes with any target or prediction."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), 0.0)
    w = np.ones(len(cm))
    if ignore is not None:
        w[ignore] = 0
    w[(cm.sum(0) + cm.sum(1)) == 0] = 0
    return float((w * iou).sum() / w.sum())


def _finish(task: str, conf, hist: torch.Tensor, preds: list, labels: np.ndarray, extra: Dict) -> Dict:
    from eo.maps import palette as P
    t = P.TASKS[task]
    res = conf.result()
    share = (hist / hist.sum()).numpy()
    scored = [c for c in range(t.n_classes) if c != t.ignore]
    p = share[scored] / max(share[scored].sum(), 1e-12)
    pred = np.concatenate(preds)
    cms = per_sample_cm(pred, labels, t.n_classes, t.ignore)
    whole = miou_of(cms.sum(0), t.ignore)
    if abs(whole - res["mIoU"]) > 1e-12 or not np.array_equal(cms.sum(0), conf.matrix()):
        raise RuntimeError(f"per-sample matrices sum to mIoU {whole}, the scorer read {res['mIoU']}")
    out = {"task": task, "samples": int(len(labels)), "result": res, "pred_share": share.tolist(),
           "pred_entropy_nats": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
           "pred_entropy_max_nats": float(np.log(len(scored))), "pred_top_share": float(p.max()),
           "per_sample_sum_mIoU": whole, "per_sample_unit": "tile" if task == "cloudsen12" else "patch"}
    out.update(extra)
    return out, cms, pred


def score_cells(task: str, codes: np.ndarray, views, device: str = "cpu", maps_per_chunk: Optional[int] = None):
    """Route B: codes (V, K, 784) (or (V, 784)) of these val views, rows aligned -> (record, per-sample matrices,
    predictions (S, native, native) uint8)."""
    from eo.eval import seg as SEG
    from eo.maps import cells as CE, decode as D, palette as P, scoring as S, tiling
    problems = SEG.crop_order_problems(task, views)
    if problems:
        raise RuntimeError("; ".join(problems[:3]))
    t = P.TASKS[task]
    if codes.ndim == 2:
        codes = codes[:, None]
    V, K, G = codes.shape
    g = CE.GRID
    if G != g * g or V != len(views):
        raise ValueError(f"codes {codes.shape} for {len(views)} views: want (views, K, {g * g})")
    if int(codes.max()) >= t.n_classes:
        raise ValueError(f"a code {int(codes.max())} outside {task}'s {t.n_classes} classes")
    labels, s_idx = SEG.gt_for_views(task, views)
    n_crops = len(views) // len(labels)
    if n_crops * len(labels) != len(views) or not np.array_equal(s_idx, np.repeat(np.arange(len(labels)), n_crops)):
        raise RuntimeError("views are not grouped by sample, a whole number of crops each")
    crops = tiling.plan(task, t.native, "crop224")
    crop_native = crops[0][2]
    conf = S.Confusion(t.n_classes, t.ignore, device)
    hist = torch.zeros(t.n_classes, dtype=torch.float64)
    per = maps_per_chunk or {"pastis": 64, "cloudsen12": 4}[task]
    preds = []
    with torch.no_grad():
        for s in range(0, len(labels), per):
            e = min(s + per, len(labels))
            gt = torch.as_tensor(np.ascontiguousarray(labels[s:e]), device=device).long()
            ks = []
            for j in range(K):
                c = torch.as_tensor(codes[s * n_crops:e * n_crops, j].astype(np.int64), device=device).view(-1, g, g)
                sc = tiling.stitch(CE.cell_scores(c, t.n_classes, crop_native), crops, t.native)
                ks.append(D.apply_rule(sc, t.ignore, {"rule": "nearest"}, "logits"))
            pr = SEG.vote(torch.stack(ks), t.n_classes)
            conf.update(gt, pr)
            hist += torch.bincount(pr.flatten().cpu(), minlength=t.n_classes).double()
            preds.append(pr.to(torch.uint8).cpu().numpy())
    return _finish(task, conf, hist, preds, labels, {"route": "B", "views": int(V), "K": int(K)})


def score_pred(task: str, pred: np.ndarray, views, device: str = "cpu"):
    """Route C (or any per-sample prediction): pred (S, native, native) for the samples of these val views, in their
    first-seen order -> (record, per-sample matrices, pred)."""
    from eo.eval import seg as SEG
    from eo.maps import palette as P, scoring as S
    t = P.TASKS[task]
    labels, s_idx = SEG.gt_for_views(task, views)
    if pred.shape != labels.shape:
        raise ValueError(f"predictions {pred.shape} against labels {labels.shape}")
    if t.ignore is not None and bool((pred == t.ignore).any()):
        raise ValueError("a prediction is the ignored class")
    conf = S.Confusion(t.n_classes, t.ignore, device)
    hist = torch.zeros(t.n_classes, dtype=torch.float64)
    preds = []
    for s in range(0, len(labels), 32):
        gt = torch.as_tensor(np.ascontiguousarray(labels[s:s + 32]), device=device).long()
        pr = torch.as_tensor(np.ascontiguousarray(pred[s:s + 32]), device=device).long()
        conf.update(gt, pr)
        hist += torch.bincount(pr.flatten().cpu(), minlength=t.n_classes).double()
        preds.append(pred[s:s + 32].astype(np.uint8))
    return _finish(task, conf, hist, preds, labels, {"route": "C", "views": int(len(views))})


def bootstrap_mean_diff(cms_a: list, cms_b: list, ignore: Optional[int], n: int = 1000, seed: int = 0) -> Dict:
    """Step 4.1's paired bootstrap of a three-seed mean difference: resample the samples (patches or tiles) with
    replacement, the same resample for every run; per resample each run's macro mIoU from its summed matrices; the
    difference of the arms' seed means. cms_*: lists of (S, C, C) per seed, samples aligned."""
    S = len(cms_a[0])
    if any(len(c) != S for c in cms_a + cms_b):
        raise ValueError("the runs' samples are not aligned")
    rng = np.random.default_rng(seed)
    point = np.mean([miou_of(c.sum(0), ignore) for c in cms_b]) - np.mean([miou_of(c.sum(0), ignore) for c in cms_a])
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, S, S)
        d[i] = (np.mean([miou_of(c[idx].sum(0), ignore) for c in cms_b])
                - np.mean([miou_of(c[idx].sum(0), ignore) for c in cms_a]))
    return {"mean_diff": float(point), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "resamples": n, "seed": seed, "samples": S}
