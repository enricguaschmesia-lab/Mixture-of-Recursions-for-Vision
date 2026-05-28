"""
Pass 2: score scene_desc predictions against ground-truth scene descriptions.

Reads a predictions.jsonl produced by generate_scene_desc.py and the GT JSONs
from <data-dir>/scene_desc/<id>.json. CPU-only, no torch.

Metrics, per sample:
    - n_pred, n_gt: object counts
    - tp, fp, fn under Hungarian matching with strict attribute equality
        (shape, color, material) and position L2 as tiebreaker
    - jaccard = tp / (tp + fp + fn)
    - precision, recall, f1
    - pos_l2 = mean L2 over matched pairs (NaN if no matches)
    - parse_ok = was the regex able to extract at least one object?

Aggregates written to <out-dir>/metrics.json (mean over samples) and
<out-dir>/per_sample.csv. Also keeps a small set of diagnostic counters:
    - n_count_match: samples where n_pred == n_gt
    - mean_count_diff: mean |n_pred - n_gt|
    - per-attribute accuracy under a looser closest-position matching

Usage:
    python scripts/score_scene_desc.py \\
        --predictions /results/eval/scene_desc/random_router/predictions.jsonl \\
        --gt-dir /home/.../clevr_com_304/test/scene_desc \\
        --out-dir /results/eval/scene_desc/random_router \\
        --aug-idx 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# Tolerant regex: captures one object descriptor. The format in CLEVR scene_desc:
#   "Object 1 - Position: x=49 y=34 Shape: cube Color: yellow Material: rubber."
# We don't require punctuation/whitespace to be exact — only that the labelled
# fields appear in order.
_OBJ_PATTERN = re.compile(
    r"Object\s+\d+\s*-\s*"
    r"Position:\s*x\s*=\s*(-?\d+)\s*y\s*=\s*(-?\d+)\s*"
    r"Shape:\s*(\w+)\s*"
    r"Color:\s*(\w+)\s*"
    r"Material:\s*(\w+)",
    re.IGNORECASE,
)

# Cost used when attribute equality fails; must dominate any plausible position L2.
_ATTR_MISMATCH_COST = 1.0e6


@dataclass(frozen=True)
class Obj:
    x: int
    y: int
    shape: str
    color: str
    material: str

    @property
    def attrs(self) -> tuple[str, str, str]:
        return (self.shape, self.color, self.material)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True,
                   help="Path to predictions.jsonl from pass 1")
    p.add_argument("--gt-dir", required=True,
                   help="Directory of GT scene_desc JSONs (one per id)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--aug-idx", type=int, default=0,
                   help="Which GT augmentation to score against")
    return p.parse_args()


def parse_scene(text: str) -> list[Obj]:
    """Extract objects from a scene description string. Lenient: returns
    whatever matches the regex; malformed tails are silently dropped."""
    out: list[Obj] = []
    if not text:
        return out
    for m in _OBJ_PATTERN.finditer(text):
        try:
            x = int(m.group(1)); y = int(m.group(2))
            shape = m.group(3).lower()
            color = m.group(4).lower()
            material = m.group(5).lower()
        except (ValueError, IndexError):
            continue
        out.append(Obj(x, y, shape, color, material))
    return out


def hungarian_match(pred: list[Obj], gt: list[Obj]) -> list[tuple[int, int]]:
    """Match predicted objects to GT objects.

    Cost: 0 if all three attributes match, +position_L2 * small_weight as
    tiebreaker; otherwise _ATTR_MISMATCH_COST. After solving, drop any pair
    whose cost is >= _ATTR_MISMATCH_COST — those are not real matches.
    """
    if not pred or not gt:
        return []
    P, G = len(pred), len(gt)
    cost = np.full((P, G), _ATTR_MISMATCH_COST, dtype=np.float64)
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if p.attrs == g.attrs:
                d = math.hypot(p.x - g.x, p.y - g.y)
                cost[i, j] = d * 1e-3  # tiny weight, just to disambiguate ties
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < _ATTR_MISMATCH_COST / 2:
            matches.append((int(r), int(c)))
    return matches


def per_attribute_diagnostic(pred: list[Obj], gt: list[Obj]) -> dict:
    """Looser matching for per-attribute analysis: pair each predicted object
    with its closest-position GT object (no attribute constraint), then count
    agreements per attribute. Each GT object can be claimed by at most one
    predicted object (greedy by ascending distance).

    Returns counts: matched (n pairs formed), shape_ok, color_ok, material_ok.
    These are over the same set of pairs, so accuracy = ok / matched.
    """
    if not pred or not gt:
        return {"matched": 0, "shape_ok": 0, "color_ok": 0, "material_ok": 0}

    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            pairs.append((math.hypot(p.x - g.x, p.y - g.y), i, j))
    pairs.sort()
    used_p, used_g = set(), set()
    matched_pairs: list[tuple[int, int]] = []
    for d, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matched_pairs.append((i, j))

    shape_ok = color_ok = material_ok = 0
    for i, j in matched_pairs:
        p, g = pred[i], gt[j]
        shape_ok += int(p.shape == g.shape)
        color_ok += int(p.color == g.color)
        material_ok += int(p.material == g.material)
    return {
        "matched": len(matched_pairs),
        "shape_ok": shape_ok,
        "color_ok": color_ok,
        "material_ok": material_ok,
    }


def score_one(pred_text: str, gt_text: str) -> dict:
    """All metrics for one (pred, gt) pair."""
    pred = parse_scene(pred_text)
    gt = parse_scene(gt_text)
    matches = hungarian_match(pred, gt)
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - tp
    denom_j = tp + fp + fn
    jaccard = tp / denom_j if denom_j > 0 else 0.0
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    pos_l2 = float("nan")
    if matches:
        pos_l2 = float(np.mean([
            math.hypot(pred[i].x - gt[j].x, pred[i].y - gt[j].y)
            for i, j in matches
        ]))

    diag = per_attribute_diagnostic(pred, gt)
    return {
        "n_pred": len(pred),
        "n_gt": len(gt),
        "tp": tp, "fp": fp, "fn": fn,
        "jaccard": jaccard,
        "precision": precision, "recall": recall, "f1": f1,
        "pos_l2": pos_l2,
        "parse_ok": bool(pred),
        **{f"diag_{k}": v for k, v in diag.items()},
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean over samples for top-line metrics. NaNs (no matches) skipped for
    pos_l2; everything else averages over all samples."""
    n = len(rows)
    if n == 0:
        return {"n_samples": 0}

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    pos_l2_vals = [r["pos_l2"] for r in rows if not math.isnan(r["pos_l2"])]
    mean_pos_l2 = float(np.mean(pos_l2_vals)) if pos_l2_vals else float("nan")

    tot_diag = Counter()
    for r in rows:
        for k in ("diag_matched", "diag_shape_ok", "diag_color_ok", "diag_material_ok"):
            tot_diag[k] += r[k]
    diag_matched = max(tot_diag["diag_matched"], 1)

    out = {
        "n_samples": n,
        "jaccard": mean("jaccard"),
        "f1": mean("f1"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "mean_pos_l2": mean_pos_l2,
        "n_samples_with_match": len(pos_l2_vals),
        "mean_n_pred": mean("n_pred"),
        "mean_n_gt": mean("n_gt"),
        "mean_count_diff": float(np.mean([abs(r["n_pred"] - r["n_gt"]) for r in rows])),
        "frac_count_match": float(np.mean([r["n_pred"] == r["n_gt"] for r in rows])),
        "parse_ok_rate": float(np.mean([r["parse_ok"] for r in rows])),
        "per_attr_acc": {
            "shape":    tot_diag["diag_shape_ok"] / diag_matched,
            "color":    tot_diag["diag_color_ok"] / diag_matched,
            "material": tot_diag["diag_material_ok"] / diag_matched,
        },
    }
    return out


def load_gt_text(gt_dir: Path, sid: str, aug_idx: int) -> Optional[str]:
    p = gt_dir / f"{sid}.json"
    if not p.is_file():
        return None
    with open(p) as f:
        captions = json.load(f)
    if not isinstance(captions, list) or aug_idx >= len(captions):
        return None
    return captions[aug_idx]


def main():
    args = parse_args()
    pred_path = Path(args.predictions)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_missing_gt = 0
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["id"]
            gt_text = load_gt_text(gt_dir, sid, args.aug_idx)
            if gt_text is None:
                n_missing_gt += 1
                continue
            metrics = score_one(rec.get("generated_text", ""), gt_text)
            metrics["id"] = sid
            metrics["n_new_tokens"] = rec.get("n_new_tokens")
            metrics["stopped_on_eo"] = rec.get("stopped_on_eo")
            rows.append(metrics)

    if n_missing_gt:
        print(f"  [warn] {n_missing_gt} predictions had no matching GT and were skipped")

    agg = aggregate(rows)
    print(json.dumps(agg, indent=2))

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Wrote: {metrics_path}")

    csv_path = out_dir / "per_sample.csv"
    fieldnames = ["id", "n_pred", "n_gt", "tp", "fp", "fn",
                  "jaccard", "f1", "precision", "recall", "pos_l2",
                  "parse_ok", "stopped_on_eo", "n_new_tokens",
                  "diag_matched", "diag_shape_ok", "diag_color_ok", "diag_material_ok"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()