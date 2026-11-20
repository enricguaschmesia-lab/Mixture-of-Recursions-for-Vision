"""A2 Step 3's done-criterion: the map-extended base's UNTRAINED map rows give the class-prior control (no input,
eo/eval/seg.py) a map at chance, and the harness reads chance as chance (`a2seg`, CPU).

Reads every step3/prior/<name>/score.json (eval_seg.py score) and sets each beside two references computed on
the same ground truth (PASTIS fold 4, CloudSen12 val), by the same scorer (eo/maps/scoring.Confusion):
  uniform   every pixel an independent uniform draw over the scored classes (seed 0)
  constant  every pixel the most frequent scored class (the best constant map)
The prior control reads chance if its mIoU is near `uniform`'s (sampled) or no better than `constant`'s (greedy,
which on rows at their mean writes one code and so one colour), and its predicted-class shares are those of a
code drawn uniformly from the slot (sampled) or one class (greedy).

    python -m eo.experiments.segmentation.step3_prior_read [--out /data/enric/reports/a2_seg/step3/prior/summary.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eo.maps import palette as P, scoring as S, sets

PRIOR = Path("/data/enric/reports/a2_seg/step3/prior")


def references(task: str, seed: int = 0) -> dict:
    t = P.TASKS[task]
    labels, _ = sets.load(task, "val")
    scored = [c for c in range(t.n_classes) if c != t.ignore]
    g = torch.Generator().manual_seed(seed)
    uni = S.Confusion(t.n_classes, t.ignore)
    hist = np.zeros(t.n_classes)
    for s in range(0, len(labels), 32):
        gt = torch.as_tensor(np.ascontiguousarray(labels[s:s + 32])).long()
        hist += np.bincount(gt.flatten().numpy(), minlength=t.n_classes)
        draw = torch.as_tensor(scored)[torch.randint(len(scored), gt.shape, generator=g)]
        uni.update(gt, draw)
    h = hist.copy()
    if t.ignore is not None:
        h[t.ignore] = 0
    top = int(h.argmax())
    con = S.Confusion(t.n_classes, t.ignore)
    for s in range(0, len(labels), 32):
        gt = torch.as_tensor(np.ascontiguousarray(labels[s:s + 32])).long()
        con.update(gt, torch.full_like(gt, top))
    return {"uniform_mIoU": uni.result()["mIoU"], "constant_mIoU": con.result()["mIoU"], "constant_class": top,
            "gt_share": (h / h.sum()).tolist(), "n_scored": len(scored)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PRIOR / "summary.json"))
    a = ap.parse_args()
    refs, rows = {}, []
    for f in sorted(PRIOR.glob("*/score.json")):
        d = json.loads(f.read_text())
        task = d["task"]
        if task not in refs:
            refs[task] = references(task)
        r = d["result"]
        src = d["source"]
        rows.append({"name": f.parent.name, "task": task, "arm": d["arm"], "K": d["K"],
                     "decoding": "greedy" if src.get("greedy") else f"sampled T={src.get('temperature')}",
                     "mIoU": r["mIoU"], "pixel_acc": r["pixel_acc"], "pred_entropy": d["pred_entropy_nats"],
                     "pred_entropy_max": d["pred_entropy_max_nats"], "pred_top_share": d["pred_top_share"],
                     "slot_mass": src.get("slot_mass_mean"), "uniform_mIoU": refs[task]["uniform_mIoU"],
                     "constant_mIoU": refs[task]["constant_mIoU"]})
    out = {"references": refs, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'name':28s} {'mIoU':>7s} {'unif':>7s} {'const':>7s} {'H/Hmax':>12s} {'top':>6s} {'slot mass':>9s}")
    for r in rows:
        print(f"{r['name']:28s} {r['mIoU']:7.4f} {r['uniform_mIoU']:7.4f} {r['constant_mIoU']:7.4f} "
              f"{r['pred_entropy']:5.2f}/{r['pred_entropy_max']:5.2f} {r['pred_top_share']:6.3f} {r['slot_mass']:9.4f}")


if __name__ == "__main__":
    main()
