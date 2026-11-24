#!/usr/bin/env python
"""A2 Step 5.2, either arm: the read the plan pre-registered (docs/plans/a2_segmentation.md, Step 5.2, items 1-11, and
"R1's arm, and Step 5.3's arithmetic"; json only, any env with numpy).

For each task (PASTIS-1 fold 4, CloudSen12 val) and seed (42, 43, 44):
  * the run's records: select41.json (the lowest-CE save, the plateau stop's epoch recomputed from the curve, health);
  * Z8 at bf16 (4 rounding steps) and fp32 (the gate's rule), through step41_read.z8_of;
  * item 5's fp32 branch (adopted by Enric 2026-11-22): a run failing the gate's fp32 rule is still read on its fp32
    maps, the failure reported beside it, if its fp32 drift is at least 1,000x below its own control's (the cache fed
    positions one off), Step 4.1's rule for a failing checkpoint (`fp32_readable`);
  * K = 8 mIoU at bf16 and at fp32, each from its per-sample matrices (patch or tile), which must sum to the score;
  * Z9's functional check where the controls exist (Step 4.0's rule: |m_wrong - m_prior| < |m_wrong - m_full K=1|
    passes; its control, the own input's second sample, must fail the same check).
Per task: the three seeds' mean, SD and range at each precision, and the precision item 5 would choose from R0's three
alone. The verdict's precision also needs R1's three selected checkpoints, and Step 5's verdict waits for R1's arm:
nothing here decides.
R0: PASTIS seeds 42 and 43 are Step 4.1's recipe runs (item 2), read from step41/gen; seed 44 and CloudSen12 from
step52/gen. R1 (2026-11-23): all six runs from step52/gen, run ids r1p1s52_s<seed>_<stamp> and r1c12s52_s<seed>_<stamp>.
Writes <arm>_read.{json,md}; Step 5.3's verdict (step53_verdict.py) reads both.

    python eo/experiments/segmentation/step52_read.py [--arm r0] [--rts-c12 <stamp>] [--rts-p44 <stamp>]
    python eo/experiments/segmentation/step52_read.py --arm r1 [--rts-r1 <stamp>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step41_read import z8_of             # noqa: E402

RUNS = Path("/data/enric/runs/pretrain/a2_seg")
GEN41 = Path("/data/enric/reports/a2_seg/step41/gen")
GEN52 = Path("/data/enric/reports/a2_seg/step52/gen")
OUT = Path("/data/enric/reports/a2_seg/step52")
IGNORE = {"pastis": 19, "cloudsen12": None}          # eo/maps/palette.TASKS[task].ignore
SEEDS = (42, 43, 44)
STEP41 = {42: "r0p1s41_l16c_s42_20261122_123818", 43: "r0p1s41_l16c_s43_20261122_123818"}
FP32_ROUNDING = 1000.0      # item 5's fp32 branch: drift at least this many times below the control's is rounding


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


def scored(d: Path, task: str):
    if not ((d / "score.json").exists() and (d / "per_patch_cm.npy").exists()):
        return None
    s = json.loads((d / "score.json").read_text())
    cm = np.load(d / "per_patch_cm.npy")
    m = float(s["result"]["mIoU"])
    if abs(miou(cm.sum(0), IGNORE[task]) - m) > 1e-12:
        raise RuntimeError(f"{d}: per-sample matrices do not sum to the scored mIoU")
    out = {"mIoU": m, "pixel_acc": float(s["result"]["pixel_acc"]), "pred_top_share": float(s["pred_top_share"]),
           "K": int(s["K"]), "samples": int(cm.shape[0])}
    if (d / "record.json").exists():
        r = json.loads((d / "record.json").read_text())
        out.update({"checkpoint": r.get("checkpoint"), "precision": r.get("precision")})
    return out


def find_run(task: str, seed: int, rts: dict, arm: str = "r0") -> tuple:
    if arm == "r0" and task == "pastis" and seed in STEP41:
        return STEP41[seed], GEN41
    prefix = {"pastis": f"{arm}p1s52", "cloudsen12": f"{arm}c12s52"}[task]
    key = {"pastis": "p44", "cloudsen12": "c12"}[task] if arm == "r0" else "r1"
    if rts.get(key):
        return f"{prefix}_s{seed}_{rts[key]}", GEN52
    hits = sorted(p.name for p in RUNS.glob(f"{prefix}_s{seed}_*") if (p / "select41.json").exists())
    if len(hits) > 1:
        raise RuntimeError(f"{task} seed {seed}: several runs {hits}; name the stamp")
    return (hits[0], GEN52) if hits else (None, GEN52)


def read_run(task: str, seed: int, rts: dict, arm: str = "r0") -> dict:
    name, gen = find_run(task, seed, rts, arm)
    if name is None or not (RUNS / name / "select41.json").exists():
        return {"task": task, "seed": seed, "run": name, "missing": True}
    sel = json.loads((RUNS / name / "select41.json").read_text())
    gb, gf = gen / name, gen / f"{name}_fp32"
    r = {"task": task, "seed": seed, "run": name, "missing": False,
         "selected": sel["selected"], "plateau_stop": sel.get("plateau_stop"), "health": sel.get("health"),
         "z8_bf16": z8_of(gb, "bf16"), "z8_fp32": z8_of(gb, "fp32")}
    zf = r["z8_fp32"]
    r["fp32_control_ratio"] = None if zf is None else zf["control_abs"] / max(zf["max_abs_logit_diff"], 1e-30)
    r["fp32_readable"] = None if zf is None else bool(
        zf["pass"] or zf["max_abs_logit_diff"] * FP32_ROUNDING <= zf["control_abs"])
    for p, g in (("bf16", gb), ("fp32", gf)):
        x = {k: scored(g / sub, task) for k, sub in (("k8", "full_k8"), ("k1", "full_k8/k1"), ("k1b", "full_k8/k1b"),
                                                    ("prior", "prior_k1"), ("wrong", "wrong_k1"),
                                                    ("greedy", "full_greedy"))}
        if x["k8"] is not None and x["k8"].get("checkpoint") and \
                not str(x["k8"]["checkpoint"]).endswith(sel["selected"]["checkpoint"]):
            raise RuntimeError(f"{name}: K = 8 maps from {x['k8']['checkpoint']}, the rule selected "
                               f"{sel['selected']['checkpoint']}")
        if None not in (x["wrong"], x["prior"], x["k1"], x["k1b"]):
            mw, mp, mf, mf2 = x["wrong"]["mIoU"], x["prior"]["mIoU"], x["k1"]["mIoU"], x["k1b"]["mIoU"]
            x["z9"] = {"check_pass": bool(abs(mw - mp) < abs(mw - mf)),
                       "control_fails_check": bool(not (abs(mf2 - mp) < abs(mf2 - mf)))}
        r[p] = x
    return r


def summary(rows: list) -> dict:
    out = {}
    for p in ("bf16", "fp32"):
        v = [r[p]["k8"]["mIoU"] for r in rows if not r["missing"] and r[p]["k8"] is not None]
        out[p] = {"n": len(v), "per_seed": v, "mean": float(np.mean(v)) if v else None,
                  "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                  "range": [float(min(v)), float(max(v))] if v else None}
    z8b = [r["z8_bf16"]["pass"] for r in rows if not r["missing"] and r["z8_bf16"] is not None]
    out["arm_alone_precision"] = ("bf16" if len(z8b) == len(SEEDS) and all(z8b) else
                                 "fp32" if len(z8b) == len(SEEDS) else None)
    fr = [r["fp32_readable"] for r in rows if not r["missing"] and r["fp32_readable"] is not None]
    out["fp32_all_readable"] = all(fr) if len(fr) == len(SEEDS) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rts-c12", default=None)
    ap.add_argument("--rts-p44", default=None)
    ap.add_argument("--arm", choices=("r0", "r1"), default="r0")
    ap.add_argument("--rts-r1", default=None)
    a = ap.parse_args()
    rts = {"c12": a.rts_c12, "p44": a.rts_p44, "r1": a.rts_r1}
    A = a.arm.upper()
    res = {"what": f"A2 Step 5.2, {A}'s arm (plan Step 5.2 items 1-11); deciding nothing", "arm": a.arm, "tasks": {}}
    lines = [f"# A2 Step 5.2, {A}'s arm", "",
             "| task | seed | run | selected (CE) | plateau epoch | Z8 bf16 (steps) | Z8 fp32 | K = 8 bf16 | K = 8 fp32 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task in ("pastis", "cloudsen12"):
        rows = [read_run(task, s, rts, a.arm) for s in SEEDS]
        res["tasks"][task] = {"runs": rows, "summary": summary(rows)}
        for r in rows:
            if r["missing"]:
                lines.append(f"| {task} | {r['seed']} | {r['run'] or '-'} | missing | | | | | |")
                continue
            ps = r["plateau_stop"] or {}
            fired = (ps.get("fired") or {}).get("epoch")
            zb, zf = r["z8_bf16"], r["z8_fp32"]
            zbs = "-" if zb is None else f"{'pass' if zb['pass'] else 'FAIL'} ({zb['max_ulp_diff']:.2f})"
            zfs = "-" if zf is None else ("pass" if zf["pass"] else
                                          f"fail, rounding ({r['fp32_control_ratio']:,.0f}x below its control)"
                                          if r["fp32_readable"] else "FAIL")
            kb = "-" if r["bf16"]["k8"] is None else f"{r['bf16']['k8']['mIoU']:.4f}"
            kf = "-" if r["fp32"]["k8"] is None else f"{r['fp32']['k8']['mIoU']:.4f}"
            sl = r["selected"]
            lines.append(f"| {task} | {r['seed']} | `{r['run']}` | {sl['checkpoint']} ({sl['eval_ce']:.3f}, epoch "
                         f"{sl['epoch']}) | {fired if fired is not None else 'none'} | {zbs} | {zfs} | {kb} | {kf} |")
    lines += ["", f"| task | precision | n | mean K = 8 | SD | range | {A}-alone precision (item 5) | every run readable at "
              "fp32 (item 5's fp32 branch) |", "|---|---|---|---|---|---|---|---|"]
    for task, t in res["tasks"].items():
        s = t["summary"]
        for p in ("bf16", "fp32"):
            x = s[p]
            if x["n"]:
                sd = "-" if x["sd"] is None else f"{x['sd']:.4f}"
                lines.append(f"| {task} | {p} | {x['n']} | {x['mean']:.4f} | {sd} | {x['range'][0]:.4f}-"
                             f"{x['range'][1]:.4f} | {s['arm_alone_precision']} | {s['fp32_all_readable']} |")
    for task, t in res["tasks"].items():
        for r in t["runs"]:
            for p in ("bf16", "fp32"):
                x = r.get(p) or {}
                if "z9" in x:
                    lines.append(f"- {task} seed {r['seed']} at {p}: prior K = 1 {x['prior']['mIoU']:.4f}, wrong input "
                                 f"{x['wrong']['mIoU']:.4f}, own input K = 1 {x['k1']['mIoU']:.4f} (second sample "
                                 f"{x['k1b']['mIoU']:.4f}); Z9 {'pass' if x['z9']['check_pass'] else 'FAIL'}, control "
                                 f"{'fails it' if x['z9']['control_fails_check'] else 'PASSES it'}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.arm}_read.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (OUT / f"{a.arm}_read.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
