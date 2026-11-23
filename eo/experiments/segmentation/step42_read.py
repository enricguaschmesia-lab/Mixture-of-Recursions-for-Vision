#!/usr/bin/env python
"""A2 Step 4.2: Routes B and C read by the rule pre-registered in the plan (docs/plans/a2_segmentation.md, Step 4.2,
items 1-9; json and numpy only).

  * Route A is Step 5.2's R0 arm as read (step52/r0_read.json): K = 8 at bf16 over seeds 42-44, PASTIS-1 fold 4 0.1322,
    CloudSen12 val 0.6188 (checked against the pre-registered values).
  * Route C's head LR (item 3): PASTIS-1 seed 42 at seg_head.lr 3e-4, 1e-3 and 3e-3; the highest mIoU (bf16, the
    selected save) wins, and that run is C's PASTIS seed 42.
  * Per route, task and seed: the run's records (select41.json: the lowest-CE save, the plateau stop, health), Route B's
    Z8 at both precisions (step41_read.z8_of) and maps (K = 8; K = 1 from its first sample; greedy, prior and wrong where
    run), Route C's head maps at bf16 (deciding) and fp32, and its wrong-input control.
  * Route B's precision per task (item 2): bf16 iff all three selected checkpoints pass Z8 at bf16; otherwise fp32, with
    Step 5.2's 1,000x rule for a run failing the gate's fp32 rule.
  * THE RULE (item 5): a route replaces A iff its three-seed mean exceeds A's by at least 2.0 points (0.020) on PASTIS-1
    fold 4 AND on CloudSen12 val, compared at full precision. Otherwise it is an ablation row.
  * Beside it (item 6): SDs, the paired bootstrap of the three-seed mean difference from A over patches or tiles (1,000
    resamples, seed 0), per-class IoU, the ceilings, B's controls with Z9's functional check, the A100-hours (cell logs).
Writes /data/enric/reports/a2_seg/step42/read.{json,md}.

    python eo/experiments/segmentation/step42_read.py --rts <stamp> [--rts-lr <stamp of the head-LR cells>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step41_read import z8_of                       # noqa: E402
from step52_read import find_run as r0_run, miou, scored   # noqa: E402

from eo.eval.seg_routes import bootstrap_mean_diff  # noqa: E402

RUNS = Path("/data/enric/runs/pretrain/a2_seg")
GEN = Path("/data/enric/reports/a2_seg/step42/gen")
OUT = Path("/data/enric/reports/a2_seg/step42")
R0_READ = Path("/data/enric/reports/a2_seg/step52/r0_read.json")
IGNORE = {"pastis": 19, "cloudsen12": None}
SEEDS = (42, 43, 44)
A_PREREG = {"pastis": 0.1322, "cloudsen12": 0.6188}
BAR = 0.020
HEAD_LRS = ("3e-4", "1e-3", "3e-3")
FP32_ROUNDING = 1000.0
TAG = {("B", "pastis"): "b42p1", ("B", "cloudsen12"): "b42c12", ("C", "pastis"): "c42p1", ("C", "cloudsen12"): "c42c12"}


def run_name(route: str, task: str, seed: int, rts: str, hlr: str = None) -> str:
    t = TAG[(route, task)]
    return f"{t}_s{seed}_{rts}" if route == "B" else f"{t}_s{seed}_h{hlr}_{rts}"


def cell_hours(name: str):
    """A100-hours of one job: its cell log's first '## <time>' line to its 'cell done' line (every attempt summed)."""
    f = RUNS / name / "joblog" / f"cell-{name}.log"
    if not f.exists():
        return None
    ts = [datetime.fromisoformat(m.group(1)) for m in re.finditer(r"^## (\S+) ", f.read_text(errors="replace"), re.M)]
    return None if len(ts) < 2 else (ts[-1] - ts[0]).total_seconds() / 3600.0


def records(name: str) -> dict:
    sel = json.loads((RUNS / name / "select41.json").read_text())
    return {"selected": sel["selected"], "plateau_stop": sel.get("plateau_stop"), "health": sel.get("health")}


def read_b(task: str, seed: int, rts: str) -> dict:
    name = run_name("B", task, seed, rts)
    if not (RUNS / name / "select41.json").exists():
        return {"task": task, "seed": seed, "run": name, "missing": True}
    gb, gf = GEN / name, GEN / f"{name}_fp32"
    r = {"task": task, "seed": seed, "run": name, "missing": False, **records(name),
         "z8_bf16": z8_of(gb, "bf16"), "z8_fp32": z8_of(gb, "fp32"), "a100_hours": cell_hours(name)}
    zf = r["z8_fp32"]
    r["fp32_readable"] = None if zf is None else bool(zf["pass"] or zf["max_abs_logit_diff"] * FP32_ROUNDING <= zf["control_abs"])
    for p, g in (("bf16", gb), ("fp32", gf)):
        x = {k: scored(g / sub, task) for k, sub in (("k8", "full_k8"), ("k1", "full_k8/k1"), ("k1b", "full_k8/k1b"),
                                                    ("prior", "prior_k1"), ("wrong", "wrong_k1"), ("greedy", "full_greedy"))}
        if x["k8"] is not None and x["k8"].get("checkpoint") and \
                not str(x["k8"]["checkpoint"]).endswith(r["selected"]["checkpoint"]):
            raise RuntimeError(f"{name}: K = 8 maps from {x['k8']['checkpoint']}, the rule selected {r['selected']['checkpoint']}")
        if None not in (x["wrong"], x["prior"], x["k1"], x["k1b"]):
            mw, mp, mf, mf2 = x["wrong"]["mIoU"], x["prior"]["mIoU"], x["k1"]["mIoU"], x["k1b"]["mIoU"]
            x["z9"] = {"check_pass": bool(abs(mw - mp) < abs(mw - mf)),
                       "control_fails_check": bool(not (abs(mf2 - mp) < abs(mf2 - mf)))}
        x["cm_path"] = str(g / "full_k8" / "per_patch_cm.npy") if x["k8"] is not None else None
        r[p] = x
    return r


def read_c(task: str, seed: int, rts: str, hlr: str) -> dict:
    name = run_name("C", task, seed, rts, hlr)
    if not (RUNS / name / "select41.json").exists():
        return {"task": task, "seed": seed, "run": name, "head_lr": hlr, "missing": True}
    gb, gf = GEN / name, GEN / f"{name}_fp32"
    r = {"task": task, "seed": seed, "run": name, "head_lr": hlr, "missing": False, **records(name),
         "a100_hours": cell_hours(name)}
    for p, g in (("bf16", gb), ("fp32", gf)):
        x = {"head": _scored_head(g / "head", task), "wrong": _scored_head(g / "head_wrong", task)}
        if x["head"] is not None and not str(x["head"]["checkpoint"]).endswith(r["selected"]["checkpoint"]):
            raise RuntimeError(f"{name}: head maps from {x['head']['checkpoint']}, the rule selected {r['selected']['checkpoint']}")
        x["cm_path"] = str(g / "head" / "per_patch_cm.npy") if x["head"] is not None else None
        r[p] = x
    return r


def _scored_head(d: Path, task: str):
    """A head-predict directory: the RCP score and the workstation's re-score (rescore/) must agree."""
    if not (d / "score.json").exists():
        return None
    s = json.loads((d / "score.json").read_text())
    cm = np.load(d / "per_patch_cm.npy")
    m = float(s["result"]["mIoU"])
    if abs(miou(cm.sum(0), IGNORE[task]) - m) > 1e-12:
        raise RuntimeError(f"{d}: per-sample matrices do not sum to the scored mIoU")
    out = {"mIoU": m, "pixel_acc": float(s["result"]["pixel_acc"]), "boundary_acc": float(s["result"]["boundary_acc"]),
           "pred_top_share": float(s["pred_top_share"]), "samples": int(cm.shape[0]),
           "checkpoint": s["source"]["checkpoint"], "precision": s["source"]["precision"]}
    rs = d / "rescore" / "score.json"
    out["rescore_equal"] = None if not rs.exists() else bool(
        abs(json.loads(rs.read_text())["result"]["mIoU"] - m) < 1e-12)
    return out


def mean_sd(v):
    return (float(np.mean(v)) if v else None, float(np.std(v, ddof=1)) if len(v) > 1 else None)


def per_class(cm_paths, ignore):
    """Per-class IoU of each seed's summed matrix, then the mean over seeds."""
    ious = []
    for p in cm_paths:
        cm = np.load(p).sum(0).astype(np.float64)
        tp = np.diag(cm)
        den = cm.sum(0) + cm.sum(1) - tp
        iou = np.where(den > 0, tp / np.where(den > 0, den, 1), np.nan)
        if ignore is not None:
            iou[ignore] = np.nan
        ious.append(iou)
    return np.nanmean(np.stack(ious), 0).tolist() if ious else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rts", required=True, help="the stamp of the main runs")
    ap.add_argument("--rts-lr", default=None, help="the stamp of Route C's head-LR cells (default --rts)")
    a = ap.parse_args()
    rts_lr = a.rts_lr or a.rts
    r0 = json.loads(R0_READ.read_text())
    A, A_cms = {}, {}
    for task in ("pastis", "cloudsen12"):
        s = r0["tasks"][task]["summary"]["bf16"]
        A[task] = float(s["mean"])
        if round(A[task], 4) != A_PREREG[task]:
            raise RuntimeError(f"R0's {task} mean {A[task]} is not the pre-registered {A_PREREG[task]}")
        A_cms[task] = []
        for seed in SEEDS:
            name, gen = r0_run(task, seed, {}, "r0")
            A_cms[task].append(np.load(gen / name / "full_k8" / "per_patch_cm.npy"))
    res = {"what": "A2 Step 4.2: Routes B and C against Step 5.2's R0 arm, by the pre-registered rule", "A": A,
           "bar_points": 100 * BAR, "routes": {}}

    # item 3: Route C's head LR
    lr_cells = {h: read_c("pastis", 42, rts_lr, h) for h in HEAD_LRS}
    scored_lr = {h: r["bf16"]["head"]["mIoU"] for h, r in lr_cells.items()
                 if not r["missing"] and r["bf16"]["head"] is not None}
    hlr = max(scored_lr, key=lambda h: (scored_lr[h], -float(h))) if len(scored_lr) == len(HEAD_LRS) else None
    res["head_lr"] = {"cells": lr_cells, "mIoU": scored_lr, "chosen": hlr}

    for route in ("B", "C"):
        out = {"tasks": {}}
        for task in ("pastis", "cloudsen12"):
            if route == "B":
                rows = [read_b(task, s, a.rts) for s in SEEDS]
                z8b = [r["z8_bf16"]["pass"] for r in rows if not r["missing"] and r["z8_bf16"] is not None]
                prec = "bf16" if len(z8b) == 3 and all(z8b) else ("fp32" if len(z8b) == 3 else None)
                key = "k8"
            else:
                rows = []
                for s in SEEDS:
                    if hlr is None:
                        rows.append({"task": task, "seed": s, "missing": True, "note": "head LR not chosen yet"})
                    elif task == "pastis" and s == 42:
                        rows.append(lr_cells[hlr])
                    else:
                        rows.append(read_c(task, s, a.rts, hlr))
                prec, key = "bf16", "head"
            got = [r for r in rows if not r["missing"] and prec and r[prec][key] is not None]
            v = [r[prec][key]["mIoU"] for r in got]
            m, sd = mean_sd(v)
            t = {"runs": rows, "precision": prec, "per_seed": v, "mean": m, "sd": sd, "n": len(v)}
            if len(v) == 3:
                t["delta_points"] = 100 * (m - A[task])
                t["clears_bar"] = bool(m - A[task] >= BAR)
                cms = [np.load(r[prec]["cm_path"]) for r in got]
                t["bootstrap_vs_A"] = bootstrap_mean_diff(A_cms[task], cms, IGNORE[task])
                t["per_class_iou"] = per_class([r[prec]["cm_path"] for r in got], IGNORE[task])
            other = "fp32" if prec == "bf16" else "bf16"
            vo = [r[other][key]["mIoU"] for r in rows if not r["missing"] and r.get(other, {}).get(key) is not None]
            t["other_precision"] = {"precision": other, "per_seed": vo, "mean": mean_sd(vo)[0]}
            hrs = [r.get("a100_hours") for r in rows if not r["missing"] and r.get("a100_hours") is not None]
            t["a100_hours"] = float(sum(hrs)) if hrs else None
            out["tasks"][task] = t
        ok = [out["tasks"][k].get("clears_bar") for k in ("pastis", "cloudsen12")]
        out["verdict"] = (None if None in ok else ("replaces A" if all(ok) else "an ablation row"))
        res["routes"][route] = out
    lr_hours = [r.get("a100_hours") for h, r in lr_cells.items() if h != hlr and not r["missing"] and r.get("a100_hours")]
    res["a100_hours"] = {"B": sum(filter(None, (res["routes"]["B"]["tasks"][k]["a100_hours"] for k in ("pastis", "cloudsen12")))),
                         "C": sum(filter(None, (res["routes"]["C"]["tasks"][k]["a100_hours"] for k in ("pastis", "cloudsen12")))),
                         "C_lr_cells_other": float(sum(lr_hours))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "read.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    lines = ["# A2 Step 4.2: Routes B and C against Step 5.2's R0 arm", "",
             f"Route A (R0, K = 8, bf16): PASTIS-1 fold 4 {A['pastis']:.4f}, CloudSen12 val {A['cloudsen12']:.4f}. "
             f"Bar: +{100 * BAR:.1f} points on both.", "",
             f"Route C's head LR (PASTIS-1 seed 42, bf16): " + ", ".join(f"{h}: {v:.4f}" for h, v in scored_lr.items())
             + f" -> **{hlr}**", ""]
    for route, o in res["routes"].items():
        lines += ["| route | task | precision | per seed | mean | SD | minus A (points) [95%] | clears 2 points |",
                  "|---|---|---|---|---|---|---|---|"]
        for task, t in o["tasks"].items():
            if t["n"] == 0:
                lines.append(f"| {route} | {task} | {t['precision']} | - | - | - | - | - |")
                continue
            b = t.get("bootstrap_vs_A")
            bi = "-" if b is None else f"{100 * b['mean_diff']:+.2f} [{100 * b['lo']:+.2f}, {100 * b['hi']:+.2f}]"
            lines.append(f"| {route} | {task} | {t['precision']} | {', '.join(f'{x:.4f}' for x in t['per_seed'])} | "
                         f"{t['mean']:.4f} | {t['sd'] if t['sd'] is None else round(t['sd'], 4)} | {bi} | "
                         f"{t.get('clears_bar', '-')} |")
        lines.append(f"\n**Route {route}: {o['verdict']}**\n")
    (OUT / "read.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
