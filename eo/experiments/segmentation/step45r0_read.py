#!/usr/bin/env python
"""A2 Step 4.5 item 10: R0's arm at 784, read by Step 5.2's protocol (docs/plans/a2_segmentation.md, Step 4, "4.5, the
grid decision", item 10, written 2026-11-24 before any CloudSen12 784 map; json only, any env with numpy).

Runs only if 4.5's rule decided (c). The arm is Step 5.2's R0 arm with the 784 overlays (eo_seg/s43_grid784 on PASTIS-1
fold 4, eo_seg/s44_grid784_cloudsen12 on CloudSen12 val), seeds 42, 43 and 44. The runs 4.3 and 4.5 already made at
this recipe count (item 10, as Step 5.2's item 2 counted Step 4.1's):
  PASTIS 42 g784s43_s42_20261123_175739, 43 g784s43_s43_20261124_113201 (step43/gen); 44 g784s43_s44_<stamp>;
  CloudSen12 42 g784cs45_s42_20261124_114054 (step45/gen); 43 and 44 g784cs45_s<seed>_<stamp>.
Each run is read by step52_read.read_run (the records, both Z8s, item 5's fp32 branch, K = 8 at both precisions, Z9
where its controls exist) and each task by step52_read.summary (mean, SD, range, the arm's precision). Reported beside,
deciding nothing: 784 - 196 per seed and in the mean, at each precision, against Step 5.2's R0 read
(/data/enric/reports/a2_seg/step52/r0_read.json). Writes /data/enric/reports/a2_seg/step45/r0_784_read.{json,md}.

    python eo/experiments/segmentation/step45r0_read.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step52_read as S52                  # noqa: E402

RUNS = S52.RUNS
GEN43 = Path("/data/enric/reports/a2_seg/step43/gen")
GEN45 = Path("/data/enric/reports/a2_seg/step45/gen")
OUT = Path("/data/enric/reports/a2_seg/step45")
R0_196 = Path("/data/enric/reports/a2_seg/step52/r0_read.json")
FIXED = {("pastis", 42): "g784s43_s42_20261123_175739", ("pastis", 43): "g784s43_s43_20261124_113201",
         ("cloudsen12", 42): "g784cs45_s42_20261124_114054"}
PREFIX = {"pastis": ("g784s43", GEN43), "cloudsen12": ("g784cs45", GEN45)}


def find_run(task: str, seed: int, rts: dict, arm: str = "r0") -> tuple:
    prefix, gen = PREFIX[task]
    if (task, seed) in FIXED:
        return FIXED[(task, seed)], gen
    hits = sorted(p.name for p in RUNS.glob(f"{prefix}_s{seed}_*") if (p / "select41.json").exists())
    if len(hits) > 1:
        raise RuntimeError(f"{task} seed {seed}: several runs {hits}")
    return (hits[0], gen) if hits else (None, gen)


def main():
    S52.find_run = find_run          # read_run resolves its run through the module's find_run
    ref = json.loads(R0_196.read_text())["tasks"]
    res = {"what": "A2 Step 4.5 item 10: R0's arm at 784 by Step 5.2's protocol; 784 - 196 beside, deciding nothing",
           "tasks": {}}
    lines = ["# A2 Step 4.5 item 10: R0's arm at 784", "",
             "| task | seed | run | selected (CE) | plateau epoch | Z8 bf16 (steps) | Z8 fp32 | K = 8 bf16 | K = 8 fp32 "
             "| 196's K = 8 bf16 / fp32 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for task in ("pastis", "cloudsen12"):
        rows = [S52.read_run(task, s, {}, "r0") for s in S52.SEEDS]
        r196 = {r["seed"]: r for r in ref[task]["runs"]}
        summ = S52.summary(rows)
        beside = {}
        for p in ("bf16", "fp32"):
            d = {r["seed"]: (r[p]["k8"]["mIoU"] - r196[r["seed"]][p]["k8"]["mIoU"]) * 100
                 for r in rows if not r["missing"] and r[p]["k8"] is not None
                 and (r196.get(r["seed"]) or {}).get(p, {}).get("k8")}
            beside[p] = {"per_seed_points": d, "mean_points": float(np.mean(list(d.values()))) if d else None}
        res["tasks"][task] = {"runs": rows, "summary": summ, "summary_196": ref[task]["summary"],
                              "minus_196": beside}
        for r in rows:
            if r["missing"]:
                lines.append(f"| {task} | {r['seed']} | {r['run'] or '-'} | missing | | | | | | |")
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
            o = r196.get(r["seed"]) or {}
            ob = "-" if not (o.get("bf16") or {}).get("k8") else f"{o['bf16']['k8']['mIoU']:.4f}"
            of = "-" if not (o.get("fp32") or {}).get("k8") else f"{o['fp32']['k8']['mIoU']:.4f}"
            sl = r["selected"]
            lines.append(f"| {task} | {r['seed']} | `{r['run']}` | {sl['checkpoint']} ({sl['eval_ce']:.3f}, epoch "
                         f"{sl['epoch']}) | {fired if fired is not None else 'none'} | {zbs} | {zfs} | {kb} | {kf} "
                         f"| {ob} / {of} |")
    lines += ["", "| task | precision | n | mean K = 8 | SD | range | arm's precision (item 5) | all readable at fp32 "
              "| 196's mean | 784 - 196, mean (points) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for task, t in res["tasks"].items():
        s = t["summary"]
        for p in ("bf16", "fp32"):
            x = s[p]
            if x["n"]:
                sd = "-" if x["sd"] is None else f"{x['sd']:.4f}"
                m196 = t["summary_196"][p]["mean"]
                mb = t["minus_196"][p]["mean_points"]
                lines.append(f"| {task} | {p} | {x['n']} | {x['mean']:.4f} | {sd} | {x['range'][0]:.4f}-"
                             f"{x['range'][1]:.4f} | {s['arm_alone_precision']} | {s['fp32_all_readable']} | "
                             f"{m196:.4f} | {'-' if mb is None else f'{mb:+.2f}'} |")
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
    (OUT / "r0_784_read.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (OUT / "r0_784_read.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
