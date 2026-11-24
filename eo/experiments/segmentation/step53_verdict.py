#!/usr/bin/env python
"""A2 Step 5.3: R0 against R1, by Step 5's rule (docs/plans/a2_segmentation.md, "The test (Step 5) and its rule, fixed
now", and Step 5.2's "R1's arm, and Step 5.3's arithmetic", written 2026-11-23 before any R1 number; `.venv`, CPU).

Reads both arms' reads (step52_read.py: r0_read.json, r1_read.json) and the per-patch (PASTIS) or per-tile
(CloudSen12) confusion matrices of their K = 8 maps. Per task:
  * the precision (item 5): bf16 iff all six selected checkpoints pass Z8 at bf16 (4 rounding steps); otherwise fp32
    for all six, each run readable by the gate's fp32 rule or the 1,000x rule (Enric, 2026-11-22); a run readable by
    neither stops the read (the rule does not say what then: Enric decides);
  * Delta = m(R1) - m(R0) in mIoU points, m the mean K = 8 mIoU of an arm's three seeds at that precision;
  * the pooled seed SD s_p = sqrt((s(R0)^2 + s(R1)^2) / 2), each s the sample SD (n - 1) of an arm's three seeds;
  * the bar max(1.0, 2 s_p) points.
Verdict: adopt R1 iff Delta >= its bar on at least one task, Delta >= -0.5 on both, and R1's forgetting <= +0.02 nats
(R1's rule 4, read 2026-11-23); otherwise R0.
Reported beside it, deciding nothing: the per-seed differences, the other precision's means, a paired bootstrap of the
three-seed mean difference over patches or tiles (1,000 resamples, seed 0: Step 4.1's form), per-class IoU, the plateau
epochs and kept saves, Z9 and the controls, and whether R1 clears 2 points on a task (R2's condition).

    python eo/experiments/segmentation/step53_verdict.py [--out /data/enric/reports/a2_seg/step53]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

READS = Path("/data/enric/reports/a2_seg/step52")
GEN41 = Path("/data/enric/reports/a2_seg/step41/gen")
GEN52 = Path("/data/enric/reports/a2_seg/step52/gen")
FORGET = Path("/data/enric/reports/a2_seg/r1/forgetting_r1_checkpoint-4000.json")
TASKS = ("pastis", "cloudsen12")
IGNORE = {"pastis": 19, "cloudsen12": None}
FLOOR, SD_MULT, NOT_WORSE, FORGET_MARGIN, R2_BAR = 1.0, 2.0, -0.5, 0.02, 2.0     # points, x s_p, points, nats, points
N_BOOT, BOOT_SEED = 1000, 0


def iou_per_class(cm: np.ndarray, ignore) -> np.ndarray:
    """Per-class IoU as eo/maps/scoring computes it; nan for the ignored class and for classes absent in target and
    prediction (so the macro mean over the finite entries is the scorer's mIoU)."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), np.nan)
    if ignore is not None:
        iou[ignore] = np.nan
    return iou


def miou(cm: np.ndarray, ignore) -> float:
    return float(np.nanmean(iou_per_class(cm, ignore)))


def maps_dir(name: str, prec: str) -> Path:
    gen = GEN41 if name.startswith("r0p1s41") else GEN52
    return gen / (name if prec == "bf16" else f"{name}_fp32") / "full_k8"


def task_verdict(task: str, rows0: list, rows1: list) -> dict:
    six = rows0 + rows1
    if any(r["missing"] for r in six):
        return {"complete": False, "missing": [r["run"] for r in six if r["missing"]]}
    z8b = {r["run"]: bool(r["z8_bf16"]["pass"]) for r in six}
    prec = "bf16" if all(z8b.values()) else "fp32"
    out = {"complete": True, "precision": prec, "z8_bf16_pass": z8b}
    if prec == "fp32":
        readable = {r["run"]: bool(r["fp32_readable"]) for r in six}
        out["fp32_readable"] = readable
        if not all(readable.values()):
            out["stopped"] = "a run is readable at fp32 by neither the gate's rule nor the 1,000x rule: Enric decides"
            return out
    arms = {}
    for arm, rows in (("R0", rows0), ("R1", rows1)):
        v = np.array([r[prec]["k8"]["mIoU"] for r in rows]) * 100
        o = np.array([r["fp32" if prec == "bf16" else "bf16"]["k8"]["mIoU"] for r in rows]) * 100
        arms[arm] = {"runs": [r["run"] for r in rows], "seeds": [r["seed"] for r in rows], "points": v.tolist(),
                     "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                     "other_precision_points": o.tolist(), "other_precision_mean": float(o.mean())}
    if arms["R0"]["seeds"] != arms["R1"]["seeds"]:
        raise RuntimeError(f"{task}: the arms' seeds differ {arms['R0']['seeds']} {arms['R1']['seeds']}")
    sp = math.sqrt((arms["R0"]["sd"] ** 2 + arms["R1"]["sd"] ** 2) / 2)
    delta = arms["R1"]["mean"] - arms["R0"]["mean"]
    bar = max(FLOOR, SD_MULT * sp)
    out.update({"arms": arms, "delta": delta, "pooled_sd": sp, "bar": bar,
                "clears_bar": bool(delta >= bar), "not_worse": bool(delta >= NOT_WORSE),
                "per_seed_delta": [b - a for a, b in zip(arms["R0"]["points"], arms["R1"]["points"])],
                "other_precision_delta": arms["R1"]["other_precision_mean"] - arms["R0"]["other_precision_mean"],
                "clears_r2_bar": bool(delta >= R2_BAR)})
    # beside it: the paired bootstrap over patches or tiles, per-class IoU, the runs' records
    cms = {arm: [np.load(maps_dir(r["run"], prec) / "per_patch_cm.npy") for r in rows]
           for arm, rows in (("R0", rows0), ("R1", rows1))}
    n = {c.shape[0] for v in cms.values() for c in v}
    if len(n) != 1:
        raise RuntimeError(f"{task}: the runs' maps cover different patch counts {n}")
    n = n.pop()
    for arm, rows in (("R0", rows0), ("R1", rows1)):
        for c, r in zip(cms[arm], rows):
            if abs(miou(c.sum(0), IGNORE[task]) * 100 - arms[arm]["points"][rows.index(r)]) > 1e-9:
                raise RuntimeError(f"{r['run']}: its per-sample matrices do not sum to its read mIoU")
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    stat = {arm: np.array([np.mean([miou(c[i].sum(0), IGNORE[task]) for c in v]) for i in idx]) * 100
            for arm, v in cms.items()}
    d = stat["R1"] - stat["R0"]
    out["bootstrap"] = {"n": n, "resamples": N_BOOT, "seed": BOOT_SEED, "mean": float(d.mean()),
                        "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
                        "share_positive": float((d > 0).mean())}
    from eo.maps import palette as P
    names = P.TASKS[task].class_names
    pc = {arm: np.nanmean(np.stack([iou_per_class(c.sum(0), IGNORE[task]) for c in v]), axis=0) * 100
          for arm, v in cms.items()}
    out["per_class"] = [{"class": names[k], "R0": float(pc["R0"][k]), "R1": float(pc["R1"][k]),
                         "delta": float(pc["R1"][k] - pc["R0"][k])}
                        for k in range(len(names)) if not np.isnan(pc["R0"][k]) or not np.isnan(pc["R1"][k])]
    out["records"] = {arm: [{"run": r["run"], "seed": r["seed"], "kept": r["selected"]["checkpoint"],
                             "kept_epoch": r["selected"]["epoch"], "kept_ce": r["selected"]["eval_ce"],
                             "plateau_epoch": ((r.get("plateau_stop") or {}).get("fired") or {}).get("epoch"),
                             "z8_bf16_steps": r["z8_bf16"]["max_ulp_diff"], "z8_fp32_pass": r["z8_fp32"]["pass"],
                             "fp32_control_ratio": r["fp32_control_ratio"],
                             "z9": {p: r[p].get("z9") for p in ("bf16", "fp32") if r[p].get("z9")},
                             "controls": {p: {k: r[p][k]["mIoU"] for k in ("prior", "wrong", "k1", "k1b", "greedy")
                                              if r[p].get(k)} for p in ("bf16", "fp32")}}
                            for r in rows] for arm, rows in (("R0", rows0), ("R1", rows1))}
    return out


def decide(deltas: dict, forgetting: float) -> tuple:
    """Step 5's rule on {task: (Delta, bar)} in points and R1's forgetting in nats -> (verdict, conditions)."""
    c = {"clears_bar_on_a_task": any(d >= b for d, b in deltas.values()),
         "not_worse_on_both": all(d >= NOT_WORSE for d, _ in deltas.values()),
         "forgetting_within_margin": bool(forgetting <= FORGET_MARGIN)}
    return ("R1" if all(c.values()) else "R0"), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/enric/reports/a2_seg/step53")
    ap.add_argument("--reads", default=str(READS), help="the directory holding r0_read.json and r1_read.json")
    a = ap.parse_args()
    r0 = json.loads((Path(a.reads) / "r0_read.json").read_text())
    r1 = json.loads((Path(a.reads) / "r1_read.json").read_text())
    if r0.get("arm", "r0") != "r0" or r1.get("arm") != "r1":
        raise RuntimeError("r0_read.json / r1_read.json are not the two arms' reads")
    fg = json.loads(FORGET.read_text())
    res = {"what": "A2 Step 5.3: R0 against R1 by Step 5's rule (plan Step 5.2, 'R1's arm, and Step 5.3's arithmetic')",
           "rule": {"floor_points": FLOOR, "sd_mult": SD_MULT, "not_worse_points": NOT_WORSE,
                    "forgetting_margin": FORGET_MARGIN, "r2_bar_points": R2_BAR},
           "forgetting": {"diff": fg["diff"], "ci95": fg["diff_ci95"], "admissible": bool(fg["diff"] <= FORGET_MARGIN),
                          "record": str(FORGET)},
           "tasks": {t: task_verdict(t, r0["tasks"][t]["runs"], r1["tasks"][t]["runs"]) for t in TASKS}}
    T = res["tasks"]
    if not all(T[t]["complete"] and "stopped" not in T[t] for t in TASKS):
        res["verdict"] = None
        res["why"] = "incomplete or stopped: " + "; ".join(
            f"{t}: {T[t].get('missing') or T[t].get('stopped')}" for t in TASKS if not T[t]["complete"] or "stopped" in T[t])
    else:
        res["verdict"], res["conditions"] = decide({t: (T[t]["delta"], T[t]["bar"]) for t in TASKS},
                                                   res["forgetting"]["diff"])
        res["r2_condition_points"] = bool(res["verdict"] == "R1" and any(T[t]["clears_r2_bar"] for t in TASKS))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    L = [f"# A2 Step 5.3: R0 against R1, verdict {res['verdict']}", ""]
    if res["verdict"] is None:
        L.append(res["why"])
    else:
        L += ["| task | precision | R0 mean (SD) | R1 mean (SD) | Delta | pooled SD | bar | clears | not worse |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t in TASKS:
            x = T[t]
            L.append(f"| {t} | {x['precision']} | {x['arms']['R0']['mean']:.2f} ({x['arms']['R0']['sd']:.2f}) | "
                     f"{x['arms']['R1']['mean']:.2f} ({x['arms']['R1']['sd']:.2f}) | {x['delta']:+.2f} | "
                     f"{x['pooled_sd']:.2f} | {x['bar']:.2f} | {x['clears_bar']} | {x['not_worse']} |")
        L += ["", f"forgetting {res['forgetting']['diff']:+.4f} nats (margin +{FORGET_MARGIN}): "
                  f"{res['forgetting']['admissible']}; conditions {res['conditions']}; R2's 2-point condition "
                  f"{res['r2_condition_points']}", ""]
        for t in TASKS:
            x = T[t]
            b = x["bootstrap"]
            L.append(f"- {t}: per seed {[round(v, 2) for v in x['per_seed_delta']]}; other precision Delta "
                     f"{x['other_precision_delta']:+.2f}; paired bootstrap over {b['n']} {b['mean']:+.2f} "
                     f"[{b['lo95']:+.2f}, {b['hi95']:+.2f}], positive {b['share_positive']:.3f}")
            L.append("  per class: " + "; ".join(f"{c['class']} {c['R0']:.1f} -> {c['R1']:.1f} ({c['delta']:+.1f})"
                                                 for c in x["per_class"]))
    (out / "verdict.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
