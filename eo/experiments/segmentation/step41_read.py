#!/usr/bin/env python
"""A2 Step 4.1: the recipe search's read, by the rule pre-registered on 2026-11-22 (docs/plans/a2_segmentation.md,
Step 4.1, items 6-11 and 13). numpy and json only; reads what the RCP cells (a2s41_cell.sh: step41_select.py, eval_seg.py
cachecheck and generate) and step4_score.py (a2seg) wrote:

  /data/enric/runs/pretrain/a2_seg/r0p1s41_<tag>_s<seed>_<stamp>/select41.json      the run's checkpoint, plateau, health
  /data/enric/reports/a2_seg/step41/gen/<run>/z8_bf16.json                          Z8 at bf16 on the selected checkpoint
  .../gen/<run>/full_k8/        K = 8 at T 1.0 seed 0: THE DECIDING NUMBER (score.json, per_patch_cm.npy; k1/ = sample 0)
  .../gen/<run>/full_greedy/    greedy
  (the recipe's runs also: full_k16/ with k1 k2 k4 k8 k16 subsets, prior_k1/, wrong_k1/, final_k8/)

  Stage A (item 8): the nine seed-42 cells. A failed run (item 7) or a missing number cannot lead.
  Contenders (item 9): the leader (the highest K = 8); in each other regime, its best cell if within 1.0 point of the
    leader (leader - m <= 0.010, inclusive, full precision); the runner-up if no other cell is within 1.0 point.
  The recipe (item 10): among the contenders, each with its seed-42 and seed-43 runs, the highest mean K = 8; among the
    contenders whose mean is within 1.0 point of it, the regime with the fewest trainable parameters (LoRA r16, then
    r64, then full), then the higher mean.
  Decoding (item 11): K = 8 unless greedy's mean (over the recipe's two runs) is within 0.5 point of K = 8's or above.
  Reported, not deciding: the bootstrap (1,000 resamples of the 482 patches, seed 0, the same for every map set) of
    every seed-42 cell against Step 4.0's aligned maps, and of each contender's two-seed mean against the recipe's.

    python eo/experiments/segmentation/step41_read.py --rts <grid stamp> [--out /data/enric/reports/a2_seg/step41]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS = Path("/data/enric/runs/pretrain/a2_seg")
GEN = Path("/data/enric/reports/a2_seg/step41/gen")
STEP40 = Path("/data/enric/reports/a2_seg/step4/gen/aligned/full_k8")
CELLS = {"l16a": ("lora16", 3e-4), "l16b": ("lora16", 1e-3), "l16c": ("lora16", 3e-3),
         "l64a": ("lora64", 3e-4), "l64b": ("lora64", 1e-3), "l64c": ("lora64", 3e-3),
         "fula": ("full", 3e-5), "fulb": ("full", 1e-4), "fulc": ("full", 3e-4)}
REGIME_ORDER = {"lora16": 0, "lora64": 1, "full": 2}            # fewest trainable parameters first
TRAINABLE = {"lora16": 51_014_208, "lora64": 65_179_200, "full": 195_782_400}
WITHIN, GREEDY_TOL = 0.010, 0.005
VOID, N_BOOT, BOOT_SEED = 19, 1000, 0


def miou(cm: np.ndarray, ignore=VOID) -> float:
    """eo/maps/scoring.Confusion.result's mIoU (as step4_read.miou)."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), 0.0)
    w = np.ones(len(cm))
    w[ignore] = 0
    w[(cm.sum(0) + cm.sum(1)) == 0] = 0
    return float((w * iou).sum() / w.sum())


def scored(d: Path):
    if not ((d / "score.json").exists() and (d / "per_patch_cm.npy").exists()):
        return None
    s = json.loads((d / "score.json").read_text())
    cm = np.load(d / "per_patch_cm.npy")
    m = float(s["result"]["mIoU"])
    if abs(miou(cm.sum(0)) - m) > 1e-12:
        raise RuntimeError(f"{d}: per-patch matrices do not sum to the scored mIoU")
    out = {"mIoU": m, "pixel_acc": float(s["result"]["pixel_acc"]), "boundary_acc": float(s["result"]["boundary_acc"]),
           "pred_top_share": float(s["pred_top_share"]), "K": int(s["K"]), "cm": cm}
    if (d / "slot_mass.npy").exists():
        out["slot_mass"] = float(np.load(d / "slot_mass.npy").mean())
    if (d / "record.json").exists():
        r = json.loads((d / "record.json").read_text())
        out.update({"checkpoint": r.get("checkpoint"), "precision": r.get("precision"), "gpu": r.get("gpu")})
    return out


def z8_of(g: Path, prec: str):
    f = g / f"z8_{prec}.json"
    if not f.exists():
        return None
    z = json.loads(f.read_text())
    # eval_seg.py cachecheck's own "pass" is X15's 4-step rule at either precision: item 4 reads it at bf16. At fp32 the
    # plan (item 15) reads the gate's fp32 rule, max(4 steps, 2 x the measured floor): the record's check.within
    ok = bool(z["pass"]) if prec == "bf16" else bool(z["check"]["within"] and z["check"]["codes_in_slot"])
    return {"pass": ok, "pass_4step": bool(z["pass"]), "control_fires": bool(z["control_fires"]),
            "max_ulp_diff": z["check"]["max_ulp_diff"],
            "max_abs_logit_diff": z["check"]["max_abs_logit_diff"], "max_abs_logit": z["check"]["max_abs_logit"],
            "tol": z["check"].get("tol"), "control_ulp": z["control_positions_one_off"]["max_ulp_diff"],
            "control_abs": z["control_positions_one_off"]["max_abs_logit_diff"]}


def run_of(tag: str, seed: int, rts: str, prec: str = "bf16"):
    """The run's records, its Z8 at both precisions (both in the bf16 directory) and its maps at `prec`."""
    name = f"r0p1s41_{tag}_s{seed}_{rts}"
    d, gb = RUNS / name, GEN / name
    g = gb if prec == "bf16" else GEN / f"{name}_fp32"
    if not (d / "select41.json").exists():
        return None
    sel = json.loads((d / "select41.json").read_text())
    r = {"run": name, "tag": tag, "seed": seed, "regime": CELLS[tag][0], "lr": CELLS[tag][1], "select": sel,
         "maps_precision": prec, "z8": z8_of(gb, "bf16"), "z8_fp32": z8_of(gb, "fp32")}
    for k, p in (("k8", "full_k8"), ("k1", "full_k8/k1"), ("greedy", "full_greedy"), ("k16", "full_k16"),
                 ("prior", "prior_k1"), ("wrong", "wrong_k1"), ("final_k8", "final_k8")):
        r[k] = scored(g / p)
    for k in (1, 2, 4, 8, 16):
        r[f"k16_{k}"] = scored(g / "full_k16" / f"k{k}")
    if r["k8"] is not None and r["k8"].get("checkpoint") and not str(r["k8"]["checkpoint"]).endswith(sel["selected"]["checkpoint"]):
        raise RuntimeError(f"{name}: K = 8 maps from {r['k8']['checkpoint']}, the rule selected {sel['selected']['checkpoint']}")
    return r


def z8_rounding_only(r) -> bool:
    """Item 15 (written before any selected checkpoint): a checkpoint failing Z8 at BOTH precisions stays in the search
    only if its fp32 drift is at least 1,000x below its own control's (the cache fed positions one off)."""
    zb, zf = r.get("z8"), r.get("z8_fp32")
    if zb is None or zb["pass"] or zf is None or zf["pass"]:
        return True
    return zf["max_abs_logit_diff"] * 1000.0 <= zf["control_abs"]


def usable(r) -> bool:
    return (r is not None and r["k8"] is not None and not r["select"]["health"]["failed"]
            and z8_rounding_only(r))


def contenders(a: dict) -> dict:
    ok = {t: r["k8"]["mIoU"] for t, r in a.items() if usable(r)}
    leader = max(ok, key=lambda t: (ok[t], -REGIME_ORDER[CELLS[t][0]]))
    lm = ok[leader]
    picks = [leader]
    for reg in REGIME_ORDER:
        if reg == CELLS[leader][0]:
            continue
        cand = [t for t in ok if CELLS[t][0] == reg]
        if cand:
            b = max(cand, key=lambda t: ok[t])
            if lm - ok[b] <= WITHIN:
                picks.append(b)
    runner = None
    if len(picks) == 1:                                # read 2026-11-22, before any data: whenever the leader would be alone
        rest = [t for t in ok if t != leader]
        if rest:
            runner = max(rest, key=lambda t: ok[t])
            picks.append(runner)
    return {"leader": leader, "leader_k8": lm, "contenders": picks, "runner_up_added": runner,
            "k8_seed42": ok, "rule": "the leader; in each other regime its best cell if leader - m <= 0.010; the "
                                     "runner-up if no other cell is within 1.0 point"}


def recipe(means: dict) -> dict:
    best = max(means.values())
    within = [t for t in means if best - means[t] <= WITHIN]
    chosen = min(within, key=lambda t: (REGIME_ORDER[CELLS[t][0]], -means[t]))
    return {"best_mean": best, "best_tag": max(means, key=means.get), "within_1pt": within, "recipe": chosen,
            "recipe_mean": means[chosen], "regime": CELLS[chosen][0], "lr": CELLS[chosen][1],
            "rule": "the highest two-seed mean K = 8; among the contenders within 1.0 point of it, the fewest trainable "
                    "parameters (LoRA r16, then r64, then full), then the higher mean"}


def bootstrap_pairs(cms: dict, pairs: list, n: int) -> dict:
    """cms: name -> list of per-patch matrices (one per seed; the mean mIoU over them is the statistic)."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    vals = {k: np.array([np.mean([miou(c[i].sum(0)) for c in v]) for i in idx]) for k, v in cms.items()}
    point = {k: float(np.mean([miou(c.sum(0)) for c in v])) for k, v in cms.items()}
    out = {}
    for a, b in pairs:
        d = vals[a] - vals[b]
        out[f"{a} - {b}"] = {"point": point[a] - point[b], "mean": float(d.mean()), "lo95": float(np.percentile(d, 2.5)),
                             "hi95": float(np.percentile(d, 97.5)), "share_positive": float((d > 0).mean())}
    return out


def strip(r):
    if r is None:
        return None
    out = {k: v for k, v in r.items() if k not in ("select",)}
    for k, v in list(out.items()):
        if isinstance(v, dict) and "cm" in v:
            out[k] = {kk: vv for kk, vv in v.items() if kk != "cm"}
    s = r["select"]
    out["select"] = {"selected": s["selected"], "plateau_stop": s["plateau_stop"], "health": s["health"],
                     "final": s["final"], "trainable_params": s["trainable_params"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rts", required=True)
    ap.add_argument("--out", default="/data/enric/reports/a2_seg/step41")
    a = ap.parse_args()
    pre = [run_of(t, sd, a.rts) for t in CELLS for sd in (42, 43)]
    z8b = [r["z8"] for r in pre if r is not None and r["z8"] is not None]
    contingency = bool(any(not z["pass"] for z in z8b))
    prec = "fp32" if contingency else "bf16"
    A = {t: run_of(t, 42, a.rts, prec) for t in CELLS}
    B = {t: run_of(t, 43, a.rts, prec) for t in CELLS}
    z8f = [r["z8_fp32"] for r in list(A.values()) + list(B.values()) if r is not None and r["z8_fp32"] is not None]
    res = {"protocol": "A2 Step 4.1, pre-registered 2026-11-22 (docs/plans/a2_segmentation.md, Step 4.1)",
           "rts": a.rts, "z8_contingency": contingency, "maps_precision": prec,
           "z8_bf16_fail": [r["run"] for r in pre if r is not None and r["z8"] is not None and not r["z8"]["pass"]],
           "z8_fp32_all_pass": bool(z8f) and all(z["pass"] for z in z8f),
           "z8_controls_fire": all(z["control_fires"] for z in z8b + z8f),
           "z8_both_fail": [r["run"] for r in pre if r is not None and r["z8"] is not None and r["z8_fp32"] is not None
                            and not r["z8"]["pass"] and not r["z8_fp32"]["pass"]],
           "z8_both_fail_not_rounding": [r["run"] for r in pre if r is not None and not z8_rounding_only(r)],
           "stage_a": {t: strip(r) for t, r in A.items()}}
    lines = ["| cell | regime | LR | stop (fire epoch) | selected (CE) | train loss there | **K = 8** | K = 1 | greedy | "
             "slot mass | pixel acc | Z8 bf16 | max grad norm |", "|" + "---|" * 13]
    for t, r in A.items():
        if r is None:
            lines.append(f"| {t} | {CELLS[t][0]} | {CELLS[t][1]:g} | not read yet | | | | | | | | | |")
            continue
        s = r["select"]
        fz = s["plateau_stop"]["fired"]
        stop = f"{s['plateau_stop']['stopped_at_step'] // s['steps_per_epoch']} ({fz['epoch'] if fz else 'none'})"
        m = lambda k: f"{r[k]['mIoU']:.4f}" if r.get(k) else "-"           # noqa: E731
        z8 = "-" if r["z8"] is None else f"{r['z8']['max_ulp_diff']:.2f} ({'pass' if r['z8']['pass'] else 'FAIL'})"
        if r.get("z8_fp32"):
            z8 += f"; fp32 {r['z8_fp32']['max_ulp_diff']:.1f} ({'pass' if r['z8_fp32']['pass'] else 'FAIL'})"
        k8 = r.get("k8")
        sm = f"{k8['slot_mass']:.3f}" if k8 and "slot_mass" in k8 else "-"
        pa = f"{k8['pixel_acc']:.3f}" if k8 else "-"
        lines.append(f"| {t} | {CELLS[t][0]} | {CELLS[t][1]:g} | {stop} | {s['selected']['checkpoint']} "
                     f"({s['selected']['eval_ce']:.4f}) | {s['selected']['train_loss_near']} | **{m('k8')}** | {m('k1')} | "
                     f"{m('greedy')} | {sm} | {pa} | {z8} | {s['health']['max_grad_norm']:.3g}"
                     f"{' FAILED' if s['health']['failed'] else ''} |")
    if all(r is not None and r["k8"] is not None for r in A.values()):
        c = contenders(A)
        res["contenders"] = c
        lines += ["", f"Stage A leader: **{c['leader']}** (K = 8 {c['leader_k8']:.4f}); contenders {c['contenders']}"
                  f"{' (runner-up added)' if c['runner_up_added'] else ''}."]
        # described, not deciding: does the selection number (fold-4 CE) order the cells as the deciding one does?
        okc = [t for t in A if usable(A[t])]
        ce = np.array([A[t]["select"]["selected"]["eval_ce"] for t in okc])
        mi = np.array([A[t]["k8"]["mIoU"] for t in okc])
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)                # noqa: E731
        rho = float(np.corrcoef(rk(-ce), rk(mi))[0, 1]) if len(okc) > 2 else None
        res["describe"] = {"spearman_lowCE_vs_K8": rho, "cells": okc,
                           "gap_train_minus_val_at_selected": {t: (A[t]["select"]["selected"]["train_loss_near"] or 0)
                                                               - A[t]["select"]["selected"]["eval_ce"] for t in okc}}
        lines.append(f"Described: Spearman rank correlation of low fold-4 CE with K = 8 over the usable cells {rho}.")
        cm40 = np.load(STEP40 / "per_patch_cm.npy")
        n = len(cm40)
        cms = {t: [r["k8"]["cm"]] for t, r in A.items() if usable(r)}
        cms["step40"] = [cm40]
        res["bootstrap_vs_step40"] = bootstrap_pairs(cms, [(t, "step40") for t in A if usable(A[t])], n)
        if all(B[t] is not None and B[t]["k8"] is not None for t in c["contenders"]):
            means = {t: float(np.mean([A[t]["k8"]["mIoU"], B[t]["k8"]["mIoU"]])) for t in c["contenders"]
                     if usable(A[t]) and usable(B[t])}
            rc = recipe(means)
            res["stage_b"] = {t: strip(B[t]) for t in c["contenders"]}
            res["means"] = means
            res["recipe"] = rc
            two = {t: [A[t]["k8"]["cm"], B[t]["k8"]["cm"]] for t in means}
            res["bootstrap_vs_recipe"] = bootstrap_pairs(two, [(t, rc["recipe"]) for t in means if t != rc["recipe"]], n)
            t = rc["recipe"]
            g = [A[t]["greedy"], B[t]["greedy"]]
            if all(x is not None for x in g):
                gm = float(np.mean([x["mIoU"] for x in g]))
                res["decoding"] = {"k8_mean": means[t], "greedy_mean": gm,
                                   "choice": "greedy" if gm >= means[t] - GREEDY_TOL else "K = 8",
                                   "rule": "K = 8 unless greedy's mean is within 0.5 point of K = 8's or above"}
                kc = {}
                for k in (1, 2, 4, 8, 16):
                    v = [x[f"k16_{k}"] for x in (A[t], B[t])]
                    if all(x is not None for x in v):
                        kc[str(k)] = {"mean": float(np.mean([x["mIoU"] for x in v])), "per_seed": [x["mIoU"] for x in v]}
                res["decoding"]["k_curve_from_k16"] = kc
            # item 13, for the recipe: the controls with Z9's functional check (Step 4.0's reading), the final save
            ctl = {}
            for r in (A[t], B[t]):
                k1b = scored(GEN / (r["run"] if prec == "bf16" else r["run"] + "_fp32") / "full_k8" / "k1b")
                x = {"k8": r["k8"]["mIoU"], "k1": r["k1"]["mIoU"] if r["k1"] else None,
                     "k1_sample1": k1b["mIoU"] if k1b else None,
                     "prior_k1": r["prior"]["mIoU"] if r["prior"] else None,
                     "wrong_k1": r["wrong"]["mIoU"] if r["wrong"] else None,
                     "final_k8": r["final_k8"]["mIoU"] if r["final_k8"] else None,
                     "greedy": r["greedy"]["mIoU"] if r["greedy"] else None}
                if None not in (x["k1"], x["k1_sample1"], x["prior_k1"], x["wrong_k1"]):
                    mw, mp, mf, mf2 = x["wrong_k1"], x["prior_k1"], x["k1"], x["k1_sample1"]
                    x["z9"] = {"check_pass": bool(abs(mw - mp) < abs(mw - mf)),
                               "control_fails_check": bool(not (abs(mf2 - mp) < abs(mf2 - mf)))}
                ctl[f"seed{r['seed']}"] = x
            res["recipe_controls"] = ctl
            lines += ["", "| contender | regime | LR | K = 8 seed 42 | K = 8 seed 43 | mean |", "|---|---|---|---|---|---|"]
            for x, mv in sorted(means.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {x} | {CELLS[x][0]} | {CELLS[x][1]:g} | {A[x]['k8']['mIoU']:.4f} | {B[x]['k8']['mIoU']:.4f} | "
                             f"{mv:.4f} |")
            lines += ["", f"**Recipe: {t}** ({rc['regime']}, LR {rc['lr']:g}); mean K = 8 {rc['recipe_mean']:.4f}; within "
                      f"1.0 point of the best ({rc['best_tag']}, {rc['best_mean']:.4f}): {rc['within_1pt']}."]
            if "decoding" in res:
                dd = res["decoding"]
                lines.append(f"Decoding: {dd['choice']} (K = 8 mean {dd['k8_mean']:.4f}, greedy mean {dd['greedy_mean']:.4f}); "
                             f"K curve {json.dumps({k: round(v['mean'], 4) for k, v in dd['k_curve_from_k16'].items()})}.")
            for sd, x in res["recipe_controls"].items():
                lines.append(f"Recipe {sd}: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                                          for k, v in x.items()))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "read.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (Path(a.out) / "read.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Z8 contingency (any selected checkpoint above 4 steps at bf16): {res['z8_contingency']} -> maps read at "
          f"{prec}; bf16 failures {res['z8_bf16_fail']}; fp32 Z8 all pass {res['z8_fp32_all_pass']}; controls fire "
          f"{res['z8_controls_fire']}; both precisions fail {res['z8_both_fail']} (not rounding, excluded: "
          f"{res['z8_both_fail_not_rounding']})")


if __name__ == "__main__":
    main()
