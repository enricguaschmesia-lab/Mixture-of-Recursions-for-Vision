#!/usr/bin/env python
"""A2 Step 7.0d's read (docs/plans/a2_segmentation.md, Step 7, "7.0d, the date curriculum in one run at B1's budget";
the rule pre-registered 2026-11-26 before any of its numbers). `.venv`, CPU; reads the scored maps (step52_score.py) and
the pulled records. Fold 4 only.

    python eo/experiments/segmentation/step70d_read.py --run <p7d_s42_...> [--out <dir>]

  1. validity of G (the run): 7.0c's items (training finite, the plateau record agreeing with its curve, which holds with
     no plateau stop, Z8 by the precision rule, Z9); finetune_init.json naming B1's base and no continuation; the date
     record (date_curriculum.json) holding k(e) = min(7, 1 + e // 2) at global_step 37 e for every epoch 0..22; the
     logged LR equal to B1's at every logged step; the CE at step 0 within 0.005 of B1's. Any failure: no verdict;
  2. the verdict, Delta(G - B1) by 7.0c's pair rule (paired bootstrap over the 482 views, 1,000 resamples, seed 0; gains
     iff Delta >= +1.2 and its lower end > 0, costs iff Delta <= -1.2 and its upper end < 0): (a) gains, (b) neither,
     (c) loses;
  3. beside, deciding nothing: full - wrong_early on G and B1 and the difference of the drops; G - P_5; G - C_7; per-class
     IoU against B1; the held-out CE of G and B1 at every epoch; one_date and shuffled_dates against B1's.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step70_read as S70                           # noqa: E402
import step70k_read as K                            # noqa: E402
from step52_read import IGNORE, RUNS                # noqa: E402
from step53_verdict import iou_per_class            # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70d")
GEN = OUT / "gen"
B1_RUN = K.B1_RUN
P5_RUN, C6_RUN, C7_RUN = ("p7k5_s42_20261126_140510", "p7c6_s42_20261126_140510", "p7c7_s42_20261126_140510")
PER, STEPS_PER_EPOCH, N_EPOCHS, N_GROUPS = 2, 37, 23, 7
CE0_TOL, LR_RTOL = 0.005, 1e-9
TASK = S70.TASK


# ---------------------------------------------------------------- the rules (pure; the gate tests them)
def k_of(epoch: int, per: int = PER) -> int:
    return min(N_GROUPS, 1 + int(epoch) // int(per))


def schedule_problems(rec: Optional[dict], per: int = PER, n_epochs: int = N_EPOCHS,
                      steps_per_epoch: int = STEPS_PER_EPOCH) -> List[str]:
    """Reasons the run's date record is not k(e) at global_step 37 e for every epoch 0..n_epochs-1."""
    if rec is None:
        return ["no date_curriculum.json"]
    why = []
    if rec.get("date_curriculum") != per:
        why.append(f"date_curriculum {rec.get('date_curriculum')}, not {per}")
    got = {int(x["draw_epoch"]): x for x in rec.get("epochs", [])}
    if sorted(got) != list(range(n_epochs)):
        why.append(f"epochs {sorted(got)}, not 0..{n_epochs - 1}")
    for e, x in sorted(got.items()):
        k = k_of(e, per)
        if (int(x["global_step"]) != e * steps_per_epoch or int(x["k"]) != k
                or list(x["groups"]) != list(range(N_GROUPS - k, N_GROUPS))):
            why.append(f"epoch {e}: step {x['global_step']} k {x['k']} groups {x['groups']}, expected step "
                       f"{e * steps_per_epoch} k {k}")
    return why


def lr_problems(hist_a: list, hist_b: list, rtol: float = LR_RTOL) -> List[str]:
    """Reasons run a's logged learning rates are not run b's (the same logged steps, equal within rtol)."""
    a = {int(h["step"]): float(h["learning_rate"]) for h in hist_a if "learning_rate" in h and "loss" in h}
    b = {int(h["step"]): float(h["learning_rate"]) for h in hist_b if "learning_rate" in h and "loss" in h}
    if sorted(a) != sorted(b):
        return [f"logged steps differ: {sorted(set(a) ^ set(b))[:6]}"]
    bad = [s for s in sorted(a) if abs(a[s] - b[s]) > rtol * max(abs(b[s]), 1e-12)]
    return [f"LR differs at {len(bad)} logged steps, first {bad[0]}: {a[bad[0]]!r} vs {b[bad[0]]!r}"] if bad else []


def band_g(p: dict) -> str:
    c = K.call(p)
    if c == "gains":
        return ("(a) gains: at B1's budget the direct jump to seven dates costs something; a second seed of both "
                "before anything is adopted")
    if c == "costs":
        return "(c) loses: at equal budget the curriculum costs: fewer epochs at seven dates outweigh a gentler start"
    return "(b) neither: a gradual start does not help at equal budget; B1 is not held back by the jump"


# ---------------------------------------------------------------- the records
def history(run: str) -> list:
    return json.loads((RUNS / run / "trainer_state.json").read_text())["log_history"]


def read_g(run: str) -> dict:
    """G's records, read as 7.0c reads a run (K.validity, K.lineage) plus 7.0d's own validity items."""
    try:
        R = S70.read_run(run, GEN)
    except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError) as e:
        return {"run": run, "valid": False, "why": [f"records: {type(e).__name__}: {e}"]}
    prec, why = K.validity(R)
    why += K.lineage(run, None, K.B1_BASE_SHA)
    f = RUNS / run / "date_curriculum.json"
    R["date_record"] = json.loads(f.read_text()) if f.is_file() else None
    why += schedule_problems(R["date_record"])
    why += lr_problems(history(run), history(B1_RUN))
    b1_ce0 = S70.curve(B1_RUN)["eval"][0]
    R["b1_step0_ce"] = b1_ce0[1] if b1_ce0[0] == 0 else None
    if R.get("step0_ce") is None or R["b1_step0_ce"] is None or abs(R["step0_ce"] - R["b1_step0_ce"]) > CE0_TOL:
        why.append(f"CE at step 0 {R.get('step0_ce')} against B1's {R['b1_step0_ce']} (tolerance {CE0_TOL})")
    R["precision"], R["valid"], R["why"] = prec, not why, why
    if R["valid"]:
        R["mIoU"] = float(R[prec]["k8"]["mIoU"]) * 100
    return R


def cm_g(run: str, prec: str, line: str = "full_k8") -> np.ndarray:
    d = GEN / (run if prec == "bf16" else f"{run}_fp32")
    return S70.cm_of(d / line)


def pair_g(g: dict, other: dict, line: str = "full_k8") -> dict:
    """G - another run's maps on the same line, at their common precision (7.0c's maps through K.cm)."""
    p = K.common(g, other)
    try:
        out = S70.pair(cm_g(g["run"], p, line), K.cm(other["run"], p, line))
    except FileNotFoundError as e:
        return {"pending": f"{p} maps missing: {e}"}
    out["precision"] = p
    return out


def early(g: dict, b1: dict) -> dict:
    """full - wrong_early on G and B1 (bf16 maps, as 7.0c's), and the G - B1 difference of the drops."""
    out, have = {}, {}
    for name, R, full_d, ctl_d in (("G", g, GEN / g["run"] / "full_k8", GEN / g["run"] / "wrong_early_k8"),
                                   ("B1", b1, K.gen_dir(B1_RUN) / "full_k8", K.GEN / B1_RUN / "wrong_early_k8")):
        if not R.get("valid"):
            continue
        if not (ctl_d / "per_patch_cm.npy").exists():
            out[name] = {"pending": f"{ctl_d} not scored"}
            continue
        rec = json.loads((ctl_d / "record.json").read_text())
        if not str(rec.get("checkpoint", "")).endswith(R["selected"]["checkpoint"]):
            raise RuntimeError(f"{ctl_d}: from {rec.get('checkpoint')}, not {R['selected']['checkpoint']}")
        if S70.views_of(ctl_d) != S70.views_of(full_d):
            raise RuntimeError(f"{ctl_d}: views differ from the full maps'")
        full, ctl = S70.cm_of(full_d), S70.cm_of(ctl_d)
        out[name] = S70.pair(full, ctl)
        have[name] = (full, ctl)
    if "G" in have and "B1" in have:
        out["G_minus_B1_drop"] = K.pair_of_drops(*have["G"], *have["B1"])
    return out


def per_class(g: dict, others: Dict[str, dict]) -> list:
    from eo.maps import palette as P
    names = P.TASKS[TASK].class_names
    cols = {}
    if g.get("valid"):
        cols["G"] = iou_per_class(cm_g(g["run"], g["precision"]).sum(0), IGNORE[TASK]) * 100
    for role, R in others.items():
        if R.get("valid"):
            cols[role] = iou_per_class(K.cm(R["run"], R["precision"]).sum(0), IGNORE[TASK]) * 100
    return [{"class": names[i], **{k: float(v[i]) for k, v in cols.items()}} for i in range(len(names))
            if cols and not all(np.isnan(v[i]) for v in cols.values())]


def ce_by_epoch(run: str) -> Dict[int, float]:
    return {s // STEPS_PER_EPOCH: v for s, v in S70.curve(run)["eval"] if s % STEPS_PER_EPOCH == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="G's run id, p7d_s42_<stamp>")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g = read_g(a.run)
    b1 = K.read_one(B1_RUN)
    p5 = K.read_one(P5_RUN)
    c7 = K.read_one(C7_RUN, parent=C6_RUN)
    res = {"rule": "A2 plan, Step 7, 7.0d (pre-registered 2026-11-26)", "bar": K.BAR, "run": a.run,
           "validity": {"valid": g.get("valid"), "why": g.get("why"), "precision": g.get("precision"),
                        "b1_valid": b1.get("valid"), "b1_why": b1.get("why")}}
    if g.get("valid") and b1.get("valid"):
        p = pair_g(g, b1)
        res["G_minus_B1"] = p
        res["verdict"] = band_g(p) if "delta" in p else f"pending: {p.get('pending')}"
    else:
        res["verdict"] = "not read: G or B1 invalid (see validity)"
    beside = {}
    if g.get("valid"):
        beside["early"] = early(g, b1)
        for name, R in (("G_minus_P5", p5), ("G_minus_C7", c7)):
            beside[name] = pair_g(g, R) if R.get("valid") else {"not_read": R.get("why")}
        for line in ("one_date_k8", "shuffled_dates_k8"):
            beside[f"{line}_G_minus_B1"] = pair_g(g, b1, line) if b1.get("valid") else None
            beside[f"full_minus_{line}_G"] = (S70.pair(cm_g(g["run"], g["precision"]), cm_g(g["run"], g["precision"], line))
                                              if (GEN / (g["run"] if g["precision"] == "bf16" else g["run"] + "_fp32")
                                                  / line / "per_patch_cm.npy").exists() else None)
        beside["per_class"] = per_class(g, {"B1": b1, "C7": c7, "P5": p5})
        cg, cb = ce_by_epoch(a.run), ce_by_epoch(B1_RUN)
        beside["ce_by_epoch"] = {str(e): {"G": cg.get(e), "B1": cb.get(e), "k": k_of(e - 1) if e >= 1 else None}
                                 for e in sorted(set(cg) | set(cb))}
        beside["selected"] = {"G": {"step": g.get("selected_step"), "ce": g.get("selected_ce")},
                              "B1": {"step": b1.get("selected_step"), "ce": b1.get("selected_ce")}}
        beside["mIoU"] = {"G": g.get("mIoU"), "B1": b1.get("mIoU"), "P5": p5.get("mIoU"), "C7": c7.get("mIoU")}
    res["beside"] = beside
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    L = ["# A2 Step 7.0d read", "", f"G = {a.run}; valid {g.get('valid')} {g.get('why') or ''}"]
    if "G_minus_B1" in res and "delta" in res["G_minus_B1"]:
        p = res["G_minus_B1"]
        L.append(f"- G {p['x']:.2f}, B1 {p['y']:.2f}: G - B1 {p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}] "
                 f"({p['precision']})")
    L.append(f"- verdict: {res['verdict']}")
    if beside:
        L += ["", "## Beside, deciding nothing"]
        for k, v in beside.get("early", {}).items():
            L.append(f"- full - wrong_early, {k}: " + (f"{v['delta']:+.2f} [{v['lo95']:+.2f}, {v['hi95']:+.2f}]"
                                                      if "delta" in v else str(v)))
        for k in ("G_minus_P5", "G_minus_C7", "one_date_k8_G_minus_B1", "shuffled_dates_k8_G_minus_B1"):
            v = beside.get(k) or {}
            L.append(f"- {k}: " + (f"{v['delta']:+.2f} [{v['lo95']:+.2f}, {v['hi95']:+.2f}]" if "delta" in v else str(v)))
        L.append("- held-out CE by epoch (G / B1): " + ", ".join(
            f"{e}: {v['G']:.3f}/{v['B1']:.3f}" for e, v in beside["ce_by_epoch"].items()
            if v["G"] is not None and v["B1"] is not None))
    (out / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
