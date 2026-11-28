#!/usr/bin/env python
"""A2 Step 7.0's read (docs/plans/a2_segmentation.md, Step 7, "7.0, PASTIS-7 on A2's current base"; the rule pre-registered
2026-11-25 before any PASTIS-7 number). `.venv`, CPU; reads the scored maps (step52_score.py) and the pulled records.

    python eo/experiments/segmentation/step70_read.py --run <p7_s42_...> [--gen <maps root>] [--out <dir>]

  1. validity: training finite, the selection's plateau record agreeing with its curve; Z8 by the precision rule; Z9
     (prior and wrong below the own maps, K = 1). If one fails, nothing below is read;
  2. precision: bf16 iff the run passes Z8 at bf16 by the 4-step rule; otherwise fp32 (the cell's f lines) under Step
     5.2's fp32 rules, against R0's PASTIS-1 fp32 maps;
  3. the number: K = 8 fold-4 mIoU of PASTIS-7 (p7), against the target 41.1 (TerraMind-L 43.1 - 2), caveats in the plan;
  4. Delta(p7 - p1) against R0's PASTIS-1 seed 42 at the same recipe (r0p1s41_l16c_s42_20261122_123818), and the date
     controls Delta(full - one_date), Delta(full - shuffled_dates): paired bootstrap over the 482 views, 1,000
     resamples, seed 0 (Step 4.3's form: one index draw per resample, mIoU of each side's summed matrices);
  5. the verdict band: (a) p7 >= 36.1 on track; (b) 25.0 <= p7 < 36.1 the temporal base must close the rest, plan B
     written now; (c) p7 < 25.0 the reporting rule for PASTIS-7 goes to Enric and Gianfranco now;
  6. beside, deciding nothing: per-class IoU against p1, the CE curve, step-0 CE, the extension test, K = 1 (two
     samples), slot mass, generation and training time, Z8's drift at the long prompt.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step41_read import z8_of                       # noqa: E402
from step52_read import IGNORE, RUNS, scored        # noqa: E402
from step53_verdict import iou_per_class, miou      # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70")
GEN = OUT / "gen"
P1_RUN, P1_GEN = "r0p1s41_l16c_s42_20261122_123818", Path("/data/enric/reports/a2_seg/step41/gen")
TARGET, BAND_A, BAND_C = 41.1, 36.1, 25.0
N_BOOT, BOOT_SEED, TASK, STEPS_PER_EPOCH = 1000, 0, "pastis", 37
SUBS = {"k8": "full_k8", "k1": "full_k8/k1", "k1b": "full_k8/k1b", "prior": "prior_k1", "wrong": "wrong_k1",
        "one_date": "one_date_k8", "shuffled": "shuffled_dates_k8"}


def curve(run: str) -> dict:
    st = json.loads((RUNS / run / "trainer_state.json").read_text())
    hist = st["log_history"]
    ev = [(int(h["step"]), float(h["eval_loss"])) for h in hist if "eval_loss" in h]
    tr = [(int(h["step"]), float(h["loss"]), float(h.get("grad_norm", float("nan")))) for h in hist if "loss" in h]
    rt = [h["train_runtime"] for h in hist if "train_runtime" in h]
    return {"eval": ev, "train": tr, "max_steps": st.get("max_steps"), "global_step": st.get("global_step"),
            "train_runtime_s": rt[-1] if rt else None}


def extension_test(c: dict, plateau) -> dict:
    """Step 4.0's form: the CE at the last eval before the decay against the eval three epochs earlier (> 1% fires)."""
    ev = dict(c["eval"])
    fired = ((plateau or {}).get("fired") or {}).get("epoch")
    e = int(fired) if fired else 40
    s, s3 = e * STEPS_PER_EPOCH, (e - 3) * STEPS_PER_EPOCH
    if s not in ev or s3 not in ev:
        return {"epoch": e, "missing": True}
    rel = (ev[s3] - ev[s]) / ev[s3]
    return {"epoch": e, "ce": ev[s], "ce_3_earlier": ev[s3], "rel_gain": rel, "fires": bool(rel > 0.01)}


def read_run(run: str, gen: Path) -> dict:
    sel = json.loads((RUNS / run / "select41.json").read_text())
    gb, gf = gen / run, gen / f"{run}_fp32"
    r = {"run": run, "selected": sel["selected"], "plateau_stop": sel.get("plateau_stop"), "health": sel.get("health"),
         "z8_bf16": z8_of(gb, "bf16"), "z8_fp32": z8_of(gb, "fp32")}
    for p, g in (("bf16", gb), ("fp32", gf)):
        x = {k: scored(g / sub, TASK) for k, sub in SUBS.items()}
        for k, sub in (("k8", "full_k8"), ("one_date", "one_date_k8"), ("shuffled", "shuffled_dates_k8")):
            rec = g / sub / "record.json"
            if rec.exists():
                rj = json.loads(rec.read_text())
                x[f"{k}_seconds"] = rj.get("seconds")
                x[f"{k}_slot_mass"] = rj.get("slot_mass_mean")
                if not str(rj.get("checkpoint", "")).endswith(sel["selected"]["checkpoint"]):
                    raise RuntimeError(f"{run}: {sub} from {rj.get('checkpoint')}, not {sel['selected']['checkpoint']}")
        r[p] = x
    c = curve(run)
    r["curve"] = c
    r["step0_ce"] = dict(c["eval"]).get(0)
    st = int(sel["selected"]["step"]) if "step" in sel["selected"] else int(sel["selected"]["checkpoint"].split("-")[1])
    r["selected_step"], r["selected_ce"] = st, dict(c["eval"]).get(st)
    r["max_grad_norm"] = max((t[2] for t in c["train"] if math.isfinite(t[2])), default=None)
    r["extension_test"] = extension_test(c, r["plateau_stop"])
    r["finite"] = all(math.isfinite(v) for _, v in c["eval"]) and all(math.isfinite(t[1]) for t in c["train"])
    r["plateau_agrees"] = bool(((r["plateau_stop"] or {}).get("agrees", True)))
    return r


def cm_of(d: Path) -> np.ndarray:
    return np.load(d / "per_patch_cm.npy")


def views_of(d: Path):
    return json.loads((d / "record.json").read_text())["views"]


def pair(x: np.ndarray, y: np.ndarray) -> dict:
    """x - y in mIoU points, paired bootstrap over patches (one index draw per resample)."""
    if x.shape != y.shape:
        raise RuntimeError(f"shapes {x.shape} {y.shape}")
    ig = IGNORE[TASK]
    mx, my = miou(x.sum(0), ig) * 100, miou(y.sum(0), ig) * 100
    rng = np.random.default_rng(BOOT_SEED)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d = np.array([miou(x[i].sum(0), ig) - miou(y[i].sum(0), ig) for i in idx]) * 100
    return {"x": mx, "y": my, "delta": mx - my, "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5)), "share_positive": float((d > 0).mean())}


def band(p7: float) -> str:
    if p7 >= BAND_A:
        return "(a) on track: Step 7 as planned"
    if p7 >= BAND_C:
        return "(b) the temporal base must close the rest: write plan B for the temporal row now"
    return "(c) more than 16 points short: the PASTIS-7 reporting rule goes to Enric and Gianfranco now"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gen", default=str(GEN))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    gen, out = Path(a.gen), Path(a.out)
    R = read_run(a.run, gen)
    zb = R["z8_bf16"]
    prec = "bf16" if (zb and zb["pass"]) else "fp32"
    res = {"rule": "A2 plan, Step 7, 7.0 (pre-registered 2026-11-25)", "precision": prec, "run": R,
           "reference_p1": {"run": P1_RUN, "z8_bf16": z8_of(P1_GEN / P1_RUN, "bf16")}}
    x = R[prec]
    why = []
    if not R["finite"]:
        why.append("training not finite")
    if not R["plateau_agrees"]:
        why.append("the plateau record disagrees with the curve")
    if prec == "fp32" and not (R["z8_fp32"] and R["z8_fp32"]["pass"]):
        why.append("Z8 fails at bf16 and at fp32")
    if x.get("k8") is None:
        why.append(f"K = 8 maps at {prec} missing")
    if all(x.get(k) for k in ("prior", "wrong", "k1")):
        own, pr, wr = x["k1"]["mIoU"], x["prior"]["mIoU"], x["wrong"]["mIoU"]
        R["z9"] = {"pass": bool(pr < own and wr < own), "own_k1": own, "prior": pr, "wrong": wr,
                   "functional": bool(abs(wr - pr) < abs(wr - own))}
        if not R["z9"]["pass"]:
            why.append("Z9: prior or wrong not below the own maps")
    else:
        why.append("Z9's maps missing")
    if why:
        res["valid"], res["why"] = False, why
    else:
        res["valid"] = True
        sub = "" if prec == "bf16" else "_fp32"
        d7, d1 = gen / f"{a.run}{sub}", P1_GEN / f"{P1_RUN}{sub}"
        if views_of(d7 / "full_k8") != views_of(d1 / "full_k8"):
            raise RuntimeError("the PASTIS-7 and PASTIS-1 views differ: the bootstrap would not pair")
        c7, c1 = cm_of(d7 / "full_k8"), cm_of(d1 / "full_k8")
        p7 = miou(c7.sum(0), IGNORE[TASK]) * 100
        res["p7"], res["target"], res["gap_to_target"] = p7, TARGET, p7 - TARGET
        res["p7_minus_p1"] = pair(c7, c1)
        for k, s in (("one_date", "one_date_k8"), ("shuffled", "shuffled_dates_k8")):
            if (d7 / s / "per_patch_cm.npy").exists():
                if views_of(d7 / s) != views_of(d7 / "full_k8"):
                    raise RuntimeError(f"{s}'s views differ from full_k8's")
                res[f"full_minus_{k}"] = pair(c7, cm_of(d7 / s))
        res["verdict"] = band(p7)
        from eo.maps import palette as P
        names = P.TASKS[TASK].class_names
        per = {"p7": iou_per_class(c7.sum(0), IGNORE[TASK]) * 100, "p1": iou_per_class(c1.sum(0), IGNORE[TASK]) * 100}
        for k, s in (("one_date", "one_date_k8"), ("shuffled", "shuffled_dates_k8")):
            if (d7 / s / "per_patch_cm.npy").exists():
                per[k] = iou_per_class(cm_of(d7 / s).sum(0), IGNORE[TASK]) * 100
        res["per_class"] = [{"class": names[i], **{k: float(v[i]) for k, v in per.items()}} for i in range(len(names))
                            if not all(np.isnan(v[i]) for v in per.values())]
    out.mkdir(parents=True, exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    lines = [f"# A2 Step 7.0 read, precision {prec}", ""]
    if res["valid"]:
        lines.append(f"- p7 (K = 8, fold 4): {res['p7']:.2f}; target {TARGET} (gap {res['gap_to_target']:+.2f})")
        for k in ("p7_minus_p1", "full_minus_one_date", "full_minus_shuffled"):
            if k in res:
                p = res[k]
                lines.append(f"- {k}: {p['x']:.2f} - {p['y']:.2f} = {p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}]")
        lines += ["", f"- verdict: {res['verdict']}"]
    else:
        lines.append(f"not read: {'; '.join(res['why'])}")
    (out / "read.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
