#!/usr/bin/env python
"""A2 Step 7.0b's B1 read (docs/plans/a2_segmentation.md, Step 7, "7.0b", B1; the rule pre-registered 2026-11-26 before
any B1 number). `.venv`, CPU; reads the scored maps (step52_score.py) and the pulled records.

    python eo/experiments/segmentation/step70b1_read.py --run <p7b1_s42_...> [--gen <maps root>] [--out <dir>]

B1 is 7.0's cell (eo_seg/s70_p7, seed 42, LR 3e-3, the same controls) with only the base changed: the nc temporal pilot's
checkpoint-2960 (one pass of A1's temporal stage, every parameter trained, from the very weights A2's base extends) with
A2's map rows by A2's rule (b10cltrxnct_ext_map_nc_ckpt2960), in place of A2's base (b10cltrxnc_ext_map_nc_ckpt33000).
  1. validity, as 7.0's: training finite, the selection's plateau record agreeing with its curve; Z8 by the precision
     rule; Z9 (prior and wrong below the own maps, K = 1). If one fails, nothing below is read;
  2. precision: bf16 iff B1 passes Z8 at bf16 by the 4-step rule (7.0 did); otherwise fp32 under Step 5.2's fp32 rules,
     against 7.0's fp32 maps of the same kept save (its cell's f lines, generated first: without them this does not read);
  3. the number: K = 8 fold-4 mIoU on B1's base (p7_B1), against the target 41.1 (TerraMind-L 43.1 - 2);
  4. Delta(B1 - 7.0) on the same 482 views, and on B1 the date controls Delta(full - shuffled_dates) and
     Delta(full - one_date): the paired bootstrap over views, 1,000 resamples, seed 0 (7.0's form, step70_read.pair);
  5. the verdict: B1 GAINS iff Delta(B1 - 7.0) >= +1.2 and its interval's lower end is above 0; it LOSES iff
     Delta <= -1.2 and the upper end is below 0. 1.2 is two SDs of the difference between two single-seed runs, from
     PASTIS-1's pooled seed SD (0.42 point; sqrt(2) x 0.42 x 2), since the bootstrap over views holds no seed noise:
       (a) gains and p7_B1 >= 36.1 (7.0's band (a), within 5 points of the target);
       (b) gains and p7_B1 < 36.1;
       (c) neither gains nor loses;
       (d) loses;
     each band's action is in the plan;
  6. beside, deciding nothing: Delta(full - shuffled_dates) for item 84 (an interval above 0: the date tokens' values
     are read on this base), per-class IoU against 7.0, the CE curve and step-0 CE against 7.0's, K = 1 (two samples),
     slot mass, generation and training time, Z8's drift.
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

import step70_read as S70                           # noqa: E402
from step52_read import IGNORE                      # noqa: E402
from step53_verdict import iou_per_class, miou      # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70b1")
GEN = OUT / "gen"
P70_RUN, P70_GEN = "p7_s42_20261125_211637", S70.GEN
B1_BASE_SHA = "e9caba474221966571a30a6a5f8f5f7288e3eb58cf362cb81898552c10db50f8"   # its extension.json's weights_sha256
A2_BASE_SHA = "6ed2e0cd8b7da5538f4823746c4c417e297ed530ccfee13b621c350d29fbb009"
SEED_SD = 0.42                                      # PASTIS-1's pooled seed SD (A2 Step 5.2), mIoU points
GAIN = round(2 * math.sqrt(2) * SEED_SD, 1)         # 1.2
TASK = S70.TASK
BANDS = {
    "a": "(a) B1 gains and reads >= 36.1: one temporal pass brings Route A within 5 points of the target",
    "b": "(b) B1 gains and reads < 36.1: a temporal stage helps; one pass does not close the gap",
    "c": "(c) B1 neither gains nor loses: one temporal pass does not move Route A on PASTIS-7",
    "d": "(d) B1 loses: the pilot's base is worse for Route A on PASTIS-7",
}


def verdict_of(p7_b1: float, d: dict) -> str:
    """The band of the rule's item 5, from p7_B1 and Delta(B1 - 7.0)'s pair() record."""
    gains = d["delta"] >= GAIN and d["lo95"] > 0
    loses = d["delta"] <= -GAIN and d["hi95"] < 0
    if gains:
        return "a" if p7_b1 >= S70.BAND_A else "b"
    return "d" if loses else "c"


def validity(R: dict, prec: str) -> list:
    """7.0's validity items (step70_read.main), on one read_run record; fills R['z9']. Returns the failures."""
    x, why = R[prec], []
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
    return why


def base_of(run: str) -> str:
    """The weights a run started from, by its finetune_init.json's hash: 'B1', 'A2' or the hash itself."""
    fi = S70.RUNS / run / "finetune_init.json"
    sha = json.loads(fi.read_text()).get("weights_sha256") if fi.exists() else None
    return {B1_BASE_SHA: "B1", A2_BASE_SHA: "A2"}.get(sha, str(sha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gen", default=str(GEN))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    gen, out = Path(a.gen), Path(a.out)
    R = S70.read_run(a.run, gen)
    zb = R["z8_bf16"]
    prec = "bf16" if (zb and zb["pass"]) else "fp32"
    res = {"rule": "A2 plan, Step 7, 7.0b, B1 (pre-registered 2026-11-26)", "precision": prec, "gain_threshold": GAIN,
           "run": R, "reference_70": {"run": P70_RUN}}
    why = validity(R, prec)
    if base_of(a.run) != "B1":
        why.append(f"{a.run} did not start from B1's base (finetune_init.json: {base_of(a.run)})")
    if base_of(P70_RUN) != "A2":
        why.append(f"7.0's run {P70_RUN} did not start from A2's base ({base_of(P70_RUN)})")
    sub = "" if prec == "bf16" else "_fp32"
    d_b1, d_70 = gen / f"{a.run}{sub}", P70_GEN / f"{P70_RUN}{sub}"
    if not (d_70 / "full_k8" / "per_patch_cm.npy").exists():
        why.append(f"7.0's {prec} K = 8 maps missing ({d_70 / 'full_k8'}): generate them on 7.0's kept save first")
    if why:
        res["valid"], res["why"] = False, why
    else:
        res["valid"] = True
        if S70.views_of(d_b1 / "full_k8") != S70.views_of(d_70 / "full_k8"):
            raise RuntimeError("B1's and 7.0's views differ: the bootstrap would not pair")
        cb, c7 = S70.cm_of(d_b1 / "full_k8"), S70.cm_of(d_70 / "full_k8")
        p = miou(cb.sum(0), IGNORE[TASK]) * 100
        res["p7_b1"], res["target"], res["gap_to_target"] = p, S70.TARGET, p - S70.TARGET
        res["b1_minus_70"] = S70.pair(cb, c7)
        for k, s in (("shuffled", "shuffled_dates_k8"), ("one_date", "one_date_k8")):
            if (d_b1 / s / "per_patch_cm.npy").exists():
                if S70.views_of(d_b1 / s) != S70.views_of(d_b1 / "full_k8"):
                    raise RuntimeError(f"{s}'s views differ from full_k8's")
                res[f"full_minus_{k}"] = S70.pair(cb, S70.cm_of(d_b1 / s))
        band = verdict_of(p, res["b1_minus_70"])
        res["band"], res["verdict"] = band, BANDS[band]
        r70 = json.loads((S70.OUT / "read.json").read_text())
        res["reference_70"].update({k: r70.get(k) for k in ("p7", "full_minus_shuffled", "full_minus_one_date")})
        R70 = S70.read_run(P70_RUN, P70_GEN)
        res["reference_70"].update({"selected_step": R70["selected_step"], "selected_ce": R70["selected_ce"],
                                    "step0_ce": R70["step0_ce"], "curve_eval": R70["curve"]["eval"],
                                    "train_runtime_s": R70["curve"]["train_runtime_s"]})
        from eo.maps import palette as P
        names = P.TASKS[TASK].class_names
        per = {"b1": iou_per_class(cb.sum(0), IGNORE[TASK]) * 100, "p70": iou_per_class(c7.sum(0), IGNORE[TASK]) * 100}
        for k, s in (("shuffled", "shuffled_dates_k8"), ("one_date", "one_date_k8")):
            if (d_b1 / s / "per_patch_cm.npy").exists():
                per[k] = iou_per_class(S70.cm_of(d_b1 / s).sum(0), IGNORE[TASK]) * 100
        res["per_class"] = [{"class": names[i], **{k: float(v[i]) for k, v in per.items()}} for i in range(len(names))
                            if not all(np.isnan(v[i]) for v in per.values())]
    out.mkdir(parents=True, exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    lines = [f"# A2 Step 7.0b, B1 read, precision {prec}", ""]
    if res["valid"]:
        lines.append(f"- p7_B1 (K = 8, fold 4): {res['p7_b1']:.2f}; target {S70.TARGET} (gap {res['gap_to_target']:+.2f})")
        for k in ("b1_minus_70", "full_minus_shuffled", "full_minus_one_date"):
            if k in res:
                q = res[k]
                lines.append(f"- {k}: {q['x']:.2f} - {q['y']:.2f} = {q['delta']:+.2f} [{q['lo95']:+.2f}, {q['hi95']:+.2f}]")
        lines += ["", f"- verdict: {res['verdict']} (gain threshold {GAIN})"]
    else:
        lines.append(f"not read: {'; '.join(res['why'])}")
    (out / "read.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
