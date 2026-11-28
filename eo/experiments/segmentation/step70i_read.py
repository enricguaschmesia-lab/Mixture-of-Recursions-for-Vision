#!/usr/bin/env python
"""A2 Step 7.0i's read (docs/plans/a2_segmentation.md, Step 7, "7.0i"; the rule pre-registered 2026-11-27 before any of
its numbers). `.venv`, CPU; 7.0e's and 7.0g's functions and decoded maps, and Step 4.5's K = 40 scores.

    python eo/experiments/segmentation/step70i_read.py --cells <E784-s44 run> <s45> <s46> [--c12 <R0 CloudSen12 run>...]
                                                       [--out <dir>]

  PASTIS-7    validity per new seed (7.0e's items: training records, the plateau record, Z8 by the precision rule, Z9,
              B1's base); each seed replicates iff Delta(E784-sN K8 - B1 K8) gains by B1's 1.2 bar; the 784 map
              replicates on all five iff seeds 42-46 each gain. Reported, deciding nothing: the five-seed mean and SD of
              K8, plain and corrected (alpha cross-fitted as 7.0e); the five models' pooled vote over their K8 samples
              (40 votes), plain and corrected (a seed ensemble, beside); per class, the five-seed mean.
  CloudSen12  validity: each K = 40 run's first 8 samples equal its K = 8 run's (step45_kcurve's k8_equals_full_k8.json).
              Reported: the three-seed mean at K = 40 against K = 8, the 61.9 target and TerraMind-L's 63.9; each seed's
              vote curve. (The corrected vote on CloudSen12 is read later, by its own tools.)
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

import step70_read as S70                           # noqa: E402
import step70e_read as RD                           # noqa: E402
import step70e_votes as VT                          # noqa: E402
import step70k_read as K                            # noqa: E402
from step52_read import IGNORE                      # noqa: E402
from step53_verdict import iou_per_class, miou      # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70i")
EARLIER = ("p7e784_s42_20261127_004451", "p7e784_s43_20261127_133608")
C12_K8 = Path("/data/enric/reports/a2_seg/step52/gen")
C12_K40 = Path("/data/enric/reports/a2_seg/step45/gen")
C12_EARLIER = ("r0c12s52_s42_20261122_190655",)
C12_TARGET, C12_TM = 61.9, 63.9


def replicates(p: dict) -> str:
    return "replicates" if K.call(p) == "gains" else "does not replicate"


def all_five(calls: list) -> str:
    return ("replicates on all five seeds" if len(calls) == 5 and all(c == "replicates" for c in calls)
            else "does not replicate on every seed")


def seed_read(run: str, b1_plain: np.ndarray, prior: dict) -> tuple:
    """(the seed's record, reasons it is not valid)."""
    R = S70.read_run(run, RD.GEN)
    prec, why = K.validity(R)
    why = list(why) + K.lineage(run, None, K.B1_BASE_SHA)
    cm = S70.cm_of(RD.GEN / (run if prec == "bf16" else run + "_fp32") / "full_k8")
    rep = S70.pair(cm, b1_plain if prec == "bf16" else K.cm(K.B1_RUN, "fp32"))
    r = {"run": run, "precision": prec, "selected": R["selected"], "plateau": R.get("plateau_stop"),
         "k8_plain": rep["x"], "k8_minus_b1": rep, "call": replicates(rep), "k8_corrected": None}
    if prec == "bf16":
        t2 = RD.test2_of(f"{run}_k8", prior)
        if not RD.stored_plain_equal(f"{run}_k8", RD.GEN / run / "full_k8", t2):
            why.append(f"{run}: the decoded K8 maps' plain vote differs from the scorer's matrices")
        r["k8_corrected"], r["alphas"] = t2["cross_fit"]["x"], t2["chosen"]
    else:
        why.append(f"{run}: read at fp32, so it is left out of the corrected and pooled reads (bf16 maps only)")
    return r, why


def pooled(runs: list, prior: dict) -> dict:
    """The pooled vote over several models' K8 samples, plain and corrected (alpha cross-fitted as 7.0e)."""
    maps, sids = [], None
    for run in runs:
        m, vj, _ = RD.load_maps(f"{run}_k8")
        s = [str(x) for x in vj["sample_id"]]
        if sids is None:
            sids = s
        elif s != sids:
            raise RuntimeError(f"{run}: its K8 maps hold the patches in another order; not pooling")
        maps.append(m)
    labels = RD.labels_for(sids)
    fn = lambda i: np.concatenate([np.asarray(m[i]) for m in maps])          # noqa: E731
    cms = {a: RD.cms_of(fn, len(sids), labels, VT.class_weights(prior, a, RD.N_CLASSES, IGNORE[RD.TASK]))
           for a in RD.ALPHAS}
    cf, chosen = RD.cross_fit(cms, RD.halves(sids))
    return {"models": len(runs), "votes": 8 * len(runs), "plain": miou(cms[0.0].sum(0), IGNORE[RD.TASK]) * 100,
            "corrected": miou(cf.sum(0), IGNORE[RD.TASK]) * 100, "alphas": {str(k): v for k, v in chosen.items()},
            "_cms": cms, "_cf": cf}


def c12_read(runs: list) -> tuple:
    rows, why = [], []
    for run in runs:
        d40, d8 = C12_K40 / run / "full_k40", C12_K8 / run / "full_k8"
        eq = json.loads((d40 / "k8_equals_full_k8.json").read_text()) if (d40 / "k8_equals_full_k8.json").exists() else {}
        if not eq.get("all_equal"):
            why.append(f"{run}: the K = 40 run's first 8 samples are not its K = 8 run's ({eq or 'no record'})")
        curve = {}
        for k in range(8, 41, 8):
            f = d40 / f"k{k}" / "score.json"
            if f.exists():
                curve[k] = json.loads(f.read_text())["result"]["mIoU"] * 100
        k40 = json.loads((d40 / "score.json").read_text())["result"]["mIoU"] * 100 if (d40 / "score.json").exists() else None
        k8 = json.loads((d8 / "score.json").read_text())["result"]["mIoU"] * 100
        if k40 is None:
            why.append(f"{run}: no K = 40 score")
        rows.append({"run": run, "k8": k8, "k40": k40, "curve": curve, "first8_equal": bool(eq.get("all_equal"))})
    got = [r for r in rows if r["k40"] is not None]
    summ = {"seeds": len(got), "k8_mean": float(np.mean([r["k8"] for r in got])) if got else None,
            "k40_mean": float(np.mean([r["k40"] for r in got])) if got else None,
            "k40_sd": float(np.std([r["k40"] for r in got], ddof=1)) if len(got) > 1 else None,
            "target": C12_TARGET, "terramind_l": C12_TM}
    return {"rows": rows, "summary": summ}, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, help="the new E784 runs (seeds 44-46)")
    ap.add_argument("--c12", nargs="*", default=[], help="R0's CloudSen12 runs read at K = 40 tonight")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    prior = json.loads(RD.PRIOR.read_text())
    b1_plain = np.load(K.gen_dir(K.B1_RUN) / "full_k8" / "per_patch_cm.npy")
    res = {"rule": "A2 plan, Step 7, 7.0i (pre-registered 2026-11-27)", "seeds": {}}
    why, failed = [], []
    for run in list(EARLIER) + list(a.cells):
        try:
            r, w = seed_read(run, b1_plain, prior)
            res["seeds"][run] = r
            why += w
        except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError) as e:
            why.append(f"{run}: {type(e).__name__}: {e}")
            failed.append(run)
    got = [r for r in res["seeds"].values()]
    res["verdict_all_five"] = all_five([r["call"] for r in got]) if not failed else "not read (a seed is missing)"
    plain = [r["k8_plain"] for r in got]
    corr = [r["k8_corrected"] for r in got if r["k8_corrected"] is not None]
    res["five_seed"] = {"n": len(got), "k8_plain_mean": float(np.mean(plain)) if plain else None,
                        "k8_plain_sd": float(np.std(plain, ddof=1)) if len(plain) > 1 else None,
                        "k8_corrected_mean": float(np.mean(corr)) if corr else None,
                        "k8_corrected_sd": float(np.std(corr, ddof=1)) if len(corr) > 1 else None}
    bf = [r["run"] for r in got if r["precision"] == "bf16"]
    pc = {}
    if bf:
        P = pooled(bf, prior)
        res["pooled"] = {k: v for k, v in P.items() if not k.startswith("_")}
        res["pooled"]["runs"] = bf
        for run in bf:
            pc[run] = iou_per_class(RD.test2_of(f"{run}_k8", prior)["_cms"][0.0].sum(0), IGNORE[RD.TASK]) * 100
    from eo.maps import palette as Pal
    names = Pal.TASKS[RD.TASK].class_names
    if pc:
        mean_pc = np.nanmean(np.stack(list(pc.values())), 0)
        res["per_class_k8_plain_mean"] = [{"class": names[i], "iou": float(mean_pc[i])} for i in range(len(names))
                                          if i != IGNORE[RD.TASK]]
    if a.c12:
        c12, w = c12_read(list(C12_EARLIER) + list(a.c12))
        res["cloudsen12"] = c12
        why += [f"CloudSen12: {x}" for x in w]
    res["valid"], res["why"] = not why, why
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda p: f"{p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}]"       # noqa: E731
    L = ["# A2 Step 7.0i read: E784's seeds 44-46, then CloudSen12 at 40 votes", "", f"- valid {res['valid']} {why}", ""]
    L += ["| seed run | precision | K8 plain | K8 corrected | K8 - B1 | call |", "|---|---|---|---|---|---|"]
    for r in got:
        kc = f"{r['k8_corrected']:.2f}" if r["k8_corrected"] is not None else "-"
        L.append(f"| {r['run']} | {r['precision']} | {r['k8_plain']:.2f} | {kc} | {f(r['k8_minus_b1'])} | {r['call']} |")
    fs = res["five_seed"]
    L += ["", f"- verdict: {res['verdict_all_five']}",
          f"- {fs['n']}-seed K8 mean: plain {fs['k8_plain_mean']:.2f} (SD {fs['k8_plain_sd'] or 0:.2f}), corrected "
          f"{(fs['k8_corrected_mean'] or 0):.2f} (SD {fs['k8_corrected_sd'] or 0:.2f})"]
    if "pooled" in res:
        p = res["pooled"]
        L.append(f"- pooled vote of {p['models']} models ({p['votes']} votes; beside, a seed ensemble): plain "
                 f"{p['plain']:.2f}, corrected {p['corrected']:.2f} (alphas {p['alphas']})")
    if "cloudsen12" in res:
        s = res["cloudsen12"]["summary"]
        L += ["", "| CloudSen12 run | K8 | K40 | first 8 equal | vote curve |", "|---|---|---|---|---|"]
        for r in res["cloudsen12"]["rows"]:
            k40 = f"{r['k40']:.2f}" if r["k40"] is not None else "-"
            L.append(f"| {r['run']} | {r['k8']:.2f} | {k40} | {r['first8_equal']} | "
                     + ", ".join(f"{k}: {v:.2f}" for k, v in r["curve"].items()) + " |")
        if s["k40_mean"] is not None:
            L.append(f"- {s['seeds']}-seed mean: K8 {s['k8_mean']:.2f}, K40 {s['k40_mean']:.2f} (SD {s['k40_sd'] or 0:.2f}); "
                     f"target {C12_TARGET}, TerraMind-L {C12_TM}")
    (out / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
