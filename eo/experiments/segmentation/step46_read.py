#!/usr/bin/env python
"""A2 Step 4.6's read (`.venv`, CPU): the forgetting-bounded recipe, by the rule written before any of its numbers.

Plan: docs/plans/a2_segmentation.md, Step 4, "4.6, the forgetting-bounded recipe (item 90)" (2026-11-24, ~19:20).
  item 1  F = CE over the full softmax after - before (step44_map_mass.py --decompose --per-row, fp32, GTX Titan X, image
          tokens of the first 256 held-out rows), paired by token: per row, sum(after) - sum(before) over the same
          positions; F = their total / the token count; a bootstrap over the 256 rows (1,000, seed 0). Bounded iff
          F <= +0.02. "Before" (the map base) must reproduce 4.4's 5.067649 to 1e-5.
  item 2  K = 8 mIoU against R0's seed-42 run on the task (PASTIS-1 fold 4 r0p1s41_l16c_s42, CloudSen12 val
          r0c12s52_s42); bf16 iff both pass Z8 at bf16; Delta in points with a paired bootstrap over patches or tiles
          (1,000, seed 0). Kept iff Delta >= -M and the interval's lower end >= -2M, M = max(1.0, 2 x R0's seed SD).
  item 3  candidates: E (the lowest-map-CE save among the screened saves with F <= 0.02), L (l16b's and l16a's kept
          saves), RP-1/2 (the decision runs, seed 42); RP-1/4 and seed 43 beside it.
  item 4  passes = bounded and kept on both tasks; adopted = the cheapest passing candidate by its two fine-tunes'
          training A100-hours (ties within 10%: the lower larger-of-two F); none passes: Step 4.1's recipe stays, the best
          bounded candidate named (lowest larger-of-two F among those kept on both tasks).

    python eo/experiments/segmentation/step46_read.py      -> /data/enric/reports/a2_seg/step46/read.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step43_read import read_named                                # noqa: E402
from step52_read import IGNORE, RUNS, scored                      # noqa: E402
from step53_verdict import iou_per_class, miou                     # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step46")
F_DIR = OUT / "F"
GEN41 = Path("/data/enric/reports/a2_seg/step41/gen")
GEN52 = Path("/data/enric/reports/a2_seg/step52/gen")
GEN46 = OUT / "gen"
RTS = Path("/data/enric/rcp_a2s4/s46/RTS")
BASE_CE = 5.067648963092788           # 4.4's map base (step44/map_mass/decomp/map_base.json)
MARGIN_F = 0.02
R0_SD = {"pastis": 0.10, "cloudsen12": 0.19}       # Step 5.2, three seeds, bf16, points
M = {t: max(1.0, 2 * s) for t, s in R0_SD.items()}
N_BOOT, BOOT_SEED = 1000, 0
ROUTE_A = {"pastis": ("r0p1s41_l16c_s42_20261122_123818", GEN41),
           "cloudsen12": ("r0c12s52_s42_20261122_190655", GEN52)}
SCREEN = {"pastis": {"r0p1s41_l16c_s42_20261122_123818": [37, 74, 148, 296, 592, 851],
                     "r0p1s41_l16b_s42_20261122_123818": [37, 74, 148, 296, 592, 1295],
                     "r0p1s41_l16a_s42_20261122_123818": [37, 74, 148, 296, 592, 1739]},
          "cloudsen12": {"r0c12s52_s42_20261122_190655": [125, 250, 500, 1000, 2000, 3625]}}
LR_OF = {"l16c": 3e-3, "l16b": 1e-3, "l16a": 3e-4, "r0c12": 3e-3}


def F_of(name: str) -> dict:
    """F of one measured checkpoint against the map base, paired by row, with its decomposition beside it."""
    p = F_DIR / f"{name}.json"
    if not p.exists():
        return None
    c, b = json.loads(p.read_text()), json.loads((F_DIR / "map_base.json").read_text())
    if abs(b["ce_image_tokens"]["ce_full"] - BASE_CE) > 1e-5:
        raise RuntimeError(f"the map base reads {b['ce_image_tokens']['ce_full']}, not 4.4's {BASE_CE}")
    rc, rb = np.asarray(c["per_row"]["rows"]), np.asarray(b["per_row"]["rows"])
    if rc.shape != rb.shape or not np.array_equal(rc[:, 0], rb[:, 0]):
        raise RuntimeError(f"{name}: not the base's rows and positions")
    n, d = rb[:, 0], rc[:, 1] - rb[:, 1]
    F = float(d.sum() / n.sum())
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(n), size=(N_BOOT, len(n)))
    boot = d[idx].sum(1) / n[idx].sum(1)
    ci, cb = c["ce_image_tokens"], b["ce_image_tokens"]
    return {"F": F, "lo95": float(np.percentile(boot, 2.5)), "hi95": float(np.percentile(boot, 97.5)),
            "bounded": bool(F <= MARGIN_F), "map_mass": c["mean_map_mass"],
            "decomp_minus_base": {k: ci[k] - cb[k] for k in ("ce_full", "ce_base_rows", "ce_own_slot")},
            "per_modality_full_minus_base": {m: v["ce_full"] - b["ce_by_modality"][m]["ce_full"]
                                             for m, v in c["ce_by_modality"].items()}}


def train_hours(run: str, upto: int = None) -> float:
    """Training A100-hours of a run, or of its first `upto` steps (pro rata)."""
    r = json.loads((RUNS / run / "all_results.json").read_text())
    st = json.loads((RUNS / run / "trainer_state.json").read_text())
    h = float(r["train_runtime"]) / 3600
    gs = int(st["global_step"])
    return h * (min(upto, gs) / gs if upto else 1.0)


def pair(task: str, a: dict, b: dict, ref_gen: Path, cand_gen: Path) -> dict:
    """A candidate b against R0's seed-42 run a, by item 2 (a non-inferiority form of 4.3's pair)."""
    za, zb = a["z8_bf16"], b["z8_bf16"]
    if za is None or zb is None:
        return {"complete": False, "why": "a Z8 reading is missing"}
    prec = "bf16" if (za["pass"] and zb["pass"]) else "fp32"
    out = {"complete": True, "precision": prec, "z8_bf16": {"R0": za["pass"], "candidate": zb["pass"]}}
    if prec == "fp32" and not (a["fp32_readable"] and b["fp32_readable"]):
        out["stopped"] = "a run is readable at fp32 by neither rule: Enric decides"
        return out
    if a[prec]["k8"] is None or b[prec]["k8"] is None:
        return {"complete": False, "why": f"K = 8 maps at {prec} missing"}
    sub = "" if prec == "bf16" else "_fp32"
    ca = np.load(ref_gen / f"{a['run']}{sub}" / "full_k8" / "per_patch_cm.npy")
    cb = np.load(cand_gen / f"{b['run']}{sub}" / "full_k8" / "per_patch_cm.npy")
    if ca.shape != cb.shape:
        raise RuntimeError(f"the runs' maps differ in shape {ca.shape} {cb.shape}")
    ma, mb = miou(ca.sum(0), IGNORE[task]) * 100, miou(cb.sum(0), IGNORE[task]) * 100
    for m, r in ((ma, a), (mb, b)):
        if abs(m - r[prec]["k8"]["mIoU"] * 100) > 1e-9:
            raise RuntimeError(f"{r['run']}: its per-sample matrices do not sum to its read mIoU")
    idx = np.random.default_rng(BOOT_SEED).integers(0, ca.shape[0], size=(N_BOOT, ca.shape[0]))
    d = np.array([miou(cb[i].sum(0), IGNORE[task]) - miou(ca[i].sum(0), IGNORE[task]) for i in idx]) * 100
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    delta = mb - ma
    from eo.maps import palette as P
    names = P.TASKS[task].class_names
    pa, pb = iou_per_class(ca.sum(0), IGNORE[task]) * 100, iou_per_class(cb.sum(0), IGNORE[task]) * 100
    other = "fp32" if prec == "bf16" else "bf16"
    out.update({"R0": ma, "candidate": mb, "delta": delta, "lo95": lo, "hi95": hi, "M": M[task],
                "kept": bool(delta >= -M[task] and lo >= -2 * M[task]),
                "other_precision_delta": (None if a[other]["k8"] is None or b[other]["k8"] is None else
                                          (b[other]["k8"]["mIoU"] - a[other]["k8"]["mIoU"]) * 100),
                "per_class": [{"class": names[k], "R0": float(pa[k]), "candidate": float(pb[k]),
                               "delta": float(pb[k] - pa[k])} for k in range(len(names))
                              if not (np.isnan(pa[k]) and np.isnan(pb[k]))]})
    return out


def run_record(r: dict) -> dict:
    ps = (r.get("plateau_stop") or {}).get("fired") or {}
    rec = {"run": r["run"], "kept": r["selected"]["checkpoint"], "kept_epoch": r["selected"]["epoch"],
           "kept_ce": r["selected"]["eval_ce"], "plateau_epoch": ps.get("epoch"), "stop_step": ps.get("stop_step"),
           "health": r.get("health"), "z8_bf16_pass": (r["z8_bf16"] or {}).get("pass"),
           "z8_fp32_pass": (r["z8_fp32"] or {}).get("pass"),
           "maps": {p: {k: (v or {}).get("mIoU") for k, v in r[p].items() if isinstance(v, dict) or v is None}
                    for p in ("bf16", "fp32")}}
    x = r["bf16"]
    if None not in (x.get("wrong"), x.get("prior"), x.get("k1"), x.get("k1b")):
        mw, mp, mf, mf2 = x["wrong"]["mIoU"], x["prior"]["mIoU"], x["k1"]["mIoU"], x["k1b"]["mIoU"]
        rec["z9_bf16"] = {"check_pass": bool(abs(mw - mp) < abs(mw - mf)),
                          "control_fails_check": bool(not (abs(mf2 - mp) < abs(mf2 - mf)))}
    return rec


def e_record(run: str, step: int, task: str) -> dict:
    """read_named's record for candidate E's maps (jobs/a2s46_gen.sh: GEN46/E_<run>_<step>[_fp32]): Z8 and K = 8 at both
    precisions, from the screened save itself, not the run's kept save."""
    from step41_read import z8_of
    from step52_read import FP32_ROUNDING
    name = f"E_{run}_{step}"
    gb, gf = GEN46 / name, GEN46 / f"{name}_fp32"
    r = {"task": task, "run": name, "z8_bf16": z8_of(gb, "bf16"), "z8_fp32": z8_of(gb, "fp32")}
    zf = r["z8_fp32"]
    r["fp32_readable"] = None if zf is None else bool(
        zf["pass"] or zf["max_abs_logit_diff"] * FP32_ROUNDING <= zf["control_abs"])
    for p, g in (("bf16", gb), ("fp32", gf)):
        k8 = scored(g / "full_k8", task)
        if k8 is not None and k8.get("checkpoint") and not str(k8["checkpoint"]).endswith(f"checkpoint-{step}"):
            raise RuntimeError(f"{name}: K = 8 maps from {k8['checkpoint']}, not checkpoint-{step}")
        r[p] = {"k8": k8}
    return r


def screen() -> dict:
    out = {}
    for task, runs in SCREEN.items():
        rows = []
        for run, steps in runs.items():
            sel = json.loads((RUNS / run / "select41.json").read_text())
            for s in steps:
                f = F_of(f"{run}__{s}")
                rows.append({"run": run, "lr": LR_OF[run.split("_")[1] if "s41" in run else "r0c12"], "step": s,
                             "epoch": s // int(sel["steps_per_epoch"]), "map_ce": sel["eval_ce"].get(str(s)),
                             "kept_save": s == int(sel["selected"]["step"]), "F": f})
        measured = [r for r in rows if r["F"] is not None]
        bounded = [r for r in measured if r["F"]["bounded"]]
        e_pick = min(bounded, key=lambda r: r["map_ce"]) if bounded else None
        out[task] = {"saves": rows, "complete": len(measured) == len(rows), "bounded": len(bounded),
                     "E_checkpoint": e_pick and {"run": e_pick["run"], "step": e_pick["step"], "F": e_pick["F"]["F"],
                                                 "map_ce": e_pick["map_ce"]}}
    return out


def rp_runs() -> dict:
    rts = RTS.read_text().strip() if RTS.exists() else None
    names = {}
    for tag, task in (("rp50p", "pastis"), ("rp50c", "cloudsen12"), ("rp25p", "pastis")):
        for seed in (42, 43):
            hits = sorted(p.name for p in RUNS.glob(f"{tag}s46_s{seed}_*") if (p / "select41.json").exists())
            if hits:
                names[(tag, seed)] = (hits[-1], task)
    return names


def main():
    doc = {"rule": "docs/plans/a2_segmentation.md, Step 4, 4.6 (2026-11-24, ~19:20)", "M": M, "margin_F": MARGIN_F}
    base = json.loads((F_DIR / "map_base.json").read_text())["ce_image_tokens"]["ce_full"]
    doc["base_check"] = {"ce_full": base, "want": BASE_CE, "pass": abs(base - BASE_CE) <= 1e-5}
    doc["screen"] = screen()
    refs = {t: read_named(n, g, t) for t, (n, g) in ROUTE_A.items()}
    doc["references"] = {t: run_record(r) for t, r in refs.items()}
    cands = {}
    # E: the screened checkpoint, if any, with its maps (generated only when bounded; GEN46/E_<run>_<step>)
    for task in SCREEN:
        e = doc["screen"][task]["E_checkpoint"]
        entry = {"F": e and e["F"], "bounded": e is not None, "checkpoint": e}
        if e is not None:
            er = e_record(e["run"], e["step"], task)
            entry["hours"] = train_hours(e["run"], upto=e["step"])
            entry["pair"] = pair(task, refs[task], er, ROUTE_A[task][1], GEN46) if er["z8_bf16"] else None
            entry["maps"] = {p: (er[p]["k8"] or {}).get("mIoU") for p in ("bf16", "fp32")}
        cands.setdefault("E", {})[task] = entry
    # L: the kept saves of the lower LRs (PASTIS); CloudSen12 has no run at a lower LR unless one was trained
    for run in ("r0p1s41_l16b_s42_20261122_123818", "r0p1s41_l16a_s42_20261122_123818"):
        sel = json.loads((RUNS / run / "select41.json").read_text())
        f = F_of(f"{run}__{sel['selected']['step']}")
        cands.setdefault(f"L_{run.split('_')[1]}", {})["pastis"] = {"F": f and f["F"], "bounded": bool(f and f["bounded"]),
                                                                    "F_detail": f, "hours": train_hours(run)}
    # RP: the replay runs
    for (tag, seed), (name, task) in rp_runs().items():
        r = read_named(name, GEN46, task)
        f = F_of(f"{name}__{r['selected']['step']}")
        entry = {"run": run_record(r), "F": f and f["F"], "bounded": bool(f and f["bounded"]), "F_detail": f,
                 "hours": train_hours(name), "pair": pair(task, refs[task], r, ROUTE_A[task][1], GEN46)
                 if seed == 42 else None}
        if seed != 42:            # a replication: against R0's own seed 43 run, beside the rule
            entry["note"] = "seed 43, beside the rule"
        cands.setdefault(f"{tag}_s{seed}", {})[task] = entry
    doc["candidates"] = cands
    # item 4: RP-1/2 seed 42 is one candidate over both tasks
    rp = {"pastis": cands.get("rp50p_s42", {}).get("pastis"), "cloudsen12": cands.get("rp50c_s42", {}).get("cloudsen12")}
    verdict = {"RP50": None, "E": None, "L": None}
    if all(rp.values()) and all(v["F"] is not None and (v["pair"] or {}).get("complete") for v in rp.values()):
        kept = all((v["pair"] or {}).get("kept") for v in rp.values())
        bounded = all(v["bounded"] for v in rp.values())
        verdict["RP50"] = {"bounded_both": bounded, "kept_both": kept, "passes": bounded and kept,
                           "F_max": max(v["F"] for v in rp.values()), "hours": sum(v["hours"] for v in rp.values()),
                           "F": {t: v["F"] for t, v in rp.items()},
                           "delta": {t: (v["pair"] or {}).get("delta") for t, v in rp.items()},
                           "ci": {t: [(v["pair"] or {}).get("lo95"), (v["pair"] or {}).get("hi95")] for t, v in rp.items()}}
    verdict["E"] = {"bounded_both": all(cands["E"][t]["bounded"] for t in SCREEN),
                    "passes": False if not all(cands["E"][t]["bounded"] for t in SCREEN) else None}
    verdict["L"] = {k: v["pastis"]["bounded"] for k, v in cands.items() if k.startswith("L_")}
    passing = [k for k, v in verdict.items() if isinstance(v, dict) and v.get("passes")]
    doc["verdict"] = verdict
    doc["adopted"] = None
    if passing:
        # the cheapest by training hours (only RP50 can carry hours here unless E or L were completed)
        doc["adopted"] = min(passing, key=lambda k: verdict[k].get("hours", float("inf")))
    else:
        kept_both = [k for k, v in verdict.items() if isinstance(v, dict) and v.get("kept_both")]
        doc["best_bounded"] = min(kept_both, key=lambda k: verdict[k]["F_max"]) if kept_both else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "read.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(json.dumps({"base_check": doc["base_check"], "verdict": verdict, "adopted": doc["adopted"],
                      "best_bounded": doc.get("best_bounded")}, indent=1, default=float))
    # the screen as a table
    for task, s in doc["screen"].items():
        print(f"\n{task}: E's screen (F in nats, map CE)")
        for r in s["saves"]:
            f = r["F"]
            print(f"  {r['run'][:20]:20s} lr {r['lr']:.0e} step {r['step']:5d} ep {r['epoch']:2d} CE "
                  f"{r['map_ce'] if r['map_ce'] is None else round(r['map_ce'], 3)}{' kept' if r['kept_save'] else ''}"
                  f"  F {('%+.4f [%+.4f, %+.4f] mass %.3f' % (f['F'], f['lo95'], f['hi95'], f['map_mass'])) if f else '-'}")
    for k, v in cands.items():
        for t, e in v.items():
            if "pair" in e and e["pair"]:
                p = e["pair"]
                print(f"\n{k} {t}: F {'-' if e['F'] is None else '%+.4f' % e['F']}; K=8 {p.get('candidate', float('nan')):.2f} vs R0 "
                      f"{p.get('R0', float('nan')):.2f}: delta {p.get('delta', float('nan')):+.2f} "
                      f"[{p.get('lo95', float('nan')):+.2f}, {p.get('hi95', float('nan')):+.2f}] kept {p.get('kept')} "
                      f"({p.get('precision')}); {e['hours']:.2f} h")
            elif "run" in e:
                print(f"\n{k} {t}: F {e['F']}; {e['hours']:.2f} h")


if __name__ == "__main__":
    main()
