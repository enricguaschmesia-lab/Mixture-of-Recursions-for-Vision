#!/usr/bin/env python
"""A2 Step 4.5's read (docs/plans/a2_segmentation.md, Step 4, "4.5, the grid decision"; rule written 2026-11-24 before
any of its numbers). `.venv`, CPU: reads scored maps only.

  - The pairs: the 784 grid (c4375_224n at Step 4.1's recipe) against R0's run of the same seed (Route A, K = 8):
    PASTIS-1 fold 4 seeds 42 and 43, CloudSen12 val seed 42. Precision per task: bf16 iff every run of the task
    passes Z8 at bf16; otherwise fp32 for all, each run readable by the 1,000x rule.
  - Delta = 784 - A in mIoU points, a paired bootstrap (1,000, seed 0) over patches (PASTIS, 482) or tiles
    (CloudSen12, 535). PASTIS's two-seed mean uses one index draw for both pairs.
  - The matched reference: Route A at K_m (= 8 x round(t784 / t196)), whose first 8 samples must equal its K = 8 run's
    (step45_kcurve's k8_equals_full_k8.json), against the 784 grid at K = 8.
  - The rule: (c) iff C1 (Delta_P(43) > +1.0, lo > 0) and C2 (mean Delta_P >= +2.0, lo > 0) and C3 (Delta_C >= +2.0,
    lo > 0) and C4 (matched Delta > +1.0 with lo > 0 on both tasks); otherwise (a).
Writes /data/enric/reports/a2_seg/step45/read.{json,md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step43_read import BOOT_SEED, N_BOOT, read_named              # noqa: E402
from step52_read import IGNORE                                     # noqa: E402
from step53_verdict import iou_per_class, miou                     # noqa: E402

GEN41 = Path("/data/enric/reports/a2_seg/step41/gen")
GEN52 = Path("/data/enric/reports/a2_seg/step52/gen")
GEN43 = Path("/data/enric/reports/a2_seg/step43/gen")
GEN45 = Path("/data/enric/reports/a2_seg/step45/gen")
OUT = Path("/data/enric/reports/a2_seg/step45")
S45 = Path("/data/enric/rcp_a2s4/s45")
C_RUN = (S45 / "RUN_c12").read_text().strip() if (S45 / "RUN_c12").exists() else None
PAIRS = {   # task: [(seed, (A run, A gen), (784 run, 784 gen))]
    "pastis": [(42, ("r0p1s41_l16c_s42_20261122_123818", GEN41), ("g784s43_s42_20261123_175739", GEN43)),
               (43, ("r0p1s41_l16c_s43_20261122_123818", GEN41), ("g784s43_s43_20261124_113201", GEN43))],
    "cloudsen12": [(42, ("r0c12s52_s42_20261122_190655", GEN52), (C_RUN, GEN45))],
}
BAR_ROUTE, BAR_ABL = 2.0, 1.0


def cm(gen: Path, run: str, prec: str, sub: str) -> np.ndarray:
    return np.load(gen / f"{run}{'' if prec == 'bf16' else '_fp32'}" / sub / "per_patch_cm.npy")


def boot(pairs, ign, idx):
    """pairs: [(cm_A, cm_784)]; the mean over pairs of mIoU(784) - mIoU(A), in points, and its bootstrap."""
    point = float(np.mean([miou(b.sum(0), ign) - miou(a.sum(0), ign) for a, b in pairs])) * 100
    d = np.array([np.mean([miou(b[i].sum(0), ign) - miou(a[i].sum(0), ign) for a, b in pairs]) for i in idx]) * 100
    return {"delta": point, "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "share_positive": float((d > 0).mean())}


def km_of(task: str, run_a: str, prec: str) -> int | None:
    sfx = "" if prec == "bf16" else "_fp32"
    ks = sorted(int(p.name[6:]) for p in (GEN45 / f"{run_a}{sfx}").glob("full_k*") if p.name[6:].isdigit()
                and int(p.name[6:]) > 8) if (GEN45 / f"{run_a}{sfx}").exists() else []
    return ks[-1] if ks else None


def main():
    out, problems = {"tasks": {}}, []
    for task, pairs in PAIRS.items():
        ign = IGNORE[task]
        runs = []
        for seed, (ra, ga), (rb, gb) in pairs:
            if rb is None or not (Path("/data/enric/runs/pretrain/a2_seg") / rb / "select41.json").exists():
                problems.append(f"{task} s{seed}: the 784 run is not here yet")
                continue
            runs.append((seed, read_named(ra, ga, task), read_named(rb, gb, task), ga, gb))
        if len(runs) != len(pairs):
            continue
        z8 = all(r["z8_bf16"] is not None and r["z8_bf16"]["pass"] for _, a, b, _, _ in runs for r in (a, b))
        prec = "bf16" if z8 else "fp32"
        if prec == "fp32" and not all(r["fp32_readable"] for _, a, b, _, _ in runs for r in (a, b)):
            problems.append(f"{task}: a run is readable at fp32 by neither rule")
            continue
        t = {"precision": prec, "seeds": {}}
        cms = []
        for seed, a, b, ga, gb in runs:
            ca, cb = cm(ga, a["run"], prec, "full_k8"), cm(gb, b["run"], prec, "full_k8")
            if ca.shape != cb.shape:
                raise RuntimeError(f"{task} s{seed}: shapes {ca.shape} {cb.shape}")
            K = km_of(task, a["run"], prec)
            m = None
            if K is not None:
                d = GEN45 / f"{a['run']}{'' if prec == 'bf16' else '_fp32'}" / f"full_k{K}"
                eqf = d / "k8_equals_full_k8.json"
                if not eqf.exists() or not json.loads(eqf.read_text())["all_equal"]:
                    raise RuntimeError(f"{d}: its first 8 samples are not the K = 8 run's (plan item 3): stop")
                if (d / "per_patch_cm.npy").exists():
                    m = np.load(d / "per_patch_cm.npy")
                curve = {}
                for kd in sorted((q for q in d.glob("k[0-9]*") if q.is_dir() and q.name[1:].isdigit()), key=lambda p: int(p.name[1:])):
                    if (kd / "score.json").exists() and kd.name not in ("k1", "k1b"):
                        curve[int(kd.name[1:])] = json.loads((kd / "score.json").read_text())["result"]["mIoU"] * 100
            names = __import__("eo.maps.palette", fromlist=["x"]).TASKS[task].class_names
            pa, pb = iou_per_class(ca.sum(0), ign) * 100, iou_per_class(cb.sum(0), ign) * 100
            t["seeds"][seed] = {
                "A": {k: a.get(k) for k in ("run", "selected", "plateau_stop", "health", "z8_bf16", "fp32_readable")},
                "g784": {k: b.get(k) for k in ("run", "selected", "plateau_stop", "health", "z8_bf16", "fp32_readable")},
                "mIoU_A": miou(ca.sum(0), ign) * 100, "mIoU_784": miou(cb.sum(0), ign) * 100,
                "other_precision": {p: {"A": (a[p]["k8"] or {}).get("mIoU"), "784": (b[p]["k8"] or {}).get("mIoU")}
                                    for p in ("bf16", "fp32")},
                "k8_seconds": {"A": a[prec].get("k8_seconds"), "784": b[prec].get("k8_seconds")},
                "K_m": K, "mIoU_A_Km": None if m is None else miou(m.sum(0), ign) * 100,
                "A_vote_curve": curve if K is not None else None,
                "per_class": [{"class": names[k], "A": float(pa[k]), "784": float(pb[k]), "delta": float(pb[k] - pa[k])}
                              for k in range(len(names)) if not (np.isnan(pa[k]) and np.isnan(pb[k]))]}
            cms.append((seed, ca, cb, m))
        n = cms[0][1].shape[0]
        idx = np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))
        for seed, ca, cb, _ in cms:
            t["seeds"][seed]["boot"] = boot([(ca, cb)], ign, idx)
        t["mean"] = boot([(ca, cb) for _, ca, cb, _ in cms], ign, idx)
        if all(m is not None for *_, m in cms):
            t["matched"] = boot([(m, cb) for _, _, cb, m in cms], ign, idx)
        else:
            t["matched"] = None
            problems.append(f"{task}: Route A at K_m not scored yet")
        out["tasks"][task] = t
    out["problems"] = problems
    T = out["tasks"]
    if not problems and set(T) == set(PAIRS):
        p43 = T["pastis"]["seeds"][43]["boot"]
        up = lambda b, bar, strict: (b["delta"] > bar if strict else b["delta"] >= bar) and b["lo95"] > 0  # noqa: E731
        C = {"C1 PASTIS seed 43 alone > +1.0": up(p43, BAR_ABL, True),
             "C2 PASTIS two-seed mean >= +2.0": up(T["pastis"]["mean"], BAR_ROUTE, False),
             "C3 CloudSen12 >= +2.0": up(T["cloudsen12"]["mean"], BAR_ROUTE, False),
             "C4 matched > +1.0 on both tasks": up(T["pastis"]["matched"], BAR_ABL, True)
             and up(T["cloudsen12"]["matched"], BAR_ABL, True)}
        out["conditions"] = C
        out["decision"] = "(c) move the pipeline to 784" if all(C.values()) else "(a) keep 196; 784 is an ablation row"
    else:
        out["decision"] = None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "read.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    md = ["# A2 Step 4.5's read", "", f"decision: **{out['decision']}**", ""]
    md += ["| task | seed | precision | A (K = 8) | 784 (K = 8) | Delta [95%] | A at K_m | 784 - A(K_m) [95%] |",
           "|---|---|---|---|---|---|---|---|"]
    for task, t in T.items():
        for seed, s in t["seeds"].items():
            b = s["boot"]
            km = "" if s["mIoU_A_Km"] is None else f"{s['mIoU_A_Km']:.2f} (K {s['K_m']})"
            md.append(f"| {task} | {seed} | {t['precision']} | {s['mIoU_A']:.2f} | {s['mIoU_784']:.2f} | "
                      f"{b['delta']:+.2f} [{b['lo95']:+.2f}, {b['hi95']:+.2f}] | {km} | |")
        mb, mm = t["mean"], t["matched"]
        mt = "" if mm is None else f"{mm['delta']:+.2f} [{mm['lo95']:+.2f}, {mm['hi95']:+.2f}]"
        md.append(f"| {task} | mean | | | | {mb['delta']:+.2f} [{mb['lo95']:+.2f}, {mb['hi95']:+.2f}] | | {mt} |")
    md += ["", *(f"- {k}: {v}" for k, v in out.get("conditions", {}).items()), *(f"- problem: {p}" for p in problems)]
    (OUT / "read.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
