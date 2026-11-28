#!/usr/bin/env python
"""A2 Step 7.0b, B1: the winter/summer check (docs/plans/a2_segmentation.md, 7.0b, "B1's read", "The winter/summer
check"; the rule written 2026-11-26 before any interval or seed-noise number). `.venv`, CPU; reads scored maps of fold
4 only (step52_score.py's per_patch_cm.npy), never the test fold.

    python eo/experiments/segmentation/step70b1_classes.py [--out <dir>]

  1. groups, fixed by sowing season from the class names: winter (autumn-sown) and spring/summer (spring-sown); the
     other seven classes are in neither;
  2. a group's IoU is the mean of its classes' IoU, each from the confusion matrix summed over the views (the scorer's
     definition; K = 8, bf16); Delta = B1 - 7.0 and C = Delta_winter - Delta_summer;
  3. view noise: 7.0's paired bootstrap over the 482 views (1,000 resamples, seed 0; one index draw per resample);
  4. seed noise: C between two seeds of one arm and recipe, over every PASTIS-1 seed pair A2 has at bf16 (8 pairs);
     sigma_C is their RMS;
  5. reading: beyond view noise iff C's interval excludes 0; beyond seed noise iff |C| > 2 sigma_C; both are needed
     before a GPU test, and either way it decides nothing;
  6. beside: C over all 792 splits of the same 12 crop classes into groups of 5 and 7.
Control: the same bootstrap with B1 replaced by 7.0 itself reads C = 0 with a zero-width interval, and a deliberately
swapped grouping (winter <-> summer) reads -C exactly.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step70_read as S70                           # noqa: E402
import step70b1_read as B                           # noqa: E402
from step52_read import GEN41, GEN52, IGNORE        # noqa: E402
from step53_verdict import iou_per_class            # noqa: E402

TASK = S70.TASK
B1_RUN = "p7b1_s42_20261126_084434"
WINTER = ("soft winter wheat", "winter barley", "winter rapeseed", "winter triticale", "winter durum wheat")
SUMMER = ("spring barley", "corn", "sunflower", "beet", "potatoes", "soybeans", "sorghum")
SEED_ARMS = {                                       # PASTIS-1, bf16 K = 8, one arm and recipe per entry
    "R0 (l16c, Step 4.1's recipe)": [GEN41 / "r0p1s41_l16c_s42_20261122_123818",
                                     GEN41 / "r0p1s41_l16c_s43_20261122_123818",
                                     GEN52 / "r0p1s52_s44_20261122_185744"],
    "Step 4.1 l64b": [GEN41 / "r0p1s41_l64b_s42_20261122_123818", GEN41 / "r0p1s41_l64b_s43_20261122_123818"],
    "Step 4.1 fulc": [GEN41 / "r0p1s41_fulc_s42_20261122_123818", GEN41 / "r0p1s41_fulc_s43_20261122_123818"],
    "R1": [GEN52 / "r1p1s52_s42_20261123_100922", GEN52 / "r1p1s52_s43_20261123_100922",
           GEN52 / "r1p1s52_s44_20261123_100922"],
}


def class_index():
    from eo.maps import palette as P
    names = list(P.TASKS[TASK].class_names)
    w = [names.index(c) for c in WINTER]
    s = [names.index(c) for c in SUMMER]
    return names, w, s


def groups(cm: np.ndarray, w, s) -> tuple:
    iou = iou_per_class(cm, IGNORE[TASK]) * 100
    return float(np.nanmean(iou[w])), float(np.nanmean(iou[s]))


def contrast(cx: np.ndarray, cy: np.ndarray, w, s) -> tuple:
    """Delta_winter, Delta_summer and C for x - y, from summed matrices."""
    xw, xs = groups(cx, w, s)
    yw, ys = groups(cy, w, s)
    return xw - yw, xs - ys, (xw - yw) - (xs - ys)


def boot(x: np.ndarray, y: np.ndarray, w, s) -> dict:
    """x - y over the views, paired (step70_read.pair's draw: one index draw per resample, seed 0)."""
    if x.shape != y.shape:
        raise RuntimeError(f"shapes {x.shape} {y.shape}")
    dw, ds, c = contrast(x.sum(0), y.sum(0), w, s)
    rng = np.random.default_rng(S70.BOOT_SEED)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(S70.N_BOOT, n))
    d = np.array([contrast(x[i].sum(0), y[i].sum(0), w, s) for i in idx])
    out = {}
    for j, (k, v) in enumerate((("delta_winter", dw), ("delta_summer", ds), ("C", c))):
        out[k] = {"delta": v, "lo95": float(np.percentile(d[:, j], 2.5)), "hi95": float(np.percentile(d[:, j], 97.5)),
                  "share_positive": float((d[:, j] > 0).mean())}
    return out


def seed_pairs(w, s) -> dict:
    runs, pairs = {}, []
    for arm, dirs in SEED_ARMS.items():
        cms = []
        for d in dirs:
            f = d / "full_k8" / "per_patch_cm.npy"
            if not f.exists():
                raise RuntimeError(f"missing {f}")
            cm = np.load(f).sum(0)
            gw, gs = groups(cm, w, s)
            runs[d.name] = {"arm": arm, "winter": gw, "summer": gs,
                            "mIoU": float(np.nanmean(iou_per_class(cm, IGNORE[TASK]))) * 100}
            cms.append((d.name, cm))
        for (na, ca), (nb, cb) in itertools.combinations(cms, 2):
            dw, ds, c = contrast(ca, cb, w, s)
            pairs.append({"arm": arm, "a": na, "b": nb, "delta_winter": dw, "delta_summer": ds, "C": c})
    cs = np.array([p["C"] for p in pairs])
    return {"runs": runs, "pairs": pairs, "n_pairs": len(pairs), "sigma_C_rms": float(np.sqrt((cs ** 2).mean())),
            "max_abs_C": float(np.abs(cs).max()),
            "sigma_C_rms_R0_only": float(np.sqrt(np.mean([p["C"] ** 2 for p in pairs if p["arm"].startswith("R0")])))}


def label_null(x: np.ndarray, y: np.ndarray, w, s, observed: float) -> dict:
    crops = list(w) + list(s)
    ix, iy = iou_per_class(x, IGNORE[TASK]) * 100, iou_per_class(y, IGNORE[TASK]) * 100
    d = ix - iy
    cs = []
    for g in itertools.combinations(crops, len(w)):
        rest = [k for k in crops if k not in g]
        cs.append(float(np.nanmean(d[list(g)]) - np.nanmean(d[rest])))
    cs = np.array(cs)
    return {"n_splits": len(cs), "share_at_or_above": float((cs >= observed - 1e-12).mean()),
            "rank_from_top": int((cs > observed + 1e-12).sum()) + 1, "max": float(cs.max()), "min": float(cs.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(B.OUT))
    a = ap.parse_args()
    names, w, s = class_index()
    d70, db1 = S70.GEN / B.P70_RUN / "full_k8", B.GEN / B1_RUN / "full_k8"
    if S70.views_of(d70) != S70.views_of(db1):
        raise RuntimeError("7.0's and B1's views differ: the bootstrap would not pair")
    c70, cb1 = S70.cm_of(d70), S70.cm_of(db1)
    res = {"rule": "A2 plan, 7.0b, B1's read, 'The winter/summer check' (written 2026-11-26 before its numbers)",
           "winter": WINTER, "summer": SUMMER, "precision": "bf16", "views": int(c70.shape[0])}
    # the read's own per-class numbers, as a path check against read.json
    rj = json.loads((B.OUT / "read.json").read_text())
    pc = {r["class"]: r for r in rj["per_class"]}
    i70, ib1 = iou_per_class(c70.sum(0), IGNORE[TASK]) * 100, iou_per_class(cb1.sum(0), IGNORE[TASK]) * 100
    worst = max(max(abs(i70[names.index(c)] - pc[c]["p70"]), abs(ib1[names.index(c)] - pc[c]["b1"]))
                for c in WINTER + SUMMER)
    if worst > 1e-9:
        raise RuntimeError(f"per-class IoU off read.json by {worst}")
    res["path_check_vs_read_json"] = worst
    res["per_class"] = [{"class": c, "group": "winter" if c in WINTER else "summer", "p70": float(i70[names.index(c)]),
                         "b1": float(ib1[names.index(c)])} for c in WINTER + SUMMER]
    res["groups"] = {"p70": dict(zip(("winter", "summer"), groups(c70.sum(0), w, s))),
                     "b1": dict(zip(("winter", "summer"), groups(cb1.sum(0), w, s)))}
    res["b1_minus_70"] = boot(cb1, c70, w, s)
    # controls: the check can read nothing, and it can read the opposite sign
    ctl_self = boot(c70, c70, w, s)["C"]
    ctl_swap = contrast(cb1.sum(0), c70.sum(0), s, w)[2]
    res["control_self"] = ctl_self
    res["control_swap_C"] = ctl_swap
    res["controls_pass"] = bool(ctl_self["delta"] == 0 and ctl_self["lo95"] == 0 and ctl_self["hi95"] == 0
                                and abs(ctl_swap + res["b1_minus_70"]["C"]["delta"]) < 1e-12)
    res["seed_noise"] = seed_pairs(w, s)
    C = res["b1_minus_70"]["C"]
    sig = res["seed_noise"]["sigma_C_rms"]
    res["beyond_view_noise"] = bool(C["lo95"] > 0 or C["hi95"] < 0)
    res["beyond_seed_noise"] = bool(abs(C["delta"]) > 2 * sig)
    res["worth_gpu_test"] = res["beyond_view_noise"] and res["beyond_seed_noise"]
    res["label_null"] = label_null(cb1.sum(0), c70.sum(0), w, s, C["delta"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(res, indent=1, default=float))
    g, bm = res["groups"], res["b1_minus_70"]
    ln = res["label_null"]
    lines = ["# A2 Step 7.0b, B1: the winter/summer check (bf16, K = 8, fold 4)", "",
             f"- controls pass: {res['controls_pass']} (self C {ctl_self['delta']:+.3f} "
             f"[{ctl_self['lo95']:+.3f}, {ctl_self['hi95']:+.3f}]; swapped groups C {ctl_swap:+.3f})",
             f"- winter: 7.0 {g['p70']['winter']:.2f}, B1 {g['b1']['winter']:.2f}; summer: 7.0 {g['p70']['summer']:.2f}, "
             f"B1 {g['b1']['summer']:.2f}"]
    for k in ("delta_winter", "delta_summer", "C"):
        p = bm[k]
        lines.append(f"- {k}: {p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}], {100 * p['share_positive']:.1f}% > 0")
    sn = res["seed_noise"]
    lines += [f"- seed noise (PASTIS-1, {sn['n_pairs']} pairs): sigma_C {sn['sigma_C_rms']:.2f} (R0 only "
              f"{sn['sigma_C_rms_R0_only']:.2f}), max |C| {sn['max_abs_C']:.2f}; 2 sigma_C = {2 * sig:.2f}",
              f"- beyond view noise: {res['beyond_view_noise']}; beyond seed noise: {res['beyond_seed_noise']}; "
              f"worth a GPU test: {res['worth_gpu_test']}",
              f"- label null: rank {ln['rank_from_top']} of {ln['n_splits']} splits, share at or above "
              f"{100 * ln['share_at_or_above']:.1f}% (range {ln['min']:+.2f} to {ln['max']:+.2f})"]
    (out / "classes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for p in sn["pairs"]:
        print(f"  {p['arm']:30s} {p['a'][:22]} - {p['b'][:22]}: dW {p['delta_winter']:+.2f} dS {p['delta_summer']:+.2f}"
              f" C {p['C']:+.2f}")


if __name__ == "__main__":
    main()
