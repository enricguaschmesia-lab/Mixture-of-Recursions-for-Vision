#!/usr/bin/env python
"""A2 Step 4.0's follow-ups (Enric, 2026-11-20, evening): the read, by the rules pre-registered in
docs/plans/a2_segmentation.md (Step 4.0, "4.0 follow-ups", Readings) before any of their data. numpy and json only.

Reads Step 4.0's maps (/data/enric/reports/a2_seg/step4/gen/<arm>/) and the follow-ups' (.../step4/followup/gen/<run>/),
both scored by step4_score.py (score.json + per_patch_cm.npy), the follow-ups' select.json and Z8 records.
  * every difference is of K = 8 fold-4 mIoU, with a paired bootstrap: the SAME 1,000 resamples of the 482 patches
    (seed 0) for all eight runs; each run's macro mIoU (void out) from its resampled per-patch matrices;
  1. alignment: D1 = aligned - aligned_shift99. SHOWN if D1 >= 0.020 and its interval lies above 0; CLOSED if D1 < 0.010
     or its interval includes 0; otherwise a small effect.
  2. geometry: each nested arm against the pick, D = nested - aligned: it BEATS THE FROZEN PICK if D > 0.010 and its
     interval lies above 0 (then the freeze is reopened for Enric, never changed here). The layout: D2 = c1000_224an -
     c1000_224ar, the four-block layout HELPS if D2 > 0 with its interval above 0. Length at 1,000 codes: c1000_224ar
     (784) against long (1,024), described only.
  3. codes: s = (m_b256_64000 - m_short) / (m_aligned - m_short). The codebook EXPLAINS the gap if s >= 0.5 and the
     interval of m_b256_64000 - m_short lies above 0; otherwise it does not.

    python eo/experiments/segmentation/step4f_read.py [--root /data/enric/reports/a2_seg/step4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4_read import load_scored, miou, CHANCE   # noqa: E402

BASE = ("aligned", "short", "long")
FOLLOW = ("aligned_shift99", "c4375_224n", "c1000_224an", "c1000_224ar", "b256_64000")
TOKENIZER = {"aligned_shift99": "aligned", "c4375_224n": "c4375_224", "c1000_224an": "c1000_224a",
             "c1000_224ar": "c1000_224a", "b256_64000": "b256_64000"}
N_BOOT, SEED = 1000, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/enric/reports/a2_seg/step4")
    a = ap.parse_args()
    R = Path(a.root)
    F = R / "followup"
    sel = json.loads((F / "select.json").read_text())
    k8, cms, rows = {}, {}, {}
    for arm in BASE:
        s = load_scored(R / "gen" / arm / "full_k8")
        k8[arm], cms[arm] = s["mIoU"], s["cm"]
    for run in FOLLOW:
        g = F / "gen" / run
        s = {k: load_scored(g / p) for k, p in (("k8", "full_k8"), ("k1", "full_k8/k1"), ("k1b", "full_k8/k1b"),
                                                ("greedy", "full_greedy"), ("final_greedy", "final_greedy"),
                                                ("prior", "prior_k1"), ("wrong", "wrong_k1"))}
        tok = TOKENIZER[run]
        ceil = load_scored((R / "ceiling" / "ceiling_aligned") if tok == "aligned" else (F / "ceiling" / f"ceiling_{tok}"))
        z8 = json.loads((g / "z8_bf16.json").read_text())
        sa = sel["arms"][run]
        if not str(s["k8"]["checkpoint"]).endswith(sa["selected"]["checkpoint"]):
            raise RuntimeError(f"{run}: K = 8 maps from {s['k8']['checkpoint']}, the rule selected {sa['selected']['checkpoint']}")
        mw, mp, mf, mf2 = s["wrong"]["mIoU"], s["prior"]["mIoU"], s["k1"]["mIoU"], s["k1b"]["mIoU"]
        k8[run], cms[run] = s["k8"]["mIoU"], s["k8"]["cm"]
        rows[run] = {"tokenizer": tok, "selected": sa["selected"], "extension_test": sa["extension_test"],
                     "health": sa["health"],
                     "mIoU": {"k8": s["k8"]["mIoU"], "k1": mf, "k1_sample1": mf2, "greedy": s["greedy"]["mIoU"],
                              "final_greedy": s["final_greedy"]["mIoU"], "prior_k1": mp, "wrong_k1": mw,
                              "ceiling": ceil["mIoU"]},
                     "pixel_acc_k8": s["k8"]["pixel_acc"], "slot_mass_k8": s["k8"].get("slot_mass"),
                     "z8_bf16": {"pass": z8["pass"], "control_fires": z8["control_fires"],
                                 "max_ulp_diff": z8["check"]["max_ulp_diff"],
                                 "control_ulp": z8["control_positions_one_off"]["max_ulp_diff"]},
                     "z9_functional": {"check_pass": bool(abs(mw - mp) < abs(mw - mf)),
                                       "control_fails_check": bool(not (abs(mf2 - mp) < abs(mf2 - mf)))},
                     "gen": {"precision": s["k8"].get("precision"), "gpu": s["k8"].get("gpu"), "n_views": s["k8"].get("n_views")}}
    n = len(cms["aligned"])
    if any(len(c) != n for c in cms.values()):
        raise RuntimeError("runs scored on different patch sets")
    idx = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    boot = {r: np.array([miou(c[i].sum(0)) for i in idx]) for r, c in cms.items()}

    def diff(x, y):
        d = boot[x] - boot[y]
        return {"point": k8[x] - k8[y], "mean": float(d.mean()), "lo95": float(np.percentile(d, 2.5)),
                "hi95": float(np.percentile(d, 97.5))}

    d1 = diff("aligned", "aligned_shift99")
    t1 = ("shown" if d1["point"] >= 0.020 and d1["lo95"] > 0 else
          "closed" if d1["point"] < 0.010 or d1["lo95"] <= 0 <= d1["hi95"] else "small effect")
    nested = {r: diff(r, "aligned") for r in ("c4375_224n", "c1000_224an")}
    beats = {r: bool(d["point"] > 0.010 and d["lo95"] > 0) for r, d in nested.items()}
    d2 = diff("c1000_224an", "c1000_224ar")
    length = diff("c1000_224ar", "long")
    gap = k8["aligned"] - k8["short"]
    d3 = diff("b256_64000", "short")
    s = d3["point"] / gap
    t3 = "explains" if (s >= 0.5 and d3["lo95"] > 0) else "does not explain"
    out = {"protocol": "A2 Step 4.0 follow-ups, pre-registered 2026-11-20 (docs/plans/a2_segmentation.md)",
           "k8_mIoU": k8, "runs": rows, "bootstrap": {"resamples": N_BOOT, "seed": SEED, "n_patches": n},
           "test1_alignment": {"D1": d1, "verdict": t1},
           "test2_geometry": {"nested_vs_aligned": nested, "beats_frozen_pick": beats,
                              "reopen_freeze": bool(any(beats.values())),
                              "layout_D2": d2, "layout_helps": bool(d2["point"] > 0 and d2["lo95"] > 0),
                              "length_784_raster_vs_long_1024": length},
           "test3_codes": {"gap_aligned_minus_short": gap, "b256_minus_short": d3, "share_closed": s, "verdict": t3},
           "checks": {"z8_all_pass": all(r["z8_bf16"]["pass"] for r in rows.values()),
                      "z8_controls_fire": all(r["z8_bf16"]["control_fires"] for r in rows.values()),
                      "z9_all_pass": all(r["z9_functional"]["check_pass"] for r in rows.values()),
                      "z9_controls_fail": all(r["z9_functional"]["control_fails_check"] for r in rows.values())},
           "chance": {k: json.loads(CHANCE.read_text())["references"]["pastis"][k] for k in ("uniform_mIoU", "constant_mIoU")}}
    (F / "read.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["| run | tokenizer | checkpoint (CE) | **K = 8** | K = 1 | greedy | greedy, final | ceiling | no input | wrong input | "
         "Z8 bf16 | Z9 | extension test |", "|" + "---|" * 13]
    for r in FOLLOW:
        x = rows[r]
        m = x["mIoU"]
        L.append(f"| {r} | {x['tokenizer']} | {x['selected']['checkpoint']} ({x['selected']['eval_ce']:.4f}) | **{m['k8']:.4f}** | "
                 f"{m['k1']:.4f} | {m['greedy']:.4f} | {m['final_greedy']:.4f} | {m['ceiling']:.4f} | {m['prior_k1']:.4f} | "
                 f"{m['wrong_k1']:.4f} | {x['z8_bf16']['max_ulp_diff']:.2f} ({'pass' if x['z8_bf16']['pass'] else 'FAIL'}) | "
                 f"{'pass' if x['z9_functional']['check_pass'] else 'FAIL'} | "
                 f"{100 * x['extension_test']['relative_drop']:.2f}% ({'fires' if x['extension_test']['fires'] else 'no'}) |")
    fmt = lambda d: f"{d['point']:+.4f} [{d['lo95']:+.4f}, {d['hi95']:+.4f}]"
    L += ["", f"1. Alignment: aligned - aligned_shift99 = {fmt(d1)} -> **{t1}**.",
          f"2. Geometry: c4375_224n - aligned = {fmt(nested['c4375_224n'])}, c1000_224an - aligned = {fmt(nested['c1000_224an'])}"
          f" -> beats the frozen pick: {beats}; layout c1000_224an - c1000_224ar = {fmt(d2)} -> helps: "
          f"{out['test2_geometry']['layout_helps']}; length c1000_224ar - long = {fmt(length)} (described).",
          f"3. Codes: b256_64000 - short = {fmt(d3)}, share of the aligned - short gap ({gap:.4f}) closed s = {s:.2f} -> **{t3}**.",
          f"Step 4.0's arms (K = 8): aligned {k8['aligned']:.4f}, short {k8['short']:.4f}, long {k8['long']:.4f}."]
    (F / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps(out["checks"]))


if __name__ == "__main__":
    main()
