#!/usr/bin/env python
"""A2 Step 4.0: the read and the pick, by the protocol pre-registered on 2026-11-20 (docs/plans/a2_segmentation.md,
Step 4.0, items 4-9 and "How the protocol is read"). numpy and json only; reads what step4_select.py, eval_seg.py
(generate, cachecheck; on RCP) and step4_score.py (a2seg) wrote under /data/enric/reports/a2_seg/step4/:

  select.json                       the selected checkpoint per arm and the extension test
  gen/<arm>/z8_bf16.json            Z8 at bf16 on an A100 on the selected checkpoint (the 4-step rule, its control)
  gen/<arm>/full_k8/                K = 8 at T 1.0, seed 0: THE DECIDING NUMBER; k1/ = its sample 0 (the K = 1 full
                                    map), k1b/ = its sample 1 (Z9's functional control)
  gen/<arm>/full_greedy/            greedy at the selected checkpoint
  gen/<arm>/final_greedy/           greedy at the final save
  gen/<arm>/prior_k1/, wrong_k1/    the no-input and wrong-input controls (K = 1, T 1.0, seed 0)
  ceiling/ceiling_<arm>/            the stored true tokens through the same decode and scorer
  each scored dir: score.json (eo/eval/seg.score) and per_patch_cm.npy (482 x 20 x 20)

  * Z9 functional: |m_wrong - m_prior| < |m_wrong - m_full| passes; the control, the own input's second sample m_full',
    must FAIL the same check: |m_full' - m_prior| < |m_full' - m_full| false.
  * Bootstrap: the same 1,000 resamples of the 482 patches (seed 0) for every arm; each arm's macro mIoU (void out) from
    the resampled sum of its per-patch matrices; each pairwise difference's mean and 2.5-97.5 % interval. Not deciding.
  * Pick (Step 1.3's three-arm rule): the highest K = 8 mIoU; among the arms within 1.0 point of it (best - m <= 0.010,
    full precision), the fewest tokens: aligned (196), then short (256), then long (1,024).
  * Guard (Enric, 2026-11-20): if the long arm is not picked and the pick's K = 8 mIoU minus the long arm's < 0.03, the
    arms are compared again after a short R1 mid-training before anything is frozen.

    python eo/experiments/segmentation/step4_read.py [--root /data/enric/reports/a2_seg/step4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("aligned", "short", "long")
TOKENS = {"aligned": 196, "short": 256, "long": 1024}
VOID = 19                    # PASTIS's ignored class (eo/maps/palette.TASKS["pastis"].ignore)
WITHIN = 0.010
GUARD = 0.03
N_BOOT, BOOT_SEED = 1000, 0
CHANCE = Path("/data/enric/reports/a2_seg/step3/prior/summary.json")
FROZEN = Path("/data/enric/reports/a2_seg/step1/frozen")
FROZEN_DIRS = {"aligned": "aligned_B_b224_64000_long", "short": "short_B_b256_15360_long", "long": "long_C_c1000_256L"}


def miou(cm: np.ndarray, ignore=VOID) -> float:
    """eo/maps/scoring.Confusion.result's mIoU (step4_score.miou): macro over the classes with any target or
    prediction, the ignored class out."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), 0.0)
    w = np.ones(len(cm))
    w[ignore] = 0
    w[(cm.sum(0) + cm.sum(1)) == 0] = 0
    return float((w * iou).sum() / w.sum())


def load_scored(d: Path) -> dict:
    s = json.loads((d / "score.json").read_text())
    cm = np.load(d / "per_patch_cm.npy")
    m = float(s["result"]["mIoU"])
    if abs(miou(cm.sum(0)) - m) > 1e-12:
        raise RuntimeError(f"{d}: per-patch matrices do not sum to the scored mIoU")
    out = {"dir": str(d), "mIoU": m, "pixel_acc": float(s["result"]["pixel_acc"]),
           "boundary_acc": float(s["result"]["boundary_acc"]), "pred_top_share": float(s["pred_top_share"]),
           "pred_entropy_nats": float(s["pred_entropy_nats"]), "K": int(s["K"]), "cm": cm}
    rec = d / "record.json"
    if (d / "slot_mass.npy").exists():
        out["slot_mass"] = float(np.load(d / "slot_mass.npy").mean())
    if rec.exists():
        r = json.loads(rec.read_text())
        out.update({"checkpoint": r.get("checkpoint"), "variant": r.get("variant"), "precision": r.get("precision"),
                    "greedy": r.get("greedy"), "temperature": r.get("temperature"), "seed": r.get("seed"),
                    "gpu": r.get("gpu"), "fork_commit": r.get("fork_commit"), "n_views": len(r.get("views", []))})
    return out


def pick(k8: dict) -> dict:
    best = max(k8.values())
    within = [a for a in ARMS if best - k8[a] <= WITHIN]
    chosen = min(within, key=lambda a: TOKENS[a])
    top = max(ARMS, key=lambda a: k8[a])
    trail = k8[chosen] - k8["long"]
    guard = chosen != "long" and trail < GUARD
    return {"best_arm": top, "best_mIoU": best, "within_1pt": within, "pick": chosen, "pick_mIoU": k8[chosen],
            "long_trails_pick_by": trail, "guard_fires": bool(guard),
            "rule": "the highest K=8 mIoU; among the arms within 1.0 point of it (best - m <= 0.010), the fewest "
                    "tokens: aligned 196, short 256, long 1,024 (Step 1.3, three-arm rule)",
            "guard_rule": "the long arm not picked and pick - long < 0.03: compare again after a short R1 mid-training"}


def bootstrap(cms: dict) -> dict:
    n = len(next(iter(cms.values())))
    if any(len(c) != n for c in cms.values()):
        raise RuntimeError("arms scored on different patch sets")
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    vals = {a: np.array([miou(c[i].sum(0)) for i in idx]) for a, c in cms.items()}
    out = {"n_patches": n, "resamples": N_BOOT, "seed": BOOT_SEED, "per_arm_sd": {a: float(v.std()) for a, v in vals.items()}}
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            d = vals[a] - vals[b]
            out[f"{a}-{b}"] = {"point": miou(cms[a].sum(0)) - miou(cms[b].sum(0)), "mean": float(d.mean()),
                               "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
                               "share_positive": float((d > 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/enric/reports/a2_seg/step4")
    a = ap.parse_args()
    R = Path(a.root)
    sel = json.loads((R / "select.json").read_text())
    chance = json.loads(CHANCE.read_text())["references"]["pastis"]
    for arm in ARMS:
        rec = json.loads((FROZEN / FROZEN_DIRS[arm] / "record.json").read_text())
        if int(rec["tokens"]) != TOKENS[arm]:
            raise RuntimeError(f"{arm}: frozen record says {rec['tokens']} tokens, not {TOKENS[arm]}")
    rows, cms = {}, {}
    for arm in ARMS:
        g = R / "gen" / arm
        z8 = json.loads((g / "z8_bf16.json").read_text()) if (g / "z8_bf16.json").exists() else None
        s = {k: load_scored(g / p) for k, p in (("k8", "full_k8"), ("k1", "full_k8/k1"), ("k1b", "full_k8/k1b"),
                                                ("greedy", "full_greedy"), ("final_greedy", "final_greedy"),
                                                ("prior", "prior_k1"), ("wrong", "wrong_k1"))}
        ceil = load_scored(R / "ceiling" / f"ceiling_{arm}")
        sa = sel["arms"][arm]
        if s["k8"]["checkpoint"] is None or not str(s["k8"]["checkpoint"]).endswith(sa["selected"]["checkpoint"]):
            raise RuntimeError(f"{arm}: K = 8 maps are from {s['k8']['checkpoint']}, the protocol selected "
                               f"{sa['selected']['checkpoint']}")
        mw, mp, mf, mf2 = s["wrong"]["mIoU"], s["prior"]["mIoU"], s["k1"]["mIoU"], s["k1b"]["mIoU"]
        z9 = {"m_wrong": mw, "m_prior": mp, "m_full_k1": mf, "m_full_k1_sample1": mf2,
              "check_pass": bool(abs(mw - mp) < abs(mw - mf)),
              "control_fails_check": bool(not (abs(mf2 - mp) < abs(mf2 - mf)))}
        cms[arm] = s["k8"]["cm"]
        rows[arm] = {"tokens": TOKENS[arm], "selected": sa["selected"], "extension_test": sa["extension_test"],
                     "health": sa["health"],
                     "z8_bf16": None if z8 is None else {"pass": z8["pass"], "control_fires": z8["control_fires"],
                                                         "max_ulp_diff": z8["check"]["max_ulp_diff"],
                                                         "control_ulp": z8["control_positions_one_off"]["max_ulp_diff"],
                                                         "floor": z8["check"]["floor"], "gpu": z8["gpu"]},
                     "z9_functional": z9,
                     "mIoU": {"k8": s["k8"]["mIoU"], "k1": mf, "k1_sample1": mf2, "greedy": s["greedy"]["mIoU"],
                              "final_greedy": s["final_greedy"]["mIoU"], "prior_k1": mp, "wrong_k1": mw,
                              "ceiling": ceil["mIoU"]},
                     "pixel_acc_k8": s["k8"]["pixel_acc"], "boundary_acc_k8": s["k8"]["boundary_acc"],
                     "slot_mass": {k: s[k].get("slot_mass") for k in ("k8", "greedy", "final_greedy", "prior", "wrong")},
                     "gen": {k: {kk: s[k].get(kk) for kk in ("checkpoint", "precision", "gpu", "fork_commit", "n_views",
                                                             "K", "temperature", "seed")} for k in s}}
    k8 = {arm: rows[arm]["mIoU"]["k8"] for arm in ARMS}
    out = {"protocol": "A2 Step 4.0, pre-registered 2026-11-20 (docs/plans/a2_segmentation.md)",
           "chance": {"uniform_mIoU": chance["uniform_mIoU"], "constant_mIoU": chance["constant_mIoU"],
                      "source": str(CHANCE)},
           "arms": rows, "bootstrap_k8": bootstrap(cms), "pick": pick(k8),
           "checks_before_reading": {"z8_bf16_all_pass": all(r["z8_bf16"] and r["z8_bf16"]["pass"] for r in rows.values()),
                                     "z8_controls_fire": all(r["z8_bf16"] and r["z8_bf16"]["control_fires"] for r in rows.values()),
                                     "z9_functional_pass": all(r["z9_functional"]["check_pass"] for r in rows.values()),
                                     "z9_controls_fail": all(r["z9_functional"]["control_fails_check"] for r in rows.values())},
           "extension_fired": sel["extend"]}
    (R / "read.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["| arm | tokens | checkpoint (CE) | **K = 8** | K = 1 | greedy | greedy, final save | ceiling | no input (K = 1) "
             "| wrong input (K = 1) | slot mass (K = 8) | Z8 bf16 (steps) | Z9 functional |", "|" + "---|" * 13]
    for arm in ARMS:
        r = rows[arm]
        m = r["mIoU"]
        z8s = "-" if r["z8_bf16"] is None else f"{r['z8_bf16']['max_ulp_diff']:.2f} ({'pass' if r['z8_bf16']['pass'] else 'FAIL'})"
        lines.append(f"| {arm} | {TOKENS[arm]} | {r['selected']['checkpoint']} ({r['selected']['eval_ce']:.4f}) | "
                     f"**{m['k8']:.4f}** | {m['k1']:.4f} | {m['greedy']:.4f} | {m['final_greedy']:.4f} | {m['ceiling']:.4f} | "
                     f"{m['prior_k1']:.4f} | {m['wrong_k1']:.4f} | {r['slot_mass']['k8']:.4f} | {z8s} | "
                     f"{'pass' if r['z9_functional']['check_pass'] else 'FAIL'}, control "
                     f"{'fails it' if r['z9_functional']['control_fails_check'] else 'PASSES it'} |")
    b = out["bootstrap_k8"]
    lines += ["", "| difference (K = 8) | point | bootstrap mean | 95% interval |", "|---|---|---|---|"]
    for i, x in enumerate(ARMS):
        for y in ARMS[i + 1:]:
            d = b[f"{x}-{y}"]
            lines.append(f"| {x} - {y} | {d['point']:+.4f} | {d['mean']:+.4f} | [{d['lo95']:+.4f}, {d['hi95']:+.4f}] |")
    p = out["pick"]
    lines += ["", f"Pick: **{p['pick']}** (best {p['best_arm']} {p['best_mIoU']:.4f}; within 1.0 point: {p['within_1pt']}); "
              f"long trails the pick by {p['long_trails_pick_by']:+.4f}; guard {'FIRES' if p['guard_fires'] else 'does not fire'}. "
              f"Chance: uniform {chance['uniform_mIoU']:.4f}, best constant {chance['constant_mIoU']:.4f}."]
    (R / "read.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(out["checks_before_reading"]))


if __name__ == "__main__":
    main()
