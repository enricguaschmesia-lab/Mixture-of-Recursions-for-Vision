#!/usr/bin/env python
"""A2 Step 7.0h's read (docs/plans/a2_segmentation.md, Step 7, "7.0h, soft voting on the stored samples"; the rule
pre-registered 2026-11-27 before any of its numbers). `.venv`, CPU; reads step70h_soft.py's per-patch sums and 7.0e's
stored class maps.

    python eo/experiments/segmentation/step70h_read.py [--out <dir>]

  validity  every sample's argmax of the log-pooled scores equals its stored class map (soft.json's check); the hard vote
            recomputed from the stored maps equals the scorer's per-patch matrices (K8 sets);
  primary   corrected soft - corrected hard (log-pooled; alpha cross-fitted over fold 4's halves for each, 7.0e's grid
            and prior) on the stack (E784's TTA-64): gains iff its interval is above 0 and the difference is above 0 on
            B1 K8, E784 K8 and B1 TTA-64; costs on the mirror; else level;
  beside    plain soft - plain hard on all four; prob-pooled - log-pooled (corrected soft); the alphas; per class.
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
from step4_score import per_patch_cm                # noqa: E402
from step52_read import IGNORE                      # noqa: E402
from step53_verdict import iou_per_class, miou      # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70h")
SOFT = OUT / "soft"
SETS = {   # name: (soft dir, subset, hard: ("k8", votes name) or ("tta", votes name)), the stack first
    "E784 TTA-64 (the stack)": ("e784_tta", "tta64", ("tta", "p7fstack_20261127_080904")),
    "B1 K8": ("b1_k8", "k8", ("k8", "b1_k8")),
    "E784 K8": ("e784_k8", "k8", ("k8", "p7e784_s42_20261127_004451_k8")),
    "B1 TTA-64": ("b1_tta", "tta64", ("tta", "p7etta_20261127_004451")),
}
BESIDE_SETS = {"E784 S8x8's own K8": ("e784_tta", "k8", ("tta_k8", "p7fstack_20261127_080904")),
               "B1 T8x8's own K8": ("b1_tta", "k8", ("tta_k8", "p7etta_20261127_004451"))}
STORED_CM = {"b1_k8": K.gen_dir(K.B1_RUN) / "full_k8" / "per_patch_cm.npy",
             "p7e784_s42_20261127_004451_k8": RD.GEN / "p7e784_s42_20261127_004451" / "full_k8" / "per_patch_cm.npy"}
PRIMARY = "E784 TTA-64 (the stack)"


def soft_cms(S: np.ndarray, labels: np.ndarray, w) -> np.ndarray:
    """(n, C, H, W) summed per-sample distributions -> per-patch matrices of argmax_c w_c S_c (ties to the lower class)."""
    pred = np.empty((S.shape[0],) + S.shape[2:], dtype=np.uint8)
    wt = np.asarray(w, dtype=np.float32)[None, :, None, None]
    for s in range(0, S.shape[0], 64):
        pred[s:s + 64] = np.argmax(np.asarray(S[s:s + 64]) * wt, axis=1)
    return per_patch_cm(pred, labels, RD.N_CLASSES, IGNORE[RD.TASK])


def hard_of(kind: str, name: str, prior: dict) -> dict:
    if kind == "k8":
        return RD.test2_of(name, prior)
    maps, vj, _ = RD.load_maps(name)
    return RD.test2_of(None, prior, stacks=RD.tta_stacks(maps, vj, "tta64" if kind == "tta" else "k8"))


def soft_of(dirname: str, subset: str, pool: str, sids, labels, prior: dict) -> dict:
    S = np.load(SOFT / dirname / f"{subset}_{pool}.npy", mmap_mode="r")
    if S.shape[0] != len(sids):
        raise RuntimeError(f"{dirname}/{subset}_{pool}: {S.shape[0]} patches, not {len(sids)}")
    cms = {a: soft_cms(S, labels, VT.class_weights(prior, a, RD.N_CLASSES, IGNORE[RD.TASK])) for a in RD.ALPHAS}
    cf, chosen = RD.cross_fit(cms, RD.halves(sids))
    return {"plain": cms[0.0], "cf": cf, "chosen": chosen,
            "by_alpha": {str(a): miou(c.sum(0), IGNORE[RD.TASK]) * 100 for a, c in cms.items()}}


def verdict(primary: dict, others: list) -> str:
    c = RD.decode_call(primary)
    if c == "gains" and all(o["delta"] > 0 for o in others):
        return "gains"
    if c == "costs" and all(o["delta"] < 0 for o in others):
        return "costs"
    return "level"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    prior = json.loads(RD.PRIOR.read_text())
    res = {"rule": "A2 plan, Step 7, 7.0h (pre-registered 2026-11-27)", "sets": {}}
    why = []
    for d in {v[0] for v in SETS.values()}:
        sj = json.loads((SOFT / d / "soft.json").read_text())
        chk = sj.get("argmax_check") or {}
        res.setdefault("argmax_checks", {})[d] = chk
        if not chk or chk.get("mismatches") != 0:
            why.append(f"{d}: the re-decoded argmax differs from the stored class maps ({chk})")
    per_class = {}
    for label, (d, subset, (kind, name)) in list(SETS.items()) + list(BESIDE_SETS.items()):
        h = hard_of(kind, name, prior)
        sids, labels = h["_sids"], h["_labels"]
        if kind == "k8" and name in STORED_CM and not np.array_equal(np.load(STORED_CM[name]), h["_cms"][0.0]):
            why.append(f"{label}: the hard vote recomputed differs from the scorer's matrices")
        sl = soft_of(d, subset, "log", sids, labels, prior)
        sp = soft_of(d, subset, "prob", sids, labels, prior)
        r = {"hard_plain": miou(h["_cms"][0.0].sum(0), IGNORE[RD.TASK]) * 100, "hard_corrected": h["cross_fit"]["x"],
             "hard_alphas": h["chosen"],
             "soft_plain": miou(sl["plain"].sum(0), IGNORE[RD.TASK]) * 100,
             "soft_corrected": miou(sl["cf"].sum(0), IGNORE[RD.TASK]) * 100, "soft_alphas": sl["chosen"],
             "soft_by_alpha": sl["by_alpha"],
             "soft_prob_corrected": miou(sp["cf"].sum(0), IGNORE[RD.TASK]) * 100, "soft_prob_alphas": sp["chosen"],
             "primary_corrected_soft_minus_hard": S70.pair(sl["cf"], h["_cf"]),
             "plain_soft_minus_hard": S70.pair(sl["plain"], h["_cms"][0.0]),
             "prob_minus_log_corrected": S70.pair(sp["cf"], sl["cf"])}
        res["sets"][label] = r
        if label == PRIMARY:
            per_class = {"hard corrected": iou_per_class(h["_cf"].sum(0), IGNORE[RD.TASK]) * 100,
                         "soft corrected": iou_per_class(sl["cf"].sum(0), IGNORE[RD.TASK]) * 100}
    res["valid"], res["why"] = not why, why
    prim = res["sets"][PRIMARY]["primary_corrected_soft_minus_hard"]
    others = [res["sets"][k]["primary_corrected_soft_minus_hard"] for k in SETS if k != PRIMARY]
    res["verdict"] = verdict(prim, others) if not why else "not read"
    from eo.maps import palette as P
    names = P.TASKS[RD.TASK].class_names
    res["per_class_stack"] = [{"class": names[i], **{k: float(v[i]) for k, v in per_class.items()}}
                              for i in range(len(names)) if i != IGNORE[RD.TASK]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda p: f"{p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}]"       # noqa: E731
    L = ["# A2 Step 7.0h read: soft voting", "", f"- valid {res['valid']} {why}; argmax checks {res['argmax_checks']}", ""]
    L.append("| set | hard plain | soft plain | hard corrected | soft corrected | corrected soft - hard | plain soft - hard | prob - log (corr.) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, r in res["sets"].items():
        L.append(f"| {k} | {r['hard_plain']:.2f} | {r['soft_plain']:.2f} | {r['hard_corrected']:.2f} ({r['hard_alphas']}) | "
                 f"{r['soft_corrected']:.2f} ({r['soft_alphas']}) | {f(r['primary_corrected_soft_minus_hard'])} | "
                 f"{f(r['plain_soft_minus_hard'])} | {f(r['prob_minus_log_corrected'])} |")
    L += ["", f"- verdict: {res['verdict']}"]
    (out / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
