#!/usr/bin/env python
"""A2 Step 7.0e's read (docs/plans/a2_segmentation.md, Step 7, "7.0e, three generation-side tests on PASTIS-7"; the
rules pre-registered 2026-11-26 before any of their numbers). `.venv`, CPU; reads the decoded class maps
(step70e_votes.py), the scored maps (step52_score.py) and the pulled records.

    python eo/experiments/segmentation/step70e_read.py [--e784 <run>] [--tta <gen name>] [--k64 <gen name>] [--out <dir>]

  test 1  E784 (B1's recipe, the 784-token nested map): 7.0's validity, B1's base, Delta(E784 - B1) by B1's 1.2 bar:
          (a) gains, (b) neither, (c) loses;
  test 2  frequency-corrected voting: argmax_c f_c^-alpha V_c with f the fixed class prior (class_prior.json), alpha
          cross-fitted over two halves of fold 4 (sorted patch id, alternating; the best of ALPHAS on one half is
          applied to the other); gains iff on B1 the cross-fitted Delta's interval is above 0 and its Delta is above 0
          on 7.0 and on G; costs on the mirror; else level;
  test 3  votes over the 8 dihedral variants (class maps brought back to k = 0 first): (A) TTA-64 - K8, (B) TTA-64 -
          K64, (C) TTA-8 - K8; (A) gains iff its interval is above 0, costs on the mirror, else level; (B) by the same
          rule; validity: the true tokens of every variant score within 0.005 of the k = 0 ceiling, T8x8's K8 level
          with B1's own K8 maps (B1's 1.2 bar);
  beside  alpha fixed at 0.5; test 2's rule on E784, TTA-64 and K64; per class.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step70_read as S70                           # noqa: E402
import step70k_read as K                            # noqa: E402
import step70e_votes as VT                          # noqa: E402
from step4_score import per_patch_cm                # noqa: E402
from step52_read import IGNORE, RUNS                # noqa: E402
from step53_verdict import iou_per_class, miou      # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70e")
GEN, VOTES = OUT / "gen", OUT / "votes"
PRIOR = OUT / "class_prior.json"
B1_RUN, R70_RUN, G_RUN = K.B1_RUN, K.R70_RUN, "p7d_s42_20261126_192845"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CEILING_K0, CEILING_TOL = 0.9143, 0.005
TASK, N_CLASSES = "pastis", 20


# ---------------------------------------------------------------- the rules (pure; the gate tests them)
def halves(sample_ids: List[str]) -> np.ndarray:
    """0 / 1 per patch: sorted patch id, alternating (the cross-fitting split, fixed in the plan)."""
    order = np.argsort(np.asarray([int(s) for s in sample_ids]), kind="stable")
    h = np.zeros(len(sample_ids), dtype=np.int64)
    h[order[1::2]] = 1
    return h


def best_alpha(cms_by_alpha: Dict[float, np.ndarray], mask: np.ndarray) -> float:
    """The alpha whose summed matrices over the masked patches read the highest mIoU (ties to the smaller alpha)."""
    best, val = None, -1.0
    for a in sorted(cms_by_alpha):
        m = miou(cms_by_alpha[a][mask].sum(0), IGNORE[TASK])
        if m > val + 1e-15:
            best, val = a, m
    return best


def cross_fit(cms_by_alpha: Dict[float, np.ndarray], h: np.ndarray) -> tuple:
    """(per-patch matrices, each patch at the alpha its OTHER half chose; {half: alpha})."""
    a0, a1 = best_alpha(cms_by_alpha, h == 0), best_alpha(cms_by_alpha, h == 1)
    out = np.where((h == 0)[:, None, None], cms_by_alpha[a1], cms_by_alpha[a0])
    return out, {0: a0, 1: a1}


def decode_call(p: Optional[dict]) -> str:
    """Tests 2 and 3: gains iff the paired interval is above 0, costs iff below 0, else level."""
    if p is None or "delta" not in p:
        return "hole"
    return "gains" if p["lo95"] > 0 else "costs" if p["hi95"] < 0 else "level"


def test2_verdict(b1: Optional[dict], others: List[Optional[dict]]) -> str:
    c = decode_call(b1)
    if c == "hole" or any(o is None or "delta" not in o for o in others):
        return "hole"
    if c == "gains" and all(o["delta"] > 0 for o in others):
        return "gains"
    if c == "costs" and all(o["delta"] < 0 for o in others):
        return "costs"
    return "level"


def band_e784(p: dict) -> str:
    c = K.call(p)
    return {"gains": "(a) gains: the 784-token map beats B1's 196 on PASTIS-7",
            "costs": "(c) loses: the 784-token map costs on PASTIS-7"}.get(
        c, "(b) neither: the 784-token map is level with B1's 196 on PASTIS-7")


# ---------------------------------------------------------------- the reads
def labels_for(sample_ids: List[str]) -> np.ndarray:
    from eo.maps import sets
    lab, ids = sets.load(TASK, "val")
    where = {int(i): n for n, i in enumerate(ids)}
    return np.asarray(lab[[where[int(s)] for s in sample_ids]])


def load_maps(name: str):
    d = VOTES / name
    meta = json.loads((d / "decode.json").read_text())
    return np.load(d / "classmaps.npy", mmap_mode="r"), json.loads((d / "views.json").read_text()), meta


def cms_of(stacks_fn, n: int, labels: np.ndarray, w=None) -> np.ndarray:
    """Per-patch matrices of the vote over stacks_fn(i) ((M, H, W) uint8) for patches i < n."""
    import torch
    preds = np.empty((n,) + labels.shape[1:], dtype=np.uint8)
    for i in range(n):
        st = torch.as_tensor(np.ascontiguousarray(stacks_fn(i)))[:, None]          # (M, 1, H, W)
        preds[i] = VT.weighted_vote(st, N_CLASSES, w)[0].numpy().astype(np.uint8)
    return per_patch_cm(preds, labels, N_CLASSES, IGNORE[TASK])


def test2_of(name: str, prior: dict, stacks=None) -> dict:
    """Plain, the cross-fitted rule and alpha 0.5 on one K-sample run (or on given stacks for test 3's votes)."""
    maps, vj, _ = load_maps(name) if stacks is None else (None, stacks[1], None)
    sids = vj["sample_id"] if stacks is None else stacks[1]["sample_id"]
    n = len(sids)
    labels = labels_for(sids)
    fn = (lambda i: maps[i]) if stacks is None else stacks[0]
    cms = {a: cms_of(fn, n, labels, VT.class_weights(prior, a, N_CLASSES, IGNORE[TASK])) for a in ALPHAS}
    h = halves(sids)
    cf, chosen = cross_fit(cms, h)
    return {"plain_mIoU": miou(cms[0.0].sum(0), IGNORE[TASK]) * 100,
            "by_alpha": {str(a): miou(c.sum(0), IGNORE[TASK]) * 100 for a, c in cms.items()},
            "chosen": {str(k): v for k, v in chosen.items()}, "cross_fit": S70.pair(cf, cms[0.0]),
            "alpha_0.5": S70.pair(cms[0.5], cms[0.0]), "_cms": cms, "_cf": cf, "_labels": labels, "_sids": sids}


def stored_plain_equal(name: str, gen_dir: Path, t2: dict) -> bool:
    """Validity: the plain vote of the decoded maps equals the scorer's own per-patch matrices."""
    f = gen_dir / "per_patch_cm.npy"
    return f.exists() and np.array_equal(np.load(f), t2["_cms"][0.0])


def tta_stacks(maps, vj, which: str):
    """Per-patch stacks from T8x8's maps (patch-major, 8 variants a patch): all 64, k = 0's 8, or sample 0 of each."""
    ks = np.asarray(vj["k"])
    n = len(ks) // 8
    if not (ks.reshape(n, 8) == np.arange(8)).all():
        raise RuntimeError("T8x8's views are not patch-major k = 0..7")

    def fn(i):
        rows = [VT.to_k0(np.asarray(maps[8 * i + k]), k) for k in range(8)]          # each (K, H, W)
        if which == "k8":
            return rows[0]
        if which == "tta8":
            return np.stack([r[0] for r in rows])
        return np.concatenate(rows)
    sids = [vj["sample_id"][8 * i] for i in range(n)]
    return fn, {"sample_id": sids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e784", default=None)
    ap.add_argument("--tta", default=None, help="T8x8's votes name under votes/")
    ap.add_argument("--k64", default=None, help="K64's votes name under votes/")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    prior = json.loads(PRIOR.read_text())
    res = {"rule": "A2 plan, Step 7, 7.0e (pre-registered 2026-11-26)", "alphas": ALPHAS}
    b1 = K.read_one(B1_RUN)
    # ---- test 1
    t1 = {"run": a.e784}
    if a.e784:
        try:
            R = S70.read_run(a.e784, GEN)
            prec, why = K.validity(R)
            why += K.lineage(a.e784, None, K.B1_BASE_SHA)
            t1.update({"valid": not why, "why": why, "precision": prec})
            if not why:
                p = "bf16" if (prec == "bf16" and b1["precision"] == "bf16") else "fp32"
                d = GEN / (a.e784 if p == "bf16" else a.e784 + "_fp32")
                pr = S70.pair(S70.cm_of(d / "full_k8"), K.cm(B1_RUN, p))
                t1.update({"pair": dict(pr, precision=p), "band": band_e784(pr), "selected": R["selected"]})
        except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError) as e:
            t1.update({"valid": False, "why": [f"records: {type(e).__name__}: {e}"]})
    else:
        t1["band"] = "hole: E784 not given"
    res["test1"] = t1
    # ---- test 2
    t2 = {}
    for role, name, gen in (("B1", "b1_k8", K.gen_dir(B1_RUN) / "full_k8"), ("7.0", "p70_k8", S70.GEN / R70_RUN / "full_k8"),
                            ("G", "g_k8", Path("/data/enric/reports/a2_seg/step70d/gen") / G_RUN / "full_k8")):
        if not (VOTES / name / "decode.json").exists():
            t2[role] = None
            continue
        r = test2_of(name, prior)
        r["plain_equals_scorer"] = stored_plain_equal(name, gen, r)
        t2[role] = r
    ok = all(t2.get(r) and t2[r]["plain_equals_scorer"] for r in ("B1", "7.0", "G"))
    res["test2"] = {r: ({k: v for k, v in x.items() if not k.startswith("_")} if x else None) for r, x in t2.items()}
    res["test2"]["verdict"] = (test2_verdict(t2["B1"]["cross_fit"], [t2["7.0"]["cross_fit"], t2["G"]["cross_fit"]])
                               if ok else "not read: a decode is missing or its plain vote differs from the scorer's")
    # ---- test 3
    t3 = {}
    if a.tta and (VOTES / a.tta / "decode.json").exists():
        maps, vj, _ = load_maps(a.tta)
        st = {w: tta_stacks(maps, vj, w) for w in ("tta64", "k8", "tta8")}
        sids = st["k8"][1]["sample_id"]
        labels = labels_for(sids)
        cm = {w: cms_of(st[w][0], len(sids), labels) for w in st}
        if a.k64 and (VOTES / a.k64 / "decode.json").exists():
            m64, v64, _ = load_maps(a.k64)
            if [str(s) for s in v64["sample_id"]] != [str(s) for s in sids]:
                raise RuntimeError("K64's patches are not T8x8's, in order")
            cm["k64"] = cms_of(lambda i: m64[i], len(sids), labels)
        t3["mIoU"] = {w: miou(c.sum(0), IGNORE[TASK]) * 100 for w, c in cm.items()}
        t3["A_tta64_minus_k8"] = S70.pair(cm["tta64"], cm["k8"])
        t3["B_tta64_minus_k64"] = S70.pair(cm["tta64"], cm["k64"]) if "k64" in cm else None
        t3["C_tta8_minus_k8"] = S70.pair(cm["tta8"], cm["k8"])
        t3["call_A"], t3["call_B"] = decode_call(t3["A_tta64_minus_k8"]), decode_call(t3["B_tta64_minus_k64"])
        b1cm = np.load(K.gen_dir(B1_RUN) / "full_k8" / "per_patch_cm.npy")
        t3["k8_vs_b1_k8"] = S70.pair(cm["k8"], b1cm) if len(cm["k8"]) == len(b1cm) else None
        truth = VOTES / "tta_truth"
        if (truth / "decode.json").exists():
            tm, tv, _ = load_maps("tta_truth")
            ks = np.asarray(tv["k"])
            ceil = {}
            for k in range(8):
                rows = np.where(ks == k)[0]
                ok_ids = [tv["sample_id"][r] for r in rows]
                lab = labels_for(ok_ids)
                pred = np.stack([VT.to_k0(np.asarray(tm[r, 0]), k) for r in rows])
                ceil[k] = miou(per_patch_cm(pred, lab, N_CLASSES, IGNORE[TASK]).sum(0), IGNORE[TASK])
            t3["truth_ceiling_by_k"] = ceil
        else:
            t3["truth_ceiling_by_k"] = None
        why = []
        if not t3["truth_ceiling_by_k"] or any(abs(v - CEILING_K0) > CEILING_TOL for v in t3["truth_ceiling_by_k"].values()):
            why.append(f"the true tokens' ceilings by variant {t3['truth_ceiling_by_k']} are not all within "
                       f"{CEILING_TOL} of {CEILING_K0}")
        if t3["k8_vs_b1_k8"] is None:
            why.append(f"T8x8 covers {len(sids)} patches, not B1's 482 (a smoke)")
        elif K.call(t3["k8_vs_b1_k8"]) != "level":
            why.append(f"T8x8's K8 is not level with B1's own K8: {t3['k8_vs_b1_k8']['delta']:+.2f}")
        t3["valid"], t3["why"] = not why, why
        t3["verdict_A"] = t3["call_A"] if not why else "not read"
        t3["verdict_B"] = t3["call_B"] if not why else "not read"
        tf = test2_of(None, prior, stacks=st["tta64"])
        t3["tta64_freq"] = {k: v for k, v in tf.items() if not k.startswith("_")}
    else:
        t3["verdict_A"] = t3["verdict_B"] = "hole: T8x8 not decoded"
    res["test3"] = t3
    # ---- beside: per class
    pc = {}
    if t2.get("B1"):
        pc["B1"] = iou_per_class(t2["B1"]["_cms"][0.0].sum(0), IGNORE[TASK]) * 100
        pc["B1 freq (cross-fit)"] = iou_per_class(t2["B1"]["_cf"].sum(0), IGNORE[TASK]) * 100
    if t3.get("mIoU"):
        pc["TTA-64"] = iou_per_class(cm["tta64"].sum(0), IGNORE[TASK]) * 100
    if t1.get("pair"):
        d = GEN / (a.e784 if t1["pair"]["precision"] == "bf16" else a.e784 + "_fp32")
        pc["E784"] = iou_per_class(S70.cm_of(d / "full_k8").sum(0), IGNORE[TASK]) * 100
    from eo.maps import palette as P
    names = P.TASKS[TASK].class_names
    res["per_class"] = [{"class": names[i], **{k: float(v[i]) for k, v in pc.items()}} for i in range(len(names))
                        if pc and i != IGNORE[TASK]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda p: f"{p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}]" if p and "delta" in p else str(p)   # noqa: E731
    L = ["# A2 Step 7.0e read", "", "## Test 1: the 784-token map"]
    L.append(f"- E784 {a.e784}: valid {t1.get('valid')} {t1.get('why') or ''}")
    if t1.get("pair"):
        L.append(f"- E784 {t1['pair']['x']:.2f} vs B1 {t1['pair']['y']:.2f}: {f(t1['pair'])} -> {t1['band']}")
    L += ["", "## Test 2: frequency-corrected voting"]
    for r in ("B1", "7.0", "G"):
        x = res["test2"].get(r)
        if x:
            L.append(f"- {r}: plain {x['plain_mIoU']:.2f}; cross-fit {f(x['cross_fit'])} (alphas {x['chosen']}); "
                     f"alpha 0.5 {f(x['alpha_0.5'])}; plain = scorer: {x['plain_equals_scorer']}")
    L.append(f"- verdict: {res['test2']['verdict']}")
    L += ["", "## Test 3: votes over the 8 flips and rotations"]
    if t3.get("mIoU"):
        L.append(f"- mIoU: {', '.join(f'{k} {v:.2f}' for k, v in t3['mIoU'].items())}")
        L.append(f"- (A) TTA-64 - K8 {f(t3['A_tta64_minus_k8'])} -> {t3['verdict_A']}")
        L.append(f"- (B) TTA-64 - K64 {f(t3['B_tta64_minus_k64'])} -> {t3['verdict_B']}")
        L.append(f"- (C) TTA-8 - K8 {f(t3['C_tta8_minus_k8'])}")
        L.append(f"- validity {t3['valid']} {t3['why']}; true-token ceilings {t3['truth_ceiling_by_k']}")
        L.append(f"- beside: frequency-corrected TTA-64 {f(t3['tta64_freq']['cross_fit'])}")
    else:
        L.append(f"- {t3.get('verdict_A')}")
    (out / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
