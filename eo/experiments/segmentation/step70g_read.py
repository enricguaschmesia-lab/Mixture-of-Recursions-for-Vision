#!/usr/bin/env python
"""A2 Step 7.0g's read (docs/plans/a2_segmentation.md, Step 7, "7.0g, E784's second seed with the full stack"; the rule
pre-registered 2026-11-27 before any of its numbers). `.venv`, CPU; 7.0e's and 7.0f's functions and decoded maps.

    python eo/experiments/segmentation/step70g_read.py --cell <E784-s43 run> --stack <S8x8-s43 votes name> [--out <dir>]

  validity   7.0e's items for E784-s43 (training, the plateau record, Z8 by the precision rule, Z9, B1's base); 7.0f's
             for S8x8-s43 (its own K8 plain level with E784-s43's K8 maps; the c4375 true tokens are 7.0f's, checked);
  replicates the 784 map replicates iff Delta(E784-s43 K8 - B1) gains by B1's 1.2 bar;
  holds      the stack holds iff Delta(stack-s43 - stack-s42) does not cost by B1's 1.2 bar;
  reported   the two-seed mean of the stack against 41.1 (and at alpha fixed at 0.5); seed 43's (S1) and (S2); per class.
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
import step70f_read as F                            # noqa: E402
import step70k_read as K                            # noqa: E402
from step52_read import IGNORE                      # noqa: E402
from step53_verdict import iou_per_class, miou      # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70g")
STACK42 = "p7fstack_20261127_080904"
TARGET = 41.1


def replicates(p: dict) -> str:
    return "replicates" if K.call(p) == "gains" else "does not replicate"


def holds(p: dict) -> str:
    return "does not hold: seed 43's stack costs against seed 42's" if K.call(p) == "costs" else "holds"


def stack_of(name: str, prior: dict) -> dict:
    """7.0f's quantities for one S8x8 run: TTA-64 corrected (cross-fitted), K8 plain and corrected, TTA-64 plain."""
    maps, vj, meta = RD.load_maps(name)
    if meta["arm"] != "c4375_224":
        raise RuntimeError(f"{name} is arm {meta['arm']}, not E784's c4375_224")
    st = {w: RD.tta_stacks(maps, vj, w) for w in ("tta64", "k8")}
    sids = st["k8"][1]["sample_id"]
    labels = RD.labels_for(sids)
    k8_plain = RD.cms_of(st["k8"][0], len(sids), labels)
    return {"sids": sids, "stack": RD.test2_of(None, prior, stacks=st["tta64"]),
            "k8c": RD.test2_of(None, prior, stacks=st["k8"]), "k8_plain": k8_plain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="E784-s43's run id")
    ap.add_argument("--stack", required=True, help="S8x8-s43's votes name under step70e/votes/")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    prior = json.loads(RD.PRIOR.read_text())
    res = {"rule": "A2 plan, Step 7, 7.0g (pre-registered 2026-11-27)", "cell": a.cell, "stack": a.stack, "target": TARGET}
    why = []
    b1_plain = np.load(K.gen_dir(K.B1_RUN) / "full_k8" / "per_patch_cm.npy")
    # E784-s43
    try:
        R = S70.read_run(a.cell, RD.GEN)
        prec, w = K.validity(R)
        w += K.lineage(a.cell, None, K.B1_BASE_SHA)
        why += [f"E784-s43: {x}" for x in w]
        e43 = S70.cm_of(RD.GEN / (a.cell if prec == "bf16" else a.cell + "_fp32") / "full_k8")
        rep = S70.pair(e43, b1_plain if prec == "bf16" else K.cm(K.B1_RUN, "fp32"))
        res["e784_s43"] = {"precision": prec, "selected": R["selected"], "plateau": R.get("plateau_stop"),
                           "k8_minus_b1": rep, "call": replicates(rep)}
    except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError) as e:
        why.append(f"E784-s43 records: {type(e).__name__}: {e}")
        e43 = None
    # the two stacks
    s43, s42 = stack_of(a.stack, prior), stack_of(STACK42, prior)
    if [str(x) for x in s43["sids"]] != [str(x) for x in s42["sids"]]:
        raise RuntimeError("the two stacks do not hold the same patches in order")
    if e43 is not None:
        own = S70.pair(s43["k8_plain"], e43) if len(e43) == len(s43["k8_plain"]) else None
        res["s8x8_s43_k8_vs_e784_s43_k8"] = own
        if own is None:
            why.append(f"S8x8-s43 covers {len(s43['sids'])} patches, not 482 (a smoke)")
        elif K.call(own) != "level":
            why.append(f"S8x8-s43's K8 is not level with E784-s43's own K8: {own['delta']:+.2f}")
    res["valid"], res["why"] = not why, why
    hp = S70.pair(s43["stack"]["_cf"], s42["stack"]["_cf"])
    m43, m42 = s43["stack"]["cross_fit"]["x"], s42["stack"]["cross_fit"]["x"]
    f05 = {s: float(x["stack"]["by_alpha"]["0.5"]) for s, x in (("s42", s42), ("s43", s43))}
    b1m, b1v, _ = RD.load_maps(F.B1_TTA)
    b1t = RD.test2_of(None, prior, stacks=RD.tta_stacks(b1m, b1v, "tta64"))
    res["stack"] = {"s42": m42, "s43": m43, "mean": (m42 + m43) / 2, "s43_minus_s42": hp,
                    "alphas": {"s42": s42["stack"]["chosen"], "s43": s43["stack"]["chosen"]},
                    "alpha_0.5": f05, "alpha_0.5_mean": (f05["s42"] + f05["s43"]) / 2,
                    "s43_minus_b1_plain": S70.pair(s43["stack"]["_cf"], b1_plain) if len(b1_plain) == len(s43["sids"]) else None,
                    "S1_s43": S70.pair(s43["stack"]["_cf"], s43["k8c"]["_cf"]),
                    "S2_s43": S70.pair(s43["stack"]["_cf"], b1t["_cf"])}
    res["verdict_replicates"] = res.get("e784_s43", {}).get("call", "not read") if not why else "not read"
    res["verdict_holds"] = holds(hp) if not why else "not read"
    res["against_target"] = {"mean": res["stack"]["mean"], "target": TARGET, "short_by": TARGET - res["stack"]["mean"]}
    from eo.maps import palette as P
    names = P.TASKS[RD.TASK].class_names
    pc = {"stack s42": iou_per_class(s42["stack"]["_cf"].sum(0), IGNORE[RD.TASK]) * 100,
          "stack s43": iou_per_class(s43["stack"]["_cf"].sum(0), IGNORE[RD.TASK]) * 100}
    res["per_class"] = [{"class": names[i], **{k: float(v[i]) for k, v in pc.items()}} for i in range(len(names))
                        if i != IGNORE[RD.TASK]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda p: f"{p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}]" if p and "delta" in p else str(p)   # noqa: E731
    e = res.get("e784_s43", {})
    L = ["# A2 Step 7.0g read: E784's second seed with the full stack", "", f"- valid {res['valid']} {why}"]
    if e:
        L.append(f"- E784-s43 K8 {e['k8_minus_b1']['x']:.2f} vs B1 {e['k8_minus_b1']['y']:.2f}: {f(e['k8_minus_b1'])} -> "
                 f"{res['verdict_replicates']} (seed 42: +6.85)")
    L += [f"- stack: s42 {m42:.2f}, s43 {m43:.2f}, mean {res['stack']['mean']:.2f}; s43 - s42 {f(hp)} -> {res['verdict_holds']}",
          f"- alpha 0.5 fixed: s42 {f05['s42']:.2f}, s43 {f05['s43']:.2f}, mean {res['stack']['alpha_0.5_mean']:.2f}; "
          f"alphas chosen {res['stack']['alphas']}",
          f"- seed 43: (S1) {f(res['stack']['S1_s43'])}; (S2) {f(res['stack']['S2_s43'])}; vs B1 plain "
          f"{f(res['stack']['s43_minus_b1_plain'])}",
          f"- the two-seed mean against {TARGET}: short by {res['against_target']['short_by']:.2f}"]
    (out / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
