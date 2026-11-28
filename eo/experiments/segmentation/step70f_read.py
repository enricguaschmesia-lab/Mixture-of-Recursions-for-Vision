#!/usr/bin/env python
"""A2 Step 7.0f's read (docs/plans/a2_segmentation.md, Step 7, "7.0f, the stack: E784 with the 8 orientations and
corrected voting"; the rule pre-registered 2026-11-27 before any of its numbers). `.venv`, CPU; reads 7.0e's decoded
class maps (step70e_votes.py) and functions (step70e_read.py).

    python eo/experiments/segmentation/step70f_read.py --stack <votes name of S8x8> [--out <dir>]

  validity  the true c4375 tokens of every orientation, brought back, within 0.005 of k = 0's; S8x8's own K8 (plain)
            level with E784's original K8 maps (B1's 1.2 bar); E784's Z8 at bf16 (7.0e's record);
  (S1)      the stack (TTA-64 corrected, alpha cross-fitted) - S8x8's K8 corrected, paired on one model: gains iff the
            interval is above 0, costs iff below, else level;
  (S2)      the stack - B1's TTA-64 corrected, two single-seed models: B1's 1.2 bar, (a) gains, (b) neither, (c) loses;
  number    the stack's mIoU against 41.1, with its interval against B1 plain (deciding nothing);
  beside    TTA-64 plain, TTA-8, per class, the alphas.
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

OUT = Path("/data/enric/reports/a2_seg/step70f")
E784_RUN = "p7e784_s42_20261127_004451"
E784_GEN = RD.GEN / E784_RUN
B1_TTA = "p7etta_20261127_004451"
TRUTH = "tta_truth_c4375"
TARGET = 41.1


def band_s2(p: dict) -> str:
    c = K.call(p)
    return {"gains": "(a) gains: the 784-token map still adds on top of the rotation votes and the correction",
            "costs": "(c) loses: the stack trails B1's corrected rotation votes"}.get(
        c, "(b) neither: the stack is level with B1's corrected rotation votes")


def truth_ceilings(name: str) -> dict:
    tm, tv, _ = RD.load_maps(name)
    ks = np.asarray(tv["k"])
    out = {}
    for k in range(8):
        rows = np.where(ks == k)[0]
        lab = RD.labels_for([tv["sample_id"][r] for r in rows])
        back = np.stack([VT.to_k0(np.asarray(tm[r, 0]), k) for r in rows])
        out[k] = miou(per_patch_cm(back, lab, RD.N_CLASSES, IGNORE[RD.TASK]).sum(0), IGNORE[RD.TASK])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, help="S8x8's votes name under step70e/votes/")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    prior = json.loads(RD.PRIOR.read_text())
    res = {"rule": "A2 plan, Step 7, 7.0f (pre-registered 2026-11-27)", "stack": a.stack, "target": TARGET}
    maps, vj, meta = RD.load_maps(a.stack)
    if meta["arm"] != "c4375_224":
        raise RuntimeError(f"{a.stack} is arm {meta['arm']}, not E784's c4375_224")
    st = {w: RD.tta_stacks(maps, vj, w) for w in ("tta64", "k8", "tta8")}
    sids = st["k8"][1]["sample_id"]
    labels = RD.labels_for(sids)
    plain = {w: RD.cms_of(st[w][0], len(sids), labels) for w in st}
    stack = RD.test2_of(None, prior, stacks=st["tta64"])
    k8c = RD.test2_of(None, prior, stacks=st["k8"])
    # the reference cross-fits: B1's TTA-64 corrected (7.0e's T8x8) and E784's own K8 maps
    b1m, b1v, _ = RD.load_maps(B1_TTA)
    b1fn, b1ids = RD.tta_stacks(b1m, b1v, "tta64")
    n = len(stack["_sids"])
    if [str(s) for s in b1ids["sample_id"][:n]] != [str(s) for s in stack["_sids"]]:
        raise RuntimeError("S8x8's patches are not B1's T8x8's first patches, in order")
    b1t = RD.test2_of(None, prior, stacks=(b1fn, {"sample_id": list(stack["_sids"])}))   # a smoke reads a prefix
    e784_cm = np.load(E784_GEN / "full_k8" / "per_patch_cm.npy")
    b1_plain = np.load(K.gen_dir(K.B1_RUN) / "full_k8" / "per_patch_cm.npy")
    # validity
    why = []
    ceil = truth_ceilings(TRUTH)
    if any(abs(v - ceil[0]) > 0.005 for v in ceil.values()):
        why.append(f"the true c4375 tokens by orientation {ceil} are not all within 0.005 of k = 0's")
    k8_vs = S70.pair(plain["k8"], e784_cm) if len(plain["k8"]) == len(e784_cm) else None
    if k8_vs is None:
        why.append(f"S8x8 covers {len(sids)} patches, not 482 (a smoke)")
    elif K.call(k8_vs) != "level":
        why.append(f"S8x8's K8 is not level with E784's own K8: {k8_vs['delta']:+.2f}")
    z8 = json.loads((E784_GEN / "z8_bf16.json").read_text())
    if not z8.get("pass"):
        why.append("E784's Z8 fails at bf16")
    res["validity"] = {"valid": not why, "why": why, "truth_ceiling_by_k": ceil, "k8_vs_e784_k8": k8_vs,
                       "e784_z8_bf16": z8.get("pass")}
    s1 = S70.pair(stack["_cf"], k8c["_cf"])
    s2 = S70.pair(stack["_cf"], b1t["_cf"])
    vs_b1 = S70.pair(stack["_cf"], b1_plain) if len(stack["_cf"]) == len(b1_plain) else None
    res["mIoU"] = {"stack_tta64_corrected": stack["cross_fit"]["x"], "tta64_plain": miou(plain["tta64"].sum(0), IGNORE[RD.TASK]) * 100,
                   "tta8_plain": miou(plain["tta8"].sum(0), IGNORE[RD.TASK]) * 100,
                   "k8_plain": miou(plain["k8"].sum(0), IGNORE[RD.TASK]) * 100, "k8_corrected": k8c["cross_fit"]["x"],
                   "b1_tta64_corrected": b1t["cross_fit"]["x"], "b1_plain": miou(b1_plain.sum(0), IGNORE[RD.TASK]) * 100}
    res["alphas"] = {"stack": stack["chosen"], "k8": k8c["chosen"], "b1_tta64": b1t["chosen"],
                     "stack_by_alpha": stack["by_alpha"]}
    res["S1_stack_minus_k8_corrected"] = s1
    res["S2_stack_minus_b1_tta64_corrected"] = s2
    res["stack_minus_b1_plain"] = vs_b1
    res["verdict_S1"] = RD.decode_call(s1) if not why else "not read"
    res["verdict_S2"] = band_s2(s2) if not why else "not read"
    res["against_target"] = {"stack": stack["cross_fit"]["x"], "target": TARGET, "short_by": TARGET - stack["cross_fit"]["x"]}
    from eo.maps import palette as P
    names = P.TASKS[RD.TASK].class_names
    pc = {"B1 plain": iou_per_class(b1_plain.sum(0), IGNORE[RD.TASK]) * 100,
          "E784 K8 plain": iou_per_class(e784_cm.sum(0), IGNORE[RD.TASK]) * 100,
          "B1 TTA-64 corrected": iou_per_class(b1t["_cf"].sum(0), IGNORE[RD.TASK]) * 100,
          "stack": iou_per_class(stack["_cf"].sum(0), IGNORE[RD.TASK]) * 100}
    res["per_class"] = [{"class": names[i], **{k: float(v[i]) for k, v in pc.items()}} for i in range(len(names))
                        if i != IGNORE[RD.TASK]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda p: f"{p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}]" if p and "delta" in p else str(p)   # noqa: E731
    m = res["mIoU"]
    L = ["# A2 Step 7.0f read: the stack", "",
         f"- validity {res['validity']['valid']} {why}; c4375 truth by k {({k: round(v, 4) for k, v in ceil.items()})}",
         f"- mIoU: stack (TTA-64 corrected) {m['stack_tta64_corrected']:.2f}; TTA-64 plain {m['tta64_plain']:.2f}; TTA-8 "
         f"{m['tta8_plain']:.2f}; K8 plain {m['k8_plain']:.2f}, corrected {m['k8_corrected']:.2f}; B1 TTA-64 corrected "
         f"{m['b1_tta64_corrected']:.2f}; B1 plain {m['b1_plain']:.2f}",
         f"- alphas: {res['alphas']['stack']} (stack), {res['alphas']['k8']} (K8), {res['alphas']['b1_tta64']} (B1 TTA-64)",
         f"- (S1) stack - K8 corrected {f(s1)} -> {res['verdict_S1']}",
         f"- (S2) stack - B1 TTA-64 corrected {f(s2)} -> {res['verdict_S2']}",
         f"- stack - B1 plain {f(vs_b1)}; against the target {TARGET}: short by {res['against_target']['short_by']:.2f}"]
    (out / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
