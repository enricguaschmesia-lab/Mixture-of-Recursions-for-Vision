#!/usr/bin/env python
"""A2 Step 4.0's follow-up 2 (Enric, 2026-11-22): the read of the fine-tuned Cosmos arm `cosmos16_224`, by the rule
pre-registered in docs/plans/a2_segmentation.md (Step 4.0, "4.0 follow-up 2") before any of its data. numpy and json.

Reads Step 4.0's aligned maps (/data/enric/reports/a2_seg/step4/gen/aligned/) and this run's
(.../step4/followup2/gen/cosmos16_224/), both scored by step4_score.py (score.json + per_patch_cm.npy), the run's
select_cosmos16_224.json and Z8 record, and the tokenizer's frozen record (its Step 1.2 ceilings).
  * Dc = m(cosmos16_224) - m(aligned) in K = 8 fold-4 mIoU, with the paired bootstrap: the SAME 1,000 resamples of the
    482 patches (seed 0) as Step 4.0's and the follow-ups' reads; each run's macro mIoU (void out) from its resampled
    per-patch matrices;
  * the fine-tuned Cosmos BEATS THE FROZEN PICK iff Dc > 0.010, its interval lies above 0, and its tokenizer passes
    Step 1.3's bar (ceilings >= 0.85 / 0.95 / 0.95 on PASTIS fold 4 / CloudSen12 val / LULC). Then, by Enric's
    instruction, R1 and Step 4.1 revert to it. Otherwise the pick stands; a tokenizer under the bar is described only.

    python eo/experiments/segmentation/step4c_read.py [--root /data/enric/reports/a2_seg/step4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4_read import load_scored, miou, CHANCE   # noqa: E402

RUN, TOK, FROZEN = "cosmos16_224", "cosmos16_224", "/data/enric/reports/a2_seg/step1/frozen/aligned_D_d224_cosmos16_lg"
BAR = {"pastis": 0.85, "cloudsen12": 0.95, "lulc": 0.95}
N_BOOT, SEED = 1000, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/enric/reports/a2_seg/step4")
    a = ap.parse_args()
    R = Path(a.root)
    F = R / "followup2"
    sel = json.loads((F / f"select_{RUN}.json").read_text())["arms"][RUN]
    al = load_scored(R / "gen" / "aligned" / "full_k8")
    g = F / "gen" / RUN
    s = {k: load_scored(g / p) for k, p in (("k8", "full_k8"), ("k1", "full_k8/k1"), ("k1b", "full_k8/k1b"),
                                            ("greedy", "full_greedy"), ("final_greedy", "final_greedy"),
                                            ("prior", "prior_k1"), ("wrong", "wrong_k1"))}
    ceil = load_scored(F / "ceiling" / f"ceiling_{TOK}")
    z8 = json.loads((g / "z8_bf16.json").read_text())
    rec = json.loads((Path(FROZEN) / "record.json").read_text())
    if not str(s["k8"]["checkpoint"]).endswith(sel["selected"]["checkpoint"]):
        raise RuntimeError(f"K = 8 maps from {s['k8']['checkpoint']}, the rule selected {sel['selected']['checkpoint']}")
    mw, mp, mf, mf2 = s["wrong"]["mIoU"], s["prior"]["mIoU"], s["k1"]["mIoU"], s["k1b"]["mIoU"]
    bar = {d: bool(rec["ceilings"][d] >= v) for d, v in BAR.items()}
    row = {"tokenizer": TOK, "frozen": FROZEN, "step1_ceilings": rec["ceilings"], "passes_bar": all(bar.values()),
           "bar": bar, "selected": sel["selected"], "extension_test": sel["extension_test"], "health": sel["health"],
           "mIoU": {"k8": s["k8"]["mIoU"], "k1": mf, "k1_sample1": mf2, "greedy": s["greedy"]["mIoU"],
                    "final_greedy": s["final_greedy"]["mIoU"], "prior_k1": mp, "wrong_k1": mw, "ceiling": ceil["mIoU"]},
           "pixel_acc_k8": s["k8"]["pixel_acc"], "slot_mass_k8": s["k8"].get("slot_mass"),
           "z8_bf16": {"pass": z8["pass"], "control_fires": z8["control_fires"],
                       "max_ulp_diff": z8["check"]["max_ulp_diff"],
                       "control_ulp": z8["control_positions_one_off"]["max_ulp_diff"]},
           "z9_functional": {"check_pass": bool(abs(mw - mp) < abs(mw - mf)),
                             "control_fails_check": bool(not (abs(mf2 - mp) < abs(mf2 - mf)))},
           "gen": {"precision": s["k8"].get("precision"), "gpu": s["k8"].get("gpu"), "n_views": s["k8"].get("n_views")}}
    n = len(al["cm"])
    if len(s["k8"]["cm"]) != n:
        raise RuntimeError("runs scored on different patch sets")
    idx = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    bc = np.array([miou(s["k8"]["cm"][i].sum(0)) for i in idx])
    ba = np.array([miou(al["cm"][i].sum(0)) for i in idx])
    d = bc - ba
    dc = {"point": s["k8"]["mIoU"] - al["mIoU"], "mean": float(d.mean()), "lo95": float(np.percentile(d, 2.5)),
          "hi95": float(np.percentile(d, 97.5))}
    beats = bool(dc["point"] > 0.010 and dc["lo95"] > 0 and row["passes_bar"])
    out = {"protocol": "A2 Step 4.0 follow-up 2, pre-registered 2026-11-22 (docs/plans/a2_segmentation.md)",
           "k8_mIoU": {"aligned": al["mIoU"], RUN: s["k8"]["mIoU"]}, "run": row,
           "bootstrap": {"resamples": N_BOOT, "seed": SEED, "n_patches": n},
           "Dc": dc, "beats_frozen_pick": beats,
           "verdict": ("revert R1 and Step 4.1 to the fine-tuned Cosmos" if beats else "the pick stands"),
           "checks": {"z8_pass": row["z8_bf16"]["pass"], "z8_control_fires": row["z8_bf16"]["control_fires"],
                      "z9_pass": row["z9_functional"]["check_pass"],
                      "z9_control_fails": row["z9_functional"]["control_fails_check"]},
           "chance": {k: json.loads(CHANCE.read_text())["references"]["pastis"][k] for k in ("uniform_mIoU", "constant_mIoU")}}
    (F / "read.json").write_text(json.dumps(out, indent=1) + "\n")
    m = row["mIoU"]
    fmt = lambda x: f"{x['point']:+.4f} [{x['lo95']:+.4f}, {x['hi95']:+.4f}]"
    L = ["| run | checkpoint (CE) | **K = 8** | K = 1 | greedy | greedy, final | ceiling | no input | wrong input | "
         "Z8 bf16 | Z9 | extension test |", "|" + "---|" * 12,
         f"| {RUN} | {sel['selected']['checkpoint']} ({sel['selected']['eval_ce']:.4f}) | **{m['k8']:.4f}** | {m['k1']:.4f} | "
         f"{m['greedy']:.4f} | {m['final_greedy']:.4f} | {m['ceiling']:.4f} | {m['prior_k1']:.4f} | {m['wrong_k1']:.4f} | "
         f"{row['z8_bf16']['max_ulp_diff']:.2f} ({'pass' if row['z8_bf16']['pass'] else 'FAIL'}) | "
         f"{'pass' if row['z9_functional']['check_pass'] else 'FAIL'} | "
         f"{100 * sel['extension_test']['relative_drop']:.2f}% ({'fires' if sel['extension_test']['fires'] else 'no'}) |",
         "", f"Tokenizer ceilings (Step 1.2): " + " / ".join(f"{rec['ceilings'][k]:.4f}" for k in BAR) +
         f" -> bar {'passed' if row['passes_bar'] else 'FAILED'}.",
         f"Dc = {RUN} - aligned = {fmt(dc)} (aligned K = 8 {al['mIoU']:.4f}) -> beats the frozen pick: {beats}: "
         f"**{out['verdict']}**."]
    (F / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps(out["checks"]))


if __name__ == "__main__":
    main()
