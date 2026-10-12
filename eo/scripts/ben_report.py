#!/usr/bin/env python
"""BEN results table (Phase 6 Step 5, D6.7). `.venv`, CPU.

    python eo/scripts/ben_report.py --models b10_full b10_lora --out /data/enric/reports/ben/report_b10

For each model tag (outputs of eval_ben.py under /data/enric/reports/ben/<tag>/<ckpt>/<role>/):
  1. SELECT the checkpoint on ft_valsel: the best mean micro-mAP over S1 / S2 / S1+S2. Never
     on test.
  2. Choose the marginal threshold for F1 on ft_valsel (micro-F1-optimal), then apply it on test.
  3. Report on each test role present (test_a = option (a), test_c = option (c)), per condition:
     greedy-set F1 (micro / macro / sample), F1 at the val threshold, micro / macro mAP,
     the first-step slot mass, and the CLASS-PRIOR predictor beside it (constant scores =
     ft_train frequencies; its F1 threshold also chosen on val).
  4. The wrong-input control where it was run: the share of the gain over the prior that
     survives when the inputs come from another patch.
  5. Paired bootstrap (over patches) of the micro-mAP difference between the first two models.

Every number names its role, condition, checkpoint and n.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.ben import labels as L  # noqa: E402
from eo.eval import ben as E  # noqa: E402

ROOT = Path("/data/enric/reports/ben")
CONDS = ("S1", "S2", "S1S2")
COND_NAME = {"S1": "S1 → BEN", "S2": "S2 → BEN", "S1S2": "S1+S2 → BEN"}


def load(tag, ck, role, cond, wrong=False):
    f = ROOT / tag / ck / role / f"{cond}{'_wrong' if wrong else ''}.npz"
    return dict(np.load(f)) if f.exists() else None


def train_labels():
    from eo.data.ben_dataset import load_index
    idx = load_index()
    return np.stack([np.isin(np.arange(L.N_CLASSES), L.from_mask(int(m))) for m in idx.labels19[idx.ft_train]])


def boot_diff(y, s1, s2, n=200, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y)
    d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        d.append(E.map_scores(y[i], s1[i])["micro_map"] - E.map_scores(y[i], s2[i])["micro_map"])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tests", nargs="+", default=["test_a", "test_c"])
    ap.add_argument("--root", type=Path, default=ROOT, help="eval_ben.py's --out-root")
    a = ap.parse_args()
    ROOT = a.root
    a.out.mkdir(parents=True, exist_ok=True)
    tr = train_labels()
    doc = {"models": {}, "train_prevalence": tr.mean(0).round(4).tolist()}
    lines = ["# BigEarthNet: B10 fine-tunes\n"]

    sel = {}
    for tag in a.models:
        cks = sorted({p.parent.parent.name for p in (ROOT / tag).glob("*/ft_valsel/S1S2.npz")})
        val = {}
        for ck in cks:
            ms = [E.map_scores(d["truth"], d["marginal"])["micro_map"]
                  for d in (load(tag, ck, "ft_valsel", c) for c in CONDS) if d is not None]
            if len(ms) == len(CONDS):
                val[ck] = float(np.mean(ms))
        if not val:
            print(f"{tag}: no complete ft_valsel evaluation")
            continue
        best = max(val, key=val.get)
        sel[tag] = best
        doc["models"][tag] = {"val_micro_map_by_checkpoint": val, "selected": best}
        lines.append(f"**{tag}**: val micro-mAP (mean of 3 conditions) by checkpoint: "
                     + ", ".join(f"{k} {v:.4f}" for k, v in val.items()) + f" → selected **{best}**\n")

    for tag, ck in sel.items():
        thr, thr_prior = {}, {}
        for c in CONDS:
            v = load(tag, ck, "ft_valsel", c)
            thr[c] = E.best_threshold(v["truth"], v["marginal"])
            thr_prior[c] = E.best_threshold(v["truth"], E.prior_scores(tr, len(v["truth"])))
        doc["models"][tag]["thresholds_from_val"] = thr
        for role in a.tests:
            rows = []
            for c in CONDS:
                d = load(tag, ck, role, c)
                if d is None:
                    continue
                y = d["truth"].astype(bool)
                pri = E.prior_scores(tr, len(y))
                g = E.f1_scores(y, d["greedy"])
                t = E.f1_scores(y, d["marginal"] >= thr[c])
                m = E.map_scores(y, d["marginal"])
                mp = E.map_scores(y, pri)
                fp = E.f1_scores(y, pri >= thr_prior[c])
                r = {"condition": c, "n": int(len(y)),
                     "greedy_micro_f1": g["micro_f1"], "greedy_macro_f1": g["macro_f1"], "greedy_sample_f1": g["sample_f1"],
                     "thr_micro_f1": t["micro_f1"], "thr_macro_f1": t["macro_f1"],
                     "micro_map": m["micro_map"], "macro_map": m["macro_map"],
                     "prior_micro_map": mp["micro_map"], "prior_macro_map": mp["macro_map"],
                     "prior_micro_f1": fp["micro_f1"], "prior_macro_f1": fp["macro_f1"],
                     "slot_mass": float(d["slot_mass"].mean()),
                     "per_class_ap": m["per_class_ap"], "per_class_f1_greedy": g["per_class_f1"]}
                w = load(tag, ck, role, c, wrong=True)
                if w is not None:
                    assert (w["rows"] == d["rows"]).all()
                    mw = E.map_scores(w["truth"], w["marginal"])["micro_map"]
                    r["wrong_input_micro_map"] = mw
                    r["wrong_input_gain_surviving"] = E.collapse_fraction(mw, m["micro_map"], mp["micro_map"])
                rows.append(r)
            if not rows:
                continue
            doc["models"][tag].setdefault("test", {})[role] = rows
            lines.append(f"\n## {tag} ({ck}) on {role}\n")
            lines.append("| condition | n | micro-F1 (greedy) | macro-F1 (greedy) | micro-F1 (val thr) | "
                         "micro-mAP | macro-mAP | prior micro-mAP | prior micro-F1 | slot mass | wrong-input gain kept |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                lines.append(f"| {COND_NAME[r['condition']]} | {r['n']} | {r['greedy_micro_f1']:.4f} | "
                             f"{r['greedy_macro_f1']:.4f} | {r['thr_micro_f1']:.4f} | {r['micro_map']:.4f} | "
                             f"{r['macro_map']:.4f} | {r['prior_micro_map']:.4f} | {r['prior_micro_f1']:.4f} | "
                             f"{r['slot_mass']:.3f} | "
                             f"{r.get('wrong_input_gain_surviving', float('nan')):.3f} |")

    tags = list(sel)
    if len(tags) >= 2:
        t1, t2 = tags[:2]
        lines.append(f"\n## Paired bootstrap: micro-mAP {t1} − {t2} (95% CI over patches, 200 resamples)\n")
        for role in a.tests:
            for c in CONDS:
                d1, d2 = load(t1, sel[t1], role, c), load(t2, sel[t2], role, c)
                if d1 is None or d2 is None:
                    continue
                assert (d1["rows"] == d2["rows"]).all()
                y = d1["truth"].astype(bool)
                diff = E.map_scores(y, d1["marginal"])["micro_map"] - E.map_scores(y, d2["marginal"])["micro_map"]
                lo, hi = boot_diff(y, d1["marginal"], d2["marginal"])
                doc.setdefault("paired", {})[f"{role}/{c}"] = {"diff": diff, "ci95": [lo, hi]}
                lines.append(f"- {role} {COND_NAME[c]}: {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    (a.out / "report.json").write_text(json.dumps(doc, indent=2) + "\n")
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
