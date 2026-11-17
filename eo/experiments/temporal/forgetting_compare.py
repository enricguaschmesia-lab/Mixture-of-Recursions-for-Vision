#!/usr/bin/env python
"""A1 Step 9: a pilot checkpoint's forgetting, paired per token with T360-T's extended base (`.venv`, CPU).

`eval_forgetting.py --model temporal --arm-config eo_terramesh/arm_t360_vanilla --precision fp32` scores the
4,416 held-out rows in T360's layout. It writes forgetting.json (exact means) and tokens_ce.npz (per-token CE,
stored as fp16). This pairs a checkpoint's reading with the base's. The base's reading is Step 6's,
/data/enric/reports/temporal/forgetting/t360t_base/extended_base, which equals T360's to 4.5e-09 (finding 24).

It reports:
- the difference overall and per modality, from the exact means;
- a paired bootstrap over the 4,416 rows (10,000 draws, a ratio of sums per draw) from the per-token arrays;
- Step 9's margin: at most +0.02 nats overall and +0.05 on any modality.

It refuses a pair that differs in precision, rows, token set or arm config, or in which tokens are scored
(label_mod). `--self-test` runs the base against itself (0 everywhere) and against a copy with one modality
raised by 0.06, which must fail the margin.

THE COORDS-FREE STAGE (2026-11-17, docs/plans/a1_nc_stage.md). Without --base, the base is the reading's own
layout's: T360-T's at 87,664 ids, the nc stage's (BASES) at 81,296, scored with `--arm-config
eo_terramesh/arm_b10ctrxnc_vanilla` (its 4,416 rows without Coords). Modality ids are v1's on both layouts, so
NAMES holds; Coords is simply absent from an nc reading.

    python eo/experiments/temporal/forgetting_compare.py --reading <forgetting-root>/<tag>/<checkpoint> [--base DIR]
           [--out OUT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path("/data/enric/reports/temporal/forgetting/t360t_base/extended_base")
BASES = {87_664: BASE, 81_296: Path("/data/enric/reports/temporal/forgetting/b10cltrxnct_base/extended_base")}
NAMES = {1: "S2L2A", 2: "S1GRD", 3: "S1RTC", 4: "DEM", 5: "NDVI", 6: "LULC", 7: "Coords"}   # eo_vocab.ID_TO_MODALITY
MARGIN_ALL, MARGIN_MOD = 0.02, 0.05
DRAWS, SEED = 10_000, 0


def load(d: Path):
    doc = json.loads((d / "forgetting.json").read_text())
    z = np.load(d / "tokens_ce.npz")
    return doc, {k: z[k] for k in z.files}


def per_row(ce_c, ce_b, lm):
    """Per row, per group (0 = all body tokens, then each modality): the summed difference and the count."""
    d = ce_c.astype(np.float64) - ce_b.astype(np.float64)
    S = np.zeros((len(d), 8))
    N = np.zeros((len(d), 8))
    valid = lm > 0
    S[:, 0], N[:, 0] = (d * valid).sum(1), valid.sum(1)
    for mid in NAMES:
        m = lm == mid
        S[:, mid], N[:, mid] = (d * m).sum(1), m.sum(1)
    return S, N


def boot(S, N):
    rng = np.random.default_rng(SEED)
    n = len(S)
    est = []
    for i in range(0, DRAWS, 500):
        c = rng.multinomial(n, np.full(n, 1.0 / n), size=min(500, DRAWS - i)).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            est.append((c @ S) / (c @ N))
    return np.concatenate(est)


def compare(doc_c, tok_c, doc_b, tok_b):
    for k in ("precision", "n_rows", "token_set", "arm_config", "vocab"):
        if doc_c.get(k) != doc_b.get(k):
            raise SystemExit(f"refused: {k} {doc_c.get(k)!r} against the base's {doc_b.get(k)!r}")
    if not np.array_equal(tok_c["rows"], tok_b["rows"]):
        raise SystemExit("refused: another row set or order")
    if not np.array_equal(tok_c["label_mod"], tok_b["label_mod"]):
        raise SystemExit("refused: other tokens scored (label_mod differs)")
    S, N = per_row(tok_c["ce"], tok_b["ce"], tok_b["label_mod"])
    B = boot(S, N)
    out = {"precision": doc_c["precision"], "n_rows": int(doc_c["n_rows"]), "draws": DRAWS,
           "margin": {"all": MARGIN_ALL, "any_modality": MARGIN_MOD}, "groups": {}}
    exact = {0: (doc_c["ce_all_body_tokens"], doc_b["ce_all_body_tokens"])}
    for mid, nm in NAMES.items():
        if nm in doc_c["per_modality"]:
            exact[mid] = (doc_c["per_modality"][nm]["ce"], doc_b["per_modality"][nm]["ce"])
    worst_repro = 0.0
    for g, (c, b) in exact.items():
        nm = "all" if g == 0 else NAMES[g]
        paired = float(S[:, g].sum() / N[:, g].sum())
        worst_repro = max(worst_repro, abs(paired - (c - b)))
        lo, hi = np.nanpercentile(B[:, g], [2.5, 97.5])
        cell = {"checkpoint": c, "base": b, "diff": c - b, "paired_fp16": paired, "ci95": [float(lo), float(hi)],
                "n_tokens": int(N[:, g].sum())}
        if g:
            for k in ("top1", "top5"):
                cell[f"{k}_diff"] = doc_c["per_modality"][nm][k] - doc_b["per_modality"][nm][k]
        cell["within_margin"] = bool(c - b <= (MARGIN_ALL if g == 0 else MARGIN_MOD))
        out["groups"][nm] = cell
    out["paired_vs_exact_max_abs"] = worst_repro
    out["passed"] = all(v["within_margin"] for v in out["groups"].values())
    return out


def show(rep, title):
    print(title)
    for nm, v in rep["groups"].items():
        print(f"  {nm:7s} {v['checkpoint']:.5f} - {v['base']:.5f} = {v['diff']:+.5f}  [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"
              f"  {'ok' if v['within_margin'] else 'OVER MARGIN'}")
    print(f"  paired fp16 vs exact means: {rep['paired_vs_exact_max_abs']:.2e}; margin passed: {rep['passed']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reading", type=Path)
    ap.add_argument("--base", type=Path, default=None, help="default: BASES by the reading's vocab (T360-T's at 87,664)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.base is None:
        vocab = json.loads((a.reading / "forgetting.json").read_text())["vocab"] if a.reading else 87_664
        if vocab not in BASES:
            raise SystemExit(f"no base reading for a {vocab}-id model: pass --base")
        a.base = BASES[vocab]
    doc_b, tok_b = load(a.base)
    if a.self_test:
        r0 = compare(doc_b, tok_b, doc_b, tok_b)
        show(r0, "self-test 1, the base against itself (0 everywhere):")
        ok0 = r0["passed"] and all(v["diff"] == 0 and v["ci95"] == [0.0, 0.0] for v in r0["groups"].values())
        doc_x = json.loads(json.dumps(doc_b))
        tok_x = dict(tok_b)
        doc_x["per_modality"]["NDVI"]["ce"] += 0.06
        n_all = sum(v["n_tokens"] for v in doc_b["per_modality"].values())
        doc_x["ce_all_body_tokens"] += 0.06 * doc_b["per_modality"]["NDVI"]["n_tokens"] / n_all
        tok_x["ce"] = np.where(tok_b["label_mod"] == 5, tok_b["ce"].astype(np.float32) + 0.06, tok_b["ce"]).astype(np.float16)
        r1 = compare(doc_x, tok_x, doc_b, tok_b)
        show(r1, "self-test 2, NDVI raised by 0.06 (must fail the margin on NDVI only):")
        ok1 = (not r1["passed"]) and [k for k, v in r1["groups"].items() if not v["within_margin"]] == ["NDVI"]
        print(f"self-test {'passed' if ok0 and ok1 else 'FAILED'}")
        return 0 if ok0 and ok1 else 1
    doc_c, tok_c = load(a.reading)
    rep = compare(doc_c, tok_c, doc_b, tok_b)
    rep.update(reading=str(a.reading), base=str(a.base), checkpoint=doc_c.get("checkpoint"))
    show(rep, f"forgetting of {a.reading} against {a.base}:")
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
