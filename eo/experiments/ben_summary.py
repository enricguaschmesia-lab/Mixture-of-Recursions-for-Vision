"""Phase 6 Part 1 on B10, the summary table (PHASE6_PLAN §2b L9). `.venv`, CPU.

One row per B10 model, every number read from the files the runs wrote (never retyped):
  * the checkpoint SELECTED on its own val role (mean micro-mAP over S1 / S2 / S1+S2), never test;
  * test_a micro-mAP and greedy micro-F1 per condition, macro-mAP (S1+S2), the wrong-input control;
  * teacher-forced forgetting on TerraMesh's 4,416 held-out rows, against the EXTENDED base.
Paired bootstrap CIs (over test_a patches, same rows) for the comparisons that answer §2b's questions:
  pretraining (full vs scratch; LoRA-1% vs scratch-1%), adapters (LoRA vs head), the LR confound
  (LoRA 1e-3 vs LoRA 1e-4; full vs LoRA at the same LR 1e-4), label budget (LoRA 10% vs 1%).
PHASE6_PLAN §2c (2026-10-18) adds L11b (distillation on the unlabelled pool), its attribution control,
and L10 (trained on generated S2), each against b10_lora; models whose files do not exist yet are
skipped. The location-only kNN (L2, no pixels) is printed beside every result as a reference (item 41).

    python eo/experiments/ben_summary.py --out /data/enric/reports/ben/summary_b10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.eval import ben as E  # noqa: E402

REP = Path("/data/enric/reports/ben")
CONDS = ("S1", "S2", "S1S2")
CKS = ("checkpoint-487", "checkpoint-974", "checkpoint-1461")
#: tag, what it is, val role, labels
MODELS = [
    ("b10_scratch", "L4: from scratch (B10 architecture, random init), full recipe", "ft_valsel", "10%"),
    ("b10_head", "L5: head only (frozen B10; the 21 label rows train), lr 1e-3", "ft_valsel", "10%"),
    ("b10_lora_lr1e4", "L6: LoRA r16, lr 1e-4 (= full's)", "ft_valsel", "10%"),
    ("b10_lora", "LoRA r16, lr 1e-3", "ft_valsel", "10%"),
    ("b10_full", "full fine-tune, lr 1e-4", "ft_valsel", "10%"),
    ("b10_full_li", "S6.b: full, loss on labels + inputs", "ft_valsel", "10%"),
    ("b10_scratch_1pct", "L8: from scratch, SMARTIES 1% list", "ft_valsel1", "1%"),
    ("b10_lora_1pct", "L8: LoRA lr 1e-3, SMARTIES 1% list", "ft_valsel1", "1%"),
    ("b10_lora_l11b", "L11b: b10_lora + distillation from S1+S2 into S1 on 272,767 unlabelled pairs", "ft_valsel", "10% + unlab."),
    ("b10_lora_l11b_s1t", "L11b.6: the same, teacher given S1 only (self-training control)", "ft_valsel", "10% + unlab."),
    ("b10_lora_l10", "L10: b10_lora + a fourth condition, S1 with a generated S2", "ft_valsel", "10%"),
]
PAIRS = [("b10_full", "b10_scratch", "pretraining, 10% (full vs scratch)"),
         ("b10_lora_1pct", "b10_scratch_1pct", "pretraining, 1% (LoRA vs scratch)"),
         ("b10_lora", "b10_head", "what the adapters add (LoRA vs head, both lr 1e-3)"),
         ("b10_lora", "b10_lora_lr1e4", "the LR confound (LoRA 1e-3 vs 1e-4)"),
         ("b10_full", "b10_lora_lr1e4", "full vs LoRA at the same LR 1e-4"),
         ("b10_lora", "b10_lora_1pct", "label budget (LoRA 10% vs 1%)"),
         ("b10_lora_l11b", "b10_lora", "L11b: distillation from S1+S2 on the unlabelled pool"),
         ("b10_lora_l11b", "b10_lora_l11b_s1t", "L11b.6: S2's knowledge beyond self-training"),
         ("b10_lora_l10", "b10_lora", "L10: the classifier also trained on generated S2 (real inputs here)")]


def jl(p):
    return json.loads(Path(p).read_text())


def select(tag, val):
    s = {}
    for ck in CKS:
        f = [REP / tag / ck / val / f"{c}.json" for c in CONDS]
        if all(x.exists() for x in f):
            s[ck] = float(np.mean([jl(x)["micro_map"] for x in f]))
    return (max(s, key=s.get), s) if s else (None, s)


def boot(y, a, b, n=100, seed=0):
    rng = np.random.default_rng(seed)
    mm = lambda yy, ss: E.average_precision(yy.ravel(), ss.ravel())
    d = [mm(y[i], a[i]) - mm(y[i], b[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(mm(y, a) - mm(y, b)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    base_f = jl(REP / "forgetting/b10_base/extended_base/forgetting.json")
    prior = jl(REP / "report_b10_test_a/report.json") if (REP / "report_b10_test_a/report.json").exists() else None
    rows, sel, md = {}, {}, []
    l2 = REP / "analysis_b10" / "l2.json"
    loc = jl(l2)["location_knn"]["10"]["micro_map"] if l2.exists() else None
    for tag, what, val, lab in MODELS:
        ck, scores = select(tag, val)
        r = {"what": what, "val_role": val, "labels": lab, "val_mean_micro_map": scores, "selected": ck}
        if ck:
            sel[tag] = ck
            t = REP / tag / ck / "test_a"
            for c in CONDS:
                if (t / f"{c}.json").exists():
                    d = jl(t / f"{c}.json")
                    r[f"test_{c}"] = {"micro_map": d["micro_map"], "macro_map": d["macro_map"], "micro_f1": d["greedy_micro_f1"]}
            if (t / "S1S2_wrong.json").exists():
                r["test_S1S2_wrong"] = jl(t / "S1S2_wrong.json")["micro_map"]
            ff = REP / "forgetting" / tag / ck / "forgetting.json"
            if ff.exists():
                f = jl(ff)
                r["forgetting_ce_all"] = f["ce_all_body_tokens"] - base_f["ce_all_body_tokens"]
                r["forgetting_per_modality"] = {m: f["per_modality"][m]["ce"] - base_f["per_modality"][m]["ce"]
                                                for m in f["per_modality"]}
        if ck or (REP / tag).exists():
            rows[tag] = r
    # paired comparisons on test_a
    comps = {}
    for x, y, what in PAIRS:
        if x not in sel or y not in sel:
            continue
        comps[f"{x} - {y}"] = {"what": what}
        for c in CONDS:
            fx, fy = REP / x / sel[x] / "test_a" / f"{c}.npz", REP / y / sel[y] / "test_a" / f"{c}.npz"
            if not (fx.exists() and fy.exists()):
                continue
            zx, zy = np.load(fx), np.load(fy)
            assert (zx["rows"] == zy["rows"]).all()
            d, ci = boot(zx["truth"].astype(bool), zx["marginal"].astype(np.float64), zy["marginal"].astype(np.float64), a.boot)
            comps[f"{x} - {y}"][c] = {"diff": d, "ci95": ci}
        print(x, "-", y, {c: round(v["diff"], 4) for c, v in comps[f"{x} - {y}"].items() if c != "what"}, flush=True)
    doc = {"models": rows, "comparisons": comps, "forgetting_reference": "extended base (b10_ext_ckpt3300)",
           "location_only_knn10_micro_map": loc,
           "test": "test_a = SMARTIES MM val, 103,944 patches"}
    (a.out / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")

    g = lambda r, k: f"{r[k]['micro_map']:.3f}" if k in r else "—"
    md += ["# Phase 6 Part 1 on B10: every model on BigEarthNet (test_a, micro-mAP)", "",
           "Class prior 0.311 (micro-F1 0.415). Checkpoint selected on each model's val role. Forgetting: teacher-forced CE "
           "on TerraMesh's 4,416 held-out rows, minus the extended base's (nats; + = forgot)."
           + (f" Reference, no pixels: location-only kNN (k = 10, L2) {loc:.3f}." if loc is not None else ""), "",
           "| model | labels | selected | S1 | S2 | S1+S2 | macro-mAP S1+S2 | micro-F1 S1+S2 | wrong input | forgetting, all tokens |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, what, val, lab in MODELS:
        if tag not in rows:
            continue
        r = rows[tag]
        fo = f"{r['forgetting_ce_all']:+.3f}" if "forgetting_ce_all" in r else "—"
        ms = f"{r['test_S1S2']['macro_map']:.3f}" if "test_S1S2" in r else "—"
        f1 = f"{r['test_S1S2']['micro_f1']:.3f}" if "test_S1S2" in r else "—"
        wr = f"{r['test_S1S2_wrong']:.3f}" if "test_S1S2_wrong" in r else "—"
        md.append(f"| **{tag}**: {what} | {lab} | {(r['selected'] or '—').replace('checkpoint-', '')} | {g(r, 'test_S1')} | "
                  f"{g(r, 'test_S2')} | {g(r, 'test_S1S2')} | {ms} | {f1} | {wr} | {fo} |")
    md += ["", "## Paired differences on test_a (micro-mAP, 95% bootstrap CI over patches)", "",
           "| comparison | question | S1 | S2 | S1+S2 |", "|---|---|---|---|---|"]
    for k, v in comps.items():
        cell = lambda c: f"{v[c]['diff']:+.3f} [{v[c]['ci95'][0]:+.3f}, {v[c]['ci95'][1]:+.3f}]" if c in v else "—"
        md.append(f"| {k} | {v['what']} | {cell('S1')} | {cell('S2')} | {cell('S1S2')} |")
    md += ["", "## Forgetting per modality (CE minus the extended base's)", "",
           "| model | " + " | ".join(base_f["per_modality"]) + " |", "|" + "---|" * (1 + len(base_f["per_modality"]))]
    for tag, *_ in MODELS:
        r = rows.get(tag, {})
        if "forgetting_per_modality" in r:
            md.append(f"| {tag} | " + " | ".join(f"{r['forgetting_per_modality'][m]:+.3f}" for m in base_f["per_modality"]) + " |")
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
