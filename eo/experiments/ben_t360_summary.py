"""BEN with T360 (2026-11-02, docs/notes/ben_t360_2026-11-02.md): every model on test_a, and the paired
comparisons that answer the plan's questions. `.venv`, CPU.

ben_summary.py's method, extended to three bases (B10, B10c, T360):
  * each model at the checkpoint SELECTED on its own val role (mean micro-mAP over S1 / S2 / S1+S2), never test;
  * test_a micro-mAP and greedy micro-F1 per condition, macro-mAP (S1+S2), the wrong-input control;
  * forgetting: teacher-forced CE on TerraMesh's 4,416 held-out rows minus ITS OWN extended base's, both
    scored at the same precision (asserted; T360 fp32, B10 and B10c fp16);
  * paired bootstrap CIs over test_a patches (same rows, asserted) for the plan's comparisons.
SMARTIES (arXiv 2506.19585, Tables 1 and 8; other protocol: encoders, linear probe for S1 and S1+S2,
fine-tuned for S2, 100 epochs) is printed beside, as context.

    python eo/experiments/ben_t360_summary.py --out /data/enric/reports/ben/summary_t360
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
#: tag, base, regime, val role, labels
MODELS = [
    ("b10_head", "B10", "head only", "ft_valsel", "10%"),
    ("b10_lora", "B10", "LoRA", "ft_valsel", "10%"),
    ("b10_full", "B10", "full", "ft_valsel", "10%"),
    ("b10_lora_1pct", "B10", "LoRA", "ft_valsel1", "1%"),
    ("b10c_lora", "B10c", "LoRA", "ft_valsel", "10%"),
    ("t360_head", "T360", "head only", "ft_valsel", "10%"),
    ("t360_lora", "T360", "LoRA", "ft_valsel", "10%"),
    ("t360_full", "T360", "full", "ft_valsel", "10%"),
    ("t360_lora_1pct", "T360", "LoRA", "ft_valsel1", "1%"),
    ("t360_full_lr3e5", "T360", "full, LR 3e-5", "ft_valsel", "10%"),   # worklog item 48
]
BASE_TAG = {"B10": "b10_base", "B10c": "b10c_base", "T360": "t360_base"}
PAIRS = [("t360_lora", "b10c_lora", "data and size, same tokens (LoRA)"),
         ("t360_lora", "b10_lora", "T360 against BEN's small model (LoRA)"),
         ("t360_head", "b10_head", "the frozen representation (head only)"),
         ("t360_full", "b10_full", "full fine-tune"),
         ("t360_lora_1pct", "b10_lora_1pct", "1% of the labels (LoRA)"),
         ("b10c_lora", "b10_lora", "the token set alone (LoRA)"),
         ("t360_lora", "t360_lora_1pct", "label budget on T360 (10% vs 1%)"),
         ("b10_lora", "b10_lora_1pct", "label budget on B10 (10% vs 1%)"),
         ("t360_lora", "t360_head", "what the adapters add on T360"),
         ("t360_full", "t360_lora", "full vs LoRA on T360"),
         ("t360_full_lr3e5", "t360_full", "full fine-tune at 3e-5 vs 1e-4 on T360"),
         ("t360_full_lr3e5", "t360_lora", "full at 3e-5 vs LoRA on T360")]
SMARTIES = {"ViT-B, 10%": {"S1": 0.789, "S2": 0.869, "S1S2": 0.854},
            "ViT-L, 10%": {"S1": 0.805, "S2": 0.877, "S1S2": 0.867},
            "ViT-L, 1%": {"S1S2": 0.792}}
PRIOR = 0.311


def jl(p):
    return json.loads(Path(p).read_text())


def select(tag, val):
    s = {}
    for ck in CKS:
        f = [REP / tag / ck / val / f"{c}.json" for c in CONDS]
        if all(x.exists() for x in f):
            s[ck] = float(np.mean([jl(x)["micro_map"] for x in f]))
    return (max(s, key=s.get), s) if s else (None, s)


def _sorted(y, s):
    o = np.argsort(-s.ravel(), kind="mergesort")
    ss = s.ravel()[o]
    return o, y.ravel()[o].astype(np.float64), np.r_[np.where(np.diff(ss))[0], ss.size - 1]


def _wap(w, o, ys, thr):
    """E.average_precision of the sample in which element i appears w[i] times (ties grouped, as there)."""
    ww = w[o]
    tps, tot = np.cumsum(ww * ys)[thr], np.cumsum(ww)[thr]
    prec = np.divide(tps, tot, out=np.zeros_like(tps), where=tot > 0)
    return float(np.sum(np.diff(np.r_[0.0, tps / tps[-1]]) * prec))


def boot(y, a, b, n=100, seed=0):
    """Paired bootstrap over patches, with ben_summary's draws. A resample is a count per patch, so each model's
    scores are sorted once and each resample is a weighted AP: equal to the AP of the duplicated sample (tested to
    3e-16, and exactly with ties), ~5x faster."""
    rng = np.random.default_rng(seed)
    mm = lambda yy, ss: E.average_precision(yy.ravel(), ss.ravel())
    sa, sb = _sorted(y, a), _sorted(y, b)
    d = []
    for _ in range(n):
        w = np.repeat(np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(np.float64), y.shape[1])
        d.append(_wap(w, *sa) - _wap(w, *sb))
    return float(mm(y, a) - mm(y, b)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def forgetting(tag, ck, base):
    ff = REP / "forgetting" / tag / ck / "forgetting.json"
    fb = REP / "forgetting" / BASE_TAG[base] / "extended_base" / "forgetting.json"
    if not (ff.exists() and fb.exists()):
        return None
    f, b = jl(ff), jl(fb)
    pf, pb = f.get("precision", "fp16"), b.get("precision", "fp16")
    assert pf == pb, f"{ff} at {pf} against its base at {pb}"
    assert f["n_rows"] == b["n_rows"] == 4416 and f["token_set"] == b["token_set"], (ff, fb)
    return {"precision": pf, "token_set": f["token_set"], "ce_all": f["ce_all_body_tokens"] - b["ce_all_body_tokens"],
            "per_modality": {m: f["per_modality"][m]["ce"] - b["per_modality"][m]["ce"] for m in f["per_modality"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, sel = {}, {}
    for tag, base, regime, val, lab in MODELS:
        ck, scores = select(tag, val)
        r = {"base": base, "regime": regime, "val_role": val, "labels": lab, "val_mean_micro_map": scores, "selected": ck}
        if ck:
            sel[tag] = ck
            t = REP / tag / ck / "test_a"
            for c in CONDS:
                if (t / f"{c}.json").exists():
                    d = jl(t / f"{c}.json")
                    r[f"test_{c}"] = {"micro_map": d["micro_map"], "macro_map": d["macro_map"],
                                      "micro_f1": d["greedy_micro_f1"], "n": d["n"]}
            if (t / "S1S2_wrong.json").exists():
                r["test_S1S2_wrong"] = jl(t / "S1S2_wrong.json")["micro_map"]
            fo = forgetting(tag, ck, base)
            if fo:
                r["forgetting"] = fo
        rows[tag] = r
        print(tag, ck, {c: round(r[f"test_{c}"]["micro_map"], 4) for c in CONDS if f"test_{c}" in r}, flush=True)
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
            assert (zx["rows"] == zy["rows"]).all() and (zx["truth"] == zy["truth"]).all(), (fx, fy)
            d, ci = boot(zx["truth"].astype(bool), zx["marginal"].astype(np.float64), zy["marginal"].astype(np.float64), a.boot)
            comps[f"{x} - {y}"][c] = {"diff": d, "ci95": ci}
        print(x, "-", y, {c: round(v["diff"], 4) for c, v in comps[f"{x} - {y}"].items() if c != "what"}, flush=True)
    doc = {"models": rows, "comparisons": comps, "smarties": SMARTIES, "class_prior": PRIOR,
           "forgetting_reference": "each model's own extended base", "n_boot": a.boot,
           "test": "test_a = SMARTIES MM val, 103,944 patches"}
    (a.out / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")

    g = lambda r, k: f"{r[k]['micro_map']:.3f}" if k in r else "—"
    md = ["# BEN with T360: every model on test_a (micro-mAP)", "",
          f"Class prior {PRIOR}. Checkpoint selected on each model's val role. Forgetting: teacher-forced CE on "
          "TerraMesh's 4,416 held-out rows minus the model's own extended base (nats; + = forgot). SMARTIES "
          "(other protocol): " + "; ".join(f"{k} " + " / ".join(f"{v[c]:.3f}" if c in v else "—" for c in CONDS)
                                           for k, v in SMARTIES.items()) + ".", "",
          "| model | base | regime | labels | selected | S1 | S2 | S1+S2 | macro-mAP S1+S2 | micro-F1 S1+S2 | wrong input | forgetting (precision) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, base, regime, val, lab in MODELS:
        r = rows[tag]
        fo = f"{r['forgetting']['ce_all']:+.3f} ({r['forgetting']['precision']})" if "forgetting" in r else "—"
        ms = f"{r['test_S1S2']['macro_map']:.3f}" if "test_S1S2" in r else "—"
        f1 = f"{r['test_S1S2']['micro_f1']:.3f}" if "test_S1S2" in r else "—"
        wr = f"{r['test_S1S2_wrong']:.3f}" if "test_S1S2_wrong" in r else "—"
        md.append(f"| {tag} | {base} | {regime} | {lab} | {(r['selected'] or '—').replace('checkpoint-', '')} | "
                  f"{g(r, 'test_S1')} | {g(r, 'test_S2')} | {g(r, 'test_S1S2')} | {ms} | {f1} | {wr} | {fo} |")
    md += ["", "## Paired differences on test_a (micro-mAP, 95% bootstrap CI over patches)", "",
           "| comparison | question | S1 | S2 | S1+S2 |", "|---|---|---|---|---|"]
    for k, v in comps.items():
        cell = lambda c: f"{v[c]['diff']:+.3f} [{v[c]['ci95'][0]:+.3f}, {v[c]['ci95'][1]:+.3f}]" if c in v else "—"
        md.append(f"| {k} | {v['what']} | {cell('S1')} | {cell('S2')} | {cell('S1S2')} |")
    mods = next((list(r["forgetting"]["per_modality"]) for r in rows.values() if "forgetting" in r), [])
    if mods:
        md += ["", "## Forgetting per modality (CE minus the model's own extended base)", "",
               "| model | " + " | ".join(mods) + " |", "|" + "---|" * (1 + len(mods))]
        for tag, *_ in MODELS:
            if "forgetting" in rows[tag]:
                md.append(f"| {tag} | " + " | ".join(f"{rows[tag]['forgetting']['per_modality'][m]:+.3f}" for m in mods) + " |")
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
