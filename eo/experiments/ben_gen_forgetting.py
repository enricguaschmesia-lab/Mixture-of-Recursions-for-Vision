#!/usr/bin/env python
"""Phase 6 L7 (PHASE6_PLAN Steps 6.2 / 6.2a): does BEN fine-tuning damage GENERATION on TerraMesh?

The teacher-forced half (eval_forgetting.py) measured CE. This half generates, decodes and scores,
before and after each fine-tune, on the same held-out scenes and seeds:

  S2L2A column  DEM -> S2L2A (Gianfranco's example of a direction BEN cannot refresh),
                S1GRD -> S2L2A (from BEN's own S1 modality), none -> S2L2A (unconditional)
  LULC column   S2L2A -> LULC (a map; misclassified share and mIoU)

Rows: Phase 3's stratified_target_rows(..., T, 64), the 12.6b rows, so the extended base can be
read against Phase 3's B10. Generation: slot-masked, T = 1, one sample, KV-cached
(eo/eval/ben_chain.generate_blocks), the same seed per (cell, batch) for every model.
"before" is the EXTENDED base (Step 2.4). Models: base, b10_full, b10_lora, b10_full_li (S6.b),
and the L5/L6 runs. The head-only model is also a control: its pretrained weights are frozen, so
under the slot mask it must decode like the base (only the new label rows changed).

Three stages, three environments:
  generate  .venv, GPU (RCP A100)  -> <gen-root>/<tag>/<T>/{cells.npz, manifest.json}, Phase 3's format
  decode    `mor`, GPU (Titan X)   -> eo/experiments/ben_gen_forgetting_decode.py
  report    .venv, CPU             -> <out-root>/report.{json,md}

6.2a (offset regimes): S2L2A errors are split by the REAL scene's BOA-offset label
(clean.load_labels), and each decoded generation is classified offset-like when every band's
1st percentile exceeds 900 DN (clean.py's cross-check rule), calibrated on the decoded TRUE tokens.

    python eo/experiments/ben_gen_forgetting.py generate --model base
    python eo/experiments/ben_gen_forgetting.py generate --tag b10_head --config eo_ben/b10_head --checkpoint <dir>
    python eo/experiments/ben_gen_forgetting.py report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

P6 = "/data/enric/runs/pretrain/phase6"
CELLS = {"S2L2A": ["DEM", "S1GRD", "none"], "LULC": ["S2L2A"],
         # 6.2a, regime-balanced (2026-10-13): 12.6b's rows hold only 2 offset scenes in 64, so this
         # column takes 32 offset + 32 clean held-out majortom scenes (clean.load_labels, seed 0)
         "S2L2A_regime": ["DEM", "none"]}
#: column -> the modality generated and decoded
MOD = {"S2L2A": "S2L2A", "LULC": "LULC", "S2L2A_regime": "S2L2A"}


def regime_rows(ds, eval_rows, root, n_each=32, seed=0):
    from eo.terramesh_tok import clean as CL
    lab = CL.load_labels(CL.clean_root(Path(root))).set_index("row")
    ok = [int(r) for r in eval_rows if lab.at[int(r), "corpus"] == "majortom"
          and {"S2L2A", "DEM"} <= set(ds.present_modalities(int(r)))]
    off = np.array([r for r in ok if bool(lab.at[r, "offset"])])
    cln = np.array([r for r in ok if not bool(lab.at[r, "offset"])])
    rng = np.random.default_rng(seed)
    return sorted(int(r) for r in np.concatenate([rng.choice(off, n_each, replace=False),
                                                  rng.choice(cln, n_each, replace=False)]))
ROWS, SEED = 64, 42
MODELS = {"base": ("eo_ben/b10_full", None),
          "b10_full": ("eo_ben/b10_full", "b10full_20261012_214747/checkpoint-1461"),
          "b10_lora": ("eo_ben/b10_lora", "b10lora_20261012_221451/checkpoint-1461"),
          "b10_full_li": ("eo_ben/b10_full_li", "b10fullli_20261013_070123/checkpoint-1461")}


def generate(a):
    import torch
    import prepare_decode as PD
    from eo.data.eo_vocab import PAD_ID
    from eo.data.eval_split import load_eval_rows, load_row_table
    from eo.eval import ben_chain as C
    from eo.eval import teacher_forced as TF
    from eo.finetune.init import build_finetuned
    from eo.generate import conditional as G
    gpu = torch.cuda.get_device_name(0)
    if "TITAN V" in gpu:
        sys.exit("refusing the TITAN V")
    if a.model:
        cfg_name, ck = MODELS[a.model]
        tag, ck = a.model, (None if ck is None else f"{a.runs_root}/{ck}")
    else:
        tag, cfg_name, ck = a.tag, a.config, a.checkpoint
    root = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
    _, tcfg = G.build_model("eo_terramesh/arm_b10_vanilla", None, device="cpu")
    ds, _ = TF.eval_loader(tcfg, num_workers=0)
    eval_rows, corpus = load_eval_rows(root_dir=root), load_row_table(root).corpus.values
    model, _ = build_finetuned(cfg_name, ck, device="cuda", overrides=[f"finetune.init_from={a.init_from}"])
    ac = torch.float16 if a.autocast == "fp16" else None
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    for ti, (col, cells) in enumerate(CELLS.items()):
        T = MOD[col]
        out = Path(a.gen_root) / tag / col
        if (out / "manifest.json").exists():
            print(f"[skip] {tag} {col}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        R = (regime_rows(ds, eval_rows, root) if col == "S2L2A_regime"
             else TF.stratified_target_rows(ds, eval_rows, corpus, T, a.rows))
        off, size, _, eo = C.slot(T)
        arrays = {"column_rows": np.array(R)}
        man = {"arm_tag": tag, "arm_config": cfg_name, "checkpoint": ck, "token_set": "original", "target": T, "column": col,
               "seed": SEED, "temperature": 1.0, "slot_masked": True, "kv_cache": True, "autocast": a.autocast,
               "gpu": gpu, "fork_commit": commit, "cells": {}}
        for si, S in enumerate(cells):
            rows = R if S == "none" else [r for r in R if S in ds.present_modalities(r)]
            built = [G.build_prompt(ds, r, T, sources=[] if S == "none" else [S]) for r in rows]
            assert all(b["context"] == ([] if S == "none" else [S]) for b in built)
            toks, mass, t0 = [], [], time.time()
            for b0 in range(0, len(built), a.batch):
                pr = torch.stack([b["prompt"] for b in built[b0:b0 + a.batch]]).cuda()
                r = C.generate_blocks(model, pr, T, n_samples=1, temperature=1.0,
                                      seed=SEED * 1_000_003 + (ti * 10 + si) * 100_003 + b0, autocast=ac)
                toks.append(r["tokens"][:, 0].astype(np.int64)); mass.append(r["slot_mass"][:, 0])
            tok, ms = np.concatenate(toks), np.concatenate(mass)
            ids = np.concatenate([tok + off, np.full((len(tok), 1), eo)], 1)
            grids, valid, rep = PD.to_grids(ids, T)
            truth = np.stack([b["truth"].numpy() for b in built]) - off
            arrays.update({f"{S}|rows": np.array(rows), f"{S}|grids": grids, f"{S}|valid": valid, f"{S}|slot_mass": ms})
            man["cells"][S] = {"n": len(rows), "seconds": round(time.time() - t0, 1), "repair": rep,
                               "token_accuracy": float((tok == truth).mean()), "mean_slot_mass": float(ms.mean()),
                               "codes_used": int(len(np.unique(tok))), "codes_used_truth": int(len(np.unique(truth)))}
            print(f"{tag} {S:>6}->{T:<5} n {len(rows):3d}  slot mass {ms.mean():.4f}  token acc "
                  f"{man['cells'][S]['token_accuracy']:.4f}  ({man['cells'][S]['seconds']} s)", flush=True)
        tmp = out / "cells.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, out / "cells.npz")
        (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")   # last: the done marker


def report(a):
    from eo.terramesh_tok import clean as CL
    root = Path(a.out_root)
    lab = CL.load_labels(CL.clean_root(Path(os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val"))))
    offset = dict(zip(lab.row.astype(int), lab.offset.astype(bool)))
    rng = np.random.default_rng(0)

    def ci(d, n=2000):
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    doc, md = {}, ["# L7: generation forgetting on TerraMesh (decoded, 12.6b rows)", ""]
    for T, cells in CELLS.items():
        tdir = root / T
        if not (tdir / "errors_base.npz").exists():
            continue
        modality = MOD[T]
        tags = sorted(p.stem[len("errors_"):] for p in tdir.glob("errors_*.npz") if p.stem not in ("errors_ceiling",))
        tags = ["base"] + [t for t in tags if t != "base"]
        E = {t: np.load(tdir / f"errors_{t}.npz") for t in tags}
        ceil = np.load(tdir / "errors_ceiling.npz")
        metric = "err" if modality == "LULC" else "err_ml7"
        md += [f"## {T} ({'misclassified share' if modality == 'LULC' else 'MSE after 7x7 multilook, z units'}; "
               f"decoded true tokens = {ceil[f'ceiling|{T}|{metric}'].mean():.4f})", "",
               "| cell | n | " + " | ".join(tags) + " | " + " | ".join(f"{t} − base [95% CI]" for t in tags[1:]) + " |",
               "|" + "---|" * (2 + 2 * len(tags) - 1)]
        for S in cells:
            base_rows = E["base"][f"base|{S}|rows"]
            vals = {t: E[t][f"{t}|{S}|{metric}"] for t in tags}
            assert all((E[t][f"{t}|{S}|rows"] == base_rows).all() for t in tags)
            diffs = {t: vals[t] - vals["base"] for t in tags[1:]}
            doc[f"{T}/{S}"] = {"n": int(len(base_rows)), "mean": {t: float(v.mean()) for t, v in vals.items()},
                               "minus_base": {t: {"mean": float(d.mean()), "ci95": ci(d)} for t, d in diffs.items()}}
            md.append(f"| {S} → {modality} | {len(base_rows)} | " + " | ".join(f"{vals[t].mean():.4f}" for t in tags) + " | "
                      + " | ".join(f"{diffs[t].mean():+.4f} [{ci(diffs[t])[0]:+.4f}, {ci(diffs[t])[1]:+.4f}]" for t in tags[1:]) + " |")
            if modality == "LULC":
                miou = {t: float(E[t][f"{t}|{S}|miou"]) for t in tags}
                doc[f"{T}/{S}"]["miou"] = miou
                md.append(f"| mIoU | | " + " | ".join(f"{miou[t]:.4f}" for t in tags) + " | |")
        if modality == "S2L2A":
            # 6.2a: by the real scene's offset regime, and the generated-offset rate
            R = ceil[f"ceiling|{T}|rows"]
            cp = ceil[f"ceiling|{T}|p1min"]
            ro = np.array([offset[int(r)] for r in R])
            calib = {"decoded_truth_offset_like_if_offset": float((cp[ro] > 900).mean()),
                     "decoded_truth_offset_like_if_clean": float((cp[~ro] > 900).mean()),
                     "n_offset": int(ro.sum()), "n_clean": int((~ro).sum())}
            doc[f"{T}/offset_rule_on_decoded_truth"] = calib
            md += ["", f"**6.2a.** Offset rule on the decoded TRUE tokens: offset scenes flagged {calib['decoded_truth_offset_like_if_offset']:.2f}, "
                       f"clean scenes flagged {calib['decoded_truth_offset_like_if_clean']:.2f} ({calib['n_offset']} offset / {calib['n_clean']} clean rows).", "",
                   "| cell | regime | n | " + " | ".join(tags) + " | generated offset-like rate: " + " / ".join(tags) + " |",
                   "|" + "---|" * (4 + len(tags))]
            for S in cells:
                rows = E["base"][f"base|{S}|rows"]
                reg = np.array([offset[int(r)] for r in rows])
                for name, sel in (("offset", reg), ("clean", ~reg)):
                    if sel.sum() == 0:
                        continue
                    m = {t: float(E[t][f"{t}|{S}|{metric}"][sel].mean()) for t in tags}
                    g = {t: float((E[t][f"{t}|{S}|p1min"][sel] > 900).mean()) for t in tags}
                    doc[f"{T}/{S}/{name}"] = {"n": int(sel.sum()), "mean": m, "generated_offset_rate": g}
                    md.append(f"| {S} → S2L2A | {name} | {int(sel.sum())} | " + " | ".join(f"{m[t]:.4f}" for t in tags)
                              + " | " + " / ".join(f"{g[t]:.2f}" for t in tags) + " |")
        md.append("")
    gen = Path(a.gen_root)
    md += ["## Unconstrained slot mass during generation (leakage onto other slots / BEN ids)", "",
           "| model | " + " | ".join(f"{S}→{T}" for T, cs in CELLS.items() for S in cs) + " |", "|" + "---|" * (1 + sum(map(len, CELLS.values())))]
    for t in sorted(p.name for p in gen.iterdir()):
        row = []
        for T, cs in CELLS.items():
            f = gen / t / T / "manifest.json"
            man = json.loads(f.read_text()) if f.exists() else {"cells": {}}
            row += [f"{man['cells'][S]['mean_slot_mass']:.4f}" if S in man["cells"] else "" for S in cs]
        md.append(f"| {t} | " + " | ".join(row) + " |")
    (root / "report.json").write_text(json.dumps(doc, indent=1) + "\n")
    (root / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "report"])
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--tag"); ap.add_argument("--config"); ap.add_argument("--checkpoint")
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--autocast", choices=["none", "fp16"], default="fp16")
    ap.add_argument("--runs-root", default=P6)
    ap.add_argument("--init-from", default=f"{P6}/base/b10_ext_ckpt3300")
    ap.add_argument("--gen-root", default="/data/enric/generations/p6_gen_forgetting")
    ap.add_argument("--out-root", default="/data/enric/reports/ben/gen_forgetting")
    a = ap.parse_args()
    if a.stage == "generate":
        assert a.model or (a.tag and a.config and a.checkpoint), "--model, or --tag --config --checkpoint"
        generate(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
