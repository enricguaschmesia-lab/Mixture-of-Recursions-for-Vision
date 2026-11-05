#!/usr/bin/env python
"""Our models against TerraMind on its own generation tasks: OUR half (2026-11-03).

TerraMind (arXiv 2504.11171, Table 3; Table 14 with token inputs) scores nine one-to-one
generations in pixel space: S1GRD and S1RTC -> S2L2A, DEM, NDVI, and S2L2A -> S1GRD, S1RTC,
DEM. This runs our decoder-only models on the same nine, on held-out scenes, so that
TerraMind can be run on exactly these scenes and both be decoded and scored the same way.

Two halves, two environments (as Step 12.6b):
  generate  .venv, GPU -- this script. Per model, per pair, per sampler: the 196 target
            tokens of every scene, as LOCAL codebook grids (N, 14, 14).
  rest      `mor`, GPU -- eo/experiments/generation/tm_compare_eval.py. TerraMind's tokens on the
            same rows, then decoding, scoring and the report.

ROWS (written once to rows.json, asserted equal by every later run and by the `mor` half):
held-out rows (eval_split v1, the 4,416 of every Phase 3+ number) that carry both S and T,
alternating corpora as stratified_target_rows does, first N, sorted. S1GRD
rows are SSL4EO-S12 only and S1RTC rows Major TOM only, so those pairs are single-corpus.

GENERATION: the prompt is build_prompt(sources=[S]) = [BO_S, S, EO_S, BO_T]; slot-masked,
exactly 196 tokens, with a KV cache (vanilla models only, as eval/ben_chain.py). Before a
pair is generated, ben_chain.gen_cache_check compares cached with uncached logits on the
real continuation of its first batch (must be < CACHE_TOL), and its shift=1 control must
disagree (> CONTROL_MIN). Two samplers:
  t1     temperature 1.0 over the whole slot: the project's protocol (D3.11, 12.6b);
  p08    temperature 1.0, top-p 0.8: TerraMind's default sampler (terratorch
         TerraMindGeneration: temps 1.0, top_p 0.8), so the two models also meet on one.
One sample per scene, seed SEED + batch index. fp32.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(python -m eo.train.gpu titanx) \
      python eo/experiments/generation/tm_compare_gen.py --model-tag t360 \
        --arm-config eo_terramesh/arm_t360_vanilla \
        --checkpoint /data/enric/runs/pretrain/phase3/vanilla360t_20261026_040535/checkpoint-17441

Outputs under /data/enric/generations/tm_compare/: rows.json, and <model>/<S>-<T>.npz with
a manifest. Runs in `.venv`; never imports terratorch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eo.data.eval_split import load_eval_rows, load_row_table  # noqa: E402
from eo.eval import ben_chain as BC  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402
from eo.terramesh_tok.contract import GRID, TOKENS_PER_SAMPLE  # noqa: E402

OUT = Path("/data/enric/generations/tm_compare")
ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
#: All nine of TerraMind's Table 3 (and Table 14, the same with token inputs); the first five ran first.
PAIRS = [("S1GRD", "S2L2A"), ("S1RTC", "S2L2A"), ("S2L2A", "S1GRD"), ("S2L2A", "DEM"), ("S1GRD", "NDVI"),
         ("S1GRD", "DEM"), ("S1RTC", "DEM"), ("S1RTC", "NDVI"), ("S2L2A", "S1RTC")]
SAMPLERS = {"t1": {"temperature": 1.0, "top_p": 1.0}, "p08": {"temperature": 1.0, "top_p": 0.8}}
N_ROWS, BATCH, SEED = 256, 32, 1000
CACHE_TOL, CONTROL_MIN = 1e-3, 1.0


def pair_name(S, T) -> str:
    return f"{S}-{T}"


def pair_rows(ds, eval_rows, corpus, S, T, n) -> list[int]:
    """The first n held-out rows carrying S and T, alternating corpora (stratified_target_rows'
    rule, with both modalities required), sorted."""
    ok = sorted(int(r) for r in eval_rows if {S, T} <= set(ds.present_modalities(int(r))))
    by = {c: [r for r in ok if corpus[r] == c] for c in ("majortom", "ssl4eos12")}
    seq = [by[c][i] for i in range(max(map(len, by.values()))) for c in ("majortom", "ssl4eos12")
           if i < len(by[c])]
    return sorted(seq[:n])


@torch.no_grad()
def sample(model, prompts, target, *, temperature, top_p, seed):
    """(B, L) prompts ending in BO_target -> (B, 196) raw codebook indices, and the mean
    unconstrained softmax mass on the slot. ben_chain.generate_blocks plus top-p over the slot
    (HF's rule: a token is kept while the mass before it is below top_p). The slot sort is
    15,360 wide, never vocabulary-sized; run it on the Titan X."""
    off, size, _, _ = BC.slot(target)
    out = model(input_ids=prompts, use_cache=True, logits_to_keep=1)
    cache, logits = out.past_key_values, out.logits[:, -1, :].float()
    g = torch.Generator(device=prompts.device).manual_seed(int(seed))
    toks, mass = [], torch.zeros(len(prompts), device=prompts.device)
    for t in range(TOKENS_PER_SAMPLE):
        mass += torch.softmax(logits, -1)[:, off:off + size].sum(-1)
        p = torch.softmax(logits[:, off:off + size] / temperature, -1)
        if top_p < 1.0:
            sp, si = torch.sort(p, -1, descending=True)
            sp = sp * ((sp.cumsum(-1) - sp) < top_p)
            p = torch.zeros_like(p).scatter(-1, si, sp)
            p = p / p.sum(-1, keepdim=True)
        x = torch.multinomial(p, 1, generator=g).squeeze(1)
        toks.append(x)
        if t == TOKENS_PER_SAMPLE - 1:
            break
        o = model(input_ids=(x + off).unsqueeze(1), past_key_values=cache, use_cache=True)
        cache, logits = o.past_key_values, o.logits[:, -1, :].float()
    return torch.stack(toks, 1), mass / TOKENS_PER_SAMPLE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--arm-config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--expect-gpu", default="TITAN X")
    ap.add_argument("--n", type=int, default=N_ROWS, help="rows per pair (smoke tests only)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from eo.generate import conditional as G
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU")
    gpu = torch.cuda.get_device_name(0)
    if args.expect_gpu not in gpu:
        raise SystemExit(f"CUDA sees {gpu!r}, expected {args.expect_gpu!r}")
    model, cfg = G.build_model(args.arm_config, args.checkpoint)
    if cfg.recursive.get("enable") or ("mor" in cfg and cfg.mor.get("enable")):
        raise SystemExit("the KV cache here is vanilla-only (eval/ben_chain.py)")
    token_set = (cfg.get("multimodal") or {}).get("token_set", "original")
    ds, _ = TF.eval_loader(cfg, num_workers=0)
    eval_rows, corpus = load_eval_rows(root_dir=ROOT), load_row_table(ROOT).corpus.values

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {pair_name(S, T): pair_rows(ds, eval_rows, corpus, S, T, args.n) for S, T in PAIRS}
    rfile = out / "rows.json"
    have = json.loads(rfile.read_text()) if rfile.exists() else {"token_set": token_set, "n": args.n,
                                                                    "eval_split": "v1", "rows": {}}
    if have["token_set"] != token_set or have["n"] != args.n or any(
            have["rows"].get(k, v) != v for k, v in rows.items()):
        raise SystemExit(f"{rfile}: rows, n or token set differ from this run's; use another --out")
    if set(rows) - set(have["rows"]):                    # a new pair joins; the others stay as they are
        have["rows"].update(rows)
        rfile.with_suffix(".tmp").write_text(json.dumps(have) + "\n")
        os.replace(rfile.with_suffix(".tmp"), rfile)

    mdir = out / args.model_tag
    mdir.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    for S, T in PAIRS:
        name = pair_name(S, T)
        if (mdir / f"{name}.json").exists():
            print(f"[skip] {args.model_tag} {name}: done", flush=True)
            continue
        R = rows[name]
        built = [G.build_prompt(ds, r, T, sources=[S]) for r in R]
        assert all(b["context"] == [S] for b in built)
        prompts = torch.stack([b["prompt"] for b in built]).cuda()
        off = BC.slot(T)[0]
        truth = torch.stack([b["truth"] for b in built]) - off            # raw, (N, 196)
        assert int(truth.min()) >= 0 and int(truth.max()) < BC.slot(T)[1]
        chk = BC.gen_cache_check(model, prompts[:8], truth[:8], T, n_steps=4)
        ctl = BC.gen_cache_check(model, prompts[:8], truth[:8], T, n_steps=4, shift=1)
        if not (chk < CACHE_TOL and ctl > CONTROL_MIN):
            raise SystemExit(f"{name}: cache check {chk:.2e} (< {CACHE_TOL}), control {ctl:.2f} (> {CONTROL_MIN})")
        arrays = {"rows": np.array(R), "truth": truth.numpy().astype(np.uint16).reshape(-1, GRID, GRID)}
        man = {"model_tag": args.model_tag, "arm_config": args.arm_config, "checkpoint": args.checkpoint,
               "token_set": token_set, "gpu": gpu, "source": S, "target": T, "n": len(R),
               "cache_check": chk, "cache_control": ctl, "precision": "fp32", "samplers": {}}
        for sname, kw in SAMPLERS.items():
            tg, toks, mass = time.perf_counter(), [], []
            for b, i in enumerate(range(0, len(R), BATCH)):
                x, m = sample(model, prompts[i:i + BATCH], T, seed=SEED + b, **kw)
                toks.append(x.cpu()); mass.append(m.cpu())
            tok = torch.cat(toks)
            acc = float((tok == truth).float().mean())
            arrays[f"{sname}|tokens"] = tok.numpy().astype(np.uint16).reshape(-1, GRID, GRID)
            arrays[f"{sname}|slot_mass"] = torch.cat(mass).numpy()
            man["samplers"][sname] = {**kw, "seed_base": SEED, "batch": BATCH, "token_accuracy": acc,
                                      "slot_mass_mean": float(torch.cat(mass).mean()),
                                      "codes_used": int(len(np.unique(arrays[f"{sname}|tokens"]))),
                                      "seconds": round(time.perf_counter() - tg, 1)}
            print(f"{args.model_tag} {name:12s} {sname:3s} n {len(R)}  token acc {acc:.4f}  "
                  f"slot mass {man['samplers'][sname]['slot_mass_mean']:.4f}  "
                  f"{man['samplers'][sname]['seconds']}s  ({time.perf_counter() - t0:.0f}s)", flush=True)
        tmp = mdir / f"{name}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, mdir / f"{name}.npz")
        (mdir / f"{name}.json").write_text(json.dumps(man, indent=1) + "\n")    # last: the done marker
    print(f"-> {mdir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
