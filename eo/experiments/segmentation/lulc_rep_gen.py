#!/usr/bin/env python
"""LULC as RGB map tokens against TerraMind's LULC tokens: the GENERATION half (`.venv`, GTX Titan X).
Enric, 2026-11-23, after Step 5.3. Protocol and reading rule: docs/plans/a2_segmentation.md, "Follow-up: LULC through
the map tokenizer against TerraMind's LULC tokens"; written before any number of it.

Two models in one architecture (eo_seg/r1_aligned, 161,665 ids), the same prompts, the same rows:
  base   A2's map-extended base (b10cltrxnc_ext_map_nc_ckpt33000): B10cL-tr-x10-NC, identity on every base logit.
         Its map rows are untrained, so it writes LULC only through TerraMind's tokens.
  r1     R1's base (r1aligned_n4000_ckpt4000): the same, mid-trained 4,000 steps with each ssl4eos12 row's LULC as
         a map block beside TerraMind's LULC block. It writes LULC both ways.

PART 1, `gen1`: LULC from inputs. Target `LULC` (TerraMind's 196 LULC tokens, slot-constrained) or `MAP` (196 codes
of the aligned slot after BO_MAP_LULC). Sources:
  none   [BO_target] alone: the class prior the model writes (a floor).
  s2     [S2L2A][BO_target].
  all    [S2L2A][S1][DEM][NDVI][BO_target], the row's own S1 (S1GRD on ssl4eos12 rows, S1RTC on Major TOM rows).
Conditions: base>LULC, r1>LULC, r1>MAP, each with none/s2/all. K samples at temperature 1 over the slot (A2's
reading: K = 8, voted per pixel by the scorer); sample j of batch s takes seed SEED * 1,000,003 + j * 7,919 + s, the
same in every condition. Rows: 1,024 held-out rows (eval_rows_v1 on val_clean; all carry every block and a val map),
the first 512 of each corpus in row order, alternating (`alternating`); run in four shards of 256 rows (`--shard`).

PART 3, `gen3`: other modalities from LULC. Targets S2L2A, S1GRD (ssl4eos12 rows), S1RTC (Major TOM rows), DEM, NDVI.
Sources: [LULC] (TerraMind's tokens) or [MAP] (the row's true map codes), or none. Conditions base<LULC, r1<LULC,
r1<MAP, base<none, r1<none. One sample a scene at temperature 1 (`t1`) and at top-p 0.8 (`p08`), seed 1000 + batch
index, as eo/experiments/generation/tm_compare_gen.py. Rows: the first 256 held-out rows carrying the target,
alternating corpora (tm_compare_gen.pair_rows' rule).

PART 3, `tf3`: the same five conditions teacher-forced on every held-out row: CE of the target's 196 tokens (full
vocabulary softmax), per row. Exact and paired: the target's tokens are the same in every condition.

Every generating condition first passes A2's cache check (eo/eval/seg.cache_check: the fp32 rule of gate Z8) on its
first 8 prompts, and its control (every cached step one position off) must fail it. Positions are sequential from 0,
as both models trained. fp32. Prompts in one batch share a length (one source layout a condition).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python eo/experiments/segmentation/lulc_rep_gen.py gen1 --model base

Outputs: /data/enric/generations/a2_lulc_rep/<part>/<condition>.npz + .json (the json last: the done marker).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = Path(os.environ.get("LULC_REP_OUT", "/data/enric/generations/a2_lulc_rep"))
CONFIG = "eo_seg/r1_aligned"
BASES = {"base": Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000"),
         "r1": Path("/data/enric/runs/pretrain/a2_seg/base/r1aligned_n4000_ckpt4000")}
MAP = "MAP"
SOURCES1 = ("none", "s2", "all")
GEN1 = {"base": ["LULC"], "r1": ["LULC", MAP]}                  # part 1's targets per model
TARGETS3 = ("S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI")
GEN3 = {"base": ["LULC", "none"], "r1": ["LULC", MAP, "none"]}  # part 3's sources per model
SAMPLERS3 = {"t1": 1.0, "p08": 0.8}
SEED, SEED3, K, N1, N3, BATCH = 20261123, 1000, 8, 1024, 256, 32
SHARD = 256                     # part 1's unit of work: 256 rows (resumable; a TITAN V stall costs one)
RATIO_RULE = 1000.0


def setup():
    """(scenes, held-out rows, corpus per row) on val_clean, through R1's own eval builder (every refusal it has)."""
    from hydra import compose, initialize_config_dir
    from eo.data import map_mix_dataset as MD
    from util.env import load_dotenv
    load_dotenv()
    ov = []
    if not os.environ.get("TERRAMESH_TRAIN_TOK_ROOT"):       # r1_read.corpus: build_eval resolves a train root, reads val only
        tm = Path("/data/enric/data/TerraMesh")
        os.environ["TERRAMESH_TRAIN_TOK_ROOT"] = str(tm / "train_tok_partial_r1")
        ov += ["++multimodal.allow_partial_train_root=true", f"r1.train_maps={tm / 'maps/mini_r1/MAP_aligned'}"]
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        cfg = compose(config_name=CONFIG, overrides=ov)
    ev = MD.build_eval(cfg)
    scenes, rows = ev["map"].scenes, np.asarray(ev["map"].rows, np.int64)
    import pandas as pd
    corp = pd.read_parquet(Path(os.environ["TERRAMESH_TOK_ROOT"]) / "tok_index.parquet",
                           columns=["row", "corpus"]).sort_values("row").corpus.values
    return scenes, rows, corp


def slot_of(scenes, target):
    """(offset, size, bo) of a target: TerraMind's slot in the nc layout, or the aligned map slot."""
    if target == MAP:
        off, size = scenes.lay.slot("aligned")
        return off, size, scenes.lay.bo("lulc")
    info = scenes.base._V.get_modality(target)
    return info.codebook_offset, info.codebook_size, info.bo_id


def s1_of(scenes, row):
    s1 = [m for m in scenes.base.present_modalities(int(row)) if m.startswith("S1")]
    if len(s1) != 1:
        raise RuntimeError(f"row {row}: S1 blocks {s1}")
    return s1[0]


def sources_of(scenes, row, src):
    if src == "none":
        return []
    if src == "s2":
        return ["S2L2A"]
    if src == "all":
        return ["S2L2A", s1_of(scenes, row), "DEM", "NDVI"]
    return [src]                                              # part 3: "LULC" or "MAP"


def truth(scenes, row, target):
    """The row's true raw codes of `target` (196,), checked against its block's BO and slot."""
    off, size, bo = slot_of(scenes, target)
    ch = scenes.chunk(int(row), target)
    if int(ch[0]) != bo or len(ch) != 198:
        raise RuntimeError(f"row {row} {target}: block starts {int(ch[0])}, BO {bo}; length {len(ch)}")
    raw = ch[1:-1] - off
    if int(raw.min()) < 0 or int(raw.max()) >= size:
        raise RuntimeError(f"row {row} {target}: codes outside the slot")
    return raw


def prompt(scenes, row, names, target):
    bo = slot_of(scenes, target)[2]
    return torch.cat([scenes.chunk(int(row), n) for n in names] + [torch.tensor([bo], dtype=torch.long)])


def load_model(tag, device):
    from eo.finetune.init import strict_load
    from eo.generate import conditional as G
    model, _ = G.build_model(CONFIG, None, device="cpu")
    rec = strict_load(model, BASES[tag])
    model.to(device).eval()
    return model, rec


def check_cache(model, prompts, slot):
    """seg.cache_check on 8 prompts, and its control one position off, which must fail."""
    from eo.eval import seg as SG
    pos = torch.arange(prompts.shape[1], device=prompts.device).expand(len(prompts), -1)
    chk = SG.cache_check(model, prompts, pos, slot, n_steps=4)
    ctl = SG.cache_check(model, prompts, pos, slot, n_steps=4, cache_offset=1)
    # Z8's fp32 rule, or Step 4.1's 1,000x rule (adopted by Enric for Step 5.2, 2026-11-22): at fp32 a cached decode
    # 1,000 times closer to recomputation than its one-position control is a rounding difference, not a misplacement
    ratio = ctl["max_abs_logit_diff"] / max(chk["max_abs_logit_diff"], 1e-30)
    ok = chk["within"] or ratio >= RATIO_RULE
    if not (ok and chk["codes_in_slot"]) or ctl["within"]:
        raise SystemExit(f"cache check failed: {chk} / control {ctl} (ratio {ratio:.3g})")
    return {"check": chk, "control": ctl, "control_ratio": ratio, "by": "z8" if chk["within"] else "ratio_1000"}


def write(path: Path, arrays: dict, man: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path.with_suffix(".npz"))
    path.with_suffix(".json").write_text(json.dumps(man, indent=1, default=str) + "\n")


def gpu_name(expect):
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU")
    name = torch.cuda.get_device_name(0)
    if expect not in name:
        raise SystemExit(f"CUDA sees {name!r}, expected {expect!r}")
    return name


@torch.no_grad()
def gen1(a, scenes, rows, corp, model, gpu, rec):
    from eo.eval import seg as SG
    rows = alternating(rows, corp, N1) if a.n is None else rows[:a.n]
    if a.shard is not None:
        rows = rows[a.shard * SHARD:(a.shard + 1) * SHARD]
    sub = ("part1" if a.n is None else f"part1_n{a.n}") + ("" if a.shard is None else f"/shard{a.shard}")
    for target in GEN1[a.model]:
        off, size, _ = slot_of(scenes, target)
        tr = np.stack([truth(scenes, r, target).numpy() for r in rows]).astype(np.uint16)
        for src in (a.sources or SOURCES1):
            name = f"{a.model}_{target.lower()}_{src}"
            path = OUT / sub / name
            if path.with_suffix(".json").exists():
                print(f"[skip] {name}", flush=True)
                continue
            k = a.k
            codes = np.zeros((len(rows), k, 196), np.uint16)
            mass = np.zeros((len(rows), k), np.float32)
            t0 = time.time()
            cache = None
            for s in range(0, len(rows), a.batch):
                rs = rows[s:s + a.batch]
                ids = torch.stack([prompt(scenes, r, sources_of(scenes, r, src), target) for r in rs]).cuda()
                pos = torch.arange(ids.shape[1], device="cuda").expand(len(rs), -1)
                if cache is None:
                    cache = check_cache(model, ids[:8], (off, size))
                for j in range(k):
                    c, m = SG.generate(model, ids, pos, (off, size), 196, greedy=False, temperature=1.0,
                                       seed=SEED * 1_000_003 + j * 7919 + s)
                    codes[s:s + len(rs), j] = c.cpu().numpy().astype(np.uint16)
                    mass[s:s + len(rs), j] = m.cpu().numpy()
                if (s // a.batch) % 20 == 0:
                    print(f"  {name} {s + len(rs)}/{len(rows)} ({time.time() - t0:.0f} s)", flush=True)
            acc = float((codes == tr[:, None]).mean())
            man = {"model": a.model, "weights": str(BASES[a.model]), "load": rec, "target": target, "source": src,
                   "rows": len(rows), "K": k, "temperature": 1.0, "seed_rule": "SEED*1000003 + j*7919 + batch start",
                   "SEED": SEED, "batch": a.batch, "precision": "fp32", "gpu": gpu, "cache": cache,
                   "slot": [off, size], "token_accuracy": acc, "slot_mass_mean": float(mass.mean()),
                   "seconds": round(time.time() - t0, 1)}
            write(path, {"rows": rows, "corpus": corp[rows].astype(str), "codes": codes, "slot_mass": mass,
                         "truth": tr}, man)
            print(f"{name}: token acc {acc:.4f}, slot mass {mass.mean():.4f}, {man['seconds']} s", flush=True)


def alternating(rows, corp, n):
    """The first n of `rows` alternating corpora, sorted (tm_compare_gen.pair_rows' rule)."""
    rows = sorted(int(r) for r in rows)
    by = {c: [r for r in rows if corp[r] == c] for c in ("majortom", "ssl4eos12")}
    seq = [by[c][i] for i in range(max(map(len, by.values()))) for c in ("majortom", "ssl4eos12") if i < len(by[c])]
    return np.asarray(sorted(seq[:n]), np.int64)


def rows3(rows, corp, target, scenes):
    """The first N3 held-out rows carrying `target`, alternating corpora, sorted."""
    return alternating([r for r in rows if target in scenes.base.present_modalities(int(r))], corp, N3)


@torch.no_grad()
def gen3(a, scenes, rows, corp, model, gpu, rec):
    from eo.generate.cached import sample
    for T in (a.targets or TARGETS3):
        R = rows3(rows, corp, T, scenes)
        off, size, _ = slot_of(scenes, T)
        tr = np.stack([truth(scenes, r, T).numpy() for r in R]).astype(np.uint16)
        for src in GEN3[a.model]:
            name = f"{a.model}_{src.lower()}_{T}"
            path = OUT / "part3" / name
            if path.with_suffix(".json").exists():
                print(f"[skip] {name}", flush=True)
                continue
            names = [] if src == "none" else [src]
            prompts = torch.stack([prompt(scenes, r, names, T) for r in R]).cuda()
            cache = check_cache(model, prompts[:8], (off, size))
            arrays = {"rows": R, "corpus": corp[R].astype(str), "truth": tr}
            man = {"model": a.model, "weights": str(BASES[a.model]), "load": rec, "target": T, "source": src,
                   "rows": len(R), "precision": "fp32", "gpu": gpu, "cache": cache, "samplers": {}}
            for sname, top_p in SAMPLERS3.items():
                t0, toks = time.time(), []
                for b, i in enumerate(range(0, len(R), BATCH)):
                    x, _ = sample(model, prompts[i:i + BATCH], T, temperature=1.0, top_p=top_p, seed=SEED3 + b,
                                  slot_range=(off, size))
                    toks.append(x.cpu())
                tok = torch.cat(toks).numpy().astype(np.uint16)
                arrays[f"{sname}|tokens"] = tok
                man["samplers"][sname] = {"temperature": 1.0, "top_p": top_p, "seed_base": SEED3, "batch": BATCH,
                                          "token_accuracy": float((tok == tr).mean()),
                                          "seconds": round(time.time() - t0, 1)}
            write(path, arrays, man)
            print(f"{name}: " + ", ".join(f"{k} acc {v['token_accuracy']:.4f}" for k, v in man["samplers"].items()),
                  flush=True)


@torch.no_grad()
def tf3(a, scenes, rows, corp, model, gpu, rec):
    """CE of every target's 196 tokens given each source, per held-out row (the row's own S1 for 'S1')."""
    rows = rows if a.n is None else rows[:a.n]
    path = OUT / ("tf3" if a.n is None else f"tf3_n{a.n}") / a.model
    if path.with_suffix(".json").exists():
        print(f"[skip] tf3 {a.model}", flush=True)
        return
    targets = ("S2L2A", "S1", "DEM", "NDVI")
    out, t0 = {}, time.time()
    part = path.parent / f"partial_{a.model}"                  # one file per (source, target): a stall costs one pass
    part.mkdir(parents=True, exist_ok=True)
    for src in GEN3[a.model]:
        for T in targets:
            pf = part / f"{src.lower()}_{T}.npz"
            if pf.exists():
                z = np.load(pf)
                if not np.array_equal(z["rows"], rows):
                    raise SystemExit(f"{pf}: rows differ")
                out[f"{src.lower()}|{T}"] = z["ce"]
                print(f"tf3 {a.model} {src:>4s} -> {T:5s}: CE {z['ce'].mean():.4f} (kept from {pf.name})", flush=True)
                continue
            ce = np.zeros(len(rows), np.float64)
            for s in range(0, len(rows), a.batch):
                rs = rows[s:s + a.batch]
                names = [] if src == "none" else [src]
                seqs = []
                for r in rs:
                    tgt = s1_of(scenes, r) if T == "S1" else T
                    seqs.append(torch.cat([scenes.chunk(int(r), n) for n in names] + [scenes.chunk(int(r), tgt)]))
                ids = torch.stack(seqs).cuda()
                L = ids.shape[1]
                # the target's 196 codes sit at L-197 .. L-2 (BO before them, EO last); logits at p predict p + 1,
                # so the last 198 positions' logits (L-198 .. L-1) hold the predictions in their first 196
                pred = model(input_ids=ids, logits_to_keep=198).logits[:, :196].float()
                lab = ids[:, L - 197:L - 1]
                c = torch.nn.functional.cross_entropy(pred.transpose(1, 2), lab, reduction="none").mean(1)
                ce[s:s + len(rs)] = c.double().cpu().numpy()
            out[f"{src.lower()}|{T}"] = ce
            np.savez(pf.with_suffix(".tmp.npz"), rows=rows, ce=ce)
            os.replace(pf.with_suffix(".tmp.npz"), pf)
            print(f"tf3 {a.model} {src:>4s} -> {T:5s}: CE {ce.mean():.4f} ({time.time() - t0:.0f} s)", flush=True)
    man = {"model": a.model, "weights": str(BASES[a.model]), "load": rec, "rows": len(rows), "precision": "fp32",
           "gpu": gpu, "targets": targets, "sources": GEN3[a.model], "means": {k: float(v.mean()) for k, v in out.items()},
           "seconds": round(time.time() - t0, 1)}
    write(path, {"rows": rows, "corpus": corp[rows].astype(str), **out}, man)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gen1", "gen3", "tf3"])
    ap.add_argument("--model", required=True, choices=sorted(BASES))
    ap.add_argument("--sources", nargs="*", choices=SOURCES1)
    ap.add_argument("--targets", nargs="*", choices=TARGETS3)
    ap.add_argument("--n", type=int, default=None, help="first n held-out rows (a timing smoke; its own directory)")
    ap.add_argument("--shard", type=int, choices=range(N1 // SHARD), help="part 1: rows [256 i, 256 (i + 1)) only")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--expect-gpu", default="TITAN X")
    a = ap.parse_args()
    gpu = gpu_name(a.expect_gpu)
    scenes, rows, corp = setup()
    model, rec = load_model(a.model, "cuda")
    print(f"{a.model}: {rec['tensors']} tensors from {BASES[a.model]}; {len(rows)} held-out rows; {gpu}", flush=True)
    {"gen1": gen1, "gen3": gen3, "tf3": tf3}[a.cmd](a, scenes, rows, corp, model, gpu, rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
