#!/usr/bin/env python
"""Our models against TerraMind on its own generation tasks: TERRAMIND'S half, decoding, scoring.

⚠ RUNS IN THE `mor` CONDA ENV (terratorch). Our models' tokens come from
eo/experiments/tm_compare_gen.py (.venv), which also fixes the rows (rows.json).

  terramind  TerraMind v1-B and v1-L (terratorch's terramind_v1_{base,large}_generate) generate
             each pair's target on the same rows (TM_RUNS):
               pixels  the source's pixels, its native input and Table 3's (standardized with our
                       contract's stats, which equal its own -- checked at start; NaN filled as
                       our tokenizers see it);
               tokens  the source as TOKENS, Table 14's setting and the input our models get:
                       TerraMind tokenizes the same standardized pixels itself, and those tokens
                       are checked against our stored ones.
             Its sampler at terratorch's defaults (temperature 1.0, top-p 0.8, decoding_steps 1;
             the paper states none). Its tokens are captured where it would decode them, so
             every model's tokens go through one decoder below.
  decode     every condition's tokens -- each model and sampler, and the TRUE tokens (the
             ceiling) -- decoded with TerraMind's protocol: the released sampler, STEPS = 10
             diffusion steps (Table 3), seed 0 re-applied per batch, so every condition shares
             the decoder's noise scene by scene. Saved (float16, standardized) for rescoring.
  score      against the ORIGINAL raster (224 centre crop; the cleaned S2L2A, as the tokens
             encode it), in Table 3's physical units: S2L2A reflectance (DN / 10,000), S1 dB,
             DEM m, NDVI. MAE and RMSE over every valid pixel and band; PSNR from that pooled
             MSE and SSIM per scene, both with the data range R = max - min of the pair's
             reference pixels. TerraMind states neither; every PSNR in its Tables 3, 10 and 14
             equals 20 log10(R / RMSE) with one R per evaluated set (torchmetrics' PSNR with
             data_range=None), and a commented-out line of its source says SSIM and PSNR "are
             calibrated on the entire range of input data". Paired bootstrap over scenes for
             the differences, and the wrong-scene control (the ceiling of the row half the pair's list away,
             scored against this row): what a sharp but unrelated image scores.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<titanx> \
      python eo/experiments/tm_compare_eval.py terramind
    ... decode [--pairs S-T ...] [--allow-titanv], then score (CPU)

The TITAN V decodes the same images as the Titan X (max |diff| 0.0005 σ, the float16 storage;
2026-11-03) at ~4x the speed; run it there only under the power watchdog
(/data/enric/logs/tm_compare/run_decode_titanv.sh). TerraMind's top-p sorts its 15,360-way
logits: generate on the Titan X.

Outputs: tokens and decoded images under /data/enric/generations/tm_compare/, the report
under /data/enric/reports/tm_compare/.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import random
import sys
import time
import warnings

os.environ.setdefault("HF_HOME", "/data/enric/hf")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                 # eo/ -> terramesh_tok
sys.path.insert(0, str(HERE.parents[1] / "scripts"))     # decode_eo

import numpy as np  # noqa: E402
import torch  # noqa: E402

from terramesh_tok import contract as C, preprocess as P, tokenizers as TK  # noqa: E402
import decode_eo as D  # noqa: E402

GEN = pathlib.Path(os.environ.get("TMC_GEN", "/data/enric/generations/tm_compare"))     # override: smoke tests
REPORT = pathlib.Path(os.environ.get("TMC_REPORT", "/data/enric/reports/tm_compare"))
#: condition -> (terratorch model, input)
TM_RUNS = {"tm_base": ("terramind_v1_base_generate", "pixels"),
           "tm_large": ("terramind_v1_large_generate", "pixels"),
           "tm_base_tok": ("terramind_v1_base_generate", "tokens")}
TOK_KEY = {"S2L2A": "tok_sen2l2a@224", "S1GRD": "tok_sen1grd@224", "S1RTC": "tok_sen1rtc@224"}
OURS = ("t360", "b10c")
SAMPLERS = ("t1", "p08")
STEPS, DECODE_SEED, BATCH, TM_BATCH, TM_SEED = 10, 0, 8, 16, 1000
#: S2L2A is scored as reflectance; everything else in the units it is stored in.
SCALE = {"S2L2A": 1e-4}
COMPARE = [("t360_p08", "tm_base"), ("t360_t1", "tm_base"), ("t360_p08", "tm_large"), ("t360_t1", "tm_large"),
           ("t360_p08", "tm_base_tok"), ("t360_t1", "tm_base_tok"), ("tm_base_tok", "tm_base"),
           ("b10c_p08", "tm_base"), ("t360_p08", "b10c_p08"), ("t360_t1", "b10c_t1"), ("t360_t1", "t360_p08"),
           ("tm_large", "tm_base")]


def spec():
    s = json.loads((GEN / "rows.json").read_text())
    if s["token_set"] != "clean":
        raise SystemExit(f"rows.json: token set {s['token_set']!r}; this comparison is on the clean set")
    return s


def pairs(s, only=None):
    """[(S, T, rows)] of rows.json, or of the pairs named in `only` ("S-T")."""
    return [tuple(k.split("-")) + (np.array(v),) for k, v in s["rows"].items() if not only or k in only]


# ------------------------------------------------------------------ TerraMind
def check_stats():
    import terratorch.models.backbones.terramind.model.terramind_register as R
    key = {"S2L2A": "sen2l2a", "S1GRD": "sen1grd", "S1RTC": "sen1rtc", "DEM": "dem"}
    for mod, k in key.items():
        for pre in ("untok_", "tok_"):
            if not (np.allclose(R.v1_pretraining_mean[f"{pre}{k}@224"], C.V1_TOK_MEAN[mod]) and
                    np.allclose(R.v1_pretraining_std[f"{pre}{k}@224"], C.V1_TOK_STD[mod])):
                raise SystemExit(f"TerraMind's {pre}{k} stats differ from contract.V1_TOK_*[{mod}]")


@torch.no_grad()
def terramind_cmd(args) -> int:
    from terratorch.registry import FULL_MODEL_REGISTRY
    gpu = D.assert_gpu(False)
    check_stats()
    s = spec()
    for tag in args.models:
        model_name, inp = TM_RUNS[tag]
        odir = GEN / tag
        odir.mkdir(parents=True, exist_ok=True)
        for S, T, R in pairs(s):
            name = f"{S}-{T}"
            if (odir / f"{name}.json").exists():
                print(f"[skip] {tag} {name}", flush=True)
                continue
            t0 = time.perf_counter()
            key = S if inp == "pixels" else TOK_KEY[S]
            m = FULL_MODEL_REGISTRY.build(model_name, modalities=[key], output_modalities=[T], pretrained=True,
                                          standardize=False).to(TK.DEVICE).eval()
            cap, enc = {}, []
            for k in m.tokenizer.keys():         # keep the tokens; skip TerraMind's own decode
                def grab(tok, *a, _k=k, **kw):
                    cap[_k] = tok.detach().cpu()
                    return torch.zeros(tok.shape[0], 1, 1, 1)
                m.tokenizer[k].decode_tokens = grab
            if inp == "tokens":                  # keep the input tokens TerraMind computes, to check them
                orig = m.tokenizer[key].encode
                def keep(x, *a, _o=orig, **kw):
                    r = _o(x, *a, **kw)
                    enc.append((r if isinstance(r, torch.Tensor) else r[-1]).detach().cpu())
                    return r
                m.tokenizer[key].encode = keep
            src = D.read_rasters(S, R, "clean")
            toks = []
            for b, i in enumerate(range(0, len(R), TM_BATCH)):
                x = torch.cat([P.prepare(a, S, device="cpu")[0] for a in src[i:i + TM_BATCH]]).to(TK.DEVICE)
                random.seed(TM_SEED + b); torch.manual_seed(TM_SEED + b)
                cap.clear()
                m(x)
                (tok,) = cap.values()
                toks.append(tok.numpy().astype(np.uint16))
            tok = np.concatenate(toks)
            assert tok.shape == (len(R), C.GRID, C.GRID), tok.shape
            truth = D.read_tokens(T, R, "clean")
            man = {"model": model_name, "source": S, "target": T, "n": len(R), "gpu": gpu, "input": inp,
                   "temperature": m.temps, "top_p": m.top_p, "top_k": m.top_k,
                   "decoding_steps": m.decoding_steps, "seed_base": TM_SEED, "batch": TM_BATCH,
                   "token_accuracy": float((tok == truth).mean()), "codes_used": int(len(np.unique(tok))),
                   "seconds": round(time.perf_counter() - t0, 1)}
            if inp == "tokens":                  # TerraMind's tokens of the input = our stored ones?
                got = torch.cat(enc).numpy().reshape(len(R), C.GRID, C.GRID)
                man["input_tokens_equal_stored"] = float((got == D.read_tokens(S, R, "clean")).mean())
                if man["input_tokens_equal_stored"] < 0.999:
                    raise SystemExit(f"{tag} {name}: TerraMind's input tokens match ours only "
                                     f"{man['input_tokens_equal_stored']:.4f}")
            np.savez(odir / f"{name}.npz", rows=R, tokens=tok)
            (odir / f"{name}.json").write_text(json.dumps(man, indent=1) + "\n")
            print(f"{tag} {name:12s} n {len(R)}  token acc {man['token_accuracy']:.4f}  "
                  f"codes {man['codes_used']}  {man['seconds']}s", flush=True)
            del m, cap, enc                      # the wrapped methods hold the model in a cycle
            gc.collect()
            torch.cuda.empty_cache()
    return 0


# ------------------------------------------------------------------ decode
def conditions(S, T, R):
    """{condition: (N, 14, 14) local token grids} for one pair, every source checked against R."""
    name, out = f"{S}-{T}", {}
    truth = D.read_tokens(T, R, "clean")
    for tag in OURS:
        f = GEN / tag / f"{name}.npz"
        if not (GEN / tag / f"{name}.json").exists():
            continue
        z = np.load(f)
        if not np.array_equal(z["rows"], R):
            raise SystemExit(f"{f}: rows differ from rows.json")
        if not np.array_equal(z["truth"].astype(np.int64), truth):
            raise SystemExit(f"{f}: the prompt's truth differs from the stored clean tokens")
        for smp in SAMPLERS:
            out[f"{tag}_{smp}"] = z[f"{smp}|tokens"]
    for tag in TM_RUNS:
        f = GEN / tag / f"{name}.npz"
        if (GEN / tag / f"{name}.json").exists():
            z = np.load(f)
            if not np.array_equal(z["rows"], R):
                raise SystemExit(f"{f}: rows differ from rows.json")
            out[tag] = z["tokens"]
    out["ceiling"] = truth
    return out


@torch.no_grad()
def decode_cmd(args) -> int:
    gpu = D.assert_gpu(args.allow_titanv)
    s = spec()
    for S, T, R in pairs(s, args.pairs):
        name = f"{S}-{T}"
        tok = TK.build(T, device=TK.DEVICE)
        for cond, grids in conditions(S, T, R).items():
            f = GEN / "decoded" / cond / f"{name}.npy"
            if f.exists():
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            t0, outs = time.perf_counter(), []
            for i in range(0, len(grids), BATCH):
                g = torch.from_numpy(np.asarray(grids[i:i + BATCH], dtype=np.int64)).to(TK.DEVICE)
                outs.append(TK.decode(tok, g, timesteps=STEPS, seed=DECODE_SEED).float().cpu().numpy())
            np.save(f.with_suffix(".tmp.npy"), np.concatenate(outs).astype(np.float16))
            os.replace(f.with_suffix(".tmp.npy"), f)
            print(f"{name:12s} {cond:10s} n {len(grids)}  {time.perf_counter() - t0:.0f}s  ({gpu})", flush=True)
    (GEN / "decoded" / "protocol.json").write_text(json.dumps(
        {"steps": STEPS, "seed": DECODE_SEED, "batch": BATCH, "sampler": "released (clamp on)"}) + "\n")
    return 0


# ------------------------------------------------------------------ score
def physical(z, T):
    """(N, C, H, W) standardized -> physical units (S2L2A as reflectance)."""
    m = np.array(C.V1_TOK_MEAN[T], dtype=np.float32)[:, None, None]
    s = np.array(C.V1_TOK_STD[T], dtype=np.float32)[:, None, None]
    return (z.astype(np.float32) * s + m) * SCALE.get(T, 1.0)


def scene_metrics(pred, ref, rng):
    """Per scene: absolute and squared error SUMMED over valid pixels and bands, their count, and
    SSIM (torchmetrics, Gaussian 11 / 1.5, per band then averaged, data range `rng`). NaN reference
    pixels (the S1 fill) leave the errors; for SSIM they take the prediction's value, so they add
    no error there. Returns (N, 4): abs sum, sq sum, count, SSIM."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    out = np.zeros((len(ref), 4))
    for i, (p, r) in enumerate(zip(pred, ref)):
        ok = ~np.isnan(r)
        d = (p - r)[ok].astype(np.float64)
        rr = np.where(ok, r, p)
        sv = [float(ssim(torch.from_numpy(p[c][None, None]), torch.from_numpy(rr[c][None, None]),
                         data_range=rng)) for c in range(p.shape[0])]
        out[i] = (np.abs(d).sum(), (d ** 2).sum(), ok.sum(), np.mean(sv))
    return out


def boot_diff(a, b, n=10_000, seed=0):
    """Mean of a - b over scenes and its 95% paired bootstrap CI."""
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def score_cmd(args) -> int:
    s = spec()
    REPORT.mkdir(parents=True, exist_ok=True)
    res = {"protocol": json.loads((GEN / "decoded" / "protocol.json").read_text()),
           "units": {"S2L2A": "reflectance (DN / 10,000)", "S1GRD": "dB", "S1RTC": "dB", "DEM": "m", "NDVI": "NDVI"},
           "pairs": {}}
    for S, T, R in pairs(s):
        name = f"{S}-{T}"
        refs = np.stack([P.center_crop(a[0] if a.ndim == 4 else a).astype(np.float32)
                         for a in D.read_rasters(T, R, "clean")]) * SCALE.get(T, 1.0)
        rng = float(np.nanmax(refs) - np.nanmin(refs))   # R: the pair's reference range, all bands
        per, conds = {}, sorted(p.parent.name for p in (GEN / "decoded").glob(f"*/{name}.npy"))
        for cond in conds:
            pred = physical(np.load(GEN / "decoded" / cond / f"{name}.npy"), T)
            per[cond] = scene_metrics(pred, refs, rng)
            if cond == "ceiling":     # the wrong-scene control: the ceiling of the row half the list away (the
                # held-out rows are a geographic group split, so a neighbouring row is often the same region)
                per["wrong_scene"] = scene_metrics(np.roll(pred, len(pred) // 2, axis=0), refs, rng)
        out = {"n": len(R), "data_range": rng, "conditions": {}}
        for cond, m in per.items():
            mse = m[:, 1].sum() / m[:, 2].sum()
            out["conditions"][cond] = {
                "mae": float(m[:, 0].sum() / m[:, 2].sum()), "rmse": float(np.sqrt(mse)),
                "psnr": float(10 * np.log10(rng ** 2 / mse)), "ssim": float(m[:, 3].mean()),
                "scene_mae": (m[:, 0] / m[:, 2]).tolist(), "scene_mse": (m[:, 1] / m[:, 2]).tolist(),
                "scene_ssim": m[:, 3].tolist()}
        out["paired"] = {}
        for a, b in COMPARE:
            if a in per and b in per:
                ca, cb = out["conditions"][a], out["conditions"][b]
                out["paired"][f"{a} - {b}"] = {k: boot_diff(ca[f"scene_{k}"], cb[f"scene_{k}"])
                                               for k in ("mae", "ssim")}
        res["pairs"][name] = out
        line = "  ".join(f"{c} {v['mae']:.4g}/{v['ssim']:.3f}" for c, v in out["conditions"].items())
        print(f"{name:12s} R {rng:.4g}  MAE/SSIM  {line}", flush=True)
    (REPORT / "summary.json").write_text(json.dumps(res) + "\n")
    print("->", REPORT / "summary.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("terramind")
    t.add_argument("--models", nargs="+", default=list(TM_RUNS), choices=list(TM_RUNS))
    d = sub.add_parser("decode")
    d.add_argument("--allow-titanv", action="store_true")
    d.add_argument("--pairs", nargs="*", default=None, help="S-T names (default: all), to split the work between cards")
    sub.add_parser("score")
    args = ap.parse_args()
    return {"terramind": terramind_cmd, "decode": decode_cmd, "score": score_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
