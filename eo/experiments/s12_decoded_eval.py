#!/usr/bin/env python
"""Step 12, decoded (12.6b): decode the one-to-one generations, score them, draw them.

⚠ RUNS IN THE `mor` CONDA ENV (terratorch). The generations come from
eo/experiments/s12_decoded_grid.py (.venv) as LOCAL codebook grids, so nothing here
imports eo/data.

ONE DECODER SETTING PER COLUMN, from the code->value test (worklog 2026-10-02 early
morning; PHASE3_PLAN.md 12.6b), fixed before any decoded grid number existed:
  S2L2A, S1RTC, NDVI, DEM   unclamped: thresholding=False, clip_sample=False. The released
                            ±1 clamp hid model error (S1RTC: (a); NDVI leans (a); S2L2A and
                            DEM inconclusive -- flagged).
  S1GRD                     released sampler: the unclamped decoder departs from what the
                            S1GRD codes imply on exactly the scenes it changes.
  LULC                      its ViT decoder (no diffusion, no clamp).
The column's floor -- its TRUE tokens decoded with the same setting -- is scored beside it.
Decode seed 0 re-applied per batch of 4, 50 timesteps (D3.11's settings).

ERROR, per scene, against the ORIGINAL raster (224 centre crop):
  continuous  MSE in standardized (z) units, all bands, all valid pixels, every scene
              (no |z| filter: unclamped it moves the error only 0.89-1.08x).
  LULC        the share of misclassified pixels (= 5 x the MSE of the 10-class one-hot maps).
Error map for the illustrations: per-pixel RMS over bands, in σ (LULC: misclassified).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<titanx> \
      python eo/experiments/s12_decoded_eval.py decode --arms a10 b10 --wait
    python eo/experiments/s12_decoded_eval.py figures --arms a10 b10

Outputs under /data/enric/reports/decoded_grid/<T>/.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import warnings

os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                 # eo/ -> terramesh_tok
sys.path.insert(0, str(HERE.parents[1] / "scripts"))     # decode_eo
sys.path.insert(0, str(HERE.parent))                      # clamp_test

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from terramesh_tok import contract as C, preprocess as P, tokenizers as TK  # noqa: E402
import decode_eo as D  # noqa: E402
import clamp_test as CT  # noqa: E402

GEN = pathlib.Path("/data/enric/generations/s12_decoded")
OUT = pathlib.Path("/data/enric/reports/decoded_grid")
TARGETS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
SETTING = {"S2L2A": "unclamped", "S1RTC": "unclamped", "NDVI": "unclamped", "DEM": "unclamped",
           "S1GRD": "released", "LULC": "vit"}
SETTING_TEXT = {"unclamped": "DiVAE, clamp off (thresholding=False, clip_sample=False)",
                "released": "DiVAE, released sampler (±1 clamp on)",
                "vit": "ViT decoder (no diffusion, no clamp)"}
BATCH, SEED, STEPS = 4, 0, 50
ERR_VMAX = 2.0     # σ, the top of every continuous error map
LULC_NAMES = ["no data", "water", "trees", "flooded veg.", "crops", "built", "bare", "snow/ice",
              "clouds", "rangeland"]
LULC_CMAP = ListedColormap(["#d9d9d9", "#419bdf", "#397d49", "#7a87c6", "#e49635", "#c4281b",
                            "#a59b8f", "#a8ebff", "#616161", "#e3e2c3"])
INDEX = None


def corpus_of(rows):
    global INDEX
    if INDEX is None:
        INDEX = pd.read_parquet(D.VAL / "tok_index.parquet").set_index("row")
    return [str(INDEX.loc[int(r), "corpus"]) for r in rows]


@torch.no_grad()
def decode(tok, target, grids, sched):
    """(N,14,14) local ids -> (N,C,224,224) decoder output; seed re-applied per batch."""
    outs = []
    for i in range(0, len(grids), BATCH):
        g = torch.from_numpy(np.asarray(grids[i:i + BATCH], dtype=np.int64)).to(TK.DEVICE)
        if target == "LULC":
            outs.append(TK.decode(tok, g, timesteps=STEPS, seed=SEED).float().cpu())
            continue
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        kw = {} if sched is None else {"scheduler": sched}
        outs.append(tok.decode_tokens(g, timesteps=STEPS, **kw).float().cpu())
    return torch.cat(outs)


def score(target, out, refs):
    """Per-scene error, its 7x7-multilooked variant, and the per-pixel error map.

    ⚠ PLAIN PER-PIXEL MSE IS DOMINATED BY SPECKLE ON SAR (decode_eo.continuous_metrics,
    measured 2026-09-24: S1RTC's shuffle control collapses only 1.20x unfiltered). A faithful
    decode reproduces speckle that cannot match the real speckle, so it can score WORSE than a
    bland one (smoke 2026-10-06: S1GRD true tokens decoded 1.46 vs generations 1.17-1.33). The
    `ml7` variant applies the project's fixed 7x7 multilook (decode_eo.ML_WINDOW, = D3.11's
    rmse_z_ml7 squared) to both images first. LULC has no multilook variant (same number)."""
    errs, errs_ml, maps = [], [], []
    for i, ref in enumerate(refs):
        if target == "LULC":
            wrong = out[i].argmax(0).numpy() != ref[0].astype(np.int64)
            errs.append(float(wrong.mean())); errs_ml.append(float(wrong.mean()))
            maps.append(wrong.astype(np.float32))
            continue
        mean = np.array(C.V1_TOK_MEAN[target], dtype=np.float32)[:, None, None]
        std = np.array(C.V1_TOK_STD[target], dtype=np.float32)[:, None, None]
        ref_z = (ref - mean) / std
        rec = out[i].numpy()
        sq = (ref_z - rec) ** 2
        ok = ~np.isnan(sq)
        errs.append(float(sq[ok].mean()))
        win = (1, D.ML_WINDOW, D.ML_WINDOW)
        ref_ml = D.uniform_filter(np.where(np.isnan(ref_z), 0.0, ref_z), size=win)
        rec_ml = D.uniform_filter(rec, size=win)
        errs_ml.append(float(((ref_ml - rec_ml)[ok] ** 2).mean()))
        with np.errstate(invalid="ignore"):
            maps.append(np.sqrt(np.nanmean(sq, axis=0)).astype(np.float32))
    return np.array(errs), np.array(errs_ml), maps


def display(target, x_phys):
    """A decoded or reference raster (C,H,W, physical units) -> an image to show."""
    if target == "S2L2A":
        return np.stack([x_phys[3], x_phys[2], x_phys[1]], -1)      # B04, B03, B02
    if target == "LULC":
        return x_phys
    return x_phys[0]                                                   # VV, DEM, NDVI


def decode_one(target, arm, sched_cache, tok, refs, R, illus_rows):
    gdir = GEN / arm / target
    z = np.load(gdir / "cells.npz")
    if not np.array_equal(z["column_rows"], R):
        raise SystemExit(f"{arm} {target}: column rows differ from the reference arm's")
    man = json.loads((gdir / "manifest.json").read_text())
    pos = {int(r): i for i, r in enumerate(R)}
    arrays, illus, t0 = {}, {}, time.perf_counter()
    for S in man["cells"]:
        rows, grids = z[f"{S}|rows"], z[f"{S}|grids"]
        out = decode(tok, target, grids, sched_cache)
        e, eml, maps = score(target, out, [refs[pos[int(r)]] for r in rows])
        arrays[f"{arm}|{S}|rows"], arrays[f"{arm}|{S}|err"] = rows, e
        arrays[f"{arm}|{S}|err_ml7"] = eml
        arrays[f"{arm}|{S}|valid_frac"] = z[f"{S}|valid"].mean(1)
        for k, r in enumerate(rows):
            if int(r) in illus_rows:
                ph = out[k] if target == "LULC" else P.destandardize(out[k], target)
                img = ph.argmax(0).numpy() if target == "LULC" else display(target, ph.numpy())
                illus[f"{arm}|{S}|{int(r)}|img"] = img.astype(np.float32)
                illus[f"{arm}|{S}|{int(r)}|err"] = maps[k]
                illus[f"{arm}|{S}|{int(r)}|mse"] = np.float32(e[k])
        print(f"  {arm} {S:>6}->{target:<5} n {len(rows):3d}  error {e.mean():.4f}  ml7 {eml.mean():.4f}  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return arrays, illus


def illus_rows_for(target, R):
    """Candidate illustration scenes, by rule: the column's first 3 majortom and first 3
    ssl4eos12 rows. A cell uses the first candidate it contains (ssl4eos12 first for an S1GRD
    source, majortom first otherwise), so no scene is chosen by looking at a result."""
    cor = corpus_of(R)
    return {"majortom": [int(r) for r, c in zip(R, cor) if c == "majortom"][:3],
            "ssl4eos12": [int(r) for r, c in zip(R, cor) if c == "ssl4eos12"][:3]}


def pick_row(pick, S, cell_rows):
    order = (pick["ssl4eos12"] + pick["majortom"]) if S == "S1GRD" else (pick["majortom"] + pick["ssl4eos12"])
    have = {int(r) for r in cell_rows}
    return next((r for r in order if r in have), None)


def decode_cmd(args) -> int:
    gpu = D.assert_gpu(args.allow_titanv)
    print(f"gpu {gpu}", flush=True)
    todo = [(T, a) for T in TARGETS for a in args.arms]
    while todo:
        ready = [(T, a) for T, a in todo if (GEN / a / T / "manifest.json").exists()]
        if not ready:
            if not args.wait:
                print(f"not generated yet: {todo}"); return 1
            time.sleep(120); continue
        T, arm = ready[0]
        odir = OUT / T
        odir.mkdir(parents=True, exist_ok=True)
        if (odir / f"errors_{arm}.npz").exists():
            todo.remove((T, arm)); continue
        R = np.load(GEN / arm / T / "cells.npz")["column_rows"]
        pick = illus_rows_for(T, R)
        illus_rows = set(pick["majortom"]) | set(pick["ssl4eos12"])
        tok = TK.build(T, device=TK.DEVICE)
        sched = CT.scheduler_like(tok, **CT.VARIANTS["noclip"]) if SETTING[T] == "unclamped" else None
        refs = [P.center_crop(a[0] if a.ndim == 4 else a).astype(np.float32)
                for a in D.read_rasters(T, R)]
        print(f"{T}: {len(R)} rows, decoder {SETTING[T]}, illustration rows {pick}", flush=True)
        if not (odir / "errors_ceiling.npz").exists():
            truth = np.asarray(np.load(D.VAL / C.tok_dir_name(T) / "tokens.npy", mmap_mode="r")[R]
                               ).astype(np.int64).reshape(-1, C.GRID, C.GRID)
            out = decode(tok, T, truth, sched)
            e, eml, maps = score(T, out, refs)
            cil = {}
            for k, r in enumerate(R):
                if int(r) in illus_rows:
                    ph = out[k] if T == "LULC" else P.destandardize(out[k], T)
                    cil[f"ceiling|{int(r)}|img"] = (ph.argmax(0).numpy() if T == "LULC"
                                                    else display(T, ph.numpy())).astype(np.float32)
                    cil[f"ceiling|{int(r)}|err"] = maps[k]
                    cil[f"ceiling|{int(r)}|mse"] = np.float32(e[k])
                    gt = refs[k]
                    cil[f"truth|{int(r)}|img"] = (gt[0] if T == "LULC" else display(T, gt)).astype(np.float32)
            np.savez(odir / "illus_ceiling.npz", **cil)
            np.savez(odir / "errors_ceiling.npz", **{f"ceiling|{T}|rows": R, f"ceiling|{T}|err": e,
                                                     f"ceiling|{T}|err_ml7": eml})
            print(f"  ceiling {T}: error {e.mean():.4f}  ml7 {eml.mean():.4f}", flush=True)
        arrays, illus = decode_one(T, arm, sched, tok, refs, R, illus_rows)
        np.savez(odir / f"illus_{arm}.npz", **illus)
        tmp = odir / f"errors_{arm}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, odir / f"errors_{arm}.npz")
        todo.remove((T, arm))
    print("decode done", flush=True)
    return 0


def source_image(src, row):
    """The conditioning modality of an illustration row, for display (None for Coords / ∅)."""
    if src in ("Coords", "none"):
        return None, src
    a = D.read_rasters(src, np.array([row]))[0]
    ref = P.center_crop(a[0] if a.ndim == 4 else a).astype(np.float32)
    return (ref[0] if src == "LULC" else display(src, ref)), src


def show(ax, target, img, ref_img=None):
    if target == "LULC":
        ax.imshow(img, cmap=LULC_CMAP, vmin=0, vmax=9, interpolation="nearest")
        return
    if target == "S2L2A":
        lo = np.nanpercentile(ref_img, 2, axis=(0, 1)); hi = np.nanpercentile(ref_img, 98, axis=(0, 1))
        ax.imshow(np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1))
        return
    lo, hi = np.nanpercentile(ref_img, 2), np.nanpercentile(ref_img, 98)
    cmap = {"DEM": "terrain", "NDVI": "RdYlGn"}.get(target, "gray")
    ax.imshow(img, cmap=cmap, vmin=lo, vmax=max(hi, lo + 1e-6))


def figures_cmd(args) -> int:
    A, B = args.arms
    summary = {}
    for T in TARGETS:
        odir = OUT / T
        parts = [np.load(odir / f) for f in ("errors_ceiling.npz",) + tuple(f"errors_{a}.npz" for a in args.arms)]
        merged = {k: p[k] for p in parts for k in p.files}
        np.savez(odir / "errors.npz", **merged)
        il = {k: v for f in ("illus_ceiling.npz",) + tuple(f"illus_{a}.npz" for a in args.arms)
              for z in [np.load(odir / f)] for k, v in ((k, z[k]) for k in z.files)}
        R = merged[f"ceiling|{T}|rows"]
        pick = illus_rows_for(T, R)
        cells = [k.split("|")[1] for k in merged if k.startswith(f"{A}|") and k.endswith("|err")]
        means = {S: {a: float(merged[f"{a}|{S}|err"].mean()) for a in args.arms} for S in cells}
        means_ml7 = {S: {a: float(merged[f"{a}|{S}|err_ml7"].mean()) for a in args.arms} for S in cells}
        unit = "misclassified pixel share" if T == "LULC" else "MSE (z units²)"
        doc = {"target": T, "unit": unit, "decoder_setting": SETTING_TEXT[SETTING[T]],
               "ceiling": float(merged[f"ceiling|{T}|err"].mean()), "cells": means,
               "ceiling_ml7": float(merged[f"ceiling|{T}|err_ml7"].mean()), "cells_ml7": means_ml7,
               "illustration_rows": pick, "decode": {"seed": SEED, "batch": BATCH, "timesteps": STEPS}}
        (odir / "decoded.json").write_text(json.dumps(doc, indent=1) + "\n")
        summary[T] = doc

        # one figure per target column, slide-shaped: each figure COLUMN is one cell (the
        # tokenizer floor first), each figure ROW one view of it, both arms
        cols_fig = ["ceiling"] + cells
        row_names = ["source given", "ground truth", f"{A} decoded", f"{A} error",
                     f"{B} decoded", f"{B} error"]
        fig, axs = plt.subplots(len(row_names), len(cols_fig),
                                figsize=(1.95 * len(cols_fig) + 0.7, 1.95 * len(row_names) + 1.1))
        fig.patch.set_facecolor("#fcfcfb")
        for j, S in enumerate(cols_fig):
            r = pick_row(pick, "S2L2A" if S == "ceiling" else S,
                         R if S == "ceiling" else merged[f"{A}|{S}|rows"])
            ref_img = il[f"truth|{r}|img"]
            if S == "ceiling":
                panels = [(None, "true tokens\n(no model)"), (ref_img, f"row {r}"),
                          (il[f"ceiling|{r}|img"], f"{il[f'ceiling|{r}|mse']:.3f}"),
                          (il[f"ceiling|{r}|err"], "err"), (None, ""), (None, "")]
                head = f"tokenizer floor\nmean {doc['ceiling']:.3f}"
            else:
                src_img, _ = source_image(S, r)
                panels = [(src_img, "∅: no source" if S == "none" else ("coordinates" if S == "Coords" else "")),
                          (ref_img, f"row {r}")]
                for a in args.arms:
                    panels += [(il[f"{a}|{S}|{r}|img"], f"{il[f'{a}|{S}|{r}|mse']:.3f}"),
                               (il[f"{a}|{S}|{r}|err"], "err")]
                head = (f"{'∅' if S == 'none' else S} → {T}\n"
                        + " / ".join(f"{means[S][a]:.3f}" for a in args.arms))
            for i, (img, title) in enumerate(panels):
                ax = axs[i, j]
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if img is None:
                    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=8.5,
                            color="#52514e", transform=ax.transAxes)
                    continue
                if title == "err":
                    if T == "LULC":
                        ax.imshow(img, cmap=ListedColormap(["#ffffff", "#d03b3b"]), vmin=0, vmax=1,
                                  interpolation="nearest")
                    else:
                        ax.imshow(img, cmap="Reds", vmin=0, vmax=ERR_VMAX)
                    continue
                if i == 0:
                    show(ax, S, img, img)
                else:
                    show(ax, T, img, ref_img)
                if title and i > 0:
                    ax.text(0.03, 0.04, title, transform=ax.transAxes, fontsize=7.5, color="#0b0b0b",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffffcc", ec="none"))
                elif title:
                    ax.set_title(title, fontsize=7.5, color="#52514e")
            axs[0, j].set_title(head, fontsize=8.5, color="#0b0b0b", pad=6)
        for i, nm in enumerate(row_names):
            axs[i, 0].set_ylabel(nm, fontsize=9, color="#0b0b0b")
        fig.suptitle(f"{T}: every one-to-one generation, decoded ({SETTING_TEXT[SETTING[T]]}). "
                     f"Column heads: the cell's mean {unit} over its scenes, {A} / {B}; "
                     f"panel labels: this scene's.", fontsize=9.5, x=0.01, ha="left")
        foot = ("Error rows: misclassified pixels in red." if T == "LULC" else
                f"Error rows: per-pixel RMS error over bands, in σ, white 0 → dark red ≥ {ERR_VMAX:g}.")
        fig.text(0.01, 0.002, foot + " Scenes chosen by rule (the first of the column's rows that the "
                 "cell contains, majortom first; ssl4eos12 first for an S1GRD source), not by looking.",
                 fontsize=8.5, color="#52514e")
        plt.tight_layout(rect=(0, 0.015, 1, 0.975))
        fig.savefig(odir / f"fig_decoded_{T}.png", dpi=110, facecolor="#fcfcfb")
        plt.close(fig)
        print(f"{T}: ceiling {doc['ceiling']:.4f}  " + "  ".join(
            f"{S} " + "/".join(f"{means[S][a]:.3f}" for a in args.arms) for S in cells))
    (OUT / "decoded_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"-> {OUT}")
    return 0


def main() -> int:
    global GEN, OUT, TARGETS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--arms", nargs="+", default=["a10", "b10"])
    d.add_argument("--wait", action="store_true", help="poll until each (target, arm) is generated")
    d.add_argument("--allow-titanv", action="store_true")
    f = sub.add_parser("figures")
    f.add_argument("--arms", nargs=2, default=["a10", "b10"])
    for p in (d, f):
        p.add_argument("--gen-root", default=str(GEN))
        p.add_argument("--out-root", default=str(OUT))
        p.add_argument("--targets", nargs="*", default=None, help="smoke tests only")
    args = ap.parse_args()
    GEN, OUT = pathlib.Path(args.gen_root), pathlib.Path(args.out_root)
    TARGETS = args.targets or TARGETS
    return decode_cmd(args) if args.cmd == "decode" else figures_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
