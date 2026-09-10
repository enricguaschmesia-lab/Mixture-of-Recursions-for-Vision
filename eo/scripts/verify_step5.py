#!/usr/bin/env python
"""Phase 1 Step 5 -- population statistics and stratified reconstruction.

verify_step4.py already covers artifact integrity (shapes/dtypes/ranges,
presence, row alignment) -- PHASE1_PLAN.md's 5.1 and 5.2. This script does
what that one explicitly leaves to Step 5:

  5.3  code-usage histogram, over the full 89,088-row population per modality
  5.4  spot reconstruction, STRATIFIED BY |z| (scene-mean z-score) for the
       five decoder-clamped modalities, plus a plain random check for LULC
       (ViT decoder, no clamp, no z -- see contract.py)

Constraints this script must respect, established in Steps 3-4
(worklog.md 2026-09-09 "Next"):
  1. Stratify 5.4 by |z| -- the DiVAE decoders clamp output to [-1,1] in
     standardized space, so |z| >~ 1 scenes reconstruct badly BY DESIGN.
     Sampling uniformly at random would misreport that as a defect.
  2. DEM's code-usage histogram is expected to be heavily skewed (real
     property: DEM is smooth/low-entropy) -- not evidence of a broken
     preprocessing contract.
  3. Cross-modality presence must be read per-corpus (S1RTC/S1GRD are exact
     complements, not a shared set) -- already covered by verify_step4.py
     section 2, not repeated here.

Output: figures + a results table under /data/enric/reports/phase1_step5/
(never under /home -- CLAUDE.md). Takes ~10-15 min on the TITAN V.
"""
import json, os, sys, warnings, pathlib
os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T

VAL = pathlib.Path("/data/enric/data/TerraMesh/val")
OUT = pathlib.Path("/data/enric/reports/phase1_step5")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
Z_MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI"]   # standardized, clamp-affected
index = pd.read_parquet(VAL / "tok_index.parquet")

results = {}

# =========================================================================
# 5.3 -- code-usage histograms, full population
# =========================================================================
print("=== 5.3 code-usage histograms (full population) ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
usage_summary = []
for ax, mod in zip(axes.flat, MODS):
    t = np.load(VAL / f"{mod}_tok/tokens.npy", mmap_mode="r")
    p = np.load(VAL / f"{mod}_tok/present.npy")
    vals = np.asarray(t[p]).ravel()
    counts = np.bincount(vals, minlength=C.CODEBOOK[mod])
    used = int((counts > 0).sum())
    usage_summary.append(dict(modality=mod, codebook=C.CODEBOOK[mod],
                               codes_used=used,
                               pct_used=round(100 * used / C.CODEBOOK[mod], 1),
                               n_scenes=int(p.sum())))
    ranked = np.sort(counts)[::-1]
    ax.plot(ranked)
    ax.set_yscale("log")
    ax.set_title(f"{mod}  ({used}/{C.CODEBOOK[mod]} codes used, "
                 f"{100*used/C.CODEBOOK[mod]:.1f}%)")
    ax.set_xlabel("code rank")
    ax.set_ylabel("count (log)")
    print(f"  {mod:6s} {used:6d}/{C.CODEBOOK[mod]:6d} codes used "
          f"({100*used/C.CODEBOOK[mod]:5.1f}%) over {int(p.sum())} scenes")
plt.tight_layout()
fig.savefig(OUT / "5.3_code_usage_histograms.png", dpi=130)
plt.close(fig)
results["usage"] = usage_summary
print(f"  -> {OUT / '5.3_code_usage_histograms.png'}")

# =========================================================================
# 5.4 -- stratified spot reconstruction
# =========================================================================
print()
print("=== 5.4 stratified spot reconstruction ===")


def scene_z(mod, arr):
    """Scalar |z| of a scene: z-score of its spatial mean, matching the
    metric established in eo/experiments/s36d_clip.py."""
    o, c = C.CROP_OFF, C.CROP
    a = arr[0] if arr.ndim == 4 else arr
    ref = a[:, o:o + c, o:o + c].astype(np.float32)
    ref = np.where(np.isnan(ref), np.nan, ref)
    mean = np.array(C.V1_TOK_MEAN[mod])
    std = np.array(C.V1_TOK_STD[mod])
    per_ch_mean = np.nanmean(ref, axis=(1, 2))
    z = (per_ch_mean - mean) / std
    return float(np.nanmean(np.abs(z)))


def candidate_shards(mod, n_shards=2):
    corpus = "ssl4eos12" if mod == "S1GRD" else (
        "majortom" if mod == "S1RTC" else None)
    sub = index if corpus is None else index[index.corpus == corpus]
    shards = sub.source_shard.unique()
    return list(RNG.choice(shards, size=min(n_shards, len(shards)), replace=False))


recon_summary = []
for mod in Z_MODS:
    print(f"  -- {mod} --")
    shards = candidate_shards(mod)
    cand = []
    for shard in shards:
        for stem, arr in tio.iter_shard(mod, shard):
            try:
                z = scene_z(mod, arr)
            except Exception:
                continue
            if np.isnan(z):
                continue
            cand.append((z, shard, stem))
    cand.sort(key=lambda r: r[0])
    n = len(cand)
    if n == 0:
        print("    no candidates found, skipping")
        continue
    # three |z| bins: [0, 0.5), [0.5, 1.0), [1.0, inf) -- clamp boundary at 1
    bins = {"low |z|<0.5": [c for c in cand if c[0] < 0.5],
            "mid 0.5<=|z|<1.0": [c for c in cand if 0.5 <= c[0] < 1.0],
            "high |z|>=1.0": [c for c in cand if c[0] >= 1.0]}
    picks = []
    for name, pool in bins.items():
        k = min(3, len(pool))
        if k:
            sel = list(RNG.choice(len(pool), size=k, replace=False))
            picks += [(name,) + pool[i] for i in sel]
    print(f"    {n} candidates scanned; bin sizes "
          f"{[(k, len(v)) for k, v in bins.items()]}; picked {len(picks)}")

    tok = T.build(mod)
    rows = []
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * max(len(picks), 1)))
    if len(picks) == 1:
        axes = axes[None, :]
    for i, (binname, z, shard, stem) in enumerate(picks):
        _, arr = tio.read_sample(mod, shard, stem)
        x, info = P.prepare(arr, mod, device=T.DEVICE)
        tokens = T.encode(tok, x)
        recon = T.decode(tok, tokens)[0].float().cpu()
        ref = P.center_crop(arr[0] if arr.ndim == 4 else arr).astype(np.float32)
        recon_phys = P.destandardize(recon, mod).numpy()
        # first channel only for display / RMSE, consistent with s36d_clip.py
        ref0, rec0 = ref[0], recon_phys[0]
        valid = ~np.isnan(ref0)
        rmse = float(np.sqrt(np.mean((ref0[valid] - rec0[valid]) ** 2)))
        rows.append(dict(modality=mod, bin=binname, z=round(z, 3),
                          stem=stem, rmse=round(rmse, 4),
                          n_nan=int(info.get("n_nan", 0))))
        for j, (im, title) in enumerate([(ref0, "source"), (rec0, "recon"),
                                          (ref0 - rec0, "diff")]):
            ax = axes[i, j]
            vmax = np.nanpercentile(np.abs(ref0), 98)
            ax.imshow(im, cmap="RdBu_r" if j == 2 else "viridis",
                      vmin=-vmax if j == 2 else None,
                      vmax=vmax if j != 2 else vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"{binname}\n|z|={z:.2f}", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / f"5.4_{mod}_stratified_reconstruction.png", dpi=130)
    plt.close(fig)
    recon_summary += rows
    for r in rows:
        print(f"    {r['bin']:18s} |z|={r['z']:5.2f}  rmse={r['rmse']:8.3f}  {r['stem']}")

# LULC: plain random spot check, no z-stratification (categorical, ViT decoder)
print("  -- LULC (random, no z-stratification -- categorical, no clamp) --")
lulc_shards = list(RNG.choice(index.source_shard.unique(), size=1))
picks = []
for shard in lulc_shards:
    for stem, arr in tio.iter_shard("LULC", shard):
        picks.append((shard, stem))
        if len(picks) >= 8:
            break
    if len(picks) >= 8:
        break
tok = T.build("LULC")
fig, axes = plt.subplots(len(picks), 2, figsize=(6, 3 * len(picks)))
for i, (shard, stem) in enumerate(picks):
    _, arr = tio.read_sample("LULC", shard, stem)
    x, info = P.prepare(arr, "LULC", device=T.DEVICE)
    tokens = T.encode(tok, x)
    recon = T.decode(tok, tokens)[0].cpu()
    ref_cls = P.center_crop(arr[0] if arr.ndim == 4 else arr)[0]
    recon_cls = recon.argmax(0).numpy()
    acc = float((ref_cls == recon_cls).mean())
    recon_summary.append(dict(modality="LULC", bin="n/a", z=None, stem=stem,
                               rmse=None, pixel_acc=round(acc, 4)))
    for j, (im, title) in enumerate([(ref_cls, "source"), (recon_cls, "recon")]):
        ax = axes[i, j]
        ax.imshow(im, cmap="tab20", vmin=0, vmax=C.LULC_N_CLASSES - 1)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(title)
        if j == 0:
            ax.set_ylabel(f"acc={acc:.3f}", fontsize=8)
    print(f"    {stem}  pixel_acc={acc:.3f}")
plt.tight_layout()
fig.savefig(OUT / "5.4_LULC_spot_reconstruction.png", dpi=130)
plt.close(fig)

results["reconstruction"] = recon_summary
with open(OUT / "step5_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print()
print(f"Results written: {OUT / 'step5_results.json'}")
print("STEP 5 DATA COLLECTION DONE")
