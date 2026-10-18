"""PHASE6_PLAN Step 1.3: the tokenizer contract on BigEarthNet, before scaling (`mor` env, GPU).

Rule 2: validate on a handful of samples, against controls that must visibly fail.

For each variant, a patch goes physical -> encoder input -> tokens -> decoder -> physical, and
the error is scored in the v1-standardized space of the PHYSICAL image (decode_eo's
`rmse_z`, `rmse_z_ml7`), so every variant is judged against the same reference:

  BEN S2   bilinear (production) · bicubic (the §4.2 alternative)
           controls: reversed band order · yaml stats · no standardization
  BEN S1   S1GRD tokenizer (production, §4.3) · S1RTC tokenizer
           controls: VV/VH swapped · no standardization
  TerraMesh S2L2A / S1GRD: the same pipeline on native 224 crops, as the reference range.

Also reported: the share of reference pixels with |z| < 1, because the DiVAE decoder clamps
its output there (CLAUDE.md); `rmse_z_in` is the error on those pixels only.

    python eo/experiments/ben_roundtrip.py --n 24 --out /data/enric/reports/ben/step1_roundtrip.json
    # L11b.2 (2026-10-18): the same check on the unlabelled pool's own pack, before tokenizing it
    python eo/experiments/ben_roundtrip.py --n 24 --packed <ROOT>/v1_stream_l11 --role ft_unlab \
        --out /data/enric/reports/ben/l11b_roundtrip.json

Reads packs in stream order through layout.load_packed (2026-10-18; the first version read a
row-order pack's filled_*.npy, which the stream-order pack replaced on 2026-10-12).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo"))
sys.path.insert(0, str(REPO / "eo" / "scripts"))

from eo.ben import layout, preprocess as BP  # noqa: E402
from eo.terramesh_tok import contract as C, preprocess as P, tokenizers as T  # noqa: E402
import decode_eo as D  # noqa: E402

TIMESTEPS, SEED, BATCH = 50, 0, 8


def roundtrip(tok, x_in: torch.Tensor, to_phys) -> np.ndarray:
    """encoder input -> tokens -> decoder output -> physical (via `to_phys`)."""
    outs = []
    for i in range(0, x_in.shape[0], BATCH):
        t = T.encode(tok, x_in[i:i + BATCH])
        outs.append(to_phys(T.decode(tok, t, timesteps=TIMESTEPS, seed=SEED).float()).cpu())
    return torch.cat(outs).numpy()


def score(ref_phys: np.ndarray, rec_phys: np.ndarray, mod: str) -> dict:
    mean = np.array(C.V1_TOK_MEAN[mod], np.float32)[:, None, None]
    std = np.array(C.V1_TOK_STD[mod], np.float32)[:, None, None]
    rows = []
    for ref, rec in zip(ref_phys, rec_phys):
        m = D.continuous_metrics((rec - mean) / std, rec, ref, mod)
        rz = (ref - mean) / std
        inr = np.abs(rz) < 1
        m["frac_in_range"] = float(inr.mean())
        m["rmse_z_in"] = float(np.sqrt(np.mean((rz[inr] - ((rec - mean) / std)[inr]) ** 2))) if inr.any() else None
        rows.append(m)
    df = pd.DataFrame(rows)
    return {k: {"median": float(df[k].median()), "mean": float(df[k].mean())}
            for k in ("rmse_z", "rmse_z_ml7", "rmse_z_in", "frac_in_range")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--packed", type=Path, default=layout.PACKED)
    ap.add_argument("--role", default="ft_train")
    a = ap.parse_args()
    dev = "cuda:0"
    gpu = torch.cuda.get_device_name(0)
    idx = pd.read_parquet(layout.INDEX)
    a2, r2p2 = layout.load_packed("S2", len(idx), a.packed)
    a1, r2p1 = layout.load_packed("S1", len(idx), a.packed)
    pool = np.flatnonzero(idx[a.role].values & (r2p2 >= 0) & (r2p1 >= 0))
    rows = np.sort(np.random.default_rng(0).choice(pool, a.n, replace=False))
    res = {"n": a.n, "ben_rows": rows.tolist(), "gpu": gpu, "timesteps": TIMESTEPS, "seed": SEED,
           "packed": str(a.packed), "role": a.role}

    # ---- BEN S2 -----------------------------------------------------------
    p10 = np.asarray(a2["S2_10m"][r2p2[rows]])
    p20 = np.asarray(a2["S2_20m"][r2p2[rows]])
    p60 = np.asarray(a2["S2_60m"][r2p2[rows]])
    tok = T.build("S2L2A", dev)
    s2 = {}
    for mode in ("bilinear", "bicubic"):
        ref = BP.s2_physical(p10, p20, p60, dev, mode)
        rec = roundtrip(tok, BP.s2_batch(p10, p20, p60, dev, mode=mode),
                        lambda z: BP.destandardize(z, "S2L2A"))
        s2[mode] = score(ref.cpu().numpy(), rec, "S2L2A")
    ref = BP.s2_physical(p10, p20, p60, dev, "bilinear")
    ref_np = ref.cpu().numpy()
    perm = list(range(12))[::-1]
    rec = roundtrip(tok, BP.s2_batch(p10, p20, p60, dev, band_perm=perm),
                    lambda z: BP.destandardize(z, "S2L2A")[:, perm])     # undo the permutation
    s2["ctrl_reversed_bands"] = score(ref_np, rec, "S2L2A")
    rec = roundtrip(tok, BP.s2_batch(p10, p20, p60, dev, stats="yaml"),
                    lambda z: BP.destandardize(z, "S2L2A", "yaml"))
    s2["ctrl_yaml_stats"] = score(ref_np, rec, "S2L2A")
    rec = roundtrip(tok, BP.s2_batch(p10, p20, p60, dev, standardize=False), lambda z: z)
    s2["ctrl_no_standardization"] = score(ref_np, rec, "S2L2A")

    # TerraMesh reference: S2L2A on native 224 crops, from one majortom shard.
    tidx = pd.read_parquet(D.VAL / "tok_index.parquet")
    shard = tidx[tidx.corpus == "majortom"].source_shard.iloc[0]
    trows = tidx[tidx.source_shard == shard].row.values[: a.n]
    raw = D.read_rasters("S2L2A", trows)
    tm_ref = np.stack([P.center_crop((r[0] if r.ndim == 4 else r).astype(np.float32)) for r in raw])
    z = torch.cat([P.prepare(r, "S2L2A", device=dev)[0] for r in raw])
    rec = roundtrip(tok, z, lambda z: BP.destandardize(z, "S2L2A"))
    s2["terramesh_reference"] = score(tm_ref, rec, "S2L2A")
    res["S2"] = s2
    del tok
    torch.cuda.empty_cache()

    # ---- BEN S1 -----------------------------------------------------------
    s1raw = np.asarray(a1["S1"][r2p1[rows]])
    s1 = {}
    for tk in ("S1GRD", "S1RTC"):
        tok = T.build(tk, dev)
        ref, nbad = BP.s1_physical(s1raw, dev, tk)
        x, _ = BP.s1_batch(s1raw, dev, tokenizer=tk)
        rec = roundtrip(tok, x, lambda z: BP.destandardize(z, tk))
        s1[tk] = score(ref.cpu().numpy(), rec, tk)
        s1[tk]["nonfinite"] = nbad
        if tk == "S1GRD":
            ref_np = ref.cpu().numpy()
            x, _ = BP.s1_batch(s1raw, dev, tokenizer=tk, reverse_bands=True)
            rec = roundtrip(tok, x, lambda z: BP.destandardize(z, tk).flip(1))
            s1["ctrl_swapped_vv_vh"] = score(ref_np, rec, tk)
            x, _ = BP.s1_batch(s1raw, dev, tokenizer=tk, standardize=False)
            rec = roundtrip(tok, x, lambda z: z)
            s1["ctrl_no_standardization"] = score(ref_np, rec, tk)
            sshard = tidx[tidx.corpus == "ssl4eos12"].source_shard.iloc[0]
            srows = tidx[tidx.source_shard == sshard].row.values[: a.n]
            raw = D.read_rasters("S1GRD", srows)
            tm_ref = np.stack([P.center_crop((r[0] if r.ndim == 4 else r).astype(np.float32)) for r in raw])
            tm_ref = np.where(np.isnan(tm_ref), np.array(C.V1_TOK_MEAN["S1GRD"], np.float32)[:, None, None], tm_ref)
            z = torch.cat([P.prepare(r, "S1GRD", device=dev)[0] for r in raw])
            rec = roundtrip(tok, z, lambda z: BP.destandardize(z, "S1GRD"))
            s1["terramesh_reference_S1GRD"] = score(tm_ref, rec, "S1GRD")
        del tok
        torch.cuda.empty_cache()
    res["S1"] = s1

    # BEN per-band statistics on these patches, against the v1 stats (Step 0.4).
    ph = BP.s2_physical(p10, p20, p60, dev).cpu().numpy()
    res["S2_band_mean"] = dict(zip(C.BAND_ORDER["S2L2A"], ph.mean((0, 2, 3)).round(1).tolist()))
    res["S2_v1_mean"] = dict(zip(C.BAND_ORDER["S2L2A"], C.V1_TOK_MEAN["S2L2A"]))
    res["S1_band_mean_dB"] = dict(zip(["VV", "VH"], np.nanmean(np.where(np.isfinite(s1raw), s1raw, np.nan), (0, 2, 3)).round(2).tolist()))
    res["S1GRD_v1_mean"] = C.V1_TOK_MEAN["S1GRD"]
    res["S1RTC_v1_mean"] = C.V1_TOK_MEAN["S1RTC"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    for mod in ("S2", "S1"):
        for k, v in res[mod].items():
            print(f"{mod:3s} {k:28s} rmse_z {v['rmse_z']['median']:.3f}  ml7 {v['rmse_z_ml7']['median']:.3f}  "
                  f"in-range {v['rmse_z_in']['median']:.3f} ({v['frac_in_range']['median']:.2f})")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{time.time() - t0:.0f} s")
