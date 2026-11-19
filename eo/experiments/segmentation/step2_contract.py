"""A2 Step 2.4 (and Part 2 Step 1.3 for PASTIS): the S2 input contract checked before scaling, and
CloudSen12's rule read (a2seg env, GPU).

Round trip on a handful of crops (24, as BEN's Step 1.3): physical -> encoder input -> tokens ->
decoder (50 steps, seed 0) -> physical, scored as decode_eo.continuous_metrics scores it
(rmse_z_ml7, in the v1-standardized space of the physical image), with the share of pixels in range
(|z| < 1; the decoder clamps beyond it). Controls that must visibly worsen it: reversed band order, raw DN
(no standardization), x10^4. Token usage: codes used, entropy and the JSD between code histograms
against TerraMesh val (S2L2A_tok224, the set BEN's 0.084 was read against) on equal counts (4,000), with
the floor from disjoint halves, as eo/experiments/bigearthnet/ben_token_stats.py computes them.

CloudSen12: 24 train tiles (seed 0), the centre 224 crop; JSD on 4,000 train views (each tile's first
random crop, k = 0: the same rows tokenize_geobench.py writes).
PASTIS: 24 train patches' p1 frames, B01/B09 imputed as s2.CHOSEN_IMPUTATION (or --method), scored on the
ten real bands; JSD on 1,200 train p1 frames (k = 0) against as many TerraMesh rows, and against
TerraMesh rows in Europe's box.

**The rule (A2 plan 2.4, fixed before reading):** CloudSen12 fails if its JSD is above 2 x BEN's
(0.0842, /data/enric/reports/ben/step1_token_stats.json) AND its round trip is worse than TerraMesh's by
more than 50% (rmse_z_ml7 median 0.1559, /data/enric/reports/ben/step1_roundtrip.json, S2
terramesh_reference). Both references are read from those files, never typed. TerraMesh's reference is
also re-run here on the same 24 rows, as a check of this script, not as the rule's input.

    python eo/experiments/segmentation/step2_contract.py --cloudsen12 --pastis --expect-gpu "GTX TITAN X"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/enric/hf")

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo"))
sys.path.insert(0, str(REPO / "eo" / "scripts"))
sys.path.insert(0, str(REPO / "eo" / "experiments" / "bigearthnet"))

from eo.geobench import s2 as G, sources as S, views as V  # noqa: E402
from eo.terramesh_tok import contract as C, preprocess as P, tokenizers as T  # noqa: E402
import decode_eo as D  # noqa: E402
from ben_token_stats import entropy, hist, jsd  # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step2/contract")
BEN_STATS = Path("/data/enric/reports/ben/step1_token_stats.json")
BEN_RT = Path("/data/enric/reports/ben/step1_roundtrip.json")
TM_VAL = Path("/data/enric/data/TerraMesh/val")
N_RT, N_JSD, TIMESTEPS, SEED, BATCH = 24, 4000, 50, 0, 32
REV = list(range(12))[::-1]


def references() -> dict:
    st = json.loads(BEN_STATS.read_text())["S2L2A"]
    rt = json.loads(BEN_RT.read_text())["S2"]
    ref = {"ben_jsd": st["jsd_ben_vs_terramesh"], "ben_jsd_floor": st["jsd_floor_terramesh_halves"],
           "terramesh_rt_ml7_median": rt["terramesh_reference"]["rmse_z_ml7"]["median"],
           "ben_rt_ml7_median": rt["bilinear"]["rmse_z_ml7"]["median"],
           "sources": [str(BEN_STATS), str(BEN_RT)]}
    ref["jsd_limit"] = 2 * ref["ben_jsd"]
    ref["rt_limit"] = 1.5 * ref["terramesh_rt_ml7_median"]
    return ref


def encode_fixed(tok, x: torch.Tensor) -> np.ndarray:
    out = []
    for s in range(0, len(x), BATCH):
        xb = x[s:s + BATCH]
        n = len(xb)
        if n < BATCH:
            xb = torch.cat([xb, xb[:1].expand(BATCH - n, *xb.shape[1:])])
        out.append(P.flatten_tokens(T.encode(tok, xb)).cpu().numpy()[:n])
    return np.concatenate(out)


def roundtrip(tok, x_in: torch.Tensor, ref_phys: np.ndarray, to_phys, exclude=()) -> dict:
    recs_std, recs_phys = [], []
    for i in range(0, len(x_in), 8):
        t = T.encode(tok, x_in[i:i + 8])
        y = T.decode(tok, t, timesteps=TIMESTEPS, seed=SEED).float()
        recs_phys.append(to_phys(y).cpu().numpy())
    rec_phys = np.concatenate(recs_phys)
    rows = []
    mean = np.array(C.V1_TOK_MEAN["S2L2A"], np.float32)[:, None, None]
    std = np.array(C.V1_TOK_STD["S2L2A"], np.float32)[:, None, None]
    for ref, rp in zip(ref_phys, rec_phys):
        ref = ref.copy()
        if exclude:
            ref[list(exclude)] = np.nan
        m = D.continuous_metrics((rp - mean) / std, rp, ref, "S2L2A")
        rz = (ref - mean) / std
        ok = np.isfinite(rz)
        inr = ok & (np.abs(np.nan_to_num(rz)) < 1)
        m["frac_in_range"] = float(inr.sum() / ok.sum())
        m["rmse_z_in"] = float(np.sqrt(np.mean((rz[inr] - ((rp - mean) / std)[inr]) ** 2))) if inr.any() else None
        rows.append(m)
    df = pd.DataFrame(rows)
    return {k: {"median": float(df[k].median()), "mean": float(df[k].mean())}
            for k in ("rmse_z", "rmse_z_ml7", "rmse_z_in", "frac_in_range")}


def terramesh_tokens(n: int, rng, europe: bool = False) -> tuple[np.ndarray, np.ndarray]:
    tt = np.load(TM_VAL / C.tok_dir_name("S2L2A") / "tokens.npy", mmap_mode="r")
    tp = np.load(TM_VAL / C.tok_dir_name("S2L2A") / "present.npy")
    mask = tp
    if europe:
        from eo.data.eval_split import load_row_table
        rt = load_row_table(TM_VAL).set_index("row")
        eu = (rt.center_lat.between(36, 71) & rt.center_lon.between(-11, 32)).reindex(np.arange(len(tp))).fillna(False).to_numpy(bool)
        mask = tp & eu
    rows = rng.permutation(np.flatnonzero(mask))[: 2 * n]
    return hist(tt[np.sort(rows[:n])]), hist(tt[np.sort(rows[n:2 * n])])


def usage(toks: np.ndarray, rng, label: str) -> dict:
    """JSD against TerraMesh on n = min(len, 4,000) rows each (BEN's 0.084 is at n = 4,000), and the
    floors of disjoint halves at n / 2 for both sets (a JSD shrinks with n, so a floor is read at its own n)."""
    n = min(len(toks), N_JSD)
    perm = rng.permutation(len(toks))
    a = hist(toks[np.sort(perm[:n])])
    h = n // 2
    a1, a2 = hist(toks[np.sort(perm[:h])]), hist(toks[np.sort(perm[h:2 * h])])
    t1, t2 = terramesh_tokens(n, rng)
    th1, th2 = terramesh_tokens(h, rng)
    e1, _ = terramesh_tokens(n, rng, europe=True)
    eh1, eh2 = terramesh_tokens(h, rng, europe=True)
    return {"n": int(n), f"codes_used_{label}": int((a > 0).sum()), "codes_used_terramesh": int((t1 > 0).sum()),
            f"entropy_{label}": entropy(a), "entropy_terramesh": entropy(t1),
            f"jsd_{label}_vs_terramesh": jsd(a, t1), "jsd_floor_terramesh_halves_at_n": jsd(t1, t2),
            f"share_on_codes_terramesh_never_used": float(a[(t1 + t2) == 0].sum() / a.sum()),
            "n_half": int(h), f"jsd_floor_{label}_halves": jsd(a1, a2), "jsd_floor_terramesh_halves": jsd(th1, th2),
            f"jsd_{label}_vs_terramesh_europe": jsd(a, e1), "jsd_floor_europe_halves": jsd(eh1, eh2)}


def terramesh_reference(tok) -> dict:
    """ben_roundtrip.py's TerraMesh reference, re-run: 24 rows of the first majortom shard."""
    tidx = pd.read_parquet(TM_VAL / "tok_index.parquet")
    shard = tidx[tidx.corpus == "majortom"].source_shard.iloc[0]
    trows = tidx[tidx.source_shard == shard].row.values[:N_RT]
    raw = D.read_rasters("S2L2A", trows)
    ref = np.stack([P.center_crop((r[0] if r.ndim == 4 else r).astype(np.float32)) for r in raw])
    z = torch.cat([P.prepare(r, "S2L2A", device="cuda")[0] for r in raw])
    return roundtrip(tok, z, ref, G.destandardize)


def cloudsen12(tok) -> dict:
    t = S.cloudsen12_table()
    tr = t[t.split == "train"].reset_index(drop=True)
    pick = np.sort(np.random.default_rng(0).choice(len(tr), N_RT, replace=False))
    o = (512 - C.CROP) // 2
    raw = np.stack([S.cloudsen12_sample(int(tr.idx[i]))[0][:, o:o + C.CROP, o:o + C.CROP] for i in pick])
    phys, n_nd = G.cloudsen12_physical(raw)
    ref = phys.cpu().numpy()
    res = {"tiles": tr.id[pick].tolist(), "nodata_pixels": n_nd,
           "band_median_dn": dict(zip(G.ORDER, np.median(ref, (0, 2, 3)).round(1).tolist())),
           "v1_mean": dict(zip(G.ORDER, C.V1_TOK_MEAN["S2L2A"]))}
    rt = {"production": roundtrip(tok, G.cloudsen12_input(raw)[0], ref, G.destandardize)}
    rt["ctrl_reversed_bands"] = roundtrip(tok, G.cloudsen12_input(raw, band_perm=REV)[0], ref,
                                          lambda z: G.destandardize(z)[:, REV])
    rt["ctrl_raw_dn"] = roundtrip(tok, G.cloudsen12_input(raw, standardize_=False)[0], ref, lambda z: z)
    rt["ctrl_x1e4"] = roundtrip(tok, G.cloudsen12_input(raw, scale=1e4)[0], ref, lambda z: G.destandardize(z) / 1e4)
    res["roundtrip"] = rt
    # token usage on 4,000 train views: each tile's first random crop, k = 0
    v = V.load("cloudsen12")["views"]
    first = v[(v.split == "train") & (v.k == 0)].groupby("sample_idx").head(1).sort_values("sample_idx")
    toks = []
    for s in range(0, len(first), 256):
        ch = first.iloc[s:s + 256]
        x = np.stack([S.cloudsen12_sample(int(r.sample_idx))[0][:, r.y0:r.y0 + 224, r.x0:r.x0 + 224]
                      for r in ch.itertuples()])
        toks.append(encode_fixed(tok, G.cloudsen12_input(x)[0]))
    toks = np.concatenate(toks)
    np.save(OUT / "cloudsen12_train_first_crop_tokens.npy", toks)
    (OUT / "cloudsen12_train_first_crop_views.json").write_text(json.dumps(first.view.tolist()))
    res["usage"] = usage(toks, np.random.default_rng(0), "cloudsen12")
    return res


def pastis(tok, method: str) -> dict:
    t = S.pastis_table()
    x = V.load("pastis")
    views, inputs = x["views"], x["inputs"]
    p1 = inputs[inputs.frame == "p1"].merge(views, on="view")
    tr = p1[(p1.split == "train") & (p1.k == 0)].reset_index(drop=True)
    pick = np.sort(np.random.default_rng(0).choice(len(tr), N_RT, replace=False))
    frames = np.stack([S.pastis_sample(int(tr.sample_idx[i]), ("s2",))["s2"][int(tr.t[i])] for i in pick])
    ref = G.pastis_physical(frames, method).cpu().numpy()
    excl = [G.ORDER.index(b) for b in G.MISSING]
    res = {"patches": tr.sample_id[pick].tolist(), "method": method,
           "band_median_dn": dict(zip(G.ORDER, np.median(ref, (0, 2, 3)).round(1).tolist()))}
    rt = {"production": roundtrip(tok, G.pastis_input(frames, method), ref, G.destandardize, excl)}
    rt["ctrl_reversed_bands"] = roundtrip(tok, G.pastis_input(frames, method, band_perm=REV), ref,
                                          lambda z: G.destandardize(z)[:, REV], excl)
    rt["ctrl_raw_dn"] = roundtrip(tok, G.pastis_input(frames, method, standardize_=False), ref, lambda z: z, excl)
    rt["ctrl_x1e4"] = roundtrip(tok, G.pastis_input(frames, method, scale=1e4), ref,
                                lambda z: G.destandardize(z) / 1e4, excl)
    res["roundtrip"] = rt
    toks = []
    for s in range(0, len(tr), 128):
        ch = tr.iloc[s:s + 128]
        fr = np.stack([S.pastis_sample(int(r.sample_idx), ("s2",))["s2"][int(r.t)] for r in ch.itertuples()])
        toks.append(encode_fixed(tok, G.pastis_input(fr, method)))
    toks = np.concatenate(toks)
    np.save(OUT / f"pastis_train_p1_k0_tokens_{method}.npy", toks)
    res["usage"] = usage(toks, np.random.default_rng(0), "pastis")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloudsen12", action="store_true")
    ap.add_argument("--pastis", action="store_true")
    ap.add_argument("--method", default=None)
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    OUT.mkdir(parents=True, exist_ok=True)
    tok = T.build("S2L2A", "cuda")
    torch.set_float32_matmul_precision("highest")
    ref = references()
    res = {"gpu": gpu, "references": ref, "timesteps": TIMESTEPS, "seed": SEED}
    t0 = time.time()
    res["terramesh_rerun"] = terramesh_reference(tok)
    print("TerraMesh re-run ml7", round(res["terramesh_rerun"]["rmse_z_ml7"]["median"], 4),
          "recorded", round(ref["terramesh_rt_ml7_median"], 4), flush=True)
    if a.cloudsen12:
        c = cloudsen12(tok)
        j = c["usage"]["jsd_cloudsen12_vs_terramesh"]
        r = c["roundtrip"]["production"]["rmse_z_ml7"]["median"]
        c["rule"] = {"jsd": j, "jsd_limit": ref["jsd_limit"], "jsd_above": j > ref["jsd_limit"],
                     "rt_ml7": r, "rt_limit": ref["rt_limit"], "rt_worse": r > ref["rt_limit"],
                     "fails": bool(j > ref["jsd_limit"] and r > ref["rt_limit"])}
        c["rule"]["verdict"] = "CloudSen12 fails: check Burn Scars" if c["rule"]["fails"] else "CloudSen12 stays"
        res["cloudsen12"] = c
        for k, v in c["roundtrip"].items():
            print(f"  CloudSen12 {k:22s} ml7 {v['rmse_z_ml7']['median']:.4f}  in-range {v['frac_in_range']['median']:.3f}", flush=True)
        print("  CloudSen12 rule", json.dumps(c["rule"]), flush=True)
    if a.pastis:
        m = a.method or G.CHOSEN_IMPUTATION
        p = pastis(tok, m)
        res["pastis"] = p
        for k, v in p["roundtrip"].items():
            print(f"  PASTIS {k:22s} ml7 {v['rmse_z_ml7']['median']:.4f}  in-range {v['frac_in_range']['median']:.3f}", flush=True)
        print("  PASTIS usage", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in p["usage"].items()}), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    name = "contract" + ("_cloudsen12" if a.cloudsen12 else "") + ("_pastis" if a.pastis else "") + ".json"
    (OUT / name).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
