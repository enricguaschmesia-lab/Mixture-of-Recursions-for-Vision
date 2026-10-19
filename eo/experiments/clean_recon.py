#!/usr/bin/env python
"""Do the TerraMind tokenizers reconstruct the CLEANED S2L2A/NDVI better or worse? (`mor` env)

Question (Enric, 2026-10-19): ~24% of val S2L2A keeps ESA's +1000 DN BOA offset
(terramesh_tok/clean.py), and the tokenizers were trained on TerraMesh, offset scenes included.
Do they reconstruct the corrected scenes worse, or better, than the same scenes as stored?
The 2026-10-11 build gate (tokenize_clean.py roundtrip) checked it on 16 rows with a 1.5x
tolerance: S2L2A 0.047 vs 0.048 z^2, NDVI 0.035 vs 0.016 (put down to the corrected NDVI's
larger spread). This is the measurement.

    python eo/experiments/clean_recon.py decode --n 256      # GPU: the GTX Titan X only
    python eo/experiments/clean_recon.py score  --n 256      # CPU
    python eo/experiments/clean_recon.py means  --step 10    # CPU

decode  n offset rows (every k-th of the 21,461), decoded from BOTH token sets: the original
        tokens (`stored`, scored against the image as stored) and the cleaned ones
        (`corrected`, against the corrected image), with seeds 0 and 1. Plus `matched`: for
        each offset row, the clean Major TOM row (pool: every --pool-step-th) nearest to its
        corrected image in per-band mean and spatial spread, decoded the same way.
        12.6b settings: unclamped DiVAE, 50 steps, the seed re-applied per batch. Scene i sits
        at the same batch position in every set, so its stored, corrected and matched decodes
        start from the same noise. Decodes are saved (z units, float16), one file per
        (modality, set, seed), so a rerun resumes.
score   per scene: mse_z (the gate's metric); nmse = SSE / SST about the reference's own
        per-band means (scale-free, THE decision metric); rmse_z_ml7 (the project's headline
        continuous metric); rmse in physical units (DN, NDVI); the per-band bias. Paired:
        corrected / stored per scene, seeds averaged first; geometric-mean ratio with a
        bootstrap 95% CI over scenes. For S2L2A the offset is a per-band constant, so SST is
        the same for both versions and the nmse ratio equals the mse ratio; NDVI's is not.
means   per-band means/stds of the stored, corrected and clean scenes, against the tokenizers'
        standardization (V1_TOK_MEAN/STD, which equal TerraMesh's published statistics).

DECISION RULE (declared 2026-10-19, before any number was seen):
  nmse ratio corrected/stored, 95% CI inside [0.90, 1.11]  -> no better or worse on cleaned data
  CI entirely above 1 / entirely below 1                    -> worse / better, by that much
  neither                                                   -> inconclusive at this n
CONTROLS (each must fire, or the result is not read):
  C1 null          seed 1 vs seed 0 on the same tokens is declared equivalent (both sets)
  C2 broken null   seed 1 with +15% injected error vs seed 0 is NOT equivalent, CI above 1
  C3 wrong pairing corrected tokens vs the stored image, and stored vs corrected: median
                   mse_z >= 2x the right pairing
  C4 wrong scene   each decode against the next scene's reference: median nmse >= 2x
Not a control, a reference: corrected vs matched clean rows (is the corrected scenes' error
what clean scenes of that brightness and spread get?).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/enric/hf")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "scripts")); sys.path.insert(0, str(HERE))
from terramesh_tok import clean as CL, contract as C, io as tio, preprocess as P  # noqa: E402

VAL = Path(tio.ROOT)
CLEAN = CL.clean_root(VAL)
OUT = Path("/data/enric/reports/clean_recon")
MODS = CL.CLEANED_MODALITIES                 # ("S2L2A", "NDVI")
SETS = ("stored", "corrected", "matched")
MARGIN = (0.90, 1.11)
INJECT = 0.15                                # C2: +15% error variance on top of seed 1
WORKERS = 3


def outdir(n: int) -> Path:
    return OUT / f"n{n}"


def git_rev() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def stats(mod):
    return (np.array(C.V1_TOK_MEAN[mod], np.float32)[:, None, None],
            np.array(C.V1_TOK_STD[mod], np.float32)[:, None, None])


def to_z(mod, a):
    mean, std = stats(mod)
    return (a.astype(np.float32) - mean) / std


def read_stems(mod, shard, stems):
    """{stem: raw array} for the wanted stems of one shard, decoding only those."""
    out = {}
    with tarfile.open(VAL / mod / shard) as tf:
        for m in tf:
            if m.name.endswith(".zarr.zip"):
                s = m.name[:-len(".zarr.zip")]
                if s in stems:
                    out[s] = tio._decode_zarr_zip(tf.extractfile(m).read())
    missing = set(stems) - set(out)
    if missing:
        raise RuntimeError(f"{mod}/{shard}: {len(missing)} stems missing, e.g. {sorted(missing)[:3]}")
    return out


def _crops_shard(args):
    shard, pairs = args
    st = {s for _, s in pairs}
    s2, nd = read_stems("S2L2A", shard, st), read_stems("NDVI", shard, st)
    return {r: (P.center_crop(s2[s][0]).copy(), P.center_crop(nd[s][0]).copy()) for r, s in pairs}


def read_crops(rows, idx):
    """{row: (S2L2A crop (12,224,224) int16, NDVI crop (1,224,224) float16)}, as stored."""
    sub = idx.loc[list(rows)]
    jobs = [(sh, list(zip(g.index.astype(int), g.stem))) for sh, g in sub.groupby("source_shard")]
    out = {}
    with Pool(WORKERS) as pool:
        for d in pool.imap_unordered(_crops_shard, jobs):
            out.update(d)
    return out


def ref_raw(mod, s2, nd, corrected):
    """The reference an offset row's decode is scored against, in physical units."""
    if mod == "S2L2A":
        return CL.correct_s2(s2, corrected)          # per pixel, so it commutes with the crop
    return CL.ndvi_from_s2(CL.correct_s2(s2, True)) if corrected else nd


def feats(z):
    """Matching features: per-band mean, and the mean over bands of the spatial std (z)."""
    return np.concatenate([np.nanmean(z, axis=(1, 2)), [np.nanstd(z, axis=(1, 2)).mean()]])


def _feats_shard(args):
    crops = _crops_shard(args)
    return {r: {"S2L2A": feats(to_z("S2L2A", a)), "NDVI": feats(to_z("NDVI", b))} for r, (a, b) in crops.items()}


def match(want, pool_rows, pool_feats):
    """Greedy nearest unused pool row, per wanted feature vector, distances scaled by the pool's std."""
    sc = pool_feats.std(0)
    sc[sc == 0] = 1
    used, out, dist = set(), [], []
    for w in want:
        d = (((pool_feats - w) / sc) ** 2).sum(1)
        for k in np.argsort(d):
            if int(pool_rows[k]) not in used:
                used.add(int(pool_rows[k])); out.append(int(pool_rows[k])); dist.append(float(np.sqrt(d[k])))
                break
    return out, dist, sc


# ---------------------------------------------------------------- the plan (rows, matches, refs)
def make_plan(a, out: Path) -> dict:
    pf = out / "plan.json"
    if pf.exists():
        return json.loads(pf.read_text())
    t0 = time.time()
    lab = CL.load_labels(CLEAN)
    idx = pd.read_parquet(VAL / "tok_index.parquet").set_index("row")
    off = np.sort(lab[lab.offset].row.values)
    rows = off[:: len(off) // a.n][: a.n]
    for root in (VAL, CLEAN):
        for mod in MODS:
            pres = np.load(root / C.tok_dir_name(mod) / "present.npy")
            assert pres[rows].all(), f"{root}/{mod}: rows not present"
    for mod in MODS:                                  # the two token sets must differ on these rows
        old = np.load(VAL / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")[rows]
        new = np.load(CLEAN / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")[rows]
        print(f"{mod}: tokens changed per row, median {np.median((old != new).mean(1)):.3f}, "
              f"min {(old != new).mean(1).min():.3f}", flush=True)
    crops = read_crops(rows, idx)
    s2 = np.stack([crops[int(r)][0] for r in rows]); nd = np.stack([crops[int(r)][1] for r in rows])
    # row alignment between the S2 and NDVI readers: TerraMesh's NDVI is the stored bands' NDVI, bit for bit
    agree = np.array([np.array_equal(CL.ndvi_from_s2(s2[i]).view(np.uint16), nd[i].view(np.uint16)) for i in range(len(rows))])
    assert agree.all(), f"S2/NDVI misaligned on {np.flatnonzero(~agree)[:5]}"
    np.save(out / "ref_offset_S2L2A.npy", s2); np.save(out / "ref_offset_NDVI.npy", nd)
    print(f"read {len(rows)} offset rows ({time.time() - t0:.0f} s)", flush=True)

    pool_rows = np.sort(lab[~lab.offset & (lab.corpus == "majortom")].row.values)[:: a.pool_step]
    sub = idx.loc[pool_rows]
    jobs = [(sh, list(zip(g.index.astype(int), g.stem))) for sh, g in sub.groupby("source_shard")]
    pf_ = {}
    with Pool(WORKERS) as pool:
        for d in pool.imap_unordered(_feats_shard, jobs):
            pf_.update(d)
    prow = np.array(sorted(pf_))
    print(f"pool: {len(prow)} clean Major TOM rows ({time.time() - t0:.0f} s)", flush=True)
    plan = {"n": a.n, "rows": [int(r) for r in rows], "pool_step": a.pool_step, "pool_size": int(len(prow)),
            "batch": a.batch, "timesteps": a.timesteps, "seeds": a.seeds, "git_rev": git_rev(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "matched": {}, "match_dist": {}}
    for mod in MODS:
        want = np.array([feats(to_z(mod, ref_raw(mod, s2[i], nd[i], True))) for i in range(len(rows))])
        have = np.array([feats(to_z(mod, ref_raw(mod, s2[i], nd[i], False))) for i in range(len(rows))])
        m, dist, sc = match(want, prow, np.array([pf_[r][mod] for r in prow]))
        plan["matched"][mod] = m
        plan["match_dist"][mod] = {"median": float(np.median(dist)), "p90": float(np.percentile(dist, 90)),
                                   # for scale: how far each corrected image is from its own stored version
                                   "corrected_to_stored_median": float(np.median(np.sqrt((((want - have) / sc) ** 2).sum(1))))}
    mrows = sorted(set(plan["matched"]["S2L2A"]) | set(plan["matched"]["NDVI"]))
    mc = read_crops(mrows, idx)
    np.save(out / "ref_matched_S2L2A.npy", np.stack([mc[r][0] for r in plan["matched"]["S2L2A"]]))
    np.save(out / "ref_matched_NDVI.npy", np.stack([mc[r][1] for r in plan["matched"]["NDVI"]]))
    pf.write_text(json.dumps(plan, indent=1))
    print(f"plan written ({time.time() - t0:.0f} s): {json.dumps(plan['match_dist'])}", flush=True)
    return plan


# ---------------------------------------------------------------- decode (GPU)
def cmd_decode(a) -> int:
    import torch
    from terramesh_tok import tokenizers as T
    import clamp_test as CT
    name = torch.cuda.get_device_name(0)
    if torch.cuda.device_count() != 1 or "TITAN X" not in name or torch.cuda.get_device_capability(0) != (5, 2):
        raise SystemExit(f"expected exactly the GTX Titan X (sm_52) visible, got {name}")
    out = outdir(a.n); out.mkdir(parents=True, exist_ok=True)
    plan = make_plan(a, out)
    assert (plan["batch"], plan["timesteps"]) == (a.batch, a.timesteps), "plan was made with other settings"
    for mod in MODS:
        tok = sched = None
        for set_ in SETS:
            rows = np.array(plan["matched"][mod] if set_ == "matched" else plan["rows"])
            root = CLEAN if set_ == "corrected" else VAL
            grids = np.load(root / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")[rows].reshape(-1, C.GRID, C.GRID)
            for seed in a.seeds:
                f = out / f"dec_{mod}_{set_}_s{seed}.npy"
                if f.exists():
                    continue
                if tok is None:
                    tok = T.build(mod, device="cuda:0")
                    sched = CT.scheduler_like(tok, **CT.VARIANTS["noclip"])
                t0, dec = time.time(), []
                with torch.no_grad():
                    for i in range(0, len(grids), a.batch):
                        g = torch.from_numpy(grids[i:i + a.batch].astype(np.int64)).to("cuda:0")
                        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                        dec.append(tok.decode_tokens(g, timesteps=a.timesteps, scheduler=sched).float().cpu().numpy().astype(np.float16))
                dec = np.concatenate(dec)
                tmp = f.with_suffix(".tmp.npy"); np.save(tmp, dec); os.replace(tmp, f)
                print(f"{mod} {set_} seed {seed}: {len(rows)} scenes in {time.time() - t0:.0f} s", flush=True)
        del tok
        torch.cuda.empty_cache()
    print("DECODE DONE", flush=True)
    return 0


# ---------------------------------------------------------------- score (CPU)
def metrics(dec, ref, std, win):
    """dec, ref: (C,H,W) float32 in z; NaN in ref = no data (filled with 0 = the mean, then excluded)."""
    from scipy.ndimage import uniform_filter
    ok = ~np.isnan(ref)
    r = np.where(ok, ref, 0.0).astype(np.float32)
    e = np.where(ok, dec - r, 0.0)
    n, nc = ok.sum(), ok.sum((1, 2))
    sse = float((e.astype(np.float64) ** 2).sum())
    cm = r.sum((1, 2), keepdims=True) / np.maximum(nc, 1)[:, None, None]
    sst = float((np.where(ok, r - cm, 0.0).astype(np.float64) ** 2).sum())
    ml = uniform_filter(r, size=(1, win, win)) - uniform_filter(dec, size=(1, win, win))
    ep = e * std
    return {"mse_z": sse / n, "nmse": sse / sst, "rmse_z_ml7": float(np.sqrt(np.mean(ml[ok] ** 2))),
            "rmse_phys": float(np.sqrt((ep.astype(np.float64) ** 2).sum() / n)),
            "bias_phys": (ep.sum((1, 2)) / np.maximum(nc, 1)).tolist(),
            "absz": float(np.mean(np.abs(cm))), "spread_z": float(np.sqrt(sst / n))}


def ratio(a, b, rng, B=10000):
    """Paired per-scene a/b: geometric mean with a bootstrap 95% CI over scenes."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    lr = np.log(a / b)
    boot = lr[rng.integers(0, len(lr), (B, len(lr)))].mean(1)
    lo, hi = np.exp(np.percentile(boot, [2.5, 97.5]))
    return {"n": int(len(lr)), "gm": float(np.exp(lr.mean())), "ci": [float(lo), float(hi)],
            "median": float(np.median(a / b)), "share_a_lower": float((a < b).mean()), "sd_log": float(lr.std()),
            "equivalent": bool(MARGIN[0] <= lo and hi <= MARGIN[1]),
            "direction": "a higher" if lo > 1 else "a lower" if hi < 1 else "none"}


def cmd_score(a) -> int:
    import decode_eo as D
    out = outdir(a.n)
    plan = json.loads((out / "plan.json").read_text())
    seeds = plan["seeds"]
    rng = np.random.default_rng(0)
    s2 = np.load(out / "ref_offset_S2L2A.npy"); nd = np.load(out / "ref_offset_NDVI.npy")
    n = len(plan["rows"])
    rep = {"plan": {k: plan[k] for k in ("n", "pool_step", "pool_size", "batch", "timesteps", "seeds", "git_rev", "date", "match_dist")},
           "margin": MARGIN, "inject": INJECT}
    keys = ("mse_z", "nmse", "rmse_z_ml7", "rmse_phys")
    allok = True
    for mod in MODS:
        _, std = stats(mod)
        mref = np.load(out / f"ref_matched_{mod}.npy")
        dec = {(s, sd): np.load(out / f"dec_{mod}_{s}_s{sd}.npy", mmap_mode="r") for s in SETS for sd in seeds}
        refz = lambda s, i: (to_z(mod, mref[i]) if s == "matched" else to_z(mod, ref_raw(mod, s2[i], nd[i], s == "corrected")))
        per = []
        noise_rng = np.random.default_rng(1)
        for i in range(n):
            ref = {s: refz(s, i) for s in SETS}
            nxt = refz("corrected", (i + 1) % n), refz("stored", (i + 1) % n)
            d = {k: v[i].astype(np.float32) for k, v in dec.items()}
            row = {f"{s}_s{sd}": metrics(d[s, sd], ref[s], std, D.ML_WINDOW) for s in SETS for sd in seeds}
            s0, s1 = seeds[0], seeds[-1]
            row["wrong_c_tok_vs_stored"] = metrics(d["corrected", s0], ref["stored"], std, D.ML_WINDOW)
            row["wrong_s_tok_vs_corrected"] = metrics(d["stored", s0], ref["corrected"], std, D.ML_WINDOW)
            row["scene_corrected"] = metrics(d["corrected", s0], nxt[0], std, D.ML_WINDOW)
            row["scene_stored"] = metrics(d["stored", s0], nxt[1], std, D.ML_WINDOW)
            for s in ("stored", "corrected"):
                v = row[f"{s}_s{s1}"]["mse_z"] * INJECT
                noisy = d[s, s1] + noise_rng.normal(0, np.sqrt(v), d[s, s1].shape).astype(np.float32)
                row[f"inject_{s}"] = metrics(noisy, ref[s], std, D.ML_WINDOW)
            per.append(row)
            if i % 32 == 0:
                print(f"{mod}: scored {i}/{n}", flush=True)
        get = lambda name, k: np.array([p[name][k] for p in per])
        avg = lambda s, k: np.mean([get(f"{s}_s{sd}", k) for sd in seeds], axis=0)
        r = {"absolute_median": {s: {k: float(np.median(avg(s, k))) for k in keys + ("absz", "spread_z")} for s in SETS},
             "main_corrected_over_stored": {k: ratio(avg("corrected", k), avg("stored", k), rng) for k in keys},
             "per_seed": {f"s{sd}": {k: ratio(get(f"corrected_s{sd}", k), get(f"stored_s{sd}", k), rng) for k in keys} for sd in seeds},
             "matched_corrected_over_matched": {k: ratio(avg("corrected", k), avg("matched", k), rng) for k in keys},
             "matched_stored_over_matched": {k: ratio(avg("stored", k), avg("matched", k), rng) for k in keys}}
        s0, s1 = f"s{seeds[0]}", f"s{seeds[-1]}"
        c = {}
        for s in ("stored", "corrected"):
            c[f"C1_null_{s}"] = ratio(get(f"{s}_{s1}", "nmse"), get(f"{s}_{s0}", "nmse"), rng)
            c[f"C2_inject_{s}"] = ratio(get(f"inject_{s}", "nmse"), get(f"{s}_{s0}", "nmse"), rng)
        c["C3_wrong_pairing"] = {
            "c_tok_vs_stored_over_right": float(np.median(get("wrong_c_tok_vs_stored", "mse_z")) / np.median(get(f"corrected_{s0}", "mse_z"))),
            "s_tok_vs_corrected_over_right": float(np.median(get("wrong_s_tok_vs_corrected", "mse_z")) / np.median(get(f"stored_{s0}", "mse_z")))}
        c["C4_wrong_scene"] = {
            "corrected": float(np.median(get("scene_corrected", "nmse")) / np.median(get(f"corrected_{s0}", "nmse"))),
            "stored": float(np.median(get("scene_stored", "nmse")) / np.median(get(f"stored_{s0}", "nmse")))}
        fired = {"C1": c["C1_null_stored"]["equivalent"] and c["C1_null_corrected"]["equivalent"],
                 "C2": all(not c[f"C2_inject_{s}"]["equivalent"] and c[f"C2_inject_{s}"]["direction"] == "a higher" for s in ("stored", "corrected")),
                 "C3": min(c["C3_wrong_pairing"].values()) >= 2,
                 "C4": min(c["C4_wrong_scene"].values()) >= 2}
        c["fired"] = fired
        allok &= all(fired.values())
        r["controls"] = c
        absz = get(f"stored_{s0}", "absz")
        terc = np.percentile(absz, [100 / 3, 200 / 3])
        bins = np.digitize(absz, terc)
        r["by_stored_absz_tercile"] = {
            f"T{b + 1} |z| {absz[bins == b].min():.2f}-{absz[bins == b].max():.2f}":
                ratio(avg("corrected", "nmse")[bins == b], avg("stored", "nmse")[bins == b], rng) for b in range(3)}
        bands = C.BAND_ORDER[mod] if mod == "S2L2A" else ["NDVI"]
        bias = lambda name: np.median(np.array([p[name]["bias_phys"] for p in per]), axis=0)
        r["median_bias_phys"] = {"bands": list(bands), **{s: bias(f"{s}_{s0}").round(4).tolist() for s in SETS},
                                 "wrong_c_tok_vs_stored": bias("wrong_c_tok_vs_stored").round(4).tolist()}
        m = r["main_corrected_over_stored"]["nmse"]
        r["verdict"] = ("no better or worse (within margin)" if m["equivalent"] else
                        "worse on the cleaned data" if m["direction"] == "a higher" else
                        "better on the cleaned data" if m["direction"] == "a lower" else "inconclusive at this n")
        if m["equivalent"] and m["direction"] != "none":
            r["verdict"] += f"; detectably {'higher' if m['direction'] == 'a higher' else 'lower'} error, but inside the margin"
        rep[mod] = r
        np.save(out / f"per_scene_{mod}.npy", np.array(per, dtype=object), allow_pickle=True)
    rep["controls_all_fired"] = bool(allok)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({m: {"verdict": rep[m]["verdict"], "nmse": rep[m]["main_corrected_over_stored"]["nmse"],
                          "mse_z": rep[m]["main_corrected_over_stored"]["mse_z"], "fired": rep[m]["controls"]["fired"]}
                      for m in MODS}, indent=1))
    return 0 if allok else 1


# ---------------------------------------------------------------- means (CPU)
def _means_shard(args):
    shard, triples = args
    st = {t[1] for t in triples}
    s2, nd = read_stems("S2L2A", shard, st), read_stems("NDVI", shard, st)
    acc = {}

    def add(key, mod, a):
        a = a.astype(np.float64).reshape(a.shape[-3], -1)
        ok = ~np.isnan(a)
        z = np.where(ok, a, 0.0)
        s = acc.setdefault((key, mod), [np.zeros(len(a)), np.zeros(len(a)), np.zeros(len(a)), 0])
        s[0] += z.sum(1); s[1] += (z ** 2).sum(1); s[2] += ok.sum(1); s[3] += 1

    for r, stem, off, corpus in triples:
        a2, an = s2[stem][0], nd[stem][0]
        for g in (["offset_stored"] if off else ["clean"]) + [f"{corpus}_stored"]:
            add(g, "S2L2A", a2); add(g, "NDVI", an)
        if off:
            add("offset_corrected", "S2L2A", CL.correct_s2(a2, True))
            add("offset_corrected", "NDVI", CL.ndvi_from_s2(CL.correct_s2(a2, True)))
    return acc


def cmd_means(a) -> int:
    lab = CL.load_labels(CLEAN).set_index("row")
    idx = pd.read_parquet(VAL / "tok_index.parquet").set_index("row")
    rows = idx.index.values[:: a.step]
    sub = idx.loc[rows]
    jobs = [(sh, [(int(r), s, bool(lab.offset[r]), "ssl4eo" if c == "ssl4eos12" else c) for r, s, c in zip(g.index, g.stem, g.corpus)])
            for sh, g in sub.groupby("source_shard")]
    acc, t0 = {}, time.time()
    with Pool(WORKERS) as pool:
        for k, d in enumerate(pool.imap_unordered(_means_shard, jobs)):
            for key, v in d.items():
                s = acc.setdefault(key, [0, 0, 0, 0])
                for j in range(4):
                    s[j] = s[j] + v[j]
            if k % 10 == 0:
                print(f"means: {k + 1}/{len(jobs)} shards ({time.time() - t0:.0f} s)", flush=True)

    def ms(*keys, mod):
        s, ss, n = (sum(acc[(k, mod)][j] for k in keys) for j in range(3))
        m = s / n
        return m, np.sqrt(ss / n - m ** 2), int(sum(acc[(k, mod)][3] for k in keys))

    rep = {"step": a.step, "n_rows": int(len(rows)), "git_rev": git_rev(), "date": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    for mod in MODS:
        groups = {"stored_all": ("clean", "offset_stored"), "corrected_all": ("clean", "offset_corrected"),
                  "clean_only": ("clean",), "offset_stored": ("offset_stored",), "offset_corrected": ("offset_corrected",),
                  "majortom_stored": ("majortom_stored",), "ssl4eo_stored": ("ssl4eo_stored",)}
        r = {"bands": list(C.BAND_ORDER[mod]) if mod == "S2L2A" else ["NDVI"],
             "V1_TOK_MEAN": C.V1_TOK_MEAN[mod], "V1_TOK_STD": C.V1_TOK_STD[mod]}
        for g, ks in groups.items():
            m, sd, cnt = ms(*ks, mod=mod)
            r[g] = {"mean": m.round(4).tolist(), "std": sd.round(4).tolist(), "n_scenes": cnt}
        r["offset_share_in_sample"] = r["offset_stored"]["n_scenes"] / r["stored_all"]["n_scenes"]
        if mod == "S2L2A":
            v1, ca, cl = np.array(C.V1_TOK_MEAN[mod]), np.array(r["corrected_all"]["mean"]), np.array(r["clean_only"]["mean"])
            # if V1 = this val population with a share p of offset scenes, V1 - corrected_all = 1000 p in every band
            r["implied_offset_share_vs_corrected_all"] = ((v1 - ca) / CL.BOA_ADD_OFFSET).round(4).tolist()
            r["v1_minus_stored_all_DN"] = (v1 - np.array(r["stored_all"]["mean"])).round(1).tolist()
            r["v1_minus_clean_only_DN"] = (v1 - cl).round(1).tolist()
        rep[mod] = r
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"means_step{a.step}.json"
    f.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--n", type=int, default=256)
    d.add_argument("--pool-step", type=int, default=10, help="matching pool: every k-th clean Major TOM row")
    d.add_argument("--batch", type=int, default=4)
    d.add_argument("--timesteps", type=int, default=50)
    d.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    s = sub.add_parser("score"); s.add_argument("--n", type=int, default=256)
    m = sub.add_parser("means"); m.add_argument("--step", type=int, default=10)
    a = ap.parse_args()
    return {"decode": cmd_decode, "score": cmd_score, "means": cmd_means}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
