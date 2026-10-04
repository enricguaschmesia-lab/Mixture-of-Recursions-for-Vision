"""Was the DiVAE clamp hiding MODEL error, or does the decoder misbehave on generated grids? (2026-10-02)

clamp_generated.py found that decoding arm A's generations without the clamp
makes 3-44% of in-range scenes more than 10% worse (S1 worst), while the
ground-truth decode improves. Two readings fit:

  (a) the model's tokens imply values beyond +-1 on scenes whose truth is
      inside, and the clamp pulled the decode back towards the truth -- the
      unclamped score is then the honest one;
  (b) the decoder renders generated (off-distribution) token grids
      unfaithfully once unclamped -- the unclamped score then adds decoder
      error to the model's.

They are separated by asking what the TOKENS say, independently of any
decoder. Each code's meaning is measured directly: the mean standardized
value of the real 16x16 patches that encode to it (the code -> value lookup),
built from the tokenized val split with every HELD-OUT row excluded
(eo/data/eval_rows_v1.json), so no scored scene informs its own lookup. A
token k <-> patch (k // 14, k % 14) on the 224 centre crop (contract
FLATTEN_ORDER). Patch statistics are in z (V1_TOK_MEAN/STD), NaN -> 0
(contract NAN_POLICY "mean").

Then, at patch scale, over the same 64 D3.11 scenes and arm-A checkpoint as
clamp_generated.py (same grids, seed, batch, timesteps), the ground-truth and
generated grids are each decoded released and unclamped, and the question is
one of DIRECTION. For every patch-channel that removing the clamp moves
noticeably (|noclip - released| >= DELTA), does it move TOWARDS what its code
implies (sign(noclip - released) == sign(implied - released))? Under (a) the
decoder follows the codes, so it does. Under (b) it goes somewhere the codes
do not say. A direction is counted only where the target is also >= DELTA
away from the released value, so a sign is never read off noise.

PASS CRITERIA. Patches whose code has fewer than MIN_N lookup samples are
masked out of every number.
  K0  lookup valid   on the ground-truth grids, implied predicts the true patch
                     means with R^2 >= 0.5 (pooled patches x channels), and a
                     lookup with its codes permuted gives R^2 <= 0.1.
  K1  positive control: on the GROUND-TRUTH grids, where removing the clamp is
                     known to be right (clamp_test.py), the moved patch-
                     channels agree in direction with the truth on >= 80% and
                     with the implied value on >= 70%.
  K1c broken control: on the WORSE generated scenes, a permuted lookup gives
                     direction agreement <= 0.6, i.e. a wrong lookup cannot
                     produce an (a) verdict.
  K2  the decisive measurement: on the WORSE generated scenes, agreement with
                     the implied value >= 0.7 -> (a); <= 0.6 -> (b); between ->
                     mixed. Reported beside it: agreement with the truth (low
                     is what made them worse), and the same on the other
                     in-range scenes.
WORSE = in-range scenes (|z| < 1) where noclip rmse_z_ml7 > 1.10 x released,
read from clamp_generated's artifacts -- the exact set G1 counted.

⚠ THE CRITERIA WERE REVISED after the S1RTC smoke run (8 scenes, 3-shard
lookup) and BEFORE the full run. The original version:
  K0  R^2 >= 0.7 and permuted R^2 <= 0.1;
  K1  median rmse(noclip, implied) over WORSE scenes <= 1.5 x its median on
      the ground-truth grids;
  K2  e_implied > e_released on >= 70% of WORSE scenes, and median
      e_implied / e_noclip in [0.67, 1.5].
Why they were changed: (1) K0's 0.7 is above what S1 can reach. The lookup's
in-sample ceiling, between-code / total variance, is 0.73 for S1RTC, and the
smoke run measured 0.59 out of sample. (2) K2 could not discriminate. A
per-code average discards the context the decoder uses, so e_implied exceeds
every decoded error on the NON-worse scenes too (smoke: 0.81 vs 0.69
released), and "e_implied > e_released" passes trivially. rmse(implied, .)
carries that same within-code floor, so magnitudes are reported but not
gated on; direction is gated on instead.
A second revision, after the same smoke run re-done with the directional
analysis: K1c was first "permuted agreement in [0.4, 0.6]", which assumed a
symmetric null. The null is not symmetric. Removing the clamp moves values
OUTWARD past +-1, while a random code's mean sits near 0, inward, so a wrong
lookup disagrees systematically (smoke: 0.14 against 0.90 for the real
lookup). The control's job is to show that a wrong lookup cannot yield
verdict (a), hence <= 0.6.

Outcome (2026-10-02, docs/worklog.md). Agreement with the code-implied value
on the WORSE scenes, by modality:
  S1RTC  0.82, all controls pass -> (a)
  NDVI   0.84 -> leans (a), formally inconclusive (K0 R^2 0.40)
  S2L2A  0.70 -> mixed (3 worse scenes)
  S1GRD  0.52 -> leans (b), formally inconclusive (K0 R^2 0.47)
  DEM    0.64 -> inconclusive (K1 0.685)
Lookup noise is ruled out: generated codes are as well sampled as true ones.

Runs in `mor`, from the repo root, on the Titan X (shared, memory-capped):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        python eo/experiments/code_value_test.py
Writes /data/enric/reports/clamp_test/code_value/{lookup_<T>.npz, cv_<T>.json, summary.json}.
"""
import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
for p in (REPO / "eo" / "experiments", REPO / "eo" / "scripts", REPO / "eo"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import clamp_test as CT  # noqa: E402
import decode_eo as D  # noqa: E402
from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T  # noqa: E402

MODS = ["S1RTC", "S1GRD", "NDVI", "S2L2A", "DEM"]
GEN_ROOT = pathlib.Path("/data/enric/generations/d311/arm_a")
REP_ROOT = pathlib.Path("/data/enric/reports/d311/arm_a")
CG_ROOT = pathlib.Path("/data/enric/reports/clamp_test/generated")
EVAL_ROWS = REPO / "eo" / "data" / "eval_rows_v1.json"
SEED, BATCH, STEPS = 0, 4, 50
MIN_N = 5
DELTA = 0.25     # z units: a move, and a target distance, that counts as a direction


def zcrop(arr, mod):
    """Raw raster -> (C, 224, 224) standardized centre crop, NaN -> 0."""
    a = P.center_crop(arr[0] if arr.ndim == 4 else arr).astype(np.float32)
    mean = np.array(C.V1_TOK_MEAN[mod], np.float32)[:, None, None]
    std = np.array(C.V1_TOK_STD[mod], np.float32)[:, None, None]
    return np.nan_to_num((a - mean) / std, nan=0.0)


def patch_means(x):
    """(..., C, 224, 224) -> (..., 196, C), token order k = row * GRID + col."""
    *lead, c, h, w = x.shape
    g, p = C.GRID, C.PATCH
    m = x.reshape(*lead, c, g, p, g, p).mean(axis=(-3, -1))      # (..., C, g, g)
    return np.moveaxis(m.reshape(*lead, c, g * g), -2, -1)


def build_lookup(mod, exclude, max_shards, rng):
    index = pd.read_parquet(D.VAL / "tok_index.parquet")
    present = np.load(D.VAL / C.tok_dir_name(mod) / "present.npy")
    tokens = np.load(D.VAL / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")
    corpus = {"S1GRD": "ssl4eos12", "S1RTC": "majortom"}.get(mod)
    sub = index if corpus is None else index[index.corpus == corpus]
    row_of = {(r.source_shard, r.stem): int(r.row) for r in sub.itertuples()}
    shards = list(rng.permutation(sub.source_shard.unique()))[:max_shards]
    K, nch = C.CODEBOOK[mod], C.N_CHANNELS[mod]
    s1, s2, cnt = np.zeros((K, nch)), np.zeros((K, nch)), np.zeros(K)
    n_scenes = 0
    for shard in shards:
        codes, pms = [], []
        for stem, arr in tio.iter_shard(mod, shard):
            row = row_of.get((shard, stem))
            if row is None or not present[row] or row in exclude:
                continue
            codes.append(np.asarray(tokens[row]).astype(np.int64))
            pms.append(patch_means(zcrop(arr, mod)))
        if not codes:
            continue
        codes, pms = np.concatenate(codes), np.concatenate(pms).astype(np.float64)
        n_scenes += len(codes) // C.TOKENS_PER_SAMPLE
        cnt += np.bincount(codes, minlength=K)
        for ch in range(nch):
            s1[:, ch] += np.bincount(codes, weights=pms[:, ch], minlength=K)
            s2[:, ch] += np.bincount(codes, weights=pms[:, ch] ** 2, minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt[:, None]
        sd = np.sqrt(np.maximum(s2 / cnt[:, None] - mean ** 2, 0))
    return mean.astype(np.float32), sd.astype(np.float32), cnt.astype(np.int64), n_scenes, len(shards)


def rmse_masked(a, b, m):
    """Per scene, over patches with m True. a, b: (N, 196, C); m: (N, 196)."""
    out = []
    for i in range(len(a)):
        k = m[i]
        out.append(float(np.sqrt(((a[i][k] - b[i][k]) ** 2).mean())) if k.any() else np.nan)
    return np.array(out)


def r2(pred, true):
    ss = ((true - pred) ** 2).sum(); st = ((true - true.mean()) ** 2).sum()
    return float(1 - ss / st)


def run_target(mod, args, out_dir, exclude):
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    lk = out_dir / f"lookup_{mod}.npz"
    if lk.exists():
        z = np.load(lk); mean, sd, cnt = z["mean"], z["sd"], z["cnt"]
        n_scenes, n_shards = int(z["n_scenes"]), int(z["n_shards"])
    else:
        mean, sd, cnt, n_scenes, n_shards = build_lookup(mod, exclude, args.max_shards, rng)
        np.savez(lk, mean=mean, sd=sd, cnt=cnt, n_scenes=n_scenes, n_shards=n_shards)
    print(f"[{mod}] lookup: {n_scenes} scenes from {n_shards} shards, "
          f"{int((cnt >= MIN_N).sum())}/{len(cnt)} codes with >= {MIN_N} patches "
          f"({time.time() - t0:.0f}s)", flush=True)

    ck = args.checkpoint
    gdir = GEN_ROOT / ck / mod / "slot_masked"
    rows = np.load(gdir / f"decode_rows_{mod}.npy")
    assert set(rows.tolist()) <= exclude, f"{mod}: scored rows are not all held out"
    n = min(len(rows), args.limit) if args.limit else len(rows)
    rows = rows[:n]
    art = np.load(D.VAL / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")
    truth = np.asarray(art[rows]).astype(np.int64).reshape(n, C.TOKENS_PER_SAMPLE)
    gen = np.load(gdir / f"decode_grids_{mod}.npy")[:n].astype(np.int64).reshape(n, -1)
    rasters = D.read_rasters(mod, rows)
    true_pm = np.stack([patch_means(zcrop(a, mod)) for a in rasters])          # (n, 196, C)

    # WORSE: exactly clamp_generated's G1 set
    cg = json.loads((CG_ROOT / f"gen_{mod}.json").read_text())
    st = json.loads((REP_ROOT / ck / f"{mod}_arm_a_{ck}" / f"metrics_{mod}.json").read_text())
    rel = np.array([s["rmse_z_ml7"] for s in st["per_scene"]["generated"][:n]])
    nc = np.array([s["rmse_z_ml7"] for s in cg["per_scene_noclip"][ck][:n]])
    inr = np.array([s["in_range"] for s in st["per_scene"]["generated"][:n]])
    worse = inr & (nc > 1.10 * rel)
    other = inr & ~worse

    # --- K0: the lookup on ground-truth grids, and its broken control ----------
    ok_t = cnt[truth] >= MIN_N
    imp_t = mean[truth]
    k0_r2 = r2(imp_t[ok_t], true_pm[ok_t])
    perm = rng.permutation(len(cnt))
    k0_r2_perm = r2(mean[perm][truth][ok_t & (cnt[perm][truth] >= MIN_N)],
                    true_pm[ok_t & (cnt[perm][truth] >= MIN_N)])
    print(f"[{mod}] K0 R2 {k0_r2:.3f}  permuted {k0_r2_perm:.3f}  coverage truth "
          f"{ok_t.mean():.3f}", flush=True)

    # --- decode: ground truth and generated, each released and unclamped -------
    tok = T.build(mod, device=T.DEVICE)
    nocl = CT.scheduler_like(tok, **CT.VARIANTS["noclip"])
    grids = {"ceiling_released": (truth, None), "ceiling_noclip": (truth, nocl),
             "gen_released": (gen, None), "gen_noclip": (gen, nocl)}
    dec = {}
    for name, (g, sched) in grids.items():
        t1 = time.time()
        out = CT.decode(tok, g.reshape(n, C.GRID, C.GRID), sched, BATCH, SEED, STEPS).numpy()
        dec[name] = patch_means(out)
        print(f"[{mod}] {name:16s} {n} scenes in {time.time() - t1:.0f}s  "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    del tok; T.build.cache_clear(); torch.cuda.empty_cache()

    ok_g = cnt[gen] >= MIN_N
    imp_g = mean[gen]
    imp_g_perm = mean[perm][gen]
    ok_g_perm = ok_g & (cnt[perm][gen] >= MIN_N)
    allsc = np.ones(n, bool)

    def direction(src, target, mask, sel):
        """Share of moved patch-channels (|noclip - released| >= DELTA, target
        also >= DELTA from released) that moved towards `target`."""
        r, q = dec[f"{src}_released"], dec[f"{src}_noclip"]
        d, t = q - r, target - r
        m = sel[:, None, None] & mask[:, :, None] & (np.abs(d) >= DELTA) & (np.abs(t) >= DELTA)
        k = int(m.sum())
        return {"share": round(float((np.sign(d[m]) == np.sign(t[m])).mean()), 4) if k else None,
                "n": k}

    dir_ = {
        "ceiling_vs_truth": direction("ceiling", true_pm, ok_t, allsc),
        "ceiling_vs_implied": direction("ceiling", imp_t, ok_t, allsc),
        "worse_vs_implied": direction("gen", imp_g, ok_g, worse),
        "worse_vs_implied_permuted": direction("gen", imp_g_perm, ok_g_perm, worse),
        "worse_vs_truth": direction("gen", true_pm, ok_g, worse),
        "other_vs_implied": direction("gen", imp_g, ok_g, other),
        "other_vs_truth": direction("gen", true_pm, ok_g, other),
        "out_of_range_vs_implied": direction("gen", imp_g, ok_g, ~inr),
    }
    print(f"[{mod}] direction {json.dumps(dir_)}", flush=True)

    # magnitudes, descriptive only (they carry the within-code floor)
    f_ceil = rmse_masked(dec["ceiling_noclip"], imp_t, ok_t)
    f_nc = rmse_masked(dec["gen_noclip"], imp_g, ok_g)
    f_rel = rmse_masked(dec["gen_released"], imp_g, ok_g)
    e_imp = rmse_masked(imp_g, true_pm, ok_g)
    e_nc = rmse_masked(dec["gen_noclip"], true_pm, ok_g)
    e_rel = rmse_masked(dec["gen_released"], true_pm, ok_g)
    # where the codes imply out-of-clamp values on in-clamp truth
    ooc = np.array([float(((np.abs(imp_g[i]) > 1) & (np.abs(true_pm[i]) < 1) & ok_g[i][:, None]).mean())
                    for i in range(n)])

    def med(x, sel):
        v = x[sel & ~np.isnan(x)]
        return round(float(np.median(v)), 4) if len(v) else None

    nw = int(worse.sum())
    k0 = bool(k0_r2 >= 0.5 and k0_r2_perm <= 0.1)
    ct, ci = dir_["ceiling_vs_truth"]["share"], dir_["ceiling_vs_implied"]["share"]
    k1 = bool(ct is not None and ci is not None and ct >= 0.8 and ci >= 0.7)
    wp = dir_["worse_vs_implied_permuted"]["share"]
    k1c = None if wp is None else bool(wp <= 0.6)
    wi = dir_["worse_vs_implied"]["share"]
    if nw == 0 or wi is None:
        verdict = "n/a (no worse scenes)"
    elif not (k0 and k1 and k1c):
        verdict = "inconclusive (a control failed)"
    else:
        verdict = ("(a) clamp hid model error" if wi >= 0.7 else
                   "(b) decoder unfaithful on generated grids" if wi <= 0.6 else "mixed")
    tests = {
        "K0_lookup": {"pass": k0, "r2": round(k0_r2, 4), "r2_permuted": round(k0_r2_perm, 4),
                      "coverage_truth": round(float(ok_t.mean()), 4),
                      "coverage_generated": round(float(ok_g.mean()), 4)},
        "K1_positive_control": {"pass": k1, "vs_truth": ct, "vs_implied": ci},
        "K1c_broken_control": {"pass": k1c, "share": wp},
        "K2_worse_vs_implied": wi,
        "direction": dir_,
        "n_worse": nw, "n_other_in_range": int(other.sum()),
        "magnitudes": {"median_f_ceiling_noclip": med(f_ceil, allsc),
                       "median_f_gen_noclip_worse": med(f_nc, worse),
                       "median_f_gen_released_worse": med(f_rel, worse),
                       "median_e_implied_worse": med(e_imp, worse),
                       "median_e_released_worse": med(e_rel, worse),
                       "median_e_noclip_worse": med(e_nc, worse),
                       "median_e_implied_other": med(e_imp, other),
                       "median_e_released_other": med(e_rel, other),
                       "median_e_noclip_other": med(e_nc, other)},
        "out_of_clamp_implied_frac": {"worse": med(ooc, worse), "other": med(ooc, other)},
        "verdict": verdict,
    }
    print(f"[{mod}] tests {json.dumps(tests)}", flush=True)

    per = {k: [None if np.isnan(v) else round(float(v), 4) for v in arr] for k, arr in
           dict(f_ceiling=f_ceil, f_gen_noclip=f_nc, f_gen_released=f_rel, e_implied=e_imp,
                e_noclip=e_nc, e_released=e_rel, ooc_implied=ooc).items()}
    doc = {"target": mod, "checkpoint": ck, "n": n, "rows": rows.tolist(),
           "worse": worse.tolist(), "in_range": inr.tolist(), "tests": tests, "per_scene": per,
           "provenance": {"gpu": torch.cuda.get_device_name(0), "seed": SEED, "batch": BATCH,
                          "timesteps": STEPS, "min_n": MIN_N, "lookup_scenes": n_scenes,
                          "lookup_shards": n_shards, "eval_rows": str(EVAL_ROWS),
                          "seconds": round(time.time() - t0)}}
    (out_dir / f"cv_{mod}.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mods", nargs="*", default=MODS)
    ap.add_argument("--checkpoint", default="checkpoint-2000")
    ap.add_argument("--max-shards", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-fraction", type=float, default=0.35)
    ap.add_argument("--out", default="/data/enric/reports/clamp_test/code_value")
    args = ap.parse_args()

    print(f"gpu: {D.assert_gpu(False)}", flush=True)
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    exclude = set(json.loads(EVAL_ROWS.read_text())["eval_rows"])
    summary = {}
    for mod in args.mods:
        f = out_dir / f"cv_{mod}.json"
        doc = json.loads(f.read_text()) if f.exists() else run_target(mod, args, out_dir, exclude)
        summary[mod] = doc["tests"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("CODE VALUE TEST DONE")


if __name__ == "__main__":
    main()
