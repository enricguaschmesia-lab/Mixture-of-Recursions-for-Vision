"""Does removing the DiVAE decoder's +-1 clamp fix out-of-range reconstruction? (2026-10-01)

Why this exists: the DiVAE decoders clamp their output to [-1, 1] in
standardized space (notes/tokenizer_bringup.md section 5), so scenes with
|z| >~ 1 reconstruct badly and D3.11 scores only |z| < 1 scenes. Reading the
code on 2026-10-01 established three things this script tests rather than
assumes:

  * The clamp is INFERENCE-ONLY. `DiVAE.train_forward` has no clamp and the
    decoder predicts x0 directly (prediction_type="sample") against unclamped
    z-scored targets; the parameter's own docstring says "at inference only".
    So the U-Net was trained to output |z| > 1, and the clamp discards it.
  * The only clamp is step 4 of the scheduler's `step()` -- terratorch's
    VENDORED DDIMScheduler (tokenizer/scheduling/scheduling_ddim.py), not
    diffusers'. With DDIM, eta=0 and set_alpha_to_one=True the last step
    returns the clamped x0 itself, which is why outputs pin at exactly +-1.000.
  * The released config sets thresholding=True with sample_max_value left at
    its default 1.0. Dynamic thresholding computes s = clamp(quantile, 1,
    sample_max_value) and returns clamp(x, -s, s) / s, so at max 1.0 it IS a
    hard clip to [-1, 1]. And at any larger max it still returns values in
    [-1, 1] -- it RESCALES, it does not widen. The way to widen the bound is
    clip_sample_range with thresholding off.

Everything here is decoding of GROUND-TRUTH tokens from the _tok224 artifact.
No MoR model is involved, the tokens are unchanged, and nothing in the training
path depends on the decoder.

SAMPLER VARIANTS (all share the released scheduler config except as noted):
  released   the tokenizer's own scheduler, untouched -- the reference
  clip0.5    thresholding off, clip_sample_range=0.5 -- the BROKEN control
  clip3      thresholding off, clip_sample_range=3
  noclip     thresholding off, clip_sample off

PASS CRITERIA, fixed before the first run:
  C0  mechanism   a scheduler rebuilt from the released config reproduces
                  T.decode's output bit-identically (first batch, every
                  modality). If not, the variant mechanism is wrong and
                  nothing below counts.
  C1  claim       thresholding off + clip_sample_range=1 matches `released`
                  to <= 1e-6 (first batch): thresholding here is a hard clip.
  C2  claim       thresholding on + sample_max_value=10 still outputs within
                  [-1, 1] (first batch): dynamic thresholding cannot widen.
  C3  clamp gone  `noclip` output exceeds |1| on some |z| >= 1 scene.
  C4  control     `clip0.5` is WORSE than `released` on rmse_z_ml7 for >= 80%
                  of scenes with |z| >= 0.5. The metric must be able to see
                  clamp damage, or C5 and C6 mean nothing.
  C5  fix         `noclip` is better than `released` on rmse_z_ml7 for >= 80%
                  of |z| >= 1 scenes.
  C6  no harm     on |z| < 0.5 scenes the median paired ratio noclip/released
                  of rmse_z_ml7 is <= 1.05, AND so is the ratio over in-clamp
                  pixels only (|ref_ml| < 1) across all scenes. This is the
                  check that removing the clamp does not destabilise sampling.

Outcome (2026-10-01, docs/worklog.md and notes/tokenizer_bringup.md 5a):
C0-C3, C5 and C6 pass on all five modalities. C4 FAILS on S1RTC (0.67) and
NDVI (0.625): on |z| >~ 1.5 dark/water scenes the released sampler does not
pin at the near bound -- it lands on the WRONG SIDE (S1RTC z ~ -2.9 decodes to
+0.84), so a tighter bound shrinks its worst case. Restricted post hoc to
0.5 <= |z| < 1.5, C4 holds 69/69 scenes. noclip is never worse than released
by more than 4.6% on any |z| < 0.5 scene, and improves S2L2A even there (2x).

Scenes: per modality, shards are scanned (corpus-restricted for S1, as in
verify_step5) until each scene-|z| bin [0,.5) [.5,1) [1,2) [2,inf) holds
--per-bin candidates or --max-shards is reached; --per-bin are drawn from each.
|z| and the metric are decode_eo's (scene_z, continuous_metrics), imported so
they cannot drift.

Runs in `mor`, from the repo root, on the Titan X (decode_eo.assert_gpu refuses
the TITAN V). It caps itself at --mem-fraction of the card because it shares
it with other jobs:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        python eo/experiments/clamp_test.py

Writes /data/enric/reports/clamp_test/{metrics_<MOD>.json, fig_<MOD>.png,
summary.json}. A modality whose metrics file exists is skipped (resumable).
"""
import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eo" / "scripts")); sys.path.insert(0, str(REPO / "eo"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

import decode_eo as D  # noqa: E402
from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T  # noqa: E402

MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI"]       # LULC: ViT decoder, no clamp
BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]
VARIANTS = {
    "released": None,
    "clip0.5": dict(thresholding=False, clip_sample=True, clip_sample_range=0.5),
    "clip3":   dict(thresholding=False, clip_sample=True, clip_sample_range=3.0),
    "noclip":  dict(thresholding=False, clip_sample=False),
}
CLAIM_VARIANTS = {   # first batch only
    "copy":    {},
    "clip1":   dict(thresholding=False, clip_sample=True, clip_sample_range=1.0),
    "dyn10":   dict(thresholding=True, sample_max_value=10.0),
}


def scheduler_like(tok, **overrides):
    """A fresh scheduler of the tokenizer's class, from its config + overrides."""
    s = tok.noise_scheduler
    cfg = {k: v for k, v in dict(s.config).items() if not k.startswith("_")}
    cfg.update(overrides)
    return type(s)(**cfg)


@torch.no_grad()
def decode(tok, grids, sched, batch, seed, timesteps):
    """decode_eo.decode_batched, with an explicit scheduler. Same seeding per
    batch, so every variant starts from the same noise for the same scene."""
    outs = []
    for i in range(0, len(grids), batch):
        g = torch.from_numpy(grids[i:i + batch].astype(np.int64)).to(T.DEVICE)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        kw = {} if sched is None else {"scheduler": sched}
        outs.append(tok.decode_tokens(g, timesteps=timesteps, **kw).float().cpu())
    return torch.cat(outs)


def pick_scenes(mod, per_bin, max_shards, rng):
    index = pd.read_parquet(D.VAL / "tok_index.parquet")
    present = np.load(D.VAL / C.tok_dir_name(mod) / "present.npy")
    corpus = {"S1GRD": "ssl4eos12", "S1RTC": "majortom"}.get(mod)
    sub = index if corpus is None else index[index.corpus == corpus]
    row_of = {(r.source_shard, r.stem): int(r.row) for r in sub.itertuples()}
    shards = list(rng.permutation(sub.source_shard.unique()))
    pools = [[] for _ in BINS]
    scanned = 0
    for shard in shards[:max_shards]:
        for stem, arr in tio.iter_shard(mod, shard):
            row = row_of.get((shard, stem))
            if row is None or not present[row]:
                continue
            z = D.scene_z(mod, arr)
            if np.isnan(z):
                continue
            scanned += 1
            for b, (lo, hi) in enumerate(BINS):
                if lo <= z < hi:
                    pools[b].append((row, z))
        if all(len(p) >= per_bin for p in pools):
            break
    picks = []
    for b, p in enumerate(pools):
        for i in rng.choice(len(p), size=min(per_bin, len(p)), replace=False):
            picks.append((b,) + p[i])
    return picks, scanned, [len(p) for p in pools]


def scene_metrics(out_std, ref_raw, mod):
    m = D.continuous_metrics(out_std, P.destandardize(torch.from_numpy(out_std), mod).numpy(),
                             ref_raw, mod)
    mean = np.array(C.V1_TOK_MEAN[mod], np.float32)[:, None, None]
    std = np.array(C.V1_TOK_STD[mod], np.float32)[:, None, None]
    ref_z = (ref_raw - mean) / std
    ok = ~np.isnan(ref_z)
    win = (1, D.ML_WINDOW, D.ML_WINDOW)
    ref_ml = uniform_filter(np.where(ok, ref_z, 0.0), size=win)
    rec_ml = uniform_filter(out_std, size=win)
    inside = ok & (np.abs(ref_ml) < 1.0)
    outside = ok & ~inside
    se = (ref_ml - rec_ml) ** 2
    m["rmse_ml7_in_clamp_px"] = float(np.sqrt(se[inside].mean())) if inside.any() else None
    m["rmse_ml7_out_clamp_px"] = float(np.sqrt(se[outside].mean())) if outside.any() else None
    m["n_in_clamp_px"], m["n_out_clamp_px"] = int(inside.sum()), int(outside.sum())
    m["out_absmax"] = float(np.abs(out_std).max())
    m["out_frac_saturated"] = float((np.abs(out_std) >= 0.999).mean())
    return m


def paired(per, a, b, key, sel):
    """Over selected scenes: share where a < b (a better), and median a/b."""
    pr = [(x[key], y[key]) for x, y, s in zip(per[a], per[b], sel)
          if s and x.get(key) is not None and y.get(key) is not None and y[key] > 0]
    if not pr:
        return {"n": 0, "a_better": None, "median_ratio": None}
    pr = np.array(pr)
    return {"n": len(pr), "a_better": round(float((pr[:, 0] < pr[:, 1]).mean()), 4),
            "median_ratio": round(float(np.median(pr[:, 0] / pr[:, 1])), 4)}


def run_modality(mod, args, out_dir):
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    picks, scanned, pool_sizes = pick_scenes(mod, args.per_bin, args.max_shards, rng)
    rows = np.array([p[1] for p in picks]); zs = np.array([p[2] for p in picks])
    bins = np.array([p[0] for p in picks])
    print(f"[{mod}] scanned {scanned} scenes, bin pools {pool_sizes}, picked {len(rows)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    art = np.load(D.VAL / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")
    grids = np.asarray(art[rows]).astype(np.int64).reshape(-1, C.GRID, C.GRID)
    rasters = D.read_rasters(mod, rows)
    refs = [P.center_crop(a[0] if a.ndim == 4 else a).astype(np.float32) for a in rasters]

    tok = T.build(mod, device=T.DEVICE)
    doc = {"modality": mod, "rows": rows.tolist(), "z": zs.round(4).tolist(),
           "bins": bins.tolist(), "bin_edges": [list(b) for b in BINS],
           "pool_sizes": pool_sizes, "scanned": scanned, "claims": {}}

    # --- C0-C2 on the first batch ------------------------------------------------
    g0 = grids[:args.batch_size]
    base = T.decode(tok, torch.from_numpy(g0).to(T.DEVICE), timesteps=args.timesteps,
                    seed=args.seed).float().cpu()
    for name, ov in CLAIM_VARIANTS.items():
        o = decode(tok, g0, scheduler_like(tok, **ov), args.batch_size, args.seed, args.timesteps)
        doc["claims"][name] = {"max_abs_diff_vs_released": float((o - base).abs().max()),
                               "out_absmax": float(o.abs().max())}
    print(f"[{mod}] claims {json.dumps(doc['claims'])}", flush=True)

    # --- the sweep ------------------------------------------------------------
    per, decoded = {}, {}
    for name, ov in VARIANTS.items():
        t1 = time.time()
        sched = None if ov is None else scheduler_like(tok, **ov)
        out = decode(tok, grids, sched, args.batch_size, args.seed, args.timesteps).numpy()
        decoded[name] = out
        per[name] = [scene_metrics(out[i], refs[i], mod) for i in range(len(rows))]
        print(f"[{mod}] {name:9s} decoded {len(rows)} in {time.time() - t1:.0f}s", flush=True)
    doc["per_scene"] = per

    z = zs
    k = "rmse_z_ml7"
    allsel = np.ones(len(z), bool)
    doc["by_bin"] = {
        name: [{"bin": list(BINS[b]), "n": int((bins == b).sum()),
                "mean": (round(float(np.mean([r[k] for r, s in zip(per[name], bins == b) if s])), 4)
                         if (bins == b).any() else None)}
               for b in range(len(BINS))]
        for name in VARIANTS}
    c0 = doc["claims"]["copy"]["max_abs_diff_vs_released"] == 0.0
    c1 = doc["claims"]["clip1"]["max_abs_diff_vs_released"] <= 1e-6
    c2 = doc["claims"]["dyn10"]["out_absmax"] <= 1.0 + 1e-6
    hi = z >= 1.0
    c3 = bool(hi.any() and max(r["out_absmax"] for r, s in zip(per["noclip"], hi) if s) > 1.0)
    p4 = paired(per, "released", "clip0.5", k, z >= 0.5)
    p5 = paired(per, "noclip", "released", k, hi)
    p6 = paired(per, "noclip", "released", k, z < 0.5)
    p6px = paired(per, "noclip", "released", "rmse_ml7_in_clamp_px", allsel)
    p5c3 = paired(per, "clip3", "released", k, hi)
    doc["tests"] = {
        "C0_mechanism": c0, "C1_threshold_is_hard_clip": c1, "C2_dyn_cannot_widen": c2,
        "C3_clamp_gone": c3,
        "C4_control": {"pass": p4["n"] > 0 and p4["a_better"] >= 0.8, **p4},
        "C5_fix": {"pass": None if p5["n"] == 0 else p5["a_better"] >= 0.8, **p5},
        "C5_clip3": p5c3,
        "C6_no_harm_scene": {"pass": None if p6["n"] == 0 else p6["median_ratio"] <= 1.05, **p6},
        "C6_no_harm_pixels": {"pass": p6px["median_ratio"] is not None
                              and p6px["median_ratio"] <= 1.05, **p6px},
    }
    print(f"[{mod}] tests {json.dumps(doc['tests'])}", flush=True)

    # --- figure: one scene per bin, every panel on the reference's scale -------
    show = [int(np.flatnonzero(bins == b)[0]) for b in range(len(BINS)) if (bins == b).any()]
    cols = ["source"] + list(VARIANTS)
    fig, axes = plt.subplots(len(show), len(cols), figsize=(2.6 * len(cols), 2.6 * len(show)),
                             squeeze=False)
    for r, i in enumerate(show):
        panels = [refs[i][0]] + [
            P.destandardize(torch.from_numpy(decoded[v][i]), mod).numpy()[0] for v in VARIANTS]
        vmin = float(np.nanpercentile(panels[0], 2))
        vmax = max(float(np.nanpercentile(panels[0], 98)), vmin + 1e-6)
        for j, (im, title) in enumerate(zip(panels, cols)):
            ax = axes[r][j]
            ax.imshow(im, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if j:
                title = f"{title}\nml7 {per[cols[j]][i][k]:.3f}"
            ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"row {rows[i]}\n|z|={z[i]:.2f}", fontsize=8)
    fig.suptitle(f"{mod} -- ground-truth tokens, decoder sampler variants (channel 0)",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"fig_{mod}.png", dpi=120)
    plt.close(fig)

    doc["provenance"] = {"gpu": torch.cuda.get_device_name(0), "seed": args.seed,
                         "timesteps": args.timesteps, "batch_size": args.batch_size,
                         "crop": C.CROP, "tok_dir": C.tok_dir_name(mod),
                         "ml_window": D.ML_WINDOW,
                         "released_scheduler": {k2: v for k2, v in dict(
                             tok.noise_scheduler.config).items() if not k2.startswith("_")},
                         "seconds": round(time.time() - t0)}
    (out_dir / f"metrics_{mod}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    del tok; T.build.cache_clear(); torch.cuda.empty_cache()
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mods", nargs="*", default=MODS)
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--max-shards", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-fraction", type=float, default=0.45)
    ap.add_argument("--out", default="/data/enric/reports/clamp_test")
    args = ap.parse_args()

    print(f"gpu: {D.assert_gpu(False)}", flush=True)
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for mod in args.mods:
        f = out_dir / f"metrics_{mod}.json"
        doc = json.loads(f.read_text()) if f.exists() else run_modality(mod, args, out_dir)
        summary[mod] = {"tests": doc["tests"], "by_bin": doc["by_bin"],
                        "claims": doc["claims"], "pool_sizes": doc["pool_sizes"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("CLAMP TEST DONE")


if __name__ == "__main__":
    main()
