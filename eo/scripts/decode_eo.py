#!/usr/bin/env python
"""Token grids -> pixels, with the ceiling and the collapse control (Step 7, side B).

⚠ RUNS IN THE `mor` CONDA ENV, NOT `.venv`:

    source /data/enric/miniforge3/etc/profile.d/conda.sh && conda activate mor

It imports terratorch (the DiVAE / ViT decoders) and therefore cannot import
anything from `eo/data/` -- not the dataset, not `eo_vocab`. Its only shared
import is `eo/terramesh_tok/contract.py`, which is pure constants and safe in
either env. Generated ids are converted to LOCAL codebook ids by
`prepare_decode.py` on the `.venv` side, precisely so that this side never
needs to know the vocabulary layout.

WHAT IT DECODES. Three token sources per run, through the SAME decoder with
the same seed, over the SAME scenes:

  ceiling    the GROUND-TRUTH tokens for these rows, straight from the
             artifact. Plan 7.4: this is the best any model could do at this
             tokenization, and it is the only sensible reference. Comparing a
             generation against the original raster instead silently charges
             tokenizer loss to the model.
  generated  the model's tokens, repaired and masked by prepare_decode.py.
  shuffled   the ceiling's tokens, spatially permuted within each scene. Plan
             7.5 / gate check W7: if the metric does not collapse here, it is
             not measuring spatial structure and no other number in this file
             means anything.

⚠ THE SHUFFLED CONTROL IS WEAKEST EXACTLY WHERE THE SCENE IS SIMPLEST. On a
single-class land-cover patch a permutation is the IDENTITY -- same tokens,
same picture -- so such a scene contributes pixel_acc 1.0 and mIoU 1.0 to the
control and pulls the collapse towards "no collapse". Measured on the 16
stratified held-out scenes (2026-09-23): several are effectively one class, and
the 1.76x mIoU collapse comes entirely from the multi-class ones. The control
is therefore a FLOOR on the collapse, not an estimate of it. `per_scene` in the
metrics artifact carries `n_classes_true`, so the collapse can be re-read over
the multi-class subset without re-decoding.

⚠ THE DECODER CLAMP DECIDES WHICH SCENES COUNT (plan 7.2, contract.py). The
DiVAE decoders clamp to [-1,1] in standardized space, so scenes with |z| >~ 1
reconstruct badly BY DESIGN -- DEM 9.8%, S2L2A 12.7%, NDVI 23.6% of val. The
continuous-modality RMSE is therefore reported over IN-RANGE scenes, with the
all-scenes number printed beside it so the restriction is visible rather than
quietly improving the result. LULC is categorical, decoded by a ViT with no
diffusion and no clamp, and is not filtered.

⚠ GPU. Arm A holds the TITAN V for ~3.7 days. This script asserts which card
it is on and refuses the TITAN V unless --allow-titanv is passed, because
`tokenizers.DEVICE` is the string "cuda:0" and which physical card that names
depends entirely on CUDA_VISIBLE_DEVICES. Run it as:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python eo/scripts/decode_eo.py ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import warnings

os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T  # noqa: E402

VAL = pathlib.Path(os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val"))

#: Multilook window, in pixels, for the continuous headline metric `rmse_z_ml7`.
#: FIXED 2026-09-24, before any arm-B number existed. Chosen on the identity
#: controls (eo/experiments/s1_metric_test.py, 64 scenes/modality), so it is
#: not blind: 5 and 7 are the conventional SAR multilook sizes and 7 sat near
#: the best of those tried. Do not retune it once arm B has produced numbers.
ML_WINDOW = 7
SOURCES = ("ceiling", "generated", "shuffled")


# ---------------------------------------------------------------- helpers

def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assert_gpu(allow_titanv: bool) -> str:
    """Name the card, and refuse the one arm A is training on.

    CLAUDE.md: the two index orderings are inverted and picking the wrong card
    is silent. Here it would be worse than slow -- it would contend with a
    3.7-day training run for memory that preflight already guards.
    """
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f"{torch.cuda.device_count()} visible CUDA devices. Pin exactly one with "
            f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<0|1>; with more than "
            f"one visible, 'cuda:0' is ambiguous.")
    name = torch.cuda.get_device_name(0)
    if "TITAN V" in name and not allow_titanv:
        raise SystemExit(
            f"refusing to run on the {name}: arm A is expected to be training there "
            f"(handover section 3a). Use CUDA_VISIBLE_DEVICES=0 under PCI_BUS_ID for "
            f"the GTX Titan X, or pass --allow-titanv if the run has finished.")
    return name


def read_rasters(modality: str, rows: np.ndarray):
    """Raw arrays for `rows`, one tar pass per shard rather than per scene.

    io.read_sample scans a tar from the start for each call, so asking it for
    N scattered scenes reopens and rescans a shard N times. The eval rows are
    scattered by construction (a geographic group split), so this matters.
    """
    import pandas as pd
    idx = pd.read_parquet(VAL / "tok_index.parquet").set_index("row")
    want = idx.loc[rows, ["stem", "source_shard"]]

    out: dict[int, np.ndarray] = {}
    for shard, grp in want.groupby("source_shard"):
        stems = {s: int(r) for r, s in zip(grp.index.values, grp.stem.values)}
        remaining = set(stems)
        for stem, arr in tio.iter_shard(modality, shard):
            if stem in remaining:
                out[stems[stem]] = arr
                remaining.discard(stem)
                if not remaining:
                    break
    missing = [int(r) for r in rows if int(r) not in out]
    if missing:
        raise RuntimeError(f"{modality}: no raster found for rows {missing[:8]}")
    return [out[int(r)] for r in rows]


def scene_z(mod: str, arr: np.ndarray) -> float:
    """Scalar |z| of a scene, identical to verify_step5.scene_z.

    Restated rather than imported because verify_step5.py is a script, not a
    module; the definition is the one established in eo/experiments/s36d_clip.py
    and it must not drift between the two.
    """
    o, c = C.CROP_OFF, C.CROP
    a = arr[0] if arr.ndim == 4 else arr
    ref = a[:, o:o + c, o:o + c].astype(np.float32)
    per_ch_mean = np.nanmean(ref, axis=(1, 2))
    z = (per_ch_mean - np.array(C.V1_TOK_MEAN[mod])) / np.array(C.V1_TOK_STD[mod])
    return float(np.nanmean(np.abs(z)))


def decode_batched(tok, grids: np.ndarray, batch: int, seed: int, device: str,
                   timesteps: int):
    """(N, GRID, GRID) local ids -> decoder output, in batches.

    ⚠ The seed is re-applied per batch by T.decode, so a scene's DiVAE noise
    depends on the batch it landed in. Keep --batch-size fixed when comparing
    two runs; LULC's ViT decoder ignores timesteps and is unaffected.
    """
    outs = []
    for i in range(0, len(grids), batch):
        g = torch.from_numpy(grids[i:i + batch].astype(np.int64)).to(device)
        outs.append(T.decode(tok, g, timesteps=timesteps, seed=seed).float().cpu())
    return torch.cat(outs) if outs else torch.empty(0)


# ---------------------------------------------------------------- metrics

def lulc_metrics(pred_cls: np.ndarray, true_cls: np.ndarray) -> dict:
    """Pixel accuracy and mIoU over the classes PRESENT IN GROUND TRUTH.

    Averaging IoU over all 10 classes instead would reward a model for
    correctly not predicting a class that does not occur in the scene, which
    on a 224x224 land-cover patch is most of them.
    """
    acc = float((pred_cls == true_cls).mean())
    ious = []
    for c in np.unique(true_cls):
        p, t = pred_cls == c, true_cls == c
        union = np.logical_or(p, t).sum()
        if union:
            ious.append(float(np.logical_and(p, t).sum() / union))
    return {"pixel_acc": round(acc, 4),
            "miou": round(float(np.mean(ious)), 4) if ious else None,
            "n_classes_true": int(len(np.unique(true_cls)))}


def continuous_metrics(recon_std: np.ndarray, recon_phys: np.ndarray,
                       ref: np.ndarray, mod: str) -> dict:
    """Three RMSEs: channel-0 physical, all-channel standardized, and multilooked.

    `rmse` (channel 0, physical) matches verify_step5 and is what W7 and the
    Step 7 tables were built on. It is kept, but it is a poor headline for the
    multi-band modalities: S2L2A's channel 0 is B01, the 60 m coastal-aerosol
    band -- upsampled, smooth, and the band least able to show spatial error --
    and S1's is VV alone. `rmse_z` scores every channel in the standardized
    space the tokenizer sees (V1_TOK_MEAN/STD), so one number covers all bands
    with equal weight and the five modalities land on a common scale.

    `rmse_z_ml7` is `rmse_z` after a ML_WINDOW x ML_WINDOW box filter on both
    images -- the standard SAR multilook -- and it is the continuous HEADLINE
    and W7 metric. ⚠ Measured 2026-09-24: on unfiltered pixels S1RTC's
    shuffle control collapses only 1.20x (1.15x on channel-0 `rmse`), failing
    W7, because per-pixel error against raw SAR is dominated by speckle that no
    token arrangement can reproduce. Multilooked it collapses 1.46x, S1GRD
    1.85x, and DEM is unchanged (6.8x -> 7.0x). The model chooses one token per
    16 px patch; detail finer than that is the decoder's, so smoothing below
    the token scale removes decoder noise and speckle, not model signal.
    NaN pixels are filled with the mean (0 in z, = contract.NAN_POLICY) so the
    filter can run, then excluded from the error.
    """
    out = {}
    r0, c0 = ref[0], recon_phys[0]
    valid = ~np.isnan(r0)
    out["rmse"] = (round(float(np.sqrt(np.mean((r0[valid] - c0[valid]) ** 2))), 4)
                   if valid.any() else None)
    mean = np.array(C.V1_TOK_MEAN[mod], dtype=np.float32)[:, None, None]
    std = np.array(C.V1_TOK_STD[mod], dtype=np.float32)[:, None, None]
    ref_z = (ref - mean) / std
    ok = ~np.isnan(ref_z)
    out["rmse_z"] = (round(float(np.sqrt(np.mean((ref_z[ok] - recon_std[ok]) ** 2))), 4)
                     if ok.any() else None)
    win = (1, ML_WINDOW, ML_WINDOW)
    ref_ml = uniform_filter(np.where(ok, ref_z, 0.0), size=win)
    rec_ml = uniform_filter(recon_std, size=win)
    out["rmse_z_ml7"] = (round(float(np.sqrt(np.mean((ref_ml[ok] - rec_ml[ok]) ** 2))), 4)
                         if ok.any() else None)
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode-dir", required=True,
                    help="directory written by prepare_decode.py")
    ap.add_argument("--target", default=None)
    ap.add_argument("--tag", default=None, help="output subdirectory name")
    ap.add_argument("--out-root", default="/data/enric/reports/phase3_step7")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-figure-scenes", type=int, default=6)
    ap.add_argument("--z-threshold", type=float, default=1.0,
                    help="|z| above which a scene is outside the decoder clamp")
    ap.add_argument("--allow-titanv", action="store_true")
    args = ap.parse_args()

    device = T.DEVICE
    gpu = assert_gpu(args.allow_titanv)
    print(f"gpu               : {gpu}")

    ddir = pathlib.Path(args.decode_dir)
    target = args.target
    if target is None:
        hits = sorted(ddir.glob("decode_grids_*.npy"))
        if len(hits) != 1:
            raise SystemExit(f"expected one decode_grids_*.npy in {ddir}, got "
                             f"{[h.name for h in hits]}; pass --target")
        target = hits[0].stem[len("decode_grids_"):]

    grids_path = ddir / f"decode_grids_{target}.npy"
    manifest = json.loads((ddir / f"decode_manifest_{target}.json").read_text())
    gen_grids = np.load(grids_path)
    valid = np.load(ddir / f"decode_valid_{target}.npy")
    rows = np.load(ddir / f"decode_rows_{target}.npy")

    if manifest["crop"] != C.CROP:
        raise SystemExit(
            f"grids were prepared at crop {manifest['crop']} but this env's contract "
            f"is {C.CROP}. Decoding them would compare different geometry.")

    print(f"target            : {target}   scenes {len(rows)}")
    print(f"repaired tokens   : {100 * manifest['repair']['repaired_fraction']:.2f}%")

    # --- the three token sources -----------------------------------------
    art = np.load(VAL / C.tok_dir_name(target) / "tokens.npy", mmap_mode="r")
    truth = np.asarray(art[rows]).astype(np.int64).reshape(-1, C.GRID, C.GRID)

    rng = np.random.default_rng(args.seed)
    shuffled = np.stack([
        s.ravel()[rng.permutation(C.TOKENS_PER_SAMPLE)].reshape(C.GRID, C.GRID)
        for s in truth])

    sources = {"ceiling": truth, "generated": gen_grids, "shuffled": shuffled}

    # --- reference rasters and the clamp filter --------------------------
    rasters = read_rasters(target, rows)
    if target == "LULC":
        in_range = np.ones(len(rows), dtype=bool)
        zs = [None] * len(rows)
    else:
        zs = [scene_z(target, a) for a in rasters]
        in_range = np.array([z < args.z_threshold for z in zs])
    print(f"in-range scenes   : {int(in_range.sum())}/{len(rows)}"
          f"  (|z| < {args.z_threshold}, LULC unfiltered)")

    # --- decode ----------------------------------------------------------
    tok = T.build(target, device=device)
    decoded, per_scene = {}, {s: [] for s in SOURCES}
    for name in SOURCES:
        print(f"  decoding {name} ...", flush=True)
        out = decode_batched(tok, sources[name], args.batch_size, args.seed,
                             device, args.timesteps)
        decoded[name] = out
        for i, arr in enumerate(rasters):
            ref = P.center_crop(arr[0] if arr.ndim == 4 else arr)
            if target == "LULC":
                m = lulc_metrics(out[i].argmax(0).numpy(), ref[0].astype(np.int64))
            else:
                m = continuous_metrics(
                    out[i].numpy(), P.destandardize(out[i], target).numpy(),
                    ref.astype(np.float32), target)
            m.update(row=int(rows[i]), z=zs[i], in_range=bool(in_range[i]),
                     valid_fraction=round(float(valid[i].mean()), 4))
            per_scene[name].append(m)

    # --- aggregate -------------------------------------------------------
    def agg(name):
        rs = per_scene[name]
        keep = [r for r in rs if r["in_range"]]
        out = {"n_scenes": len(rs), "n_in_range": len(keep)}
        for key in ("pixel_acc", "miou", "rmse", "rmse_z", "rmse_z_ml7"):
            vals = [r[key] for r in keep if r.get(key) is not None]
            allv = [r[key] for r in rs if r.get(key) is not None]
            if allv:
                out[key] = round(float(np.mean(vals)), 4) if vals else None
                out[f"{key}_all_scenes"] = round(float(np.mean(allv)), 4)
        return out

    summary = {s: agg(s) for s in SOURCES}
    key = "pixel_acc" if target == "LULC" else "rmse_z_ml7"

    # ⚠ THE HEADLINE METRIC AND THE CONTROL METRIC ARE NOT THE SAME ONE, and
    # this is a measurement, not a preference. Measured on the identity control
    # (2026-09-23, 16 held-out scenes): shuffling the token grid moves LULC
    # pixel accuracy only 0.986 -> 0.824, because a land-cover scene is
    # dominated by one or two classes and a spatial permutation preserves their
    # PROPORTIONS -- most pixels stay right by coincidence. mIoU over the
    # classes present falls 0.852 -> 0.485 on the same decode, because it
    # weights exactly the rare classes that a permutation destroys.
    #
    # So pixel accuracy is reported (it is what a reader understands at a
    # glance) but mIoU is what W7 gates on. Gating on pixel accuracy would
    # have made the control unfalsifiable: a metric with a 0.82 floor and a
    # 1.0 ceiling cannot collapse by any meaningful margin.
    collapse_key = "miou" if target == "LULC" else "rmse_z_ml7"
    better = max if collapse_key in ("pixel_acc", "miou") else min

    print()
    print(f"{'source':11s} {'n':>4s} {'in-range':>9s}  {key}")
    for s in SOURCES:
        v = summary[s].get(key)
        print(f"{s:11s} {summary[s]['n_scenes']:4d} {summary[s]['n_in_range']:9d}  "
              f"{v if v is not None else 'n/a'}"
              + (f"   miou {summary[s].get('miou')}" if target == "LULC"
                 else f"   rmse_z {summary[s].get('rmse_z')}   rmse(ch0) {summary[s].get('rmse')}"))

    # W7: the control has to collapse, or the metric is not measuring structure
    c, sh = summary["ceiling"].get(collapse_key), summary["shuffled"].get(collapse_key)
    collapsed = (c is not None and sh is not None and better(c, sh) == c and c != sh)
    ratio = (c / sh if better is max else sh / c) if (c and sh) else None

    # The ratio alone under-reads a modality with a high irreducible floor:
    # S1RTC's shuffle is worse on EVERY scene yet the mean ratio is small. So
    # the paired per-scene view is recorded beside it -- win = share of
    # in-range scenes where the shuffle scores worse, d = mean gap / its SD.
    pairs = [(a[collapse_key], b[collapse_key])
             for a, b in zip(per_scene["ceiling"], per_scene["shuffled"])
             if a["in_range"] and a.get(collapse_key) is not None
             and b.get(collapse_key) is not None]
    gaps = np.array([(a - b) if better is max else (b - a) for a, b in pairs])
    paired = {"n": len(gaps),
              "win": round(float((gaps > 0).mean()), 4) if len(gaps) else None,
              "d": (round(float(gaps.mean() / gaps.std(ddof=1)), 3)
                    if len(gaps) > 1 and gaps.std(ddof=1) > 0 else None)}
    print()
    print(f"W7 control: shuffled {collapse_key} "
          f"{'COLLAPSES' if collapsed else 'DOES NOT COLLAPSE'} against the ceiling "
          f"({sh} vs {c}" + (f", {ratio:.2f}x)" if ratio else ")"))
    print(f"            paired over {paired['n']} in-range scenes: "
          f"win {paired['win']}  d {paired['d']}")
    if target == "LULC":
        print(f"            (pixel_acc moves only "
              f"{summary['shuffled'].get('pixel_acc')} -> {summary['ceiling'].get('pixel_acc')}; "
              f"class-dominated scenes, see the comment in this script)")

    # --- figure ----------------------------------------------------------
    tag = args.tag or ddir.name
    out_dir = pathlib.Path(args.out_root) / f"{target}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_fig = min(args.max_figure_scenes, len(rows))
    cols = ["source"] + list(SOURCES)
    fig, axes = plt.subplots(n_fig, len(cols), figsize=(3 * len(cols), 3 * n_fig),
                             squeeze=False)
    for i in range(n_fig):
        ref = P.center_crop(rasters[i][0] if rasters[i].ndim == 4 else rasters[i])
        if target == "LULC":
            panels = [ref[0].astype(np.int64)] + [
                decoded[s][i].argmax(0).numpy() for s in SOURCES]
            kw = dict(cmap="tab20", vmin=0, vmax=C.LULC_N_CLASSES - 1)
        else:
            panels = [ref[0].astype(np.float32)] + [
                P.destandardize(decoded[s][i], target).numpy()[0] for s in SOURCES]
            vmax = float(np.nanpercentile(np.abs(panels[0]), 98))
            kw = dict(cmap="viridis", vmin=None, vmax=vmax)
        for j, (im, title) in enumerate(zip(panels, cols)):
            ax = axes[i][j]
            ax.imshow(im, **kw)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
            if j == 0:
                lbl = f"row {rows[i]}"
                if zs[i] is not None:
                    lbl += f"\n|z|={zs[i]:.2f}"
                ax.set_ylabel(lbl, fontsize=8)
    fig.suptitle(f"{target} -- {tag}", fontsize=11)
    # rect keeps the suptitle clear of the column headers; tight_layout alone
    # lays it over the first row's titles.
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    fig_path = out_dir / f"step7_{target}_{tag}.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # --- metrics artifact ------------------------------------------------
    doc = {
        "target": target,
        "tag": tag,
        "metric": key,
        "collapse_metric": collapse_key,
        "collapse_higher_is_better": better is max,
        "summary": summary,
        "w7_shuffled_collapses": bool(collapsed),
        "w7_paired": paired,
        "ml_window": ML_WINDOW,
        "z_threshold": args.z_threshold,
        "per_scene": per_scene,
        "figure": str(fig_path),
        "provenance": {
            # W7 reads this artifact rather than re-decoding (wrong env), so it
            # must be able to tell a stale file from a fresh one.
            "decode_dir": str(ddir),
            "grids_sha256": sha256(grids_path),
            "gpu": gpu, "seed": args.seed, "timesteps": args.timesteps,
            "crop": C.CROP, "tok_dir": C.tok_dir_name(target),
            "manifest": manifest,
        },
    }
    m_path = out_dir / f"metrics_{target}.json"
    m_path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n-> {fig_path}\n-> {m_path}")
    return 0 if collapsed else 2


if __name__ == "__main__":
    raise SystemExit(main())
