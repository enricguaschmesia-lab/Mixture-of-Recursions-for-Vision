#!/usr/bin/env python
"""Per-patch content statistics for the held-out rows (depth-vs-content analysis).

⚠ RUNS IN THE `mor` CONDA ENV (numcodecs), CPU only:

    /data/enric/miniforge3/envs/mor/bin/python eo/scripts/patch_stats.py --modality LULC

For every held-out row carrying the modality, reads the raw raster, applies the
tokenizer's centre crop, and summarises each 16x16 patch -- the pixels behind
exactly one token, `token k <-> patch (k//14, k%14)`, row-major
(contract.FLATTEN_ORDER). The statistics are simple, named complexity proxies:

  LULC    n_classes, class entropy (nats), majority-class share, majority class
  DEM     elevation std (m), elevation range (m)
  NDVI    mean, std
  S2L2A   std of the band-mean reflectance (texture), band-mean brightness
  S1*     std of VV (dB) -- speckle + structure; mean VV

Writes /data/enric/reports/arm_eval/patch_stats/<MOD>.npz with `rows` (the
row ids, eval-split order) and one (n_rows, 196) float32 array per statistic,
NaN where the row does not carry the modality.

⚠ Only a tar MEMBER NAME is inspected before deciding to decode it, so the pass
costs one sequential read of each shard, not a decode of every sample.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from terramesh_tok import contract as C, io as tio  # noqa: E402

VAL = pathlib.Path("/data/enric/data/TerraMesh/val")
REPO = pathlib.Path(__file__).resolve().parents[2]


def patches(a: np.ndarray) -> np.ndarray:
    """(ch, 264, 264) -> (196, ch, 16, 16), centre-cropped, row-major."""
    o, c, p = C.CROP_OFF, C.CROP, 16
    x = a[:, o:o + c, o:o + c].astype(np.float32)
    g = c // p
    return x.reshape(x.shape[0], g, p, g, p).transpose(1, 3, 0, 2, 4).reshape(g * g, x.shape[0], p, p)


def stats(mod: str, a: np.ndarray) -> dict:
    a = a[0] if a.ndim == 4 else a
    P = patches(a)
    if mod == "LULC":
        cls = P[:, 0].reshape(196, -1).astype(np.int64)
        cnt = np.stack([(cls == k).sum(1) for k in range(C.LULC_N_CLASSES)], 1) / cls.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(cnt > 0, cnt * np.log(cnt), 0), 1)
        return {"n_classes": (cnt > 0).sum(1), "entropy": ent, "majority_share": cnt.max(1),
                "majority_class": cnt.argmax(1)}
    if mod == "DEM":
        v = P[:, 0].reshape(196, -1)
        return {"std": np.nanstd(v, 1), "range": np.nanmax(v, 1) - np.nanmin(v, 1)}
    if mod == "NDVI":
        v = P[:, 0].reshape(196, -1)
        return {"mean": np.nanmean(v, 1), "std": np.nanstd(v, 1)}
    if mod == "S2L2A":
        v = P.mean(1).reshape(196, -1)
        return {"texture": np.nanstd(v, 1), "brightness": np.nanmean(v, 1)}
    v = P[:, 0].reshape(196, -1)                         # S1: VV, dB
    return {"vv_std": np.nanstd(v, 1), "vv_mean": np.nanmean(v, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--out", default="/data/enric/reports/arm_eval/patch_stats")
    args = ap.parse_args()
    mod = args.modality

    doc = json.loads((REPO / "eo/data/eval_rows_v1.json").read_text())
    rows = np.asarray(doc["eval_rows"], dtype=np.int64)          # eval-split order, as the loader
    present = np.load(VAL / C.tok_dir_name(mod) / "present.npy")
    idx = pd.read_parquet(VAL / "tok_index.parquet").set_index("row")
    want = idx.loc[[r for r in rows if present[r]], ["stem", "source_shard"]]
    pos = {int(r): i for i, r in enumerate(rows)}

    out = {}
    done = 0
    import warnings
    warnings.filterwarnings("ignore")
    for shard, grp in want.groupby("source_shard"):
        stems = {s: int(r) for r, s in zip(grp.index.values, grp.stem.values)}
        with tarfile.open(VAL / mod / shard) as tf:
            for m in tf:
                s = m.name[: -len(".zarr.zip")] if m.name.endswith(".zarr.zip") else None
                if s not in stems:
                    continue
                st = stats(mod, tio._decode_zarr_zip(tf.extractfile(m).read()))
                for k, v in st.items():
                    out.setdefault(k, np.full((len(rows), 196), np.nan, np.float32))[pos[stems[s]]] = v
                done += 1
                del stems[s]
                if not stems:
                    break
        if stems:
            raise RuntimeError(f"{mod}/{shard}: {len(stems)} rows not found")
    o = pathlib.Path(args.out); o.mkdir(parents=True, exist_ok=True)
    with open(o / f"{mod}.npz", "wb") as fh:
        np.savez_compressed(fh, rows=rows, **out)
    print(f"{mod}: {done} rows, stats {sorted(out)} -> {o / (mod + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
