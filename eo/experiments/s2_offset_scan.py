"""Scan TerraMesh val S2L2A for the +1000 BOA_ADD_OFFSET signature (mor env).

Every STEP-th scene of every shard, 224 centre crop: low percentiles per band, water-pixel
medians (cloud_mask class 1), and the scene mean of the visible bands.
"""
import sys, tarfile, zipfile, io as _io, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/enric/code/Mixture-of-Recursions-for-Vision/eo")
from terramesh_tok import io as tio

ROOT = "/data/enric/data/TerraMesh/val"
STEP = int(sys.argv[1]); OUT = sys.argv[2]
shards = sorted(pd.read_parquet(f"{ROOT}/../val_metadata.parquet").tar.unique())
recs = []
for sh in shards:
    with tarfile.open(f"{ROOT}/S2L2A/{sh}") as tf:
        k = 0
        for m in tf:
            if not m.name.endswith(".zarr.zip"):
                continue
            k += 1
            if (k - 1) % STEP:
                continue
            raw = tf.extractfile(m).read()
            b = tio._decode_zarr_zip(raw, "bands")[0][:, 20:244, 20:244].astype(np.float32)
            cm = tio._decode_zarr_zip(raw, "cloud_mask")[0][20:244, 20:244]
            r = {"zarr": m.name, "tar": sh}
            p1 = np.percentile(b.reshape(12, -1), 1, axis=1)
            for i, n in enumerate(["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]):
                r[f"p1_{n}"] = float(p1[i])
            r["min_all"] = float(b.min())
            r["vis_mean"] = float(b[1:4].mean())
            w = cm == 1
            r["water_frac"] = float(w.mean())
            r["water_B08"] = float(np.median(b[7][w])) if w.sum() >= 50 else np.nan
            r["water_B12"] = float(np.median(b[11][w])) if w.sum() >= 50 else np.nan
            r["cloud_frac"] = float(np.isin(cm, [3, 4]).mean())
            recs.append(r)
    print(sh, len(recs), flush=True)
pd.DataFrame(recs).to_parquet(OUT)
print("done", OUT, flush=True)
