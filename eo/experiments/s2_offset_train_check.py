"""Does val's S2 offset rule transfer to the TRAIN split? (2026-10-25, mor env, CPU)

s2_offset_scan.py's exact statistics (224 centre crop; floor = min over bands of the per-band 1st
percentile; median B12 over water pixels, cloud_mask 1) on one shard, split by clean.has_offset's
label (Major TOM baseline 05.00 acquired before 2022-01-25, matched on grid cell + S2 time).

    python eo/experiments/s2_offset_train_check.py <root> <shard> <metadata.parquet> <every-nth>

Result (worklog 2026-10-25): on val (the positive control) the labelled rows show the offset (floor
> 900 DN in 97%, water B12 ~1,028 DN); on train shards 1, 400 and 797 they look like clean rows (3.5-6%,
28-44 DN). The train split carries no offset."""
import sys, tarfile
import numpy as np, pandas as pd
sys.path.insert(0, "/home/enric/code/Mixture-of-Recursions-for-Vision/eo")
from terramesh_tok import io as tio, clean as CL
root, shard, meta, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
t = pd.read_parquet(meta, columns=["tar", "zarr", "sample_id", "S2_time"])
t = t[t.tar == shard].copy()
t["grid_cell"] = t.sample_id.str.split("_").str[:2].str.join("_")
t["timestamp"] = pd.to_datetime(t.S2_time).dt.strftime("%Y%m%dT%H%M%S")
mt = pd.read_parquet("/data/enric/data/MajorTOM/Core-S2L2A/metadata.parquet", columns=["grid_cell", "product_id", "timestamp"])
j = t.merge(mt, on=["grid_cell", "timestamp"], how="left")
j["baseline"] = j.product_id.str.extract(r"_N(\d{4})_")[0].astype(int)
j["off"] = [CL.has_offset(b, s) for b, s in zip(j.baseline, j.S2_time)]
j = j.drop_duplicates("zarr").set_index("zarr")
recs = []
k = 0
with tarfile.open(f"{root}/S2L2A/{shard}") as tf:
    for m in tf:
        if not m.name.endswith(".zarr.zip"):
            continue
        k += 1
        if (k - 1) % step:
            continue
        raw = tf.extractfile(m).read()
        b = tio._decode_zarr_zip(raw, "bands")[0][:, 20:244, 20:244].astype(np.float32)
        cm = tio._decode_zarr_zip(raw, "cloud_mask")[0][20:244, 20:244]
        w = cm == 1
        recs.append(dict(zarr=m.name, floor=float(np.percentile(b.reshape(12, -1), 1, axis=1).min()),
                         water_B12=float(np.median(b[11][w])) if w.sum() >= 50 else np.nan,
                         baseline=int(j.loc[m.name, "baseline"]), year=pd.Timestamp(j.loc[m.name, "S2_time"]).year,
                         off=bool(j.loc[m.name, "off"])))
d = pd.DataFrame(recs)
for lab, g in d.groupby("off"):
    print(f"label offset={lab}: n {len(g)} | floor>900 {np.mean(g.floor > 900):.3f} | floor median {g.floor.median():.0f} | water B12 median {g.water_B12.median():.0f} (n water {g.water_B12.notna().sum()}) | water B12>900 {np.mean(g.water_B12.dropna() > 900):.3f}")
print(d.assign(b5pre=(d.baseline == 500) & d.off).groupby(["baseline", "off"]).agg(n=("floor", "size"), floor_gt900=("floor", lambda x: round(np.mean(x > 900), 3)),
      water_B12_med=("water_B12", "median")).to_string())
