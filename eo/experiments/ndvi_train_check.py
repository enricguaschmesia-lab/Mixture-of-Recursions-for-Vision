"""Is TerraMesh's stored TRAIN NDVI exactly clean.ndvi_from_s2(stored S2L2A)? (2026-10-25, mor env, CPU)

If so, the train pipeline computes NDVI instead of downloading it (1.01 TB). Control: in the rows
val's offset rule labels, NDVI from offset-CORRECTED bands must differ from the stored NDVI.

    python eo/experiments/ndvi_train_check.py <shard>      (reads data/TerraMesh/train_probe/)

Result (worklog 2026-10-25), train majortom shard 1: 10,000/10,000 bit-identical; control 2,423/2,423."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/enric/code/Mixture-of-Recursions-for-Vision/eo")
from terramesh_tok import io as tio, clean as CL

root, shard = "/data/enric/data/TerraMesh/train_probe", sys.argv[1]
t = pd.read_parquet("/data/enric/data/TerraMesh/train_meta/train_metadata.parquet",
                    columns=["tar", "zarr", "sample_id", "S2_time"])
t = t[t.tar == shard].copy()
t["grid_cell"] = t.sample_id.str.split("_").str[:2].str.join("_")
t["timestamp"] = pd.to_datetime(t.S2_time).dt.strftime("%Y%m%dT%H%M%S")
mt = pd.read_parquet("/data/enric/data/MajorTOM/Core-S2L2A/metadata.parquet", columns=["grid_cell", "product_id", "timestamp"])
j = t.merge(mt, on=["grid_cell", "timestamp"], how="left")
j["baseline"] = j.product_id.str.extract(r"_N(\d{4})_")[0].astype(int)
j["off"] = [CL.has_offset(b, s) for b, s in zip(j.baseline, j.S2_time)]
lab = j.groupby("zarr").off.agg(lambda x: x.iloc[0] if x.nunique() == 1 else None)   # None = ambiguous
stem2off = {z[: -len(".zarr.zip")]: v for z, v in lab.items()}
print(f"{shard}: {len(stem2off)} rows, offset {sum(v is True for v in stem2off.values())}, ambiguous {sum(v is None for v in stem2off.values())}")

t0 = time.time()
n = same = diff_ctrl = n_off = floor_off = floor_clean = n_clean = 0
bad = []
for (s_stem, s2), (n_stem, nd) in zip(tio.iter_shard("S2L2A", shard, root=root), tio.iter_shard("NDVI", shard, root=root)):
    assert s_stem == n_stem, (s_stem, n_stem)
    off = stem2off[s_stem]
    rec = CL.ndvi_from_s2(s2)
    eq = rec.shape == nd.shape and np.array_equal(rec, nd.astype(np.float16), equal_nan=True)
    n += 1; same += eq
    if not eq and len(bad) < 5: bad.append((s_stem, off, rec.shape, nd.shape, nd.dtype))
    p1 = np.nanpercentile(s2.reshape(12, -1).astype(np.float64), 1, axis=1).min()
    if off is True:
        n_off += 1; floor_off += p1 > 900
        diff_ctrl += not np.array_equal(CL.ndvi_from_s2(CL.correct_s2(s2, True)), nd.astype(np.float16), equal_nan=True)
    elif off is False:
        n_clean += 1; floor_clean += p1 > 900
print(f"{n} rows in {time.time()-t0:.0f} s; stored NDVI == ndvi_from_s2(stored S2): {same}/{n}; examples of mismatch: {bad}")
print(f"CONTROL offset rows: NDVI from corrected bands differs from stored in {diff_ctrl}/{n_off}")
print(f"brightness cross-check (every band's 1st pct > 900 DN): offset rows {floor_off}/{n_off}, clean rows {floor_clean}/{n_clean}")
