"""Which DN convention does BigEarthNet v1 S2 use? (`mor` env, CPU; PHASE6_PLAN Step 0.4, partly)

Follow-up to the TerraMesh S2 BOA-offset finding (s2_offset_scan.py): ~26% of TerraMesh val S2L2A
scenes still carry ESA's +1000 DN offset. Downstream (BEN) inputs are in whichever convention BEN
uses, so measure it the same way.

Streams the S2 archive once (pigz -dc | tar stream), samples ~1 patch in STEP by a crc32 of the
patch name (deterministic), and records per patch: the darkest band's 1st percentile ("floor", as
in the TerraMesh scan), per-band means, the acquisition date, the labels, and whether every label
is a water class (the physical test: water is near 0 DN in SWIR without the offset, ~1000 with it).

    pigz -dc BigEarthNet-S2-v1.0.tar.gz | python ben_s2_offset_scan.py STEP OUT.parquet
"""
import io, json, sys, tarfile, zlib
import numpy as np, pandas as pd, tifffile

STEP, OUT = int(sys.argv[1]), sys.argv[2]
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
WATER = {"Sea and ocean", "Water bodies", "Water courses", "Coastal lagoons", "Estuaries"}

recs, cur, buf, n_seen = [], None, {}, 0


def flush(name, files):
    if name is None or len(files) < 13:
        return
    meta = json.loads(files["labels_metadata.json"])
    arrs = {b: tifffile.imread(io.BytesIO(files[f"{b}.tif"])).astype(np.float32) for b in BANDS}
    p1 = {b: float(np.percentile(a, 1)) for b, a in arrs.items()}
    r = {"patch": name, "date": meta.get("acquisition_date"), "labels": "|".join(meta["labels"]),
         "water_only": all(l in WATER for l in meta["labels"]), "floor": min(p1.values()),
         "dtype": str(tifffile.imread(io.BytesIO(files["B02.tif"])).dtype)}
    for b, a in arrs.items():
        r[f"mean_{b}"] = float(a.mean()); r[f"p1_{b}"] = p1[b]; r[f"med_{b}"] = float(np.median(a))
    r["shape_B02"] = str(arrs["B02"].shape); r["shape_B05"] = str(arrs["B05"].shape)
    recs.append(r)


with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tf:
    for m in tf:
        parts = m.name.split("/")
        if len(parts) != 3 or not m.isfile():
            continue
        patch = parts[1]
        if patch != cur:
            flush(cur, buf)
            cur, buf = patch, {}
            n_seen += 1
            if n_seen % 20000 == 0:
                print(n_seen, len(recs), flush=True)
        if zlib.crc32(patch.encode()) % STEP:
            continue
        buf[parts[2][len(patch) + 1:]] = tf.extractfile(m).read()
    flush(cur, buf)
pd.DataFrame(recs).to_parquet(OUT)
print("done", n_seen, "patches seen,", len(recs), "sampled ->", OUT, flush=True)
