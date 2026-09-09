"""Does the ENCODER share the decoder's [-1,1] ceiling?

If it does, tokens for extreme scenes would be degenerate and Step 4 would be
producing meaningless tokens for mountains, dense vegetation, etc. If it does
not, the clipping is decoder-only and irrelevant to our pipeline, which only
ever runs the encoder.

Test: take one scene and offset it by increasing amounts (pushing the mean z
far past 1). If the encoder saturates, tokens stop changing.
"""
import warnings, numpy as np, torch, pandas as pd
warnings.filterwarnings("ignore")
from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
MT = META[META.tar.str.startswith("majortom")]
r = MT.iloc[0]
_, arr = io.read_sample("DEM", r.tar, r.zarr[: -len(".zarr.zip")])
tok = T.build("DEM")
base_x, _ = P.prepare(arr, "DEM")
prev = None
print("Offsetting a DEM scene to push its z-score far past the decoder's ceiling")
print(f"  {'offset m':>9s} {'mean z':>8s} {'uniq tokens':>12s} "
      f"{'changed vs prev':>16s} {'tok checksum':>13s}")
for off in [0, 500, 1000, 2000, 4000, 8000]:
    x = base_x + off / C.V1_TOK_STD["DEM"][0]
    f = P.flatten_tokens(T.encode(tok, x))[0]
    ch = "-" if prev is None else f"{int((f != prev).sum())}/256"
    print(f"  {off:9d} {float(x.mean()):8.3f} {len(torch.unique(f)):12d} "
          f"{ch:>16s} {int(f.sum()):13d}")
    prev = f

print()
print("Same test on NDVI (bounded physical range, so use small offsets)")
tok2 = T.build("NDVI")
_, arr2 = io.read_sample("NDVI", r.tar, r.zarr[: -len(".zarr.zip")])
bx, _ = P.prepare(arr2, "NDVI")
prev = None
print(f"  {'offset':>9s} {'mean z':>8s} {'uniq tokens':>12s} "
      f"{'changed vs prev':>16s} {'tok checksum':>13s}")
for off in [0.0, 0.2, 0.5, 1.0, 2.0]:
    x = bx + off / C.V1_TOK_STD["NDVI"][0]
    f = P.flatten_tokens(T.encode(tok2, x))[0]
    ch = "-" if prev is None else f"{int((f != prev).sum())}/256"
    print(f"  {off:9.2f} {float(x.mean()):8.3f} {len(torch.unique(f)):12d} "
          f"{ch:>16s} {int(f.sum()):13d}")
    prev = f
