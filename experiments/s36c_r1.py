"""R1 test: is the NDVI/DC failure caused by diffusers version drift?

Run twice -- once with the installed diffusers 0.40.0, once with 0.30.0
prepended on sys.path -- and compare the SAME tokens decoded by each.
Encoding never touches diffusers, so tokens must be byte-identical; only the
reconstruction may change.
"""
import sys, os
if os.environ.get("USE_DIFFUSERS_030") == "1":
    sys.path.insert(0, "/data/enric/cache/diffusers030")
import warnings, numpy as np, torch, pandas as pd
warnings.filterwarnings("ignore")
import diffusers
print(f"diffusers {diffusers.__version__}   (file: {diffusers.__file__[:52]}...)")

from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

SEED = 0
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
MT = META[META.tar.str.startswith("majortom")]
CASES = [MT[MT.center_lat.abs() < 8].iloc[0],       # NDVI mean 0.819 - failing
         MT[MT.center_lat.abs() > 55].iloc[0]]      # NDVI mean 0.246 - fine


def rmse(a, b): return float(np.sqrt(((a - b) ** 2).mean()))


for mod in ["NDVI", "DEM"]:
    tok = T.build(mod)
    print(f"\n{mod}")
    for r in CASES:
        stem = r.zarr[: -len(".zarr.zip")]
        _, arr = io.read_sample(mod, r.tar, stem)
        o, c = C.CROP_OFF, C.CROP
        ref = arr[0][:, o:o + c, o:o + c].astype(np.float32)
        x, _ = P.prepare(arr, mod)
        g = T.encode(tok, x)
        f = P.flatten_tokens(g)[0]
        line = (f"  {stem}  in_mean={ref.mean():+9.4f}  "
                f"tok_checksum={int(f.sum())}")
        for ts in (10, 50, 200):
            q = P.destandardize(
                tok.decode_tokens(g, timesteps=ts,
                                  generator=torch.Generator().manual_seed(SEED)
                                  )[0].float(), mod).cpu().numpy()
            line += f"  |ts{ts}: rmse={rmse(ref, q):8.4f} mean={q.mean():+8.4f}"
        print(line)
