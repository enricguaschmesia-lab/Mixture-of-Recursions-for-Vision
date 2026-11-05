"""Is the reconstruction failure a decoder OUTPUT-RANGE clip, not a contract bug?

Hypothesis: the DiVAE decoder was configured with clip_sample=True /
thresholding=True, so its output is clamped to [-1, 1] in STANDARDIZED space.
Physical values whose z-score leaves [-1, 1] therefore cannot be reconstructed,
no matter how correct the preprocessing is.

Prediction 1: decoder output, expressed in standardized units, never leaves
              approximately [-1, 1].
Prediction 2: reconstruction error should track |z| of the input, and should
              hit DEM too -- for a high-elevation scene, which no NDVI-specific
              explanation would predict.
"""
import warnings, numpy as np, torch, pandas as pd
warnings.filterwarnings("ignore")
from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

SEED = 0
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
MT = META[META.tar.str.startswith("majortom")]


def dec(tok, g, ts=50):
    return tok.decode_tokens(g, timesteps=ts,
                             generator=torch.Generator().manual_seed(SEED))


print("=== Prediction 1: decoder output range in STANDARDIZED units ===")
print(f"  {'mod':6s} {'sample':24s} {'input z range':>22s} {'recon z range':>22s}")
for mod, idxs in [("NDVI", [0, 9]), ("DEM", [0, 9]), ("S2L2A", [0])]:
    tok = T.build(mod)
    for i in idxs:
        r = MT.iloc[i]
        stem = r.zarr[: -len(".zarr.zip")]
        _, arr = io.read_sample(mod, r.tar, stem)
        x, _ = P.prepare(arr, mod)
        out = dec(tok, T.encode(tok, x))[0].float()
        print(f"  {mod:6s} {stem:24s} "
              f"[{float(x.min()):+8.3f},{float(x.max()):+8.3f}] "
              f"[{float(out.min()):+8.3f},{float(out.max()):+8.3f}]")

print()
print("=== Prediction 2: does DEM fail on a HIGH-ELEVATION scene? ===")
print("  (v1 DEM stats: mean 670.665 m, std 951.272 m -> z=+1 at 1622 m,")
print("   so anything above ~1600 m should clip if the hypothesis holds)")
tok = T.build("DEM")
rows = []
for i in range(0, 30000, 700):
    r = MT.iloc[i]
    try:
        _, arr = io.read_sample("DEM", r.tar, r.zarr[: -len(".zarr.zip")])
    except (KeyError, FileNotFoundError):
        continue
    o, c = C.CROP_OFF, C.CROP
    ref = arr[0][:, o:o + c, o:o + c].astype(np.float32)
    x, _ = P.prepare(arr, "DEM")
    q = P.destandardize(dec(tok, T.encode(tok, x))[0].float(),
                        "DEM").cpu().numpy()
    z = (ref.mean() - C.V1_TOK_MEAN["DEM"][0]) / C.V1_TOK_STD["DEM"][0]
    rows.append((abs(z), ref.mean(), float(np.sqrt(((q - ref) ** 2).mean())),
                 ref.std()))
rows.sort()
print(f"  {'|z|':>6s} {'elev m':>9s} {'rmse m':>10s} {'scene std':>10s}  verdict")
for az, mu, e, sd in rows:
    v = "OK" if e < max(sd, 5) else "  <-- FAILS"
    print(f"  {az:6.2f} {mu:9.1f} {e:10.2f} {sd:10.2f}  {v}")
