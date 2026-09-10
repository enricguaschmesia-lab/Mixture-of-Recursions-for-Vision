"""Step 3.6b: fixed controls (consistent physical space) + NDVI diagnostic.

Fixes a methodology bug in the first control pass: the 'no standardization'
variant compared the decoder's output (which always lives in STANDARDIZED
space) against a raw-unit reference. Here every variant is destandardized with
the v1 stats before comparison, so all numbers are in physical units.
"""
import warnings, numpy as np, torch, pandas as pd
warnings.filterwarnings("ignore")
from skimage.metrics import structural_similarity as ssim_fn
from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

DEV, TS, SEED = T.DEVICE, 50, 0
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
MT = META[META.tar.str.startswith("majortom")]
S = [(r.tar, r.zarr[: -len(".zarr.zip")], r) for r in
     [MT[MT.center_lat.abs() < 8].iloc[0], MT[MT.center_lat.abs() > 55].iloc[0],
      MT.sort_values("cloud_cover", ascending=False).iloc[0]]]


def dec(tok, g, ts=TS):
    return tok.decode_tokens(g, timesteps=ts,
                             generator=torch.Generator().manual_seed(SEED))


def rmse(a, b): return float(np.sqrt(((a - b) ** 2).mean()))


def phys_ref(arr, mod, crop=None):
    o = (C.NATIVE - (crop or C.CROP)) // 2
    c = crop or C.CROP
    return arr[0][:, o:o + c, o:o + c].astype(np.float32)


print("=== 3.6b Controls, ALL in physical units (bug fixed) ===")
for mod in ["DEM", "NDVI", "S1RTC", "S2L2A"]:
    tok = T.build(mod)
    tar, stem, _ = S[0]
    _, arr = io.read_sample(mod, tar, stem)
    ref = phys_ref(arr, mod)
    row = {}

    # correct
    x, _ = P.prepare(arr, mod)
    q = P.destandardize(dec(tok, T.encode(tok, x))[0].float(), mod).cpu().numpy()
    row["correct"] = rmse(ref, q)

    # trivial baseline
    row["mean-baseline"] = rmse(ref, np.broadcast_to(
        ref.mean(axis=(1, 2), keepdims=True), ref.shape))

    # reversed bands: encode reversed, decode, un-reverse, compare to ref
    if C.N_CHANNELS[mod] > 1:
        xr, _ = P.prepare(arr, mod, reverse_bands=True)
        qr = P.destandardize(dec(tok, T.encode(tok, xr))[0].float(), mod)
        row["reversed-bands"] = rmse(ref, qr.cpu().numpy()[::-1].copy())

    # NO standardization: feed raw, but decoder output is still in
    # standardized space -> destandardize it before comparing.
    xn, _ = P.prepare(arr, mod, standardize=False)
    qn = P.destandardize(dec(tok, T.encode(tok, xn))[0].float(), mod)
    row["no-standardization"] = rmse(ref, qn.cpu().numpy())

    # wrong modality stats: standardize with other, decode, destandardize
    # with the SAME wrong stats (what a mistaken user would do)
    other = "NDVI" if mod != "NDVI" else "DEM"
    n = C.N_CHANNELS[mod]
    mw = torch.tensor((C.V1_TOK_MEAN[other] * n)[:n], device=DEV).view(1, -1, 1, 1)
    sw = torch.tensor((C.V1_TOK_STD[other] * n)[:n], device=DEV).view(1, -1, 1, 1)
    xw = (P.prepare(arr, mod, standardize=False)[0] - mw) / sw
    qw = (dec(tok, T.encode(tok, xw)) * sw + mw)[0].float().cpu().numpy()
    row[f"stats-of-{other}"] = rmse(ref, qw)

    print(f"  {mod:6s}")
    for k, v in row.items():
        r = v / row["correct"]
        tag = " <-- CORRECT" if k == "correct" else (
            "  !! NOT SEPARATED" if r < 1.5 else "")
        print(f"      {k:22s} {v:12.4f}  x{r:6.2f}{tag}")

print()
print("=== NDVI diagnostic: why does majortom_val_0000001 round-trip badly? ===")
tok = T.build("NDVI")
for tar, stem, p in S:
    _, arr = io.read_sample("NDVI", tar, stem)
    ref = phys_ref(arr, "NDVI")
    x, _ = P.prepare(arr, "NDVI")
    g = T.encode(tok, x)
    out = {}
    for ts in (10, 50, 200):
        q = P.destandardize(dec(tok, g, ts)[0].float(), "NDVI").cpu().numpy()
        out[ts] = (rmse(ref, q), float(q.mean()), float(q.std()))
    print(f"  {stem}  lat={p.center_lat:7.2f}")
    print(f"    input NDVI  mean={ref.mean():+.4f} std={ref.std():.4f} "
          f"min={ref.min():+.3f} max={ref.max():+.3f}")
    print(f"    standardized mean={float(x.mean()):+.4f} std={float(x.std()):.4f}")
    for ts, (e, m, s) in out.items():
        print(f"    ts={ts:3d}  rmse={e:.4f}  recon mean={m:+.4f} std={s:.4f}")

print()
print("=== Is the low-variance failure general? 12 more NDVI samples ===")
rows = []
for i in range(0, 12000, 1000):
    r = MT.iloc[i]
    try:
        _, arr = io.read_sample("NDVI", r.tar, r.zarr[: -len(".zarr.zip")])
    except (KeyError, FileNotFoundError):
        continue
    ref = phys_ref(arr, "NDVI")
    x, _ = P.prepare(arr, "NDVI")
    q = P.destandardize(dec(tok, T.encode(tok, x))[0].float(), "NDVI").cpu().numpy()
    rows.append((ref.std(), rmse(ref, q), ref.mean()))
rows.sort()
print(f"  {'input std':>10s} {'input mean':>11s} {'rmse':>9s} {'rmse/std':>9s}")
for sd, e, mu in rows:
    print(f"  {sd:10.4f} {mu:+11.4f} {e:9.4f} {e/sd:9.2f}"
          + ("   <-- worse than a constant" if e > sd else ""))
