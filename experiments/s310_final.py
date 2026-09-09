"""Final Step 3 pass: coords vocab bound, |z| distribution, review figures."""
import os, json, warnings, numpy as np, torch, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

FIG = "/data/enric/figures/step3"; os.makedirs(FIG, exist_ok=True)
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
MT = META[META.tar.str.startswith("majortom")]


def dec(tok, g, ts=50):
    return tok.decode_tokens(g, timesteps=ts,
                             generator=torch.Generator().manual_seed(0))


# ---------------------------------------------------------------- coords
print("=== Coords tokenizer: exact id range over ALL 89,088 val samples ===")
from terratorch.models.backbones.terramind.tokenizer.tokenizer_register import (
    terramind_v1_coords_tokenizer)
ct = terramind_v1_coords_tokenizer(pretrained=True)
coords_all = torch.tensor(META[["center_lon", "center_lat"]].values,
                          dtype=torch.float32)
lens, allids = {}, []
BAD = []
for i in range(0, len(coords_all), 256):
    chunk = coords_all[i:i + 256]
    try:
        t = ct.encode(chunk)["tensor"]
        lens[t.shape[1]] = lens.get(t.shape[1], 0) + t.shape[0]
        allids.append(t.flatten())
    except ValueError:
        # heterogeneous lengths in this chunk -> fall back to per-sample
        for j in range(len(chunk)):
            t = ct.encode(chunk[j:j + 1])["tensor"]
            L = t.shape[1]
            lens[L] = lens.get(L, 0) + 1
            allids.append(t.flatten())
            if L != 3:
                BAD.append((chunk[j].tolist(), t.flatten().tolist()))
allids = torch.cat(allids)
print(f"  token-count distribution over all {len(coords_all)} val samples: "
      f"{dict(sorted(lens.items()))}")
print(f"  id range {int(allids.min())}..{int(allids.max())}, "
      f"{len(torch.unique(allids))} distinct ids used")
print(f"  reported get_vocab_size() = {ct.text_tokenizer.get_vocab_size()}"
      f"  <-- SMALLER than the max id; Phase 2 must size the coords slot from "
      f"max_id+1 = {int(allids.max())+1}, not from get_vocab_size()")
if BAD:
    print(f"  !! {len(BAD)} samples do NOT produce 3 tokens. Examples:")
    for c, t in BAD[:5]:
        print(f"       lon={c[0]:9.4f} lat={c[1]:8.4f} -> {len(t)} tokens {t}")
    print("     ** IBM's CoordsTokenizer.encode() CRASHES on a batch whose "
          "members tokenize to different lengths (torch.tensor of ragged list).")

# ---------------------------------------------------------------- |z| dist
print()
print("=== How much of val lies outside the decoder's [-1,1] output range? ===")
print("  (encoder is unaffected; this bounds where RECONSTRUCTION is a valid check)")
zrep = {}
for mod, n in [("DEM", 500), ("NDVI", 500), ("S2L2A", 300)]:
    zs = []
    step = max(1, len(MT) // n)
    for i in range(0, len(MT), step):
        r = MT.iloc[i]
        try:
            _, arr = io.read_sample(mod, r.tar, r.zarr[: -len(".zarr.zip")])
        except (KeyError, FileNotFoundError):
            continue
        o, c = C.CROP_OFF, C.CROP
        a = arr[0][:, o:o + c, o:o + c].astype(np.float32)
        m = np.asarray(C.V1_TOK_MEAN[mod]).reshape(-1, 1, 1)
        s = np.asarray(C.V1_TOK_STD[mod]).reshape(-1, 1, 1)
        z = (a - m) / s
        zs.append((float(np.abs(z.mean())), float((np.abs(z) > 1).mean())))
        if len(zs) >= n:
            break
    sm = np.array([x[0] for x in zs]); px = np.array([x[1] for x in zs])
    zrep[mod] = dict(n=len(zs), scenes_mean_z_gt1=float((sm > 1).mean()),
                     median_px_frac_gt1=float(np.median(px)),
                     mean_px_frac_gt1=float(px.mean()))
    print(f"  {mod:6s} n={len(zs):4d}  scenes with |mean z|>1: "
          f"{100*(sm>1).mean():5.1f}%   pixels with |z|>1: "
          f"median {100*np.median(px):5.1f}%  mean {100*px.mean():5.1f}%")

# ---------------------------------------------------------------- figures
print()
print("=== Review figures (U1) ===")
# in-range samples: pick scenes whose mean |z| is small
PICKS = {}
for mod in ["DEM", "NDVI", "S2L2A", "S1RTC", "LULC"]:
    best = None
    for i in range(0, 4000, 137):
        r = MT.iloc[i]
        try:
            _, arr = io.read_sample(mod, r.tar, r.zarr[: -len(".zarr.zip")])
        except (KeyError, FileNotFoundError):
            continue
        o, c = C.CROP_OFF, C.CROP
        a = arr[0][:, o:o + c, o:o + c].astype(np.float32)
        if mod == "LULC":
            best = (0.0, r, arr); break
        m = np.asarray(C.V1_TOK_MEAN[mod]).reshape(-1, 1, 1)
        s = np.asarray(C.V1_TOK_STD[mod]).reshape(-1, 1, 1)
        zz = float(np.abs(((a - m) / s).mean()))
        var = float(a.std())
        if var > 1e-6 and (best is None or zz < best[0]):
            best = (zz, r, arr)
    PICKS[mod] = best

for mod, (zz, r, arr) in PICKS.items():
    tok = T.build(mod)
    stem = r.zarr[: -len(".zarr.zip")]
    x, _ = P.prepare(arr, mod)
    g = T.encode(tok, x)
    rec = dec(tok, g)
    o, c = C.CROP_OFF, C.CROP
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    if mod == "LULC":
        ref = arr[0, 0][o:o + c, o:o + c].astype(np.int64)
        pr = rec.argmax(1)[0].cpu().numpy()
        acc = float((ref == pr).mean())
        ax[0].imshow(ref, cmap="tab10", vmin=0, vmax=9); ax[0].set_title("input classes")
        ax[1].imshow(pr, cmap="tab10", vmin=0, vmax=9); ax[1].set_title("decoded classes")
        ax[2].imshow(ref != pr, cmap="Reds"); ax[2].set_title(f"disagreement ({100*(1-acc):.2f}% px)")
        sub = f"pixel accuracy {acc:.4f}  |  one-hot 10-class contract"
    else:
        ref = P.destandardize(x[0], mod).cpu().numpy()
        pr = P.destandardize(rec[0].float(), mod).cpu().numpy()
        b = {"S2L2A": 3}.get(mod, 0)
        lo, hi = np.percentile(ref[b], [2, 98])
        ax[0].imshow(ref[b], vmin=lo, vmax=hi, cmap="gray"); ax[0].set_title(f"input (band {b})")
        ax[1].imshow(pr[b], vmin=lo, vmax=hi, cmap="gray"); ax[1].set_title("reconstruction")
        im = ax[2].imshow(pr[b] - ref[b], cmap="RdBu_r"); ax[2].set_title("error")
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        rm = float(np.sqrt(((pr - ref) ** 2).mean()))
        sub = f"RMSE {rm:.4g} (physical units)  |  scene |mean z| = {zz:.2f} (in decoder range)"
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{mod} — {stem} — 256 crop, 50 DDIM steps\n{sub}", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/roundtrip_{mod}.png", dpi=110); plt.close(fig)
    print(f"  wrote roundtrip_{mod}.png   ({sub})")

# the clipping artifact, shown deliberately
tok = T.build("DEM")
hi_r = None
for i in range(0, 30000, 700):
    r = MT.iloc[i]
    try:
        _, arr = io.read_sample("DEM", r.tar, r.zarr[: -len(".zarr.zip")])
    except (KeyError, FileNotFoundError):
        continue
    o, c = C.CROP_OFF, C.CROP
    a = arr[0][:, o:o + c, o:o + c].astype(np.float32)
    z = (a.mean() - C.V1_TOK_MEAN["DEM"][0]) / C.V1_TOK_STD["DEM"][0]
    if z > 1.8:
        hi_r = (r, arr, a, z); break
if hi_r:
    r, arr, ref, z = hi_r
    x, _ = P.prepare(arr, "DEM")
    pr = P.destandardize(dec(tok, T.encode(tok, x))[0].float(), "DEM").cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    lo, hi = float(ref.min()), float(ref.max())
    ax[0].imshow(ref[0], vmin=lo, vmax=hi, cmap="terrain"); ax[0].set_title(f"input DEM ({ref.mean():.0f} m)")
    ax[1].imshow(pr[0], vmin=lo, vmax=hi, cmap="terrain"); ax[1].set_title(f"reconstruction ({pr.mean():.0f} m)")
    im = ax[2].imshow(pr[0] - ref[0], cmap="RdBu_r"); ax[2].set_title("error"); plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("DEM — KNOWN ARTIFACT, not a contract error —\n"
                 f"scene mean z = +{z:.2f} exceeds the decoder's [-1,1] output clamp "
                 f"(ceiling {C.V1_TOK_MEAN['DEM'][0]+C.V1_TOK_STD['DEM'][0]:.0f} m). "
                 "Tokens are unaffected.", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/artifact_DEM_clipping.png", dpi=110); plt.close(fig)
    print(f"  wrote artifact_DEM_clipping.png (mean {ref.mean():.0f} m, z=+{z:.2f})")

json.dump(zrep, open(f"{FIG}/z_distribution.json", "w"), indent=1)
print(f"\nwrote {FIG}/z_distribution.json")
