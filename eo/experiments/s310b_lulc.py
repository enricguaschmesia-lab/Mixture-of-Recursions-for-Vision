"""Regenerate the LULC figure on a genuinely multi-class scene.

The first pass picked a single-class scene, which is trivially reconstructed
and proves nothing about the one-hot contract.
"""
import warnings, numpy as np, torch, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

FIG = "/data/enric/figures/step3"
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
MT = META[META.tar.str.startswith("majortom")]
tok = T.build("LULC")
o, c = C.CROP_OFF, C.CROP

best = None
print("searching for the most class-diverse LULC scene in a 60-sample sweep")
for i in range(0, 20000, 333):
    r = MT.iloc[i]
    try:
        _, arr = io.read_sample("LULC", r.tar, r.zarr[: -len(".zarr.zip")])
    except (KeyError, FileNotFoundError):
        continue
    ref = arr[0, 0][o:o + c, o:o + c]
    u, cnt = np.unique(ref, return_counts=True)
    # entropy over class frequencies -- rewards balanced multi-class scenes
    p = cnt / cnt.sum()
    h = float(-(p * np.log(p)).sum())
    if best is None or h > best[0]:
        best = (h, r, arr, len(u))
h, r, arr, nu = best
stem = r.zarr[: -len(".zarr.zip")]
print(f"  picked {stem}: {nu} classes, entropy {h:.3f}")

x, _ = P.prepare(arr, "LULC")
g = T.encode(tok, x)
rec = tok.decode_tokens(g)
ref = arr[0, 0][o:o + c, o:o + c].astype(np.int64)
pr = rec.argmax(1)[0].cpu().numpy()
acc = float((ref == pr).mean())
ious = []
for k in range(C.LULC_N_CLASSES):
    un = np.logical_or(ref == k, pr == k).sum()
    if un > 0:
        ious.append(np.logical_and(ref == k, pr == k).sum() / un)
print(f"  pixel accuracy {acc:.4f}   mIoU {np.mean(ious):.4f} "
      f"over {len(ious)} classes")
print(f"  input classes  {sorted(np.unique(ref).tolist())}")
print(f"  output classes {sorted(np.unique(pr).tolist())}")

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
ax[0].imshow(ref, cmap="tab10", vmin=0, vmax=9); ax[0].set_title(f"input classes ({nu} present)")
ax[1].imshow(pr, cmap="tab10", vmin=0, vmax=9); ax[1].set_title("decoded classes")
ax[2].imshow(ref != pr, cmap="Reds"); ax[2].set_title(f"disagreement ({100*(1-acc):.2f}% px)")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"LULC — {stem} — 256 crop, ViT decoder (no diffusion)\n"
             f"pixel accuracy {acc:.4f}, mIoU {np.mean(ious):.4f}  |  "
             f"one-hot 10-class contract", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/roundtrip_LULC.png", dpi=110)
print(f"  wrote {FIG}/roundtrip_LULC.png")
