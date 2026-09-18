#!/usr/bin/env python
"""Step 4.8 (optional) -- Coords_tok.

NOT part of D1.5. Produced so open question #8 (is a non-image modality worth
having?) can be decided against a real artifact rather than an argument.

Two Step 3 findings are handled here:
  * 36 val samples have a latitude that rounds to '-0.00', which is not in the
    tokenizer vocabulary and becomes six [UNK] tokens. Normalising -0.0 -> 0.0
    before formatting fixes it.
  * IBM's CoordsTokenizer.encode() crashes on a batch whose members tokenize to
    different lengths (torch.tensor on a ragged list), so batches are only safe
    once every member is the same length.
"""
import json, os, time, warnings
os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
from terratorch.models.backbones.terramind.tokenizer.tokenizer_register import (
    terramind_v1_coords_tokenizer)

VAL = Path("/data/enric/data/TerraMesh/val")
OUT = VAL / C.tok_dir_name("Coords")   # unsuffixed: crop-independent
OUT.mkdir(parents=True, exist_ok=True)
index = pd.read_parquet(VAL / "tok_index.parquet")
meta = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
meta["stem"] = meta.zarr.str[: -len(".zarr.zip")]

# align coordinates to the canonical row order
m = index.merge(meta[["stem", "center_lon", "center_lat"]], on="stem", how="left")
assert len(m) == len(index) and m.center_lon.notna().all()

ct = terramind_v1_coords_tokenizer(pretrained=True)

# The -0.00 fix: snap to the tokenizer's own 0.25-degree grid first, then add
# 0.0 to turn any negative zero into positive zero before it is formatted.
lon = (np.round(m.center_lon.values * 4) / 4) + 0.0
lat = (np.round(m.center_lat.values * 4) / 4) + 0.0
lon[lon == 0] = 0.0
lat[lat == 0] = 0.0
coords = torch.tensor(np.stack([lon, lat], 1), dtype=torch.float32)

t0 = time.time()
rows, lens = [], {}
for i in range(0, len(coords), 512):
    chunk = coords[i:i + 512]
    try:
        t = ct.encode(chunk)["tensor"]
        lens[t.shape[1]] = lens.get(t.shape[1], 0) + t.shape[0]
        rows.append(t.cpu().numpy())
    except ValueError:                      # ragged -> fall back per sample
        for j in range(len(chunk)):
            t = ct.encode(chunk[j:j + 1])["tensor"]
            lens[t.shape[1]] = lens.get(t.shape[1], 0) + 1
            rows.append(t.cpu().numpy())

if len(set(lens)) != 1:
    raise SystemExit(f"still ragged after the -0.00 fix: {lens}")
tokens = np.concatenate(rows).astype(np.uint16)
n_tok = tokens.shape[1]
print(f"coords: {tokens.shape} in {time.time()-t0:.1f}s   lengths {lens}")
print(f"  id range {int(tokens.min())}..{int(tokens.max())}, "
      f"{len(np.unique(tokens))} distinct")

with open(OUT / "tokens.npy", "wb") as fh:
    np.save(fh, tokens)
with open(OUT / "present.npy", "wb") as fh:
    np.save(fh, np.ones(len(tokens), dtype=bool))
json.dump({
    "modality": "Coords", "tokenizer": "terramind_v1_coords_tokenizer",
    "repo": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-Coords",
    "tokens_per_sample": int(n_tok),
    "token_layout": "[lat, lon, EOS=3]",
    "grid_snap_degrees": 0.25,
    "observed_id_min": int(tokens.min()), "observed_id_max": int(tokens.max()),
    "vocab_size_needed": int(tokens.max()) + 1,
    "reported_get_vocab_size": int(ct.text_tokenizer.get_vocab_size()),
    "note": ("get_vocab_size() UNDER-reports: it is smaller than the largest id "
             "emitted, so Phase 2 must size the embedding slot from "
             "observed_id_max+1. The -0.00 latitude bug (36 samples -> six [UNK]) "
             "is fixed here by snapping to the 0.25deg grid and normalising "
             "negative zero before formatting."),
    "dtype": "uint16", "n_samples": int(len(tokens)),
    "row_order": "val/tok_index.parquet", "not_part_of": "D1.5",
    # Crop-independence, stated in the artifact rather than left to be inferred
    # from a directory listing. Coords is tokenized from the scene's
    # centre_lon/centre_lat, and every crop is centred on that same point, so
    # the value -- and every token derived from it -- is identical at any crop.
    # This is why Coords_tok/ keeps an unsuffixed name and carries an older date
    # than the image modalities: it is not re-tokenized when the crop changes.
    # A loader asserting metadata['crop'] against contract.CROP must treat null
    # as "applies at every crop", NOT as a missing field.
    "crop": None,
    "crop_independent": True,
    "crop_independent_reason": (
        "tokens derive from the scene centre coordinate, which is invariant to "
        "the crop size because every crop is centred on that point; see "
        "eo/terramesh_tok/contract.py CROP_INDEPENDENT"),
    "tok_dir_unsuffixed": True,
    "no_shards_subdir": ("written in a single pass, not per source tar -- the "
                         "absence of shards/ is by design, not a truncated run"),
    "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
}, open(OUT / "metadata.json", "w"), indent=1)
print(f"wrote {OUT}/tokens.npy ({tokens.nbytes/1e6:.2f} MB), present.npy, metadata.json")
