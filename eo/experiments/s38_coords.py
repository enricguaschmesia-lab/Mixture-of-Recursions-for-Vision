"""Step 3.8: coords tokenizer feasibility probe (open question #8).

The builder is NOT registered in FULL_MODEL_REGISTRY -- unlike the six image
tokenizers -- so it must be imported and called directly. encode() takes a
[B, 2] tensor of [lon, lat], not strings.
"""
import os, warnings, torch, pandas as pd
os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")
from terratorch.models.backbones.terramind.tokenizer.tokenizer_register import (
    terramind_v1_coords_tokenizer)

ct = terramind_v1_coords_tokenizer(pretrained=True)
print(f"built: {type(ct).__name__}")
vocab = ct.text_tokenizer.get_vocab_size()
print(f"text tokenizer vocab size: {vocab}")

d = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
sub = d.iloc[[0, 10, 5000, 40000, 88000]]
coords = torch.tensor(sub[["center_lon", "center_lat"]].values,
                      dtype=torch.float32)
out = ct.encode(coords)
ids = out['tensor']
print(f"\nencode() returned {type(ids).__name__} shape "
      f"{tuple(ids.shape) if hasattr(ids,'shape') else len(ids)}")
print(f"\n  {'lon':>10s} {'lat':>10s}  -> tokens")
for (lon, lat), t in zip(coords.tolist(), ids):
    t = t.tolist() if hasattr(t, "tolist") else t
    print(f"  {lon:10.4f} {lat:10.4f}  -> {t}")

print(f"\n  tokens per sample: {ids.shape[1] if hasattr(ids,'shape') else '?'}")
print(f"  token id range across these samples: "
      f"{int(ids.min())}..{int(ids.max())}")
print("  NOTE: coords are snapped to a 0.25 degree grid before tokenizing,")
print("        so this is a ~28 km positional label, not a precise location.")
