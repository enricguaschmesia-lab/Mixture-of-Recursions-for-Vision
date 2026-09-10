#!/usr/bin/env python
"""Diagnose the batch-size sensitivity the equivalence gate caught.

Questions, in order of what would change the decision:
  A. How BIG is it?  fraction of tokens differing, bs=1 vs bs=8/32.
  B. Is it deterministic AT a fixed batch size?  (run twice)
  C. Is it a bin-boundary effect?  compare the pre-quantization latents.
  D. Do determinism knobs remove it?
"""
import os, warnings
os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")
import numpy as np, torch, pandas as pd
from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T

DEV = T.DEVICE
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
MT = META[META.tar.str.startswith("majortom")].head(32)


@torch.no_grad()
def enc(mod, tok, arrs, bs, latents=False):
    outs, lats, buf = [], [], []
    def flush():
        nonlocal buf
        x = torch.cat(buf).to(DEV)
        q, _, t = tok.encode(x)
        outs.append(P.flatten_tokens(t).cpu().numpy())
        if latents:
            lats.append(q.float().cpu().numpy())
        buf = []
    for a in arrs:
        buf.append(P.prepare(a, mod, device="cpu")[0])
        if len(buf) >= bs:
            flush()
    if buf:
        flush()
    tk = np.concatenate(outs)
    return (tk, np.concatenate(lats)) if latents else tk


print("=== A. magnitude of the batch-size effect ===")
print(f"  {'mod':6s} {'bs1 vs bs8':>22s} {'bs1 vs bs32':>22s}")
cache = {}
for mod in ["DEM", "NDVI", "LULC", "S1RTC", "S2L2A"]:
    tok = T.build(mod)
    arrs = [tio.read_sample(mod, r.tar, r.zarr[: -len(".zarr.zip")])[1]
            for _, r in MT.iterrows()]
    cache[mod] = (tok, arrs)
    a1 = enc(mod, tok, arrs, 1)
    a8, a32 = enc(mod, tok, arrs, 8), enc(mod, tok, arrs, 32)
    n = a1.size
    print(f"  {mod:6s} {int((a1!=a8).sum()):6d}/{n} ({100*(a1!=a8).mean():6.3f}%) "
          f"{int((a1!=a32).sum()):9d}/{n} ({100*(a1!=a32).mean():6.3f}%)")

print()
print("=== B. deterministic AT a fixed batch size? (bs=32 run twice) ===")
for mod in ["DEM", "NDVI", "LULC", "S1RTC", "S2L2A"]:
    tok, arrs = cache[mod]
    x, y = enc(mod, tok, arrs, 32), enc(mod, tok, arrs, 32)
    print(f"  {mod:6s} {'IDENTICAL' if np.array_equal(x, y) else 'NONDETERMINISTIC'}")

print()
print("=== C. is it an FSQ bin-boundary effect? (DEM, bs=1 vs bs=32) ===")
mod = "DEM"; tok, arrs = cache[mod]
t1, q1 = enc(mod, tok, arrs, 1, latents=True)
t32, q32 = enc(mod, tok, arrs, 32, latents=True)
d = np.abs(q1 - q32)
print(f"  max |latent difference|      : {d.max():.3e}")
print(f"  mean |latent difference|     : {d.mean():.3e}")
print(f"  latent magnitude (mean |q|)  : {np.abs(q1).mean():.3e}")
print(f"  relative                     : {d.max()/max(np.abs(q1).mean(),1e-12):.3e}")
print(f"  tokens differing             : {int((t1!=t32).sum())}/{t1.size}")
print("  -> a difference this small can only flip a token whose latent sits")
print("     essentially exactly on an FSQ bin boundary.")

print()
print("=== D. do determinism knobs remove it? ===")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
for mod in ["DEM", "S2L2A"]:
    tok, arrs = cache[mod]
    a1, a32 = enc(mod, tok, arrs, 1), enc(mod, tok, arrs, 32)
    print(f"  {mod:6s} bs1 vs bs32 with deterministic+noTF32: "
          f"{int((a1!=a32).sum())}/{a1.size} differ")
print("  (TITAN V is Volta -- TF32 does not exist there, so the knobs mostly")
print("   affect kernel selection, not precision.)")
