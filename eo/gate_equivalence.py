#!/usr/bin/env python
"""Step 4.2 -- the equivalence gate. Nothing scales until this passes.

The first version of this gate demanded bit-identical tokens across batch
sizes. That is not achievable on a GPU and the gate correctly failed: batch
SHAPE selects a different cuBLAS/cuDNN kernel, whose different reduction order
perturbs the pre-quantization latent by ~1e-6, which flips a token whenever a
latent sits essentially exactly on an FSQ bin boundary. Measured rate: 1 token
in 8,192 (0.012%), and confirmed to be exactly one latent element moving one
bin, not diffuse drift.

So the gate now asserts what is actually required for reproducibility:

  1. DETERMINISM AT A FIXED BATCH SHAPE  -- must be bit-identical. This is what
     makes a re-run reproduce the artifact.
  2. PADDING INVARIANCE                  -- a padded tail batch must give the
     same tokens for its real samples as a genuinely full batch. This is what
     lets a 1000-sample shard avoid encoding its last 8 samples on a different
     kernel path.
  3. AGREEMENT WITH STEP 3               -- the batched path must reproduce the
     Step 3 single-sample tokens to within the measured bin-flip rate, not
     exactly. Threshold 0.1%; anything larger is a real contract divergence.
"""
import json, os, sys, warnings
os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")
import numpy as np, torch, pandas as pd
from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T

REF = json.load(open("/data/enric/figures/step3/tokens.json"))
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
STEM2TAR = dict(zip(META.zarr.str[: -len(".zarr.zip")], META.tar))
DEV, BS = T.DEVICE, 32
TOL = 0.001          # 0.1%; measured bin-flip rate is 0.012%
fail = 0


@torch.no_grad()
def encode(mod, tok, arrs, bs=BS, pad=True):
    """The production path: fixed batch shape, tail padded and sliced off."""
    out, buf = [], []

    def flush():
        nonlocal buf
        n = len(buf)
        b = torch.cat(buf)
        if pad and n < bs:
            b = torch.cat([b, b[-1:].repeat(bs - n, *([1] * (b.ndim - 1)))])
        out.append(P.flatten_tokens(T.encode(tok, b.to(DEV))).cpu().numpy()[:n])
        buf = []

    for a in arrs:
        buf.append(P.prepare(a, mod, device="cpu")[0])
        if len(buf) >= bs:
            flush()
    if buf:
        flush()
    return np.concatenate(out).astype(np.uint16)


def load(mod, stems):
    return [tio.read_sample(mod, STEM2TAR[s], s)[1] for s in stems]


MODS = ["DEM", "NDVI", "LULC", "S1RTC", "S1GRD", "S2L2A"]
mt = META[META.tar.str.startswith("majortom")].head(40)
ss = META[META.tar.str.startswith("ssl4eos12")].head(40)

print(f"=== 1. determinism at a fixed batch shape (bs={BS}, run twice) ===")
cache = {}
for mod in MODS:
    tok = T.build(mod)
    rows = ss if mod == "S1GRD" else mt
    arrs = [tio.read_sample(mod, r.tar, r.zarr[: -len(".zarr.zip")])[1]
            for _, r in rows.iterrows()]
    cache[mod] = (tok, arrs)
    ok = np.array_equal(encode(mod, tok, arrs), encode(mod, tok, arrs))
    fail += not ok
    print(f"  {mod:6s} {len(arrs)} samples  {'IDENTICAL' if ok else 'NONDETERMINISTIC -- FAIL'}")

print()
print(f"=== 2. padding invariance (tail of {40 % BS} padded to {BS}) ===")
for mod in MODS:
    tok, arrs = cache[mod]
    full = encode(mod, tok, arrs)                     # 32 full + 8 padded
    # the same tail 8 samples, encoded as their own padded batch
    tail = encode(mod, tok, arrs[BS:])
    ok = np.array_equal(full[BS:], tail)
    fail += not ok
    print(f"  {mod:6s} padded tail == standalone padded batch: "
          f"{'YES' if ok else 'NO -- FAIL'}")

print()
print(f"=== 3. agreement with Step 3's verified single-sample tokens ===")
print(f"  {'mod':6s} {'differing':>12s} {'rate':>9s}   verdict (tol {100*TOL:.1f}%)")
for mod, samples in REF.items():
    tok = T.build(mod)
    stems = list(samples)
    got = encode(mod, tok, load(mod, stems))
    want = np.asarray([samples[s] for s in stems], dtype=np.uint16)
    n_diff, n = int((got != want).sum()), want.size
    rate = n_diff / n
    ok = rate <= TOL
    fail += not ok
    print(f"  {mod:6s} {n_diff:6d}/{n:5d} {100*rate:8.3f}%   "
          f"{'OK' if ok else 'FAIL -- exceeds bin-flip tolerance'}")

print()
print("GATE PASSED" if not fail else f"GATE FAILED ({fail} check(s))")
sys.exit(1 if fail else 0)
