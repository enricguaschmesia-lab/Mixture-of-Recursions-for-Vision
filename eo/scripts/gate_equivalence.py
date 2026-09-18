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
  3. AGREEMENT WITH THE SINGLE-SAMPLE PATH -- the batched path must reproduce
     the single-sample reference to within the measured bin-flip rate, not
     exactly.

     The threshold is an ABSOLUTE COUNT derived from the sample size, not a
     rate. A fixed 0.1% rate was the original formulation and it is wrong here:
     the reference is only 3 samples x TOKENS_PER_SAMPLE per modality (588 at
     the 224 crop), and 0.1% of 588 is 0.588 tokens -- i.e. the threshold
     silently demanded ZERO flips, which is exactly the criterion this gate was
     rewritten to stop demanding. At the 256 crop it passed only because no
     token happened to flip in a 768-token sample; it was always one unlucky
     flip from a false failure. Found 2026-09-18, when NDVI flipped 1 token in
     588 (0.170%) and was decoded to be a single FSQ digit moving a single bin
     -- the documented mechanism, not a divergence.

     So: allow up to the Poisson tail bound at BIN_FLIP_RATE, which scales with
     the sample size. A real contract divergence (wrong stats, wrong crop, wrong
     band order) moves tens of percent of tokens -- Step 3 measured 19-51/256
     for S2L2A and up to 72/256 for DEM -- so this still separates the two cases
     by orders of magnitude.
"""
import json, math, os, sys, warnings
os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")
import numpy as np, torch, pandas as pd
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T

# Single-sample reference at the CURRENT contract crop. Crop-specific by
# necessity: a different crop means a different patch grid, the ViT interpolates
# its position embeddings, and every token changes -- so the 256 reference is not
# comparable at 224 (contract.py, CROP comment). Regenerate with
# eo/scripts/make_gate_reference.py whenever the crop changes.
_REF_PATH = _pathlib.Path(f"/data/enric/figures/step3/tokens{C.CROP}.json")
if not _REF_PATH.exists():
    sys.exit(f"missing single-sample reference for crop {C.CROP}: {_REF_PATH}\n"
             f"generate it first:  python eo/scripts/make_gate_reference.py")
REF = json.load(open(_REF_PATH))
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
STEM2TAR = dict(zip(META.zarr.str[: -len(".zarr.zip")], META.tar))
DEV, BS = T.DEVICE, 32
BIN_FLIP_RATE = 1 / 8192   # measured in Step 4: one token in 8,192 (0.012%)
FALSE_ALARM = 1e-3         # tolerated probability of a spurious gate failure


def flip_budget(n_tokens: int, rate: float = BIN_FLIP_RATE,
                alpha: float = FALSE_ALARM) -> int:
    """Max flips consistent with `rate`, at a false-alarm probability of alpha.

    Smallest k with P(X > k) < alpha for X ~ Poisson(n * rate). Computed
    directly -- k stays tiny, and this keeps the gate free of scipy.
    """
    lam = n_tokens * rate
    cdf, term, k = math.exp(-lam), math.exp(-lam), 0
    while 1.0 - cdf >= alpha:
        k += 1
        term *= lam / k
        cdf += term
    return k
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
print(f"=== 3. agreement with the single-sample path (crop {C.CROP}) ===")
_n_ref = 3 * C.TOKENS_PER_SAMPLE
print(f"  budget: <= {flip_budget(_n_ref)} flips in {_n_ref} tokens "
      f"(Poisson bound at {BIN_FLIP_RATE:.2e}, false alarm {FALSE_ALARM:.0e})")
print(f"  {'mod':6s} {'differing':>12s} {'rate':>9s} {'budget':>7s}   verdict")
for mod, samples in REF.items():
    tok = T.build(mod)
    stems = list(samples)
    got = encode(mod, tok, load(mod, stems))
    want = np.asarray([samples[s] for s in stems], dtype=np.uint16)
    n_diff, n = int((got != want).sum()), want.size
    budget = flip_budget(n)
    ok = n_diff <= budget
    fail += not ok
    print(f"  {mod:6s} {n_diff:6d}/{n:5d} {100*n_diff/n:8.3f}% {budget:7d}   "
          f"{'OK' if ok else 'FAIL -- exceeds bin-flip budget, real divergence'}")

print()
print("GATE PASSED" if not fail else f"GATE FAILED ({fail} check(s))")
sys.exit(1 if fail else 0)
