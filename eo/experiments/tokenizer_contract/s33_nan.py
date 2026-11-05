"""Step 3.3: NaN prevalence across the val split, and the consequence."""
import warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

MT = [f"majortom_shard_{i:06d}.tar" for i in range(1, 82, 8)]
SS = [f"ssl4eos12_shard_{i:06d}.tar" for i in range(1, 10)]
PLAN = {"S1RTC": (MT, 200), "S1GRD": (SS, 220), "NDVI": (MT + SS, 110),
        "S2L2A": (MT + SS, 110), "DEM": (MT + SS, 110), "LULC": (MT + SS, 110)}

print("=== NaN prevalence (val split) ===")
print(f"{'MOD':6s} {'n':>6s} {'samples w/ NaN':>15s} {'%':>7s} "
      f"{'mean NaN px frac':>17s} {'worst':>8s}")
res = {}
for mod, (shards, per) in PLAN.items():
    n = nsamp = 0
    fracs = []
    for sh in shards:
        try:
            for stem, a in io.iter_shard(mod, sh, limit=per):
                n += 1
                af = a.astype(np.float32)
                f = float(np.isnan(af).mean())
                if f > 0:
                    nsamp += 1
                    fracs.append(f)
        except FileNotFoundError:
            continue
    pct = 100.0 * nsamp / max(n, 1)
    mf = float(np.mean(fracs)) if fracs else 0.0
    wf = float(np.max(fracs)) if fracs else 0.0
    res[mod] = (n, nsamp, pct, mf, wf)
    print(f"{mod:6s} {n:6d} {nsamp:15d} {pct:6.2f}% {mf:17.5f} {wf:7.4f}")

print()
print("=== Consequence: does one NaN pixel corrupt all 256 tokens? ===")
mod = "S1RTC"
tok = T.build(mod)
stem, arr = io.read_sample(mod, "majortom_shard_000009.tar")
x_clean, _ = P.prepare(arr, mod)
base = P.flatten_tokens(T.encode(tok, x_clean))[0]

# inject NaN into the RAW array, then run the real prepare() (which fills)
raw_nan = arr.astype(np.float32).copy()
raw_nan[0, 0, 100, 100] = np.nan
x_filled, info = P.prepare(raw_nan, mod)
filled = P.flatten_tokens(T.encode(tok, x_filled))[0]
print(f"  policy='{C.NAN_POLICY}' filled {info['n_nan']} px -> "
      f"{int((filled != base).sum())}/256 tokens differ")

# and WITHOUT the fill, to show what the policy is protecting against
xn = x_clean.clone()
xn[0, 0, 100, 100] = float("nan")
unfilled = P.flatten_tokens(T.encode(tok, xn))[0]
n_nan_tok = int(torch.isnan(unfilled.float()).sum())
print(f"  NO fill, 1 NaN px       -> {int((unfilled != base).sum())}/256 "
      f"tokens differ, {n_nan_tok} are NaN")
print(f"  unfilled token range: {int(unfilled.min())}..{int(unfilled.max())}")
