"""Step 3.2 + 3.4(2): build every tokenizer strictly, encode one sample each."""
import warnings, time, torch, numpy as np
warnings.filterwarnings("ignore")
from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

SHARD = {m: "majortom_shard_000009.tar" for m in C.MODALITIES}
SHARD["S1GRD"] = "ssl4eos12_shard_000002.tar"

print(f"{'MOD':6s} {'strict load':11s} {'in shape':22s} {'tok grid':12s} "
      f"{'min':>5s} {'max':>6s} {'cb':>6s} {'uniq':>5s} {'nan':>5s}  build_s")
for mod in ["DEM", "LULC", "NDVI", "S1RTC", "S1GRD", "S2L2A"]:
    t0 = time.time()
    try:
        tok = T.build(mod)
        loaded = "OK"
    except Exception as e:
        print(f"{mod:6s} FAILED {type(e).__name__}: {str(e)[:100]}")
        continue
    tb = time.time() - t0
    stem, arr = io.read_sample(mod, SHARD[mod])
    x, info = P.prepare(arr, mod)
    g = T.encode(tok, x)
    f = P.flatten_tokens(g)
    lo, hi = int(f.min()), int(f.max())
    ok = "OK" if hi < C.CODEBOOK[mod] and lo >= 0 else "RANGE-FAIL"
    print(f"{mod:6s} {loaded:11s} {str(tuple(x.shape)):22s} "
          f"{str(tuple(g.shape)):12s} {lo:5d} {hi:6d} {C.CODEBOOK[mod]:6d} "
          f"{len(torch.unique(f)):5d} {info['n_nan']:5d}  {tb:6.1f}  {ok}")
    if mod == "DEM":
        print(f"       stem={stem}  first 16 tokens: {f[0,:16].tolist()}")
        print(f"       flatten check: token[17] == grid[1,1] ->",
              int(f[0, 17]) == int(g[0, 1, 1]))
