"""Step 3.9 throughput benchmark + 3.8 coords probe."""
import os, time, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

DEV = T.DEVICE
MOD, SHARD = "S2L2A", "majortom_shard_000009.tar"
N = 64

print("=== 3.9 Stage timings (S2L2A, the heaviest modality) ===")
# (a) IO: tar member read + blosc decode
t0 = time.time(); arrs = []
for stem, a in io.iter_shard(MOD, SHARD, limit=N):
    arrs.append(a)
io_s = time.time() - t0
print(f"  (a) read+decode {N} samples : {io_s:6.2f}s -> {N/io_s:6.1f} samples/s")

# (b) preprocessing (CPU: crop, cast, standardize) - measure on CPU only
t0 = time.time()
xs = [P.prepare(a, MOD, device="cpu")[0] for a in arrs]
pp_s = time.time() - t0
print(f"  (b) preprocess  {N} samples : {pp_s:6.2f}s -> {N/pp_s:6.1f} samples/s")

# (c) encode, batched, fp32 and fp16
tok = T.build(MOD)
print(f"  (c) encode on {DEV} (TITAN V):")
for dtype in (torch.float32, torch.float16):
    tk = tok.half() if dtype == torch.float16 else tok.float()
    for bs in (1, 8, 16, 32):
        try:
            batch = torch.cat(xs[:bs]).to(DEV).to(dtype)
            torch.cuda.reset_peak_memory_stats(DEV)
            with torch.no_grad():
                tk.encode(batch)                        # warmup
            torch.cuda.synchronize(DEV); t0 = time.time()
            reps = max(1, 32 // bs)
            with torch.no_grad():
                for _ in range(reps):
                    tk.encode(batch)
            torch.cuda.synchronize(DEV)
            dt = (time.time() - t0) / reps
            mem = torch.cuda.max_memory_allocated(DEV) / 2**30
            print(f"      {str(dtype).split('.')[-1]:8s} bs={bs:3d}  "
                  f"{dt*1000:7.1f} ms/batch  {bs/dt:7.1f} samples/s  "
                  f"peak {mem:5.2f} GiB")
        except RuntimeError as e:
            print(f"      {str(dtype).split('.')[-1]:8s} bs={bs:3d}  FAILED: {str(e)[:60]}")
            break
tok.float()

print()
print("=== Step 4 extrapolation ===")
n_samples, n_mods = 89088, 6
per_mod_io = n_samples / (N / io_s) / 60
print(f"  IO+decode alone, per modality : {per_mod_io:6.1f} min "
      f"({per_mod_io*n_mods/60:.1f} h for all {n_mods})")
b = n_samples * C.TOKENS_PER_SAMPLE * 2 / 1e6
print(f"  Output size per modality      : {b:6.1f} MB  "
      f"(all {n_mods}: {b*n_mods/1000:.2f} GB) as uint16")

print()
print("=== 3.8 Coords tokenizer probe (optional) ===")
try:
    from terratorch.registry import FULL_MODEL_REGISTRY
    ct = FULL_MODEL_REGISTRY.build("terramind_v1_coords_tokenizer", pretrained=True)
    import pandas as pd
    d = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet").head(3)
    for _, r in d.iterrows():
        s = f"lat={r.center_lat:.4f} lon={r.center_lon:.4f}"
        try:
            out = ct.encode([s]) if hasattr(ct, "encode") else None
            ids = out[-1] if isinstance(out, (tuple, list)) else out
            ids = ids.flatten().tolist() if hasattr(ids, "flatten") else ids
            print(f"  {s:34s} -> {len(ids)} tokens {ids}")
        except Exception as e:
            print(f"  {s:34s} -> encode() failed: {type(e).__name__}: {str(e)[:70]}")
    print(f"  tokenizer type: {type(ct).__name__}")
except Exception as e:
    print(f"  coords tokenizer unavailable: {type(e).__name__}: {str(e)[:120]}")
