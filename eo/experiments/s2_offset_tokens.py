"""S2L2A brightness / BOA-offset through the tokenizer and the chain (`mor` env, one GPU).

Question (Enric, 2026-10-11): the chained S1RTC -> S2L2A -> NDVI failures look like the
generated S2 has the right shapes but the wrong brightness. TerraMesh's val S2L2A mixes
scenes that still carry ESA's +1000 BOA_ADD_OFFSET with scenes that do not (scan:
s2_offset_scan.py). This script measures, on the 256 chain rows of A10:

  1. regime of the real scene, of its true-token decode, and of the generated S2 (t1),
     where regime = 'offset' iff the darkest band's 1st percentile (the floor) > 900 DN;
  2. what a brightness change does to the tokens and to their decode (16 scenes, 8 per regime):
     shift by 1000 DN toward the other regime, +100 DN, x1.25;
  3. control: re-encoding the unmodified scene reproduces the stored tokens.

Encoding follows the contract: batch 32, tail padded by repeating the last sample
(tokenize_terramesh.py). Decoding follows 12.6b: unclamped DiVAE, 50 steps, batch 4, seed 0
re-applied per batch (s12_decoded_eval.decode).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "scripts")); sys.path.insert(0, str(HERE))
from terramesh_tok import contract as C, preprocess as P, tokenizers as TK  # noqa: E402
import decode_eo as D  # noqa: E402
import clamp_test as CT  # noqa: E402

MOD = "S2L2A"
CHAIN = Path("/data/enric/reports/chain_S1RTC_S2L2A_NDVI/a10")
OUT = Path("/data/enric/reports/s2_offset"); OUT.mkdir(parents=True, exist_ok=True)
THR = 900.0
DEV = TK.DEVICE

name = torch.cuda.get_device_name(0)
assert torch.cuda.device_count() == 1 and "TITAN X" in name, name
tok = TK.build(MOD, device=DEV)
sched = CT.scheduler_like(tok, **CT.VARIANTS["noclip"])


def floor(x):            # (12,H,W) DN -> min over bands of the 1st percentile
    return float(np.percentile(x.reshape(12, -1), 1, axis=1).min())


def visible_mean(x):
    return float(x[1:4].mean())


@torch.no_grad()
def encode(arrs):        # list of (12,224,224) float32 DN -> (N,14,14) int64
    out = []
    for i in range(0, len(arrs), 32):
        xs = [P.prepare(a[None], MOD, device=DEV, crop=224, crop_off=0)[0] for a in arrs[i:i + 32]]
        n = len(xs); b = torch.cat(xs)
        if n < 32:
            b = torch.cat([b, b[-1:].repeat(32 - n, 1, 1, 1)])
        out.append(TK.encode(tok, b)[:n].cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def decode(grids):       # (N,14,14) -> (N,12,224,224) DN, float32
    outs = []
    for i in range(0, len(grids), 4):
        g = torch.from_numpy(np.asarray(grids[i:i + 4], dtype=np.int64)).to(DEV)
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        z = tok.decode_tokens(g, timesteps=50, scheduler=sched).float()
        outs.append(P.destandardize(z, MOD).cpu().numpy())
    return np.concatenate(outs)


gen = np.load(CHAIN / "gen_mid.npz")
rows = gen["rows"]
raw = [P.center_crop(a[0] if a.ndim == 4 else a).astype(np.float32) for a in D.read_rasters(MOD, rows)]
stored = np.load(D.VAL / C.tok_dir_name(MOD) / "tokens.npy", mmap_mode="r")

# ---- 1. regimes along the chain
dec_true = decode(stored[rows].reshape(-1, 14, 14))
dec_gen = decode(gen["grids"])
res = {"rows": rows,
       "floor_real": np.array([floor(x) for x in raw]),
       "floor_true": np.array([floor(x) for x in dec_true]),
       "floor_gen": np.array([floor(x) for x in dec_gen]),
       "vis_real": np.array([visible_mean(x) for x in raw]),
       "vis_true": np.array([visible_mean(x) for x in dec_true]),
       "vis_gen": np.array([visible_mean(x) for x in dec_gen])}
np.savez(OUT / "chain_regimes_a10.npz", **res)
print("chain regimes saved", flush=True)

# ---- 2/3. perturbation and control, 8 scenes per regime (lowest row ids, by rule)
off = [i for i in np.argsort(rows) if res["floor_real"][i] > 1000][:8]
cln = [i for i in np.argsort(rows) if res["floor_real"][i] < 500][:8]
sel = off + cln
base = [raw[i] for i in sel]
variants = {
    "asis": base,
    "shift1000": [x - 1000 if k < 8 else x + 1000 for k, x in enumerate(base)],
    "plus100": [x + 100 for x in base],
    "x1.25": [x * 1.25 for x in base],
}
enc = {k: encode(v) for k, v in variants.items()}
report = {"rows": rows[sel].tolist(), "regime": ["offset"] * 8 + ["clean"] * 8, "variants": {}}
report["control_asis_vs_stored"] = float((enc["asis"].reshape(16, -1) == stored[rows[sel]]).mean())
dec = {k: decode(g) for k, g in enc.items()}
for k in variants:
    changed = (enc[k] != enc["asis"]).reshape(16, -1).mean(1)
    applied = np.array([(variants[k][j] - base[j]).mean(axis=(1, 2)) for j in range(16)])     # (16,12)
    got = np.array([(dec[k][j] - dec["asis"][j]).mean(axis=(1, 2)) for j in range(16)])
    report["variants"][k] = {
        "tokens_changed": changed.round(3).tolist(),
        "applied_band_mean_shift": applied.mean(1).round(1).tolist(),
        "decoded_band_mean_shift": got.mean(1).round(1).tolist(),
        "decoded_floor": [round(floor(x), 1) for x in dec[k]],
        "input_floor": [round(floor(x), 1) for x in variants[k]],
    }
(OUT / "perturb_a10.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v if k != "variants" else {kk: {m: vv[m] for m in ("tokens_changed",)} for kk, vv in v.items()}
                  for k, v in report.items()}), flush=True)
print("done", flush=True)
