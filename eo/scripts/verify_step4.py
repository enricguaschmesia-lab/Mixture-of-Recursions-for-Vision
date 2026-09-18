#!/usr/bin/env python
"""Step 4 hand-off verification -- integrity of the produced artifacts.

This is NOT Step 5. Step 5 does the population statistics (code-usage
histograms, spot reconstruction). This checks that what Step 4 wrote is
structurally sound and, critically, that consolidation did not scramble rows --
the one failure mode this format introduces (R9), which would silently pair the
wrong S2L2A patch with the wrong LULC label and poison Phase 4.
"""
import json, os, sys, warnings
os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T

VAL = Path("/data/enric/data/TerraMesh/val")
index = pd.read_parquet(VAL / "tok_index.parquet")
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
N = 89088
fail = 0


def bad(msg):
    global fail
    fail += 1
    print(f"    FAIL: {msg}")


print("=== 1. shapes, dtypes, ranges, presence ===")
print(f"  {'mod':6s} {'shape':16s} {'dtype':8s} {'present':>8s} {'tok range':>16s} "
      f"{'uniq codes':>11s} {'nan':>5s}")
tokens, present = {}, {}
for mod in MODS:
    d = VAL / C.tok_dir_name(mod)
    t = np.load(d / "tokens.npy", mmap_mode="r")
    p = np.load(d / "present.npy")
    meta = json.load(open(d / "metadata.json"))
    tokens[mod], present[mod] = t, p
    tp = np.asarray(t[p])
    u = len(np.unique(tp))
    print(f"  {mod:6s} {str(t.shape):16s} {str(t.dtype):8s} {int(p.sum()):8d} "
          f"[{int(tp.min()):5d},{int(tp.max()):6d}] {u:11d} "
          f"{meta['n_nan_total']:5d}")
    if t.shape != (N, C.TOKENS_PER_SAMPLE):
        bad(f"{mod}: shape {t.shape}")
    if t.dtype != np.uint16:
        bad(f"{mod}: dtype {t.dtype}")
    if int(tp.max()) >= C.CODEBOOK[mod] or int(tp.min()) < 0:
        bad(f"{mod}: tokens outside [0,{C.CODEBOOK[mod]})")
    const = (tp == tp[:, :1]).all(axis=1)
    if const.any():
        bad(f"{mod}: {int(const.sum())} constant token sequences")

print()
print("=== 2. presence masks match the corpus structure ===")
is_mt = (index.corpus == "majortom").values
for mod in ["S2L2A", "DEM", "NDVI", "LULC"]:
    if not present[mod].all():
        bad(f"{mod}: expected full coverage, got {int(present[mod].sum())}")
print(f"  S2L2A/DEM/NDVI/LULC full coverage : "
      f"{all(present[m].all() for m in ['S2L2A','DEM','NDVI','LULC'])}")
ok_rtc = np.array_equal(present["S1RTC"], is_mt)
ok_grd = np.array_equal(present["S1GRD"], ~is_mt)
comp = np.array_equal(present["S1RTC"] ^ present["S1GRD"], np.ones(N, bool))
print(f"  S1RTC == majortom rows            : {ok_rtc}")
print(f"  S1GRD == ssl4eos12 rows           : {ok_grd}")
print(f"  S1RTC XOR S1GRD covers every row  : {comp}")
for c, m in ((ok_rtc, "S1RTC mask"), (ok_grd, "S1GRD mask"),
             (comp, "S1 complementarity")):
    if not c:
        bad(m)

print()
print("=== 3. row alignment end-to-end (R9) -- re-encode random rows ===")
print("  looks up the stem for a row, re-reads it from the source tar, encodes")
print("  it, and compares to the stored row. Catches any consolidation scramble.")
rng = np.random.default_rng(0)
for mod in MODS:
    rows = rng.choice(np.flatnonzero(present[mod]), size=4, replace=False)
    okc = 0
    for r in rows:
        rec = index.iloc[int(r)]
        _, arr = tio.read_sample(mod, rec.source_shard, rec.stem)
        x, _ = P.prepare(arr, mod, device="cpu")
        b = torch.cat([x] + [x[-1:]] * 31).to(T.DEVICE)     # pad to batch 32
        f = P.flatten_tokens(T.encode(T.build(mod), b)).cpu().numpy()[0]
        okc += int(np.array_equal(f.astype(np.uint16), np.asarray(tokens[mod][r])))
    print(f"  {mod:6s} {okc}/4 rows reproduce exactly"
          + ("" if okc == 4 else "   <-- ROW MISALIGNMENT"))
    if okc != 4:
        bad(f"{mod}: row alignment")

print()
print("=== 4. cross-modality join sanity ===")
r = int(rng.choice(np.flatnonzero(present["S1RTC"])))
rec = index.iloc[r]
print(f"  row {r} -> stem {rec.stem} ({rec.corpus}, {rec.source_shard})")
for mod in MODS:
    tag = "present" if present[mod][r] else "ABSENT (expected for S1GRD)"
    print(f"    {mod:6s} {tag:28s} first 6 tokens "
          f"{list(np.asarray(tokens[mod][r][:6])) if present[mod][r] else '-'}")

print()
print("=== 5. provenance recorded ===")
revs = {json.load(open(VAL / C.tok_dir_name(m) / 'metadata.json'))["contract_git_rev"][:7]
        for m in MODS}
stats = {json.load(open(VAL / C.tok_dir_name(m) / 'metadata.json'))
         ["standardization"]["source"] for m in MODS}
print(f"  contract git rev across modalities : {revs}")
print(f"  standardization source             : {stats}")
if len(revs) != 1:
    bad(f"modalities built from different contract revisions: {revs}")

print()
print("STEP 4 VERIFICATION PASSED" if not fail else f"FAILED ({fail} check(s))")
sys.exit(1 if fail else 0)
