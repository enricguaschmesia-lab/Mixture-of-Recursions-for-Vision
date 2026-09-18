#!/usr/bin/env python
"""Phase 1 Step 4 — batch tokenization of the TerraMesh val split.

Applies the Step 3 contract (terramesh_tok) to every sample of one modality and
writes val/<MOD>_tok<CROP>/. Resumable, atomic per shard, provenance-recording.

    python tokenize_terramesh.py --modality DEM
    python tokenize_terramesh.py --modality DEM --dry-run          # one shard
    python tokenize_terramesh.py --modality DEM --shards majortom_shard_000001.tar

Output layout (see STEP4_PLAN.md 2.2). The directory carries the crop, from
contract.TOK_DIR_SUFFIX -- two token sets coexist (Phase 1's 256 and the
ratified 224) and the naming is what makes mixing them impossible:
    val/tok_index.parquet             canonical row order, all 89,088 samples
    val/<MOD>_tok224/shards/*.npy     (N,196) uint16, per source tar (resume unit)
    val/<MOD>_tok224/tokens.npy       (89088,196) uint16 -- SAME rows every modality
    val/<MOD>_tok224/present.npy      (89088,) bool
    val/<MOD>_tok224/metadata.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/enric/hf")

import numpy as np
import pandas as pd
import torch

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T

VAL = Path("/data/enric/data/TerraMesh/val")
INDEX_PATH = VAL / "tok_index.parquet"
MANIFEST = Path("/data/enric/weights/tokenizer_manifest.json")
REPO = Path(__file__).resolve().parent

# Canonical shard order: all 81 majortom shards, then all 9 ssl4eos12 shards.
MAJORTOM = [f"majortom_shard_{i:06d}.tar" for i in range(1, 82)]
SSL4EOS12 = [f"ssl4eos12_shard_{i:06d}.tar" for i in range(1, 10)]
CANONICAL = MAJORTOM + SSL4EOS12

# S1RTC covers majortom only, S1GRD ssl4eos12 only -- exact complements,
# verified on disk. Everything else covers the whole split.
MOD_SHARDS = {
    "S2L2A": CANONICAL, "DEM": CANONICAL, "NDVI": CANONICAL, "LULC": CANONICAL,
    "S1RTC": MAJORTOM, "S1GRD": SSL4EOS12,
}
N_TOTAL = 89088


# ----------------------------------------------------------------- index
def build_index(anchor: str = "LULC") -> pd.DataFrame:
    """Canonical row order, taken from the actual tar member order."""
    rows = []
    for sh in CANONICAL:
        with tarfile.open(VAL / anchor / sh) as tf:
            stems = [m.name[: -len(".zarr.zip")] for m in tf
                     if m.name.endswith(".zarr.zip")]
        corpus = sh.split("_")[0]
        # NB: bind the base BEFORE extend(). list.extend consumes a generator
        # lazily while the list grows, so `len(rows)` inside the generator would
        # advance on every item and produce 0,2,4,6,... instead of 0,1,2,3,...
        base = len(rows)
        rows.extend((base + i, s, sh, corpus) for i, s in enumerate(stems))
    df = pd.DataFrame(rows, columns=["row", "stem", "source_shard", "corpus"])
    if len(df) != N_TOTAL:
        raise RuntimeError(f"index has {len(df)} rows, expected {N_TOTAL}")
    if not np.array_equal(df.row.values, np.arange(N_TOTAL)):
        raise RuntimeError(
            f"index rows are not 0..{N_TOTAL-1} "
            f"(min {df.row.min()}, max {df.row.max()}) -- row numbering is broken")
    if df.stem.duplicated().any():
        raise RuntimeError("duplicate stems in index")
    # cross-check against the official metadata (set equality, not order)
    meta = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
    want = set(meta.zarr.str[: -len(".zarr.zip")])
    if set(df.stem) != want:
        raise RuntimeError("index stems disagree with val_metadata.parquet")
    return df


def get_index(rebuild: bool = False) -> pd.DataFrame:
    if INDEX_PATH.exists() and not rebuild:
        return pd.read_parquet(INDEX_PATH)
    print(f"building canonical index -> {INDEX_PATH}", flush=True)
    df = build_index()
    tmp = INDEX_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, INDEX_PATH)
    print(f"  {len(df)} rows, "
          f"{(df.corpus == 'majortom').sum()} majortom / "
          f"{(df.corpus == 'ssl4eos12').sum()} ssl4eos12", flush=True)
    return df


# ----------------------------------------------------------------- encode
@torch.no_grad()
def tokenize_shard(mod: str, shard: str, tok, batch_size: int, device: str):
    """Returns (stems, tokens uint16 (N, TOKENS_PER_SAMPLE), nan_counts)."""
    stems, nans, out, buf = [], [], [], []

    def flush():
        # Always encode a full batch_size batch, padding the tail by repeating
        # the last sample and discarding the padding rows. Batch SHAPE changes
        # the cuBLAS/cuDNN kernel path, which can flip a token whose latent sits
        # on an FSQ bin boundary (measured: 1 in 8,192). Padding keeps every
        # real sample on one kernel path, so a shard of 1000 at batch 32 does
        # not encode its last 8 samples differently from the other 992.
        if not buf:
            return
        n_real = len(buf)
        batch = torch.cat(buf)
        if n_real < batch_size:
            batch = torch.cat([batch, batch[-1:].repeat(
                batch_size - n_real, *([1] * (batch.ndim - 1)))])
        g = T.encode(tok, batch.to(device))
        f = P.flatten_tokens(g).cpu().numpy()[:n_real]
        lo, hi = int(f.min()), int(f.max())
        if lo < 0 or hi >= C.CODEBOOK[mod]:
            raise RuntimeError(
                f"{mod}/{shard}: token out of range [{lo},{hi}] "
                f"vs codebook {C.CODEBOOK[mod]}")
        out.append(f.astype(np.uint16))
        buf.clear()

    for stem, arr in tio.iter_shard(mod, shard):
        x, info = P.prepare(arr, mod, device="cpu")
        buf.append(x)
        stems.append(stem)
        nans.append(info["n_nan"])
        if len(buf) >= batch_size:
            flush()
    flush()
    return stems, np.concatenate(out), np.asarray(nans, dtype=np.int32)


def check_invariants(mod, shard, stems, toks, expect_stems):
    """4.6 -- each aborts the shard loudly rather than writing bad data."""
    n = len(expect_stems)
    if len(stems) != n:
        raise RuntimeError(f"{mod}/{shard}: {len(stems)} samples, index says {n}")
    if stems != list(expect_stems):
        bad = next(i for i, (a, b) in enumerate(zip(stems, expect_stems)) if a != b)
        raise RuntimeError(
            f"{mod}/{shard}: ROW MISALIGNMENT at position {bad}: "
            f"read '{stems[bad]}' but index row expects '{expect_stems[bad]}'")
    if len(set(stems)) != n:
        raise RuntimeError(f"{mod}/{shard}: duplicate stems")
    if toks.shape != (n, C.TOKENS_PER_SAMPLE):
        raise RuntimeError(
            f"{mod}/{shard}: shape {toks.shape}, "
            f"expected {(n, C.TOKENS_PER_SAMPLE)}")
    if toks.dtype != np.uint16:
        raise RuntimeError(f"{mod}/{shard}: dtype {toks.dtype}, expected uint16")
    # the NaN-poisoning signature: every token collapsing to one value
    const = (toks == toks[:, :1]).all(axis=1)
    if const.any():
        raise RuntimeError(
            f"{mod}/{shard}: {int(const.sum())} sample(s) have a constant token "
            f"sequence -- the NaN-poisoning signature. First: {stems[int(np.argmax(const))]}")


# ----------------------------------------------------------------- driver
def run(mod: str, batch_size: int, device: str, shards, force: bool,
        index: pd.DataFrame):
    outdir = VAL / C.tok_dir_name(mod)
    (outdir / "shards").mkdir(parents=True, exist_ok=True)
    tok = T.build(mod, device=device)
    by_shard = {s: g for s, g in index.groupby("source_shard")}

    t_all = time.time()
    n_done = n_skip = 0
    stats = dict(tmin=10**9, tmax=-1, nan=0, n=0)
    for shard in shards:
        expect = by_shard[shard].sort_values("row")
        dst = outdir / "shards" / shard.replace(".tar", ".npy")
        if dst.exists() and not force:
            a = np.load(dst, mmap_mode="r")
            if a.shape == (len(expect), C.TOKENS_PER_SAMPLE):
                n_skip += 1
                stats["tmin"] = min(stats["tmin"], int(a.min()))
                stats["tmax"] = max(stats["tmax"], int(a.max()))
                stats["n"] += a.shape[0]
                continue
            print(f"  {shard}: existing output has wrong shape {a.shape}, redoing",
                  flush=True)

        t0 = time.time()
        stems, toks, nans = tokenize_shard(mod, shard, tok, batch_size, device)
        check_invariants(mod, shard, stems, toks, list(expect.stem))

        tmp = dst.with_suffix(".npy.tmp")
        with open(tmp, "wb") as fh:      # np.save() would append '.npy' to a path
            np.save(fh, toks)
        os.replace(tmp, dst)
        np.save(outdir / "shards" / shard.replace(".tar", ".nan.npy"), nans)

        dt = time.time() - t0
        stats["tmin"] = min(stats["tmin"], int(toks.min()))
        stats["tmax"] = max(stats["tmax"], int(toks.max()))
        stats["nan"] += int(nans.sum())
        stats["n"] += len(stems)
        n_done += 1
        print(f"  {shard:32s} {len(stems):5d} samples  {dt:6.1f}s  "
              f"{len(stems)/dt:6.1f}/s  tok[{int(toks.min())},{int(toks.max())}]  "
              f"nan={int(nans.sum())}", flush=True)

    print(f"{mod}: {n_done} shards written, {n_skip} skipped (resume), "
          f"{time.time()-t_all:.1f}s total", flush=True)
    return outdir, stats


def consolidate(mod: str, outdir: Path, index: pd.DataFrame, stats, args):
    """Build the row-aligned tokens.npy + present.npy + metadata.json."""
    tokens = np.zeros((N_TOTAL, C.TOKENS_PER_SAMPLE), dtype=np.uint16)
    present = np.zeros(N_TOTAL, dtype=bool)
    nan_total = 0
    for shard in MOD_SHARDS[mod]:
        rows = index[index.source_shard == shard].sort_values("row").row.values
        a = np.load(outdir / "shards" / shard.replace(".tar", ".npy"))
        tokens[rows] = a
        present[rows] = True
        nf = outdir / "shards" / shard.replace(".tar", ".nan.npy")
        if nf.exists():
            nan_total += int(np.load(nf).sum())

    n_present = int(present.sum())
    expected = sum(len(index[index.source_shard == s]) for s in MOD_SHARDS[mod])
    if n_present != expected:
        raise RuntimeError(f"{mod}: present={n_present}, expected {expected}")

    for name, arr in (("tokens.npy", tokens), ("present.npy", present)):
        tmp = outdir / (name + ".tmp")
        with open(tmp, "wb") as fh:      # same np.save() suffix gotcha
            np.save(fh, arr)
        os.replace(tmp, outdir / name)

    man = json.load(open(MANIFEST))[C.TOKENIZER[mod]]
    import terratorch, numcodecs, zarr
    meta = {
        "modality": mod,
        "tokenizer": {"name": C.TOKENIZER[mod], "repo": man["repo"],
                      "filename": man["filename"], "revision": man["revision"]},
        "codebook_size": C.CODEBOOK[mod],
        "patch_size": C.PATCH, "grid": [C.GRID, C.GRID],
        "tokens_per_sample": C.TOKENS_PER_SAMPLE,
        "crop": {"native": C.NATIVE, "crop": C.CROP, "offset": C.CROP_OFF},
        "flatten_order": C.FLATTEN_ORDER,
        "standardization": {
            "source": "v1_pretraining_* tok_* keys (decision D3.2)",
            "mean": C.V1_TOK_MEAN[mod], "std": C.V1_TOK_STD[mod]},
        "lulc_one_hot": mod == "LULC",
        "lulc_n_classes": C.LULC_N_CLASSES if mod == "LULC" else None,
        "nan_policy": C.NAN_POLICY, "n_nan_total": nan_total,
        "dtype": "uint16", "batch_size": args.batch_size,
        "device": f"{args.device} (TITAN V)", "precision": "fp32",
        "contract_git_rev": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "versions": {"torch": torch.__version__,
                     "terratorch": __import__("importlib.metadata", fromlist=["x"])
                     .version("terratorch"),
                     "numcodecs": numcodecs.__version__, "zarr": zarr.__version__,
                     "numpy": np.__version__},
        "n_samples_total_rows": N_TOTAL, "n_samples_present": n_present,
        "shards": MOD_SHARDS[mod],
        "token_min": stats["tmin"], "token_max": stats["tmax"],
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    tmp = outdir / "metadata.json.tmp"
    json.dump(meta, open(tmp, "w"), indent=1)
    os.replace(tmp, outdir / "metadata.json")
    mb = tokens.nbytes / 1e6
    print(f"{mod}: tokens.npy {tokens.shape} {mb:.1f} MB, present={n_present}, "
          f"nan_total={nan_total}, tok[{stats['tmin']},{stats['tmax']}]", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=list(MOD_SHARDS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=T.DEVICE)
    ap.add_argument("--shards", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="one shard only; skip consolidation")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args()

    index = get_index(rebuild=args.rebuild_index)
    shards = args.shards or MOD_SHARDS[args.modality]
    if args.dry_run:
        shards = shards[:1]

    print(f"=== {args.modality}: {len(shards)} shard(s), batch {args.batch_size}, "
          f"{args.device}, contract "
          f"{subprocess.run(['git','-C',str(REPO),'rev-parse','--short','HEAD'], capture_output=True, text=True).stdout.strip()} ===",
          flush=True)
    outdir, stats = run(args.modality, args.batch_size, args.device, shards,
                        args.force, index)
    if args.dry_run:
        print("dry run: skipping consolidation")
        return
    consolidate(args.modality, outdir, index, stats, args)


if __name__ == "__main__":
    main()
