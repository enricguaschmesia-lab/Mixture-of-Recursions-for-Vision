#!/usr/bin/env python
"""Continuous-I/O pilot, data half: raw TerraMesh pixels + the FSQ tokens of the same rows. (`mor` env)

Question (Enric, 2026-10-28): can our decoder-only model ingest and emit continuous patches
instead of FSQ tokens? The pilot (cio_train.py) trains small decoders on the SAME rows seen
either as tokens or as pixels, so this script writes both, row-aligned.

    python eo/experiments/cio_data.py extract      # ~10 min, 4 processes
    python eo/experiments/cio_data.py verify       # NDVI-from-S2 alignment check + its control

ROWS. Only rows with S1RTC (the majortom corpus, 80,384 of 89,088), so every row carries the same
four modalities: S2L2A (12 bands), S1RTC (2), DEM (1), LULC (10 classes). NDVI is left out: it is
a fixed function of two S2L2A bands. Train: 16,384 rows drawn from the complement of the held-out
split (eo/data/eval_rows_v1.json). Test: 1,024 rows drawn from the held-out split. Both are stored
in a random order (seed 0), so a contiguous block of rows is a random sample.

PIXELS. The 224 centre crop (contract.CROP_OFF), standardized with the tokenizers' own statistics
(contract.V1_TOK_MEAN/STD), NaN -> the modality mean (z = 0), stored as float16 z-scores.
S2L2A has ESA's +1000 DN BOA offset removed in the rows clean.has_offset labels, so the pixels
match the cleaned tokens (val_clean) this script copies next to them. LULC is stored as uint8
class indices.

OUTPUT /data/enric/experiments/continuous_io/data/
    {train,test}_{S2L2A,S1RTC,DEM}.npy   (N, C, 224, 224) float16 z-scores
    {train,test}_LULC.npy                (N, 224, 224) uint8 classes
    {train,test}_tok.npy                 (N, 4, 196) uint16 FSQ ids, modality order MODS
    meta.json                            rows, offsets, stats, git revision

VERIFY. NDVI recomputed from the extracted S2L2A (de-standardized, clean.ndvi_from_s2) must match
TerraMesh's stored NDVI crop on rows WITHOUT the offset (it is computed from the same bands), to
float16 precision. Control: the same comparison against an NDVI crop shifted by one pixel must fail.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eo.terramesh_tok import clean as CL                      # noqa: E402
from eo.terramesh_tok import contract as C                    # noqa: E402
from eo.terramesh_tok.io import _decode_zarr_zip              # noqa: E402

VAL = Path("/data/enric/data/TerraMesh/val")
VAL_CLEAN = Path("/data/enric/data/TerraMesh/val_clean")
OUT = Path("/data/enric/experiments/continuous_io/data")
MODS = ("S2L2A", "S1RTC", "DEM", "LULC")
N_TRAIN, N_TEST, SEED = 16384, 1024, 0


def git_rev() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def pick_rows(n_train: int = N_TRAIN, n_test: int = N_TEST):
    index = pd.read_parquet(VAL / "tok_index.parquet")
    s1 = np.load(VAL / C.tok_dir_name("S1RTC") / "present.npy")
    held = np.zeros(len(index), bool)
    held[json.load(open(REPO / "eo/data/eval_rows_v1.json"))["eval_rows"]] = True
    rng = np.random.default_rng(SEED)
    train = rng.permutation(np.flatnonzero(s1 & ~held))[:n_train]
    test = rng.permutation(np.flatnonzero(s1 & held))[:n_test]
    return index, train, test


def to_z(mod: str, a: np.ndarray, offset: bool = False) -> np.ndarray:
    """(C, 264, 264) raw -> (C, 224, 224) float32 z-scores, the tokenizers' standardization."""
    o, c = C.CROP_OFF, C.CROP
    a = a[..., o:o + c, o:o + c].astype(np.float32)
    if mod == "S2L2A" and offset:
        a = a - CL.BOA_ADD_OFFSET
    mean = np.asarray(C.V1_TOK_MEAN[mod], np.float32).reshape(-1, 1, 1)
    std = np.asarray(C.V1_TOK_STD[mod], np.float32).reshape(-1, 1, 1)
    a = np.where(np.isnan(a), mean, a)
    return (a - mean) / std


def _shard_task(args):
    mod, shard, split, want = args          # want: {stem: (out_index, offset)}
    path = OUT / (f"{split}_{mod}.npy")
    arr = np.load(path, mmap_mode="r+")
    done = 0
    with tarfile.open(VAL / mod / shard) as tf:
        for m in tf:
            if not m.name.endswith(".zarr.zip"):
                continue
            stem = m.name[: -len(".zarr.zip")]
            if stem not in want:
                continue
            i, offset = want[stem]
            raw = _decode_zarr_zip(tf.extractfile(m).read())
            raw = raw[0] if raw.ndim == 4 else raw
            if mod == "LULC":
                o, c = C.CROP_OFF, C.CROP
                cls = raw[0, o:o + c, o:o + c]
                assert cls.max() < C.LULC_N_CLASSES, f"LULC class {cls.max()} in {stem}"
                arr[i] = cls.astype(np.uint8)
            else:
                assert raw.shape[0] == C.N_CHANNELS[mod], (mod, raw.shape)
                arr[i] = to_z(mod, raw, offset).astype(np.float16)
            done += 1
    arr.flush()
    return mod, shard, split, done


def cmd_extract(a) -> int:
    global OUT
    if a.smoke:                       # 64 + 32 rows into a sibling directory, to test the code
        OUT = OUT.parent / "data_smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    index, train, test = pick_rows(*((64, 32) if a.smoke else (N_TRAIN, N_TEST)))
    labels = CL.load_labels(VAL_CLEAN).set_index("row")
    assert (labels.index.values == np.arange(len(index))).all()
    offset = labels["offset"].values.astype(bool)
    tasks = []
    for split, rows in (("train", train), ("test", test)):
        n = len(rows)
        for mod in MODS:
            shape = (n, C.CROP, C.CROP) if mod == "LULC" else (n, C.N_CHANNELS[mod], C.CROP, C.CROP)
            dtype = np.uint8 if mod == "LULC" else np.float16
            np.lib.format.open_memmap(OUT / f"{split}_{mod}.npy", mode="w+", dtype=dtype, shape=shape).flush()
        sub = index.iloc[rows]
        pos = {r: k for k, r in enumerate(rows)}
        for shard, g in sub.groupby("source_shard"):
            want = {s: (pos[r], bool(offset[r])) for r, s in zip(g.row.values, g.stem.values)}
            for mod in MODS:
                tasks.append((mod, shard, split, want))
        # the tokens of the same rows, from the cleaned root (S2L2A cleaned, the rest symlinked)
        tok = np.stack([np.load(VAL_CLEAN / C.tok_dir_name(m) / "tokens.npy", mmap_mode="r")[rows]
                        for m in MODS], axis=1)
        np.save(OUT / f"{split}_tok.npy", tok.astype(np.uint16))
    t0, counts = time.time(), {}
    with Pool(4) as pool:
        for k, (mod, shard, split, done) in enumerate(pool.imap_unordered(_shard_task, tasks), 1):
            counts[(split, mod)] = counts.get((split, mod), 0) + done
            if k % 40 == 0:
                print(f"[{time.time() - t0:6.0f} s] {k}/{len(tasks)} shard tasks", flush=True)
    for split, rows in (("train", train), ("test", test)):
        for mod in MODS:
            got = counts.get((split, mod), 0)
            assert got == len(rows), f"{split} {mod}: wrote {got} of {len(rows)} rows"
    meta = {"created": time.strftime("%F %T"), "git_rev": git_rev(), "mods": MODS,
            "train_rows": train.tolist(), "test_rows": test.tolist(),
            "train_offset_rows": int(offset[train].sum()), "test_offset_rows": int(offset[test].sum()),
            "crop": C.CROP, "crop_off": C.CROP_OFF, "patch": C.PATCH,
            "stats": "contract.V1_TOK_MEAN/STD z-scores, NaN -> 0, S2L2A offset removed per clean.has_offset",
            "tokens": "val_clean <MOD>_tok224/tokens.npy, same rows", "seed": SEED,
            "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(f"done in {meta['elapsed_s']} s; offset rows train {meta['train_offset_rows']}, "
          f"test {meta['test_offset_rows']}")
    return 0


def cmd_verify(a) -> int:
    """NDVI from the extracted S2L2A == stored NDVI on offset-free rows; shifted crop must fail."""
    global OUT
    if a.smoke:
        OUT = OUT.parent / "data_smoke"
    meta = json.loads((OUT / "meta.json").read_text())
    index = pd.read_parquet(VAL / "tok_index.parquet")
    offset = CL.load_labels(VAL_CLEAN).set_index("row")["offset"].values.astype(bool)
    test_rows = np.asarray(meta["test_rows"])
    s2 = np.load(OUT / "test_S2L2A.npy", mmap_mode="r")
    picks = [k for k, r in enumerate(test_rows) if not offset[r]][: a.n]
    mean = np.asarray(C.V1_TOK_MEAN["S2L2A"], np.float32).reshape(-1, 1, 1)
    std = np.asarray(C.V1_TOK_STD["S2L2A"], np.float32).reshape(-1, 1, 1)
    o, c = C.CROP_OFF, C.CROP
    err_ok, err_ctl = [], []
    for k in picks:
        r = test_rows[k]
        stem, shard = index.stem[r], index.source_shard[r]
        with tarfile.open(VAL / "NDVI" / shard) as tf:
            nd = _decode_zarr_zip(tf.extractfile(f"{stem}.zarr.zip").read())
        nd = (nd[0] if nd.ndim == 4 else nd)[0].astype(np.float32)
        dn = np.rint(s2[k].astype(np.float32) * std + mean)            # back to DN (z stored as fp16)
        mine = CL.ndvi_from_s2(dn[None])[0, 0].astype(np.float32)
        ok = np.isfinite(nd[o:o + c, o:o + c])
        err_ok.append(np.abs(mine - nd[o:o + c, o:o + c])[ok].mean())
        ctl = nd[o + 1:o + 1 + c, o:o + c]
        err_ctl.append(np.abs(mine - ctl)[np.isfinite(ctl)].mean())
    e, x = float(np.median(err_ok)), float(np.median(err_ctl))
    rep = {"n": len(picks), "median_abs_ndvi_err": e, "control_shifted_crop": x,
           "pass": bool(e < 0.01 and x > 5 * e)}
    print(json.dumps(rep, indent=1))
    (OUT / "verify.json").write_text(json.dumps(rep))
    return 0 if rep["pass"] else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("extract")
    x.add_argument("--smoke", action="store_true")
    v = sp.add_parser("verify")
    v.add_argument("--n", type=int, default=64)
    v.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    return {"extract": cmd_extract, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
