#!/usr/bin/env python
"""A2 Step 2.6 (prepared, not run): TerraMesh train LULC as map tokens, row-aligned with the train tokens
(`a2seg` / `mor-eo` env, GPU; for R1's mid-training, A2 Step 5).

⚠ Downloads 96 tars, 24.5 GB (the `ssl4eos12` corpus's LULC, 958,176 rows), which waits for Enric's yes
and for Step 4.0's tokenizer pick. `--plan` only lists what would be fetched and written; nothing is
downloaded without `--download`.

Per shard, in tok_index.parquet's order: fetch <LULC>/<shard> from the HF listing (size and sha256
checked on the fly, eo/terramesh_tok/train_split._fetch), read its members in tar order, check the stems
against tok_index's rows for that shard, centre-crop each map to 224 (train_split.crop, the contract's
crop), render it with the universal palette and LULC's fixed assignment (class c -> colour c, no data on
white), encode with the frozen arm at its fixed batch shape (eo/scripts/tokenize_maps.encode_maps), and
write the codes at those rows; then delete the tar. Majortom rows stay absent (`present` False).

    <out>/MAP_<arm>/tokens.npy   (8,929,841, tokens_per_map) uint16, row = tok_index.row
                   /present.npy  bool
                   /done/<shard>.json, metadata.json

    python eo/scripts/tokenize_train_lulc_maps.py --root $G/tm_train/root --arm aligned --plan
    python eo/scripts/tokenize_train_lulc_maps.py --root $G/tm_train/root --work $G/tm_train/lulc_raw \\
        --arm aligned --download --expect-gpu A100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "scripts"))
from eo.maps import frozen  # noqa: E402
from eo.terramesh_tok import train_split as TS  # noqa: E402
from tokenize_maps import encode_maps  # noqa: E402

LISTING = Path("/data/enric/data/TerraMesh/train_meta/hf_train_listing.json")
CHUNK = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="the train token root (tok_index.parquet, plan.json)")
    ap.add_argument("--work", type=Path, default=None, help="where the tars land transiently")
    ap.add_argument("--arm", choices=list(frozen.ARMS), required=True)
    ap.add_argument("--listing", type=Path, default=LISTING)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--shards", type=int, default=None, help="only the first N ssl4eos12 shards")
    ap.add_argument("--conns", type=int, default=8)
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    idx = pd.read_parquet(a.root / "tok_index.parquet")
    files = json.loads(a.listing.read_text())["files"]["LULC"]
    revision = json.loads((a.root / "plan.json").read_text()).get("revision", "main")
    shards = sorted(idx[idx.corpus == "ssl4eos12"].source_shard.unique())[: a.shards]
    size = sum(files[s]["size"] for s in shards)
    print(f"{len(shards)} ssl4eos12 LULC tars, {size / 1e9:.2f} GB, revision {revision}; "
          f"{int(idx.source_shard.isin(shards).sum())} rows of {len(idx)}", flush=True)
    if a.plan or not a.download:
        if not a.plan:
            print("nothing downloaded: pass --download (needs Enric's yes, A2 Step 2.6)")
        return
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    TS.strict_fp32()
    tok = frozen.load(a.arm)
    T = tok.frozen["tokens"]
    out = a.root / f"MAP_{a.arm}"
    (out / "done").mkdir(parents=True, exist_ok=True)
    n = len(idx)
    tpath, ppath = out / "tokens.npy", out / "present.npy"
    tokens = (np.load(tpath, mmap_mode="r+") if tpath.exists()
              else np.lib.format.open_memmap(tpath, mode="w+", dtype=np.uint16, shape=(n, T)))
    present = np.load(ppath) if ppath.exists() else np.zeros(n, bool)
    base = f"https://huggingface.co/datasets/{TS.HF_REPO}/resolve/{revision}/train/LULC"
    for s in shards:
        if (out / "done" / f"{s}.json").exists():
            continue
        t0 = time.time()
        dest = a.work / "LULC" / s
        TS._fetch(f"{base}/{s}", dest, files[s]["size"], files[s]["sha256"], conns=a.conns)
        rows = idx[idx.source_shard == s]
        want = dict(zip(rows.stem, rows.row))
        stems, maps = [], []
        for stem, arr in TS._iter_tar(a.work, "LULC", s):
            stems.append(stem)
            maps.append(np.asarray(TS.crop(arr)).reshape(224, 224))
        if set(stems) != set(want):
            raise RuntimeError(f"{s}: tar stems differ from tok_index ({len(set(stems) ^ set(want))} differ)")
        r = np.array([want[st] for st in stems])
        lab = np.stack(maps)
        for c0 in range(0, len(lab), CHUNK):
            codes = encode_maps(tok, "lulc", lab[c0:c0 + CHUNK])
            tokens[r[c0:c0 + CHUNK]] = codes.astype(np.uint16)
        tokens.flush()
        present[r] = True
        with open(str(ppath) + ".tmp", "wb") as fh:
            np.save(fh, present)
        os.replace(str(ppath) + ".tmp", ppath)
        TS.atomic_json({"shard": s, "rows": int(len(r)), "sha256": files[s]["sha256"], "gpu": gpu,
                        "seconds": round(time.time() - t0, 1)}, out / "done" / f"{s}.json")
        dest.unlink()
        print(f"  {s}: {len(r)} maps, {time.time() - t0:.0f} s", flush=True)
    TS.atomic_json({"arm": a.arm, "frozen": tok.frozen, "n_rows": n, "n_present": int(present.sum()),
                    "revision": revision, "gpu": gpu, "written": time.strftime("%Y-%m-%d %H:%M:%S")},
                   out / "metadata.json")


if __name__ == "__main__":
    main()
