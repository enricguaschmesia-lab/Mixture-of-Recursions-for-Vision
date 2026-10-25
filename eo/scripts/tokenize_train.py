#!/usr/bin/env python
"""Tokenize the TerraMesh TRAIN split, streamed shard by shard (mor env). Logic: eo/terramesh_tok/train_split.py.

    python eo/scripts/tokenize_train.py plan     --out OUT --revision SHA
    python eo/scripts/tokenize_train.py download --out OUT --work WORK --positions 0:894 --revision SHA
    python eo/scripts/tokenize_train.py encode   --out OUT --work WORK --positions 0:894 --expect-gpu "TITAN V"
    python eo/scripts/tokenize_train.py verify-val --shards majortom_shard_000001.tar ssl4eos12_shard_000001.tar
    python eo/scripts/tokenize_train.py consolidate --out OUT [--allow-partial --dest DIR]
    python eo/scripts/tokenize_train.py coords   --root DIR      (after consolidate)
    python eo/scripts/tokenize_train.py verify-coords            (val: must equal val/Coords_tok)

Positions index order.json (a seeded permutation of the 894 shards); "a:b" with a > b walks backward,
so two platforms can work from opposite ends. `touch OUT/control/STOP` stops both loops cleanly.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/enric/hf")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eo.terramesh_tok import train_split as TS  # noqa: E402

TM = Path("/data/enric/data/TerraMesh")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--revision", required=True, help="the HF dataset commit sha to pin")
    p.add_argument("--meta", type=Path, default=TM / "train_meta/train_metadata.parquet")
    p.add_argument("--listing", type=Path, default=TM / "train_meta/hf_train_listing.json")
    p.add_argument("--majortom-meta", type=Path, default=Path("/data/enric/data/MajorTOM/Core-S2L2A/metadata.parquet"))
    p.add_argument("--seed", type=int, default=20261025)
    for name in ("download", "encode"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--positions", default="0:894")
        p.add_argument("--poll", type=float, default=20.0)
        if name == "download":
            p.add_argument("--revision", required=True)
            p.add_argument("--listing", type=Path, default=TM / "train_meta/hf_train_listing.json")
            p.add_argument("--ahead", type=int, default=4)
            p.add_argument("--streams", type=int, default=8)
        else:
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--expect-gpu", default=None)
            p.add_argument("--keep-raw", action="store_true")
            p.add_argument("--worker", default="0/1", help="i/n: this encoder takes every n-th position from i "
                                                         "(one downloader feeding n encoders, one per GPU)")
    p = sub.add_parser("verify-val")
    p.add_argument("--val-root", type=Path, default=TM / "val")
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--expect-gpu", default=None)
    p.add_argument("--control-crop-off", type=int, default=None, help="deliberately-broken control")
    p.add_argument("--max-rate", type=float, default=0.0, help="token disagreement allowed (0 = exact)")
    p.add_argument("--report", type=Path, default=None)
    p = sub.add_parser("consolidate")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-root", type=Path, default=TM / "val")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--dest", type=Path, default=None)
    p = sub.add_parser("coords")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--meta", type=Path, default=TM / "train_meta/train_metadata.parquet")
    p.add_argument("--val-root", type=Path, default=TM / "val")
    p = sub.add_parser("verify-coords")
    p.add_argument("--val-root", type=Path, default=TM / "val")
    p.add_argument("--meta", type=Path, default=TM / "val_metadata.parquet")
    a = ap.parse_args()

    if a.cmd == "plan":
        print(json.dumps(TS.build_plan(a.out, a.meta, a.listing, a.majortom_meta, a.revision, a.seed), indent=1))
    elif a.cmd == "download":
        TS.download_loop(a.out, a.work, TS.parse_positions(a.positions, 894), a.revision, a.listing,
                         ahead=a.ahead, streams=a.streams, poll=a.poll)
    elif a.cmd == "encode":
        i, n = (int(x) for x in a.worker.split("/"))
        TS.encode_loop(a.out, a.work, TS.parse_positions(a.positions, 894)[i::n], device=a.device,
                       expect_gpu=a.expect_gpu, delete_raw=not a.keep_raw, poll=a.poll)
    elif a.cmd == "verify-val":
        rep = TS.verify_val(a.val_root, a.shards, a.device, a.expect_gpu, crop_off=a.control_crop_off)
        worst = max(m["rate"] for s in rep["shards"].values() for m in s.values())
        rep["max_rate_allowed"], rep["worst_rate"] = a.max_rate, worst
        rep["pass"] = worst <= a.max_rate
        if a.report:
            TS.atomic_json(rep, a.report)
        print(("PASS" if rep["pass"] else "FAIL") + f": worst disagreement {worst:.6f} (allowed {a.max_rate})")
        return 0 if rep["pass"] else 1
    elif a.cmd == "consolidate":
        b = TS.consolidate(a.out, a.val_root, allow_partial=a.allow_partial, dest=a.dest)
        return 0 if b["offset_free"] else 1
    elif a.cmd == "coords":
        print(json.dumps(TS.write_coords(a.root, a.meta, a.val_root)))
    elif a.cmd == "verify-coords":
        import numpy as np
        import pandas as pd
        idx = pd.read_parquet(a.val_root / "tok_index.parquet")
        meta = pd.read_parquet(a.meta, columns=["zarr", "center_lon", "center_lat"])
        meta["stem"] = meta.zarr.str[: -len(".zarr.zip")]
        m = idx.merge(meta[["stem", "center_lon", "center_lat"]], on="stem", how="left", validate="one_to_one")
        mine = TS.coords_tokens(m.center_lon.values, m.center_lat.values)
        ref = np.load(a.val_root / "Coords_tok" / "tokens.npy")
        ok = mine.shape == ref.shape and bool((mine == ref).all())
        print(f"{'PASS' if ok else 'FAIL'}: val Coords {mine.shape} vs stored {ref.shape}, "
              f"rows differing {int((mine != ref).any(axis=1).sum()) if mine.shape == ref.shape else 'n/a'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
