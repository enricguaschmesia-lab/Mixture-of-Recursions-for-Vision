#!/usr/bin/env python
"""Pack one BigEarthNet v1 archive into native-resolution memmaps (`mor` env; CPU).

    python eo/scripts/pack_ben.py --kind S2
    python eo/scripts/pack_ben.py --kind S1

Run in tmux with a log under /data/enric/logs/. Packs the rows of layout.PACK_ROLES only,
in archive order. Not resumable: a failed pass is rerun (see eo/ben/pack.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.ben import layout, pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["S2", "S1"], required=True)
    ap.add_argument("--roles", default=",".join(layout.PACK_ROLES))
    ap.add_argument("--out", type=Path, default=layout.PACKED)
    ap.add_argument("--exclude-packed", action="store_true",
                    help="skip rows already in the main pack (layout.PACKED): an extra pack, e.g. L8's")
    a = ap.parse_args()
    roles = [r for r in a.roles.split(",") if r]
    index = pd.read_parquet(layout.INDEX)
    wanted = np.flatnonzero(np.logical_or.reduce([index[r].to_numpy(bool) for r in roles]))
    if a.exclude_packed:
        assert a.out.resolve() != layout.PACKED.resolve(), "--exclude-packed needs another --out"
        wanted = np.setdiff1d(wanted, np.load(layout.PACKED / f"order_{a.kind}.npy"))
    print(f"packing {len(wanted)} rows ({', '.join(roles)}) of {len(index)} into {a.out}", flush=True)
    stats = pack.pack(a.kind, index[["row", "s2", "s1"]], wanted, out=a.out)
    stats["roles"] = roles
    (a.out / f"pack_{a.kind}.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
