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
    a = ap.parse_args()
    index = pd.read_parquet(layout.INDEX)
    wanted = np.flatnonzero(np.logical_or.reduce([index[r].to_numpy(bool) for r in layout.PACK_ROLES]))
    print(f"packing {len(wanted)} rows ({', '.join(layout.PACK_ROLES)}) of {len(index)}", flush=True)
    stats = pack.pack(a.kind, index[["row", "s2", "s1"]], wanted)
    stats["roles"] = list(layout.PACK_ROLES)
    (layout.PACKED / f"pack_{a.kind}.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
