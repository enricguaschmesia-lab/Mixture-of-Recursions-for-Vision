#!/usr/bin/env python
"""A2 Step 4.2, Route B: write the class-cell "map tokens" of every train and val view (`.venv` or `a2seg`, CPU).

    python eo/scripts/make_cell_tokens.py --task pastis [--root /data/enric/data/geobench2]
    python eo/scripts/make_cell_tokens.py --task cloudsen12

One row per view of <root>/<task>/tok/views.parquet, in the TerraMesh token format beside MAP_aligned:
    <root>/<task>/tok/MAP_cells_<task>/tokens.npy   (views, 784) uint16, the 28 x 28 cells in raster order
                                      /present.npy  True for train and val views; test views are absent (never read)
                                      /metadata.json the rule (eo/maps/cells.RULE), label set hashes, counts
The labels are eo/maps/sets' fixed train and val sets (sha256-checked against their manifest); each view takes its crop
and dihedral variant as Step 2 cut the map tokens. The rule and both of its paths: eo/maps/cells.py; gate Z19-Z20.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("pastis", "cloudsen12"))
    ap.add_argument("--root", default="/data/enric/data/geobench2")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--check-numpy", type=int, default=64, help="views per chunk recomputed by the numpy path")
    a = ap.parse_args()
    from eo.maps import cells
    meta = cells.build_artifact(a.task, a.root, chunk=a.chunk, check_numpy=a.check_numpy,
                                log=lambda m: print(m, flush=True))
    print(json.dumps({k: meta[k] for k in ("arm", "n_views", "tokens_sha256", "checked_by_numpy", "seconds")}))


if __name__ == "__main__":
    main()
