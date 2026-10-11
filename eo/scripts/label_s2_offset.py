#!/usr/bin/env python
"""Label every TerraMesh val row by whether its S2L2A still carries the BOA offset (clean.py).

Major TOM rows: the product's processing baseline, from Major TOM Core-S2L2A's metadata.parquet,
matched on grid cell (the first two fields of TerraMesh's sample_id) and S2 time, which must
match every row exactly once. SSL4EO-S12 rows carry no product id and are labelled clean on
the evidence in clean.py. Writes <clean root>/s2_offset_labels.parquet and a .json with the
provenance, the counts, and the cross-check against the 1-in-10 brightness scan
(s2_offset_scan.py), which must separate as measured or the script fails.

    python eo/scripts/label_s2_offset.py \\
        --majortom-meta /data/enric/data/MajorTOM/Core-S2L2A/metadata.parquet \\
        --scan /data/enric/reports/s2_offset/val_scan_step10_meta.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terramesh_tok import clean as CL  # noqa: E402

VAL = Path("/data/enric/data/TerraMesh/val")
META = VAL.parent / "val_metadata.parquet"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--majortom-meta", type=Path, required=True)
    ap.add_argument("--scan", type=Path, default=None, help="s2_offset_scan.py output with S2_time (cross-check)")
    args = ap.parse_args()

    ix = pd.read_parquet(VAL / "tok_index.parquet")
    meta = pd.read_parquet(META, columns=["zarr", "sample_id", "S2_time"])
    meta["stem"] = meta.zarr.str[:-len(".zarr.zip")]
    d = ix.merge(meta[["stem", "sample_id", "S2_time"]], on="stem", how="left", validate="one_to_one")
    assert len(d) == len(ix) and d.sample_id.notna().all(), "index and val_metadata disagree"

    mt = pd.read_parquet(args.majortom_meta, columns=["grid_cell", "product_id", "timestamp"])
    m = d[d.corpus == "majortom"].copy()
    m["grid_cell"] = m.sample_id.str.split("_").str[:2].str.join("_")
    m["timestamp"] = pd.to_datetime(m.S2_time).dt.strftime("%Y%m%dT%H%M%S")
    j = m.merge(mt, on=["grid_cell", "timestamp"], how="left")
    if len(j) != len(m) or j.product_id.isna().any():
        raise SystemExit(f"Major TOM match is not one-to-one: {len(m)} rows -> {len(j)}, "
                         f"{int(j.product_id.isna().sum())} unmatched")
    j["baseline"] = j.product_id.str.extract(r"_N(\d{4})_")[0].astype(int)
    j["offset"] = [CL.has_offset(b, t) for b, t in zip(j.baseline, j.S2_time)]

    s = d[d.corpus != "majortom"].copy()
    s["product_id"], s["baseline"], s["offset"] = None, -1, False
    lab = pd.concat([j[["row", "stem", "corpus", "S2_time", "product_id", "baseline", "offset"]],
                     s[["row", "stem", "corpus", "S2_time", "product_id", "baseline", "offset"]]])
    lab = lab.sort_values("row").reset_index(drop=True)
    assert np.array_equal(lab.row.values, np.arange(len(ix))), "labels do not cover every row once"

    report = {"n_rows": int(len(lab)), "n_offset": int(lab.offset.sum()),
              "offset_frac": round(float(lab.offset.mean()), 4),
              "offset_baselines": {str(k): int(v) for k, v in lab[lab.offset].baseline.value_counts().items()},
              "rule": f"baseline in [{CL.OFFSET_BASELINES.start}, {CL.OFFSET_BASELINES.stop}) and "
                      f"S2 time < {CL.TERRAMESH_OFFSET_CUTOFF}; SSL4EO-S12 clean",
              "majortom_meta": str(args.majortom_meta), "majortom_meta_sha256": sha256(args.majortom_meta),
              "val_metadata_sha256": sha256(META), "tok_index_sha256": sha256(VAL / "tok_index.parquet"),
              "date": time.strftime("%Y-%m-%d %H:%M:%S %Z")}

    if args.scan is not None:          # cross-check against the brightness scan; must separate
        sc = pd.read_parquet(args.scan)[["zarr", "floor", "water_B12"]]
        sc["stem"] = sc.zarr.str[:-len(".zarr.zip")]
        x = sc.merge(lab, on="stem", how="inner", validate="one_to_one")
        tab = {}
        for name, g in x.groupby("offset"):
            w = g.water_B12.dropna()
            tab["offset" if name else "clean"] = {
                "n": int(len(g)), "floor_gt_900": round(float((g.floor > 900).mean()), 4),
                "n_water": int(len(w)), "water_B12_gt_900": round(float((w > 900).mean()), 4) if len(w) else None,
                "water_B12_median": float(w.median()) if len(w) else None}
        report["scan_crosscheck"] = tab
        o, c = tab.get("offset", {}), tab.get("clean", {})
        # what was measured on 2026-10-11; a label that no longer separates this way is wrong
        if not (o.get("floor_gt_900", 0) > 0.9 and c.get("floor_gt_900", 1) < 0.06
                and o.get("water_B12_gt_900", 0) > 0.95 and c.get("water_B12_gt_900", 1) < 0.05):
            print(json.dumps(tab, indent=1))
            raise SystemExit("cross-check FAILED: the labels do not separate the scan as measured")

    root = CL.clean_root(VAL)
    root.mkdir(exist_ok=True)
    tmp = root / (CL.LABELS_FILE + ".tmp")
    lab.to_parquet(tmp, index=False)
    tmp.replace(root / CL.LABELS_FILE)
    (root / "s2_offset_labels.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"-> {root / CL.LABELS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
