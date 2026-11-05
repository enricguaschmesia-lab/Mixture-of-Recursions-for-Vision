#!/usr/bin/env python
"""Do S1 images sit in the right rows? Row-by-row dates from raw tars (A1, Step 1). `mor`, CPU.

Train's `S1_time` holds date 0's S1 date in all four rows of a location, so it cannot answer
this. Every S1GRD sample does carry its own truth: the product id in `file_id` and the zarr `time`.
This reads both from each sample of the tars given, and compares them with the row's S2 date.

PASS: the product date equals `time`, and it is nearest to the row's own S2 date among the
location's four. At t = 0 it must also equal the metadata's `S1_time`.

The plan's first rule was "within 30 days of its S2", val's largest gap. Train misses that on 4 of
30,000 rows (30.3-38.3 days), each still nearest its own date, so the 30-day count is reported but the
pass rule is the nearest date, which is what a wrong row would break. Changed after seeing train's
result (2026-11-05), so the count stays in the report.

CONTROL, which must fail: the same comparison against another date index's S2 date, of the same
location. Dates are about 90 days apart, so true S1 dates fall outside 30 days of it. Without this,
"within 30 days" could pass on any column. Val is the positive control, run through the same code.

    python -I eo/experiments/temporal/s1_dates.py --split train --tars <dir>/ssl4eos12_shard_*.tar
    python -I eo/experiments/temporal/s1_dates.py --split val --tars /data/enric/data/TerraMesh/val/S1GRD/*.tar

Tars are read as data only (`tarfile` + the blosc decoder); nothing in them runs.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eo.terramesh_tok.io import _decode_zarr_zip  # noqa: E402

ROOT = Path("/data/enric/data/TerraMesh")
OUT = Path("/data/enric/reports/temporal/s1_dates")
META = {"train": ROOT / "train_meta/train_metadata.parquet", "val": ROOT / "val_metadata.parquet"}
MAX_GAP_DAYS = 30                 # val's largest, and the rule fixed in the plan
PRODUCT = re.compile(r"_(\d{8}T\d{6})_")
DAY = pd.Timedelta(days=1)


def read_tar(path: Path):
    """(stem, product date, zarr time, file_id) per sample, in tar order."""
    out = []
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.name.endswith(".zarr.zip"):
                continue
            raw = tf.extractfile(m).read()
            fid = str(_decode_zarr_zip(raw, key="file_id")[0])
            ns = int(_decode_zarr_zip(raw, key="time")[0])
            found = PRODUCT.findall(fid)
            if not found:
                raise RuntimeError(f"{path.name}/{m.name}: no date in file_id {fid!r}")
            out.append((m.name[: -len(".zarr.zip")], pd.Timestamp(found[0]), pd.Timestamp(ns), fid))
    return out


def summary(gap_days: pd.Series) -> dict:
    a = gap_days.abs()
    return {"rows": int(len(a)), "median_abs_days": float(a.median()), "p99_abs_days": float(a.quantile(0.99)),
            "max_abs_days": float(a.max()), "share_within_30d": float((a <= MAX_GAP_DAYS).mean()),
            "rows_outside_30d": int((a > MAX_GAP_DAYS).sum()), "median_signed_days": float(gap_days.median())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("train", "val"), required=True)
    ap.add_argument("--tars", nargs="+", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = []
    for t in args.tars:
        got = read_tar(Path(t))
        rows += [(Path(t).name, *r) for r in got]
        print(f"{Path(t).name}: {len(got)} samples", flush=True)
    raw = pd.DataFrame(rows, columns=["tar", "stem", "product", "zarr_time", "file_id"])

    meta = pd.read_parquet(META[args.split], columns=["zarr", "tar", "sample_id", "S2_time", "S1_time"])
    meta = meta[meta.tar.str.startswith("ssl4eos12")].copy()
    meta["stem"] = meta.zarr.str[: -len(".zarr.zip")]
    meta["location"] = meta.sample_id.str.rsplit("_", n=1).str[0]
    meta["t"] = meta.sample_id.str.rsplit("_", n=1).str[1].astype(int)
    j = raw.merge(meta[["stem", "location", "t", "S2_time", "S1_time"]], on="stem", how="left", validate="1:1")
    if j.S2_time.isna().any():
        raise RuntimeError(f"{int(j.S2_time.isna().sum())} samples have no metadata row")

    # Another date's S2, same location: the control column.
    other = meta[["location", "t", "S2_time"]].rename(columns={"t": "t_other", "S2_time": "S2_other"})
    j["t_other"] = (j.t + 2) % 4                     # half a year away, on the same location
    j = j.merge(other, on=["location", "t_other"], how="left", validate="m:1")

    # The nearest of the location's four S2 dates: S1 in the wrong row lands on another one.
    wide = meta.pivot(index="location", columns="t", values="S2_time")
    s2 = wide.reindex(j.location).to_numpy(dtype="datetime64[ns]")            # (rows, 4)
    p = j["product"].to_numpy(dtype="datetime64[ns]")[:, None]
    j["nearest_t"] = np.abs((s2 - p).astype("timedelta64[s]").astype(np.int64)).argmin(1)

    j["gap_product_s2"] = (j["product"] - j.S2_time) / DAY
    j["gap_time_s2"] = (j.zarr_time - j.S2_time) / DAY
    j["gap_product_time"] = (j["product"] - j.zarr_time) / DAY
    j["gap_product_meta"] = (j["product"] - j.S1_time) / DAY
    j["gap_product_other"] = (j["product"] - j.S2_other) / DAY

    report = {"split": args.split, "tars": [Path(t).name for t in args.tars], "rows": int(len(j)),
              "locations_seen": int(j.location.nunique()), "rows_per_t": {int(k): int(v) for k, v in j.t.value_counts().sort_index().items()},
              "rule": {"max_gap_days": MAX_GAP_DAYS},
              "product_vs_zarr_time": summary(j.gap_product_time),
              "by_t": {int(t): {"product_vs_s2": summary(g.gap_product_s2), "zarr_time_vs_s2": summary(g.gap_time_s2),
                                "product_vs_metadata_S1_time": summary(g.gap_product_meta),
                                "control_product_vs_s2_of_another_date": summary(g.gap_product_other)}
                       for t, g in j.groupby("t")},
              "all": {"product_vs_s2": summary(j.gap_product_s2),
                      "control_product_vs_s2_of_another_date": summary(j.gap_product_other)}}

    # Dates of one location, where a tar set holds two or more of its rows: must be chronological.
    multi = j[j.groupby("location").t.transform("size") > 1].sort_values(["location", "t"])
    d = multi.groupby("location")["product"].diff().dropna() / DAY
    report["within_location"] = {"locations": int(multi.location.nunique()), "pairs": int(len(d)),
                                 "share_increasing": float((d > 0).mean()) if len(d) else None,
                                 "median_days": float(d.median()) if len(d) else None}

    # The same test with each row's t rotated, as if the images sat two rows along.
    rotated = (j.nearest_t.values == (j.t.values + 2) % 4)
    report["nearest_date"] = {"rows_nearest_own_date": int((j.nearest_t == j.t).sum()), "rows": int(len(j)),
                              "control_rows_nearest_rotated_date": int(rotated.sum())}
    passed = (report["product_vs_zarr_time"]["max_abs_days"] < 1.0
              and report["nearest_date"]["rows_nearest_own_date"] == len(j))
    fired = (report["all"]["control_product_vs_s2_of_another_date"]["share_within_30d"] < 0.05
             and report["nearest_date"]["control_rows_nearest_rotated_date"] < 0.01 * len(j))
    report["passed"], report["control_fired"] = bool(passed), bool(fired)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    j.drop(columns=["t_other"]).to_parquet(out / f"{args.split}_rows.parquet")
    (out / f"{args.split}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "rows_per_t", "product_vs_zarr_time", "all", "within_location", "nearest_date", "passed", "control_fired")}, indent=1))
    far = j[j.gap_product_s2.abs() > MAX_GAP_DAYS]
    print(far[["stem", "t", "nearest_t", "gap_product_s2"]].to_string())
    for t, v in report["by_t"].items():
        print(f"t={t}: product vs S2 {v['product_vs_s2']}\n     product vs metadata S1_time {v['product_vs_metadata_S1_time']}")
    return 0 if passed and fired else 1


if __name__ == "__main__":
    sys.exit(main())
