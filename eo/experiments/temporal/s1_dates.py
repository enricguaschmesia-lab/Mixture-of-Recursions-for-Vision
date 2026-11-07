#!/usr/bin/env python
"""Do S1 images sit in the right rows? Row-by-row dates from raw tars (A1, Step 1). `mor`, CPU.

Train's `S1_time` holds date 0's S1 date in all four rows of a location, so it cannot answer
this. Every S1GRD sample does carry its own truth: the product id in `file_id` and the zarr `time`.
This reads both from each sample of the tars given, and compares them with the row's S2 date.

PASS: the product date equals `time`, it is nearest to the row's own S2 date among the location's
four, and at t = 0 it equals the metadata's `S1_time`.

The plan's first rule was "within 30 days of its S2", val's largest gap. Train misses that on 4 of
30,000 rows (30.3-38.3 days), each still nearest its own date, so the 30-day count is reported but the
pass rule is the nearest date, which is what a wrong row would break. Changed after seeing train's
result (2026-11-05), so the count stays in the report.

CONTROL, which must fail: the same check on the same samples with S1 deliberately misplaced, moved
one row along inside every location seen with two rows or more, inside 1% of them, and inside one.
Each must come back not passed. The first control compared true rows with another date's S2, which
could not fail: it follows from the pass. Val is the positive control, run through the same code.

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
S1_FIELDS = ("product", "zarr_time", "file_id")
SEED = 20261105


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


def judge(raw: pd.DataFrame, meta: pd.DataFrame, tars) -> tuple:
    """(report, joined rows) for the samples in `raw`."""
    j = raw.merge(meta[["stem", "location", "t", "S2_time", "S1_time"]], on="stem", how="left", validate="1:1")
    if j.S2_time.isna().any():
        raise RuntimeError(f"{int(j.S2_time.isna().sum())} samples have no metadata row")

    # Another date's S2, same location: how far S1 sits from a date that is not its own.
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

    report = {"tars": tars, "rows": int(len(j)), "locations_seen": int(j.location.nunique()),
              "rows_per_t": {int(k): int(v) for k, v in j.t.value_counts().sort_index().items()},
              "rule": {"max_gap_days": MAX_GAP_DAYS},
              "product_vs_zarr_time": summary(j.gap_product_time),
              "by_t": {int(t): {"product_vs_s2": summary(g.gap_product_s2), "zarr_time_vs_s2": summary(g.gap_time_s2),
                                "product_vs_metadata_S1_time": summary(g.gap_product_meta),
                                "product_vs_s2_of_another_date": summary(g.gap_product_other)}
                       for t, g in j.groupby("t")},
              "all": {"product_vs_s2": summary(j.gap_product_s2),
                      "product_vs_s2_of_another_date": summary(j.gap_product_other)}}

    # Dates of one location, where a tar set holds two or more of its rows: must be chronological.
    multi = j[j.groupby("location").t.transform("size") > 1].sort_values(["location", "t"])
    d = multi.groupby("location")["product"].diff().dropna() / DAY
    report["within_location"] = {"locations": int(multi.location.nunique()), "pairs": int(len(d)),
                                 "share_increasing": float((d > 0).mean()) if len(d) else None,
                                 "median_days": float(d.median()) if len(d) else None}
    report["nearest_date"] = {"rows_nearest_own_date": int((j.nearest_t == j.t).sum()), "rows": int(len(j))}
    report["passed"] = bool(report["product_vs_zarr_time"]["max_abs_days"] < 1.0
                            and report["nearest_date"]["rows_nearest_own_date"] == len(j)
                            and report["by_t"][0]["product_vs_metadata_S1_time"]["max_abs_days"] < 1.0)
    return report, j


def misplaced(raw: pd.DataFrame, j: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """`raw` with S1 moved one row along inside n of the locations seen with two rows or more."""
    seen = j[j.groupby("location").t.transform("size") > 1].sort_values(["location", "t"])
    pick = set(rng.permutation(seen.location.unique())[:n].tolist())
    where = pd.Series(np.arange(len(raw)), index=raw.stem.values)
    source = np.arange(len(raw))
    for _, g in seen[seen.location.isin(pick)].groupby("location"):
        stems = g.stem.values
        source[where[stems].values] = where[np.roll(stems, -1)].values
    out = raw.copy()
    for c in S1_FIELDS:
        out[c] = raw[c].values[source]
    return out


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

    tars = [Path(t).name for t in args.tars]
    report, j = judge(raw, meta, tars)
    report["split"] = args.split

    # The control: the same samples and the same check, with S1 deliberately in the wrong rows.
    n_multi = int(j[j.groupby("location").t.transform("size") > 1].location.nunique())
    rng = np.random.default_rng(SEED)
    controls = {}
    for name, n in (("every_location", n_multi), ("one_percent", max(1, round(0.01 * n_multi))), ("one_location", 1)):
        r, _ = judge(misplaced(raw, j, n, rng), meta, tars)
        controls[name] = {"locations_moved": n, "rows_nearest_own_date": r["nearest_date"]["rows_nearest_own_date"],
                          "passed": r["passed"]}
    report["control_s1_misplaced"] = controls
    passed, fired = report["passed"], not any(c["passed"] for c in controls.values())
    report["control_fired"] = bool(fired)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    j.drop(columns=["t_other"]).to_parquet(out / f"{args.split}_rows.parquet")
    (out / f"{args.split}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "rows_per_t", "product_vs_zarr_time", "all", "within_location", "nearest_date", "control_s1_misplaced", "passed", "control_fired")}, indent=1))
    far = j[j.gap_product_s2.abs() > MAX_GAP_DAYS]
    print(far[["stem", "t", "nearest_t", "gap_product_s2"]].to_string())
    for t, v in report["by_t"].items():
        print(f"t={t}: product vs S2 {v['product_vs_s2']}\n     product vs metadata S1_time {v['product_vs_metadata_S1_time']}")
    return 0 if passed and fired else 1


if __name__ == "__main__":
    sys.exit(main())
