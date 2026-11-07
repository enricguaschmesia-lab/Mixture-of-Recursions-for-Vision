#!/usr/bin/env python
"""Is a row's date in dates.npy the date of its own S2 image? (A1, Step 2.) `mor`, CPU.

dates.npy comes from the metadata's `S2_time`, and the same metadata's `S1_time` proved to be a join
artefact on train. Every S2L2A sample carries its own truth: the zarr `time`, and the acquisition in
`file_id`. This reads both from each sample of the tars given and compares them with the sample's row.

PASS: on every sample, `time` equals the row's `S2_time`, `file_id` names the same second where there
is one, and the year id and day bin of `time` equal the row's line in dates.npy.

CONTROL, which must fail: the same comparison against another row's line in dates.npy. For
ssl4eos12, the next date of the same location, so it is the misplacement that matters here. For
Major TOM, the next row of the file. Below 5% of samples may still agree, by chance, and no more.

    python -I eo/experiments/temporal/s2_dates.py --root <token root> --metadata <parquet> \\
        --tars <dir>/*.tar --out <report.json>

Tars are read as data only (`tarfile` + the blosc decoder); nothing in them runs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eo.data.temporal_index import date_ids, load_temporal_index  # noqa: E402
from eo.data.temporal_split import row_dates  # noqa: E402
from eo.terramesh_tok.io import _decode_zarr_zip  # noqa: E402

ACQUIRED = re.compile(r"(\d{8}T\d{6})")      # the first stamp: `S2B_MSIL2A_<here>_...` and `<here>_<later>_T10SFH`
SECOND = pd.Timedelta(seconds=1)


def read_tar(path: Path):
    """(stem, zarr time, acquisition in file_id) per sample, in tar order."""
    out = []
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.name.endswith(".zarr.zip"):
                continue
            raw = tf.extractfile(m).read()
            ns = int(_decode_zarr_zip(raw, key="time")[0])
            try:
                fid = str(_decode_zarr_zip(raw, key="file_id")[0])
            except KeyError:                 # val's Major TOM samples have none
                out.append((m.name[: -len(".zarr.zip")], pd.Timestamp(ns), pd.NaT))
                continue
            found = ACQUIRED.search(fid)
            if found is None:
                raise RuntimeError(f"{path.name}/{m.name}: no date in file_id {fid!r}")
            out.append((m.name[: -len(".zarr.zip")], pd.Timestamp(ns), pd.Timestamp(found.group(1))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--tars", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for t in args.tars:
        got = read_tar(Path(t))
        rows += [(Path(t).name, *r) for r in got]
        print(f"{Path(t).name}: {len(got)} samples", flush=True)
    raw = pd.DataFrame(rows, columns=["tar", "stem", "zarr_time", "acquired"])

    index = load_temporal_index(args.root)
    table = row_dates(args.root, args.metadata)
    j = raw.merge(table[["row", "stem", "corpus", "S2_time"]], on="stem", how="left", validate="1:1")
    if j.row.isna().any():
        raise RuntimeError(f"{int(j.row.isna().sum())} samples are not in this root")
    j["row"] = j.row.astype(np.int64)

    # The other row: the next date of the location, or the next row of the file.
    n = len(table)
    next_row = (np.arange(n) + 1) % n
    next_row[index.locations] = np.roll(index.locations, -1, axis=1)
    other = next_row[j.row.to_numpy()]

    own = date_ids(j.zarr_time)
    stored = np.asarray(index.dates[j.row.to_numpy()])
    control = np.asarray(index.dates[other])
    same_time = (j.zarr_time - j.S2_time).abs() < SECOND
    has_id = j.acquired.notna()
    same_id = (j.zarr_time - j.acquired).abs() < SECOND
    same_date = (own == stored).all(axis=1)
    moved = (own == control).all(axis=1)

    per = {}
    for corpus, g in j.groupby("corpus"):
        i = g.index.to_numpy()
        gap = (g.zarr_time - g.S2_time).abs() / SECOND
        per[str(corpus)] = {"samples": int(len(g)), "tars": sorted(g.tar.unique().tolist()),
                            "time_equals_S2_time": int(same_time[i].sum()), "largest_gap_seconds": float(gap.max()),
                            "with_file_id": int(has_id[i].sum()), "file_id_equals_time": int(same_id[i].sum()),
                            "dates_npy_equal": int(same_date[i].sum()),
                            "control_other_row_equal": int(moved[i].sum())}
    passed = bool(same_time.all() and (same_id | ~has_id).all() and same_date.all())
    fired = bool(moved.mean() < 0.05)
    report = {"root": str(args.root), "samples": int(len(j)), "by_corpus": per, "passed": passed, "control_fired": fired}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if passed and fired else 1


if __name__ == "__main__":
    sys.exit(main())
