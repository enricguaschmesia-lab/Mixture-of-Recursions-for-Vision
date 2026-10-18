"""Stream a BigEarthNet v1 archive once and pack the wanted patches (`mor` env; tifffile).

    S2: <PACKED>/S2_10m.npy (M,4,120,120) uint16 · S2_20m.npy (M,6,60,60) · S2_60m.npy (M,2,20,20)
    S1: <PACKED>/S1.npy     (M,2,120,120) float32, dB, [VV, VH]
    both: order_<kind>.npy (M,) int64 = the index row at each packed position
          meta_<kind>.parquet (row, patch, labels, date, ...)

⚠ STREAM ORDER, NOT ROW ORDER. Position p holds the p-th wanted patch in ARCHIVE order, and
`order_<kind>.npy` maps it to its index row (`layout.load_packed` gives the inverse).
The first version wrote each patch at its index row. That is a scattered write into a 60 GB
file for every patch, and on /data (a QLC SSD under btrfs) it collapsed to ~3 MB/s of
writeback with 72% iowait once the drive's write cache filled (2026-10-12). Appending in
stream order makes every write sequential.

Only the WANTED rows are packed (the fine-tuning and test roles, ~244k of 519k), not the
whole archive. Rasters are written exactly as stored: no resampling, scaling or NaN
filling; that belongs to the tokenizer contract (`preprocess.py`).

Not resumable by design: a pass is one sequential sweep (~30-60 min), so a failed pass is
rerun from the start, and the outputs are written under .tmp names and renamed at the end.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tifffile

from . import layout


def _utm(projection: str):
    m = re.search(r"UTM zone (\d+)([NS])", projection or "")
    return f"{m.group(1)}{m.group(2)}" if m else None


class _Appender:
    """Sequential writes into a preallocated .npy, one row at a time."""

    def __init__(self, path: Path, shape, dtype):
        self.final = path
        self.path = path.with_suffix(".npy.tmp")
        m = np.lib.format.open_memmap(self.path, mode="w+", dtype=dtype, shape=tuple(shape))
        self.offset = m.offset
        self.row_shape, self.dtype = tuple(shape[1:]), np.dtype(dtype)
        del m
        self.fh = open(self.path, "r+b", buffering=16 << 20)
        self.fh.seek(self.offset)
        self.n = 0

    def write(self, a: np.ndarray):
        assert a.shape == self.row_shape and a.dtype == self.dtype, (a.shape, a.dtype)
        self.fh.write(np.ascontiguousarray(a).tobytes())
        self.n += 1

    def close(self, expected: int):
        self.fh.close()
        if self.n != expected:
            raise RuntimeError(f"{self.path}: wrote {self.n} rows, expected {expected}")
        self.path.replace(self.final)


def pack(kind: str, index: pd.DataFrame, wanted: np.ndarray, out: Path = layout.PACKED,
         log_every: int = 20000, labels: bool = True) -> Dict:
    """labels=False (L11b's unlabelled pool) writes meta_<kind>.parquet WITHOUT the label column:
    the rasters, dates and coordinates are kept, the labels are never stored."""
    assert kind in ("S2", "S1")
    out.mkdir(parents=True, exist_ok=True)
    name_col = "s2" if kind == "S2" else "s1"
    want_rows = set(int(r) for r in wanted)
    row_of: Dict[str, int] = {n: int(r) for n, r in zip(index[name_col].values, index["row"].values)
                              if int(r) in want_rows}
    m_total = len(row_of)
    archive = layout.S2_ARCHIVE if kind == "S2" else layout.S1_ARCHIVE

    if kind == "S2":
        app = {g: _Appender(out / f"{g}.npy", (m_total, len(b), s, s), layout.S2_DTYPE)
               for g, (b, s) in layout.S2_GROUPS.items()}
        needed = [b for bands, _ in layout.S2_GROUPS.values() for b in bands]
    else:
        bands, s = layout.S1_BANDS
        app = {"S1": _Appender(out / "S1.npy", (m_total, len(bands), s, s), layout.S1_DTYPE)}
        needed = list(bands)

    order: List[int] = []
    meta: List[dict] = []
    stats = {"seen": 0, "packed": 0, "skipped_not_wanted": 0, "incomplete": 0, "duplicate": 0}
    done_rows = set()
    t0 = time.time()

    def flush(patch: str, files: Dict[str, bytes]):
        if patch is None:
            return
        stats["seen"] += 1
        row = row_of.get(patch)
        if row is None:
            stats["skipped_not_wanted"] += 1
            return
        if row in done_rows:
            stats["duplicate"] += 1
            return
        if any(b not in files for b in needed) or "labels_metadata.json" not in files:
            stats["incomplete"] += 1
            return
        md = json.loads(files["labels_metadata.json"])
        rec = {"row": row, "pos": len(order), "patch": patch,
               "ulx": md["coordinates"].get("ulx"), "uly": md["coordinates"].get("uly"),
               "utm": _utm(md.get("projection"))}
        if labels:
            rec["labels"] = json.dumps(md["labels"])
        if kind == "S2":
            for g, (bands, s) in layout.S2_GROUPS.items():
                app[g].write(np.stack([tifffile.imread(io.BytesIO(files[b])) for b in bands]))
            rec["date"] = md.get("acquisition_date")
            rec["tile_source"] = md.get("tile_source")
        else:
            a = np.stack([tifffile.imread(io.BytesIO(files[b])) for b in needed])
            rec["nonfinite"] = int((~np.isfinite(a)).sum())
            app["S1"].write(a)
            rec["date"] = md.get("acquisition_time")
            rec["s2_named"] = md.get("corresponding_s2_patch")
        order.append(row)
        done_rows.add(row)
        meta.append(rec)
        stats["packed"] += 1

    proc = subprocess.Popen(["pigz", "-dc", str(archive)], stdout=subprocess.PIPE, bufsize=1 << 20)
    cur, buf = None, {}
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
        for m in tf:
            parts = m.name.split("/")
            if len(parts) != 3 or not m.isfile():
                continue
            patch = parts[1]
            if patch != cur:
                flush(cur, buf)
                cur, buf = patch, {}
                if stats["seen"] and stats["seen"] % log_every == 0:
                    dt = time.time() - t0
                    print(f"[{kind}] seen {stats['seen']} packed {stats['packed']}/{m_total} "
                          f"({stats['seen'] / dt:.0f} patches/s, {dt / 60:.1f} min)", flush=True)
            if patch not in row_of:
                continue                          # not wanted: never read the bytes
            suffix = parts[2][len(patch) + 1:]     # 'B02.tif' / 'VV.tif' / 'labels_metadata.json'
            buf[suffix[:-4] if suffix.endswith(".tif") else suffix] = tf.extractfile(m).read()
        flush(cur, buf)
    if proc.wait() != 0:
        raise RuntimeError("pigz failed")
    for a in app.values():
        a.close(expected=len(order))
    if len(order) != m_total:
        missing = sorted(want_rows - done_rows)
        raise RuntimeError(f"{kind}: {len(missing)} wanted rows never appeared, e.g. {missing[:5]}")
    np.save(out / f"order_{kind}.npy", np.asarray(order, dtype=np.int64))
    meta_df = pd.DataFrame(meta)
    assert labels or "labels" not in meta_df.columns
    meta_df.to_parquet(out / f"meta_{kind}.parquet")
    stats.update({"wanted": m_total, "minutes": round((time.time() - t0) / 60, 2), "labels_stored": labels})
    return stats
