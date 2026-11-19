"""GEO-Bench-2's PASTIS and CloudSen12: pinned files and their readers (A2 Step 2.1-2.2).

Both are tortillas (tacoreader 0.5.6).
  - PASTIS on disk: aialliance/pastis at b2e184f2 (2025-11-17), the revision GeoBenchV2 0.9 (the latest
    release) pins by sha256; three parts, 47.8 GB. It holds train 1,200, val 482, test 496: its train is
    a subsample of PASTIS-R folds 1-3 (seed 24 in GEO-Bench-2's generator).
  - The Hub's head, b4918975 (2025-12-15), holds all of PASTIS-R: train 1,455 = folds 1-3, val 482 =
    fold 4, test 496 = fold 5, which is GEO-Bench-2's Table 2 and the revision torchgeo's
    geo-eval-harness pins. A2 takes it as the PASTIS protocol (worklog, 2026-11-19).
    Its 255 training patches that the pinned revision lacks are read from the local PASTIS-R, after
    eo/experiments/segmentation/step2_splits.py --dec checked every one of them (S2 series, semantic mask,
    dates) byte for byte against the head revision, read remotely. Every other patch is read from the
    pinned tortilla, which equals PASTIS-R too (the same script).
  - CloudSen12: aialliance/cloudsen12 at 7b72e755, one file, 9.94 GB: 4,000 / 535 / 975, the "high"
    label tag, 512 px. It is L2A (12 bands, then AOT and WVP), not L1C (A2 Step 2.4).

`pastis_table()` has one row per patch of the 1,455 / 482 / 496 protocol; `in_pinned` marks the 2,178 the
pinned revision holds. Test rows may be read for tokenizing only; nothing here scores. Split names are
ours: train, val, test.
"""
from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/enric/data/geobench2")
PASTIS_DIR = ROOT / "pastis"
PASTIS_FILES = ("geobench_pastis.0000.part.tortilla", "geobench_pastis.0001.part.tortilla",
                "geobench_pastis.0002.part.tortilla")
PASTIS_SHA256 = ("56b1490c6dc7345fdff79e94d9132753ee28d8504bb061d8db39d19e888f7ca3",
                 "7d0463a695a0822a1f25638598b2d54daf28f387d6a80d353be7c323069060db",
                 "22d27389b1ccee4f250f4a187d034d12dbf15c5610a5dd8d502ee8783e94c81e")
PASTIS_REV = "b2e184f2c09c76b5e54daaa66d936e38f7680488"
PASTIS_REV_HEAD = "b4918975bf6650250715a0f922fd496266737a96"
PASTIS_HEAD_SHA256 = ("56b1490c6dc7345fdff79e94d9132753ee28d8504bb061d8db39d19e888f7ca3",   # as geo-eval-harness pins
                      "3f1e98e304bf6021fd522c0b24f80a3a028feab9bd7238db364398580baf1c2e",
                      "b249a8e71d1c6a755457621caf51ebb67f9174eca12eeb6750953c7cefcc9575")
CLOUDSEN12_FILE = ROOT / "cloudsen12" / "geobench_cloudsen12.tortilla"
CLOUDSEN12_SHA256 = "16b3c03d7b15cf42f6ef0cee6d453b6ad8ebbe7744674c4b58657511f7f5d0c0"
CLOUDSEN12_REV = "7b72e755b787231c2263ad70c9d04eba38bdbb93"
PASTIS_R = Path("/data/enric/data/PASTIS/PASTIS-R")

SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}
COUNTS = {"pastis": {"train": 1455, "val": 482, "test": 496},
          "cloudsen12": {"train": 4000, "val": 535, "test": 975}}
PINNED_COUNTS = {"pastis": {"train": 1200, "val": 482, "test": 496}}
TABLE2 = {"pastis": {"train": 1455, "val": 482, "test": 496},
          "cloudsen12": {"train": 4000, "val": 535, "test": 975}}


def _subfile(path: str) -> bytes:
    """Bytes of a '/vsisubfile/<offset>_<size>,<file>' path."""
    m = re.search(r"(\d+)_(\d+),(.+)", path)
    off, size, fn = int(m.group(1)), int(m.group(2)), m.group(3)
    with open(fn, "rb") as fh:
        fh.seek(off)
        return fh.read(size)


@lru_cache(maxsize=None)
def _pastis_taco():
    import tacoreader
    return tacoreader.load([str(PASTIS_DIR / f) for f in PASTIS_FILES])


@lru_cache(maxsize=None)
def _cloudsen12_taco():
    import tacoreader
    return tacoreader.load(str(CLOUDSEN12_FILE))


def pinned_pastis_table():
    """The pinned tortilla as stored: idx (its position), patch_id, split, tile, dates, lon, lat."""
    df = _pastis_taco()
    out = df[["tortilla:data_split", "patch_id", "tile", "dates", "lon", "lat"]].copy()
    out["split"] = out.pop("tortilla:data_split").map(SPLIT_NAMES)
    out["patch_id"] = out.patch_id.astype(int)
    out.insert(0, "idx", np.arange(len(out)))
    return pd.DataFrame(out).reset_index(drop=True)


@lru_cache(maxsize=None)
def _pastis_table() -> pd.DataFrame:
    pin = pinned_pastis_table()
    folds = pastis_r_fold_ids()
    have = set(pin.patch_id)
    extra = sorted(p for p, f in folds.items() if f in (1, 2, 3) and p not in have)
    meta = _pastis_r_meta()
    ex = pd.DataFrame({"taco_idx": -1, "patch_id": extra, "tile": [meta[p]["TILE"] for p in extra],
                       "dates": [np.array(pastis_r_dates(p)) for p in extra], "lon": np.nan, "lat": np.nan,
                       "split": "train", "in_pinned": False})
    pin = pin.rename(columns={"idx": "taco_idx"}).assign(in_pinned=True)
    t = pd.concat([pin, ex], ignore_index=True)
    order = {"train": 0, "val": 1, "test": 2}
    t = t.sort_values(["split", "patch_id"], key=lambda c: c.map(order) if c.name == "split" else c).reset_index(drop=True)
    t.insert(0, "idx", np.arange(len(t)))
    return t


def pastis_table() -> pd.DataFrame:
    """GEO-Bench-2 PASTIS at the head revision's splits (1,455 / 482 / 496), one row per patch, ordered by
    split then patch id: idx (this table's position, the views' sample_idx), taco_idx (position in the
    pinned tortilla, -1 for the 255 read from PASTIS-R), patch_id, split, tile, dates (the S2 dates in
    series order), lon, lat, in_pinned. A fresh copy on every call."""
    return _pastis_table().copy()


def _h5(path: str) -> np.ndarray:
    import h5py
    with h5py.File(io.BytesIO(_subfile(path)), "r") as h:
        return h["data"][()]


def pastis_sample(idx: int, what=("s2", "semantic")) -> dict:
    """{'s2': (T, 10, 128, 128) int16 DN, 'semantic': (128, 128) uint8 (channel 0)} of the patch at
    pastis_table() position `idx`: from the pinned tortilla, or PASTIS-R for the 255 it lacks."""
    r = _pastis_table().iloc[int(idx)]
    out = {}
    if r.taco_idx >= 0:
        row = _pastis_taco().read(int(r.taco_idx))
        pos = {"s2": 0, "s1a": 1, "s1d": 2, "semantic": 3, "instance": 4}
        for w in what:
            a = _h5(row.read(pos[w]))
            out[w] = a[0] if w == "semantic" else a
        return out
    for w in what:
        if w == "s2":
            out[w] = np.load(PASTIS_R / "DATA_S2" / f"S2_{int(r.patch_id)}.npy")
        elif w == "semantic":
            out[w] = np.load(PASTIS_R / "ANNOTATIONS" / f"TARGET_{int(r.patch_id)}.npy")[0]
        else:
            raise KeyError(w)
    return out


def pastis_frames_p7(n_dates: int, n: int = 7) -> list[int]:
    """GEO-Bench-2's frame indices at num_time_steps n (datasets/pastis.py, `_load_image`):
    int(i * T / n). Their dates are read from the patch's own `dates` at these indices (finding 9:
    the loader's own date metadata takes the LAST n dates, which are not the frames it returns)."""
    if n_dates < n:
        raise ValueError(f"{n_dates} dates < {n}: GEO-Bench-2 zero-pads in front, not handled here")
    step = n_dates / n
    return [int(i * step) for i in range(n)]


def cloudsen12_table():
    """GEO-Bench-2 CloudSen12, one row per 512-px tile: idx, id (tortilla id), split, s2_id, date."""
    df = _cloudsen12_taco()
    out = df[["tortilla:id", "tortilla:data_split", "s2_id", "roi_id", "clear_percentage"]].copy()
    out.columns = ["id", "split", "s2_id", "roi_id", "clear_percentage"]
    out["split"] = out.split.map(SPLIT_NAMES)
    out["date"] = out.s2_id.str.extract(r"MSI\w{3}_(\d{8})")[0].astype(int)
    out.insert(0, "idx", np.arange(len(out)))
    return pd.DataFrame(out).reset_index(drop=True)


def cloudsen12_sample(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """(image (14, 512, 512) uint16 DN: B01..B12 without B10, then AOT, WVP; mask (512, 512) uint8)."""
    import rasterio
    taco = _cloudsen12_taco()
    row = taco.read(int(idx))
    with rasterio.open(row.read(0)) as s:
        img = s.read()
    with rasterio.open(row.read(1)) as s:
        m = s.read()
    assert img.shape == (14, 512, 512) and m.shape == (1, 512, 512), (img.shape, m.shape)
    return img, m[0]


@lru_cache(maxsize=None)
def _pastis_r_meta() -> dict:
    import json
    return {int(ft["properties"]["ID_PATCH"]): ft["properties"]
            for ft in json.loads((PASTIS_R / "metadata.geojson").read_text())["features"]}


def pastis_r_fold_ids() -> dict[int, int]:
    """PASTIS-R's own fold per patch id (metadata.geojson)."""
    return {p: int(m["Fold"]) for p, m in _pastis_r_meta().items()}


def pastis_r_dates(patch_id: int) -> list[int]:
    import json
    d = _pastis_r_meta()[int(patch_id)]["dates-S2"]
    d = json.loads(d.replace("'", '"')) if isinstance(d, str) else d
    return [int(d[k]) for k in sorted(d, key=int)]
