# eo/data/temporal_split.py
"""Val's four-date locations, and their dev/test split for the temporal stage (`.venv`).

TerraMesh's `ssl4eos12` corpus stores one location's four dates as four rows, linked only by
`sample_id` = `<location>_<date index>`. This module joins them, and draws the split of val's
2,176 locations once:

    dev    544 locations: every choice the temporal stage makes is read here
    test   the other 1,632, of which 1,578 stay unread until the frozen base is scored once

Both are held out from every T360-family model, which never saw a val row. The split exists
because the pilot picks settings: scored on the same locations, the final numbers would be
selected on themselves.

⚠ THE OTHER 54. eval_rows_v1.json holds 96 of these locations, 42 in dev and 54 in test, and they
are the only S1GRD rows in it. Forgetting is read on those rows while settings are chosen, so the 54
are not unread on S1GRD. `unread_test` drops them; the file itself stays as drawn.

WHY WHOLE SPATIAL GROUPS. Locations are drawn around cities, so some lie within a tile of each
other. Locations closer than 5 km (eval_split.SPLIT_RADIUS_KM) stay on one side, and the same
groups are what the paired bootstrap resamples. Unlike eval_split, the clustering runs over
these locations alone: its reason for clustering across corpora was training on val's other
rows, which a train-split model does not do.

⚠ As with eval_rows_v1.json the file stores row INDICES, so load_temporal_split checks the
hashes of what they index: a regenerated tok_index.parquet would make them name other scenes,
silently.

Runs in `.venv`: numpy and pandas only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from eo.data.eval_split import (DEFAULT_METADATA_PATH, SPLIT_RADIUS_KM, _sha256, load_eval_rows, load_row_table,
                                spatial_groups)

SPLIT_TAG = "v1"
SPLIT_SEED = 20261105          # fixed; the split is drawn exactly once
N_DATES = 4
N_DEV = 544                    # a quarter of 2,176; divides by 16 x 2, the in-training eval's batch


def split_path(tag: str = SPLIT_TAG) -> Path:
    return Path(__file__).resolve().parent / f"temporal_rows_{tag}.json"


def location_table(root_dir, metadata_path=None):
    """One row per location, four `row_<t>` columns in chronological order. Raises if a location
    is not a complete, chronological quadruple."""
    import pandas as pd

    meta = pd.read_parquet(metadata_path or DEFAULT_METADATA_PATH, columns=["zarr", "S2_time"])
    meta["stem"] = meta["zarr"].str[: -len(".zarr.zip")]
    table = load_row_table(root_dir, metadata_path).merge(meta[["stem", "S2_time"]], on="stem", validate="1:1")
    ssl = table[table.corpus == "ssl4eos12"].copy()
    ssl["location"] = ssl.sample_id.str.rsplit("_", n=1).str[0]
    ssl["t"] = ssl.sample_id.str.rsplit("_", n=1).str[1].astype(int)
    size = ssl.groupby("location").t.agg(["size", "nunique"])
    if not ((size["size"] == N_DATES) & (size["nunique"] == N_DATES)).all():
        raise RuntimeError(f"{int(((size['size'] != N_DATES) | (size['nunique'] != N_DATES)).sum())} locations "
                           f"are not complete quadruples")
    ssl = ssl.sort_values(["location", "t"])
    if not (ssl.groupby("location").S2_time.diff().dropna() > pd.Timedelta(0)).all():
        raise RuntimeError("a location's S2 dates are not strictly increasing with its date index")
    wide = ssl.pivot(index="location", columns="t", values="row")
    wide.columns = [f"row_{t}" for t in wide.columns]
    pos = ssl.groupby("location")[["center_lat", "center_lon"]].first()
    if (ssl.groupby("location").center_lat.nunique() != 1).any():
        raise RuntimeError("a location's rows disagree on its centre")
    return wide.join(pos).reset_index()


def _draw_stratum(groups: np.ndarray, sizes: np.ndarray, want: int, rng: np.random.Generator) -> np.ndarray:
    """Groups adding up to exactly `want` locations, in a seeded random order. A group that would
    overshoot is skipped and the walk goes on, so the count lands exactly."""
    picked, total = [], 0
    for i in rng.permutation(len(groups)):
        if total == want:
            break
        if total + sizes[i] <= want:
            picked.append(groups[i])
            total += int(sizes[i])
    if total != want:
        raise RuntimeError(f"could not reach {want} locations by whole groups (got {total})")
    return np.asarray(picked, dtype=np.int64)


def build_split(root_dir, metadata_path=None, seed: int = SPLIT_SEED, tag: str = SPLIT_TAG,
                n_dev: int = N_DEV, created: str = "2026-11-05") -> Dict:
    """Draw dev by whole groups, within each hemisphere, so both sides keep its seasons' mix."""
    loc = location_table(root_dir, metadata_path)
    loc["group"] = spatial_groups(loc.center_lat.values, loc.center_lon.values, SPLIT_RADIUS_KM)
    lat = loc.groupby("group").center_lat.mean()
    size = loc.groupby("group").size()
    south = lat.values < 0
    rng = np.random.default_rng(seed)
    n_south = int(round(n_dev * (loc.center_lat < 0).mean()))
    dev_groups = np.concatenate([
        _draw_stratum(lat.index.values[~south], size.values[~south], n_dev - n_south, rng),
        _draw_stratum(lat.index.values[south], size.values[south], n_south, rng)])
    is_dev = loc.group.isin(set(dev_groups.tolist())).values
    rows = [f"row_{t}" for t in range(N_DATES)]

    def side(mask):
        d = loc[mask]
        return {"locations": d.location.tolist(), "groups": [int(g) for g in d.group],
                "rows": [[int(r) for r in x] for x in d[rows].values]}

    root = Path(root_dir)
    return {
        "tag": tag, "created": created, "seed": seed,
        "group_key": {"kind": "single_linkage_spatial", "radius_km": SPLIT_RADIUS_KM, "over": "ssl4eos12 locations"},
        "n_locations": int(len(loc)), "n_groups": int(loc.group.nunique()),
        "n_dev": int(is_dev.sum()), "n_test": int((~is_dev).sum()),
        "n_dev_groups": int(len(dev_groups)),
        "south_share": {"dev": float((loc.center_lat[is_dev] < 0).mean()),
                        "test": float((loc.center_lat[~is_dev] < 0).mean())},
        "inputs": {"tok_index.parquet": _sha256(root / "tok_index.parquet"),
                   "val_metadata.parquet": _sha256(Path(metadata_path or DEFAULT_METADATA_PATH))},
        "dev": side(is_dev), "test": side(~is_dev),
    }


def unread_test(doc: Dict, root_dir, metadata_path=None) -> Dict:
    """The test side without the locations whose rows are in eval_rows_v1.json (module docstring)."""
    held = set(int(r) for r in load_eval_rows(root_dir=root_dir, metadata_path=metadata_path))
    test = doc["test"]
    keep = [i for i, rows in enumerate(test["rows"]) if not held.intersection(rows)]
    return {k: [test[k][i] for i in keep] for k in ("locations", "groups", "rows")}


def load_temporal_split(root_dir=None, tag: str = SPLIT_TAG, path=None, metadata_path=None) -> Dict:
    """The committed split, checked against the data it indexes when `root_dir` is given."""
    path = Path(path) if path is not None else split_path(tag)
    doc = json.loads(path.read_text())
    if root_dir is not None:
        actual = {"tok_index.parquet": _sha256(Path(root_dir) / "tok_index.parquet"),
                  "val_metadata.parquet": _sha256(Path(metadata_path or DEFAULT_METADATA_PATH))}
        for name, want in doc["inputs"].items():
            if actual[name] != want:
                raise RuntimeError(f"{path.name} was built against another {name}: {want[:12]} vs "
                                   f"{actual[name][:12]}. Its row indices would name other scenes.")
    dev, test = doc["dev"], doc["test"]
    seen = set(dev["locations"]) | set(test["locations"])
    if len(seen) != doc["n_dev"] + doc["n_test"]:
        raise RuntimeError(f"{path.name}: dev and test overlap or repeat a location")
    if set(dev["groups"]) & set(test["groups"]):
        raise RuntimeError(f"{path.name}: a spatial group sits on both sides")
    return doc
