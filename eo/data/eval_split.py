# eo/data/eval_split.py
"""The held-out evaluation split for Phase 3 (D3.1).

PLAN: docs/PHASE3_PLAN.md Step 1. Where this file and the plan disagree about
the GROUP KEY, this file is right and the plan is superseded -- see "The group
key is geographic, not textual" below. Everything else follows the plan.

WHY A GROUP SPLIT AT ALL. A random row split leaks, measured on all 89,088
rows (2026-09-20, plan Step 1.1): `majortom` is 5,187 MajorTom cells x up to 16
adjacent sub-tiles, so 15 of a cell's 16 neighbours land in train and 1 in
eval; `ssl4eos12` is 2,176 locations x exactly 4 seasonal revisits of the
IDENTICAL coordinate (lat/lon std measured at exactly 0.0). Shard-level holdout
buys nothing either -- shards are globally shuffled, not geographic.

THE GROUP KEY IS GEOGRAPHIC, NOT TEXTUAL, and this is a correction to the plan.
Plan Step 1.2 specifies parsing `sample_id`: `{row}_{col}` for majortom,
`{location}` for ssl4eos12, grouped and held out WITHIN each corpus. Measured
2026-09-22, that is not sufficient, because the two corpora are not spatially
independent:

  * Every tile is exactly 2,640 m square (measured off `bounds`, both corpora).
  * 7,896 of 8,704 ssl4eos12 tiles (90.7%) physically OVERLAP a majortom tile.
    Closest cross-corpus pair: 20 metres.
  * 604 of 2,176 ssl4eos12 locations (27.8%) overlap a DIFFERENT ssl4eos12
    location, so even the location id does not isolate one patch of ground.

So a per-corpus textual key holds out an ssl4eos12 location and leaves the same
ground in training inside a majortom tile, nine times out of ten. It satisfies
"train and eval are disjoint at the group level" while leaking exactly what the
group was introduced to prevent -- the same shape of silent-absence bug this
project keeps finding.

The key used instead: single-linkage clustering of TILE CENTROIDS at
SPLIT_RADIUS_KM, across both corpora at once. Measured cluster counts:

  | radius  | clusters | median rows | largest cluster | spans both corpora |
  |---------|----------|-------------|-----------------|--------------------|
  | 2.6 km  |   78,362 |           1 |  112 (0.13%)    |              1,296 |
  | 5.0 km  |    5,103 |          16 |  112 (0.13%)    |                753 |
  | 10.0 km |    5,069 |          16 |  152 (0.17%)    |                750 |

5 km is the operating point. 2.6 km (one tile width, i.e. literal pixel
overlap) leaves the 4x4 cell adjacency leak the plan identified -- adjacent
tiles touch at exactly 2,640 m and so never merge. 5 km recovers the MajorTom
cell almost exactly (5,103 clusters at median 16 rows, against 5,187
hand-parsed cells) while additionally absorbing the seasonal revisits and the
cross-corpus duplicates. 10 km changes almost nothing, so the choice is not on
a cliff edge.

⚠ Single-linkage chains, and that was the risk worth measuring: adjacent
MajorTom cells could in principle merge into one continent-sized cluster and
make the split unsplittable. Measured, they do not -- the largest cluster is
112 rows (0.13% of the corpus) even at 10 km, because MajorTom coverage is
patchy rather than contiguous. Re-measure this before raising the radius.

STRATIFICATION is by whether a cluster contains any ssl4eos12 row, not by
corpus (a cluster can hold both). ⚠ S1GRD exists ONLY on ssl4eos12 rows and
S1RTC ONLY on majortom rows -- they are exact complements -- so a holdout drawn
without regard to this can silently contain zero S1GRD scenes and produce no
S1GRD eval loss at all. Same class of bug as Phase 2's missing Coords.

⚠ ONE-WAY DOOR. Once an arm is trained against a split, changing it invalidates
every comparison made on it. The split is therefore MATERIALIZED to a committed
JSON artifact and read back, never recomputed from a seed at launch -- a split
recomputed per launch is a split that moves when anything upstream does. The
artifact records hashes of both parquet inputs and `load_eval_rows` verifies
them.

Runs in the repo `.venv`, NOT the `mor` conda env: numpy/pandas/pyarrow only,
never terratorch.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- the split's defining constants ------------------------------------
# Changing any of these changes the split, which invalidates every checkpoint
# trained against it. They are constants rather than CLI defaults so that a
# regenerated artifact is bit-identical unless the change was deliberate.
SPLIT_RADIUS_KM = 5.0        # single-linkage radius for the geographic group
EVAL_GROUP_FRACTION = 0.05   # share of GROUPS held out, per stratum
SPLIT_SEED = 20260922        # fixed; the split is drawn exactly once
SPLIT_TAG = "v1"

EARTH_RADIUS_KM = 6371.0
TILE_WIDTH_M = 2640.0        # measured off `bounds`, identical in both corpora

DEFAULT_METADATA_PATH = "/data/enric/data/TerraMesh/val_metadata.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_row_table(root_dir, metadata_path=None):
    """Join the canonical row order to the per-sample geographic metadata.

    `tok_index.parquet` defines the row order every modality's tokens.npy uses;
    `val_metadata.parquet` carries the coordinates but has no row column. They
    join on the zarr stem. The join is validated one-to-one: a many-to-one join
    here would silently duplicate or drop rows and every downstream count would
    still look plausible.
    """
    import pandas as pd

    root = Path(root_dir)
    idx_path = root / "tok_index.parquet"
    meta_path = Path(metadata_path or DEFAULT_METADATA_PATH)

    idx = pd.read_parquet(idx_path)
    meta = pd.read_parquet(meta_path)
    meta["stem"] = meta["zarr"].str[: -len(".zarr.zip")]

    joined = idx.merge(
        meta[["stem", "sample_id", "center_lon", "center_lat"]],
        on="stem",
        how="left",
        validate="one_to_one",
    )
    if joined["sample_id"].isna().any():
        missing = int(joined["sample_id"].isna().sum())
        raise RuntimeError(
            f"{missing} of {len(joined)} rows in {idx_path} have no match in "
            f"{meta_path}. The split cannot be built without coordinates for "
            f"every row."
        )
    return joined.sort_values("row").reset_index(drop=True)


def _unit_positions(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Tile centroids as 3D cartesian kilometres.

    Clustering happens in 3D rather than in degrees on purpose: a degree of
    longitude is 111 km at the equator and 38 km at 70 deg N, so a
    degree-spaced bucket grid would silently use a different radius at
    different latitudes -- and TerraMesh is global.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    return np.stack(
        [
            EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon),
            EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon),
            EARTH_RADIUS_KM * np.sin(lat),
        ],
        axis=1,
    )


def spatial_groups(lat_deg, lon_deg, radius_km: float = SPLIT_RADIUS_KM) -> np.ndarray:
    """Single-linkage spatial clustering. Returns one group label per row.

    Chord distance is used rather than great-circle: below a few hundred km the
    two agree to far better than the tile size, and the buckets have to be
    cartesian anyway.

    Labels are a pure function of the coordinates -- clusters are renumbered in
    order of the lowest row index they contain -- so rebuilding the split on the
    same inputs reproduces the same labels, with no dependence on dict order or
    on the seed.
    """
    pos = _unit_positions(lat_deg, lon_deg)
    n = len(pos)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    # Bucket at the radius, then only compare within the 27 touching buckets:
    # two points closer than `radius_km` cannot be further apart than one
    # bucket in any axis.
    keys = np.floor(pos / radius_km).astype(np.int64)
    buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i, key in enumerate(map(tuple, keys)):
        buckets[key].append(i)

    r2 = radius_km * radius_km
    for i in range(n):
        kx, ky, kz = keys[i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in buckets.get((kx + dx, ky + dy, kz + dz), ()):
                        if q <= i:
                            continue
                        d = pos[i] - pos[q]
                        if float(d @ d) < r2:
                            ra, rb = find(i), find(q)
                            if ra != rb:
                                parent[ra] = rb

    roots = np.array([find(i) for i in range(n)])
    # Renumber by first appearance in row order -> deterministic labels.
    order: Dict[int, int] = {}
    labels = np.empty(n, dtype=np.int64)
    for i, r in enumerate(roots):
        r = int(r)
        if r not in order:
            order[r] = len(order)
        labels[i] = order[r]
    return labels


def build_split(
    root_dir,
    metadata_path=None,
    radius_km: float = SPLIT_RADIUS_KM,
    eval_fraction: float = EVAL_GROUP_FRACTION,
    seed: int = SPLIT_SEED,
    tag: str = SPLIT_TAG,
) -> Dict:
    """Draw the held-out split and return the artifact as a dict.

    Holds out `eval_fraction` of GROUPS (not rows) within each stratum, where
    the stratum is "does this group contain any ssl4eos12 row". Groups vary in
    size, so the row count lands near but not exactly on the fraction -- that is
    the correct behaviour and the artifact records what was actually drawn.
    """
    import pandas as pd

    table = load_row_table(root_dir, metadata_path)
    labels = spatial_groups(table.center_lat.values, table.center_lon.values, radius_km)
    table = table.assign(group=labels)

    # A group is "ssl4eos12-bearing" if any of its rows is; that is the stratum
    # that carries S1GRD, and it is a property of the group, not of the row.
    has_ssl = (
        table.assign(is_ssl=table.corpus.eq("ssl4eos12"))
        .groupby("group")
        .is_ssl.any()
    )

    rng = np.random.default_rng(seed)
    eval_groups: List[int] = []
    strata_counts: Dict[str, Dict[str, int]] = {}
    for stratum_name, want in (("ssl4eos12_bearing", True), ("majortom_only", False)):
        members = np.sort(has_ssl.index.values[has_ssl.values == want])
        if len(members) == 0:
            raise RuntimeError(f"stratum '{stratum_name}' is empty; the split would be unstratified.")
        k = int(np.ceil(eval_fraction * len(members)))
        picked = rng.choice(members, size=k, replace=False)
        eval_groups.extend(int(g) for g in picked)
        strata_counts[stratum_name] = {"groups_total": int(len(members)), "groups_held_out": int(k)}

    eval_group_set = set(eval_groups)
    is_eval = table.group.isin(eval_group_set).values
    eval_rows = np.sort(table.row.values[is_eval]).astype(np.int64)

    per_corpus = (
        table.assign(is_eval=is_eval).groupby("corpus").is_eval.agg(["sum", "size"])
    )
    counts_per_corpus = {
        str(c): {"eval_rows": int(r["sum"]), "total_rows": int(r["size"])}
        for c, r in per_corpus.iterrows()
    }

    root = Path(root_dir)
    return {
        "tag": tag,
        "created": "2026-09-22",
        "deliverable": "D3.1",
        "group_key": {
            "kind": "single_linkage_spatial",
            "radius_km": radius_km,
            "corpus_agnostic": True,
            "note": (
                "Tile centroids clustered across both corpora. NOT the plan's "
                "per-corpus sample_id parse: 90.7% of ssl4eos12 tiles overlap a "
                "majortom tile, so a per-corpus key leaks. See module docstring."
            ),
        },
        "seed": seed,
        "eval_fraction_of_groups": eval_fraction,
        "groups_total": int(table.group.nunique()),
        "groups_held_out": len(eval_group_set),
        "strata": strata_counts,
        "counts_per_corpus": counts_per_corpus,
        "n_eval_rows": int(len(eval_rows)),
        "n_total_rows": int(len(table)),
        "inputs": {
            "tok_index.parquet": _sha256(root / "tok_index.parquet"),
            "val_metadata.parquet": _sha256(Path(metadata_path or DEFAULT_METADATA_PATH)),
        },
        "eval_rows": [int(r) for r in eval_rows],
    }


def default_split_path(tag: str = SPLIT_TAG) -> Path:
    return Path(__file__).resolve().parent / f"eval_rows_{tag}.json"


def load_eval_rows(
    path=None,
    root_dir=None,
    metadata_path=None,
    verify_inputs: bool = True,
) -> np.ndarray:
    """Read the committed split back, verifying it still describes this data.

    ⚠ The hash check is the point. The split is a list of integer row indices
    into `tok_index.parquet`'s row order; if that file is ever regenerated the
    indices still resolve, still look like valid rows, and silently name
    different scenes. Nothing downstream could detect it.
    """
    path = Path(path) if path is not None else default_split_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No eval split at {path}. Build it with:\n"
            f"  python eo/scripts/build_eval_split.py"
        )
    doc = json.loads(path.read_text())

    if verify_inputs and root_dir is not None:
        root = Path(root_dir)
        expected = doc.get("inputs", {})
        actual = {
            "tok_index.parquet": _sha256(root / "tok_index.parquet"),
            "val_metadata.parquet": _sha256(Path(metadata_path or DEFAULT_METADATA_PATH)),
        }
        for name, want in expected.items():
            if name in actual and actual[name] != want:
                raise RuntimeError(
                    f"{path.name} was built against a different {name}.\n"
                    f"  recorded: {want}\n  on disk:  {actual[name]}\n"
                    f"  The split is a list of row INDICES, so a regenerated "
                    f"artifact makes it name different scenes with no error. "
                    f"Rebuild the split and retrain, or restore the original file."
                )

    rows = np.asarray(doc["eval_rows"], dtype=np.int64)
    if len(rows) != doc["n_eval_rows"]:
        raise RuntimeError(
            f"{path.name}: n_eval_rows says {doc['n_eval_rows']} but the list holds {len(rows)}."
        )
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError(f"{path.name}: eval_rows contains duplicates.")
    return rows


def train_rows_from_eval(eval_rows: Sequence[int], n_total: int) -> np.ndarray:
    """The complement. Train is defined as everything not held out, so the two
    cannot overlap by construction rather than by a later check."""
    mask = np.ones(int(n_total), dtype=bool)
    ev = np.asarray(eval_rows, dtype=np.int64)
    if ev.size and (ev.min() < 0 or ev.max() >= n_total):
        raise RuntimeError(
            f"eval rows fall outside [0, {n_total}); the split does not match this artifact."
        )
    mask[ev] = False
    return np.nonzero(mask)[0].astype(np.int64)
