"""The BigEarthNet v1 patch index: one row per clean S2/S1 pair, with every split it is in.

THE ROW ORDER of `index_v1.parquet` is the row order of every packed raster and every
token array under `/data/enric/data/BigEarthNet/`, exactly as `tok_index.parquet` is for
TerraMesh. Rows are the official S1/S2 mapping minus the official cloud/shadow and
seasonal-snow lists, sorted by S2 name.

Splits (PHASE6_PLAN.md §4.1, worklog §4 item 28, still open with the supervisor):

  sm_train / sm_train10 / sm_train1 / sm_val   the SMARTIES MM lists (pinned SHA)
  sm_none     clean, but in no SMARTIES list: presumably SMARTIES' unreleased test
  official    BEN v1's own train/val/test (NOT used for training or testing: SMARTIES'
              lists are a random re-split that draws from it, so it leaks, 2026-10-04)

Fine-tuning roles, chosen so that the fine-tunes do NOT depend on item 28's answer:

  ft_train    = sm_train10                           (31,166 pairs, Gianfranco's "10% train")
  ft_valsel   = a fixed random subsample of sm_train \\ sm_train10, for checkpoint
                selection. Disjoint from both test candidates, so it is a legitimate val set
                under option (a) AND option (c).
  test_a      = sm_val                               option (a)
  test_c      = sm_none                              option (c)

⚠ All of these are patch-level random splits, so neighbouring patches of one S2 tile sit on
both sides of every boundary. A spatial-autocorrelation CAVEAT (plan 3.6), not fixed here.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import labels as L
from . import layout

INDEX_VERSION = "v1"
VALSEL_SIZE = 5000
VALSEL_SEED = 20261012

SMARTIES_MM = {
    "sm_train": "BigEarthNet-MM-train.txt",
    "sm_train10": "BigEarthNet-MM-train10prcnt.txt",
    "sm_train1": "BigEarthNet-MM-train1prcnt.txt",
    "sm_val": "BigEarthNet-MM-val.txt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_pairs_csv_gz(path: Path) -> List[Tuple[str, str]]:
    out = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                s2, s1 = line.split(",")
                out.append((s2, s1))
    return out


def _read_names_csv_gz(path: Path) -> List[str]:
    """The cloud/snow lists: one S2 patch name per line (first field if there are more)."""
    with gzip.open(path, "rt") as fh:
        return [ln.strip().split(",")[0] for ln in fh if ln.strip()]


def read_smarties_mm(path: Path) -> List[Tuple[str, str]]:
    with open(path) as fh:
        return [tuple(ln.split()) for ln in fh if ln.strip()]


def build_index(smarties_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """The index WITHOUT labels (labels come from the archives, in `attach_labels`)."""
    v1 = layout.V1
    mapping = _read_pairs_csv_gz(v1 / "s1s2_mapping.csv.gz")
    cloud = set(_read_names_csv_gz(v1 / "patches_with_cloud_and_shadow.csv.gz"))
    snow = set(_read_names_csv_gz(v1 / "patches_with_seasonal_snow.csv.gz"))
    official = {}
    for split in ("train", "val", "test"):
        for s2, _ in _read_pairs_csv_gz(v1 / f"{split}.csv.gz"):
            official[s2] = split

    s2_to_s1 = dict(mapping)
    assert len(s2_to_s1) == len(mapping), "an S2 patch appears twice in the mapping"
    assert len(set(s2_to_s1.values())) == len(mapping), "an S1 patch appears twice in the mapping"

    clean = sorted(s2 for s2 in s2_to_s1 if s2 not in cloud and s2 not in snow)
    df = pd.DataFrame({"s2": clean})
    df["s1"] = df["s2"].map(s2_to_s1)
    df["official"] = df["s2"].map(official).fillna("none")
    df.insert(0, "row", np.arange(len(df), dtype=np.int64))

    prov: Dict = {"smarties_sha": L.SMARTIES_SHA, "smarties_files": {}}
    pos = pd.Series(df.index.values, index=df["s2"])
    for col, fname in SMARTIES_MM.items():
        f = smarties_dir / fname
        pairs = read_smarties_mm(f)
        prov["smarties_files"][fname] = {"sha256": sha256(f), "lines": len(pairs)}
        s2s = [p[0] for p in pairs]
        missing = [s for s in s2s if s not in pos.index]
        if missing:
            raise RuntimeError(f"{fname}: {len(missing)} S2 patches not in the clean index, "
                               f"e.g. {missing[:3]}. The lists were expected to be clean.")
        bad_pair = [p for p in pairs if s2_to_s1.get(p[0]) != p[1]]
        if bad_pair:
            raise RuntimeError(f"{fname}: {len(bad_pair)} pairs disagree with s1s2_mapping, "
                               f"e.g. {bad_pair[:2]}")
        df[col] = False
        df.loc[pos[s2s].values, col] = True

    assert not (df.sm_train & df.sm_val).any(), "SMARTIES train and val overlap"
    assert (df.sm_train10 <= df.sm_train).all(), "the 10% list is not a subset of train"
    df["sm_none"] = ~(df.sm_train | df.sm_val)

    # Fine-tuning roles.
    df["ft_train"] = df.sm_train10
    pool = df.index[df.sm_train & ~df.sm_train10].to_numpy()
    rng = np.random.default_rng(VALSEL_SEED)
    pick = np.sort(rng.choice(pool, size=VALSEL_SIZE, replace=False))
    df["ft_valsel"] = False
    df.loc[pick, "ft_valsel"] = True
    df["test_a"] = df.sm_val
    df["test_c"] = df.sm_none
    df = add_l8_roles(df)

    prov.update({
        "index_version": INDEX_VERSION,
        "n_mapping": len(mapping), "n_cloud_list": len(cloud), "n_snow_list": len(snow),
        "n_clean": len(df),
        "valsel": {"size": VALSEL_SIZE, "seed": VALSEL_SEED,
                   "pool": "sm_train & ~sm_train10", "pool_size": int(len(pool))},
        "counts": {c: int(df[c].sum()) for c in
                   list(SMARTIES_MM) + ["sm_none", "ft_train", "ft_valsel", "test_a", "test_c"] + list(L8_DERIVED)},
        "official_counts": df["official"].value_counts().to_dict(),
    })
    return df, prov


#: PHASE6_PLAN L8, label efficiency (2026-10-13). SMARTIES' 1% list is kept intact as the training
#: role; it is NOT a subset of the 10% list, and 47 of its patches fall in ft_valsel, so the 1% runs
#: select on ft_valsel minus those (ft_valsel1). Derived from existing columns only.
L8_DERIVED = ("ft_train1", "ft_valsel1")


def add_l8_roles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ft_train1"] = out.sm_train1.astype(bool)
    out["ft_valsel1"] = out.ft_valsel.astype(bool) & ~out.sm_train1.astype(bool)
    assert not (out.ft_train1 & (out.test_a | out.test_c | out.ft_valsel1)).any(), "ft_train1 overlaps an eval role"
    return out


def attach_labels(df: pd.DataFrame, meta_s2: pd.DataFrame,
                  meta_s1: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    """Add the 19-class labels (from the S2 metadata, as SMARTIES does) and check S1 agrees.

    Only PACKED rows have metadata (pack.py packs layout.PACK_ROLES); every other row gets
    labels19 = -1, which BenTokenDataset refuses to serve.
    """
    m2 = meta_s2.set_index("row")
    assert m2.index.is_unique
    have = df.row.isin(m2.index).to_numpy()
    labels43 = [None] * len(df)
    masks = np.full(len(df), -1, dtype=np.int64)
    n_dropped, dropped_names = 0, {}
    for i in np.flatnonzero(have):
        labs = m2.at[int(df.row.iat[i]), "labels"]
        labels43[i] = labs
        idx, dropped = L.to_19(json.loads(labs))
        masks[i] = L.to_mask(idx)
        n_dropped += bool(dropped)
        for d in dropped:
            dropped_names[d] = dropped_names.get(d, 0) + 1
    out = df.copy()
    out["labels43"] = labels43
    out["labels19"] = masks
    out["n_labels19"] = [bin(int(m)).count("1") if m >= 0 else -1 for m in masks]
    out["date"] = out.row.map(m2["date"])
    lab = out[have]
    rep = {"rows_labelled": int(have.sum()), "rows_unlabelled": int((~have).sum()),
           "rows_with_a_dropped_label": n_dropped, "dropped_label_counts": dropped_names,
           "rows_with_no_19_label": int((lab.n_labels19 == 0).sum()),
           "max_labels19": int(lab.n_labels19.max()),
           "label_count_hist": lab.n_labels19.value_counts().sort_index().to_dict()}
    for role in ("ft_train", "ft_valsel", "test_a", "test_c") + L8_DERIVED:
        if role in out.columns:
            sub = out[out[role]]
            rep[f"{role}_unlabelled"] = int((sub.labels19 < 0).sum())
            rep[f"{role}_no_19_label"] = int((sub.n_labels19 == 0).sum())
    if meta_s1 is not None:
        m1 = meta_s1.set_index("row")
        common = np.intersect1d(m2.index.values, m1.index.values)
        a = [sorted(json.loads(x)) for x in m1.loc[common, "labels"]]
        b = [sorted(json.loads(x)) for x in m2.loc[common, "labels"]]
        named = m1.loc[common, "s2_named"].tolist() if "s2_named" in m1.columns else None
        rep["s1_s2_label_agreement"] = {"compared": int(len(common)),
                                        "disagree": int(sum(x != y for x, y in zip(a, b)))}
        if named is not None:
            s2_of = df.set_index("row").loc[common, "s2"].tolist()
            rep["s1_metadata_names_its_s2"] = {"compared": len(common),
                                               "mismatch": int(sum(x != y for x, y in zip(named, s2_of)))}
    return out, rep
