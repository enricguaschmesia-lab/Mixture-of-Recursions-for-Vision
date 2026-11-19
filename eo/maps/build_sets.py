"""Build A2 Step 1's fixed map sets and their manifest (a2seg env).

    python -m eo.maps.build_sets            # all three datasets, then the manifest

PASTIS: PASTIS-R's semantic targets (channel 0 of TARGET_<id>.npy), folds from metadata.geojson.
CloudSen12: GEO-Bench-2's tortilla (geobench_v2 0.9), train and validation masks only; the test rows
are counted from the metadata and never opened.
LULC: TerraMesh val's raw LULC tars, centre-cropped to 224 (eo.terramesh_tok.preprocess.center_crop),
for the eligible rows only (see eo/maps/sets.py), split by 5 km group into a read set and a pool.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eo.maps.sets import MANIFEST, ROOT, _sha256

PASTIS = Path("/data/enric/data/PASTIS/PASTIS-R")
CLOUDSEN12 = Path("/data/enric/data/geobench2/cloudsen12/geobench_cloudsen12.tortilla")
CLOUDSEN12_SHA = "16b3c03d7b15cf42f6ef0cee6d453b6ad8ebbe7744674c4b58657511f7f5d0c0"
CLOUDSEN12_REV = "7b72e755b787231c2263ad70c9d04eba38bdbb93"   # aialliance/cloudsen12 on the Hub
TM_VAL = "/data/enric/data/TerraMesh/val"
SEED = 20261118
LULC_READ_ROWS = 2000
SUB = {"cloudsen12": 1000, "lulc": 5000}


def _save(name: str, labels: np.ndarray, ids: np.ndarray) -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    lp, ip = ROOT / f"{name}_labels.npy", ROOT / f"{name}_ids.npy"
    np.save(lp, np.ascontiguousarray(labels.astype(np.uint8)))
    np.save(ip, ids)
    vals, cnt = np.unique(labels, return_counts=True)
    return {"labels": lp.name, "ids": ip.name, "labels_sha256": _sha256(lp), "ids_sha256": _sha256(ip),
            "shape": list(labels.shape), "n": int(len(labels)),
            "class_pixels": {int(v): int(c) for v, c in zip(vals, cnt)}}


def _subset(name: str, n_total: int, k: int | None) -> dict:
    sub = np.arange(n_total) if k is None or k >= n_total else \
        np.sort(np.random.default_rng(SEED).permutation(n_total)[:k])
    p = ROOT / f"{name}_sub.npy"
    np.save(p, sub.astype(np.int64))
    return {"subset": p.name, "subset_n": int(len(sub)), "subset_sha256": _sha256(p)}


def build_pastis() -> dict:
    meta = json.loads((PASTIS / "metadata.geojson").read_text())
    fold = {int(f["properties"]["ID_PATCH"]): int(f["properties"]["Fold"]) for f in meta["features"]}
    out = {}
    for split, folds in (("train", (1, 2, 3)), ("val", (4,))):
        ids = np.array(sorted(i for i, f in fold.items() if f in folds), dtype=np.int64)
        lab = np.stack([np.load(PASTIS / "ANNOTATIONS" / f"TARGET_{i}.npy")[0] for i in ids])
        ent = _save(f"pastis_{split}", lab, ids)
        ent["source"] = f"PASTIS-R folds {list(folds)}, TARGET_<id>.npy channel 0"
        if split == "train":
            ent.update(_subset("pastis_train", len(ids), None))
        out[f"pastis/{split}"] = ent
        print(f"pastis/{split}: {len(ids)} maps", flush=True)
    return out


def build_cloudsen12() -> dict:
    import rasterio
    import tacoreader
    df = tacoreader.load(str(CLOUDSEN12))
    counts = df["tortilla:data_split"].value_counts().to_dict()
    assert counts == {"train": 4000, "validation": 535, "test": 975}, counts
    out = {}
    for split, tsplit in (("train", "train"), ("val", "validation")):
        sub = df[df["tortilla:data_split"] == tsplit].reset_index(drop=True)
        labs, ids = [], []
        for i in range(len(sub)):
            row = sub.read(i)
            with rasterio.open(row.read(1)) as src:
                m = src.read()
            assert m.shape == (1, 512, 512), m.shape
            labs.append(m[0])
            ids.append(sub.iloc[i]["tortilla:id"])
        lab = np.stack(labs)
        assert lab.max() <= 3, lab.max()
        ent = _save(f"cloudsen12_{split}", lab, np.array(ids))
        ent["source"] = f"GEO-Bench-2 CloudSen12 '{tsplit}' masks, {CLOUDSEN12.name} sha256 {CLOUDSEN12_SHA}"
        if split == "train":
            ent.update(_subset("cloudsen12_train", len(ids), SUB["cloudsen12"]))
        out[f"cloudsen12/{split}"] = ent
        print(f"cloudsen12/{split}: {len(ids)} tiles", flush=True)
    out["_cloudsen12_split_counts"] = counts
    return out


def build_lulc() -> dict:
    import pandas as pd
    from eo.data.eval_split import load_row_table, spatial_groups
    from eo.terramesh_tok import io as tio
    from eo.terramesh_tok.preprocess import center_crop

    repo = Path(__file__).resolve().parents[2]
    t = load_row_table(TM_VAL)
    t["group"] = spatial_groups(t.center_lat.values, t.center_lon.values, 5.0)
    ev = set(json.loads((repo / "eo/data/eval_rows_v1.json").read_text())["eval_rows"])
    tr = json.loads((repo / "eo/data/temporal_rows_v1.json").read_text())
    locs = set(tr["dev"]["locations"]) | set(tr["test"]["locations"])
    t["held"] = t.row.isin(ev)
    t["temporal"] = (t.corpus == "ssl4eos12") & t.sample_id.str.split("_").str[0].isin(locs)
    bad = t.groupby("group").held.any() | t.groupby("group").temporal.any()
    el = t[~t.group.map(bad)].copy()
    groups = np.sort(el.group.unique())
    perm = np.random.default_rng(SEED).permutation(groups)
    size = el.groupby("group").size()
    cum = np.cumsum(size.loc[perm].values)
    n_read_groups = int(np.searchsorted(cum, LULC_READ_ROWS) + 1)
    read_groups = set(perm[:n_read_groups].tolist())
    el["split"] = np.where(el.group.isin(read_groups), "val", "train")
    want = dict(zip(el.source_shard + "/" + el.stem, el.row))
    split_of = dict(zip(el.row, el.split))

    got: dict[int, np.ndarray] = {}
    t0 = time.time()
    for k, shard in enumerate(sorted(el.source_shard.unique())):
        for stem, arr in tio.iter_shard("LULC", shard):
            r = want.get(f"{shard}/{stem}")
            if r is not None:
                got[r] = np.asarray(center_crop(arr)).reshape(224, 224).copy()
        if k % 10 == 0:
            print(f"  lulc shard {k}: {len(got)} maps ({time.time() - t0:.0f} s)", flush=True)
    missing = set(want.values()) - set(got)
    assert not missing, f"{len(missing)} eligible rows not found in the tars"

    out = {}
    for split in ("train", "val"):
        rows = np.array(sorted(r for r in got if split_of[r] == split), dtype=np.int64)
        lab = np.stack([got[r] for r in rows])
        ent = _save(f"lulc_{split}", lab, rows)
        ent["source"] = ("TerraMesh val LULC tars, 224 px centre crop; rows outside eval_rows_v1 and outside "
                         "any 5 km group (eval_split.spatial_groups) holding a held-out row or a temporal "
                         f"dev/test location; groups split by seed {SEED}, read set ~{LULC_READ_ROWS} rows")
        ent["groups"] = int(el[el.split == split].group.nunique())
        if split == "train":
            ent.update(_subset("lulc_train", len(rows), SUB["lulc"]))
        out[f"lulc/{split}"] = ent
        print(f"lulc/{split}: {len(rows)} maps in {ent['groups']} groups", flush=True)
    g_tr = set(el[el.split == "train"].group)
    g_va = set(el[el.split == "val"].group)
    assert not (g_tr & g_va)
    out["_lulc_audit"] = {
        "val_rows": int(len(t)), "eligible_rows": int(len(el)), "eligible_groups": int(len(groups)),
        "excluded_groups_heldout_or_temporal": int(bad.sum()),
        "pool_groups": len(g_tr), "read_groups": len(g_va), "groups_in_both": 0,
        "step0_525_maps": _audit_step0(t),
    }
    return out


def _audit_step0(t) -> dict:
    """Where Step 0's 525 LULC maps came from (lulc_val600_stems.txt), against the held-out sets."""
    p = Path("/data/enric/reports/a2_seg/data/lulc_val600_stems.txt")
    stems = [s.strip() for s in p.read_text().splitlines() if s.strip()]
    m = t.set_index(t.source_shard + "/" + t.stem).loc[stems]
    bad = t.groupby("group").held.any() | t.groupby("group").temporal.any()
    return {"n": len(stems), "in_eval_rows_v1": int(m.held.sum()), "temporal_dev_test": int(m.temporal.sum()),
            "in_a_heldout_or_temporal_group": int(m.group.map(bad).sum())}


MONITOR = {"pastis": 100, "cloudsen12": 200, "lulc": 1000}


def build_monitor() -> dict:
    """A monitor set carved from each training pool, for a trained tokenizer's early stopping and
    checkpoint choice, so that the read sets are read once: PASTIS 100 patches and CloudSen12 200 tiles
    (seeded), LULC whole 5 km groups of the pool until 1,000 maps. A tokenizer trains on `fit` (the
    pool less the monitor set) only."""
    from eo.data.eval_split import load_row_table, spatial_groups
    out = {}
    rng = np.random.default_rng(SEED + 1)
    for d in ("pastis", "cloudsen12"):
        n = int(np.load(ROOT / f"{d}_train_ids.npy").shape[0])
        mon = np.sort(rng.permutation(n)[:MONITOR[d]])
        out[d] = mon
    rows = np.load(ROOT / "lulc_train_ids.npy")
    t = load_row_table(TM_VAL)
    grp = spatial_groups(t.center_lat.values, t.center_lon.values, 5.0)
    g = grp[rows]                                         # rows are TerraMesh row indices = table order
    assert (t.row.values[rows] == rows).all()
    order = np.random.default_rng(SEED + 2).permutation(np.unique(g))
    picked, n = [], 0
    for gg in order:
        picked.append(gg)
        n += int((g == gg).sum())
        if n >= MONITOR["lulc"]:
            break
    out["lulc"] = np.sort(np.nonzero(np.isin(g, picked))[0])
    res = {}
    for d, mon in out.items():
        p = ROOT / f"{d}_train_monitor.npy"
        np.save(p, mon.astype(np.int64))
        res[d] = {"monitor": p.name, "monitor_n": int(len(mon)), "monitor_sha256": _sha256(p)}
        if d == "lulc":
            res[d]["monitor_groups"] = len(picked)
            fit_groups = set(np.unique(np.delete(g, mon)).tolist())
            assert not (fit_groups & set(picked))
        print(f"{d}: monitor {len(mon)} maps", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="pastis,cloudsen12,lulc")
    ap.add_argument("--monitor", action="store_true", help="add the monitor subsets to the manifest")
    a = ap.parse_args()
    if a.monitor:
        man = json.loads(MANIFEST.read_text())
        for d, ent in build_monitor().items():
            man["sets"][f"{d}/train"].update(ent)
        MANIFEST.write_text(json.dumps(man, indent=1))
        print(f"manifest -> {MANIFEST}", flush=True)
        return
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"sets": {}}
    sets = dict(old.get("sets", {}))
    extra = dict(old.get("audit", {}))
    for name in a.only.split(","):
        res = {"pastis": build_pastis, "cloudsen12": build_cloudsen12, "lulc": build_lulc}[name]()
        for k, v in res.items():
            (extra if k.startswith("_") else sets)[k.lstrip("_") if k.startswith("_") else k] = v
    man = {"built": time.strftime("%Y-%m-%d %H:%M"), "seed": SEED, "sets": sets, "audit": extra,
           "cloudsen12": {"file": str(CLOUDSEN12), "sha256": CLOUDSEN12_SHA, "hub_revision": CLOUDSEN12_REV,
                          "geobench_v2": "0.9"},
           "never_extracted": ["pastis fold 5", "cloudsen12 test"]}
    MANIFEST.write_text(json.dumps(man, indent=1))
    print(f"manifest -> {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
