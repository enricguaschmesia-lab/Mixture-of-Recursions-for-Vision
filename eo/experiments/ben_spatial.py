"""Phase 6 L2 (PHASE6_PLAN §2b, Step 3.6's caveat): how much of B10's BEN score is spatial leakage? `mor` env, CPU.

Every BEN split SMARTIES ships is patch-random, so a test patch's neighbours (same S2 tile, same
acquisition, often the adjacent 1.2 km patch) can sit in ft_train. A model that memorized what
a neighbourhood looks like would score higher there for reasons unrelated to land cover.

Per test_a patch (the S2 patch's own metadata, `v1_stream/meta_S2.parquet`):
  * tile_seen     its MGRS tile (from `tile_source`) holds at least one ft_train patch;
  * product_seen  its S2 product (same tile AND acquisition) holds one;
  * d_nn          great-circle distance between its centre and the nearest ft_train patch's centre
                  (UTM corner + 600 m, through pyproj).
The model's micro-mAP is reported per d_nn bin and per seen/unseen group, with each group's own
class-prior mAP beside it (label composition differs between groups: a confound, made visible).

A LOCATION-ONLY baseline, which uses no pixels at all: score_c = the share of the k nearest
ft_train patches labelled c. It measures how much the labels are predictable from position.
Controls (rule 2):
  * the same baseline with ft_train's labels shuffled across locations must fall to the prior;
  * the model's mAP over RANDOM bins of the same sizes must be flat (no trend from binning).

    python eo/experiments/ben_spatial.py --out /data/enric/reports/ben/analysis_b10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.ben import labels as L  # noqa: E402
from eo.eval import ben as E  # noqa: E402

BEN = Path("/data/enric/data/BigEarthNet")
REP = Path("/data/enric/reports/ben")
MODELS = {"b10_full": "checkpoint-1461", "b10_lora": "checkpoint-1461"}
CONDS = ("S1", "S2", "S1S2")
EDGES_KM = [0, 0.1, 1.25, 1.75, 5, 20, np.inf]   # <0.1 km = the same place (another date); 1.2 km = edge-adjacent; 1.70 km = diagonal
R_EARTH = 6_371_008.8


def lonlat(meta):
    from pyproj import Transformer
    lon, lat = np.empty(len(meta)), np.empty(len(meta))
    x, y = meta.ulx.to_numpy() + 600.0, meta.uly.to_numpy() - 600.0
    for z in meta.utm.unique():
        sel = (meta.utm == z).to_numpy()
        epsg = (32600 if z.endswith("N") else 32700) + int(z[:-1])
        lon[sel], lat[sel] = Transformer.from_crs(epsg, 4326, always_xy=True).transform(x[sel], y[sel])
    return lon, lat


def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def chord_to_m(ch):
    return 2 * R_EARTH * np.arcsin(np.clip(ch / 2, 0, 1))


def mm(y, s):
    return E.average_precision(y.ravel(), s.ravel())


def boot_ci(y, s, n=100, seed=0):
    rng = np.random.default_rng(seed)
    v = [mm(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def multihot(masks):
    return np.stack([np.isin(np.arange(L.N_CLASSES), L.from_mask(int(m))) for m in masks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 10, 50])
    a = ap.parse_args()
    from scipy.spatial import cKDTree
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    idx = pd.read_parquet(BEN / "index_v1.parquet")
    meta = pd.read_parquet(BEN / "v1_stream/meta_S2.parquet").set_index("row")
    meta["tile"] = meta.tile_source.str.extract(r"_T(\d{2}[A-Z]{3})_", expand=False)
    assert meta.tile.notna().all(), "a tile_source without an MGRS tile id"
    tr_rows = idx.row[idx.ft_train].to_numpy()
    te_rows = idx.row[idx.test_a].to_numpy()
    mt, me = meta.loc[tr_rows], meta.loc[te_rows]
    y_tr = multihot(idx.labels19.to_numpy()[tr_rows])
    y_te = multihot(idx.labels19.to_numpy()[te_rows])
    freq = y_tr.mean(0)

    lt, le = lonlat(mt), lonlat(me)
    tree = cKDTree(unit(*lt))
    kmax = max(a.k)
    ch, nn = tree.query(unit(*le), k=kmax)
    d_nn = chord_to_m(ch[:, 0]) / 1000.0
    tiles_tr, prods_tr = set(mt.tile), set(mt.tile_source)
    tile_seen = me.tile.isin(tiles_tr).to_numpy()
    prod_seen = me.tile_source.isin(prods_tr).to_numpy()
    twin = d_nn < 0.1
    same_lab = (y_tr[nn[:, 0]] == y_te).all(1)
    out = {"n_train": int(len(tr_rows)), "n_test": int(len(te_rows)),
           "n_tiles_train": len(tiles_tr), "n_tiles_test": int(me.tile.nunique()),
           "n_products_train": len(prods_tr), "n_products_test": int(me.tile_source.nunique()),
           "twins": {"share_of_test": float(twin.mean()),
                     "labels_identical_to_twin": float(same_lab[twin].mean()) if twin.any() else None,
                     "labels_identical_to_nearest_if_not_twin": float(same_lab[~twin].mean()),
                     "twin_date_differs": float((mt.date.to_numpy()[nn[twin, 0]] != me.date.to_numpy()[twin]).mean()) if twin.any() else None},
           "test_tile_seen": float(tile_seen.mean()), "test_product_seen": float(prod_seen.mean()),
           "d_nn_km_quantiles": {q: float(np.quantile(d_nn, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
           "prior_micro_map": mm(y_te, np.broadcast_to(freq, y_te.shape))}

    # location-only baseline, and its control (labels shuffled across training locations)
    y_shuf = y_tr[rng.permutation(len(y_tr))]
    out["location_knn"] = {}
    for k in a.k:
        sc = y_tr[nn[:, :k]].mean(1)
        sc_ctl = y_shuf[nn[:, :k]].mean(1)
        out["location_knn"][k] = {"micro_map": mm(y_te, sc), "control_shuffled_labels": mm(y_te, sc_ctl)}
        if k == 10:
            knn10 = sc

    # the model, per group
    bins = np.digitize(d_nn, EDGES_KM[1:-1])
    groups = {f"d_nn {EDGES_KM[b]}–{EDGES_KM[b + 1]} km": bins == b for b in range(len(EDGES_KM) - 1)}
    groups.update({"tile seen": tile_seen, "tile unseen": ~tile_seen,
                   "product seen": prod_seen, "product unseen": ~prod_seen})
    rand_bins = rng.permutation(bins)
    out["models"] = {}
    for tag, ck in MODELS.items():
        for cond in CONDS:
            z = np.load(REP / tag / ck / "test_a" / f"{cond}.npz")
            order = pd.Series(np.arange(len(z["rows"])), index=z["rows"]).loc[te_rows].to_numpy()
            s, yz = z["marginal"][order].astype(np.float64), z["truth"][order].astype(bool)
            assert (yz == y_te).all(), "eval_ben's truth disagrees with the index labels"
            res = {"all": mm(y_te, s)}
            for name, sel in groups.items():
                if sel.sum() < 50:
                    res[name] = {"n": int(sel.sum())}
                    continue
                r = {"n": int(sel.sum()), "micro_map": mm(y_te[sel], s[sel]),
                     "prior_micro_map": mm(y_te[sel], np.broadcast_to(freq, (sel.sum(), L.N_CLASSES))),
                     "knn10_micro_map": mm(y_te[sel], knn10[sel]), "mean_labels": float(y_te[sel].sum(1).mean())}
                if tag == "b10_lora" and cond == "S1S2":
                    r["ci95"] = boot_ci(y_te[sel], s[sel])
                res[name] = r
            res["random_bins_control"] = [mm(y_te[rand_bins == b], s[rand_bins == b])
                                          for b in range(len(EDGES_KM) - 1) if (rand_bins == b).sum() >= 50]
            out["models"][f"{tag}/{cond}"] = res
    (a.out / "l2.json").write_text(json.dumps(out, indent=1, default=float) + "\n")

    md = ["# L2: spatial leakage on test_a (B10, selected checkpoints)", "",
          f"- ft_train {out['n_train']} patches on {out['n_tiles_train']} MGRS tiles; test_a {out['n_test']} on {out['n_tiles_test']}.",
          f"- S2 products (tile + date): ft_train {out['n_products_train']}, test_a {out['n_products_test']}.",
          f"- test_a patches whose tile has a training patch: {out['test_tile_seen']:.4f}; whose S2 product (tile + date) has one: {out['test_product_seen']:.4f}.",
          f"- **same-place twins** (a training patch within 0.1 km, i.e. the same patch at another date): {out['twins']['share_of_test']:.4f} of test_a; "
          f"labels identical to the twin's: {out['twins']['labels_identical_to_twin']:.4f} (nearest patch, non-twins: {out['twins']['labels_identical_to_nearest_if_not_twin']:.4f}); "
          f"twin at another date: {out['twins']['twin_date_differs']:.4f}.",
          "- distance to the nearest ft_train patch (km), quantiles 5/25/50/75/95/99%: "
          + " / ".join(f"{v:.2f}" for v in out["d_nn_km_quantiles"].values()), "",
          "**Location-only baseline** (no pixels: the share of the k nearest training patches with each label):", "",
          "| k | micro-mAP | control: labels shuffled across locations |", "|---|---|---|"]
    md += [f"| {k} | {v['micro_map']:.4f} | {v['control_shuffled_labels']:.4f} |" for k, v in out["location_knn"].items()]
    md += [f"\nClass prior: {out['prior_micro_map']:.4f}.", "",
           "**b10_lora, S1+S2 → BEN, by group** (each group's own class-prior and location-kNN mAP beside it):", "",
           "| group | n | mean labels | model micro-mAP [95% CI] | prior | location k=10 | model − prior |", "|---|---|---|---|---|---|---|"]
    r = out["models"]["b10_lora/S1S2"]
    for name in groups:
        g = r[name]
        if "micro_map" not in g:
            md.append(f"| {name} | {g['n']} | | | | | |")
            continue
        md.append(f"| {name} | {g['n']} | {g['mean_labels']:.2f} | {g['micro_map']:.4f} [{g['ci95'][0]:.4f}, {g['ci95'][1]:.4f}] | "
                  f"{g['prior_micro_map']:.4f} | {g['knn10_micro_map']:.4f} | {g['micro_map'] - g['prior_micro_map']:+.4f} |")
    md += ["", "**Every model and condition, micro-mAP per distance bin** (control: random bins of the same sizes):", "",
           "| model / cond | all | " + " | ".join(n for n in groups if n.startswith("d_nn")) + " | random bins (control) |",
           "|" + "---|" * (3 + sum(n.startswith("d_nn") for n in groups))]
    for key, r in out["models"].items():
        md.append(f"| {key} | {r['all']:.4f} | " + " | ".join(
            f"{r[n]['micro_map']:.4f}" if "micro_map" in r[n] else f"(n={r[n]['n']})" for n in groups if n.startswith("d_nn"))
            + " | " + " ".join(f"{v:.4f}" for v in r["random_bins_control"]) + " |")
    (a.out / "l2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
