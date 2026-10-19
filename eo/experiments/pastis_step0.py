"""Phase 6 Part 2, Step 0 (PHASE6_PART2_PLAN 0.2-0.4): inspect PASTIS-R before any preprocessing.

CPU only; numpy + pandas + matplotlib, so it runs in `.venv` (no tokenizer). Reads the extracted
Zenodo archive (record 5735646) and writes, under --out:
  * patches.parquet  one row per patch: fold, tile, S2/S1 lengths and dates, the protocol-image
                     candidates (the chronologically last acquisition; the clearest Jul-Aug one) with
                     their cloud and no-data fractions, pixel counts per class, the centroid;
  * summary.json     everything the worklog quotes, computed here, never retyped;
  * cloud_sheet.png  the cloud rule checked by eye: last images sorted by score, and the clearest
                     Jul-Aug images.

The cloud rule is deliberately simple: a pixel is "bright" when B02, B03 and B04 all exceed 1,500
(reflectance x 1e4). Clear land rarely does in the blue band; clouds and snow do. It is checked by
eye on the contact sheet, and against a broken control that must fail: the same threshold on B08
(NIR), which is bright over clear vegetation, must flag clear summer images.
The bright rule misses HAZE (found on the first full run: 36 t31tfj patches on 2019-08-13 had a blue
floor of ~950 and nothing "bright"). Haze is read from the image's blue floor, the 1st percentile of
B02 (dark-object idea): clear summer floors are ~150, hazy ones ~950, and SWIR (B11) is unchanged,
unlike a DN offset, which raises every band. The second full run showed CLOUD SHADOW too: THEIA's
correction drives shadowed blue reflectance NEGATIVE (a last image at floor -540 is almost black), so
"lowest floor" picked shadowed dates. Three indicators per acquisition, all read by eye on the sheet:
  bright  (thick cloud): share of pixels with B02, B03, B04 > 1,500;
  floor   (haze):        B02 1st percentile; > 400 is hazy (clear Jul-Aug: median ~60, p90 ~230);
  negblue (shadow):      share of pixels with B02 < 0.
"Clear" = bright < 1%, negblue < 1%, floor <= 400. The Jul-Aug candidate is the non-hazy date with
the least bright + no-data + negblue (the least bad one if every date is hazy).

    python eo/experiments/pastis_step0.py --root /data/enric/data/PASTIS/PASTIS-R \
        --out /data/enric/reports/pastis/step0
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.terramesh_tok import contract as C  # noqa: E402

# The archive's S2 band order (PANGAEA's pastis.yaml; checked here against the pretraining means).
PASTIS_S2 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
B = {b: i for i, b in enumerate(PASTIS_S2)}
CLASSES = ["background", "meadow", "soft winter wheat", "corn", "winter barley", "winter rapeseed",
           "spring barley", "sunflower", "grapevine", "beet", "winter triticale", "winter durum wheat",
           "fruits, vegetables, flowers", "potatoes", "leguminous fodder", "soybeans", "orchard",
           "mixed cereal", "sorghum", "void"]
N_CLS = len(CLASSES)
BRIGHT = 1500            # cloud rule threshold, reflectance x 1e4
SUMMER = (20190701, 20190831)
PANGAEA_MAX_T = 50       # pangaea/datasets/pastis.py shuffles longer series before taking [-1]
HAZE_FLOOR = 400         # B02 1st percentile above this: hazy (clear Jul-Aug p90 ~230, the hazy date ~950)
CLEAR_MAX = 0.01         # "clear": bright < 1% and negblue < 1% and floor <= HAZE_FLOOR


def _dates(v):
    d = v if isinstance(v, dict) else ast.literal_eval(v)
    return [int(d[k]) for k in sorted(d, key=int)]


def _centroid(geom):
    """Mean of the outer ring's vertices; the archive mixes Polygon (1,346) and MultiPolygon (1,087)."""
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    ring = np.concatenate([np.asarray(poly[0], dtype=np.float64)[:-1] for poly in polys])
    return float(ring[:, 0].mean()), float(ring[:, 1].mean())


def _per_date(s2):
    """(T,) no-data, negative, bright (cloud rule), NIR-bright (broken control) fractions, and the
    blue floor (B02 1st percentile over pixels with data)."""
    nodata_px = (s2 == 0).all(axis=1)
    nodata = nodata_px.mean(axis=(1, 2))
    neg = (s2 < 0).any(axis=1).mean(axis=(1, 2))
    vis = s2[:, [B["B02"], B["B03"], B["B04"]]]
    bright = (vis > BRIGHT).all(axis=1).mean(axis=(1, 2))
    nir = (s2[:, B["B08"]] > BRIGHT).mean(axis=(1, 2))
    floor = np.array([np.percentile(s2[t, B["B02"]][~nodata_px[t]], 1) if (~nodata_px[t]).sum() > 50 else np.nan
                      for t in range(s2.shape[0])])
    negblue = ((s2[:, B["B02"]] < 0) & ~nodata_px).mean(axis=(1, 2))
    return nodata, neg, bright, nir, floor, negblue


def inspect_patch(args):
    root, props = args
    pid = int(props["ID_PATCH"])
    out = {"id": pid, "fold": int(props["Fold"]), "tile": props["TILE"],
           "n_parcel": int(props["N_Parcel"]), "parcel_cover": float(props["Parcel_Cover"])}
    d2 = _dates(props["dates-S2"])
    s2 = np.load(root / "DATA_S2" / f"S2_{pid}.npy")                       # allow_pickle=False
    out.update(s2_T=int(s2.shape[0]), s2_shape_ok=s2.shape[1:] == (10, 128, 128),
               s2_dtype=str(s2.dtype), s2_dates_n=len(d2), s2_sorted=d2 == sorted(d2),
               first_date=d2[0], last_date=d2[-1])
    nodata, neg, bright, nir, floor, negblue = _per_date(s2)
    last = int(np.argmax(d2))                                               # chronologically last
    out.update(last_bright=float(bright[last]), last_nodata=float(nodata[last]),
               last_neg=float(neg[last]), last_nir_bright=float(nir[last]), last_floor=float(floor[last]), last_negblue=float(negblue[last]),
               pangaea_random_last=bool(len(d2) > PANGAEA_MAX_T),
               n_bright_lt5=int((bright < 0.05).sum()),
               n_clear=int(((bright < CLEAR_MAX) & (negblue < CLEAR_MAX) & (np.nan_to_num(floor, nan=1e9) <= HAZE_FLOOR)).sum()))
    summer = [i for i, d in enumerate(d2) if SUMMER[0] <= d <= SUMMER[1]]
    out["n_summer"] = len(summer)
    if summer:
        bad = bright + nodata + negblue
        fl = np.where(np.isfinite(floor), floor, np.inf)
        nonhazy = [i for i in summer if fl[i] <= HAZE_FLOOR]
        j = (min(nonhazy, key=lambda i: bad[i]) if nonhazy
             else min(summer, key=lambda i: bad[i] + (fl[i] - HAZE_FLOOR) / 1000.0))
        k = summer[int(np.argmin((bright + nodata)[summer]))]   # the bright-rule-only pick, for the record
        clear = [i for i in summer if bright[i] < CLEAR_MAX and negblue[i] < CLEAR_MAX and fl[i] <= HAZE_FLOOR]
        out.update(summer_date=d2[j], summer_bright=float(bright[j]), summer_nodata=float(nodata[j]),
                   summer_nir_bright=float(nir[j]), summer_floor=float(floor[j]), summer_negblue=float(negblue[j]),
                   summer_n_ok=len(nonhazy), summer_n_clear=len(clear),
                   brightonly_date=d2[k], brightonly_floor=float(floor[k]), brightonly_negblue=float(negblue[k]))
        img = s2[j].astype(np.float32)
        ok = (img != 0).any(axis=0) & ~(img[[B["B02"], B["B03"], B["B04"]]] > BRIGHT).all(axis=0)
        ndvi = (img[B["B08"]] - img[B["B04"]]) / (img[B["B08"]] + img[B["B04"]] + 1e-6)
        veg = ok & (ndvi > 0.6)
        out["summer_veg_px"] = int(veg.sum())
        out["summer_veg_median"] = (np.median(img[:, veg], axis=1).tolist() if veg.sum() >= 50 else None)
        out["summer_p1"] = np.percentile(img[:, ok], 1, axis=1).tolist() if ok.sum() >= 50 else None
    else:
        out.update(summer_date=None, summer_bright=None, summer_nodata=None, summer_nir_bright=None,
                   summer_floor=None, summer_negblue=None, summer_n_ok=0, summer_n_clear=0,
                   brightonly_date=None, brightonly_floor=None, brightonly_negblue=None,
                   summer_veg_px=0, summer_veg_median=None, summer_p1=None)
    t = np.load(root / "ANNOTATIONS" / f"TARGET_{pid}.npy")
    sem = t[0]
    out.update(target_shape=list(t.shape), target_dtype=str(t.dtype),
               sem_in_range=bool(sem.max() < N_CLS))
    out["counts"] = np.bincount(sem.ravel(), minlength=N_CLS)[:N_CLS].tolist()
    for orb in ("S1A", "S1D"):
        a = np.load(root / f"DATA_{orb}" / f"{orb}_{pid}.npy")
        da = _dates(props[f"dates-{orb}"])
        a32 = a.astype(np.float32)
        fin = np.isfinite(a32)
        diff = np.abs(a32[:, 2] - (a32[:, 0] - a32[:, 1]))[fin.all(axis=1)]
        out.update({f"{orb}_T": int(a.shape[0]), f"{orb}_dates_n": len(da), f"{orb}_dtype": str(a.dtype),
                    f"{orb}_nan": float(1 - fin.mean()),
                    f"{orb}_median": np.nanmedian(a32, axis=(0, 2, 3)).tolist(),
                    f"{orb}_ratio_maxdiff": float(diff.max()) if diff.size else None})
    return out


def _rgb(s2_img):
    rgb = s2_img[[B["B04"], B["B03"], B["B02"]]].astype(np.float32).transpose(1, 2, 0) / 3000.0
    return np.clip(rgb, 0, 1)


def contact_sheet(root, df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = df.sort_values("last_bright")
    n = len(s)
    picks = [("last image, lowest score", s.iloc[:8]),
             ("last image, middle", s.iloc[n // 2 - 4: n // 2 + 4]),
             ("last image, score 5-25%", s[(s.last_bright > 0.05) & (s.last_bright < 0.25)].head(8)),
             ("last image, highest score", s.iloc[-8:]),
             ("Jul-Aug pick, random", df.sample(min(8, len(df)), random_state=0)),
             ("Jul-Aug pick, highest floor", df.sort_values("summer_floor").tail(8)),
             ("last image, hazy, not bright", s[(s.last_floor > HAZE_FLOOR) & (s.last_bright < 0.01)].head(8)),
             ("last image, CLEAR by all 3", s[(s.last_bright < CLEAR_MAX) & (s.last_negblue < CLEAR_MAX) & (s.last_floor <= HAZE_FLOOR)].sample(8, random_state=1) if len(s) > 8 else s.head(0)),
             ("Jul-Aug pick, highest negblue", df.sort_values("summer_negblue").tail(8))]
    fig, axes = plt.subplots(len(picks), 8, figsize=(16, 2.25 * len(picks)))
    for r, (title, rows) in enumerate(picks):
        for c in range(8):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(rows):
                continue
            row = rows.iloc[c]
            s2 = np.load(root / "DATA_S2" / f"S2_{int(row.id)}.npy", mmap_mode="r")
            d2 = list(range(s2.shape[0]))
            if r < 4 or r in (6, 7):
                j, score, date = d2[-1], row.last_bright, row.last_date
            else:
                if row.summer_date is None or pd.isna(row.summer_date):
                    continue
                dates = sorted(_dates(META[int(row.id)]["dates-S2"]))
                j, score, date = dates.index(int(row.summer_date)), row.summer_bright, int(row.summer_date)
            ax.imshow(_rgb(np.asarray(s2[j])))
            last = r < 4 or r in (6, 7)
            fl = row.last_floor if last else row.summer_floor
            nb = row.last_negblue if last else row.summer_negblue
            ax.set_title(f"{int(row.id)} {date}\nbr {score:.2f} fl {fl:.0f} sh {nb:.2f}", fontsize=8)
        axes[r, 0].text(-0.15, 0.5, title, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


META = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/enric/data/PASTIS/PASTIS-R"))
    ap.add_argument("--out", type=Path, default=Path("/data/enric/reports/pastis/step0"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    g = json.load(open(a.root / "metadata.geojson"))
    feats = g["features"]
    for f in feats:
        META[int(f["properties"]["ID_PATCH"])] = f["properties"]
    if a.limit:
        feats = feats[: a.limit]
    raw = a.out / "patches_raw.parquet"          # the expensive pass, kept so a later failure costs nothing
    if raw.exists() and not a.limit:
        df = pd.read_parquet(raw)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(inspect_patch, [(a.root, f["properties"]) for f in feats], chunksize=8))
        df = pd.DataFrame(rows)
        if not a.limit:
            df.to_parquet(raw)
    cx, cy = zip(*[_centroid(f["geometry"]) for f in feats])
    df["cx"], df["cy"] = cx, cy

    # --- files against metadata
    ids = set(df.id)
    files = {m: {int(p.stem.split("_")[1]) for p in (a.root / f"DATA_{m}").glob(f"{m}_*.npy")}
             for m in ("S2", "S1A", "S1D")}
    targets = {int(p.stem.split("_")[1]) for p in (a.root / "ANNOTATIONS").glob("TARGET_*.npy")}
    # --- distances (EPSG:2154, metres) to the nearest training patch (folds 1-3)
    xy = df[["cx", "cy"]].to_numpy()
    train = df.fold.isin([1, 2, 3]).to_numpy()
    dmat = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dmat, np.inf)
    df["d_train_km"] = dmat[:, train].min(axis=1) / 1000.0
    other = df.fold.to_numpy()[:, None] != df.fold.to_numpy()[None, :]
    df["d_otherfold_km"] = np.where(other, dmat, np.inf).min(axis=1) / 1000.0
    df["d_any_km"] = dmat.min(axis=1) / 1000.0

    counts = np.stack(df["counts"].to_numpy())
    shares = {}
    for name, sel in [("all", np.ones(len(df), bool))] + [(f"fold{k}", (df.fold == k).to_numpy()) for k in range(1, 6)]:
        c = counts[sel].sum(0)
        shares[name] = {CLASSES[i]: round(float(c[i] / c.sum()), 5) for i in range(N_CLS)}
    nonvoid = counts[:, :19].sum(0)
    pangaea_like = {CLASSES[i]: round(float(counts[:, i].sum() / counts[:, 1:].sum()), 5) for i in range(1, N_CLS)}

    veg = np.array([v for v in df.summer_veg_median if v is not None])
    s2norm = json.load(open(a.root / "NORM_S2_patch.json"))
    tm = dict(zip(C.BAND_ORDER["S2L2A"], C.V1_TOK_MEAN["S2L2A"]))
    p1 = np.array([v for v in df.summer_p1 if v is not None])
    month = (df.last_date // 100).value_counts().sort_index()
    sm = df.dropna(subset=["summer_bright"])
    summary = {
        "patches": len(df), "folds": df.fold.value_counts().sort_index().to_dict(),
        "tiles": df.tile.value_counts().to_dict(),
        "files_not_in_metadata": {m: sorted(files[m] - ids)[:50] for m in files},
        "n_files_not_in_metadata": {m: len(files[m] - ids) for m in files},
        "metadata_without_file": {m: len(ids - files[m]) for m in files} | {"TARGET": len(ids - targets)},
        "s2": {"dtype": sorted(df.s2_dtype.unique().tolist()), "shape_ok": bool(df.s2_shape_ok.all()),
               "T_equals_dates": bool((df.s2_T == df.s2_dates_n).all()), "dates_sorted": bool(df.s2_sorted.all()),
               "T_min_max": [int(df.s2_T.min()), int(df.s2_T.max())],
               "first_date_range": [int(df.first_date.min()), int(df.first_date.max())],
               "last_date_range": [int(df.last_date.min()), int(df.last_date.max())],
               "last_date_by_month": {int(k): int(v) for k, v in month.items()},
               "pangaea_random_last (T > 50)": int(df.pangaea_random_last.sum()),
               "norm_fold1_mean_vs_pretraining_mean": {b: [round(s2norm["Fold_1"]["mean"][i], 1), round(tm[b], 1)]
                                                        for i, b in enumerate(PASTIS_S2)},
               "summer_vegetation_median_by_band": dict(zip(PASTIS_S2, np.median(veg, 0).round(0).tolist())) if len(veg) else None,
               "summer_p1_by_band_median": dict(zip(PASTIS_S2, np.median(p1, 0).round(0).tolist())) if len(p1) else None,
               # A +1000 DN offset (TerraMesh's baseline-05.00 problem) raises EVERY acquisition's blue floor to
               # ~1000+. A B12 floor test misfires here: dry Aug fields have SWIR floors of 1,300-1,900 with
               # normal blue floors (16 patches on the third run), so the test is the blue floor of the pick.
               "offset_test: patches whose Jul-Aug pick has a blue floor <= 400 (an offset is then impossible)":
                   int((df.summer_floor <= HAZE_FLOOR).sum()),
               "offset_test: the others (id, floor)": [[int(i), round(float(f), 1)] for i, f in
                                                       df.loc[df.summer_floor > HAZE_FLOOR, ["id", "summer_floor"]].to_numpy()]},
        "protocol_image": {
            "last": {"bright_mean": round(float(df.last_bright.mean()), 4),
                     "share_bright_gt_0.05": round(float((df.last_bright > 0.05).mean()), 4),
                     "share_bright_gt_0.25": round(float((df.last_bright > 0.25).mean()), 4),
                     "share_bright_gt_0.50": round(float((df.last_bright > 0.50).mean()), 4),
                     "share_nodata_gt_0.05": round(float((df.last_nodata > 0.05).mean()), 4),
                     "floor_quantiles_10_50_90": np.nanpercentile(df.last_floor, [10, 50, 90]).round(0).tolist(),
                     "share_hazy (floor > 400)": round(float((df.last_floor > HAZE_FLOOR).mean()), 4),
                     "share_shadow (negblue >= 0.01)": round(float((df.last_negblue >= CLEAR_MAX).mean()), 4),
                     "share_CLEAR (all three)": round(float(((df.last_bright < CLEAR_MAX) & (df.last_negblue < CLEAR_MAX) & (df.last_floor <= HAZE_FLOOR)).mean()), 4),
                     "CLEAR_by_fold": {int(k): round(float(v), 4) for k, v in ((df.last_bright < CLEAR_MAX) & (df.last_negblue < CLEAR_MAX) & (df.last_floor <= HAZE_FLOOR)).groupby(df.fold).mean().items()},
                     "CLEAR_by_tile_and_date": {f"{t} {d}": round(float(v), 3) for (t, d), v in ((df.last_bright < CLEAR_MAX) & (df.last_negblue < CLEAR_MAX) & (df.last_floor <= HAZE_FLOOR)).groupby([df.tile, df.last_date]).mean().items() if ((df.tile == t) & (df.last_date == d)).sum() >= 20}},
            "summer_clearest": {"patches_with_summer_dates": int(len(sm)),
                                "n_summer_dates_min_median_max": [int(df.n_summer.min()), int(df.n_summer.median()), int(df.n_summer.max())],
                                "bright_mean": round(float(sm.summer_bright.mean()), 4),
                                "share_bright_gt_0.05": round(float((sm.summer_bright > 0.05).mean()), 4),
                                "share_bright_gt_0.25": round(float((sm.summer_bright > 0.25).mean()), 4),
                                "share_nodata_gt_0.05": round(float((sm.summer_nodata > 0.05).mean()), 4),
                                "floor_quantiles_10_50_90": np.nanpercentile(sm.summer_floor, [10, 50, 90]).round(0).tolist(),
                                "share_hazy (floor > 400)": round(float((sm.summer_floor > HAZE_FLOOR).mean()), 4),
                                "share_shadow (negblue >= 0.01)": round(float((sm.summer_negblue >= CLEAR_MAX).mean()), 4),
                                "share_CLEAR (all three)": round(float(((sm.summer_bright < CLEAR_MAX) & (sm.summer_negblue < CLEAR_MAX) & (sm.summer_floor <= HAZE_FLOOR)).mean()), 4),
                                "patches_without_a_nonhazy_date": int((sm.summer_n_ok == 0).sum()),
                                "patches_without_a_clear_date": int((sm.summer_n_clear == 0).sum()),
                                "n_clear_dates_jul_aug_min_median_max": [int(sm.summer_n_clear.min()), int(sm.summer_n_clear.median()), int(sm.summer_n_clear.max())],
                                "bright_rule_only_pick: hazy / shadow": [int((sm.brightonly_floor > HAZE_FLOOR).sum()), int((sm.brightonly_negblue >= CLEAR_MAX).sum())],
                                "date_by_month": {int(k): int(v) for k, v in (sm.summer_date // 100).value_counts().sort_index().items()}},
            "whole_series_clear_dates_per_patch_min_median_max": [int(df.n_clear.min()), int(df.n_clear.median()), int(df.n_clear.max())],
            "broken_control_nir_rule_on_clearest_summer": {"mean_flagged": round(float(sm.summer_nir_bright.mean()), 4),
                                                          "share_gt_0.25": round(float((sm.summer_nir_bright > 0.25).mean()), 4)}},
        "labels": {"target_shapes": sorted({tuple(s) for s in df.target_shape}), "dtypes": sorted(df.target_dtype.unique().tolist()),
                   "semantic_in_range": bool(df.sem_in_range.all()), "pixel_shares": shares,
                   "pangaea_style_shares_excluding_background": pangaea_like,
                   "patches_mostly_background (>0.9)": int((counts[:, 0] / counts.sum(1) > 0.9).sum()),
                   "patches_void_gt_0.5": int((counts[:, 19] / counts.sum(1) > 0.5).sum()),
                   "classes_absent_from_test_fold": [CLASSES[i] for i in range(19) if counts[(df.fold == 5).to_numpy(), i].sum() == 0]},
        "s1": {o: {"dtype": sorted(df[f"{o}_dtype"].unique().tolist()), "T_equals_dates": bool((df[f"{o}_T"] == df[f"{o}_dates_n"]).all()),
                   "T_min_max": [int(df[f"{o}_T"].min()), int(df[f"{o}_T"].max())],
                   "nan_mean": round(float(df[f"{o}_nan"].mean()), 5),
                   "median_by_channel": np.median(np.stack(df[f"{o}_median"].to_numpy()), 0).round(2).tolist(),
                   "ch2_minus_(ch0-ch1)_max": round(float(np.nanmax(df[f"{o}_ratio_maxdiff"].astype(float))), 3)}
               for o in ("S1A", "S1D")} | {"pretraining_means": {"S1GRD": C.V1_TOK_MEAN["S1GRD"], "S1RTC": C.V1_TOK_MEAN["S1RTC"]}},
        "distances_km": {f"fold{k}": {"to_nearest_train_median": round(float(df[df.fold == k].d_train_km.median()), 3),
                                     "to_nearest_train_min": round(float(df[df.fold == k].d_train_km.min()), 3),
                                     "share_within_1.5km_of_train": round(float((df[df.fold == k].d_train_km < 1.5).mean()), 4)}
                         for k in (4, 5)} | {"nearest_any_patch_median": round(float(df.d_any_km.median()), 3),
                                             "nearest_other_fold_min": round(float(df.d_otherfold_km.min()), 3),
                                             "share_nearest_other_fold_lt_1.5km": round(float((df.d_otherfold_km < 1.5).mean()), 4)},
    }
    df.to_parquet(a.out / "patches.parquet")
    json.dump(summary, open(a.out / "summary.json", "w"), indent=1, default=str)
    contact_sheet(a.root, df, a.out / "cloud_sheet.png")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
