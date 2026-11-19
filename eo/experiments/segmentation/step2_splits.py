"""A2 Step 2.2: GEO-Bench-2's splits against Table 2 and against PASTIS-R (a2seg env, CPU).

    python eo/experiments/segmentation/step2_splits.py --out /data/enric/reports/a2_seg/step2/splits.json

PASTIS (GEO-Bench-2 at the revision GeoBenchV2 0.9 pins):
  - counts per split against Table 2; patch ids unique and disjoint across splits;
  - train inside PASTIS-R folds 1-3, val equal to fold 4, test equal to fold 5, by patch id;
  - every train and val patch's semantic mask equal to PASTIS-R's TARGET_<id>.npy channel 0 (the test
    masks are not opened: ids and dates only);
  - every patch's S2 dates equal to PASTIS-R's dates-S2, in order;
  - the S2 series byte-equal to PASTIS-R's DATA_S2 on a fixed sample of train and val patches.
CloudSen12: counts against Table 2; tile ids unique and disjoint; ROI and S2 product overlap between
splits (reported, not refused: GEO-Bench-2 keeps CloudSen12's own split).
Also the file hashes as recorded at download.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.geobench import sources as S  # noqa: E402

N_S2_CHECK = 24


def pastis() -> dict:
    """The pinned tortilla as stored (2,178 patches)."""
    t = S.pinned_pastis_table()
    folds = S.pastis_r_fold_ids()
    out = {"counts": t.split.value_counts().to_dict(), "table2": S.TABLE2["pastis"]}
    ids = {s: set(t.patch_id[t.split == s]) for s in ("train", "val", "test")}
    out["unique"] = bool(t.patch_id.is_unique)
    out["disjoint"] = all(not (ids[a] & ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    f = {k: {i for i, v in folds.items() if v in k} for k in ((1, 2, 3), (4,), (5,))}
    out["train_in_folds_1_3"] = ids["train"] <= f[(1, 2, 3)]
    out["val_eq_fold_4"] = ids["val"] == f[(4,)]
    out["test_eq_fold_5"] = ids["test"] == f[(5,)]
    left = sorted(f[(1, 2, 3)] - ids["train"])
    out["folds_1_3_not_in_train"] = {"n": len(left), "ids": left}
    out["train_by_fold"] = {int(k): int(sum(folds[i] == k for i in ids["train"])) for k in (1, 2, 3)}

    # dates, all patches
    import json as _json
    meta = _json.loads((S.PASTIS_R / "metadata.geojson").read_text())
    rdates = {}
    for ft in meta["features"]:
        d = ft["properties"]["dates-S2"]
        d = _json.loads(d.replace("'", '"')) if isinstance(d, str) else d
        rdates[int(ft["properties"]["ID_PATCH"])] = [int(d[k]) for k in sorted(d, key=int)]
    bad_dates = [int(p) for p, d in zip(t.patch_id, t.dates) if list(map(int, d)) != rdates[int(p)]]
    out["dates_equal"] = {"n_checked": len(t), "n_differ": len(bad_dates), "differ": bad_dates[:20]}

    # masks, train and val
    t0, bad_mask, n = time.time(), [], 0
    for r in t[t.split.isin(["train", "val"])].itertuples():
        m = S._h5(S._pastis_taco().read(int(r.idx)).read(3))[0]
        ref = np.load(S.PASTIS_R / "ANNOTATIONS" / f"TARGET_{r.patch_id}.npy")[0]
        n += 1
        if m.shape != ref.shape or not np.array_equal(m, ref):
            bad_mask.append(int(r.patch_id))
    out["masks_equal"] = {"n_checked": n, "n_differ": len(bad_mask), "differ": bad_mask[:20],
                          "seconds": round(time.time() - t0, 1)}

    # S2 series, a fixed sample of train and val
    rng = np.random.default_rng(0)
    tv = t[t.split.isin(["train", "val"])]
    pick = tv.iloc[np.sort(rng.choice(len(tv), N_S2_CHECK, replace=False))]
    bad_s2 = []
    for r in pick.itertuples():
        a = S._h5(S._pastis_taco().read(int(r.idx)).read(0))
        ref = np.load(S.PASTIS_R / "DATA_S2" / f"S2_{r.patch_id}.npy")
        if a.shape != ref.shape or a.dtype != ref.dtype or not np.array_equal(a, ref):
            bad_s2.append({"patch": int(r.patch_id), "shape": list(a.shape), "ref": list(ref.shape)})
    out["s2_equal"] = {"n_checked": int(len(pick)), "patches": pick.patch_id.astype(int).tolist(),
                       "n_differ": len(bad_s2), "differ": bad_s2}
    out["n_dates"] = {"min": int(t.dates.map(len).min()), "median": float(t.dates.map(len).median()),
                      "max": int(t.dates.map(len).max())}
    out["ok"] = bool(out["unique"] and out["disjoint"] and out["train_in_folds_1_3"] and out["val_eq_fold_4"]
                     and out["test_eq_fold_5"] and not bad_dates and not bad_mask and not bad_s2
                     and out["counts"] == S.PINNED_COUNTS["pastis"])
    return out


def cloudsen12() -> dict:
    t = S.cloudsen12_table()
    out = {"counts": t.split.value_counts().to_dict(), "table2": S.TABLE2["cloudsen12"]}
    out["unique"] = bool(t.id.is_unique)
    by = {s: t[t.split == s] for s in ("train", "val", "test")}
    out["disjoint"] = all(not (set(by[a].id) & set(by[b].id)) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    out["roi_shared"] = {f"{a}-{b}": len(set(by[a].roi_id) & set(by[b].roi_id))
                         for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    out["s2_product_shared"] = {f"{a}-{b}": len(set(by[a].s2_id) & set(by[b].s2_id))
                                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    out["baselines"] = t.s2_id.str.extract(r"_(N\d{4})_")[0].value_counts().to_dict()
    out["dates"] = {"min": int(t.date.min()), "max": int(t.date.max())}
    out["ok"] = bool(out["unique"] and out["disjoint"] and out["counts"] == S.COUNTS["cloudsen12"])
    return out


def pastis_dec() -> dict:
    """GEO-Bench-2 PASTIS at the Hub's head (b4918975, 2025-12-15), read remotely by HTTP range: its
    splits, and every patch it holds that the pinned revision does not (S2 series, semantic mask,
    dates) against PASTIS-R. Positions, never the frame's index labels (they are not 0..n-1)."""
    import io as _io
    import re
    import h5py
    import requests
    import tacoreader
    base = "https://huggingface.co/datasets/aialliance/pastis/resolve/{rev}/geobench_pastis.000{i}.part.tortilla"
    df = tacoreader.load([base.format(rev=S.PASTIS_REV_HEAD, i=i) for i in range(3)])
    pid = df.patch_id.astype(int).values
    split = df["tortilla:data_split"].map(S.SPLIT_NAMES).values
    folds = S.pastis_r_fold_ids()
    nov = set(S.pinned_pastis_table().patch_id)

    def fetch(path):
        m = re.search(r"(\d+)_(\d+),(.+)", path)
        off, size, url = int(m.group(1)), int(m.group(2)), m.group(3).replace("/vsicurl/", "")
        r = requests.get(url, headers={"Range": f"bytes={off}-{off + size - 1}"}, timeout=600)
        r.raise_for_status()
        assert len(r.content) == size
        with h5py.File(_io.BytesIO(r.content), "r") as h:
            return h["data"][()]
    out = {"revision": S.PASTIS_REV_HEAD, "counts": {s: int((split == s).sum()) for s in ("train", "val", "test")}}
    ids = {s: set(pid[split == s]) for s in ("train", "val", "test")}
    out["train_eq_folds_1_3"] = ids["train"] == {i for i, f in folds.items() if f in (1, 2, 3)}
    out["val_eq_fold_4"] = ids["val"] == {i for i, f in folds.items() if f == 4}
    out["test_eq_fold_5"] = ids["test"] == {i for i, f in folds.items() if f == 5}
    out["pinned_revision_inside"] = nov <= set(pid)
    extra = [p for p in np.flatnonzero(split == "train") if pid[p] not in nov]
    bad, t0 = [], time.time()
    for p in extra:
        row = df.read(int(p))
        s2, sem = fetch(row.read(0)), fetch(row.read(3))[0]
        dates = [int(d) for d in df.dates.iloc[int(p)]]
        if not (np.array_equal(s2, np.load(S.PASTIS_R / "DATA_S2" / f"S2_{pid[p]}.npy"))
                and np.array_equal(sem, np.load(S.PASTIS_R / "ANNOTATIONS" / f"TARGET_{pid[p]}.npy")[0])
                and dates == S.pastis_r_dates(int(pid[p]))):
            bad.append(int(pid[p]))
    out["extra_train"] = {"n": len(extra), "ids": sorted(int(pid[p]) for p in extra),
                          "n_differ_from_pastis_r": len(bad), "differ": bad, "seconds": round(time.time() - t0, 1)}
    out["ok"] = bool(out["train_eq_folds_1_3"] and out["val_eq_fold_4"] and out["test_eq_fold_5"]
                     and out["pinned_revision_inside"] and not bad)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dec", action="store_true", help="also read the Hub head's PASTIS remotely (HF_HUB_OFFLINE=0)")
    a = ap.parse_args()
    sums = (S.PASTIS_DIR / "SHA256SUMS").read_text().split()
    res = {"pastis_files": {"revision": S.PASTIS_REV,
                            "sha256": dict(zip(sums[1::2], sums[0::2])),
                            "pinned_by_geobenchv2_0.9": dict(zip(S.PASTIS_FILES, S.PASTIS_SHA256)),
                            "sizes": {f: (S.PASTIS_DIR / f).stat().st_size for f in S.PASTIS_FILES}},
           "cloudsen12_file": {"revision": S.CLOUDSEN12_REV, "sha256": S.CLOUDSEN12_SHA256,
                               "size": S.CLOUDSEN12_FILE.stat().st_size}}
    res["pastis_files"]["match"] = res["pastis_files"]["sha256"] == res["pastis_files"]["pinned_by_geobenchv2_0.9"]
    res["pastis"] = pastis()
    res["cloudsen12"] = cloudsen12()
    if a.dec:
        res["pastis_head_revision"] = pastis_dec()
        d = res["pastis_head_revision"]
        print("PASTIS head", d["counts"], "ok", d["ok"], "| extras", d["extra_train"]["n"], "differ",
              d["extra_train"]["n_differ_from_pastis_r"], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    p, c = res["pastis"], res["cloudsen12"]
    print("PASTIS", p["counts"], "ok", p["ok"], "| train in f1-3", p["train_in_folds_1_3"], "val=f4", p["val_eq_fold_4"],
          "test=f5", p["test_eq_fold_5"], "| masks differ", p["masks_equal"]["n_differ"], "dates differ",
          p["dates_equal"]["n_differ"], "s2 differ", p["s2_equal"]["n_differ"], "| by fold", p["train_by_fold"])
    print("CloudSen12", c["counts"], "ok", c["ok"], "roi shared", c["roi_shared"], "product shared", c["s2_product_shared"])
    print("files match pins", res["pastis_files"]["match"])


if __name__ == "__main__":
    main()
