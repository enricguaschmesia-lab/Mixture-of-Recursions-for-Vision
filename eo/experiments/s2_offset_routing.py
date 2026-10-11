"""Does A10 route or score S2 tokens differently in offset vs clean scenes? (`mor` env, CPU)

Follow-up to the 2026-10-11 S2 BOA-offset finding (s2_offset_scan.py, s2_offset_tokens.py).
Reads A10's held-out teacher-forced pass (eval_checkpoint.py --keep-tokens output,
tokens_router.npz: per-token depth and CE for 4,416 rows), labels each row's S2 regime from the
raw raster (floor = darkest band's 1st percentile, offset iff > 900 DN, as in the scan), and
compares per-row mean depth and mean CE by regime, per modality.

Control for scene content: S1RTC/DEM/LULC tokens are untouched by the offset, so a regime
difference that shows up in them too is geography/season, not the offset. The comparison is
also repeated inside Major TOM 2019Q3-2021Q4, where both regimes are common.
"""
import sys, tarfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terramesh_tok import io as tio, contract as C  # noqa: E402

EVAL = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/enric/reports/arm_eval/a10/checkpoint-3300")
VAL = Path("/data/enric/data/TerraMesh/val")
OUT = Path("/data/enric/reports/s2_offset")
MOD_ID = {"S2L2A": 1, "S1GRD": 2, "S1RTC": 3, "DEM": 4, "NDVI": 5, "LULC": 6, "Coords": 7}   # teacher_forced.MODALITY_TO_ID

rows = np.load(EVAL / "eval_rows.npy")
cache = OUT / "eval_rows_floor.parquet"
if cache.exists():
    fl = pd.read_parquet(cache)
else:
    idx = pd.read_parquet(VAL / "tok_index.parquet").set_index("row").loc[rows]
    rec = {}
    for shard, grp in idx.groupby("source_shard"):
        want = {f"{s}.zarr.zip": int(r) for r, s in zip(grp.index, grp.stem)}
        with tarfile.open(VAL / "S2L2A" / shard) as tf:
            for m in tf:
                if m.name in want:
                    b = tio._decode_zarr_zip(tf.extractfile(m).read(), "bands")[0]
                    b = b[:, C.CROP_OFF:C.CROP_OFF + C.CROP, C.CROP_OFF:C.CROP_OFF + C.CROP].astype(np.float32)
                    rec[want[m.name]] = float(np.percentile(b.reshape(12, -1), 1, axis=1).min())
    assert len(rec) == len(rows), (len(rec), len(rows))
    fl = pd.DataFrame({"row": list(rec), "floor": list(rec.values())})
    fl.to_parquet(cache)
fl = fl.set_index("row").loc[rows]

z = np.load(EVAL / "tokens_router.npz")
depth, imod, lmod, ce = z["depth"].astype(np.float32), z["input_mod"], z["label_mod"], z["ce"].astype(np.float32)

# control: the S2 label ids of row i must be that row's stored S2 tokens (row alignment)
s2tok = np.load(VAL / C.tok_dir_name("S2L2A") / "tokens.npy", mmap_mode="r")
ok = 0
for i in range(0, len(rows), 97):
    lab = z["label"][i][lmod[i] == MOD_ID["S2L2A"]]
    lab = lab[lab >= 0]
    ok += int(np.array_equal(np.sort(lab - lab.min()), np.sort(s2tok[rows[i]].astype(np.int64) - s2tok[rows[i]].min()))
              and len(lab) == 196)
print(f"row alignment control: {ok}/{len(range(0, len(rows), 97))} sampled rows match stored S2 tokens", flush=True)

meta = pd.read_parquet(VAL.parent / "val_metadata.parquet")[["tar", "zarr", "S2_time"]]
tix = pd.read_parquet(VAL / "tok_index.parquet").set_index("row").loc[rows]
meta = meta.set_index(["tar", "zarr"]).loc[list(zip(tix.source_shard, tix.stem + ".zarr.zip"))]
d = pd.DataFrame({"row": rows, "floor": fl.floor.values, "corpus": tix.corpus.values,
                  "t": pd.to_datetime(meta.S2_time.values)})
d["regime"] = np.where(d.floor > 900, "offset", "clean")
for m, k in MOD_ID.items():
    sel = imod == k
    n = sel.sum(1)
    d[f"depth_{m}"] = np.where(n > 0, (depth * sel).sum(1) / np.maximum(n, 1), np.nan)
    sl = lmod == k
    nl = sl.sum(1)
    d[f"ce_{m}"] = np.where(nl > 0, (ce * sl).sum(1) / np.maximum(nl, 1), np.nan)
d.to_parquet(OUT / f"eval_rows_regime_{EVAL.parent.name}_{EVAL.name}.parquet")


def boot_diff(a, b, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(a), len(b)


win = (d.corpus == "majortom") & (d.t >= "2019-07-01") & (d.t < "2022-01-01")
print(f"{EVAL}: {len(d)} rows, offset {int((d.regime == 'offset').sum())} ({(d.regime == 'offset').mean():.1%}); "
      f"window 2019Q3-2021Q4: {int(win.sum())} rows, offset {(d[win].regime == 'offset').mean():.1%}")
for scope, sub in (("all rows", d), ("Major TOM 2019Q3-2021Q4", d[win])):
    print(f"\n[{scope}]  offset − clean, mean per row [95% bootstrap CI] (n offset / clean)")
    for m in ("S2L2A", "NDVI", "S1RTC", "S1GRD", "DEM", "LULC"):
        for q in ("depth", "ce"):
            o, c = sub[sub.regime == "offset"][f"{q}_{m}"].values, sub[sub.regime == "clean"][f"{q}_{m}"].values
            if np.isfinite(o).sum() < 20 or np.isfinite(c).sum() < 20:
                continue
            v, lo, hi, no, nc = boot_diff(o, c)
            print(f"  {m:6s} {q:5s} {v:+.4f} [{lo:+.4f}, {hi:+.4f}]  ({no}/{nc})   clean mean {np.nanmean(c):.4f}")
