"""The fixed index of what A2 tokenizes (Step 2.3-2.5): views, and for PASTIS their input frames.

A *view* is one map the model writes: (sample, crop, dihedral variant k). Its input tokens and its map
tokens come from the same native raster cut and transformed the same way (eo/geobench/s2.dihedral), so
"dihedral map = dihedral image" holds by construction (gate Z5).

PASTIS (128 px, one crop: the whole patch):
  train  each of the 1,455 patches (folds 1-3, GEO-Bench-2's head revision; `in_pinned` marks the 1,200
         that GeoBenchV2 0.9's pinned revision holds) x 8 dihedral variants
  val    482 patches (fold 4), k = 0;  test 496 (fold 5), k = 0
  frames per view:
    p1     the clearest July-August acquisition (A2 Step 2.3; Part 2 Step 0's pick, `summer_date` in
           /data/enric/reports/pastis/step0/patches.parquet; clear for 91.7% of patches)
    p7_j   GEO-Bench-2's 7 evenly spaced indices int(j * T / 7), j = 0..6, dates from the patch's own
           date list at those indices (finding 9)
CloudSen12 (512-px tiles, 224-px crops):
  train  9 fixed random crops per tile (seed SEED) x 8 dihedral variants = 72 views per tile
  val, test  the 3 x 3 test crops at offsets 0, 144, 288 (eo/maps/tiling.py), k = 0

Rows are numbered once, here, and written to <task root>/views.parquet (and inputs.parquet for
PASTIS). Every token array is row-aligned with one of them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eo.geobench import sources as S
from eo.maps import tiling

SEED = 20261119
N_RANDOM_CROPS = 9
CROP = 224
TOK_ROOT = {"pastis": S.ROOT / "pastis" / "tok", "cloudsen12": S.ROOT / "cloudsen12" / "tok"}
STEP0 = Path("/data/enric/reports/pastis/step0/patches.parquet")
FRAMES_P7 = [f"p7_{j}" for j in range(7)]


def pastis_views() -> pd.DataFrame:
    t = S.pastis_table()
    rows = []
    for split in ("train", "val", "test"):
        sub = t[t.split == split].sort_values("patch_id")
        ks = range(8) if split == "train" else (0,)
        for r in sub.itertuples():
            for k in ks:
                rows.append((split, int(r.idx), int(r.patch_id), 0, 0, 128, k, bool(r.in_pinned)))
    v = pd.DataFrame(rows, columns=["split", "sample_idx", "sample_id", "y0", "x0", "size", "k", "in_pinned"])
    v.insert(0, "view", np.arange(len(v)))
    return v


def pastis_inputs(views: pd.DataFrame) -> pd.DataFrame:
    """One row per (view, frame): frame name, its index in the patch's S2 series and its date."""
    t = S.pastis_table().set_index("patch_id")
    p1 = pd.read_parquet(STEP0).set_index("id").summer_date.astype(int)
    rows = []
    for v in views.itertuples():
        dates = [int(d) for d in t.loc[v.sample_id, "dates"]]
        d1 = int(p1.loc[v.sample_id])
        rows.append((v.view, "p1", dates.index(d1), d1))
        for j, ti in enumerate(S.pastis_frames_p7(len(dates))):
            rows.append((v.view, FRAMES_P7[j], ti, dates[ti]))
    x = pd.DataFrame(rows, columns=["view", "frame", "t", "date"])
    x.insert(0, "row", np.arange(len(x)))
    return x


def cloudsen12_views() -> pd.DataFrame:
    t = S.cloudsen12_table()
    rng = np.random.default_rng(SEED)
    rows = []
    for split in ("train", "val", "test"):
        sub = t[t.split == split].sort_values("idx")
        for r in sub.itertuples():
            if split == "train":
                yx = rng.integers(0, 512 - CROP + 1, size=(N_RANDOM_CROPS, 2))
                for y0, x0 in yx:
                    for k in range(8):
                        rows.append((split, int(r.idx), r.id, int(y0), int(x0), CROP, k))
            else:
                for (y0, x0, c) in tiling.plan("cloudsen12", 512, "crop224"):
                    rows.append((split, int(r.idx), r.id, int(y0), int(x0), int(c), 0))
    v = pd.DataFrame(rows, columns=["split", "sample_idx", "sample_id", "y0", "x0", "size", "k"])
    v.insert(0, "view", np.arange(len(v)))
    return v


def build(task: str) -> dict[str, pd.DataFrame]:
    if task == "pastis":
        v = pastis_views()
        return {"views": v, "inputs": pastis_inputs(v)}
    if task == "cloudsen12":
        v = cloudsen12_views()
        return {"views": v, "inputs": v.rename(columns={"view": "row"}).assign(view=v.view.values)}
    raise KeyError(task)


def load(task: str) -> dict[str, pd.DataFrame]:
    root = TOK_ROOT[task]
    return {k: pd.read_parquet(root / f"{k}.parquet") for k in ("views", "inputs")}


def write(task: str) -> dict:
    """Write the index once; refuse to overwrite a different one (every token array is aligned to it)."""
    root = TOK_ROOT[task]
    root.mkdir(parents=True, exist_ok=True)
    new = build(task)
    out = {}
    for k, df in new.items():
        p = root / f"{k}.parquet"
        if p.exists():
            old = pd.read_parquet(p)
            if not old.equals(df.reset_index(drop=True)):
                raise RuntimeError(f"{p} exists and differs from the index built now; refusing to overwrite")
        else:
            df.reset_index(drop=True).to_parquet(p)
        out[k] = len(df)
    return out
