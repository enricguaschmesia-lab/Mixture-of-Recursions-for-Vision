"""The cleaned dataset: S2L2A without ESA's BOA offset, and NDVI recomputed (2026-10-11).

WHAT IS WRONG IN TERRAMESH. Since processing baseline 04.00, ESA's L2A products store
DN = 10000 * reflectance + 1000 (BOA_ADD_OFFSET = -1000 in the metadata). TerraMesh removes
the offset by ACQUISITION DATE ("the +1000 offset is removed from post-2022 data"), but Major
TOM also serves pre-2022 scenes from the Collection-1 reprocessing, baseline 05.00, which
carry it too. Those keep +1000 DN in every band, and TerraMesh's NDVI, computed from the
bands as stored, is compressed toward 0 in them.

EVIDENCE (every 10th val scene, matched exactly to Major TOM's metadata on grid cell + S2 time):
  baseline 05.00, acquired before 2022-01-25 : 96% have every band's 1st percentile > 900 DN;
                                               water pixels B12 median 1,048 DN (117/117 > 900)
  every other baseline, and all later scenes : 1-8% (naturally bright scenes); water B12 ~86 DN
  'N9999' (not a real baseline)              : behaves as clean (3%, the natural rate)
  SSL4EO-S12 (no product ids, 2019-12..2021-10): 2.5%, the natural rate -> clean
The 1st-percentile rule is only a cross-check: it misses ~4% of offset scenes (dark B01/B02
pixels with slightly negative reflectance) and flags ~3% of clean ones (bright surfaces).

THE CLEANED SET keeps TerraMesh's convention (no offset; negative values kept, as TerraMesh does
after 2022). Only S2L2A and NDVI change, and only in the offset rows; it lives in its own root
(CLEAN_ROOT_NAME) next to val/, with every unchanged artifact symlinked, so a run is switched to
it by TERRAMESH_TOK_ROOT alone.

Pure numpy/pandas: importable from both envs, like contract.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import contract as C

BOA_ADD_OFFSET = 1000                       # DN, every band
#: Real processing baselines carrying the offset: 04.00 up to (excluding) the 'N9999' placeholder.
OFFSET_BASELINES = range(400, 9000)
#: TerraMesh removed the offset from products acquired on or after this date.
TERRAMESH_OFFSET_CUTOFF = "2022-01-25"

#: TerraMesh's NDVI: bands clipped at 0, a small epsilon in the denominator, on DN, stored as
#: float16. Reproduces the stored NDVI bit-exactly (float16) on scenes with and without
#: negative pixels (2026-10-11); `tokenize_clean.py verify-ndvi` checks every row.
NDVI_EPS = 1e-6
_B04, _B08 = C.BAND_ORDER["S2L2A"].index("B04"), C.BAND_ORDER["S2L2A"].index("B08")

CLEAN_ROOT_NAME = "val_clean"
LABELS_FILE = "s2_offset_labels.parquet"
CLEANED_MODALITIES = ("S2L2A", "NDVI")


def has_offset(baseline: int, s2_time) -> bool:
    """True iff a Major TOM product of this baseline, acquired then, still carries the offset."""
    import pandas as pd
    return int(baseline) in OFFSET_BASELINES and pd.Timestamp(s2_time) < pd.Timestamp(TERRAMESH_OFFSET_CUTOFF)


def correct_s2(arr: np.ndarray, offset: bool) -> np.ndarray:
    """Raw S2L2A (..., 12, H, W) as stored -> the same array without the offset. Same dtype."""
    if not offset:
        return arr
    if arr.shape[-3] != C.N_CHANNELS["S2L2A"]:
        raise ValueError(f"expected 12 S2L2A bands on axis -3, got shape {arr.shape}")
    out = arr.astype(np.int32) - BOA_ADD_OFFSET
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if out.min() < info.min:
            raise ValueError(f"offset removal underflows {arr.dtype}")
    return out.astype(arr.dtype)


def ndvi_from_s2(arr: np.ndarray) -> np.ndarray:
    """S2L2A (1, 12, H, W) -> NDVI (1, 1, H, W) float16, exactly as TerraMesh stores it."""
    red = np.clip(arr[..., _B04, :, :].astype(np.float64), 0, None)
    nir = np.clip(arr[..., _B08, :, :].astype(np.float64), 0, None)
    return ((nir - red) / (nir + red + NDVI_EPS)).astype(np.float16)[..., None, :, :]


def clean_root(val_root: Path) -> Path:
    """The cleaned root that sits next to a val/ root."""
    return Path(val_root).parent / CLEAN_ROOT_NAME


def load_labels(root: Path):
    """The per-row labels (row, stem, corpus, offset, ...) written by label_s2_offset.py."""
    import pandas as pd
    return pd.read_parquet(Path(root) / LABELS_FILE)
