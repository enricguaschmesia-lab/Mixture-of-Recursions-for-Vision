"""Where BigEarthNet v1 lives on disk, and how the packed rasters are laid out.

The v1 archives ship one GeoTIFF per band per patch (S2: 12 bands at three resolutions, S1:
VV and VH). Reading ~7 M small files from a 70 GB tar.gz per pass is the cost, so `pack.py`
streams each archive ONCE and appends every wanted patch, at native resolution, to a few
arrays in ARCHIVE order; `order_<kind>.npy` maps each position to its `index_v1.parquet` row
and `load_packed` returns the inverse. Nothing is resampled at pack time: resampling to 224
is part of the tokenizer contract (`preprocess.py`), not of storage.

S2 is stored in three arrays by native resolution. The band ORDER inside each group is
fixed here; `preprocess.py` reassembles the 12 bands in `contract.BAND_ORDER["S2L2A"]` order
and asserts it, so the grouping can never leak into the tokenizer input.
"""
from __future__ import annotations

import os
from pathlib import Path

# BEN_ROOT overrides it off the workstation (RCP); the dataloader reads the same variable.
ROOT = Path(os.environ.get("BEN_ROOT", "/data/enric/data/BigEarthNet"))
V1 = ROOT / "v1"                         # the downloaded archives + official small files
PACKED = ROOT / "v1_stream"              # pack.py output, in ARCHIVE order (see pack.py)
SPLITS = ROOT / "splits"                 # pinned SMARTIES lists (copied, sha256 recorded)
INDEX = ROOT / "index_v1.parquet"        # one row per clean S2/S1 pair; THE row order

S2_ARCHIVE = V1 / "BigEarthNet-S2-v1.0.tar.gz"
S1_ARCHIVE = V1 / "BigEarthNet-S1-v1.0.tar.gz"

# Native groups. 10 m: 120 px; 20 m: 60 px; 60 m: 20 px. All cover the same 1.2 km footprint.
S2_GROUPS = {
    "S2_10m": (["B02", "B03", "B04", "B08"], 120),
    "S2_20m": (["B05", "B06", "B07", "B8A", "B11", "B12"], 60),
    "S2_60m": (["B01", "B09"], 20),
}
S1_BANDS = (["VV", "VH"], 120)            # float32, dB (inspected 2026-10-12)
S2_DTYPE = "uint16"
S1_DTYPE = "float32"

assert sorted(b for bands, _ in S2_GROUPS.values() for b in bands) == sorted(
    ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
), "S2 groups must cover the 12 L2A bands exactly once (no B10)"


PACK_ROLES = ("ft_train", "ft_valsel", "test_a", "test_c")   # what pack_ben.py packs


def load_packed(kind: str, n_rows: int):
    """({array name: memmap}, row_to_pos) for 'S2' or 'S1'. row_to_pos[row] = -1 if not packed."""
    import numpy as np
    names = list(S2_GROUPS) if kind == "S2" else ["S1"]
    arrays = {g: np.load(PACKED / f"{g}.npy", mmap_mode="r") for g in names}
    order = np.load(PACKED / f"order_{kind}.npy")
    r2p = np.full(n_rows, -1, dtype=np.int64)
    r2p[order] = np.arange(len(order))
    for a in arrays.values():
        assert a.shape[0] == len(order), (a.shape, len(order))
    return arrays, r2p
