"""BigEarthNet v1 for the Phase 6 downstream demo (docs PHASE6_PLAN.md).

    labels.py   the 19-class nomenclature and the 43 -> 19 grouping (any env)
    layout.py   paths, band groups, the packed-raster layout (any env)
    splits.py   patch index: S1/S2 pairing, official + SMARTIES splits (any env)
    pack.py     stream the v1 archives into native-resolution memmaps (`mor`: tifffile)
    preprocess.py  packed rasters -> encoder-ready 224 x 224 tensors (`mor`)

Split by dependency, not by directory, like the rest of `eo/`: everything except
`pack.py` and `preprocess.py` is stdlib/numpy/pandas only.
"""
