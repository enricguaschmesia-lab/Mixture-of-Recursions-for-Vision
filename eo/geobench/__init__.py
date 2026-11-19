"""GEO-Bench-2's segmentation tasks read into TerraMind's token contract (Paper A, A2 Step 2).

    sources   where the GEO-Bench-2 tortillas are, their pinned revisions and sha256s, and readers for
              PASTIS (S2 series, dates, semantic masks) and CloudSen12 (L2A tiles, masks)
    s2        the S2 input contract: GEO-Bench-2 band order -> contract.BAND_ORDER["S2L2A"], the B01/B09
              imputation, one bilinear resample to the crop, v1 standardization, the dihedral variants
    views     the fixed index of what gets tokenized: per task, one row per (sample, crop, dihedral
              variant), and for PASTIS one input row per (view, frame)

Runs in the `a2seg` conda env. Every constant of the token contract is imported from
eo/terramesh_tok/contract.py, never restated.
"""
