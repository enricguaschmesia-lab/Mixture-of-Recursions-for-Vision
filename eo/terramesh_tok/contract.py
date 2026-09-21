"""The TerraMesh -> TerraMind FSQ token preprocessing contract.

SINGLE SOURCE OF TRUTH. Step 4 (batch tokenization) and Phase 5 (routing
analysis) must both import from here rather than re-deriving any of it.

Every constant below was established empirically in Phase 1 Step 3; see
STEP3_PLAN.md sections 1.1-1.5 and notes/tokenizer_bringup.md for the evidence.
The one exception is CROP, which is a supervisor decision rather than a
measurement -- see its comment below.
"""
from __future__ import annotations

# --- Geometry -----------------------------------------------------------
NATIVE = 264          # TerraMesh on-disk spatial size
CROP = 224            # Ratified at Meeting 2, 2026-09-15: 224 is TerraMind's
                      # own operating point and the standard EO crop. This
                      # REVERSES the Phase 1 decision of 256, against which the
                      # D1.5 artifact was originally produced.
PATCH = 16
GRID = CROP // PATCH  # 14 -> 196 tokens/sample
CROP_OFF = (NATIVE - CROP) // 2   # 20 px off each side, exactly symmetric

# Why 224 is safe, since the obvious objection has already been raised once and
# retracted once, and the evidence for it is still sitting in terratorch:
#
#   DEM and NDVI really were trained at 256 ONLY -- terratorch records each
#   tokenizer's original training args, and those two carry
#   input_size_min=256, input_size_max=256 (tokenizer_register.py:240 and :295),
#   where S2L2A (:157), S1GRD (:184, :212) and LULC (:268) carry
#   input_size_min=224. An earlier version of this comment used that fact to
#   argue 224 would be out of distribution.
#
#   That inference was MEASURED FALSE on 2026-09-09 (worklog: STEP3_PLAN 1.3
#   retraction). 224 reconstructs within +-10% of 256 on every modality,
#   including DEM and NDVI. The fact stands; the conclusion drawn from it does
#   not. Do not re-litigate this from the training args alone.
#
# The mechanism, because it is the non-obvious part and it has a consequence:
# tokenizers.build() builds at the registry default image_size=256 and is NOT
# changed for 224. ViTEncoder.forward bicubically interpolates pos_emb from the
# built 16x16 grid to whatever patch grid the input actually produces, and
# asserts only divisibility by the patch size (terratorch .../terramind/
# tokenizer/models/vit_models.py:550-555, assert at :542). A 224 input yields a
# 14x14 grid = 196 tokens at forward time, with no builder change.
#
#   CONSEQUENCE: the 224 tokens are NOT a sub-crop of the 256 tokens.
#   Interpolating the position embeddings perturbs every patch embedding, so
#   all 196 differ from any 196 of the old set. There is no cheap cross-check
#   against the 256 artifact; the 224 set is verified on its own terms.

# --- Token layout -------------------------------------------------------
# tokenizer.encode() returns (B, H_q, W_q). We flatten ROW-MAJOR (C order):
#     token k  <->  patch (row = k // GRID, col = k % GRID)
# Phase 5's patch<->token mapping depends on this. Do not change it.
FLATTEN_ORDER = "row-major C order; row = k // GRID, col = k % GRID"
TOKENS_PER_SAMPLE = GRID * GRID   # 196

# --- Standardization ----------------------------------------------------
# Source: terratorch terramind_register.py  v1_pretraining_mean/std, 'tok_*'
# keys. These -- not terramesh_statistics.yaml -- are what TerraMind applies
# immediately before tokenizer.encode(). Decision D3.2, see STEP3_PLAN 1.2a.
V1_TOK_MEAN = {
    "S2L2A": [1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975,
              2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648],
    "S1GRD": [-12.599, -20.293],
    "S1RTC": [-10.930, -17.329],
    "DEM":   [670.665],
    "NDVI":  [0.327],
    "LULC":  None,   # categorical - NO standardization (tok_lulc mean 0 std 1)
}
V1_TOK_STD = {
    "S2L2A": [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
              1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
    "S1GRD": [5.195, 5.890],
    "S1RTC": [4.391, 4.459],
    "DEM":   [951.272],
    "NDVI":  [0.322],
    "LULC":  None,
}
# Dataset-side alternative (terramesh_statistics.yaml / terramesh.py L50).
# Kept ONLY for the A/B test in Step 3.5. Not the production contract.
YAML_MEAN = {
    "S2L2A": [1390.461, 1503.332, 1718.211, 1853.926, 2199.116, 2779.989,
              2987.025, 3083.248, 3132.235, 3162.989, 2424.902, 1857.665],
    "S1GRD": [-12.577, -20.265], "S1RTC": [-10.930, -17.329],
    "DEM": [651.663], "NDVI": [0.327], "LULC": None,
}
YAML_STD = {
    "S2L2A": [2131.157, 2163.666, 2059.311, 2152.477, 2105.179, 1912.773,
              1842.326, 1893.568, 1775.656, 1814.907, 1436.282, 1336.155],
    "S1GRD": [5.179, 5.872], "S1RTC": [4.391, 4.459],
    "DEM": [928.168], "NDVI": [0.322], "LULC": None,
}

# --- Per-modality facts -------------------------------------------------
TOKENIZER = {
    "S2L2A": "terramind_v1_tokenizer_s2l2a",
    "S1GRD": "terramind_v1_tokenizer_s1grd",
    "S1RTC": "terramind_v1_tokenizer_s1rtc",
    "DEM":   "terramind_v1_tokenizer_dem",
    "NDVI":  "terramind_v1_tokenizer_ndvi",
    "LULC":  "terramind_v1_tokenizer_lulc",
}
CODEBOOK = {m: 15360 for m in TOKENIZER}
CODEBOOK["LULC"] = 4375           # FSQ 7-5-5-5-5

# LULC is fed as a ONE-HOT stack, not class indices. Proven from the released
# checkpoint: no cls_emb.* keys, encoder.proj.weight is (768, 10, 16, 16), and
# decoder.out_proj is 2560 = 10 * 16 * 16. modality_info.py's num_channels=9
# describes the untokenized pixel path, not the tokenizer.
LULC_N_CLASSES = 10
N_CHANNELS = {"S2L2A": 12, "S1GRD": 2, "S1RTC": 2, "DEM": 1, "NDVI": 1,
              "LULC": LULC_N_CLASSES}

BAND_ORDER = {
    "S2L2A": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
              "B8A", "B09", "B11", "B12"],
    "S1GRD": ["vv", "vh"], "S1RTC": ["vv", "vh"],
    "DEM": ["DEM"], "NDVI": ["NDVI"], "LULC": ["LULC"],
}
# Modalities are stored already in this order on disk (verified against the
# zarr 'band' coordinate). No reordering is applied; preprocess.py asserts it.

MODALITIES = list(TOKENIZER)
IS_FLOAT_NODATA = {"S1GRD", "S1RTC", "NDVI"}   # float16 with fill_value=NaN

# --- NaN policy ---------------------------------------------------------
# Set in Step 3.3 from measured prevalence. 'mean' -> fill with the modality
# mean so the value becomes exactly 0 after standardization (least-disruptive).
NAN_POLICY = "mean"

# Token dtype: 15360 > 255, so uint8 is impossible; uint16 is exact and minimal.
TOKEN_DTYPE = "uint16"

# --- On-disk artifact layout --------------------------------------------
# Tokenized output lives at <root>/<MODALITY><TOK_DIR_SUFFIX>/. The crop is IN
# THE DIRECTORY NAME, deliberately: two token sets now coexist (the Phase 1 256
# set and the ratified 224 one), and the highest-consequence silent failure in
# Phase 2 is training against the wrong one. Deriving the suffix from CROP makes
# that structurally impossible rather than merely asserted -- there is no flag to
# forget and no way to write 224 arrays into a 256-named directory.
#
# Every reader of the artifact imports this instead of hardcoding "_tok":
# eo/scripts/{tokenize_terramesh,tokenize_coords,verify_step4,verify_step5,
# verify_step7}.py and eo/data/terramesh_token_dataset.py.
TOK_DIR_SUFFIX = f"_tok{CROP}"

# Modalities whose tokens do not depend on the crop, and which therefore keep an
# UNSUFFIXED directory and are never re-tokenized when the crop changes. Coords
# is tokenized from the scene's centre_lon/centre_lat (val_metadata.parquet);
# both the 256 and 224 crops are centred on that same point, so the coordinate
# -- and every token derived from it -- is identical at any crop.
# This is why Coords_tok/ carries an older date than the image modalities. The
# set is here rather than special-cased in each caller so that fact is stated
# once, in the contract, instead of being rediscovered from a directory listing.
CROP_INDEPENDENT = {"Coords"}


def tok_dir_name(modality: str) -> str:
    """Directory name holding one modality's tokens at the contract crop.

    Crop-independent modalities (Coords) keep the unsuffixed '<MOD>_tok' name,
    because there is no per-crop version of them to disambiguate.
    """
    if modality in CROP_INDEPENDENT:
        return f"{modality}_tok"
    return f"{modality}{TOK_DIR_SUFFIX}"
