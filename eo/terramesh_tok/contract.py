"""The TerraMesh -> TerraMind FSQ token preprocessing contract.

SINGLE SOURCE OF TRUTH. Step 4 (batch tokenization) and Phase 4 (routing
analysis) must both import from here rather than re-deriving any of it.

Every constant below was established empirically in Phase 1 Step 3; see
STEP3_PLAN.md sections 1.1-1.5 and notes/tokenizer_bringup.md for the evidence.
"""
from __future__ import annotations

# --- Geometry -----------------------------------------------------------
NATIVE = 264          # TerraMesh on-disk spatial size
CROP = 256            # 264 is not divisible by the 16px patch size; 256 keeps
                      # 94% of ground area and is in-distribution for ALL six
                      # tokenizers (DEM and NDVI were trained at 256 ONLY).
PATCH = 16
GRID = CROP // PATCH  # 16 -> 256 tokens/sample
CROP_OFF = (NATIVE - CROP) // 2   # 4 px off each side, exactly symmetric

# --- Token layout -------------------------------------------------------
# tokenizer.encode() returns (B, H_q, W_q). We flatten ROW-MAJOR (C order):
#     token k  <->  patch (row = k // GRID, col = k % GRID)
# Phase 4's patch<->token mapping depends on this. Do not change it.
FLATTEN_ORDER = "row-major C order; row = k // GRID, col = k % GRID"
TOKENS_PER_SAMPLE = GRID * GRID   # 256

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
