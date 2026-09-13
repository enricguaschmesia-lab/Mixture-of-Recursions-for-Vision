# eo/mor_data/eo_vocab.py
"""PROVISIONAL unified vocabulary for the TerraMesh EO modalities.

⚠ THIS FILE IS A PLACEHOLDER. Phase 2 owns the real design (see the block
below) and is expected to rewrite it. It exists so Phase 1 Step 7's dataloader
has working token ids to emit and something to load against -- not because
these are the right ids.

Layout mirrors lm_dataset/multimodal_vocab_shared_caption_scene_desc.py so the
arithmetic Phase 2 reasons about is the same one that is already in the repo:

    [0, sum_codebooks)              per-modality codebooks, disjoint
    [sum_codebooks, +2M)            BO/EO pair per modality
    sum_codebooks + 2M              <PAD>

Codebook sizes are imported from eo/terramesh_tok/contract.py, which is the
single source of truth (CLAUDE.md). Never restate one here.

⚠ Why this is a NEW registry rather than an append to the CLEVR one:
bo_id/eo_id/PAD_ID are computed as sum_codebooks + ..., so appending any
modality to the CLEVR registry shifts every BO/EO and PAD id and invalidates
the existing CLEVR checkpoints. Only codebook_offset is append-stable. Phase 3
trains from scratch (use_pretrained_weights: false), so a separate registry
costs nothing.

## Phase 2 open design questions

1. Exact vs. rounded codebook slots. This file uses exact sizes (15,360 /
   4,375 / 6,365). CLEVR rounded Cosmos 64,000 and GPT-2 50,264. Rounding buys
   room to swap a tokenizer without moving ids; exactness buys a smaller
   embedding table.
2. Shared vs. disjoint tables for the five 15,360-codebook modalities. They are
   disjoint here. Sharing would be a claim, not a saving: Step 3 measured 0/256
   token overlap between S1GRD and S1RTC on an identical SAR input, so their
   codes are unrelated despite the equal codebook size.
3. DEM uses only 1,094 of its 15,360 codes (7.1%, Step 5). Compact it, or keep
   the slot uniform with the other FSQ modalities?
4. Delimiters. BO/EO per modality is inherited from CLEVR unexamined. Whether
   EO wants a global BOS, or a preamble encoding the modality order, is open.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.terramesh_tok.contract import CODEBOOK, MODALITIES as _CONTRACT_MODALITIES, TOKENS_PER_SAMPLE

# Coords is not part of D1.5 and has no entry in contract.py. Size is the
# OBSERVED max id + 1 over all 89,088 val samples (Coords_tok/metadata.json),
# deliberately NOT the tokenizer's own get_vocab_size(), which reports 2,169 --
# smaller than the largest id it emits. Sizing from it would index out of range.
COORDS_CODEBOOK = 6365
COORDS_TOKENS_PER_SAMPLE = 3


@dataclass
class ModalityInfo:
    name: str
    codebook_size: int
    data_type: str            # 'tokens' (EO has no text modality yet)
    tokens_per_sample: int
    codebook_offset: int = 0
    bo_id: int = 0
    eo_id: int = 0


# Order is permanent once anything is trained against it. APPEND only.
# Every slot is reserved whether or not the modality is active in a given run,
# so activating Coords later does not shift any already-trained id. Same
# principle as multimodal_vocab.py's docstring.
_MODALITY_REGISTRY: List[Tuple[str, int, str, int]] = [
    (m, CODEBOOK[m], 'tokens', TOKENS_PER_SAMPLE) for m in _CONTRACT_MODALITIES
] + [
    ('Coords', COORDS_CODEBOOK, 'tokens', COORDS_TOKENS_PER_SAMPLE),
]


def build_vocab() -> Tuple[Dict[str, ModalityInfo], int, int]:
    modalities: Dict[str, ModalityInfo] = {}
    offset = 0
    for name, codebook_size, data_type, n_tok in _MODALITY_REGISTRY:
        modalities[name] = ModalityInfo(
            name=name,
            codebook_size=codebook_size,
            data_type=data_type,
            tokens_per_sample=n_tok,
            codebook_offset=offset,
        )
        offset += codebook_size

    sum_codebooks = offset
    for i, name in enumerate(modalities):
        modalities[name].bo_id = sum_codebooks + 2 * i
        modalities[name].eo_id = sum_codebooks + 2 * i + 1

    pad_id = sum_codebooks + 2 * len(modalities)
    return modalities, pad_id, pad_id + 1


MODALITIES, PAD_ID, TOTAL_VOCAB_SIZE = build_vocab()

# Per-token modality id for per-modality loss logging. 0 = no modality.
MODALITY_TO_ID: Dict[str, int] = {name: i + 1 for i, name in enumerate(MODALITIES)}
ID_TO_MODALITY: Dict[int, str] = {i: name for name, i in MODALITY_TO_ID.items()}

# Modalities carrying pixel data, i.e. every slot except Coords. Used as the
# default active set and as the patch-shuffle eligibility test.
IMAGE_MODALITIES: List[str] = list(_CONTRACT_MODALITIES)

# Import-time invariants: disjoint, in-bounds, no overlap with specials.
_prev_end = 0
for _n, _i in MODALITIES.items():
    assert _i.codebook_offset == _prev_end, f"{_n}: codebook slots must be contiguous"
    assert _i.codebook_offset + _i.codebook_size <= PAD_ID, f"{_n}: slot overruns PAD"
    assert _i.bo_id < TOTAL_VOCAB_SIZE and _i.eo_id < TOTAL_VOCAB_SIZE, f"{_n}: BO/EO out of range"
    _prev_end = _i.codebook_offset + _i.codebook_size
assert len({i.bo_id for i in MODALITIES.values()} | {i.eo_id for i in MODALITIES.values()}) == 2 * len(MODALITIES)
del _prev_end, _n, _i


def get_modality(name: str) -> ModalityInfo:
    if name not in MODALITIES:
        raise KeyError(
            f"Unknown EO modality '{name}'. Known: {list(MODALITIES.keys())}. "
            f"To add one, append to _MODALITY_REGISTRY in eo/mor_data/eo_vocab.py."
        )
    return MODALITIES[name]
