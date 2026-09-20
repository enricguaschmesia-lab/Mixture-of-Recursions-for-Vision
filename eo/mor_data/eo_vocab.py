# eo/mor_data/eo_vocab.py
"""The unified vocabulary for the TerraMesh EO modalities.

DESIGN: docs/notes/vocabulary_design.md (D2.2, decided 2026-09-20). That note
carries the evidence and the rejected alternatives; this file carries the
layout. Where they disagree, the note is the specification and this file is the
bug.

    [0, sum_codebooks)          per-modality codebooks, disjoint, exact sizes
    [sum_codebooks, +2M)        BO/EO pair per modality, in registry order
    sum_codebooks + 2M          <PAD>

    sum_codebooks = 87,541      PAD_ID = 87,555      TOTAL_VOCAB_SIZE = 87,556

⚠ THIS IS A ONE-WAY DOOR. Once anything is trained against these ids, changing
the layout invalidates the checkpoint. Only `codebook_offset` is append-stable:
appending a modality, or resizing the LAST codebook slot, shifts every BO/EO and
PAD id. Same hazard as the CLEVR registry, which is why EO has its own.

The five decisions, in short. Reasons and numbers are in D2.2.

1. EXACT codebook slots, no rounding. Rounding buys room to swap a tokenizer
   without moving ids; tokenizer revisions are SHA-pinned in
   /data/enric/weights/tokenizer_manifest.json and swapping is not on the
   roadmap, so that room is unusable. Meanwhile the vocabulary multiplies the
   largest tensor in the model -- the (B, L, V) logits -- and is paid every
   step. TerraMind itself uses exact FSQ sizes for image modalities
   (terratorch .../terramind/model/modality_info.py:193-234).

2. DISJOINT slots for the five 15,360-codebook modalities. Equal codebook SIZE
   is not shared codebook SEMANTICS: one identical SAR tensor through the S1GRD
   and S1RTC tokenizers gives 0/196 equal ids, overlap 4 against ~2.5 expected
   by chance (measured at the 224 contract). CLEVR's `codebook_group` sharing
   mechanism is deliberately not carried over -- nothing here would use it.

3. DEM is NOT compacted, though it uses only 1,026 of 15,360 codes (6.7%).
   Compaction needs a dense<->FSQ remap that Phase 4 must invert, and the figure
   is measured on val only -- a code val never hit would be unmappable if the
   train split arrives. A real 16.4% saving, declined deliberately: reduce batch
   size before revisiting this.

4. PER-MODALITY BO/EO, with no global BOS and no preamble encoding the modality
   order. The structure is INHERITED FROM CLEVR, stated rather than assumed. It
   survives on its own merits: under `modality_order: 'random'` a chunk's first
   token IS its BO id, so the sequence is already self-describing, and a
   preamble would put a position-dependent signal at the front of every sequence
   exactly where Phase 4 is trying to read whether depth tracks modality.
   TerraMind's [S_N] sentinels are span-masking markers, not chunk delimiters,
   and it keeps one embedding table per modality so it never needs either.

5. COORDS is sized from the tokenizer's own id range, not from what val emitted.
   See COORDS_CODEBOOK below -- this one was a live bug.

Runs in the repo `.venv`, NOT the `mor` conda env: stdlib + the contract only,
never terratorch. Import does no file I/O; the artifact check is an explicit
call, `assert_artifact_fits()`.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.terramesh_tok.contract import (
    CODEBOOK,
    MODALITIES as _CONTRACT_MODALITIES,
    TOKENS_PER_SAMPLE,
    tok_dir_name,
)

# --- Coords ---------------------------------------------------------------
# 6,366 = the coords tokenizer's own id range, max(get_vocab().values()) + 1.
#
# THIS NUMBER IS NOT THE ONE THE TOKENIZER REPORTS, AND NOT THE ONE VAL SHOWS.
# Both of the obvious sources are wrong, in opposite directions:
#
#   get_vocab_size()  -> 2,169. That is an ENTRY COUNT. The vocabulary is
#                        SPARSE: 2,169 entries spread over ids 0..6,365, with
#                        4,197 unused ids below the maximum. Sizing from it
#                        would index out of range on most real coordinates.
#   val-observed max  -> 6,365 (max id 6,364 + 1). This is what this file used
#                        until 2026-09-20, and it is ONE SHORT. Id 6,365 is
#                        'lon=180.00', a legitimate point on the tokenizer's own
#                        0.25-degree grid at the antimeridian that the val split
#                        simply never visits.
#
# The old value did not fail loudly. 81,175 + 6,365 = 87,540, which under the
# old layout was exactly S2L2A's bo_id -- in range, no exception, the token just
# silently became a begin-of-modality marker for another modality.
#
# It is a literal rather than a call because this module runs in .venv, which
# has no terratorch and cannot import the tokenizer (see eo/README.md). The
# derivation is one line and is recorded in D2.2 section 5 so it stays
# reproducible rather than becoming folklore:
#
#   v = terramind_v1_coords_tokenizer(pretrained=True).text_tokenizer.get_vocab()
#   max(v.values()) + 1        # -> 6366
#
# 4,197 of these ids are sparse gaps the tokenizer cannot emit. They cost
# 12.3 MiB of embedding rows and train to nothing; closing them would need a
# dense remap, rejected for the same reason as DEM compaction (D2.2 section 2.5).
COORDS_CODEBOOK = 6366
COORDS_TOKENS_PER_SAMPLE = 3

# Keys holding the largest id actually present in each artifact's metadata.json.
# The image modalities and Coords were written by different scripts and use
# different names; both are checked by assert_artifact_fits().
_MAX_ID_KEYS = ("token_max", "observed_id_max")


@dataclass
class ModalityInfo:
    name: str
    codebook_size: int
    data_type: str            # 'tokens' -- EO has no text modality
    tokens_per_sample: int
    codebook_offset: int = 0
    bo_id: int = 0
    eo_id: int = 0


# Order is permanent once anything is trained against it. APPEND only, and note
# that appending still shifts every BO/EO and PAD id -- only codebook_offset is
# stable. Every slot is reserved whether or not the modality is active in a given
# run, so activating Coords later does not shift an already-trained id.
#
# Sizes come from contract.py, the single source of truth. Never restate one
# here (CLAUDE.md); Coords is the sole exception and carries its provenance above.
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


# TOTAL_VOCAB_SIZE is DERIVED. No human maintains it, and the training config's
# model_config.vocab_size is asserted against it at startup
# (lm_dataset/modality_registry.assert_vocab_size).
MODALITIES, PAD_ID, TOTAL_VOCAB_SIZE = build_vocab()

# Per-token modality id for per-modality loss logging. 0 = no modality.
MODALITY_TO_ID: Dict[str, int] = {name: i + 1 for i, name in enumerate(MODALITIES)}
ID_TO_MODALITY: Dict[int, str] = {i: name for name, i in MODALITY_TO_ID.items()}

# Modalities carrying pixel data, i.e. every slot except Coords. Used as the
# default active set and as the patch-shuffle eligibility test.
IMAGE_MODALITIES: List[str] = list(_CONTRACT_MODALITIES)

# --- Import-time invariants ----------------------------------------------
# Pure arithmetic, no I/O, so this module stays importable without the dataset
# mounted. Cheap, and they caught real problems in Phase 1.
#
# ⚠ KNOW WHAT THESE CANNOT DO. They verify that build_vocab()'s DERIVATION is
# self-consistent. They cannot validate the codebook SIZES, because the sizes are
# inputs to that derivation: BO/EO starts wherever the codebooks end and PAD
# follows BO/EO, so ANY set of sizes produces a perfectly consistent layout.
# Setting Coords back to the buggy 6,365 trips none of these -- verified by
# mutation, 2026-09-20. An assert of the form "no codebook reaches into the BO/EO
# block" is vacuous for the same reason and was removed rather than left to give
# false assurance.
#
# What actually guards the sizes:
#   image modalities -> contract.CODEBOOK, i.e. the released FSQ codebook sizes.
#   Coords           -> assert_artifact_fits(), against `tokenizer_id_bound`
#                       recorded in Coords_tok/metadata.json by tokenize_coords.py,
#                       which runs in `mor` and can query the tokenizer directly.
_prev_end = 0
for _n, _i in MODALITIES.items():
    assert _i.codebook_size > 0, f"{_n}: empty codebook slot"
    assert _i.codebook_offset == _prev_end, f"{_n}: codebook slots must be contiguous"
    assert _i.codebook_offset + _i.codebook_size <= PAD_ID, f"{_n}: slot overruns PAD"
    assert _i.bo_id < TOTAL_VOCAB_SIZE and _i.eo_id < TOTAL_VOCAB_SIZE, f"{_n}: BO/EO out of range"
    _prev_end = _i.codebook_offset + _i.codebook_size
assert _prev_end == min(i.bo_id for i in MODALITIES.values()), \
    "codebook block and BO/EO block must be adjacent with no gap"
_specials = {i.bo_id for i in MODALITIES.values()} | {i.eo_id for i in MODALITIES.values()}
assert len(_specials) == 2 * len(MODALITIES), "BO/EO ids must be unique"
assert PAD_ID not in _specials, "PAD must not collide with a BO/EO id"
assert TOTAL_VOCAB_SIZE == PAD_ID + 1, "PAD must be the last id"
del _prev_end, _n, _i, _specials


def get_modality(name: str) -> ModalityInfo:
    if name not in MODALITIES:
        raise KeyError(
            f"Unknown EO modality '{name}'. Known: {list(MODALITIES.keys())}. "
            f"To add one, append to _MODALITY_REGISTRY in eo/mor_data/eo_vocab.py."
        )
    return MODALITIES[name]


def assert_artifact_fits(root_dir, modalities: Optional[List[str]] = None) -> Dict[str, int]:
    """Check each artifact's recorded maximum token id against its slot size.

    The import-time invariants above prove the LAYOUT is self-consistent. They
    cannot prove the DATA fits it -- that depends on what the tokenizers actually
    emitted, which is exactly where the Coords bug lived. This closes that gap.

    Deliberately not run at import: it reads metadata.json, and making the
    vocabulary unimportable without the dataset mounted would be a bad trade for
    a check that belongs where the arrays are opened. `TerraMeshTokenDataset`
    calls it once in __init__.

    Reads the recorded maximum rather than scanning the arrays -- it is O(1)
    against six 34.9 MB memory maps, and the value is written by the same run
    that produced the tokens.

    Returns {modality: recorded_max_id}. Raises RuntimeError on overflow.
    """
    root = Path(root_dir)
    names = list(modalities) if modalities else list(MODALITIES)
    seen: Dict[str, int] = {}
    for name in names:
        info = get_modality(name)
        meta_path = root / tok_dir_name(name) / "metadata.json"
        if not meta_path.exists():
            continue                      # nothing recorded; the shape check will catch a real problem
        meta = json.loads(meta_path.read_text())
        recorded = next((meta[k] for k in _MAX_ID_KEYS if meta.get(k) is not None), None)
        if recorded is None:
            continue
        # The strongest check available: the tokenizer's own id range, recorded
        # by the mor-env script that produced the artifact. Unlike the observed
        # maximum, this covers ids the tokenizer CAN emit but this split did not
        # -- which is exactly how the Coords slot came to be one id short.
        bound = meta.get("tokenizer_id_bound")
        if bound is not None and int(bound) > info.codebook_size:
            raise RuntimeError(
                f"'{name}' vocabulary slot is {info.codebook_size} wide, but its "
                f"tokenizer can emit ids up to {int(bound) - 1} "
                f"(tokenizer_id_bound={int(bound)}, {meta_path}).\n"
                f"  The slot is too small for ids this split happens not to contain, "
                f"so nothing would fail until such a sample appears -- and then it "
                f"would land on a BO/EO marker and corrupt silently rather than raise.\n"
                f"  Widen the slot in eo/mor_data/eo_vocab.py and update "
                f"docs/notes/vocabulary_design.md."
            )

        recorded = int(recorded)
        seen[name] = recorded
        if recorded >= info.codebook_size:
            raise RuntimeError(
                f"'{name}' artifact holds token id {recorded}, but its vocabulary slot "
                f"is only {info.codebook_size} wide ({meta_path}).\n"
                f"  Global id would be {info.codebook_offset + recorded}, which lands "
                f"outside this modality's slot -- and the ids immediately above it are "
                f"BO/EO markers, so this does NOT raise at training time, it silently "
                f"corrupts. Widen the slot in eo/mor_data/eo_vocab.py and update "
                f"docs/notes/vocabulary_design.md, or re-check the artifact."
            )
    return seen
