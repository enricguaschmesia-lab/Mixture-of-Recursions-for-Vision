# eo/data/eo_vocab_nc.py
"""The Coords-free EO vocabulary, layout "nc": eo_vocab's layout with the Coords slot and its BO/EO pair removed.

Enric, 2026-11-14: no model trained from now on is trained with Coords or includes Coords tokens, so Coords leaves
the embedding table and the codebook too, not only the data. eo_vocab.py's layout is a one-way door (every
checkpoint so far is trained against it), so it stays exactly as it is and this is a SECOND, frozen layout, chosen
per run by `multimodal.vocab: nc` (lm_dataset/modality_registry.get_registry; TerraMeshTokenDataset(vocab="nc")).

    [0, 81,175)          the six image codebooks, at eo_vocab's offsets and sizes exactly
    [81,175, 81,187)     a BO/EO pair per image modality, in registry order
    81,187               <PAD>                                   TOTAL_VOCAB_SIZE = 81,188

Image token ids and modality ids (S2L2A 1 ... LULC 6) are eo_vocab's. Only the BO/EO ids and PAD move, because
Coords was the LAST codebook slot. So:
  * a v1 sequence fed to an nc model raises: its BO/EO and PAD ids (>= 87,541) index past the table;
  * an nc sequence fed to a v1 model does NOT raise: its BO/EO ids (81,175-81,186) are Coords codebook rows there.
    Anything that scores a checkpoint must take the layout from the checkpoint's own vocab_size (`for_vocab_size`).
A v1 checkpoint and an nc checkpoint share no delimiter and no PAD id: never load one as the other.

Exposes eo_vocab's names (MODALITIES, PAD_ID, TOTAL_VOCAB_SIZE, MODALITY_TO_ID, ID_TO_MODALITY, IMAGE_MODALITIES,
DEFAULT_ACTIVE_MODALITIES, get_modality, assert_artifact_fits), so it is a registry like the others. The BEN and
temporal layouts are appended to v1 only: neither exists for nc yet, and get_registry refuses `vocab: nc` there.

Runs in `.venv` and in `mor`: stdlib, the contract and eo_vocab only. Import does no file I/O.
"""
from __future__ import annotations

from types import ModuleType
from typing import Dict, List, Optional

from eo.data import eo_vocab as _v1
from eo.data.eo_vocab import ModalityInfo

NAME = "nc"
COORDS = "Coords"

# eo_vocab's registry without Coords, in eo_vocab's order. Never restate a size here: they come from the contract
# through eo_vocab (CLAUDE.md).
_MODALITY_REGISTRY = [r for r in _v1._MODALITY_REGISTRY if r[0] != COORDS]


def _build():
    """eo_vocab.build_vocab's derivation over this registry (codebooks, then BO/EO pairs, then PAD)."""
    modalities: Dict[str, ModalityInfo] = {}
    offset = 0
    for name, codebook_size, data_type, n_tok in _MODALITY_REGISTRY:
        modalities[name] = ModalityInfo(name=name, codebook_size=codebook_size, data_type=data_type,
                                        tokens_per_sample=n_tok, codebook_offset=offset)
        offset += codebook_size
    for i, name in enumerate(modalities):
        modalities[name].bo_id = offset + 2 * i
        modalities[name].eo_id = offset + 2 * i + 1
    pad_id = offset + 2 * len(modalities)
    return modalities, pad_id, pad_id + 1


MODALITIES, PAD_ID, TOTAL_VOCAB_SIZE = _build()
MODALITY_TO_ID: Dict[str, int] = {name: _v1.MODALITY_TO_ID[name] for name in MODALITIES}
ID_TO_MODALITY: Dict[int, str] = {i: name for name, i in MODALITY_TO_ID.items()}
IMAGE_MODALITIES: List[str] = list(_v1.IMAGE_MODALITIES)
DEFAULT_ACTIVE_MODALITIES: List[str] = list(MODALITIES)

# --- Import-time invariants: the derivation, and that nothing but Coords changed. ------------------------------
assert COORDS not in MODALITIES and COORDS not in MODALITY_TO_ID, "nc must hold no Coords"
assert list(MODALITIES) == IMAGE_MODALITIES, "nc = eo_vocab's image modalities, in eo_vocab's order"
for _n, _i in MODALITIES.items():
    _o = _v1.MODALITIES[_n]
    assert (_i.codebook_offset, _i.codebook_size, _i.tokens_per_sample) == \
        (_o.codebook_offset, _o.codebook_size, _o.tokens_per_sample), f"{_n}: image slot differs from eo_vocab's"
    assert MODALITY_TO_ID[_n] == _v1.MODALITY_TO_ID[_n], f"{_n}: modality id differs from eo_vocab's"
_end = max(i.codebook_offset + i.codebook_size for i in MODALITIES.values())
assert _end == _v1.MODALITIES[COORDS].codebook_offset, "the image codebooks end where eo_vocab's Coords slot began"
_specials = [i.bo_id for i in MODALITIES.values()] + [i.eo_id for i in MODALITIES.values()]
assert sorted(_specials) == list(range(_end, _end + 2 * len(MODALITIES))), "BO/EO contiguous right after the codebooks"
assert PAD_ID == _end + 2 * len(MODALITIES) and TOTAL_VOCAB_SIZE == PAD_ID + 1, "PAD is the last id"
assert TOTAL_VOCAB_SIZE == 81_188 and PAD_ID == 81_187, "the size every nc config and checkpoint carries"
assert TOTAL_VOCAB_SIZE == _v1.TOTAL_VOCAB_SIZE - _v1.MODALITIES[COORDS].codebook_size - 2, \
    "nc is eo_vocab minus Coords' codebook and its BO/EO pair, nothing else"
del _n, _i, _o, _end, _specials


def get_modality(name: str) -> ModalityInfo:
    if name not in MODALITIES:
        why = (" The nc vocabulary has no Coords: no model trained since 2026-11-14 includes Coords tokens "
               "(Enric). Drop it from multimodal.active_modalities." if name == COORDS else "")
        raise KeyError(f"Unknown modality '{name}' in the nc vocabulary. Known: {list(MODALITIES)}.{why}")
    return MODALITIES[name]


def assert_artifact_fits(root_dir, modalities: Optional[List[str]] = None) -> Dict[str, int]:
    """eo_vocab.assert_artifact_fits for the image modalities, whose slots are eo_vocab's exactly; Coords refused."""
    names = list(modalities) if modalities else list(MODALITIES)
    for n in names:
        get_modality(n)
    return _v1.assert_artifact_fits(root_dir, names)


def layout(name: Optional[str]) -> ModuleType:
    """The registry module of a layout name: "v1" (eo_vocab, the default) or "nc" (this module)."""
    if name in (None, "", "v1"):
        return _v1
    if name == NAME:
        import sys
        return sys.modules[__name__]
    raise ValueError(f"unknown EO vocabulary layout {name!r}: v1 (eo_vocab) or nc (eo_vocab_nc)")


def for_vocab_size(vocab_size: int) -> ModuleType:
    """The layout a checkpoint was built for, from its vocab_size: 87,556 -> v1, 81,188 -> nc. Refuses others."""
    for mod in (_v1, layout(NAME)):
        if int(vocab_size) == mod.TOTAL_VOCAB_SIZE:
            return mod
    raise ValueError(f"vocab_size {vocab_size} is neither eo_vocab's {_v1.TOTAL_VOCAB_SIZE} (v1) nor "
                     f"{TOTAL_VOCAB_SIZE} (nc): not a TerraMesh pretraining checkpoint")
