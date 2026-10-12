# eo/data/ben_vocab.py
"""The BigEarthNet label vocabulary: 21 ids APPENDED after the EO vocabulary (Phase 6 D6.3).

    [0, 87,556)        eo_vocab, untouched: codebooks, BO/EO pairs, PAD = 87,555
    87,556 + c         BEN class c, c in 0..18, in the canonical 19-class order
                       (eo/ben/labels.NEW_LABELS, SMARTIES' released order)
    87,575             BO_BEN
    87,576             EO_BEN
                       TOTAL_VOCAB_SIZE = 87,577

WHY APPEND, AND WHY A SEPARATE MODULE. `eo_vocab.py` is final and a one-way door: A10/B10
are trained against its ids, and inserting anything before PAD shifts PAD (and every BO/EO
id if a modality were appended). So the BEN ids go strictly AFTER PAD, every pretraining id
maps to itself, and eo_vocab.py is not edited. Gianfranco's rule, verbatim: "do not have
token:8 for pasture and token:8 for DEM token" -- no BEN id coincides with any existing id.

PAD stays 87,555. It is no longer the last id, which is fine: nothing in the training path
assumes PAD == TOTAL_VOCAB_SIZE - 1 (eo_vocab asserts it for ITS layout only).

The label block is `BO_BEN c_1 ... c_k EO_BEN`: present classes only, ascending in the
canonical order, always LAST in a sequence (BEN is a target, never an input).

Registry surface. `lm_dataset/modality_registry.py` resolves dataset `ben_multimodal` to
this module, which exposes the same names as eo_vocab (`MODALITY_TO_ID`, `ID_TO_MODALITY`,
`TOTAL_VOCAB_SIZE`, `PAD_ID`, `get_modality`). The 7 EO modality ids are unchanged and BEN
is id 8, so per-modality logging gains `loss_BEN` and nothing else moves.

⚠ As with eo_vocab, import-time asserts verify the DERIVATION, not the sizes. The class
count is pinned against `eo/ben/labels.N_CLASSES`, which is pinned against SMARTIES' code.

Runs in `.venv` (stdlib + eo_vocab + eo.ben.labels only).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from eo.ben import labels as _labels
from eo.data import eo_vocab as _eo
from eo.data.eo_vocab import ModalityInfo, get_modality as _eo_get_modality

BASE = _eo.TOTAL_VOCAB_SIZE                 # 87,556: the first id that does not exist yet
N_CLASSES = _labels.N_CLASSES               # 19
CLASS_IDS: List[int] = [BASE + c for c in range(N_CLASSES)]
BO_BEN = BASE + N_CLASSES
EO_BEN = BASE + N_CLASSES + 1
TOTAL_VOCAB_SIZE = BASE + N_CLASSES + 2     # 87,577
N_NEW = TOTAL_VOCAB_SIZE - BASE             # 21
PAD_ID = _eo.PAD_ID                         # unchanged: 87,555

# The label "modality". tokens_per_sample is the MAXIMUM body length (all 19 classes).
BEN = ModalityInfo(name="BEN", codebook_size=N_CLASSES, data_type="labels",
                   tokens_per_sample=N_CLASSES, codebook_offset=BASE,
                   bo_id=BO_BEN, eo_id=EO_BEN)

MODALITIES: Dict[str, ModalityInfo] = dict(_eo.MODALITIES)
MODALITIES["BEN"] = BEN
MODALITY_TO_ID: Dict[str, int] = dict(_eo.MODALITY_TO_ID)
MODALITY_TO_ID["BEN"] = max(_eo.MODALITY_TO_ID.values()) + 1
ID_TO_MODALITY: Dict[int, str] = {i: n for n, i in MODALITY_TO_ID.items()}

# BEN inputs: the two TerraMesh slots BEN's S1/S2 are tokenized into. S1 goes through
# the S1GRD tokenizer (PHASE6_PLAN §4.3, confirmed by the Step 1.3 round trip).
INPUT_MODALITIES: Dict[str, str] = {"S2": "S2L2A", "S1": "S1GRD"}

# --- Import-time invariants ------------------------------------------------
def layout_problems(class_ids, bo_ben: int, eo_ben: int, total: int,
                    eo_total: int = _eo.TOTAL_VOCAB_SIZE, pad_id: int = _eo.PAD_ID) -> List[str]:
    """Everything wrong with a proposed BEN layout. Empty = valid. Gate X2 feeds it broken ones."""
    out = []
    new = list(class_ids) + [bo_ben, eo_ben]
    if len(set(new)) != len(new):
        out.append("BEN ids are not distinct")
    clash = sorted(i for i in new if 0 <= i < eo_total)
    if clash:
        out.append(f"{len(clash)} BEN ids collide with pretraining ids (e.g. {clash[:3]}): the 'token:8' rule")
    if pad_id != eo_total - 1:
        out.append(f"PAD moved: {pad_id} != {eo_total - 1}")
    if sorted(new) != list(range(eo_total, eo_total + len(new))) or total != eo_total + len(new):
        out.append("BEN ids are not the contiguous block right after the pretraining vocabulary")
    return out


assert N_CLASSES == 19, "BEN-19: the class count is pinned by SMARTIES' nomenclature"
assert N_NEW == 21, "19 classes + BO_BEN + EO_BEN"
assert not layout_problems(CLASS_IDS, BO_BEN, EO_BEN, TOTAL_VOCAB_SIZE), \
    layout_problems(CLASS_IDS, BO_BEN, EO_BEN, TOTAL_VOCAB_SIZE)
# Identity on every pretraining id: the EO layout this module sees is the layout eo_vocab
# derives, slot for slot, so nothing trained against eo_vocab moves.
_m, _pad, _tot = _eo.build_vocab()
assert _pad == PAD_ID and _tot == BASE
for _n, _i in _m.items():
    _j = MODALITIES[_n]
    assert (_i.codebook_offset, _i.codebook_size, _i.bo_id, _i.eo_id) == \
           (_j.codebook_offset, _j.codebook_size, _j.bo_id, _j.eo_id), f"{_n} moved"
assert all(MODALITY_TO_ID[n] == i for n, i in _eo.MODALITY_TO_ID.items()), "an EO modality id moved"
del _m, _pad, _tot, _n, _i, _j


def get_modality(name: str) -> ModalityInfo:
    if name == "BEN":
        return BEN
    return _eo_get_modality(name)


def label_block(classes: Iterable[int]) -> List[int]:
    """Sorted, de-duplicated class indices -> [BO_BEN, ids..., EO_BEN]."""
    cs = sorted(set(int(c) for c in classes))
    if any(c < 0 or c >= N_CLASSES for c in cs):
        raise ValueError(f"class index out of range: {cs}")
    return [BO_BEN] + [BASE + c for c in cs] + [EO_BEN]


def decode_label_block(ids: Sequence[int], strict: bool = True) -> List[int]:
    """[BO_BEN?] ids... EO_BEN -> sorted class indices.

    strict: the body must be BEN class ids in strictly ascending order, ending at EO_BEN.
    Non-strict: keep every class id seen before the first EO_BEN (generation scoring).
    """
    seq = list(int(t) for t in ids)
    if seq and seq[0] == BO_BEN:
        seq = seq[1:]
    if EO_BEN in seq:
        body = seq[: seq.index(EO_BEN)]
    elif strict:
        raise ValueError("label block has no EO_BEN")
    else:
        body = seq
    cls = [t - BASE for t in body if BASE <= t < BASE + N_CLASSES]
    if strict:
        if len(cls) != len(body):
            raise ValueError(f"non-class id inside a label block: {body}")
        if any(b <= a for a, b in zip(cls, cls[1:])):
            raise ValueError(f"label block not strictly ascending: {cls}")
    return sorted(set(cls))
