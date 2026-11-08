# eo/data/temporal_vocab.py
"""The date vocabulary of the temporal stage: 87 ids APPENDED after BEN's (`.venv`).

    [0, 87,556)         eo_vocab, untouched: codebooks, BO/EO pairs, PAD = 87,555
    [87,556, 87,577)    ben_vocab's 21 ids, unchanged. No data during pretraining
    87,577 + y          Year, y in 0..11, temporal_index's year id: 2015..2026
    87,589 + d          Day bin, d in 0..72, temporal_index's day bin
    87,662              BO_Date
    87,663              EO_Date
                        TOTAL_VOCAB_SIZE = 87,664

WHY AFTER BEN, AND WHY A SEPARATE MODULE. As with ben_vocab: every id below the new block means
what it always meant, so eo_vocab.py and ben_vocab.py stay as they are, PAD keeps its id, and every
existing checkpoint, tool and BEN report stays valid. Later tasks append after 87,663.

⚠ A ONE-WAY DOOR, once anything has trained against these ids. eo/scripts/extend_checkpoint.py
--layout temporal writes the rows; until a run has trained on them, the layout can still change.

A DATE BLOCK is `BO_Date Y D EO_Date`, or `BO_Date D EO_Date` with the year dropped. Y and D come
from temporal_index's `dates.npy`, which holds positions in their own ranges, not vocabulary ids.
This module is the only place that adds the offsets, and it casts first: those arrays are uint8,
and an offset near 87,589 would wrap silently. The date is the row's S2 date (temporal_index.py).

ROWS WITH NO DATA. The embedding table and lm_head are one tensor, so a row that never occurs as a
target is only ever pushed down during pretraining. That covers BEN's 21 rows, and the three years
no row of train or val falls in. `UNTRAINED_IDS` names them: reset them to mean + noise
(eo/finetune/extend.py) before a task uses any of them. `coverage_problems` holds the list to a
dates.npy.

Registry surface. `lm_dataset/modality_registry.py` resolves dataset `terramesh_temporal` to this
module, which exposes the same names as eo_vocab (`MODALITY_TO_ID`, `ID_TO_MODALITY`,
`TOTAL_VOCAB_SIZE`, `PAD_ID`, `get_modality`). The 7 EO modality ids and BEN's 8 are unchanged and
Date is 9, so per-modality logging gains `loss_Date`. BO_Date and EO_Date get modality id 0 from
assemble_sequence, as every BO and EO does.

⚠ As with eo_vocab, import-time asserts verify the DERIVATION. The sizes are pinned here and in
temporal_index.py.

Runs in `.venv`, and in `mor-eo` on RCP: temporal_index brings numpy and pandas, never terratorch.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from eo.data import ben_vocab as _ben
from eo.data import eo_vocab as _eo
from eo.data.eo_vocab import ModalityInfo
from eo.data.temporal_index import N_DOY_BINS, N_YEARS, YEAR_FIRST

BASE = _ben.TOTAL_VOCAB_SIZE                # 87,577: the first id that does not exist yet
YEAR_BASE = BASE
DAY_BASE = YEAR_BASE + N_YEARS              # 87,589
BO_DATE = DAY_BASE + N_DOY_BINS             # 87,662
EO_DATE = BO_DATE + 1                       # 87,663
TOTAL_VOCAB_SIZE = EO_DATE + 1              # 87,664
N_NEW = TOTAL_VOCAB_SIZE - BASE             # 87
PAD_ID = _eo.PAD_ID                         # unchanged: 87,555

YEAR_IDS: List[int] = [YEAR_BASE + y for y in range(N_YEARS)]
DAY_IDS: List[int] = [DAY_BASE + d for d in range(N_DOY_BINS)]

# The label "modality". tokens_per_sample is the MAXIMUM body length, year and day.
DATE = ModalityInfo(name="Date", codebook_size=N_YEARS + N_DOY_BINS, data_type="date",
                    tokens_per_sample=2, codebook_offset=YEAR_BASE, bo_id=BO_DATE, eo_id=EO_DATE)

MODALITIES: Dict[str, ModalityInfo] = dict(_ben.MODALITIES)
MODALITIES["Date"] = DATE
MODALITY_TO_ID: Dict[str, int] = dict(_ben.MODALITY_TO_ID)
MODALITY_TO_ID["Date"] = max(_ben.MODALITY_TO_ID.values()) + 1
ID_TO_MODALITY: Dict[int, str] = {i: n for n, i in MODALITY_TO_ID.items()}

UNTRAINED_YEARS: Tuple[int, ...] = (2015, 2025, 2026)     # no row of train or val
UNTRAINED_IDS: Tuple[int, ...] = (tuple(range(_ben.BASE, _ben.TOTAL_VOCAB_SIZE))
                                  + tuple(YEAR_BASE + y - YEAR_FIRST for y in UNTRAINED_YEARS))


# --- Import-time invariants ------------------------------------------------
def layout_problems(year_ids: Sequence[int], day_ids: Sequence[int], bo_date: int, eo_date: int, total: int,
                    ben_total: int = _ben.TOTAL_VOCAB_SIZE, pad_id: int = _eo.PAD_ID) -> List[str]:
    """Everything wrong with a proposed date layout. Empty = valid. Gate Y1 feeds it broken ones."""
    out = []
    new = list(year_ids) + list(day_ids) + [bo_date, eo_date]
    if len(set(new)) != len(new):
        out.append("date ids are not distinct")
    clash = sorted(i for i in new if 0 <= i < ben_total)
    if clash:
        out.append(f"{len(clash)} date ids collide with eo_vocab's or BEN's (e.g. {clash[:3]})")
    if pad_id != _eo.TOTAL_VOCAB_SIZE - 1:
        out.append(f"PAD moved: {pad_id} != {_eo.TOTAL_VOCAB_SIZE - 1}")
    if len(year_ids) != N_YEARS or len(day_ids) != N_DOY_BINS:
        out.append(f"{len(year_ids)} years and {len(day_ids)} day bins, not {N_YEARS} and {N_DOY_BINS}")
    if sorted(new) != list(range(ben_total, ben_total + len(new))) or total != ben_total + len(new):
        out.append("date ids are not the contiguous block right after BEN's")
    return out


assert N_NEW == 87, "12 years + 73 day bins + BO_Date + EO_Date"
assert TOTAL_VOCAB_SIZE == 87_664, "the size every temporal config and checkpoint carries"
assert not layout_problems(YEAR_IDS, DAY_IDS, BO_DATE, EO_DATE, TOTAL_VOCAB_SIZE), \
    layout_problems(YEAR_IDS, DAY_IDS, BO_DATE, EO_DATE, TOTAL_VOCAB_SIZE)
# Identity on every id below BASE: nothing this module sees has moved from what BEN laid out.
for _n, _i in _ben.MODALITIES.items():
    assert MODALITIES[_n] == _i, f"{_n} moved"
assert all(MODALITY_TO_ID[n] == i for n, i in _ben.MODALITY_TO_ID.items()), "a modality id moved"
del _n, _i


def get_modality(name: str) -> ModalityInfo:
    if name == "Date":
        return DATE
    return _ben.get_modality(name)


def date_block(year_id: int, day_bin: int, with_year: bool = True) -> List[int]:
    """dates.npy's two values -> [BO_Date, Y, D, EO_Date], or [BO_Date, D, EO_Date]."""
    y, d = int(year_id), int(day_bin)
    if not 0 <= y < N_YEARS:
        raise ValueError(f"year id {y} is outside 0..{N_YEARS - 1}")
    if not 0 <= d < N_DOY_BINS:
        raise ValueError(f"day bin {d} is outside 0..{N_DOY_BINS - 1}")
    body = [YEAR_BASE + y, DAY_BASE + d] if with_year else [DAY_BASE + d]
    return [BO_DATE] + body + [EO_DATE]


def decode_date_block(ids: Sequence[int]) -> Tuple[Optional[int], int]:
    """date_block's inverse: (year id or None, day bin). Anything else raises."""
    seq = [int(t) for t in ids]
    if len(seq) < 3 or seq[0] != BO_DATE or seq[-1] != EO_DATE:
        raise ValueError(f"not a date block: {seq}")
    body = seq[1:-1]
    if len(body) == 1 and body[0] in DAY_IDS:
        return None, body[0] - DAY_BASE
    if len(body) == 2 and body[0] in YEAR_IDS and body[1] in DAY_IDS:
        return body[0] - YEAR_BASE, body[1] - DAY_BASE
    raise ValueError(f"not a date block: {seq}")


def coverage_problems(year_ids: Iterable[int], untrained: Sequence[int] = UNTRAINED_YEARS) -> List[str]:
    """`year_ids`: the year ids a dates.npy holds. Empty = UNTRAINED_YEARS describes that array."""
    seen = {YEAR_FIRST + int(y) for y in year_ids}
    out = []
    has_rows = sorted(seen & set(untrained))
    if has_rows:
        out.append(f"years listed as untrained have rows: {has_rows}")
    unlisted = sorted(set(range(YEAR_FIRST, YEAR_FIRST + N_YEARS)) - seen - set(untrained))
    if unlisted:
        out.append(f"years with no rows are not listed as untrained: {unlisted}")
    return out


def layout_record() -> Dict:
    """What extension.json keeps, so that a base names the layout it was written for."""
    return {"base": BASE, "year_base": YEAR_BASE, "n_years": N_YEARS, "year_first": YEAR_FIRST,
            "day_base": DAY_BASE, "n_day_bins": N_DOY_BINS, "bo_date": BO_DATE, "eo_date": EO_DATE,
            "total": TOTAL_VOCAB_SIZE, "pad_id": PAD_ID, "untrained_ids": list(UNTRAINED_IDS)}
