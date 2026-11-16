# eo/data/temporal_vocab_nc.py
"""The temporal stage's vocabulary on the Coords-free layout "nc": temporal_vocab's layout, appended to
eo_vocab_nc instead of eo_vocab (`.venv`).

    [0, 81,188)          eo_vocab_nc, untouched: the six image codebooks, their BO/EO pairs, PAD = 81,187
    [81,188, 81,209)     21 ids RESERVED for BEN, in ben_vocab's order (19 classes, BO_BEN, EO_BEN). No data
    81,209 + y           Year, y in 0..11, temporal_index's year id: 2015..2026
    81,221 + d           Day bin, d in 0..72, temporal_index's day bin
    81,294               BO_Date
    81,295               EO_Date
                         TOTAL_VOCAB_SIZE = 81,296

WHY THE SAME SHAPE AS temporal_vocab. v1's date rows sit after BEN's 21 ids, because BEN existed first. No
BEN layout exists on nc yet, but its 21 rows are reserved here at nc + 0..20, where ben_vocab puts them on
v1, so that a BEN fine-tune of an nc temporal base finds them where a future nc BEN layout would, and
everything built for v1's temporal base (extend_checkpoint's 108 appended rows, UNTRAINED_IDS, the reset
before a task uses a row) applies unchanged. Modality ids are v1's: the six image modalities 1..6, BEN 8,
Date 9. Coords' 7 does not exist here.

⚠ A ONE-WAY DOOR, once anything has trained against these ids (as temporal_vocab). The layout is written
into an extended base's extension.json (`layout_record`, which names `eo_layout: nc`), and
eo/train/clevr_init.verify_init refuses a base whose record is not this module's as it is now.

⚠ An nc sequence fed to a v1 temporal model does not raise (its ids are below 87,664): take the layout
from the checkpoint's vocab_size (`for_vocab_size`), never from the data alone.

Same public surface as temporal_vocab: MODALITY_TO_ID, ID_TO_MODALITY, TOTAL_VOCAB_SIZE, PAD_ID,
get_modality, date_block, decode_date_block, layout_record, UNTRAINED_IDS, coverage_problems.

Runs in `.venv`, and in `mor-eo` on RCP: stdlib, the contract, eo_vocab_nc, ben_vocab and temporal_index's
constants. Import does no file I/O.
"""
from __future__ import annotations

from types import ModuleType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from eo.data import ben_vocab as _ben
from eo.data import eo_vocab_nc as _nc
from eo.data import temporal_vocab as _tv1
from eo.data.eo_vocab import ModalityInfo
from eo.data.temporal_index import N_DOY_BINS, N_YEARS, YEAR_FIRST

NAME = "nc"
EXTENSION_LAYOUT = "temporal_nc"            # extension.json's `layout` for a base of this layout

BEN_BASE = _nc.TOTAL_VOCAB_SIZE             # 81,188: the first id eo_vocab_nc does not have
N_BEN = _ben.N_NEW                          # 21, reserved: 19 classes, BO_BEN, EO_BEN
BASE = BEN_BASE + N_BEN                     # 81,209: the first date id
YEAR_BASE = BASE
DAY_BASE = YEAR_BASE + N_YEARS              # 81,221
BO_DATE = DAY_BASE + N_DOY_BINS             # 81,294
EO_DATE = BO_DATE + 1                       # 81,295
TOTAL_VOCAB_SIZE = EO_DATE + 1              # 81,296
N_NEW = TOTAL_VOCAB_SIZE - BASE             # 87 date ids
N_APPENDED = TOTAL_VOCAB_SIZE - BEN_BASE    # 108 rows an extended base appends, as on v1
PAD_ID = _nc.PAD_ID                         # unchanged: 81,187

YEAR_IDS: List[int] = [YEAR_BASE + y for y in range(N_YEARS)]
DAY_IDS: List[int] = [DAY_BASE + d for d in range(N_DOY_BINS)]

# BEN's reserved block, laid out as ben_vocab lays it out on v1, so a future nc BEN layout can be held to it.
BEN = ModalityInfo(name="BEN", codebook_size=_ben.N_CLASSES, data_type="labels",
                   tokens_per_sample=_ben.N_CLASSES, codebook_offset=BEN_BASE,
                   bo_id=BEN_BASE + _ben.N_CLASSES, eo_id=BEN_BASE + _ben.N_CLASSES + 1)
# The label "modality". tokens_per_sample is the MAXIMUM body length, year and day.
DATE = ModalityInfo(name="Date", codebook_size=N_YEARS + N_DOY_BINS, data_type="date",
                    tokens_per_sample=2, codebook_offset=YEAR_BASE, bo_id=BO_DATE, eo_id=EO_DATE)

MODALITIES: Dict[str, ModalityInfo] = dict(_nc.MODALITIES)
MODALITIES["BEN"] = BEN
MODALITIES["Date"] = DATE
MODALITY_TO_ID: Dict[str, int] = dict(_nc.MODALITY_TO_ID)
MODALITY_TO_ID["BEN"] = _tv1.MODALITY_TO_ID["BEN"]          # 8, as on v1
MODALITY_TO_ID["Date"] = _tv1.MODALITY_TO_ID["Date"]        # 9, as on v1: per-modality logging keeps loss_Date
ID_TO_MODALITY: Dict[int, str] = {i: n for n, i in MODALITY_TO_ID.items()}
IMAGE_MODALITIES: List[str] = list(_nc.IMAGE_MODALITIES)

UNTRAINED_YEARS: Tuple[int, ...] = _tv1.UNTRAINED_YEARS    # (2015, 2025, 2026): no row of train or val
UNTRAINED_IDS: Tuple[int, ...] = (tuple(range(BEN_BASE, BASE))
                                  + tuple(YEAR_BASE + y - YEAR_FIRST for y in UNTRAINED_YEARS))


# --- Import-time invariants ------------------------------------------------
def layout_problems(year_ids: Sequence[int], day_ids: Sequence[int], bo_date: int, eo_date: int, total: int,
                    reserved_end: int = BASE, nc_total: int = _nc.TOTAL_VOCAB_SIZE,
                    pad_id: int = _nc.PAD_ID) -> List[str]:
    """Everything wrong with a proposed nc date layout. Empty = valid. The gate feeds it broken ones."""
    out = []
    new = list(year_ids) + list(day_ids) + [bo_date, eo_date]
    if len(set(new)) != len(new):
        out.append("date ids are not distinct")
    clash = sorted(i for i in new if 0 <= i < reserved_end)
    if clash:
        out.append(f"{len(clash)} date ids collide with eo_vocab_nc's or the reserved BEN ids (e.g. {clash[:3]})")
    if pad_id != nc_total - 1:
        out.append(f"PAD moved: {pad_id} != {nc_total - 1}")
    if reserved_end != nc_total + N_BEN:
        out.append(f"the reserved BEN block ends at {reserved_end}, not nc's {nc_total} + {N_BEN}")
    if len(year_ids) != N_YEARS or len(day_ids) != N_DOY_BINS:
        out.append(f"{len(year_ids)} years and {len(day_ids)} day bins, not {N_YEARS} and {N_DOY_BINS}")
    if sorted(new) != list(range(reserved_end, reserved_end + len(new))) or total != reserved_end + len(new):
        out.append("date ids are not the contiguous block right after the reserved BEN ids")
    return out


assert N_NEW == _tv1.N_NEW == 87, "12 years + 73 day bins + BO_Date + EO_Date, as on v1"
assert N_APPENDED == _tv1.TOTAL_VOCAB_SIZE - _tv1._eo.TOTAL_VOCAB_SIZE == 108, "BEN's 21 + 87, as on v1"
assert TOTAL_VOCAB_SIZE == 81_296 and PAD_ID == 81_187, "the size every nc temporal config and checkpoint carries"
assert not layout_problems(YEAR_IDS, DAY_IDS, BO_DATE, EO_DATE, TOTAL_VOCAB_SIZE), \
    layout_problems(YEAR_IDS, DAY_IDS, BO_DATE, EO_DATE, TOTAL_VOCAB_SIZE)
# Every id below BEN_BASE is eo_vocab_nc's, unmoved; and the reserved block is BEN's v1 shape, shifted.
for _n, _i in _nc.MODALITIES.items():
    assert MODALITIES[_n] == _i, f"{_n} moved"
assert all(MODALITY_TO_ID[n] == i for n, i in _nc.MODALITY_TO_ID.items()), "a modality id moved"
assert "Coords" not in MODALITIES and "Coords" not in MODALITY_TO_ID, "nc holds no Coords"
assert (BEN.bo_id - BEN.codebook_offset, BEN.eo_id - BEN.codebook_offset) == \
    (_ben.BO_BEN - _ben.BASE, _ben.EO_BEN - _ben.BASE), "the reserved block is not ben_vocab's shape"
# The date block's shape is v1's, shifted: the same offsets from the first date id.
assert (DAY_BASE - BASE, BO_DATE - BASE, EO_DATE - BASE) == \
    (_tv1.DAY_BASE - _tv1.BASE, _tv1.BO_DATE - _tv1.BASE, _tv1.EO_DATE - _tv1.BASE), "not v1's date shape"
del _n, _i


def get_modality(name: str) -> ModalityInfo:
    if name in ("Date", "BEN"):
        return MODALITIES[name]
    return _nc.get_modality(name)          # refuses Coords, with the reason


def date_block(year_id: int, day_bin: int, with_year: bool = True) -> List[int]:
    """dates.npy's two values -> [BO_Date, Y, D, EO_Date], or [BO_Date, D, EO_Date]. Casts first: the
    arrays are uint8, and an offset near 81,221 would wrap silently."""
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
    """temporal_vocab.coverage_problems: the year rows are the same years on either layout."""
    return _tv1.coverage_problems(year_ids, untrained)


def layout_record() -> Dict:
    """What extension.json keeps, so that a base names the layout it was written for. v1's record has no
    `eo_layout` key and lower ids, so neither record can pass for the other."""
    return {"eo_layout": NAME, "ben_base": BEN_BASE, "base": BASE, "year_base": YEAR_BASE, "n_years": N_YEARS,
            "year_first": YEAR_FIRST, "day_base": DAY_BASE, "n_day_bins": N_DOY_BINS, "bo_date": BO_DATE,
            "eo_date": EO_DATE, "total": TOTAL_VOCAB_SIZE, "pad_id": PAD_ID, "untrained_ids": list(UNTRAINED_IDS)}


# --- The two temporal layouts by name or size ------------------------------
def layout(name: Optional[str]) -> ModuleType:
    """The temporal registry of an EO layout name: "v1" (temporal_vocab, the default) or "nc" (this module)."""
    if name in (None, "", "v1"):
        return _tv1
    if name == NAME:
        import sys
        return sys.modules[__name__]
    raise ValueError(f"unknown EO vocabulary layout {name!r}: v1 (temporal_vocab) or nc (temporal_vocab_nc)")


def for_vocab_size(vocab_size: int) -> ModuleType:
    """The temporal layout a checkpoint was built for: 87,664 -> v1, 81,296 -> nc. Refuses any other size."""
    for mod in (_tv1, layout(NAME)):
        if int(vocab_size) == mod.TOTAL_VOCAB_SIZE:
            return mod
    raise ValueError(f"vocab_size {vocab_size} is neither temporal_vocab's {_tv1.TOTAL_VOCAB_SIZE} (v1) nor "
                     f"{TOTAL_VOCAB_SIZE} (nc): not a temporal-stage checkpoint")


def extension_layout(mod: ModuleType) -> str:
    """extension.json's `layout` value for a base of this temporal registry: "temporal" (v1) or "temporal_nc"."""
    return getattr(mod, "EXTENSION_LAYOUT", "temporal")
