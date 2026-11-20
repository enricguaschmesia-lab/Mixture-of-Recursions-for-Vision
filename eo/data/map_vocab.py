# eo/data/map_vocab.py
"""A2's map vocabulary: RGB-map tokens appended after a base's last id (A2 Step 3.1, `.venv`).

    [0, B)                 the base, untouched. B is the checkpoint's vocab size: 81,296 for A2's base (the
                           nc temporal layout, eo/data/temporal_vocab_nc.py; plan decision 12)
    B + c                  the ALIGNED slot, c in 0..63,999   (b224_64000_long: 196 tokens, 14 x 14)
    B + 64,000 + c         the SHORT slot,   c in 0..15,359   (b256_15360_long: 256 tokens, 16 x 16)
    B + 79,360 + c         the LONG slot,    c in 0..999      (c1000_256L: 1,024 tokens, 32 x 32)
    B + 80,360             EO_MAP, shared by every task and arm
    B + 80,361 + t         BO_MAP_<task>, t in LULC, PASTIS, CLOUDSEN12, KUROSIWO, BURNSCARS, SPARE1..3
                           TOTAL = B + 80,369 (161,665 on A2's base)

WHY ONE SLOT PER ARM. Step 4.0 picks one of three frozen map tokenizers (eo/maps/frozen.ARMS) by an R0
fine-tune of each. Giving each arm its own slot in one layout lets the three fine-tunes start from ONE
extended base with one identity check, and keeps any arm's ids where they are after the pick. A slot's codes
mean that arm's codes only; the other two slots' rows stay untrained in that arm's runs and decoding is
constrained to the arm's slot (eo/eval/seg.py).

WHY PER-TASK BO AND ONE EO. `BO_MAP_<task>` is the task token (plan decision 8, "Task tokens"): it tells the
model which palette assignment to write. The block ends the same way for every task.

THE MODALITY. Map codes take modality id MAP_MODALITY_ID (10: the base's ids are the image modalities 1..6,
BEN 8 and Date 9), so per-modality logging gains `loss_MAP`. BO_MAP and EO_MAP get 0, like every BO/EO.

DERIVED FROM THE BASE. `for_base(vocab_size)` builds the layout over any temporal base the repo knows
(temporal_vocab_nc.for_vocab_size: 81,296 nc, 87,664 v1). The module-level names are the layout over A2's base
(nc, 81,296), and are what lm_dataset/modality_registry resolves for dataset `geobench_seg`.

⚠ A ONE-WAY DOOR once anything trains on map ids (R0's first fine-tune or R1's mid-training). Rows are
appended, never inserted: every base id maps to itself (gate Z4, and the extension's identity check).

Runs in `.venv` and `a2seg`: stdlib, the contract-free vocab modules. Import does no file I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, List, Sequence, Tuple

from eo.data import temporal_vocab_nc as _tvnc
from eo.data.eo_vocab import ModalityInfo

#: Step 4.0's arms, in slot order: (codes, tokens per map, grid side). eo/maps/frozen.ARMS names their files.
ARMS: Dict[str, Tuple[int, int, int]] = {"aligned": (64_000, 196, 14), "short": (15_360, 256, 16),
                                         "long": (1_000, 1_024, 32)}
#: The task tokens, in id order. Three spares, so a task added later does not move EO_MAP or a slot.
TASKS: Tuple[str, ...] = ("LULC", "PASTIS", "CLOUDSEN12", "KUROSIWO", "BURNSCARS", "SPARE1", "SPARE2", "SPARE3")
#: eo/maps' dataset names -> the task token.
DATASET_TASK = {"lulc": "LULC", "pastis": "PASTIS", "cloudsen12": "CLOUDSEN12", "kurosiwo": "KUROSIWO",
                "burnscars": "BURNSCARS"}
MAP_MODALITY_ID = 10
MAP = "MAP"
N_CODES = sum(c for c, _, _ in ARMS.values())        # 80,360
N_NEW = N_CODES + 1 + len(TASKS)                     # 80,369


@dataclass(frozen=True)
class MapLayout:
    """The map ids over one base. `base` is the base's registry module (a temporal layout)."""
    base_vocab: int
    base: ModuleType = field(compare=False, repr=False)
    slots: Dict[str, Tuple[int, int]] = field(compare=False)      # arm -> (offset, size)
    eo_map: int = 0
    bo_map: Dict[str, int] = field(default_factory=dict, compare=False)
    total: int = 0

    @property
    def pad_id(self) -> int:
        return int(self.base.PAD_ID)

    @property
    def n_new(self) -> int:
        return self.total - self.base_vocab

    def slot(self, arm: str) -> Tuple[int, int]:
        if arm not in self.slots:
            raise KeyError(f"unknown map arm {arm!r}: {sorted(self.slots)}")
        return self.slots[arm]

    def bo(self, task: str) -> int:
        t = DATASET_TASK.get(task, task)
        if t not in self.bo_map:
            raise KeyError(f"unknown map task {task!r}: {list(self.bo_map)}")
        return self.bo_map[t]

    def map_block(self, arm: str, task: str, codes: Sequence[int]) -> List[int]:
        """Raw codes (0..size-1) -> [BO_MAP_<task>, ids..., EO_MAP]. Casts each code to int first: the token
        arrays are uint16, and an offset past 65,535 would wrap silently."""
        off, size = self.slot(arm)
        n_tok = ARMS[arm][1]
        cs = [int(c) for c in codes]
        if len(cs) != n_tok:
            raise ValueError(f"arm {arm} writes {n_tok} tokens a map, got {len(cs)}")
        if min(cs) < 0 or max(cs) >= size:
            raise ValueError(f"a code outside arm {arm}'s 0..{size - 1}: {min(cs)}..{max(cs)}")
        return [self.bo(task)] + [off + c for c in cs] + [self.eo_map]

    def decode_block(self, arm: str, ids: Sequence[int]) -> List[int]:
        """map_block's body -> raw codes. Refuses an id outside the arm's slot."""
        off, size = self.slot(arm)
        out = []
        for t in ids:
            c = int(t) - off
            if not 0 <= c < size:
                raise ValueError(f"id {int(t)} is outside arm {arm}'s slot [{off}, {off + size})")
            out.append(c)
        return out

    def record(self) -> Dict:
        """What extension.json keeps under `map_vocab`, so a base names the layout it was written for."""
        return {"base_vocab": self.base_vocab, "base_layout": getattr(self.base, "NAME", "v1"),
                "slots": {a: list(v) for a, v in self.slots.items()}, "eo_map": self.eo_map,
                "bo_map": dict(self.bo_map), "total": self.total, "pad_id": self.pad_id,
                "modality_id": MAP_MODALITY_ID, "arms": {a: list(v) for a, v in ARMS.items()}}


def build(base_vocab: int, base: ModuleType) -> MapLayout:
    off, slots = int(base_vocab), {}
    for arm, (codes, _, _) in ARMS.items():
        slots[arm] = (off, codes)
        off += codes
    eo_map = off
    bo_map = {t: eo_map + 1 + i for i, t in enumerate(TASKS)}
    return MapLayout(base_vocab=int(base_vocab), base=base, slots=slots, eo_map=eo_map, bo_map=bo_map,
                     total=eo_map + 1 + len(TASKS))


def for_base(vocab_size: int) -> MapLayout:
    """The map layout over a temporal base of this size (81,296 nc or 87,664 v1). Refuses any other size."""
    return build(int(vocab_size), _tvnc.for_vocab_size(int(vocab_size)))


def for_total(total: int) -> MapLayout:
    """The map layout whose TOTAL is this (a fine-tuned checkpoint's vocab size)."""
    return for_base(int(total) - N_NEW)


def layout_problems(lay: MapLayout) -> List[str]:
    """Everything wrong with a proposed map layout. Empty = valid. Gate Z4 feeds it broken ones."""
    out = []
    new = []
    for arm, (o, s) in lay.slots.items():
        new += list(range(o, o + s))
    new += [lay.eo_map] + list(lay.bo_map.values())
    if len(set(new)) != len(new):
        out.append("map ids are not distinct")
    clash = [i for i in (min(new), ) if i < lay.base_vocab]
    if clash:
        out.append(f"map ids start at {min(new)}, inside the base's {lay.base_vocab} ids")
    if sorted(new) != list(range(lay.base_vocab, lay.base_vocab + len(new))) or lay.total != lay.base_vocab + len(new):
        out.append("map ids are not the contiguous block right after the base's last id")
    if lay.pad_id >= lay.base_vocab:
        out.append(f"PAD {lay.pad_id} is not a base id")
    if [s for _, s in lay.slots.values()] != [c for c, _, _ in ARMS.values()]:
        out.append("a slot's size is not its arm's codebook")
    return out


# --- A2's base: the module-level registry surface (dataset `geobench_seg`) -----------------------------
LAYOUT = for_base(_tvnc.TOTAL_VOCAB_SIZE)
BASE_VOCAB = LAYOUT.base_vocab                        # 81,296
TOTAL_VOCAB_SIZE = LAYOUT.total                       # 161,665
PAD_ID = LAYOUT.pad_id                                # 81,187, unchanged
EO_MAP = LAYOUT.eo_map
BO_MAP = dict(LAYOUT.bo_map)
SLOTS = dict(LAYOUT.slots)

MODALITIES: Dict[str, ModalityInfo] = dict(_tvnc.MODALITIES)
for _arm, (_o, _s) in SLOTS.items():
    MODALITIES[f"{MAP}_{_arm}"] = ModalityInfo(name=f"{MAP}_{_arm}", codebook_size=_s, data_type="map",
                                               tokens_per_sample=ARMS[_arm][1], codebook_offset=_o,
                                               bo_id=BO_MAP["PASTIS"], eo_id=EO_MAP)
MODALITY_TO_ID: Dict[str, int] = dict(_tvnc.MODALITY_TO_ID)
MODALITY_TO_ID[MAP] = MAP_MODALITY_ID
ID_TO_MODALITY: Dict[int, str] = {i: n for n, i in MODALITY_TO_ID.items()}

assert N_NEW == 80_369 and TOTAL_VOCAB_SIZE == 161_665, "64,000 + 15,360 + 1,000 codes + EO_MAP + 8 task tokens"
assert not layout_problems(LAYOUT), layout_problems(LAYOUT)
assert MAP_MODALITY_ID not in _tvnc.ID_TO_MODALITY, "the map's modality id is taken"
assert all(MODALITY_TO_ID[n] == i for n, i in _tvnc.MODALITY_TO_ID.items()), "a base modality id moved"
assert for_base(87_664).total == 87_664 + N_NEW, "the v1 temporal base (T360-T) takes the same layout"
del _arm, _o, _s


def get_modality(name: str) -> ModalityInfo:
    if name.startswith(MAP + "_"):
        return MODALITIES[name]
    return _tvnc.get_modality(name)
