# eo/data/temporal_dataset.py
"""The temporal stage's training data: whole locations as sequences, in homogeneous micro-batches
with single Major TOM scenes (`.venv`). A1 Step 4; the design is docs/plans/a1_temporal_stage.md,
"The sequence" and decisions 2, 3, 5 and 7-10.

THE SEQUENCE. One location, 2,793 tokens at four dates:

    statics   [BO_Coords lat lon eos EO_Coords] [BO_DEM body EO_DEM]        once, date 0's row
    group t   [BO_Date Y D EO_Date] [BO_S2L2A ..] [BO_S1GRD ..] [BO_NDVI ..] (+ LULC in one group)

  which dates     all four; in the ramp's first stages two or three of them, drawn uniformly
  group order     chronological 0.4; leave-one-out 0.4 (one date drawn uniformly goes last, the
                  others stay chronological); uniform 0.2
  statics         Coords and DEM together, in a random order, read from date 0's row (DEM can differ
                  by 2 tokens between rows encoded on two cards, finding 19). First in chronological
                  order; otherwise before a group drawn uniformly, or at the end
  LULC            once, in a group drawn uniformly, from that group's own row (decision 3)
  inside a group  S2L2A, S1GRD, NDVI (and LULC) shuffled; one date block, the row's S2 date
                  (decision 2): first 0.8, last 0.1, absent 0.1; the year dropped from 0.3 of them
  positions       LATTICE (decision 9). Sequential, except that after a date block the counter goes
                  back to where the block began, so image blocks keep T360's 198n and 198n + 5
                  offsets. Coords advances it as always: T360 reads Coords as a cue (finding 15).
                  `assemble_sequence` stays sequential and position_ids are rewritten after it

A MAJOR TOM SCENE is T360's layout, every present modality shuffled, plus one date block placed by
the same rule as a group's: 999 tokens. With the date absent it is token-identical to
TerraMeshTokenDataset's sequence at the same order (gate Y7), which is what keeps forgetting
comparable. Only Major TOM rows appear as scenes: an `ssl4eos12` row only ever reaches the model
inside its location's sequence (decision 5).

ONE DATASET INDEX IS ONE MICRO-BATCH (finding 6: padding scenes to 2,793 would cost ~1.7x). Either
`locations` sequences of the epoch's stage (10, 7, then 6) or 16 scenes, cut to its longest real
sequence (padding is trailing and attention causal, so no real logit moves: gate Y8). So
per_device_*_batch_size is 1, `collate_fn` unwraps the micro-batch, and total_batch_size counts
micro-batches. HF's sampler, DDP sharding and resume work untouched. An epoch is steps_per_epoch x
total_batch_size indices, an exact multiple of a step, so no partial accumulation chunk exists. A
permutation keyed on (seed, epoch) gives each epoch's indices to the two streams, and each stream
walks one fixed permutation of its pool across epochs, like BenMixDataset.unlab_pos.

⚠ HOW MICRO-BATCHES WEIGH IN A STEP. The model's forward takes **kwargs, so HF passes
num_items_in_batch: every micro-batch's loss is a SUM over its tokens divided by the label tokens of
the whole step. So each token weighs the same and a stream's share of the gradient is its share of
TOKENS, not of micro-batches. `token_share()` reports it, about 0.49 for Major TOM at four dates.

⚠ THE EPOCH COMES FROM global_step. Workers are forked once per epoch, so a step counter never
reaches them, and SetEpochCallback rounds state.epoch: on a resume past mid-epoch HF restores 5.6,
say, while its loop resumes in epoch 5 (finding 18). `TemporalEpochCallback` sets
global_step // steps_per_epoch instead. pretrain.py attaches it in place of SetEpochCallback.

⚠ AND NOTHING ELSE MAY SET IT, nor order an epoch. Measured on accelerate 1.13 (gate Y17): after a
mid-epoch resume, HF iterates `skip_first_batches(loader)`, a new DataLoaderShard whose __iter__
calls set_epoch(0), its own iteration count. That call reaches the sampler when it can, so on one
process the rest of epoch e >= 1 is replayed in epoch 0's shuffle; and otherwise (two processes, or
a sequential sampler) it reaches the DATASET's set_epoch, which would draw the rest of the epoch at
epoch 0, in stage 0. Hence the dataset has no `set_epoch` (`set_draw_epoch`, set by the callback
alone), and it orders its epoch itself (`orders_itself`): the trainer reads it with a
SequentialSampler, which a resume skips into exactly where the uninterrupted run stood.

⚠ DRAWS are keyed on (seed, epoch, location) and (seed, epoch, row), never on the worker id
(finding 7), so a micro-batch is the same at any worker count and rank (gates Y5, Y17).

⚠ VAL'S TEST LOCATIONS ARE NEVER BUILT. On val's root, `TemporalLocations` refuses any location on
the test side of temporal_rows_v1.json unless `allow_test=True`, which only Step 12 passes. A run
on val (development) trains on the 544 dev locations and the Major TOM rows outside the 4,416
held-out rows.

Every number above is a config value with this plan's number as default (`TemporalConfig`), because
the decisions may change on 11 Nov. Decisions 2 and 3 live in `_group_blocks` and
`TemporalLocations.draw`.

Runs in `.venv`, and in `mor-eo` on RCP.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback

from eo.data import temporal_vocab as TV
from eo.data.temporal_index import N_DATES, TemporalIndex, load_temporal_index
from eo.data.temporal_split import load_temporal_split
from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
from lm_dataset.sequence_assembly import assemble_sequence

KEYS = ("input_ids", "attention_mask", "labels", "position_ids", "modality_ids")   # the closed five
ORDER_MODES = ("chronological", "leave_one_out", "uniform")
DATE_MODES = ("first", "last", "absent")
GROUP_MODALITIES = ("S2L2A", "S1GRD", "NDVI")        # one block each per date group
LULC = "LULC"                                        # once per sequence (decision 3)
STATICS = ("Coords", "DEM")                          # once per sequence, from date 0's row
DATE = "Date"
SEQUENCE_MODALITIES = STATICS + GROUP_MODALITIES + (LULC,)

# RNG stream tags: location draws, scene draws, the two pools' walks, an epoch's stream assignment.
_TEMPORAL, _SINGLE, _LOC_WALK, _ROW_WALK, _ASSIGN = 0, 1, 2, 3, 4
EVAL_SEED = 20261108                                 # the in-training temporal set, drawn once


def width(modality: str) -> int:
    """Tokens of one block, BO and EO included (a date block's maximum, with its year)."""
    return TV.get_modality(modality).tokens_per_sample + 2


def sequence_length(n_dates: int) -> int:
    """5 + 198 + n * (3 * 198 + 4) + 198: 1,597, 2,195 and 2,793 at two, three and four dates."""
    group = sum(width(m) for m in GROUP_MODALITIES) + width(DATE)
    return sum(width(m) for m in STATICS) + int(n_dates) * group + width(LULC)


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Stage:
    """One stage of the length ramp (decision 10)."""
    dates: int                      # dates per sequence
    locations: int                  # locations per micro-batch
    epochs: Optional[int] = None    # epochs it lasts; None on the last stage, which runs to the end


@dataclass(frozen=True)
class TemporalConfig:
    """Step 4's settings, read from the run config's `temporal:` block. Defaults are the plan's."""
    order_probs: Tuple[float, ...] = (0.4, 0.4, 0.2)          # ORDER_MODES (decision 8)
    date_probs: Tuple[float, ...] = (0.8, 0.1, 0.1)           # DATE_MODES (decision 7)
    year_drop: float = 0.3                                    # share of date blocks without a year
    stages: Tuple[Stage, ...] = (Stage(2, 10, 1), Stage(3, 7, 1), Stage(4, 6, None))
    scenes_per_microbatch: int = 16                           # Major TOM scenes per micro-batch
    single_share: float = 0.5                                 # share of an epoch's micro-batches
    steps_per_epoch: int = 333
    single_max_length: int = 1048                             # scenes and the single-date eval set
    eval_scenes_per_microbatch: int = 16                      # 4,416 = 16 x 276
    eval_locations_per_microbatch: int = 8                    # 544 = 8 x 68, any of 1, 2, 4 GPUs
    eval_dates: int = 4
    eval_seed: int = EVAL_SEED

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping]) -> "TemporalConfig":
        """YAML spelling: order_probs and date_probs as {mode: p}, stages as a list of
        {dates, locations, epochs}. Unknown keys are refused, so a typo cannot fall back silently."""
        raw = dict(raw or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"temporal: unknown keys {unknown}; known {sorted(known)}")
        kw = {}
        for key, value in raw.items():
            if key == "order_probs":
                kw[key] = _probs(value, ORDER_MODES, key)
            elif key == "date_probs":
                kw[key] = _probs(value, DATE_MODES, key)
            elif key == "stages":
                kw[key] = tuple(Stage(**dict(s)) for s in value)
            else:
                kw[key] = value
        out = cls(**kw)
        out.validate()
        return out

    @classmethod
    def from_cfg(cls, cfg) -> "TemporalConfig":
        """The run config's `temporal:` block; absent means every default."""
        from omegaconf import OmegaConf

        raw = cfg.get("temporal", None) if hasattr(cfg, "get") else None
        if raw is not None and OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
        return cls.from_mapping(raw)

    def validate(self) -> None:
        for name, probs, modes in (("order_probs", self.order_probs, ORDER_MODES),
                                   ("date_probs", self.date_probs, DATE_MODES)):
            if len(probs) != len(modes) or min(probs) < 0 or abs(sum(probs) - 1) > 1e-9:
                raise ValueError(f"temporal.{name} must be {len(modes)} non-negative values summing to 1 "
                                 f"over {modes}, got {probs}")
        if not 0 <= self.year_drop <= 1:
            raise ValueError(f"temporal.year_drop must be in [0, 1], got {self.year_drop}")
        if not self.stages:
            raise ValueError("temporal.stages is empty")
        for k, s in enumerate(self.stages):
            last = k == len(self.stages) - 1
            if not 1 <= int(s.dates) <= N_DATES or int(s.locations) < 1:
                raise ValueError(f"temporal.stages[{k}]: dates in 1..{N_DATES} and locations >= 1, got {s}")
            if last != (s.epochs is None) or (not last and int(s.epochs) < 1):
                raise ValueError(f"temporal.stages[{k}]: every stage but the last lasts >= 1 epochs, and the "
                                 f"last runs to the end (epochs null); got {s}")
        if not 0 <= self.single_share < 1:
            raise ValueError(f"temporal.single_share must be in [0, 1), got {self.single_share}")
        for name in ("scenes_per_microbatch", "steps_per_epoch", "single_max_length",
                     "eval_scenes_per_microbatch", "eval_locations_per_microbatch"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"temporal.{name} must be >= 1")
        if not 1 <= self.eval_dates <= N_DATES:
            raise ValueError(f"temporal.eval_dates must be in 1..{N_DATES}")

    def stage_of(self, epoch: int) -> Tuple[int, Stage]:
        """(stage number, stage) of an epoch. A function of the epoch alone, never of a counter."""
        e = int(epoch)
        if e < 0:
            raise ValueError(f"epoch {epoch} < 0")
        for k, s in enumerate(self.stages):
            if s.epochs is None or e < s.epochs:
                return k, s
            e -= int(s.epochs)
        raise AssertionError("unreachable: the last stage has no end")

    def max_dates(self) -> int:
        return max(max(int(s.dates) for s in self.stages), int(self.eval_dates))

    def record(self) -> Dict:
        out = asdict(self)
        out["order_probs"] = dict(zip(ORDER_MODES, self.order_probs))
        out["date_probs"] = dict(zip(DATE_MODES, self.date_probs))
        return out


def _probs(value, modes: Sequence[str], name: str) -> Tuple[float, ...]:
    if isinstance(value, Mapping):
        extra = sorted(set(value) - set(modes))
        missing = sorted(set(modes) - set(value))
        if extra or missing:
            raise ValueError(f"temporal.{name}: name every mode of {tuple(modes)}; extra {extra}, missing {missing}")
        return tuple(float(value[m]) for m in modes)
    return tuple(float(v) for v in value)


# --------------------------------------------------------------------------- assembly
def lattice_positions(widths: Sequence[int], is_date: Sequence[bool], max_length: int) -> torch.Tensor:
    """position_ids for blocks of these widths: sequential, except that after a date block the counter
    goes back to where the block began (decision 9). Padding continues from the counter."""
    pos = torch.empty(int(max_length), dtype=torch.long)
    at = cursor = 0
    for w, d in zip(widths, is_date):
        pos[at: at + w] = torch.arange(cursor, cursor + w)
        at += w
        if not d:
            cursor += w
    pos[at:] = torch.arange(cursor, cursor + int(max_length) - at)
    return pos


def _assemble(chunks: List[torch.Tensor], modalities: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    """assemble_sequence, unchanged (the closed five keys, modality 0 for every BO and EO), then the
    lattice. Never shuffles patches: T360 trained without, and the lattice assumes raster order."""
    total = sum(int(c.shape[0]) for c in chunks)
    if total > max_length:
        raise RuntimeError(f"a sequence of {total} tokens exceeds max_length {max_length}")
    out = assemble_sequence(chunks=chunks, chunk_modality_ids=[TV.MODALITY_TO_ID[m] for m in modalities],
                            chunk_shufflable=[False] * len(chunks), max_length=int(max_length),
                            pad_id=TV.PAD_ID, rng=np.random.default_rng(0), shuffle_image_patches=False)
    out["position_ids"] = lattice_positions([int(c.shape[0]) for c in chunks], [m == DATE for m in modalities],
                                            max_length)
    return out


def stack_trim(items: Sequence[Dict[str, torch.Tensor]], trim: bool = True) -> Dict[str, torch.Tensor]:
    """Stack sequences into one micro-batch, cut to its longest real sequence when `trim`."""
    for it in items:
        if tuple(sorted(it)) != tuple(sorted(KEYS)):
            raise RuntimeError(f"a sequence carries keys {sorted(it)}, not the closed five {sorted(KEYS)}")
    batch = {k: torch.stack([it[k] for it in items]) for k in KEYS}
    if not trim:
        return batch
    keep = int(batch["attention_mask"].sum(1).max())
    return {k: v[:, :keep].contiguous() for k, v in batch.items()}


def unwrap_microbatch(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """The collator of every micro-batch dataset: HF's DataLoader hands it a list of one micro-batch."""
    if len(items) != 1:
        raise RuntimeError(f"one index is one micro-batch, so per_device_train_batch_size and "
                           f"per_device_eval_batch_size must be 1; the loader batched {len(items)}")
    mb = items[0]
    if tuple(sorted(mb)) != tuple(sorted(KEYS)):
        raise RuntimeError(f"a micro-batch carries keys {sorted(mb)}, not the closed five {sorted(KEYS)}: "
                           f"compute_loss would forward the extra into model(**inputs)")
    if any(v.ndim != 2 for v in mb.values()):
        raise RuntimeError("a micro-batch's tensors must be (sequences, length)")
    return mb


# --------------------------------------------------------------------------- draws
@dataclass(frozen=True)
class Block:
    modality: str           # an image modality, Coords, or Date
    t: int                  # the date index whose row it reads (statics: 0)
    with_year: bool = True  # date blocks only


@dataclass(frozen=True)
class LocationDraw:
    """Everything the RNG decided for one location's sequence."""
    location: int
    dates: Tuple[int, ...]              # the sequence's date indices, chronological
    order_mode: str
    held: int                           # leave-one-out's last date, drawn in every mode
    order: Tuple[int, ...]              # date indices in group order
    lulc_t: int
    statics: Tuple[str, str]
    statics_slot: int                   # groups before the statics; len(order) means at the end
    date_modes: Tuple[str, ...]         # per group, in group order
    with_year: Tuple[bool, ...]         # per group, in group order
    blocks: Tuple[Block, ...]


@dataclass(frozen=True)
class SceneDraw:
    row: int
    order: Tuple[str, ...]
    date_mode: str
    with_year: bool


def _group_blocks(t: int, modalities: Sequence[str], date_mode: str, with_year: bool) -> List[Block]:
    """One date group: its image blocks in the drawn order, and its date block placed by decision 7."""
    group = [Block(m, t) for m in modalities]
    date = Block(DATE, t, with_year)
    if date_mode == "first":
        return [date] + group
    if date_mode == "last":
        return group + [date]
    return group


class TemporalLocations(Dataset):
    """Whole-location sequences over a base TerraMeshTokenDataset, as TargetLast wraps one, so its
    _load_chunk (uint16 cast before the offset), presence masks and artifact checks are reused.

    As a Dataset, index i is location `location_ids[i]` at the current epoch's stage. The training
    data is MixedMicrobatches, which calls `sequence` directly."""

    def __init__(self, base: TerraMeshTokenDataset, index: TemporalIndex, location_ids: Sequence[int],
                 cfg: TemporalConfig, seed: int, allow_test: bool = False):
        super().__init__()
        if Path(base.root_dir).resolve() != Path(index.root).resolve():
            raise ValueError(f"base reads {base.root_dir} but the index describes {index.root}")
        missing = [m for m in SEQUENCE_MODALITIES if m not in base.active_modalities]
        if missing:
            raise ValueError(f"the base dataset lacks {missing}, which every location sequence holds")
        if base.n_rows != len(index.dates):
            raise ValueError(f"base has {base.n_rows} rows, the index dates {len(index.dates)}")
        self.base, self.index, self.cfg, self.seed = base, index, cfg, int(seed)
        ids = np.asarray(location_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0 or len(np.unique(ids)) != len(ids):
            raise ValueError("location_ids must be a non-empty 1-D set of distinct ids")
        if ids.min() < 0 or ids.max() >= len(index):
            raise ValueError(f"location ids fall outside [0, {len(index)})")
        self.test_ids = val_test_location_ids(index)
        if self.test_ids is not None and not allow_test:
            bad = np.intersect1d(ids, self.test_ids)
            if bad.size:
                raise RuntimeError(f"{bad.size} of these locations are on val's test side "
                                   f"(temporal_rows_v1.json); they stay unread until the base is frozen")
        self.location_ids = ids
        self.epoch = 0

    # --- the Dataset view
    def set_draw_epoch(self, epoch: int) -> None:
        """Not `set_epoch`, which accelerate calls with its own loader count (module docstring)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(len(self.location_ids))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        _, stage = self.cfg.stage_of(self.epoch)
        return self.sequence(int(self.location_ids[int(i)]), self.epoch, stage.dates)

    # --- the grammar
    def _rng(self, seed: int, epoch: int, loc: int) -> np.random.Generator:
        return np.random.default_rng((int(seed), int(epoch), int(loc), _TEMPORAL))

    def rows(self, loc: int) -> np.ndarray:
        """The location's four rows, chronological (temporal_index.py)."""
        return self.index.locations[int(loc)]

    def draw(self, loc: int, epoch: int, n_dates: int, seed: Optional[int] = None) -> LocationDraw:
        """The layout of one location at one epoch. Every draw is made in every case, in a fixed
        order, so one setting never shifts the stream of another."""
        c = self.cfg
        n = int(n_dates)
        if not 1 <= n <= N_DATES:
            raise ValueError(f"n_dates {n_dates} outside 1..{N_DATES}")
        rng = self._rng(self.seed if seed is None else seed, epoch, loc)
        subset = rng.choice(N_DATES, size=n, replace=False)
        dates = tuple(range(N_DATES)) if n == N_DATES else tuple(sorted(int(t) for t in subset))
        mode = ORDER_MODES[int(rng.choice(len(ORDER_MODES), p=c.order_probs))]
        held = dates[int(rng.integers(n))]
        perm = rng.permutation(n)
        if mode == "chronological":
            order = dates
        elif mode == "leave_one_out":
            order = tuple(t for t in dates if t != held) + (held,)
        else:
            order = tuple(dates[int(i)] for i in perm)
        lulc_t = dates[int(rng.integers(n))]
        statics = STATICS if int(rng.integers(2)) == 0 else STATICS[::-1]
        slot = int(rng.integers(n + 1))
        slot = 0 if mode == "chronological" else slot
        groups, date_modes, years = [], [], []
        for t in order:
            mods = list(GROUP_MODALITIES) + ([LULC] if t == lulc_t else [])
            mods = [mods[int(i)] for i in rng.permutation(len(mods))]
            dm = DATE_MODES[int(rng.choice(len(DATE_MODES), p=c.date_probs))]
            wy = bool(rng.random() >= c.year_drop)
            groups.append(_group_blocks(t, mods, dm, wy))
            date_modes.append(dm)
            years.append(wy)
        blocks: List[Block] = []
        for k, group in enumerate(groups):
            if k == slot:
                blocks += [Block(m, 0) for m in statics]
            blocks += group
        if slot == n:
            blocks += [Block(m, 0) for m in statics]
        return LocationDraw(int(loc), dates, mode, held, order, lulc_t, tuple(statics), slot,
                            tuple(date_modes), tuple(years), tuple(blocks))

    def sequence(self, loc: int, epoch: int, n_dates: int, seed: Optional[int] = None,
                 draw: Optional[LocationDraw] = None) -> Dict[str, torch.Tensor]:
        """The five-key sequence of one location, padded to sequence_length(n_dates)."""
        d = self.draw(loc, epoch, n_dates, seed) if draw is None else draw
        rows = self.rows(loc)
        chunks, mods = [], []
        for b in d.blocks:
            row = int(rows[b.t])
            if b.modality == DATE:
                year, day = self.index.date_of(row)
                chunks.append(torch.tensor(TV.date_block(year, day, b.with_year), dtype=torch.long))
            else:
                chunks.append(self.base._load_chunk(b.modality, row))
            mods.append(b.modality)
        return _assemble(chunks, mods, sequence_length(len(d.dates)))


class SingleScenes:
    """Single scenes (Major TOM rows): T360's layout plus one date block placed as a group's."""

    def __init__(self, base: TerraMeshTokenDataset, index: TemporalIndex, rows: Sequence[int],
                 cfg: TemporalConfig, seed: int):
        self.base, self.index, self.cfg, self.seed = base, index, cfg, int(seed)
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
            raise ValueError("rows must be a 1-D set of distinct row ids")
        if rows.size and (rows.min() < 0 or rows.max() >= base.n_rows):
            raise ValueError(f"rows fall outside [0, {base.n_rows})")
        seasonal = np.intersect1d(rows, index.locations.ravel())
        if seasonal.size:
            raise ValueError(f"{seasonal.size} of these rows belong to four-date locations: an ssl4eos12 row "
                             f"reaches the model only inside its location's sequence (decision 5)")
        self.rows = rows
        if rows.size and int(base._seq_lens[rows].max()) + width(DATE) > cfg.single_max_length:
            raise ValueError(f"a scene with its date block needs {int(base._seq_lens[rows].max()) + width(DATE)} "
                             f"tokens; single_max_length is {cfg.single_max_length}")

    def _rng(self, seed: int, epoch: int, row: int) -> np.random.Generator:
        return np.random.default_rng((int(seed), int(epoch), int(row), _SINGLE))

    def draw(self, row: int, epoch: int, seed: Optional[int] = None) -> SceneDraw:
        rng = self._rng(self.seed if seed is None else seed, epoch, row)
        present = self.base.present_modalities(int(row))
        order = tuple(present[int(i)] for i in rng.permutation(len(present)))
        dm = DATE_MODES[int(rng.choice(len(DATE_MODES), p=self.cfg.date_probs))]
        wy = bool(rng.random() >= self.cfg.year_drop)
        return SceneDraw(int(row), order, dm, wy)

    def sequence(self, row: int, epoch: int, seed: Optional[int] = None,
                 draw: Optional[SceneDraw] = None) -> Dict[str, torch.Tensor]:
        d = self.draw(row, epoch, seed) if draw is None else draw
        blocks = _group_blocks(0, d.order, d.date_mode, d.with_year)
        chunks, mods = [], []
        for b in blocks:
            if b.modality == DATE:
                year, day = self.index.date_of(d.row)
                chunks.append(torch.tensor(TV.date_block(year, day, b.with_year), dtype=torch.long))
            else:
                chunks.append(self.base._load_chunk(b.modality, d.row))
            mods.append(b.modality)
        return _assemble(chunks, mods, self.cfg.single_max_length)


# --------------------------------------------------------------------------- micro-batches
class MixedMicrobatches(Dataset):
    """The training set: one index is one homogeneous micro-batch, of locations or of scenes.

    It ORDERS ITSELF: index p of an epoch is a micro-batch drawn from a permutation keyed on
    (seed, epoch), so the trainer reads it with a SequentialSampler (`orders_itself`, honoured by
    util/trainer_pt.py). HF's shuffled sampler cannot be trusted across a resume (module docstring)."""

    collate_fn = staticmethod(unwrap_microbatch)
    orders_itself = True

    def __init__(self, temporal: TemporalLocations, single: Optional[SingleScenes], cfg: TemporalConfig,
                 microbatches_per_step: int, seed: int):
        super().__init__()
        self.temporal, self.single, self.cfg, self.seed = temporal, single, cfg, int(seed)
        self.microbatches_per_step = int(microbatches_per_step)
        if self.microbatches_per_step < 1:
            raise ValueError("microbatches_per_step must be >= 1")
        self.n = int(cfg.steps_per_epoch) * self.microbatches_per_step
        self.n_single = int(round(cfg.single_share * self.n))
        self.n_temporal = self.n - self.n_single
        if self.n_temporal < 1:
            raise ValueError(f"no temporal micro-batch in an epoch of {self.n}")
        if self.n_single and (single is None or not len(single.rows)):
            raise ValueError(f"single_share {cfg.single_share} needs Major TOM rows")
        ids = temporal.location_ids
        self.loc_walk = ids[np.random.default_rng((self.seed, _LOC_WALK)).permutation(len(ids))]
        self.row_walk = (single.rows[np.random.default_rng((self.seed, _ROW_WALK)).permutation(len(single.rows))]
                         if self.n_single else np.zeros(0, dtype=np.int64))
        self.epoch = 0
        self._assigned: Tuple[int, Optional[np.ndarray]] = (-1, None)

    def set_draw_epoch(self, epoch: int) -> None:
        """The epoch whose draws this dataset serves. Deliberately not `set_epoch`: accelerate calls a
        dataset's set_epoch with its own loader count, which is 0 in the loader that skips batches
        after a mid-epoch resume (module docstring). Only TemporalEpochCallback sets it."""
        self.epoch = int(epoch)

    def epoch_callback(self) -> "TemporalEpochCallback":
        return TemporalEpochCallback(self, self.cfg.steps_per_epoch)

    def __len__(self) -> int:
        return self.n

    def stage_of(self, epoch: int) -> Tuple[int, Stage]:
        return self.cfg.stage_of(epoch)

    def temporal_cursor(self, epoch: int) -> int:
        """Locations the temporal walk drew before this epoch."""
        total, left = 0, int(epoch)
        for s in self.cfg.stages:
            k = left if s.epochs is None else min(left, int(s.epochs))
            total += k * self.n_temporal * int(s.locations)
            left -= k
            if left == 0:
                break
        return total

    def assignment(self, epoch: int) -> np.ndarray:
        """(n,) per index: its ordinal in the temporal stream (>= 0), or -1 - its ordinal among scenes.
        Read in index order, this is the epoch's order: the two streams interleaved at random."""
        if self._assigned[0] != int(epoch):
            perm = np.random.default_rng((self.seed, int(epoch), _ASSIGN)).permutation(self.n)
            out = np.empty(self.n, dtype=np.int64)
            out[perm[: self.n_temporal]] = np.arange(self.n_temporal)
            out[perm[self.n_temporal:]] = -1 - np.arange(self.n_single)
            self._assigned = (int(epoch), out)
        return self._assigned[1]

    def members(self, i: int, epoch: Optional[int] = None) -> Tuple[str, np.ndarray, int]:
        """(stream, location ids or rows, dates per sequence) of micro-batch i."""
        epoch = self.epoch if epoch is None else int(epoch)
        if not 0 <= int(i) < self.n:
            raise IndexError(i)
        a = int(self.assignment(epoch)[int(i)])
        if a >= 0:
            _, stage = self.stage_of(epoch)
            start = self.temporal_cursor(epoch) + a * int(stage.locations)
            return "temporal", self.loc_walk[(start + np.arange(stage.locations)) % len(self.loc_walk)], stage.dates
        per = int(self.cfg.scenes_per_microbatch)
        start = (epoch * self.n_single + (-1 - a)) * per
        return "single", self.row_walk[(start + np.arange(per)) % len(self.row_walk)], 1

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        stream, ids, n_dates = self.members(int(i))
        if stream == "temporal":
            items = [self.temporal.sequence(int(loc), self.epoch, n_dates) for loc in ids]
        else:
            items = [self.single.sequence(int(row), self.epoch) for row in ids]
        return stack_trim(items)

    def token_share(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """Expected real tokens per micro-batch of each stream at an epoch's stage, and Major TOM's share
        of the step's tokens (which is its share of the gradient, module docstring)."""
        epoch = self.epoch if epoch is None else int(epoch)
        _, stage = self.stage_of(epoch)
        c = self.cfg
        date_tokens = (1 - c.date_probs[DATE_MODES.index("absent")]) * (width(DATE) - c.year_drop)
        per_location = sequence_length(stage.dates) - stage.dates * width(DATE) + stage.dates * date_tokens
        temporal = self.n_temporal * stage.locations * per_location
        single = 0.0
        if self.n_single:
            mean_scene = float(self.single.base._seq_lens[self.single.rows].mean()) + date_tokens
            single = self.n_single * c.scenes_per_microbatch * mean_scene
        return {"stage_dates": int(stage.dates), "tokens_per_location": float(per_location),
                "temporal_tokens": float(temporal), "single_tokens": float(single),
                "single_token_share": float(single / (single + temporal))}


class SingleEval(Dataset):
    """The single-date eval set in micro-batches: T360's in-training eval exactly. Consecutive rows of
    the 4,416, untrimmed at single_max_length, so micro-batch m holds the tensors T360's eval batch m
    held at the same worker count (its orders are keyed on the worker id, eval_checkpoint.py)."""

    collate_fn = staticmethod(unwrap_microbatch)

    def __init__(self, base: TerraMeshTokenDataset, per: int):
        super().__init__()
        if len(base) % int(per):
            raise ValueError(f"{len(base)} eval rows do not split into micro-batches of {per}")
        self.base, self.per = base, int(per)

    def __len__(self) -> int:
        return len(self.base) // self.per

    def __getitem__(self, m: int) -> Dict[str, torch.Tensor]:
        return stack_trim([self.base[int(m) * self.per + i] for i in range(self.per)], trim=False)


class TemporalEval(Dataset):
    """The dev locations through the training draw, at eval_dates, with a fixed seed and epoch: the
    same sequences at every evaluation. Early checkpoints read high by design (the ramp)."""

    collate_fn = staticmethod(unwrap_microbatch)

    def __init__(self, temporal: TemporalLocations, per: int, n_dates: int, seed: int = EVAL_SEED):
        super().__init__()
        if len(temporal) % int(per):
            raise ValueError(f"{len(temporal)} locations do not split into micro-batches of {per}")
        self.temporal, self.per, self.n_dates, self.seed = temporal, int(per), int(n_dates), int(seed)

    def __len__(self) -> int:
        return len(self.temporal) // self.per

    def locations(self, m: int) -> np.ndarray:
        return self.temporal.location_ids[int(m) * self.per: (int(m) + 1) * self.per]

    def __getitem__(self, m: int) -> Dict[str, torch.Tensor]:
        return stack_trim([self.temporal.sequence(int(loc), 0, self.n_dates, seed=self.seed)
                           for loc in self.locations(m)])


# --------------------------------------------------------------------------- the epoch
class TemporalEpochCallback(TrainerCallback):
    """Sets the dataset's epoch to global_step // steps_per_epoch before the epoch's workers fork.

    Never SetEpochCallback's int(round(state.epoch)): a resume past mid-epoch restores state.epoch
    (5.6, say) before on_epoch_begin, while HF's loop resumes in epoch 5 (finding 18). And it checks
    that HF's epoch is ours, so the two counts cannot drift apart."""

    def __init__(self, dataset, steps_per_epoch: int):
        self.dataset, self.steps_per_epoch = dataset, int(steps_per_epoch)

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        if args.dataloader_persistent_workers:
            raise RuntimeError("TemporalEpochCallback needs non-persistent workers: persistent ones never "
                               "see a new epoch")
        if train_dataloader is not None:
            n, ga = len(train_dataloader), int(args.gradient_accumulation_steps)
            if n % ga or n // ga != self.steps_per_epoch:
                raise RuntimeError(f"HF's epoch is {n} micro-batches / {ga} accumulation = {n / ga} steps, not "
                                   f"temporal.steps_per_epoch {self.steps_per_epoch}: total_batch_size must count "
                                   f"micro-batches, and per_device_train_batch_size must be 1")
        self.dataset.set_draw_epoch(int(state.global_step) // self.steps_per_epoch)


# --------------------------------------------------------------------------- roots and builders
def val_test_location_ids(index: TemporalIndex) -> Optional[np.ndarray]:
    """Ids of the test side's locations when `index` is val's (told by its row order's hash, as
    temporal_index.split_report does), else None."""
    doc = load_temporal_split()
    if doc["inputs"]["tok_index.parquet"] != index.build["inputs"]["tok_index.parquet"]:
        return None
    return _ids_of(index, doc["test"]["rows"])


def val_dev_location_ids(index: TemporalIndex) -> np.ndarray:
    """The 544 dev locations of val's index, as index ids. Checks the split against the root."""
    doc = load_temporal_split(index.root)
    return _ids_of(index, doc["dev"]["rows"])


def _ids_of(index: TemporalIndex, rows: Sequence[Sequence[int]]) -> np.ndarray:
    line = {tuple(int(r) for r in quad): i for i, quad in enumerate(index.locations)}
    missing = [q for q in rows if tuple(int(r) for r in q) not in line]
    if missing:
        raise RuntimeError(f"{len(missing)} split locations are not lines of {index.root}'s index")
    return np.sort(np.asarray([line[tuple(int(r) for r in q)] for q in rows], dtype=np.int64))


def _majortom_rows(root: Path, index: TemporalIndex) -> np.ndarray:
    """Every Major TOM row of a root, from tok_index.parquet, checked against the index: the two
    corpora partition the rows, and the index's locations are all of ssl4eos12's."""
    import pandas as pd

    t = pd.read_parquet(root / "tok_index.parquet", columns=["row", "corpus"])
    if not (t["row"].to_numpy() == np.arange(len(t))).all():
        raise RuntimeError(f"{root}/tok_index.parquet does not number its rows 0..{len(t) - 1} in order")
    corpus = t["corpus"].to_numpy()
    seasonal = np.zeros(len(t), dtype=bool)
    seasonal[index.locations.ravel()] = True
    if not np.array_equal(seasonal, corpus == "ssl4eos12") or not np.array_equal(~seasonal, corpus == "majortom"):
        raise RuntimeError(f"{root}: the index's locations are not exactly the ssl4eos12 rows")
    return np.flatnonzero(corpus == "majortom").astype(np.int64)


def _settings(cfg):
    mm = cfg.get("multimodal", {})
    if mm.get("shuffle_image_patches", False):
        raise ValueError("the temporal stage never shuffles patches: the lattice assumes raster order")
    if mm.get("modality_order", "fixed") != "random":
        raise ValueError("the temporal stage needs multimodal.modality_order: random (T360's, and the "
                         "single-date eval set must stay T360's)")
    for key in ("per_device_train_batch_size", "per_device_eval_batch_size"):
        if int(cfg.get(key) or 0) != 1:
            raise ValueError(f"{key} must be 1: one index of the temporal datasets is one micro-batch")
    tcfg = TemporalConfig.from_cfg(cfg)
    need = sequence_length(tcfg.max_dates())
    if int(cfg.max_length) < need:
        raise ValueError(f"max_length {cfg.max_length} < {need}, the longest sequence the ramp or the eval "
                         f"set builds ({tcfg.max_dates()} dates)")
    return mm, tcfg


def _eval_split_path(split_ref) -> Path:
    """As load_dataset._build_terramesh resolves multimodal.eval_split: a bare tag, or a path."""
    from eo.data.eval_split import default_split_path
    from paths import PROJECT_ROOT

    path = Path(str(split_ref))
    if not path.suffix:
        return default_split_path(str(split_ref))
    return path if path.is_absolute() else Path(PROJECT_ROOT) / path


def _base(root, split: str, mm, tcfg: TemporalConfig, seed: int) -> TerraMeshTokenDataset:
    return TerraMeshTokenDataset(root_dir=str(root), split=split, active_modalities=mm.get("active_modalities", None),
                                 max_length=tcfg.single_max_length, modality_order="random", seed=seed,
                                 shuffle_image_patches=False, rows=None)


def build_train(cfg, val_root_dir) -> MixedMicrobatches:
    """The training set of a temporal config. multimodal.split train: every location and Major TOM row
    of TERRAMESH_TRAIN_TOK_ROOT. split val (development only): the dev locations, and the Major TOM
    rows outside the held-out split."""
    import os

    from eo.data.eval_split import load_eval_rows
    from eo.terramesh_tok.clean import resolve_token_root

    mm, tcfg = _settings(cfg)
    seed = int(cfg.get("seed", 42))
    token_set = mm.get("token_set", "original")
    if mm.get("split", "val") == "train":
        from eo.terramesh_tok.train_split import resolve_train_root
        root = Path(resolve_train_root(os.environ.get("TERRAMESH_TRAIN_TOK_ROOT"), token_set,
                                       allow_partial=bool(mm.get("allow_partial_train_root", False))))
        index = load_temporal_index(root)
        if val_test_location_ids(index) is not None:
            raise RuntimeError(f"{root} is val's root, not train's")
        locations = np.arange(len(index), dtype=np.int64)
        rows = _majortom_rows(root, index)
        split = "train"
    else:
        root = Path(resolve_token_root(val_root_dir, token_set))
        index = load_temporal_index(root)
        locations = val_dev_location_ids(index)
        split_ref = mm.get("eval_split", None)
        if split_ref is None:
            raise ValueError("a val (development) run needs multimodal.eval_split, to keep the held-out rows out")
        held = load_eval_rows(path=_eval_split_path(split_ref), root_dir=root)
        rows = np.setdiff1d(_majortom_rows(root, index), held)
        split = "val"
    base = _base(root, split, mm, tcfg, seed)
    temporal = TemporalLocations(base, index, locations, tcfg, seed)
    single = SingleScenes(base, index, rows, tcfg, seed) if tcfg.single_share > 0 else None
    ds = MixedMicrobatches(temporal, single, tcfg, int(cfg.total_batch_size), seed)
    share = ds.token_share(epoch=0)
    print(f"Temporal train ({split}, {root}): {len(locations)} locations, {len(rows)} Major TOM rows; an epoch is "
          f"{ds.n} micro-batches ({ds.n_temporal} temporal, {ds.n_single} single) = {tcfg.steps_per_epoch} steps "
          f"of {ds.microbatches_per_step}; stages {[(s.dates, s.locations, s.epochs) for s in tcfg.stages]}; "
          f"Major TOM's token share at epoch 0 {share['single_token_share']:.3f}")
    print(f"Temporal config: {json.dumps(tcfg.record())}")
    return ds


def build_eval(cfg, val_root_dir, single_base: Optional[TerraMeshTokenDataset]) -> Dict[str, Dataset]:
    """{"single": the 4,416 held-out rows in T360's layout at their own length (finding 11),
    "temporal": the dev locations at eval_dates}. `single_base` comes from load_dataset's
    _build_terramesh, so it is T360's eval set exactly."""
    from eo.terramesh_tok.clean import resolve_token_root

    mm, tcfg = _settings(cfg)
    if single_base is None:
        raise ValueError("the temporal stage needs multimodal.eval_split for its single-date eval set")
    if single_base.max_length != tcfg.single_max_length:
        raise ValueError(f"the single-date eval set is at {single_base.max_length}, not {tcfg.single_max_length}")
    root = Path(resolve_token_root(val_root_dir, mm.get("token_set", "original")))
    index = load_temporal_index(root)
    base = _base(root, "val", mm, tcfg, int(cfg.get("seed", 42)))
    temporal = TemporalLocations(base, index, val_dev_location_ids(index), tcfg, int(cfg.get("seed", 42)))
    out = {"single": SingleEval(single_base, tcfg.eval_scenes_per_microbatch),
           "temporal": TemporalEval(temporal, tcfg.eval_locations_per_microbatch, tcfg.eval_dates, tcfg.eval_seed)}
    print(f"Temporal eval: single {len(single_base)} rows in {len(out['single'])} micro-batches of "
          f"{tcfg.eval_scenes_per_microbatch} at {single_base.max_length}; temporal {len(temporal)} dev locations "
          f"in {len(out['temporal'])} micro-batches of {tcfg.eval_locations_per_microbatch} at {tcfg.eval_dates} "
          f"dates ({sequence_length(tcfg.eval_dates)} tokens), seed {tcfg.eval_seed}")
    return out
