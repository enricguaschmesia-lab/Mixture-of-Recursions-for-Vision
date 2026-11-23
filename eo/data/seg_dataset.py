# eo/data/seg_dataset.py
"""A2's segmentation sequences: input tokens, then the map, target last (A2 Step 3.2, `.venv`).

THE LAYOUTS (plan, "How the inputs and the map are ordered"; the base's nc temporal layout):

    PASTIS-1     [BO_S2L2A s1..s196 EO_S2L2A] [BO_MAP_PASTIS m1..mG EO_MAP]                      396 + (G - 196)
    PASTIS-7     7 x ([BO_Date D EO_Date] [BO_S2L2A .. EO_S2L2A]), chronological, then the map    1,605 + (G - 196)
    CloudSen12   one 224-px crop: [BO_S2L2A .. EO_S2L2A] [BO_MAP_CLOUDSEN12 .. EO_MAP]            396 + (G - 196)

  G is the arm's tokens a map: 196 (aligned), 256 (short), 1,024 (long), eo/data/map_vocab.ARMS.
  * Date blocks carry the day bin only (decision 8: PASTIS is 2018-19, whose year rows barely trained), from
    the date eo/geobench/views.py read off each frame's index in the patch's own date list (inputs.parquet).
    Day bin = (day of year - 1) // 5, temporal_index's rule.
  * Positions are the LATTICE (temporal_dataset.lattice_positions): sequential, except that after a date
    block the counter goes back to where the block began, so every image block and the map keep 198-token
    offsets. With no date block (PASTIS-1, CloudSen12) that is 0..L-1.
  * The loss is on the map's codes and EO_MAP only: every other label is -100. BO_MAP_<task> is the prompt's
    last token, never a target.
  * Assembly is `assemble_sequence` (the closed five keys; modality 0 on every BO/EO), then the lattice.

VIEWS. A view is one map the model writes (eo/geobench/views.py): (sample, crop, dihedral k). Its input rows
and its map row are the same view's, so "dihedral map = dihedral image" holds by construction (gate Z5).
  train   ONE example per sample (patch or tile) per epoch. The sample order is a permutation keyed on
          (seed, epoch), and the view is drawn among the sample's views (PASTIS: its 8 dihedral variants;
          CloudSen12: its 9 crops x 8) from an RNG keyed on (seed, epoch, sample). An epoch is cut to a whole
          number of steps (`batch`), so no partial accumulation chunk exists (CLAUDE.md, resume).
  eval    every view of the split in index order (val: k = 0; CloudSen12: the 3 x 3 test crops).
  `in_pinned` (PASTIS train only; worklog §4 item 82): None = all 1,455 patches (A2's protocol), True = the
          1,200 GeoBenchV2 0.9's pinned revision holds, False = the other 255.

⚠ THE EPOCH. Draws depend on the epoch, so this dataset orders its own epochs (`orders_itself`: the trainer
reads it with a SequentialSampler) and has no `set_epoch`, which accelerate calls with its own count after a
mid-epoch resume (A1 finding 23; CLAUDE.md). `SegEpochCallback` sets the draw epoch to
global_step // steps_per_epoch before the epoch's workers fork, as TemporalEpochCallback does.

CONTROLS (`variant`, for evaluation; eo/eval/seg.py):
  prior           no input: the map block alone, at the positions it has in the full layout
  wrong           the inputs of view `derangement[i]` (a fixed permutation of the split's samples with no fixed
                  point, eo/eval/ben.derangement; for CloudSen12 the same crop of another tile)
  shuffled_dates  PASTIS-7: the 7 date blocks' days permuted with no fixed point (keyed on seed and view),
                  images in place
  one_date        PASTIS-7: the middle group (j = 3) alone, with its date block

MAP LAYOUTS (Step 4.0's follow-ups, Enric 2026-11-20 evening; plan Step 4.0 "follow-ups"). Default: the map block
right after the inputs at sequential (lattice) positions, its codes in raster order, as Step 4.0 ran. Two options:
  map_shift d   (raster only) the whole map block, BO_MAP to EO_MAP, moved d positions on: positions jump by d after
                the inputs, so the map leaves the 198 lattice (alignment test: d = 99, half a block)
  map_order "nested"  (a 28 x 28 map over ONE 14 x 14 input block: PASTIS-1, CloudSen12) the map's four 2 x 2 phases
                as four 196-token blocks on the lattice: sub-cell j = 2 dy + dx of input patch k = 14 r + c (map cell
                (2r + dy, 2c + dx)) sits at position BO + 1 + 198 j + k, written block by block (j = 0..3), each in
                the inputs' raster order. Every map token is 198 (j + 1) positions after its input patch's token,
                positions only increase, and EO_MAP takes the last code's position + 1 (989 on PASTIS-1).
  Codes are stored, generated-and-returned and scored in RASTER order: map_chunk reorders into sequence order,
  `to_raster` reorders back, and `step_positions` gives generation the position of each code it feeds back.
Arms: Step 4.0's three (map_vocab.ARMS) or a follow-up alias (map_vocab.ALIASES), whose codes take its parent slot's
first ids.

⚠ A TEST SPLIT IS REFUSED unless `allow_test=True` (Step 6's protocol runs only). Never score it in Steps 3-5.

Runs in `.venv`: numpy, pandas, torch. Never imports terratorch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from eo.data import map_vocab as MV
from eo.data import temporal_vocab_nc as TVNC
from eo.data.temporal_dataset import lattice_positions
from eo.terramesh_tok.contract import TOKENS_PER_SAMPLE
from lm_dataset.sequence_assembly import assemble_sequence

try:                                             # HF is only needed for the callback; the gate's CPU part may
    from transformers import TrainerCallback     # import this module without building a trainer
except Exception:                                # pragma: no cover
    TrainerCallback = object

ROOT = Path("/data/enric/data/geobench2")
TASKS = ("pastis", "cloudsen12")
FRAMES = {"pastis": ("p1", "p7"), "cloudsen12": ("single",)}
VARIANTS = ("full", "prior", "wrong", "shuffled_dates", "one_date")
INPUT = "S2L2A"
P7 = [f"p7_{j}" for j in range(7)]
ONE_DATE_GROUP = 3
DATE_WIDTH = 3                                   # [BO_Date D EO_Date]: the day bin only
DERANGE_SEED = 20261119
_DRAW_STREAM = 7                                 # the view draw's RNG key tag


def day_bin(date: int) -> int:
    """YYYYMMDD -> temporal_index's day bin, (day of year - 1) // 5, capped at 72."""
    from eo.data.temporal_index import date_ids
    s = str(int(date))
    return int(date_ids([f"{s[:4]}-{s[4:6]}-{s[6:]}"])[0, 1])


def year_id(date: int) -> int:
    from eo.data.temporal_index import YEAR_FIRST
    return int(str(int(date))[:4]) - YEAR_FIRST


MAP_ORDERS = ("raster", "nested")
NESTED_SIDE, INPUT_SIDE, BLOCK = 28, 14, 198      # the nested layout: a 28 x 28 map over the 14 x 14 input grid


def nested_order(side: int = NESTED_SIDE, inner: int = INPUT_SIDE) -> np.ndarray:
    """order[t] = the raster index (row-major over side x side) of the code written t-th under the nested layout:
    t = j * inner^2 + k, k = r * inner + c the input patch, j = 2 dy + dx the sub-cell, cell (2r + dy, 2c + dx)."""
    f = side // inner
    out = []
    for dy in range(f):
        for dx in range(f):
            for r in range(inner):
                for c in range(inner):
                    out.append((f * r + dy) * side + (f * c + dx))
    return np.asarray(out, dtype=np.int64)


def required_length(task: str, frames: str, arm: str, variant: str = "full") -> int:
    g = MV.spec(arm)[1] + 2
    img = TOKENS_PER_SAMPLE + 2
    if task == "pastis" and frames == "p7":
        n = 1 if variant == "one_date" else 7
        return n * (DATE_WIDTH + img) + g
    return img + g


@dataclass
class SegSpec:
    task: str
    arm: str
    frames: str = "p1"
    split: str = "train"
    variant: str = "full"
    in_pinned: Optional[bool] = None
    root: Path = ROOT
    base_vocab: int = MV.BASE_VOCAB
    map_shift: int = 0
    map_order: str = "raster"

    def validate(self, allow_test: bool = False) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task {self.task!r}: {TASKS}")
        if self.frames not in FRAMES[self.task]:
            raise ValueError(f"{self.task} takes frames {FRAMES[self.task]}, not {self.frames!r}")
        if self.arm not in MV.ARMS and self.arm not in MV.ALIASES:
            raise ValueError(f"arm {self.arm!r}: {sorted(MV.ARMS) + sorted(MV.ALIASES)}")
        if self.map_order not in MAP_ORDERS:
            raise ValueError(f"map_order {self.map_order!r}: {MAP_ORDERS}")
        if int(self.map_shift) < 0 or (self.map_shift and self.map_order != "raster"):
            raise ValueError(f"map_shift {self.map_shift}: a non-negative shift of the raster layout only")
        if self.map_order == "nested" and (MV.spec(self.arm)[2] != NESTED_SIDE or self.frames == "p7"):
            raise ValueError(f"the nested layout puts a {NESTED_SIDE} x {NESTED_SIDE} map over ONE input block; arm "
                             f"{self.arm} is {MV.spec(self.arm)[2]} x {MV.spec(self.arm)[2]}, frames {self.frames}")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant {self.variant!r}: {VARIANTS}")
        if self.variant in ("shuffled_dates", "one_date") and self.frames != "p7":
            raise ValueError(f"variant {self.variant} is PASTIS-7's only")
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split {self.split!r}")
        if self.split == "test" and not allow_test:
            raise PermissionError("the test split is refused: Steps 3-5 never read it (allow_test is Step 6's)")
        if self.in_pinned is not None and not (self.task == "pastis" and self.split == "train"):
            raise ValueError("in_pinned selects among PASTIS's training patches only")
        if self.split == "train" and self.variant != "full":
            raise ValueError("controls are evaluation variants")


class SegDataset(Dataset):
    """One task, one arm, one split; see the module docstring."""

    orders_itself = True

    def __init__(self, spec: SegSpec, max_length: Optional[int] = None, seed: int = 42, batch: int = 1,
                 allow_test: bool = False):
        spec.validate(allow_test)
        self.spec, self.seed, self.batch = spec, int(seed), max(int(batch), 1)
        self.lay = MV.for_base(spec.base_vocab)
        self.V = self.lay.base
        if getattr(self.V, "NAME", None) != "nc":
            raise ValueError(f"A2's sequences are on the nc layout; base {spec.base_vocab} is "
                             f"{getattr(self.V, 'NAME', 'v1')}")
        self.task_token = self.lay.bo(spec.task)
        self.map_off, self.map_size = self.lay.slot(spec.arm)
        self.n_map = MV.spec(spec.arm)[1]
        self.side = MV.spec(spec.arm)[2]
        # sequence index -> raster index of the code written there (identity for the raster layout)
        self.seq_order = nested_order(self.side) if spec.map_order == "nested" else np.arange(self.n_map)
        assert sorted(self.seq_order.tolist()) == list(range(self.n_map)), "the map order is not a permutation"
        self.required = required_length(spec.task, spec.frames, spec.arm, "full")
        self.max_length = self.required if max_length is None else int(max_length)
        if self.max_length < self.required:
            raise ValueError(f"max_length {self.max_length} < the {self.required} tokens of {spec.task} "
                             f"{spec.frames} on arm {spec.arm}")
        self._load()
        self.draw_epoch = 0

    # ------------------------------------------------------------------ data
    def _load(self) -> None:
        sp = self.spec
        tok = Path(sp.root) / sp.task / "tok"
        views = pd.read_parquet(tok / "views.parquet")
        inputs = pd.read_parquet(tok / "inputs.parquet")
        if not np.array_equal(views.view.values, np.arange(len(views))):
            raise RuntimeError(f"{tok}/views.parquet is not numbered 0..n-1")
        mdir = tok / f"MAP_{sp.arm}"
        meta = json.loads((mdir / "metadata.json").read_text())
        if meta.get("arm") != sp.arm or int(meta["frozen"]["tokens"]) != self.n_map \
                or int(meta["frozen"]["n_codes"]) != self.map_size or int(meta["n_views"]) != len(views):
            raise RuntimeError(f"{mdir}/metadata.json is not arm {sp.arm}'s map over these {len(views)} views")
        idir = tok / f"{INPUT}_tok224"
        imeta = json.loads((idir / "metadata.json").read_text())
        if int(imeta["n_rows"]) != len(inputs) or int(imeta["tokens_per_sample"]) != TOKENS_PER_SAMPLE:
            raise RuntimeError(f"{idir}/metadata.json does not match inputs.parquet")
        self.maps = np.load(mdir / "tokens.npy", mmap_mode="r")
        self.s2 = np.load(idir / "tokens.npy", mmap_mode="r")
        # The map artifact must hold every view of the split read (checked below, once the split is selected): Route B's
        # cells (A2 Step 4.2) hold train and val only, since the test split's labels are never read.
        map_present = np.load(mdir / "present.npy")
        if not np.load(idir / "present.npy").all():
            raise RuntimeError(f"{tok}: an artifact is incomplete")
        self.map_meta, self.input_meta = meta, imeta

        # input rows of each view, in sequence order
        if sp.task == "pastis":
            names = ["p1"] if sp.frames == "p1" else P7
            x = inputs[inputs.frame.isin(names)]
            piv = x.pivot(index="view", columns="frame", values="row")[names]
            dates = x.pivot(index="view", columns="frame", values="date")[names]
            if len(piv) != len(views) or piv.isna().any().any():
                raise RuntimeError("inputs.parquet does not hold every frame of every view")
            self.in_rows = piv.loc[views.view].to_numpy(np.int64)
            self.in_dates = dates.loc[views.view].to_numpy(np.int64)
        else:
            if not np.array_equal(inputs.view.values, views.view.values):
                raise RuntimeError("cloudsen12: inputs and views are not row-aligned")
            self.in_rows = inputs.row.to_numpy(np.int64)[:, None]
            self.in_dates = None

        sel = views.split == sp.split
        if sp.in_pinned is not None:
            sel &= views.in_pinned == bool(sp.in_pinned)
        self.views = views[sel].reset_index(drop=True)
        if not map_present[self.views.view.values].all():
            raise RuntimeError(f"{mdir}: the artifact lacks views of split {sp.split}")
        # samples, and each sample's views (train draws among them)
        self.samples = np.unique(self.views.sample_idx.values)
        self._sample_views = {int(s): g.view.to_numpy(np.int64)
                              for s, g in self.views.groupby("sample_idx", sort=True)}
        if sp.split == "train":
            self.epoch_size = (len(self.samples) // self.batch) * self.batch
            if self.epoch_size == 0:
                raise ValueError(f"{len(self.samples)} samples < one batch of {self.batch}")
        else:
            self.epoch_size = len(self.views)
        self._derange = None

    # ------------------------------------------------------------------ the epoch
    def set_draw_epoch(self, epoch: int) -> None:
        """Not `set_epoch`, which accelerate calls with its own loader count (module docstring)."""
        self.draw_epoch = int(epoch)

    def epoch_callback(self, steps_per_epoch: Optional[int] = None, verbose: bool = True) -> "SegEpochCallback":
        return SegEpochCallback(self, steps_per_epoch or self.steps_per_epoch(), verbose)

    def steps_per_epoch(self) -> int:
        return self.epoch_size // self.batch

    def order(self, epoch: int) -> np.ndarray:
        """The epoch's samples, in order: a permutation keyed on (seed, epoch), cut to epoch_size."""
        rng = np.random.default_rng([self.seed, int(epoch)])
        return self.samples[rng.permutation(len(self.samples))][: self.epoch_size]

    def draw_view(self, sample: int, epoch: int) -> int:
        vs = self._sample_views[int(sample)]
        rng = np.random.default_rng([self.seed, _DRAW_STREAM, int(epoch), int(sample)])
        return int(vs[rng.integers(len(vs))])

    def view_at(self, i: int, epoch: Optional[int] = None) -> int:
        if self.spec.split != "train":
            return int(self.views.view.values[i])
        e = self.draw_epoch if epoch is None else int(epoch)
        return self.draw_view(int(self.order(e)[i]), e)

    def __len__(self) -> int:
        return self.epoch_size

    # ------------------------------------------------------------------ chunks
    def _s2_chunk(self, row: int) -> torch.Tensor:
        info = self.V.get_modality(INPUT)
        body = torch.from_numpy(np.asarray(self.s2[row]).astype(np.int64)) + info.codebook_offset
        return torch.cat([torch.tensor([info.bo_id]), body, torch.tensor([info.eo_id])])

    def _date_chunk(self, date: int) -> torch.Tensor:
        return torch.tensor(self.V.date_block(year_id(date), day_bin(date), with_year=False), dtype=torch.long)

    def map_chunk(self, view: int) -> torch.Tensor:
        """[BO_MAP_<task>, the view's codes in SEQUENCE order (+ the slot's offset), EO_MAP]."""
        codes = np.asarray(self.maps[view]).astype(np.int64)        # uint16 -> int64 BEFORE the offset
        return torch.tensor(self.lay.map_block(self.spec.arm, self.spec.task, codes[self.seq_order]),
                            dtype=torch.long)

    def to_raster(self, seq_codes: np.ndarray) -> np.ndarray:
        """Codes in sequence order (..., G) -> raster order (..., G): the inverse of map_chunk's reordering."""
        seq_codes = np.asarray(seq_codes)
        out = np.empty_like(seq_codes)
        out[..., self.seq_order] = seq_codes
        return out

    def map_positions(self, bo: int) -> torch.Tensor:
        """position_ids of [BO_MAP, G codes in sequence order, EO_MAP] when the inputs leave BO_MAP's lattice slot at
        `bo`: raster = sequential from bo + map_shift; nested = BO at bo, code t = j * 196 + k at bo + 1 + 198 j + k,
        EO_MAP after the last code (module docstring, MAP LAYOUTS)."""
        g = self.n_map
        if self.spec.map_order == "raster":
            return torch.arange(bo + int(self.spec.map_shift), bo + int(self.spec.map_shift) + g + 2, dtype=torch.long)
        t = torch.arange(g)
        inner = INPUT_SIDE * INPUT_SIDE
        codes = bo + 1 + BLOCK * (t // inner) + t % inner
        return torch.cat([torch.tensor([bo]), codes, codes[-1:] + 1]).long()

    def step_positions(self, variant: Optional[str] = None) -> torch.Tensor:
        """(G,) the position of each code generation feeds back: the codes' position_ids in a built sequence (its
        labelled tokens but EO_MAP). Read off a real sequence, so it is the training layout by construction; every view
        of a dataset (and variant) shares it."""
        s = self.sequence(int(self.views.view.values[0]), variant)
        lab = (s["labels"] != -100).nonzero().flatten()
        return s["position_ids"][lab[:-1]].clone()

    def derangement(self) -> Dict[int, int]:
        """view -> the view whose inputs the `wrong` control feeds it (same crop index, another sample)."""
        if self._derange is None:
            from eo.eval.ben import derangement
            d = derangement(len(self.samples), DERANGE_SEED)
            other = {int(self.samples[i]): int(self.samples[d[i]]) for i in range(len(self.samples))}
            out = {}
            for s, vs in self._sample_views.items():
                ws = self._sample_views[other[s]]
                for j, v in enumerate(vs):
                    out[int(v)] = int(ws[j % len(ws)])
            self._derange = out
        return self._derange

    def input_chunks(self, view: int, variant: Optional[str] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """The input blocks of `view` under a variant, in order, with their modality names."""
        variant = self.spec.variant if variant is None else variant
        src = self.derangement()[view] if variant == "wrong" else view
        rows = self.in_rows[src]
        chunks, mods = [], []
        if self.spec.frames != "p7":                    # PASTIS-1 and CloudSen12: one image, no date block
            return [self._s2_chunk(int(r)) for r in rows], [INPUT] * len(rows)
        dates = self.in_dates[src]
        groups = range(len(rows))
        if variant == "one_date":
            groups = [ONE_DATE_GROUP]
        if variant == "shuffled_dates":
            rng = np.random.default_rng([DERANGE_SEED, int(view)])
            while True:
                perm = rng.permutation(len(dates))
                if not (perm == np.arange(len(dates))).any():
                    break
            dates = dates[perm]
        for j in groups:
            chunks.append(self._date_chunk(int(dates[j])))
            mods.append("Date")
            chunks.append(self._s2_chunk(int(rows[j])))
            mods.append(INPUT)
        return chunks, mods

    # ------------------------------------------------------------------ sequences
    def sequence(self, view: int, variant: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """The five-key example of `view` under a variant (training uses "full")."""
        variant = self.spec.variant if variant is None else variant
        chunks, mods = self.input_chunks(view, variant)
        m = self.map_chunk(view)
        widths = [int(c.shape[0]) for c in chunks]
        lm = int(m.shape[0])
        lattice = lattice_positions(widths + [lm], [x == "Date" for x in mods] + [False], self.max_length)
        start = sum(widths)
        mpos = self.map_positions(int(lattice[start]))             # the layout's map positions (MAP LAYOUTS)
        if variant == "prior":                         # the map alone, at its positions in the full layout
            chunks, mods, widths = [], [], []
            last = int(mpos[-1])
            pos = torch.cat([mpos, torch.arange(last + 1, last + 1 + self.max_length - lm)])
        else:
            pos = lattice.clone()
            pos[start:start + lm] = mpos
            last = int(mpos[-1])
            pos[start + lm:] = torch.arange(last + 1, last + 1 + self.max_length - start - lm)
        ids = [self.V.MODALITY_TO_ID[x] for x in mods] + [MV.MAP_MODALITY_ID]
        out = assemble_sequence(chunks=chunks + [m], chunk_modality_ids=ids,
                                chunk_shufflable=[False] * (len(chunks) + 1), max_length=self.max_length,
                                pad_id=self.lay.pad_id, rng=np.random.default_rng(0), shuffle_image_patches=False)
        out["position_ids"] = pos[: self.max_length].clone()
        start = sum(widths)
        labels = torch.full_like(out["labels"], -100)
        lo, hi = start + 1, start + int(m.shape[0])            # the codes and EO_MAP; not BO_MAP
        labels[lo:hi] = out["input_ids"][lo:hi]
        out["labels"] = labels
        return out

    def prompt(self, view: int, variant: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(input_ids, position_ids) up to and including BO_MAP_<task>: what generation is given."""
        s = self.sequence(view, variant)
        # the prompt ends at BO_MAP: the last position before the first label
        first = int((s["labels"] != -100).nonzero()[0])
        assert s["input_ids"][first - 1] == self.task_token, "the prompt does not end in BO_MAP_<task>"
        return s["input_ids"][:first].clone(), s["position_ids"][:first].clone()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.sequence(self.view_at(i))


class SegEpochCallback(TrainerCallback):
    """Sets the dataset's draw epoch to global_step // steps_per_epoch before the epoch's workers fork, and
    checks that HF's epoch is the dataset's (temporal_dataset.TemporalEpochCallback's rule, for one-example
    items)."""

    def __init__(self, dataset: SegDataset, steps_per_epoch: int, verbose: bool = True):
        self.dataset, self.steps_per_epoch, self.verbose = dataset, int(steps_per_epoch), bool(verbose)

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        if args.dataloader_persistent_workers:
            raise RuntimeError("SegEpochCallback needs non-persistent workers: persistent ones never see a new epoch")
        epoch = int(state.global_step) // self.steps_per_epoch
        if train_dataloader is not None:
            n, ga = len(train_dataloader), int(args.gradient_accumulation_steps)
            if n % ga or n // ga != self.steps_per_epoch:
                raise RuntimeError(f"HF's epoch is {n} batches / {ga} accumulation = {n / ga} steps, not the "
                                   f"dataset's {self.steps_per_epoch}: seg.batch must equal total_batch_size")
        if self.verbose and getattr(state, "is_world_process_zero", True):
            print(f"SegEpochCallback: global_step {int(state.global_step)} -> draw epoch {epoch}", flush=True)
        self.dataset.set_draw_epoch(epoch)


# --------------------------------------------------------------------------- builders (pretrain.py)
def spec_from_cfg(cfg, want: str) -> Optional[SegSpec]:
    sc = cfg.get("seg") or {}
    split = sc.get("train_split", "train") if want == "train" else sc.get("eval_split", "val")
    if split is None:
        return None
    return SegSpec(task=str(sc["task"]), arm=str(sc["arm"]), frames=str(sc.get("frames", "p1")), split=str(split),
                   in_pinned=sc.get("in_pinned", None) if want == "train" else None,
                   root=Path(str(sc.get("root", ROOT))), base_vocab=int(cfg.model_config.vocab_size) - MV.N_NEW,
                   map_shift=int(sc.get("map_shift", 0) or 0), map_order=str(sc.get("map_order", "raster") or "raster"))


def build(cfg, want: str = "train") -> Optional[SegDataset]:
    spec = spec_from_cfg(cfg, want)
    if spec is None:
        return None
    batch = int(cfg.get("total_batch_size") or 1) if want == "train" else 1
    cls = SegDataset
    if str((cfg.get("seg") or {}).get("route", "map") or "map") == "head":
        from eo.data.seg_head_dataset import SegHeadDataset as cls      # A2 Step 4.2, Route C: the S2 block twice
    ds = cls(spec, max_length=int(cfg.max_length), seed=int(cfg.get("seed", 42)), batch=batch)
    print(f"seg {want}: {spec.task} {spec.frames} arm {spec.arm} split {spec.split} in_pinned {spec.in_pinned} "
          f"map_order {spec.map_order} map_shift {spec.map_shift}: "
          f"{len(ds.samples)} samples, {len(ds)} examples an epoch, {ds.required} tokens (max_length {ds.max_length})")
    return ds
