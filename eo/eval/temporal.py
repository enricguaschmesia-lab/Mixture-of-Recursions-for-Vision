# eo/eval/temporal.py
"""The temporal tasks' layouts: sequences and prompts (`.venv`). A1 Step 7's evaluation module; its
prompt side was pulled forward into Step 6 for gate check Y13 (eo/scripts/verify_temporal.py).

Step 12's four tasks (docs/plans/a1_temporal_stage.md):

    task       context                                   target                    layout
    left_out   the other three dates, statics            S2L2A, S1GRD, NDVI at t   leave-one-out
    forecast   dates 0-2, statics                        the same at date 3        chronological
    gap_fill   the other dates, statics, S1GRD at t      S2L2A, NDVI at t          leave-one-out, S1 in t's group
    date       one date's images, statics                Y, D                      date last

HOW A TASK BECOMES A SEQUENCE. Each instance is a block list (`task_blocks`), wrapped in a forced
LocationDraw (`forced_draw`) and assembled by TemporalLocations.sequence(draw=...), the training
code path. So its tokens, modality ids and lattice positions are the training data's, and the PROMPT
is the sequence cut just after the target block's BO: a prefix by construction, its positions the
lattice's. The builder takes any block list, because Step 7 also rebuilds Step 0's stored layouts
(t360_context, t360_best_layouts) to pair with them.

DEFAULTS where the plan leaves the layout open (2026-11-09, Step 6; logged in the worklog):
  * the statics first, Coords then DEM, both from date 0's row (the grammar's rule, finding 19);
  * context groups chronological (leave-one-out: the others in order, the held date last; forecast
    is that with t = 3);
  * a date block first in every group, with its year; `with_dates=False` drops them all (T360's row,
    whose positions are then sequential, as T360 trained);
  * inside a group S2L2A, S1GRD, NDVI (GROUP_MODALITIES' order); LULC once, last in the first context
    group (the date task: last of its one group's images);
  * the target group last, its date block first; one target block per prompt, last;
  * gap fill: S1GRD at t, then the target, in t's group, and nothing else of date t;
  * date estimation: the date block last, with its year, Y and D the target; no other date block.

⚠ "NO TARGET TOKEN" MEANS PROVENANCE, NOT VALUES. Blocks of one modality at other dates share its
codebook, so a target's ids recur in context by design. What must not be in a prompt is the target's
own block, (modality, date), and for date estimation any date block of the target date. Y13 tests
both the declared blocks and the tokens against the rows (tokens.npy).

⚠ TEST LOCATIONS. `task_sequence` builds only locations of its TemporalLocations, which itself
refuses val's test side unless constructed with allow_test=True (only Step 12 does). So a dev set
cannot be asked for a test location's sequence, even by id.

Runs in `.venv`; never imports terratorch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from eo.data import temporal_vocab as TV
from eo.data.temporal_dataset import (DATE, GROUP_MODALITIES, LULC, STATICS, Block, LocationDraw,
                                      TemporalLocations, width)
from eo.data.temporal_index import N_DATES

TASKS = ("left_out", "forecast", "gap_fill", "date")
TARGETS = {"left_out": GROUP_MODALITIES, "forecast": GROUP_MODALITIES, "gap_fill": ("S2L2A", "NDVI"),
           "date": (DATE,)}


def task_dates(task: str) -> Tuple[int, ...]:
    """The target dates a task is scored at: every date, except forecast's last."""
    return (N_DATES - 1,) if task == "forecast" else tuple(range(N_DATES))


def task_blocks(task: str, t: int, target: str, with_dates: bool = True) -> List[Block]:
    """One task instance as a block list, the target block last (module docstring, DEFAULTS)."""
    if task not in TASKS:
        raise ValueError(f"task {task!r} not in {TASKS}")
    if target not in TARGETS[task]:
        raise ValueError(f"{task}'s targets are {TARGETS[task]}, not {target!r}")
    if t not in task_dates(task):
        raise ValueError(f"{task} is scored at dates {task_dates(task)}, not {t}")
    if task == "date" and not with_dates:
        raise ValueError("date estimation needs its date block: it is the target")
    blocks = [Block(m, 0) for m in STATICS]
    if task == "date":
        return blocks + [Block(m, t) for m in GROUP_MODALITIES] + [Block(LULC, t), Block(DATE, t, True)]
    for k, u in enumerate(d for d in range(N_DATES) if d != t):
        blocks += [Block(DATE, u, True)] if with_dates else []
        blocks += [Block(m, u) for m in GROUP_MODALITIES] + ([Block(LULC, u)] if k == 0 else [])
    blocks += [Block(DATE, t, True)] if with_dates else []
    blocks += [Block("S1GRD", t)] if task == "gap_fill" else []
    return blocks + [Block(target, t)]


def forced_draw(loc: int, blocks: Sequence[Block], mode: str = "task") -> LocationDraw:
    """A LocationDraw that names exactly these blocks. sequence() reads only `blocks` and `dates`;
    the other fields describe the list, best effort, for a reader of the draw."""
    blocks = tuple(blocks)
    dated = [b for b in blocks if b.modality not in STATICS]
    order = tuple(dict.fromkeys(b.t for b in dated))
    first = {u: next(b for b in dated if b.t == u) for u in order}
    lulc = [b.t for b in blocks if b.modality == LULC]
    stat = [i for i, b in enumerate(blocks) if b.modality in STATICS]
    slot = len({b.t for b in blocks[: stat[0]] if b.modality not in STATICS}) if stat else 0
    years = tuple(next((b.with_year for b in dated if b.t == u and b.modality == DATE), True) for u in order)
    modes = tuple("first" if first[u].modality == DATE else
                  ("last" if any(b.modality == DATE and b.t == u for b in dated) else "absent") for u in order)
    return LocationDraw(int(loc), tuple(sorted(order)), mode, order[-1] if order else -1, order,
                        lulc[0] if lulc else -1, tuple(blocks[i].modality for i in stat), slot, modes, years, blocks)


def block_width(b: Block) -> int:
    return (4 if b.with_year else 3) if b.modality == DATE else width(b.modality)


@dataclass(frozen=True)
class TaskSequence:
    draw: LocationDraw
    full: Dict[str, torch.Tensor]       # the five keys, (L,), cut to the real length
    target: Tuple[int, int]             # [start, end) of the target block in `full`
    prompt: Dict[str, torch.Tensor]     # full[:start + 1]: every block before the target, and its BO

    @property
    def target_tokens(self) -> Tuple[int, int]:
        """[first, last) of what is scored or generated: the target's body (Y and D for a date block)."""
        return self.target[0] + 1, self.target[1] - 1


def task_sequence(temporal: TemporalLocations, loc: int, blocks: Sequence[Block],
                  target_index: int = -1) -> TaskSequence:
    """A block list as the training code assembles it, with the prompt cut at the target block
    (default the last). Refuses a location outside `temporal`'s set (module docstring)."""
    loc = int(loc)
    if loc not in set(int(x) for x in temporal.location_ids):
        raise RuntimeError(f"location {loc} is not one of this TemporalLocations' locations")
    blocks = list(blocks)
    k = target_index % len(blocks)
    draw = forced_draw(loc, blocks)
    seq = temporal.sequence(loc, 0, len(draw.dates), draw=draw)
    real = int(seq["attention_mask"].sum())
    full = {key: v[:real].clone() for key, v in seq.items()}
    start = sum(block_width(b) for b in blocks[:k])
    end = start + block_width(blocks[k])
    target = blocks[k]
    bo = TV.BO_DATE if target.modality == DATE else TV.get_modality(target.modality).bo_id
    if int(full["input_ids"][start]) != bo or end > real:
        raise RuntimeError(f"the target block {target} does not start at {start} of {real} tokens")
    return TaskSequence(draw, full, (start, end), {key: v[: start + 1].clone() for key, v in full.items()})


def task_instances(tasks: Sequence[str] = TASKS) -> List[Tuple[str, int, str]]:
    """(task, t, target) for every instance Step 12 scores."""
    return [(task, t, tgt) for task in tasks for t in task_dates(task) for tgt in TARGETS[task]]


def batch(items: Sequence[TaskSequence], part: str = "prompt") -> Dict[str, torch.Tensor]:
    """Stack equal-length prompts (or full sequences) of one task into (B, L) tensors."""
    xs = [getattr(it, part) for it in items]
    lens = {int(x["input_ids"].shape[0]) for x in xs}
    if len(lens) != 1:
        raise ValueError(f"{part}s of lengths {sorted(lens)} do not stack: keep every date block's year")
    return {k: torch.stack([x[k] for x in xs]) for k in xs[0]}
