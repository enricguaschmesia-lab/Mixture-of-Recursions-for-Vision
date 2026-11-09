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

SCORING (A1 Step 7, 2026-11-09). Everything below the task builders.

  * LAYOUTS ARE DATA, in eo/eval/temporal_layouts_<tag>.json: every layout written out per target
    and target date as "MOD@date ..." (`parse_blocks`), the target block last. A tag is FROZEN once
    its hash is pinned in FROZEN_LAYOUTS: `load_layouts` refuses a file that no longer hashes to it,
    so a change is a new tag, never an edit. It holds Step 0's layouts exactly (T360's own, to
    reproduce its stored units), T360-T's task layouts (Step 6's defaults, `task_blocks`), T360's
    baselines per task, the pilot's reading layouts, and the date probe's settings.
  * LOCATIONS AND TARGET DATES are Step 0's (eo/experiments/temporal/t360_context.py): the first
    256 of a seeded permutation of dev, in dev order, and target date t = (i + j) % 4 for location
    i and target j of (S2L2A, S1GRD, NDVI), or the layout's own rule. So every reading pairs, per
    location, with Step 0's stored units (`step0_locations`).
  * THREE SCORING PATHS. A target block's CE goes through teacher_forced.run_pass with
    per_row_target, its labels masked to the target's body (`scored`), in trim_collate batches:
    the same CE, top-1 (rank, never topk) and autocast as every evaluation. A whole sequence's CE
    per block (`score_blocks`) reads run_pass's per-token arrays. The date task cannot use run_pass,
    whose modalities are eo_vocab's: `score_dates` reads the logits of Y and D itself, their CE over
    the whole vocabulary and the model's distribution over the 12 years and 73 day bins.
  * OPTIONS. Without date blocks (`with_dates=False`, T360's row); a WRONG DATE (`move_dates`): the
    target's date block, or every date block, with its day bin moved by `day_shift` (mod 73) and its
    year by `year_shift`, nothing else changed; the target block moved in position (`move_target`),
    for the offset curve.

Runs in `.venv`; never imports terratorch.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from eo.data import temporal_vocab as TV
from eo.data.temporal_dataset import (DATE, GROUP_MODALITIES, KEYS, LULC, SEQUENCE_MODALITIES, STATICS, Block,
                                      LocationDraw, TemporalLocations, width)
from eo.data.temporal_index import N_DATES, N_DOY_BINS, N_YEARS

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


# =========================================================================== Step 7: layouts as data
LAYOUTS_TAG = "v1"
#: Frozen layout files: tag -> sha256 of the file's bytes (module docstring). None: still being drawn
#: up, loaded only with allow_unfrozen=True, and every unit records its own layout's hash meanwhile.
FROZEN_LAYOUTS: Dict[str, Optional[str]] = {"v1": "9a5dd8ef62208246ea3e717cba61e85141ad005952d22ad35d663fca611e1583"}
STEP0_SEED = 20261105          # t360_context.SEED: the permutation of dev that picks Step 0's locations
STEP0_N = 256
BLOCK_MODALITIES = SEQUENCE_MODALITIES + (DATE,)


def layouts_path(tag: str = LAYOUTS_TAG) -> Path:
    return Path(__file__).resolve().parent / f"temporal_layouts_{tag}.json"


def canonical_sha(obj) -> str:
    """sha256 of an object's canonical JSON: what a unit records of the layout entry it read."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_blocks(spec: str) -> List[Block]:
    """'Coords@0 DEM@0 Date@1 S2L2A@1 ...' -> blocks. Every date block keeps its year."""
    out = []
    for item in spec.split():
        mod, _, t = item.partition("@")
        if mod not in BLOCK_MODALITIES or not t.isdigit() or not 0 <= int(t) < N_DATES:
            raise ValueError(f"{item!r} is not MODALITY@date with a modality of {BLOCK_MODALITIES}")
        out.append(Block(mod, int(t)))
    return out


def format_blocks(blocks: Sequence[Block]) -> str:
    if any(b.modality == DATE and not b.with_year for b in blocks):
        raise ValueError("a layout's date blocks keep their year")
    return " ".join(f"{b.modality}@{b.t}" for b in blocks)


def without_dates(blocks: Sequence[Block]) -> List[Block]:
    """T360's row: the same blocks with every date block dropped (positions then sequential)."""
    return [b for b in blocks if b.modality != DATE]


def layout_problems(doc: Dict) -> List[str]:
    """Everything wrong with a layout file's structure. Empty = valid."""
    out = []
    for key in ("tag", "locations", "probe", "layouts"):
        if key not in doc:
            out.append(f"no {key!r}")
    if out:
        return out
    for name, e in doc["layouts"].items():
        rule = e.get("dates", {}).get("rule")
        if rule not in ("step0", "fixed", "none"):
            out.append(f"{name}: date rule {rule!r}")
            continue
        for target, by_t in e.get("blocks", {}).items():
            if target not in e.get("targets", []):
                out.append(f"{name}: blocks for {target!r}, not one of its targets")
            for t, spec in by_t.items():
                try:
                    blocks = parse_blocks(spec)
                except ValueError as err:
                    out.append(f"{name} {target} t{t}: {err}")
                    continue
                if rule == "none":
                    continue
                last = blocks[-1] if blocks else None
                want = DATE if target == DATE else target
                if last is None or last.modality != want or str(last.t) != t:
                    out.append(f"{name} {target} t{t}: the last block {last} is not the target at t{t}")
                if any((b.modality, b.t) == (last.modality, last.t) for b in blocks[:-1]):
                    out.append(f"{name} {target} t{t}: the target block is also in the context")
        for target in e.get("targets", []):
            if target not in e.get("blocks", {}):
                out.append(f"{name}: no blocks for target {target!r}")
    return out


def load_layouts(tag: str = LAYOUTS_TAG, path=None, allow_unfrozen: bool = False) -> Dict:
    """A layout file, refused if it does not hash to its pinned value (FROZEN_LAYOUTS), or if it is
    not pinned yet and `allow_unfrozen` is False. Adds `sha256` (the file's) to the returned dict."""
    p = Path(path) if path is not None else layouts_path(tag)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    want = FROZEN_LAYOUTS.get(tag)
    if want is None and not allow_unfrozen:
        raise RuntimeError(f"{p.name}: tag {tag!r} is not frozen yet (FROZEN_LAYOUTS); pass allow_unfrozen")
    if want is not None and sha != want:
        raise RuntimeError(f"{p.name} hashes to {sha[:12]}, not the frozen {want[:12]}: a frozen layout file is "
                           f"never edited; write a new tag")
    doc = json.loads(raw)
    if doc.get("tag") != tag:
        raise RuntimeError(f"{p.name} says tag {doc.get('tag')!r}, not {tag!r}")
    bad = layout_problems(doc)
    if bad:
        raise RuntimeError(f"{p.name}: {bad[:5]}")
    doc["sha256"] = sha
    return doc


def target_date(entry: Dict, i: int, target: str) -> Optional[int]:
    """A layout's target date for location i (Step 0's rule, module docstring), or None."""
    d = entry["dates"]
    if d["rule"] == "none":
        return None
    if d["rule"] == "fixed":
        return int(d["t"])
    j = GROUP_MODALITIES.index(target) if d.get("j", "target") == "target" else int(d["j"])
    return (int(i) + j) % N_DATES


def layout_blocks(entry: Dict, target: str, t: Optional[int], with_dates: bool = True) -> List[Block]:
    blocks = parse_blocks(entry["blocks"][target]["*" if t is None else str(int(t))])
    return blocks if with_dates else without_dates(blocks)


# =========================================================================== Step 7: locations
def step0_locations(index, n: int = STEP0_N, seed: int = STEP0_SEED):
    """Step 0's locations (t360_context.pick_locations): the first n of a seeded permutation of dev,
    in dev order. Returns (location ids of `index`, their 5 km groups, their names), in Step 0's order,
    which is the order of every stored unit."""
    from eo.data.temporal_split import load_temporal_split
    dev = load_temporal_split(index.root)["dev"]
    idx = np.sort(np.random.default_rng(seed).permutation(len(dev["locations"]))[:n])
    line = {tuple(int(r) for r in quad): i for i, quad in enumerate(index.locations)}
    ids = np.asarray([line[tuple(int(r) for r in dev["rows"][k])] for k in idx], dtype=np.int64)
    return ids, np.asarray(dev["groups"], dtype=np.int64)[idx], [dev["locations"][k] for k in idx]


def dev_temporal(root, seed: int = 42):
    """val_clean's dev locations as a TemporalLocations (test side refused), and its index."""
    from eo.data.temporal_dataset import TemporalConfig, val_dev_location_ids
    from eo.data.temporal_index import load_temporal_index
    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    index = load_temporal_index(root)
    base = TerraMeshTokenDataset(root_dir=str(root), split="val", active_modalities=list(SEQUENCE_MODALITIES),
                                 max_length=1048, modality_order="random", seed=seed, rows=None)
    return TemporalLocations(base, index, val_dev_location_ids(index), TemporalConfig(), seed), index


# =========================================================================== Step 7: sequence options
def block_starts(blocks: Sequence[Block]) -> List[int]:
    out, at = [], 0
    for b in blocks:
        out.append(at)
        at += block_width(b)
    return out


def _with(ts: TaskSequence, full: Dict[str, torch.Tensor]) -> TaskSequence:
    return TaskSequence(ts.draw, full, ts.target, {k: v[: ts.target[0] + 1].clone() for k, v in full.items()})


def move_dates(ts: TaskSequence, which: str = "target", day_shift: int = 36, year_shift: int = 0) -> TaskSequence:
    """The same sequence with a WRONG DATE: `which` = "target" moves the target date's own date block
    (the last date block of the target's date before it), "all" every date block before the target.
    Its day bin moves by `day_shift` modulo 73, its year by `year_shift` (kept inside the 12 years);
    input_ids and labels change at those tokens and nowhere else. Refuses a date-block target (that
    would change what is scored) and a sequence without the date block asked for."""
    if which not in ("target", "all"):
        raise ValueError(f"which={which!r}: 'target' or 'all'")
    blocks = list(ts.draw.blocks)
    starts = block_starts(blocks)
    k = starts.index(ts.target[0])
    tgt = blocks[k]
    if tgt.modality == DATE:
        raise ValueError("the target is a date block: a wrong date would change what is scored")
    picks = [i for i, b in enumerate(blocks[:k]) if b.modality == DATE and (which == "all" or b.t == tgt.t)]
    picks = picks[-1:] if which == "target" else picks
    if not picks:
        raise ValueError(f"no date block to move ({which}) before the target {tgt}")
    full = {key: v.clone() for key, v in ts.full.items()}
    for i in picks:
        s, w = starts[i], block_width(blocks[i])
        year, day = TV.decode_date_block(full["input_ids"][s:s + w].tolist())
        new_year = None if year is None else year + int(year_shift)
        if new_year is not None and not 0 <= new_year < N_YEARS:
            raise ValueError(f"year id {year} + {year_shift} leaves the 12 years")
        new = torch.tensor(TV.date_block(0 if new_year is None else new_year, (day + int(day_shift)) % N_DOY_BINS,
                                         new_year is not None), dtype=torch.long)
        full["input_ids"][s:s + w] = new
        full["labels"][s:s + w] = new
    return _with(ts, full)


def move_target(ts: TaskSequence, delta: int) -> TaskSequence:
    """The target block (BO, body, EO) moved `delta` positions along, nothing in the gap: Step 0's
    long offsets (t360_context, `pos[L - BLOCK:] += delta`). The target must be the last block."""
    if ts.target[1] != len(ts.full["input_ids"]):
        raise ValueError("move_target needs the target last")
    full = {key: v.clone() for key, v in ts.full.items()}
    full["position_ids"][ts.target[0]:ts.target[1]] += int(delta)
    return _with(ts, full)


def scored(ts: TaskSequence) -> Dict[str, torch.Tensor]:
    """The full sequence with labels -100 outside the target's body: what the target's CE reads."""
    out = {k: v.clone() for k, v in ts.full.items()}
    a, b = ts.target_tokens
    lab = torch.full_like(out["labels"], -100)
    lab[a:b] = out["labels"][a:b]
    out["labels"] = lab
    return out


# =========================================================================== Step 7: scoring
def _pad(item: Dict[str, torch.Tensor], length: int) -> Dict[str, torch.Tensor]:
    n = int(item["input_ids"].shape[0])
    if n == length:
        return item
    fill = {"input_ids": TV.PAD_ID, "attention_mask": 0, "labels": -100, "modality_ids": 0}
    out = {k: torch.cat([v, torch.full((length - n,), fill[k], dtype=v.dtype)]) for k, v in item.items()
           if k != "position_ids"}
    p = item["position_ids"]
    out["position_ids"] = torch.cat([p, p[-1] + 1 + torch.arange(length - n, dtype=p.dtype)])
    return out


def batches(items: Sequence[Dict[str, torch.Tensor]], size: int) -> List[Dict[str, torch.Tensor]]:
    """Items padded to a common length and collated with teacher_forced.trim_collate, `size` at a time."""
    from eo.eval.teacher_forced import trim_collate
    for it in items:
        if tuple(sorted(it)) != tuple(sorted(KEYS)):
            raise RuntimeError(f"an item carries keys {sorted(it)}, not the closed five")
    out = []
    for k in range(0, len(items), int(size)):
        part = list(items[k:k + int(size)])
        L = max(int(x["input_ids"].shape[0]) for x in part)
        out.append(trim_collate([_pad(x, L) for x in part]))
    return out


def score_target(model, items: Sequence[Dict[str, torch.Tensor]], target: str, batch_size: int,
                 dtype=None, device: str = "cuda") -> Dict[str, np.ndarray]:
    """Per item: mean CE and top-1 over the target's body, through run_pass (labels masked by `scored`)."""
    from eo.eval import teacher_forced as TF
    from eo.terramesh_tok.contract import TOKENS_PER_SAMPLE
    res = TF.run_pass(model, batches(items, batch_size), device=device, keep_tokens=False,
                      per_row_target=target, dtype=dtype)
    rows = res["rows"]
    n = rows["n"].astype(np.int64)
    if len(n) != len(items) or (n != TOKENS_PER_SAMPLE).any():
        raise RuntimeError(f"{target}: {sorted(set(n.tolist()))} tokens scored per item, not {TOKENS_PER_SAMPLE}: "
                           f"the labels are not masked to one target block")
    return {"ce": rows["ce_sum"].astype(np.float64) / n, "top1": rows["top1"].astype(np.float64) / n, "n": n}


def score_blocks(model, items: Sequence[Dict[str, torch.Tensor]], blocks: Sequence[Block], batch_size: int,
                 dtype=None, device: str = "cuda") -> np.ndarray:
    """(items, blocks) mean CE over each block's body (Y and D for a date block), from run_pass's
    per-token arrays (fp16, as every stored per-token CE). Items: whole sequences, one block list."""
    from eo.eval import teacher_forced as TF
    L = {int(x["input_ids"].shape[0]) for x in items}
    if len(L) != 1:
        raise ValueError(f"block profiles need one length, got {sorted(L)}")
    starts = block_starts(blocks)
    bos = [TV.BO_DATE if b.modality == DATE else TV.get_modality(b.modality).bo_id for b in blocks]
    for x in items:
        ids = x["input_ids"]
        if starts[-1] + block_width(blocks[-1]) != len(ids) or any(int(ids[s]) != bo for s, bo in zip(starts, bos)):
            raise RuntimeError("the block list does not describe these sequences")
    res = TF.run_pass(model, batches(items, batch_size), device=device, keep_tokens=True, dtype=dtype)
    ce = res["tokens"]["ce"].astype(np.float64)          # (N, L-1): column j is the CE of token j + 1
    lm = res["tokens"]["label_mod"]
    out = np.empty((len(items), len(blocks)))
    for k, (s, b) in enumerate(zip(block_starts(blocks), blocks)):
        w = block_width(b)
        if not (lm[:, s:s + w - 2] > 0).all():
            raise RuntimeError(f"block {b} at {s}: a body token is not scored")
        out[:, k] = ce[:, s:s + w - 2].mean(1)
    return out


@torch.no_grad()
def score_dates(model, items: Sequence[Dict[str, torch.Tensor]], batch_size: int, dtype=None,
                device: str = "cuda") -> Dict[str, np.ndarray]:
    """The date task, its date block last with its year: per item, CE of Y and D over the whole
    vocabulary (D given the true Y: teacher-forced), their ranks (count strictly above, never topk),
    the softmax mass on the 12 year rows and 73 day rows, and the model's distribution over each
    (`p_year` (N, 12), `p_day` (N, 73)), with the true year and day."""
    model.eval()
    keys = ("ce_y", "ce_d", "rank_y", "rank_d", "mass_years", "mass_days", "year", "day", "p_year", "p_day")
    acc = {k: [] for k in keys}
    years, days = torch.tensor(TV.YEAR_IDS), torch.tensor(TV.DAY_IDS)
    for k in range(0, len(items), int(batch_size)):
        part = items[k:k + int(batch_size)]
        x = torch.stack([it["input_ids"] for it in part]).to(device)
        if len({int(it["input_ids"].shape[0]) for it in part}) != 1:
            raise ValueError("date items of one length only")
        if not (bool((x[:, -4] == TV.BO_DATE).all()) and bool((x[:, -1] == TV.EO_DATE).all())
                and bool(torch.isin(x[:, -3], years.to(device)).all()) and bool(torch.isin(x[:, -2], days.to(device)).all())):
            raise RuntimeError("the date task's items must end with a date block holding its year")
        kw = {key: torch.stack([it[key] for it in part]).to(device) for key in ("attention_mask", "position_ids")}
        with torch.autocast("cuda", dtype=dtype or torch.float16, enabled=dtype is not None):
            lg = model(input_ids=x, logits_to_keep=4, use_cache=False, **kw).logits[:, :2].float()
        lab = x[:, -3:-1]                                                  # Y, D: predicted from BO_Date, Y
        lsm = torch.log_softmax(lg, -1)
        ce = -lsm.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
        rank = (lg > lg.gather(-1, lab.unsqueeze(-1))).sum(-1)
        py = torch.softmax(lg[:, 0, years.to(device)], -1)
        pd = torch.softmax(lg[:, 1, days.to(device)], -1)
        acc["ce_y"].append(ce[:, 0]); acc["ce_d"].append(ce[:, 1])
        acc["rank_y"].append(rank[:, 0]); acc["rank_d"].append(rank[:, 1])
        acc["mass_years"].append(lsm[:, 0, years.to(device)].exp().sum(-1))
        acc["mass_days"].append(lsm[:, 1, days.to(device)].exp().sum(-1))
        acc["year"].append(lab[:, 0] - TV.YEAR_BASE); acc["day"].append(lab[:, 1] - TV.DAY_BASE)
        acc["p_year"].append(py); acc["p_day"].append(pd)
    return {k: torch.cat(v).cpu().numpy() for k, v in acc.items()}


# =========================================================================== Step 7: date measures
BIN_DAYS = 5                    # temporal_index.DOY_BIN_DAYS: 73 bins of 5 days, the last 5 or 6
QUARTERS = ("DJF", "MAM", "JJA", "SON")


def quarter(day_bin) -> np.ndarray:
    """Calendar quarter (0 DJF, 1 MAM, 2 JJA, 3 SON) of a day bin, by its centre's day of year
    (5 d + 3; March 1 is day 60 outside leap years). Hemisphere-blind, as the plan reads it."""
    c = 5 * np.asarray(day_bin, dtype=np.int64) + 3
    return np.select([(c < 60) | (c >= 335), c < 152, c < 244], [0, 1, 2], 3)


def circular_bins(a, b) -> np.ndarray:
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) % N_DOY_BINS
    return np.minimum(d, N_DOY_BINS - d)


def date_measures(r: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per item, from score_dates: top-1 year over the 12 years, the day error in days of the most
    likely bin and of the distribution's circular mean, and full-vocabulary top-1 of Y and D."""
    p_day = r["p_day"].astype(np.float64)
    ang = 2 * np.pi * np.arange(N_DOY_BINS) / N_DOY_BINS
    mean_bin = (np.angle((p_day * np.exp(1j * ang)).sum(1)) % (2 * np.pi)) * N_DOY_BINS / (2 * np.pi)
    return {"top1_year": (r["p_year"].argmax(1) == r["year"]).astype(np.float64),
            "day_err_mode": BIN_DAYS * circular_bins(r["p_day"].argmax(1), r["day"]),
            "day_err_mean": BIN_DAYS * circular_bins(mean_bin, r["day"]),
            "top1_y_vocab": (r["rank_y"] == 0).astype(np.float64), "top1_d_vocab": (r["rank_d"] == 0).astype(np.float64)}


# =========================================================================== Step 7: intervals
DRAWS = 10_000


def group_ci(values, groups, seed: int = 0, draws: int = DRAWS) -> Dict[str, float]:
    """Mean and 95% interval from a bootstrap over spatial groups: t360_context.group_ci, the same
    seed and draws, so that intervals compare with Step 0's."""
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups)
    u, inv = np.unique(groups, return_inverse=True)
    s, n = np.bincount(inv, values), np.bincount(inv).astype(float)
    pick = np.random.default_rng(seed).integers(0, len(u), (draws, len(u)))
    b = s[pick].sum(1) / n[pick].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}
