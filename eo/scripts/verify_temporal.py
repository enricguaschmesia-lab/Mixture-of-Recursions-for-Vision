#!/usr/bin/env python
"""The temporal stage's gate (`.venv`, CPU).

Same discipline as verify_phase3.py and verify_phase6.py: every check ships with a deliberately broken
control that proves it can fail, and a check lands with the step that produces what it reads. A check
whose artifact does not exist yet is SKIPPED and reported as such, never passed.

    python eo/scripts/verify_temporal.py
    python eo/scripts/verify_temporal.py --only Y1

  Y1  temporal_vocab: every id below the date rows maps to itself and every date id is used once;
      a date block round-trips for every date, with and without its year, from dates.npy's uint8;
      UNTRAINED_YEARS describes both dates.npy
                                     | date rows before BEN's, on PAD, past a gap, a repeated id and a
                                       missing bin are rejected; date blocks out of range or out of
                                       order are refused; the wrong untrained years are flagged
  Y9  The extended base: extension.json records a passed fp32 identity and both controls fired; its
      weights hash to the record and carry the current layout; the table is 87,664 rows and every
      tensor of the source checkpoint is unchanged, bit for bit; lm_head is tied, so not stored
                                     | a failed identity, an unfired control, another size, a changed
                                       weights file and one moved row are all refused

Step 4 (the dataset, eo/data/temporal_dataset.py), on val_clean's 544 dev locations and its Major TOM
rows outside the held-out split. No test location is ever read: Y3 audits every row this gate reads.

  Y3  Sequences parse back to their source rows from the tokens alone, read here straight from
      tokens.npy: every block matches a row of its own location, the date blocks match dates.npy,
      one date block per group, LULC once, the statics once from date 0's row, and the layout the
      dataset declares; only dev locations and Major TOM rows outside the held-out split are built
                                     | a row off by one; LULC in two groups; a test location
  Y4  Every micro-batch, and the dry build through load_dataset_from_config with T360's arm, is the
      closed five keys; modality_ids labels every body and Date; labels follow the mask
                                     | a sixth key; Date bodies left unlabelled
  Y5  The same micro-batches at 0, 2 and 3 workers and in a second instance (another rank); a new
      epoch draws anew               | draws keyed on the worker id; draws that ignore the epoch
  Y6  Order, date, year, statics, LULC, subset and scene draws hit their probabilities over 10,000
      draws each                     | the date mode `last` set to 0; the order mode `uniform` set to 0
  Y7  A Major TOM scene with its date dropped is TerraMeshTokenDataset's sequence at the same order,
      all five tensors; with the date in, it is T360's tokens and positions plus one date block
                                     | the date block left in
  Y8  Micro-batches are homogeneous over whole epochs; each epoch splits exactly by single_share;
      Major TOM's token share is within 0.02 of it; HF weighs every token of a step alike
      (num_items_in_batch), so that share is the gradient's; trimming moves no logit
                                     | a mixed micro-batch; half the scenes per micro-batch; a model
                                       that refuses loss kwargs; a trim that cuts a real token
  Y14 A dict of eval sets logs each set under its own keys and nothing under plain eval_loss_<mod>
                                     | the guard removed
  Y16 (dataset half) position_ids are the lattice: image blocks sit 198n apart, 198n + 5 with Coords
      between, whatever date blocks lie between
                                     | positions rewound over Coords; sequential positions
  Y17 The ramp through HF's own loop and the production trainer mixin (a probe model, CPU): the
      run serves the dataset's own order at each epoch's stage; resumes past mid-epoch and across
      stage boundaries replay it step for step, at 0 and 2 workers; two ranks through accelerate's
      resume loader keep their epoch and shards
                                     | SetEpochCallback's rounded epoch; a stage read from a counter;
                                       HF's shuffled sampler (resumed in epoch 0's order); a dataset
                                       with set_epoch on two ranks (accelerate resets it to 0)

  ⚠ Y5 and Y17 cover the rank through index purity, a second instance and accelerate's own loader
    classes for two ranks; two real processes under DDP wait for Step 8's two A100s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eo.data import ben_vocab as BV  # noqa: E402
from eo.data import eo_vocab as EV  # noqa: E402
from eo.data import temporal_vocab as TV  # noqa: E402
from eo.data.temporal_index import N_DOY_BINS, N_YEARS, load_temporal_index  # noqa: E402

ROOTS = {"val_clean": Path("/data/enric/data/TerraMesh/val_clean"),
         "train": Path("/data/enric/data/TerraMesh/train_coords")}
EXT = Path("/data/enric/runs/pretrain/temporal/base/t360_ext_temporal_ckpt17441")
TABLE = "model.embed_tokens.weight"
VAL = ROOTS["val_clean"]
ARM = "arm_t360_vanilla"         # the dry build composes T360's arm with the temporal dataset

RESULTS = []


def record(check, ok, detail, ctrl_ok=None, ctrl_detail=""):
    RESULTS.append((check, ok, ctrl_ok))
    print(f"{'PASS' if ok else 'FAIL'}  {check}: {detail}")
    if ctrl_ok is not None:
        print(f"      control {'fired' if ctrl_ok else 'DID NOT FIRE'}: {ctrl_detail}")


def skip(check, why):
    RESULTS.append((check, None, None))
    print(f"SKIP  {check}: {why}")


def refused(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except (ValueError, RuntimeError):
        return True
    return False


# ---------------------------------------------------------------- Y1
def y1():
    probs = TV.layout_problems(TV.YEAR_IDS, TV.DAY_IDS, TV.BO_DATE, TV.EO_DATE, TV.TOTAL_VOCAB_SIZE)
    same = all(TV.MODALITIES[n] == i for n, i in BV.MODALITIES.items()) \
        and all(TV.MODALITIES[n].bo_id == i.bo_id and TV.MODALITIES[n].codebook_offset == i.codebook_offset
                for n, i in EV.MODALITIES.items())
    ids_same = all(TV.MODALITY_TO_ID[n] == i for n, i in BV.MODALITY_TO_ID.items()) and TV.MODALITY_TO_ID["Date"] == 9
    used = TV.YEAR_IDS + TV.DAY_IDS + [TV.BO_DATE, TV.EO_DATE]
    ok = (not probs and same and ids_same and TV.PAD_ID == EV.PAD_ID and TV.BASE == BV.TOTAL_VOCAB_SIZE
          and sorted(used) == list(range(TV.BASE, TV.TOTAL_VOCAB_SIZE)))

    y, d = TV.YEAR_IDS, TV.DAY_IDS
    broken = {
        "date rows before BEN's": ([EV.TOTAL_VOCAB_SIZE + i for i in range(N_YEARS)],
                                   [EV.TOTAL_VOCAB_SIZE + N_YEARS + i for i in range(N_DOY_BINS)],
                                   EV.TOTAL_VOCAB_SIZE + 85, EV.TOTAL_VOCAB_SIZE + 86, TV.TOTAL_VOCAB_SIZE),
        "start at PAD": ([EV.PAD_ID + i for i in range(N_YEARS)], d, TV.BO_DATE, TV.EO_DATE, TV.TOTAL_VOCAB_SIZE),
        "gap after BEN's": ([i + 1 for i in y], [i + 1 for i in d], TV.BO_DATE + 1, TV.EO_DATE + 1,
                            TV.TOTAL_VOCAB_SIZE + 1),
        "repeated id": ([y[0]] * N_YEARS, d, TV.BO_DATE, TV.EO_DATE, TV.TOTAL_VOCAB_SIZE),
        "a day bin missing": (y, d[:-1], TV.BO_DATE, TV.EO_DATE, TV.TOTAL_VOCAB_SIZE),
    }
    fired = {k: bool(TV.layout_problems(*v)) for k, v in broken.items()}
    record("Y1 layout", ok, f"{TV.N_NEW} date ids {TV.BASE}..{TV.TOTAL_VOCAB_SIZE - 1}, PAD {TV.PAD_ID} and every "
           f"id below {TV.BASE} unchanged, Date is modality {TV.MODALITY_TO_ID['Date']}", all(fired.values()),
           ", ".join(f"{k}: {'rejected' if v else 'ACCEPTED'}" for k, v in fired.items()))

    # Every date, both forms, through dates.npy's type. Offsets added to a uint8 would wrap.
    n, good = 0, True
    for yi in np.arange(N_YEARS, dtype=np.uint8):
        for di in np.arange(N_DOY_BINS, dtype=np.uint8):
            for with_year in (True, False):
                n += 1
                block = TV.date_block(yi, di, with_year=with_year)
                good &= all(type(t) is int for t in block) and block[0] == TV.BO_DATE and block[-1] == TV.EO_DATE
                good &= TV.decode_date_block(block) == (int(yi) if with_year else None, int(di))
    bad_in = [("year 12", lambda: TV.date_block(N_YEARS, 0)), ("day bin 73", lambda: TV.date_block(0, N_DOY_BINS)),
              ("year after day", lambda: TV.decode_date_block([TV.BO_DATE, TV.DAY_BASE, TV.YEAR_BASE, TV.EO_DATE])),
              ("no EO_Date", lambda: TV.decode_date_block([TV.BO_DATE, TV.YEAR_BASE, TV.DAY_BASE])),
              ("a BEN id inside", lambda: TV.decode_date_block([TV.BO_DATE, BV.BASE, TV.DAY_BASE, TV.EO_DATE])),
              ("empty body", lambda: TV.decode_date_block([TV.BO_DATE, TV.EO_DATE]))]
    fired = {k: refused(f) for k, f in bad_in}
    record("Y1 date blocks", bool(good), f"{n} blocks round-trip from uint8", all(fired.values()),
           ", ".join(f"{k}: {'refused' if v else 'ACCEPTED'}" for k, v in fired.items()))

    seen, sizes = set(), {}
    for name, root in ROOTS.items():
        dates = np.asarray(load_temporal_index(root).dates)
        years, counts = np.unique(dates[:, 0], return_counts=True)
        seen |= set(int(v) for v in years)
        sizes[name] = len(dates)
    problems = TV.coverage_problems(seen)
    wrong = {"2020 listed as untrained": TV.coverage_problems(seen, (2015, 2020, 2025, 2026)),
             "2026 not listed": TV.coverage_problems(seen, (2015, 2025))}
    n_ids = len(TV.UNTRAINED_IDS)
    ok = not problems and n_ids == 24 and all(BV.BASE <= i < TV.BASE + N_YEARS for i in TV.UNTRAINED_IDS)
    record("Y1 untrained rows", ok, f"{n_ids} ids: BEN's 21 and years {TV.UNTRAINED_YEARS}, none of which has a row "
           f"in {sizes}", all(bool(v) for v in wrong.values()),
           ", ".join(f"{k}: {'flagged' if v else 'NOT FLAGGED'}" for k, v in wrong.items()))


# ---------------------------------------------------------------- Y9
def y9(ext: Path = EXT):
    if not (ext / "extension.json").is_file():
        skip("Y9 extended base", f"{ext} has no extension.json: Step 3's extension has not run")
        return
    from safetensors import safe_open

    from eo.finetune.init import verify_extension

    doc = verify_extension(ext, TV.TOTAL_VOCAB_SIZE, check_hash=True)
    ident, c1 = doc["identity"], doc["control_insert_before_pad"]
    ok = (doc["layout"] == "temporal" and doc["temporal_vocab"] == TV.layout_record() and doc["tied"]
          and ident["precision"] == "fp32" and ident["max_abs_diff"] <= ident["tol"] < c1["max_abs_diff"])
    record("Y9 record", ok, f"identity max |dlogit| {ident['max_abs_diff']:.3e} on {ident['positions']} positions, "
           f"insert before PAD {c1['max_abs_diff']:.2f}, device {doc['gpu']}, sha {doc['weights_sha256'][:12]}")

    source = Path(doc["source_checkpoint"]) / "model.safetensors"
    with safe_open(str(source), "pt") as old, safe_open(str(ext / doc["weights_file"]), "pt") as new:
        names_old, names_new = set(old.keys()), set(new.keys())
        moved = [k for k in sorted(names_old - {TABLE}) if not torch.equal(old.get_tensor(k), new.get_tensor(k))]
        a, b = old.get_tensor(TABLE), new.get_tensor(TABLE)
    n_old = a.shape[0]
    same_rows = bool(torch.equal(a, b[:n_old]))
    fresh = b[n_old:]
    expected = a.float().mean(0, keepdim=True)
    near_mean = float((fresh - expected).abs().max() / a.float().std(0).max())
    ok = (names_old == names_new and not moved and same_rows and b.shape == (TV.TOTAL_VOCAB_SIZE, a.shape[1])
          and "lm_head.weight" not in names_new and near_mean < 0.1)
    doctored = b.clone()
    doctored[1234, 7] += 1e-3
    shifted = torch.cat([b[:EV.PAD_ID], fresh, b[EV.PAD_ID:n_old]], 0)
    fired = {"one moved row": not torch.equal(a, doctored[:n_old]), "rows before PAD": not torch.equal(a, shifted[:n_old])}
    record("Y9 weights", ok, f"{len(names_new)} tensors, {len(moved)} moved; table {tuple(b.shape)}, rows below {n_old} "
           f"bit-identical, {len(fresh)} new rows at the mean (largest move {near_mean:.3f} of a std), lm_head not stored",
           all(fired.values()), ", ".join(f"{k}: {'seen' if v else 'MISSED'}" for k, v in fired.items()))

    tmp = tempfile.TemporaryDirectory(prefix="y9_")

    def fake(mutate, change_weights=False):
        d = Path(tempfile.mkdtemp(dir=tmp.name))
        w = doc["weights_file"]
        (d / w).symlink_to(ext / w)
        fake_doc = json.loads(json.dumps(doc))
        mutate(fake_doc)
        if change_weights:
            fake_doc["weights_sha256"] = "0" * 64
        (d / "extension.json").write_text(json.dumps(fake_doc))
        return d

    def set_key(*path, value):
        def go(x):
            for k in path[:-1]:
                x = x[k]
            x[path[-1]] = value
        return go

    cases = {"identity failed": (fake(set_key("identity", "passed", value=False)), {}),
             "control 1 did not fire": (fake(set_key("control_insert_before_pad", "fired", value=False)), {}),
             "control 2 did not fire": (fake(set_key("control_untied_resize", "fired", value=False)), {}),
             "another size": (fake(set_key("extension", "vocab_after", value=BV.TOTAL_VOCAB_SIZE)), {}),
             "weights changed": (fake(lambda x: None, change_weights=True), {"check_hash": True})}
    fired = {k: refused(verify_extension, d, TV.TOTAL_VOCAB_SIZE, **{"check_hash": False, **kw})
             for k, (d, kw) in cases.items()}
    tmp.cleanup()
    record("Y9 refusals", refused(verify_extension, ext, BV.TOTAL_VOCAB_SIZE, check_hash=False),
           "the base is refused at BEN's size", all(fired.values()),
           ", ".join(f"{k}: {'refused' if v else 'ACCEPTED'}" for k, v in fired.items()))


# ================================================================ Step 4: the dataset
MODS = ("S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords")
GROUP = ("S2L2A", "S1GRD", "NDVI")
STATIC = ("Coords", "DEM")
KEYS = ("input_ids", "attention_mask", "labels", "position_ids", "modality_ids")
READ_ROWS: set = set()          # every row read in this process, by the dataset code or by Reader
_FX: dict = {}


def _audit_dataset_reads():
    """Record every row the dataset code reads in this process, for Y3's audit. Forked loader workers
    read the same micro-batches the main process reads in Y5, and nothing else."""
    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    if getattr(TerraMeshTokenDataset._load_chunk, "audited", False):
        return
    inner = TerraMeshTokenDataset._load_chunk

    def load_chunk(self, modality, row):
        READ_ROWS.add(int(row))
        return inner(self, modality, row)
    load_chunk.audited = True
    TerraMeshTokenDataset._load_chunk = load_chunk


class Reader:
    """Blocks straight from tokens.npy and dates straight from dates.npy, offsets from the registry:
    none of the dataset's code."""

    def __init__(self, root: Path):
        from eo.terramesh_tok.contract import tok_dir_name
        self.tokens = {m: np.load(root / tok_dir_name(m) / "tokens.npy", mmap_mode="r") for m in MODS}
        self.dates = np.load(root / "temporal" / "dates.npy")

    def block(self, m: str, row: int) -> np.ndarray:
        READ_ROWS.add(int(row))
        info = TV.get_modality(m)
        body = np.asarray(self.tokens[m][int(row)], dtype=np.int64) + info.codebook_offset
        return np.concatenate([[info.bo_id], body, [info.eo_id]])

    def date(self, row: int):
        y, d = self.dates[int(row)]
        return int(y), int(d)


def fx() -> SimpleNamespace:
    """val_clean, its index, the dev locations and the Major TOM rows outside the held-out split."""
    if not _FX:
        from eo.data import temporal_dataset as T
        from eo.data.eval_split import load_eval_rows
        from eo.data.temporal_split import load_temporal_split
        from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
        _audit_dataset_reads()
        index = load_temporal_index(VAL)
        doc = load_temporal_split(VAL)
        held = load_eval_rows(root_dir=VAL)
        _FX.update(T=T, index=index, doc=doc, held=held, reader=Reader(VAL),
                   base=TerraMeshTokenDataset(root_dir=str(VAL), split="val", active_modalities=list(MODS),
                                              max_length=1048, modality_order="random", seed=42, rows=None),
                   dev=T.val_dev_location_ids(index),
                   majortom=np.setdiff1d(T._majortom_rows(VAL, index), held),
                   test_rows={int(r) for quad in doc["test"]["rows"] for r in quad},
                   dev_rows={int(r) for quad in doc["dev"]["rows"] for r in quad},
                   dev_quads={tuple(int(r) for r in quad) for quad in doc["dev"]["rows"]})
    return SimpleNamespace(**_FX)


def make(tcfg=None, per_step: int = 60, seed: int = 42, temporal_cls=None, single_cls=None, mixed_cls=None):
    """A training set on the dev locations, built as build_train builds it, with classes swappable for
    the controls."""
    f = fx()
    T = f.T
    tcfg = tcfg or T.TemporalConfig()
    temporal = (temporal_cls or T.TemporalLocations)(f.base, f.index, f.dev, tcfg, seed)
    single = (single_cls or T.SingleScenes)(f.base, f.index, f.majortom, tcfg, seed)
    return (mixed_cls or T.MixedMicrobatches)(temporal, single, tcfg, per_step, seed)


def small_cfg(**kw):
    """A ramp of the same shape, small enough to run whole epochs and HF's loop on CPU."""
    T = fx().T
    base = dict(stages=(T.Stage(2, 3, 1), T.Stage(3, 2, 1), T.Stage(4, 2, None)), scenes_per_microbatch=3,
                steps_per_epoch=3)
    base.update(kw)
    return T.TemporalConfig(**base)


_BO = {TV.get_modality(m).bo_id: m for m in MODS}
_BO[TV.BO_DATE] = "Date"


def parse(ids) -> list:
    """The blocks of a real token stream, from BO and EO ids alone: [(modality, start, end)]."""
    out, p, n = [], 0, len(ids)
    while p < n:
        m = _BO.get(int(ids[p]))
        if m is None:
            raise ValueError(f"no block starts at {p} (token {int(ids[p])})")
        if m == "Date":
            end = p + 2
            while end < min(n, p + 4) and int(ids[end - 1]) != TV.EO_DATE:
                end += 1
            if int(ids[end - 1]) != TV.EO_DATE:
                raise ValueError(f"the date block at {p} has no EO_Date within 4 tokens")
        else:
            info = TV.get_modality(m)
            end = p + info.tokens_per_sample + 2
            if end > n or int(ids[end - 1]) != info.eo_id:
                raise ValueError(f"the {m} block at {p} does not end in its EO")
        out.append((m, p, end))
        p = end
    return out


def real_ids(seq, b=None):
    am = seq["attention_mask"] if b is None else seq["attention_mask"][b]
    ids = seq["input_ids"] if b is None else seq["input_ids"][b]
    return ids[: int(am.sum())].numpy()


def grammar(draw, n: int) -> list:
    """Decisions 2, 3, 7 and 8 on a declared layout."""
    out, blocks = [], list(draw.blocks)
    stat = [i for i, b in enumerate(blocks) if b.modality in STATIC]
    if len(stat) != 2 or stat[1] != stat[0] + 1 or any(blocks[i].t != 0 for i in stat):
        return ["the statics are not one adjacent Coords-DEM pair from date 0's row"]
    if 0 < stat[0] and stat[1] < len(blocks) - 1 and blocks[stat[0] - 1].t == blocks[stat[1] + 1].t:
        out.append("the statics sit inside a group")
    runs = []
    for b in (b for i, b in enumerate(blocks) if i not in stat):
        if runs and runs[-1][0] == b.t:
            runs[-1][1].append(b)
        else:
            runs.append((b.t, [b]))
    order = [t for t, _ in runs]
    if len(order) != n or len(set(order)) != n:
        out.append(f"groups {order}: not {n} distinct dates, each one contiguous run")
    lulc = 0
    for t, g in runs:
        ms = [b.modality for b in g]
        img = [m for m in ms if m != "Date"]
        if sorted(m for m in img if m != "LULC") != sorted(GROUP):
            out.append(f"group {t} holds {img}")
        lulc += img.count("LULC")
        k = ms.count("Date")
        if k > 1 or (k == 1 and ms[0] != "Date" and ms[-1] != "Date"):
            out.append(f"group {t}: {k} date blocks, or one inside the group")
    if lulc != 1:
        out.append(f"LULC {lulc} times")
    if tuple(order) != tuple(draw.order) or list(draw.dates) != sorted(set(draw.dates)) or len(draw.dates) != n:
        out.append(f"groups {order} against the declared {draw.order}, dates {draw.dates}")
    if draw.order_mode == "chronological" and (stat[0] != 0 or order != sorted(order)):
        out.append("chronological, but the statics are not first or the dates not in order")
    if draw.order_mode == "leave_one_out" and (order[:-1] != sorted(order[:-1]) or order[-1] != draw.held):
        out.append("leave-one-out, but the others are not chronological or the held date is not last")
    return out


def parse_back(temporal, loc: int, epoch: int, n: int) -> list:
    """Y3 on one sequence: its tokens against the location's rows, read here, and its declared layout."""
    f = fx()
    seq = temporal.sequence(loc, epoch, n)
    draw = temporal.draw(loc, epoch, n)
    am, ids_all = seq["attention_mask"].numpy(), seq["input_ids"].numpy()
    real = int(am.sum())
    out = []
    if not (am[:real] == 1).all() or (ids_all[real:] != TV.PAD_ID).any():
        out.append("padding is not trailing PAD")
    rows4 = [int(r) for r in f.index.locations[int(loc)]]
    if tuple(rows4) not in f.dev_quads:
        out.append(f"location {loc}'s rows are not a dev quadruple of temporal_rows_v1.json")
    ids = ids_all[:real]
    try:
        blocks = parse(ids)
    except ValueError as e:
        return [f"loc {loc}: {e}"]
    mods = [m for m, _, _ in blocks]
    cands = []
    for m, s, e in blocks:
        if m == "Date":
            y, d = TV.decode_date_block(ids[s:e])
            c = [t for t in range(4) if f.reader.date(rows4[t])[1] == d
                 and (y is None or f.reader.date(rows4[t])[0] == y)]
        else:
            c = [t for t in range(4) if np.array_equal(f.reader.block(m, rows4[t]), ids[s:e])]
        if not c:
            out.append(f"loc {loc}: the {m} block at {s} matches no row of its location")
        cands.append(c)
    want = {"S2L2A": n, "S1GRD": n, "NDVI": n, "LULC": 1, "Coords": 1, "DEM": 1, "S1RTC": 0}
    out += [f"loc {loc}: {m} {mods.count(m)} times, not {k}" for m, k in want.items() if mods.count(m) != k]
    if mods.count("Date") > n:
        out.append(f"loc {loc}: {mods.count('Date')} date blocks for {n} dates")
    named = [c[0] for (m, _, _), c in zip(blocks, cands) if m == "S2L2A" and len(c) == 1]
    if len(set(named)) != n:
        out.append(f"loc {loc}: the S2L2A blocks name dates {named}, not {n} distinct ones")
    if [b.modality for b in draw.blocks] != mods:
        out.append(f"loc {loc}: the declared layout is not the token stream")
    else:
        for b, c, (m, s, e) in zip(draw.blocks, cands, blocks):
            if b.t not in c:
                out.append(f"loc {loc}: the {m} block at {s} is not from date {b.t}'s row")
            if m == "Date" and (e - s == 4) != b.with_year:
                out.append(f"loc {loc}: the date block at {s} drops its year against the layout")
    return out + [f"loc {loc}: {p}" for p in grammar(draw, n)]


# ---------------------------------------------------------------- Y3
def y3():
    f = fx()
    T = f.T
    ds = make()
    sample = [(int(loc), e) for e in range(4) for loc in f.dev[e::17][:24]]
    problems, seen = [], 0
    for loc, e in sample:
        n = ds.stage_of(e)[1].dates
        problems += parse_back(ds.temporal, loc, e, n)
        seen += 1
    record("Y3 parse-back", not problems, f"{seen} sequences at 2, 3 and 4 dates: every block is its own location's "
           f"row, read from tokens.npy, and the declared layout keeps the grammar"
           + (f"; {problems[:3]}" if problems else ""))

    class OffByOne(T.TemporalLocations):
        """Date 1's row read one row along, where that row is still a dev location's (never a test one)."""
        def rows(self, loc):
            r = super().rows(loc).copy()
            if int(r[1]) + 1 in fx().dev_rows:
                r[1] += 1
            return r

    class DateOffByOne(T.TemporalLocations):
        """Each date's blocks from the next date's row of the same location."""
        def rows(self, loc):
            return np.roll(super().rows(loc), -1)

    class TwoLulc(T.TemporalLocations):
        """LULC in two groups: another group's NDVI becomes LULC, so the length stays."""
        def draw(self, loc, epoch, n_dates, seed=None):
            d = super().draw(loc, epoch, n_dates, seed)
            blocks, done = list(d.blocks), False
            for i, b in enumerate(blocks):
                if not done and b.modality == "NDVI" and b.t != d.lulc_t:
                    blocks[i], done = replace(b, modality="LULC"), True
            return replace(d, blocks=tuple(blocks))

    fired = {}
    for name, cls in (("a row off by one", OffByOne), ("a date off by one", DateOffByOne),
                      ("LULC in two groups", TwoLulc)):
        bad = make(temporal_cls=cls)
        hit = ([(loc, e) for loc, e in sample
                if 1 in bad.temporal.draw(loc, e, bad.stage_of(e)[1].dates).dates
                and int(f.index.locations[loc][1]) + 1 in f.dev_rows]
               if name == "a row off by one" else sample[::4])
        flagged = sum(bool(parse_back(bad.temporal, loc, e, bad.stage_of(e)[1].dates)) for loc, e in hit)
        fired[name] = (flagged, len(hit))
    ok_ctrl = all(n > 0 and k == n for k, n in fired.values())
    record("Y3 controls", True, "parse-back against three broken datasets", ok_ctrl,
           ", ".join(f"{k}: {a} of {b} sequences flagged" for k, (a, b) in fired.items()))

    test_ids = ds.temporal.test_ids
    single_rows = ds.single.rows
    ok = (test_ids is not None and np.array_equal(ds.temporal.location_ids, f.dev)
          and not np.intersect1d(ds.temporal.location_ids, test_ids).size
          and not np.intersect1d(single_rows, f.held).size
          and not np.intersect1d(single_rows, f.index.locations.ravel()).size
          and len(single_rows) == 80384 - int((np.asarray(f.held) < 80384).sum()))
    refused_test = refused(T.TemporalLocations, f.base, f.index, np.append(f.dev[:3], test_ids[0]),
                           ds.cfg, 42)
    refused_seasonal = refused(T.SingleScenes, f.base, f.index, np.append(single_rows[:3], f.index.locations[0, 0]),
                               ds.cfg, 42)
    record("Y3 dev only", ok, f"{len(ds.temporal.location_ids)} dev locations of {len(f.index)}, none on the test "
           f"side ({len(test_ids)}); {len(single_rows)} Major TOM scene rows, none held out or seasonal",
           refused_test and refused_seasonal,
           f"a test location refused: {refused_test}; an ssl4eos12 row as a scene refused: {refused_seasonal}")


def y3_audit():
    """Run last: every row this gate read in this process, against the test side's rows."""
    f = fx()
    hit = READ_ROWS & f.test_rows
    record("Y3 no test row read", not hit, f"{len(READ_ROWS)} rows read in this process, {len(hit)} of them in the "
           f"{len(f.test_rows)} rows of the test side's 1,632 locations")


# ---------------------------------------------------------------- Y4
def expected_modality_ids(ids: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.int64)
    for m, s, e in parse(ids):
        out[s + 1: e - 1] = TV.MODALITY_TO_ID[m]
    return out


def five_key_problems(mb) -> list:
    if tuple(sorted(mb)) != tuple(sorted(KEYS)):
        return [f"keys {sorted(mb)}"]
    out = []
    shapes = {tuple(v.shape) for v in mb.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2 or any(v.dtype != torch.long for v in mb.values()):
        return [f"shapes {shapes}, dtypes {[v.dtype for v in mb.values()]}"]
    B, L = next(iter(shapes))
    for b in range(B):
        am = mb["attention_mask"][b].numpy()
        real = int(am.sum())
        ids, lab = mb["input_ids"][b].numpy(), mb["labels"][b].numpy()
        if not (am[:real] == 1).all() or (am[real:] != 0).any():
            out.append(f"row {b}: the mask is not a prefix")
        if not np.array_equal(lab[:real], ids[:real]) or (lab[real:] != -100).any():
            out.append(f"row {b}: labels do not follow the mask")
        if not np.array_equal(mb["modality_ids"][b].numpy(), expected_modality_ids(ids[:real], L)):
            out.append(f"row {b}: modality_ids do not label every body and Date")
    return out


def compose(overrides):
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision/eo_terramesh"), version_base=None):
        return hydra_compose(config_name=ARM, overrides=list(overrides))


DRY = ("dataset=terramesh_temporal", "multimodal.split=val", "max_length=2793", "total_batch_size=60",
       "per_device_train_batch_size=1", "per_device_eval_batch_size=1", "model_config.vocab_size=87664")


def y4():
    f = fx()
    T = f.T
    ds = make()
    mbs = []
    for e in (0, 1, 2):
        ds.set_draw_epoch(e)
        a = ds.assignment(e)
        for i in list(np.flatnonzero(a >= 0)[:3]) + list(np.flatnonzero(a < 0)[:2]):
            mbs.append(T.unwrap_microbatch([ds[int(i)]]))
    problems = [p for mb in mbs for p in five_key_problems(mb)]
    n_date = sum(int((mb["modality_ids"] == TV.MODALITY_TO_ID["Date"]).sum()) for mb in mbs)
    sixth = dict(mbs[0], location_ids=torch.zeros_like(mbs[0]["input_ids"]))
    blind = {k: v.clone() for k, v in mbs[0].items()}
    blind["modality_ids"][blind["modality_ids"] == TV.MODALITY_TO_ID["Date"]] = 0
    fired = {"a sixth key": bool(five_key_problems(sixth)) and refused(T.unwrap_microbatch, [sixth]),
             "Date bodies unlabelled": bool(five_key_problems(blind))}
    record("Y4 micro-batches", not problems, f"{len(mbs)} micro-batches over three stages and both streams: the closed "
           f"five keys, labels on the mask, modality_ids on every body ({n_date} Date tokens labelled "
           f"{TV.MODALITY_TO_ID['Date']})" + (f"; {problems[:3]}" if problems else ""),
           all(fired.values()), ", ".join(f"{k}: {'caught' if v else 'MISSED'}" for k, v in fired.items()))

    from lm_dataset.load_dataset import load_dataset_from_config, load_eval_dataset_from_config
    from lm_dataset.modality_registry import assert_vocab_size, is_eo
    cfg = compose(DRY)
    train = load_dataset_from_config(cfg)
    evals = load_eval_dataset_from_config(cfg)
    got, probs = {}, []
    for name, d in [("train", train)] + sorted(evals.items()):
        loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False, num_workers=0, collate_fn=d.collate_fn)
        mb = next(iter(loader))
        got[name] = tuple(mb["input_ids"].shape)
        probs += [f"{name}: {p}" for p in five_key_problems(mb)]
    single = evals["single"].base
    ok = (isinstance(train, T.MixedMicrobatches) and set(evals) == {"single", "temporal"}
          and isinstance(evals["single"], T.SingleEval) and isinstance(evals["temporal"], T.TemporalEval)
          and not probs and got["single"] == (16, 1048) and got["temporal"][0] == 8 and got["temporal"][1] <= 2793
          and len(single) == 4416 and np.array_equal(np.sort(single.rows), np.sort(f.held))
          and np.array_equal(evals["temporal"].temporal.location_ids, f.dev)
          and np.array_equal(train.temporal.location_ids, f.dev)
          and assert_vocab_size(cfg) == TV.TOTAL_VOCAB_SIZE and is_eo(cfg) and len(train) == 333 * 60)
    record("Y4 dry build", ok, f"{ARM} with {len(DRY)} overrides through load_dataset_from_config: train "
           f"{type(train).__name__} of {len(train)}, eval {sorted(evals)}; first micro-batches {got}, all the closed "
           f"five keys; vocab {assert_vocab_size(cfg)}" + (f"; {probs[:3]}" if probs else ""))


# ---------------------------------------------------------------- Y5
def _same(a, b) -> bool:
    return len(a) == len(b) and all(set(x) == set(y) and all(torch.equal(x[k], y[k]) for k in x) for x, y in zip(a, b))


def _direct(d, epoch, idx):
    d.set_draw_epoch(epoch)
    return [d[int(i)] for i in idx]


def _loaded(d, epoch, idx, workers):
    d.set_draw_epoch(epoch)
    loader = torch.utils.data.DataLoader(d, batch_size=1, sampler=[int(i) for i in idx], num_workers=workers,
                                         collate_fn=d.collate_fn)
    return list(loader)


def y5():
    f = fx()
    T = f.T
    ds = make()
    picks = {}
    for e in (0, 2):
        a = ds.assignment(e)
        picks[e] = list(np.flatnonzero(a >= 0)[:5]) + list(np.flatnonzero(a < 0)[:3])
    ref = {e: _direct(ds, e, idx) for e, idx in picks.items()}
    workers = {w: all(_same(ref[e], _loaded(ds, e, idx, w)) for e, idx in picks.items()) for w in (0, 2, 3)}
    other = make()
    rank = all(_same(ref[e], _direct(other, e, idx)) for e, idx in picks.items())
    locs = [int(x) for x in f.dev[:200]]
    new_draw = np.mean([ds.temporal.draw(loc, 2, 4).blocks != ds.temporal.draw(loc, 3, 4).blocks for loc in locs])
    new_mb = np.mean([not _same(_direct(ds, 2, [i]), _direct(ds, 3, [i])) for i in picks[2]])
    ok = all(workers.values()) and rank and new_draw > 0.95 and new_mb > 0.95
    detail = (f"{sum(len(v) for v in picks.values())} micro-batches at epochs 0 and 2 identical at "
              f"{[w for w, v in workers.items() if v]} workers and in a second instance: {rank}; epoch 3 against 2: "
              f"{new_draw:.0%} of 200 layouts and {new_mb:.0%} of micro-batches new")

    class WorkerKeyed(T.TemporalLocations):
        def _rng(self, seed, epoch, loc):
            info = torch.utils.data.get_worker_info()
            return np.random.default_rng((int(seed), int(epoch), int(loc), 0, info.id if info else 0))

    class WorkerKeyedScenes(T.SingleScenes):
        def _rng(self, seed, epoch, row):
            info = torch.utils.data.get_worker_info()
            return np.random.default_rng((int(seed), int(epoch), int(row), 1, info.id if info else 0))

    class EpochBlind(T.TemporalLocations):
        def _rng(self, seed, epoch, loc):
            return np.random.default_rng((int(seed), int(loc), 0))

    bad = make(temporal_cls=WorkerKeyed, single_cls=WorkerKeyedScenes)
    keyed = not _same(_direct(bad, 0, picks[0]), _loaded(bad, 0, picks[0], 2))
    blind = make(temporal_cls=EpochBlind)
    blind_new = np.mean([blind.temporal.draw(loc, 2, 4).blocks != blind.temporal.draw(loc, 3, 4).blocks for loc in locs])
    record("Y5 draws", ok, detail, keyed and blind_new < 0.95,
           f"keyed on the worker id: {'differs at 2 workers' if keyed else 'NOT SEEN'}; ignoring the epoch: "
           f"{blind_new:.0%} of layouts new")


# ---------------------------------------------------------------- Y6
def prop_problems(name: str, counts: dict, probs: dict, n: int, z_max: float = 4.5) -> list:
    if n == 0:
        return []           # nothing drawn in this conditional: the tally it conditions on reports that
    out = [f"{name}: {k} drawn, never expected" for k in counts if k not in probs]
    for k, p in probs.items():
        c = counts.get(k, 0)
        if p in (0, 1):
            if c != p * n:
                out.append(f"{name}={k}: {c} of {n} at p {p}")
            continue
        z = abs(c / n - p) / np.sqrt(p * (1 - p) / n)
        if z > z_max:
            out.append(f"{name}={k}: {c / n:.4f} against {p} (z {z:.1f})")
    return out


def _tally(values) -> dict:
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def location_stats(temporal, n_dates: int, cfg, n_draws: int = 10000) -> list:
    """Every draw of decisions 7, 8 and 10 against its probability."""
    from itertools import combinations
    f = fx()
    pairs = [(int(loc), e) for e in range(200) for loc in f.dev][:n_draws]
    draws = [temporal.draw(loc, e, n_dates) for loc, e in pairs]
    n = len(draws)
    probs = []
    probs += prop_problems("order", _tally(d.order_mode for d in draws), dict(zip(f.T.ORDER_MODES, cfg.order_probs)), n)
    modes = [m for d in draws for m in d.date_modes]
    probs += prop_problems("date", _tally(modes), dict(zip(f.T.DATE_MODES, cfg.date_probs)), len(modes))
    years = [y for d in draws for m, y in zip(d.date_modes, d.with_year) if m != "absent"]
    probs += prop_problems("year kept", _tally(years), {True: 1 - cfg.year_drop, False: cfg.year_drop}, len(years))
    probs += prop_problems("Coords first", _tally(d.statics[0] == "Coords" for d in draws), {True: .5, False: .5}, n)
    chrono = [d for d in draws if d.order_mode == "chronological"]
    probs += prop_problems("chronological statics slot", _tally(d.statics_slot for d in chrono), {0: 1}, len(chrono))
    other = [d for d in draws if d.order_mode != "chronological"]
    probs += prop_problems("statics slot", _tally(d.statics_slot for d in other),
                           {k: 1 / (n_dates + 1) for k in range(n_dates + 1)}, len(other))
    uniform_rank = {k: 1 / n_dates for k in range(n_dates)}
    probs += prop_problems("LULC date", _tally(d.dates.index(d.lulc_t) for d in draws), uniform_rank, n)
    loo = [d for d in draws if d.order_mode == "leave_one_out"]
    probs += prop_problems("held date", _tally(d.dates.index(d.held) for d in loo), uniform_rank, len(loo))
    uni = [d for d in draws if d.order_mode == "uniform"]
    probs += prop_problems("uniform first group", _tally(d.dates.index(d.order[0]) for d in uni), uniform_rank, len(uni))
    firsts = {3: [], 4: []}
    for d in draws:
        for t in d.order:
            g = [b.modality for b in d.blocks if b.t == t and b.modality not in STATIC + ("Date",)]
            firsts[len(g)].append(g[0])
    for k, vals in firsts.items():
        mods = list(GROUP) + (["LULC"] if k == 4 else [])
        probs += prop_problems(f"first of {k} in a group", _tally(vals), {m: 1 / k for m in mods}, len(vals))
    if n_dates < 4:
        subsets = {s: 1 / len(list(combinations(range(4), n_dates))) for s in combinations(range(4), n_dates)}
        probs += prop_problems(f"{n_dates}-date subset", _tally(d.dates for d in draws), subsets, n)
    return probs


def scene_stats(single, cfg, n_draws: int = 10000) -> list:
    f = fx()
    rows = f.majortom[:: max(1, len(f.majortom) // 2000)][:2000]
    draws = [single.draw(int(r), e) for e in range(5) for r in rows][:n_draws]
    n = len(draws)
    probs = prop_problems("scene date", _tally(d.date_mode for d in draws), dict(zip(f.T.DATE_MODES, cfg.date_probs)), n)
    present = [d for d in draws if d.date_mode != "absent"]
    probs += prop_problems("scene year kept", _tally(d.with_year for d in present),
                           {True: 1 - cfg.year_drop, False: cfg.year_drop}, len(present))
    probs += prop_problems("scene first modality", _tally(d.order[0] for d in draws),
                           {m: 1 / 6 for m in ("S2L2A", "S1RTC", "DEM", "NDVI", "LULC", "Coords")}, n)
    return probs


def y6():
    f = fx()
    T = f.T
    cfg = T.TemporalConfig()
    temporal = T.TemporalLocations(f.base, f.index, f.dev, cfg, 42)
    single = T.SingleScenes(f.base, f.index, f.majortom, cfg, 42)
    probs = {n: location_stats(temporal, n, cfg) for n in (4, 3, 2)}
    scenes = scene_stats(single, cfg)
    ok = not any(probs.values()) and not scenes
    detail = (f"10,000 location draws at each of 4, 3 and 2 dates and 10,000 scene draws: order, date mode, year, "
              f"statics order and slot, LULC's date, the held date, group order, modality order and date subsets "
              f"all within 4.5 sd of their probability")
    if not ok:
        detail += f"; {[p for v in probs.values() for p in v][:3] + scenes[:3]}"
    fired = {}
    for name, broken in (("date `last` at 0", replace(cfg, date_probs=(0.9, 0.0, 0.1))),
                         ("order `uniform` at 0", replace(cfg, order_probs=(0.5, 0.5, 0.0)))):
        bad = T.TemporalLocations(f.base, f.index, f.dev, broken, 42)
        fired[name] = len(location_stats(bad, 4, cfg))
    record("Y6 probabilities", ok, detail, all(fired.values()),
           ", ".join(f"{k}: {v} departures flagged" for k, v in fired.items()))


# ---------------------------------------------------------------- Y7
def y7():
    f = fx()
    T = f.T
    cfg = T.TemporalConfig()
    single = T.SingleScenes(f.base, f.index, f.majortom, cfg, 42)
    rows = [int(r) for r in f.majortom[:: len(f.majortom) // 64][:64]]
    same = dated = left_in = 0
    for r in rows:
        ref = f.base[r]                    # TerraMeshTokenDataset, its own order: (seed, worker 0, row)
        order = tuple(m for m, _, _ in parse(real_ids(ref)))
        mine = single.sequence(r, 0, draw=T.SceneDraw(r, order, "absent", True))
        same += all(torch.equal(mine[k], ref[k]) for k in KEYS)
        n_ref = int(ref["attention_mask"].sum())
        good = True
        for mode in ("first", "last"):
            for wy in (True, False):
                d = single.sequence(r, 0, draw=T.SceneDraw(r, order, mode, wy))
                ids = real_ids(d)
                (m, s, e), = [b for b in parse(ids) if b[0] == "Date"]
                keep = np.r_[0:s, e:len(ids)]
                year, day = TV.decode_date_block(ids[s:e])
                y_ref, d_ref = f.reader.date(r)
                good &= (np.array_equal(ids[keep], real_ids(ref))
                         and np.array_equal(d["position_ids"][:len(ids)].numpy()[keep], ref["position_ids"][:n_ref].numpy())
                         and np.array_equal(d["modality_ids"][:len(ids)].numpy()[keep], ref["modality_ids"][:n_ref].numpy())
                         and day == d_ref and (year == y_ref if wy else year is None)
                         and (s == 0) == (mode == "first"))
        dated += bool(good)
        kept = single.sequence(r, 0, draw=T.SceneDraw(r, order, "first", True))
        left_in += not all(torch.equal(kept[k], ref[k]) for k in KEYS)
    record("Y7 date dropped", same == len(rows) and dated == len(rows),
           f"{same} of {len(rows)} Major TOM scenes with the date absent equal TerraMeshTokenDataset's five tensors at its "
           f"order; with the date first or last, with and without its year, {dated} are T360's tokens, positions and "
           f"modality ids plus that row's date block", left_in == len(rows),
           f"the date block left in: {left_in} of {len(rows)} differ")


# ---------------------------------------------------------------- Y8
def classify(seq_ids: np.ndarray):
    mods = [m for m, _, _ in parse(seq_ids)]
    if "S1GRD" in mods and "S1RTC" not in mods:
        return ("temporal", mods.count("S2L2A"))
    if "S1RTC" in mods and "S1GRD" not in mods and mods.count("S2L2A") == 1:
        return ("single", 1)
    return ("neither", -1)


def homogeneity_problems(ds, epochs) -> tuple:
    out, n = [], 0
    for e in epochs:
        ds.set_draw_epoch(e)
        _, stage = ds.stage_of(e)
        for i in range(len(ds)):
            mb = ds[i]
            stream, ids, n_dates = ds.members(i)
            kinds = {classify(real_ids(mb, b)) for b in range(mb["input_ids"].shape[0])}
            size = stage.locations if stream == "temporal" else ds.cfg.scenes_per_microbatch
            if kinds != {(stream, n_dates)} or mb["input_ids"].shape[0] != size:
                out.append(f"epoch {e} index {i}: {stream} x {size} declared, {sorted(kinds)} x "
                           f"{mb['input_ids'].shape[0]} served")
            n += 1
    return out, n


def _tiny_model(accepts_loss_kwargs=None):
    from transformers import LlamaConfig

    from model.base_model.modeling_llama import LlamaForCausalLM
    torch.manual_seed(0)
    conf = LlamaConfig(vocab_size=TV.TOTAL_VOCAB_SIZE, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                       tie_word_embeddings=True)
    conf._attn_implementation = "sdpa"
    model = LlamaForCausalLM(conf).eval()
    if accepts_loss_kwargs is not None:
        model.accepts_loss_kwargs = accepts_loss_kwargs
    return model


def trim_problems(model, untrimmed, trimmed, tol: float = 1e-4) -> tuple:
    """Trimming may drop only padding, and no real logit may move."""
    out = []
    L = trimmed["input_ids"].shape[1]
    if any(not torch.equal(trimmed[k], untrimmed[k][:, :L]) for k in KEYS):
        out.append("the trimmed micro-batch is not a prefix of the untrimmed one")
    if not torch.equal(trimmed["attention_mask"].sum(1), untrimmed["attention_mask"].sum(1)):
        return out + ["trimming dropped real tokens"], float("nan")
    worst = 0.0
    with torch.no_grad():
        h = {name: model.model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"],
                               position_ids=mb["position_ids"]).last_hidden_state
             for name, mb in (("u", untrimmed), ("t", trimmed))}
        for b, real in enumerate(trimmed["attention_mask"].sum(1).tolist()):
            for s in range(0, real, 512):
                e = min(real, s + 512)
                lu, lt = model.lm_head(h["u"][b, s:e]), model.lm_head(h["t"][b, s:e])
                worst = max(worst, float((lu - lt).abs().max()))
    if worst > tol:
        out.append(f"real logits moved by {worst:.2e}")
    return out, worst


def y8():
    f = fx()
    T = f.T
    small = make(small_cfg(), per_step=2)
    problems, n = homogeneity_problems(small, range(4))
    real = make()
    split = {e: ((real.assignment(e) >= 0).sum(), (real.assignment(e) < 0).sum(),
                 np.array_equal(np.sort(real.assignment(e)[real.assignment(e) >= 0]), np.arange(real.n_temporal)))
             for e in range(4)}
    split_ok = all(t == real.n_temporal == s == real.n_single and full for t, s, full in split.values())
    sample = []
    for e in (0, 2):
        real.set_draw_epoch(e)
        a = real.assignment(e)
        for i in list(np.flatnonzero(a >= 0)[:3]) + list(np.flatnonzero(a < 0)[:6]):
            sample.append((e, int(i)))
    for e, i in sample:
        real.set_draw_epoch(e)
        problems += [f"real config, {p}" for p in homogeneity_problems_one(real, e, i)]

    class Mixed(T.MixedMicrobatches):
        """One scene padded into a micro-batch of locations."""
        def __getitem__(self, i):
            mb = super().__getitem__(i)
            stream, ids, n_dates = self.members(i)
            if stream != "temporal":
                return mb
            scene = self.single.sequence(int(self.single.rows[0]), self.epoch)
            L = mb["input_ids"].shape[1]
            pad = {k: torch.cat([v, torch.full((L - v.shape[0],), {"input_ids": TV.PAD_ID, "labels": -100}.get(k, 0),
                                                dtype=torch.long)]) for k, v in scene.items()}
            pad = {k: v[:L] for k, v in pad.items()}
            return {k: torch.cat([mb[k][:-1], pad[k][None]]) for k in KEYS}

    mixed_problems, _ = homogeneity_problems(make(small_cfg(), per_step=2, mixed_cls=Mixed), [0])
    record("Y8 homogeneous", not problems and split_ok,
           f"{n} micro-batches over 4 whole epochs of a small ramp, and {len(sample)} at the real config: every one "
           f"is one stream at its stage's dates and size; each of 4 epochs splits {real.n_temporal} + {real.n_single} "
           f"of {real.n}" + (f"; {problems[:3]}" if problems else ""),
           bool(mixed_problems), f"a mixed micro-batch: {len(mixed_problems)} flagged")

    def measured_share(d, e, k=30):
        d.set_draw_epoch(e)
        a = d.assignment(e)
        tok = {s: np.mean([int(d[int(i)]["attention_mask"].sum()) for i in idx[:k]])
               for s, idx in (("temporal", np.flatnonzero(a >= 0)), ("single", np.flatnonzero(a < 0)))}
        single = d.n_single * tok["single"]
        return single / (single + d.n_temporal * tok["temporal"])

    shares = {e: (measured_share(real, e), real.token_share(e)["single_token_share"]) for e in (0, 1, 2)}
    share_ok = all(abs(m - real.cfg.single_share) <= 0.02 and abs(m - x) <= 0.005 for m, x in shares.values())
    half = make(replace(T.TemporalConfig(), scenes_per_microbatch=8))
    half_share = measured_share(half, 2)
    record("Y8 token share", share_ok, "Major TOM's measured share of real tokens by stage (2, 3, 4 dates): "
           + ", ".join(f"{m:.3f} (expected {x:.3f})" for m, x in shares.values())
           + f", against single_share {real.cfg.single_share} +- 0.02", abs(half_share - 0.5) > 0.02,
           f"8 scenes per micro-batch: {half_share:.3f}")

    from omegaconf import OmegaConf
    from transformers import TrainingArguments

    from util.trainer_pt import EOTrainer
    real.set_draw_epoch(0)
    temporal_mb = T.stack_trim([real.temporal.sequence(int(f.dev[0]), 0, 2)])
    single_mb = T.stack_trim([real.single.sequence(int(r), 0) for r in f.majortom[:2]])

    def weighting(model):
        tmp = tempfile.mkdtemp(prefix="y8_")
        args = TrainingArguments(output_dir=tmp, use_cpu=True, report_to=[], per_device_train_batch_size=1,
                                 gradient_accumulation_steps=2, remove_unused_columns=False)
        tr = EOTrainer(model=model, args=args, cfg=OmegaConf.create({"dataset": "terramesh_temporal"}),
                       data_collator=T.unwrap_microbatch)
        # HF's own step: get_batch_samples counts the step's label tokens only for a model that takes
        # loss kwargs, and training_step divides by the accumulation only for one that does not.
        _, num_items = tr.get_batch_samples(iter([temporal_mb, single_mb]), 2, "cpu")
        n_items = sum(int(mb["labels"].ne(-100).sum()) for mb in (temporal_mb, single_mb))
        rel = []
        with torch.no_grad():
            for mb in (temporal_mb, single_mb):
                loss = tr.compute_loss(model, dict(mb), num_items_in_batch=num_items)
                if not tr.model_accepts_loss_kwargs:
                    loss = loss / args.gradient_accumulation_steps
                logits = model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"],
                               position_ids=mb["position_ids"]).logits
                ce = float(torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                             mb["labels"][:, 1:].reshape(-1), ignore_index=-100,
                                                             reduction="sum"))
                rel.append(abs(float(loss) - ce / n_items) / (ce / n_items))
        shutil.rmtree(tmp, ignore_errors=True)
        counted = num_items is not None and int(num_items) == n_items
        return tr.model_accepts_loss_kwargs and counted, max(rel), n_items

    model = _tiny_model()
    accepts, rel, num_items = weighting(model)
    refuses, rel_bad, _ = weighting(_tiny_model(accepts_loss_kwargs=False))
    record("Y8 weighting", accepts and rel < 1e-5,
           f"EOTrainer passes num_items_in_batch ({num_items} label tokens over a temporal and a single micro-batch): "
           f"each micro-batch's loss is its token sum over the step's tokens, to {rel:.1e}, so each stream's share of "
           f"the gradient is its share of tokens", (not refuses) and rel_bad > 0.1,
           f"a model that refuses loss kwargs: each micro-batch weighs as its own mean, {rel_bad:.0%} off")

    pick = [int(loc) for loc in f.dev if int(real.temporal.sequence(int(loc), 0, 2)["attention_mask"].sum()) < 1597][:2]
    seqs = [real.temporal.sequence(loc, 0, 2) for loc in pick]
    scenes = [real.single.sequence(int(r), 0) for r in f.majortom[2:4]]
    worst, probs = 0.0, []
    for items in (seqs, scenes):
        p, w = trim_problems(model, T.stack_trim(items, trim=False), T.stack_trim(items))
        probs += p
        worst = max(worst, w)
    untrimmed = T.stack_trim(seqs, trim=False)
    cut = {k: v[:, : int(untrimmed["attention_mask"].sum(1).max()) - 1] for k, v in untrimmed.items()}
    bad, _ = trim_problems(model, untrimmed, cut)
    lengths = [int(s["attention_mask"].sum()) for s in seqs] + [int(s["attention_mask"].sum()) for s in scenes]
    record("Y8 trimming", not probs, f"two locations at 2 dates ({lengths[:2]} real of 1,597) and two scenes "
           f"({lengths[2:]} of 1,048), trimmed and not, through a 2-layer model at fp32: real logits within "
           f"{worst:.1e}" + (f"; {probs}" if probs else ""), bool(bad), f"a trim one token short: {bad}")


def homogeneity_problems_one(ds, e: int, i: int) -> list:
    _, stage = ds.stage_of(e)
    mb = ds[i]
    stream, ids, n_dates = ds.members(i)
    kinds = {classify(real_ids(mb, b)) for b in range(mb["input_ids"].shape[0])}
    size = stage.locations if stream == "temporal" else ds.cfg.scenes_per_microbatch
    if kinds != {(stream, n_dates)} or mb["input_ids"].shape[0] != size:
        return [f"epoch {e} index {i}: {stream} x {size} declared, {sorted(kinds)} served"]
    return []


# ---------------------------------------------------------------- Y14
def y14():
    f = fx()
    T = f.T
    from omegaconf import OmegaConf
    from transformers import Trainer, TrainingArguments

    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    from util.trainer_pt import EOTrainer
    cfg = T.TemporalConfig()
    held_mt = [int(r) for r in f.held[:2]]
    single = T.SingleEval(TerraMeshTokenDataset(root_dir=str(VAL), split="val", active_modalities=list(MODS),
                                                max_length=1048, modality_order="random", seed=42, rows=held_mt), 1)
    temporal = T.TemporalEval(T.TemporalLocations(f.base, f.index, f.dev[:2], cfg, 42), 1, 2)
    model = _tiny_model()

    class Unguarded(EOTrainer):
        """The mixin's evaluate as it was before the guard."""
        def evaluate(self, *args, **kwargs):
            for name in self.modality_ev_loss:
                self.modality_ev_loss[name].zero_()
                self.modality_ev_count[name].zero_()
            metrics = Trainer.evaluate(self, *args, **kwargs)
            prefix = kwargs.get("metric_key_prefix", "eval")
            extra = {}
            for name in list(self.modality_ev_loss.keys()):
                tot = self._nested_gather(self.modality_ev_loss[name]).sum().item()
                cnt = self._nested_gather(self.modality_ev_count[name]).sum().item()
                if cnt > 0:
                    extra[f"{prefix}_loss_{name}"] = round(tot / cnt, 4)
            if extra and isinstance(metrics, dict):
                metrics.update(extra)
                self.log(extra)
            return metrics

    def keys(cls):
        tmp = tempfile.mkdtemp(prefix="y14_")
        args = TrainingArguments(output_dir=tmp, use_cpu=True, report_to=[], per_device_eval_batch_size=1,
                                 remove_unused_columns=False, prediction_loss_only=True, dataloader_num_workers=0,
                                 disable_tqdm=True)
        tr = cls(model=model, args=args, eval_dataset={"single": single, "temporal": temporal},
                 cfg=OmegaConf.create({"dataset": "terramesh_temporal"}), data_collator=T.unwrap_microbatch)
        metrics = tr.evaluate()
        logged = {k for entry in tr.state.log_history for k in entry}
        shutil.rmtree(tmp, ignore_errors=True)
        return set(metrics) | logged

    got = keys(EOTrainer)
    plain = sorted(k for k in got if k.startswith("eval_loss_"))
    per_set = {s: sorted(k[len(f"eval_{s}_loss_"):] for k in got if k.startswith(f"eval_{s}_loss_"))
               for s in ("single", "temporal")}
    ok = (not plain and {"eval_single_loss", "eval_temporal_loss"} <= got
          and set(per_set["single"]) == {"S2L2A", "S1RTC", "DEM", "NDVI", "LULC", "Coords"}
          and {"S2L2A", "S1GRD", "NDVI", "LULC", "DEM", "Coords"} <= set(per_set["temporal"])
          and set(per_set["temporal"]) <= {"S2L2A", "S1GRD", "NDVI", "LULC", "DEM", "Coords", "Date"})
    bad = sorted(k for k in keys(Unguarded) if k.startswith("eval_loss_"))
    record("Y14 eval keys", ok, f"two eval sets through EOTrainer.evaluate: single logs {per_set['single']}, temporal "
           f"logs {per_set['temporal']}, each under its own prefix; plain eval_loss_<mod>: {plain or 'none'}",
           bool(bad), f"the guard removed: {bad}")


# ---------------------------------------------------------------- Y16, the dataset half
def lattice_problems(ids: np.ndarray, pos: np.ndarray) -> list:
    blocks = parse(ids)
    want = np.empty(len(ids), dtype=np.int64)
    cur = 0
    for m, s, e in blocks:
        want[s:e] = np.arange(cur, cur + e - s)
        if m != "Date":
            cur += e - s
    out = [] if np.array_equal(pos, want) else ["position_ids are not the lattice"]
    imgs = [(s, k) for k, (m, s, e) in enumerate(blocks) if e - s == 198]
    for a in range(len(imgs)):
        for b in range(a + 1, len(imgs)):
            (sa, ka), (sb, kb) = imgs[a], imgs[b]
            coords = any(blocks[k][0] == "Coords" for k in range(ka + 1, kb))
            gap = 198 * (b - a) + 5 * coords
            if not (pos[sb: sb + 198] - pos[sa: sa + 198] == gap).all():
                out.append(f"{blocks[ka][0]} -> {blocks[kb][0]}: patches {sorted(set((pos[sb:sb + 198] - pos[sa:sa + 198]).tolist()))[:3]} "
                           f"apart, not {gap}")
    return out


def variant_positions(ids: np.ndarray, rewinds) -> np.ndarray:
    pos, cur = np.empty(len(ids), dtype=np.int64), 0
    for m, s, e in parse(ids):
        pos[s:e] = np.arange(cur, cur + e - s)
        if m not in rewinds:
            cur += e - s
    return pos


def y16():
    f = fx()
    ds = make()
    seqs = [ds.temporal.sequence(int(loc), e, ds.stage_of(e)[1].dates) for e in range(4) for loc in f.dev[e::29][:16]]
    seqs += [ds.single.sequence(int(r), e) for e in range(2) for r in f.majortom[e::4000][:16]]
    probs, pairs = [], 0
    # A broken rule can only show where it changes a position: Coords, or a date block, before an image block.
    controls = {"rewound over Coords": [0, 0], "sequential": [0, 0]}
    for s in seqs:
        n = int(s["attention_mask"].sum())
        ids, pos = s["input_ids"][:n].numpy(), s["position_ids"][:n].numpy()
        probs += lattice_problems(ids, pos)
        blocks = parse(ids)
        k = sum(1 for m, a, b in blocks if b - a == 198)
        pairs += k * (k - 1) // 2
        last_image = max(i for i, (m, a, b) in enumerate(blocks) if b - a == 198)
        for name, rewinds, before in (("rewound over Coords", ("Date", "Coords"), "Coords"), ("sequential", (), "Date")):
            if any(m == before for m, a, b in blocks[:last_image]):
                controls[name][1] += 1
                controls[name][0] += bool(lattice_problems(ids, variant_positions(ids, rewinds)))
    has_dates = sum(1 for s in seqs if (s["input_ids"] == TV.BO_DATE).any())
    record("Y16 lattice (dataset)", not probs, f"{len(seqs)} sequences ({has_dates} with date blocks), {pairs} pairs of "
           f"image blocks: every patch sits 198n from its counterpart, 198n + 5 with Coords between"
           + (f"; {probs[:3]}" if probs else ""),
           all(n > 0 and k == n for k, n in controls.values()),
           ", ".join(f"{k}: {a} of the {b} sequences it can affect flagged" for k, (a, b) in controls.items()))


# ---------------------------------------------------------------- Y17
def fingerprint(input_ids, position_ids) -> tuple:
    """(hash, stream, sequences, dates) of a micro-batch, from its tokens."""
    h = hashlib.sha1(input_ids.numpy().tobytes() + position_ids.numpy().tobytes()).hexdigest()[:16]
    first = input_ids[0].numpy()
    kind = "temporal" if (first == TV.get_modality("S1GRD").bo_id).any() else "single"
    return h, kind, int(input_ids.shape[0]), int((first == TV.get_modality("S2L2A").bo_id).sum())


class Probe(torch.nn.Module):
    """A model that records what HF's loop feeds it: one fingerprint per micro-batch."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward(self, input_ids=None, attention_mask=None, labels=None, position_ids=None, modality_ids=None):
        self.seen.append(fingerprint(input_ids, position_ids))
        return {"loss": (self.w - 1.0).pow(2).sum()}


def run_loop(tcfg, out: Path, workers: int, callback: str = "temporal", mixed_cls=None, resume=None,
             max_steps: int = 12, ga: int = 2) -> dict:
    """HF's own training loop, through the production trainer mixin (its sampler hook and logging),
    saving every step: {global_step: [fingerprints]}. `callback` "rounding" applies
    SetEpochCallback's rule, int(round(state.epoch)), to the dataset's epoch."""
    from omegaconf import OmegaConf
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.utils import logging as hf_logging

    from util.trainer_pt import PerModalityLossMixin
    hf_logging.set_verbosity_error()
    ds = make(tcfg, per_step=ga, mixed_cls=mixed_cls)
    probe = Probe()

    class ProbeTrainer(PerModalityLossMixin, Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.init_per_modality(OmegaConf.create({"dataset": "terramesh_temporal"}))

    class Steps(TrainerCallback):
        def __init__(self):
            self.at, self.steps = 0, {}

        def on_step_end(self, args, state, control, **kwargs):
            self.steps[state.global_step] = probe.seen[self.at:]
            self.at = len(probe.seen)

    class Rounding(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            ds.set_draw_epoch(int(round(state.epoch or 0)))

    log = Steps()
    epoch_cb = ds.epoch_callback() if callback == "temporal" else Rounding()
    args = TrainingArguments(output_dir=str(out), per_device_train_batch_size=1, gradient_accumulation_steps=ga,
                             max_steps=max_steps, save_strategy="steps", save_steps=1, logging_steps=1, report_to=[],
                             use_cpu=True, dataloader_num_workers=workers, seed=42, data_seed=42,
                             remove_unused_columns=False, learning_rate=1e-3, lr_scheduler_type="constant",
                             disable_tqdm=True, log_level="error")
    trainer = ProbeTrainer(model=probe, args=args, train_dataset=ds, data_collator=ds.collate_fn,
                           callbacks=[epoch_cb, log])
    trainer.train(resume_from_checkpoint=None if resume is None else str(resume))
    return log.steps


def skip_loader_epochs(ds, rank: int, epoch: int = 1, skip: int = 2) -> tuple:
    """accelerate's resume path on rank `rank` of two, with no process group: a DataLoaderShard over a
    BatchSamplerShard, then skip_first_batches. (dataset epoch after it, positions it served)."""
    from accelerate.data_loader import BatchSamplerShard, DataLoaderShard, skip_first_batches
    from accelerate.state import PartialState
    PartialState(cpu=True)
    sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(ds), 1, False)
    loader = DataLoaderShard(ds, batch_sampler=BatchSamplerShard(sampler, num_processes=2, process_index=rank),
                             collate_fn=ds.collate_fn)
    ds.set_draw_epoch(epoch)
    loader.set_epoch(epoch)                                  # HF's loop, then its resume
    served = [fingerprint(mb["input_ids"], mb["position_ids"]) for mb in skip_first_batches(loader, skip)]
    return ds.epoch, served


def y17():
    f = fx()
    T = f.T
    tcfg = small_cfg()
    spe, ga = tcfg.steps_per_epoch, 2
    tmp = Path(tempfile.mkdtemp(prefix="y17_"))
    ref = run_loop(tcfg, tmp / "ref", 0)
    # The uninterrupted run against the dataset itself: step s reads epoch (s - 1) // 3, in index order.
    direct = make(tcfg, per_step=ga)
    order_bad = stage_bad = 0
    for s, fps in ref.items():
        e = (s - 1) // spe
        direct.set_draw_epoch(e)
        first = ((s - 1) % spe) * ga
        want = [fingerprint(mb["input_ids"], mb["position_ids"]) for mb in (direct[first + j] for j in range(ga))]
        order_bad += fps != want
        _, st = tcfg.stage_of(e)
        stage_bad += sum((kind == "temporal" and (nd, b) != (st.dates, st.locations)) or (kind == "single" and b != 3)
                         for h, kind, b, nd in fps)
    runs = {"uninterrupted, 2 workers": (run_loop(tcfg, tmp / "w2", 2), 1),
            "resumed at step 2 (epoch 0.67, stage 0), 2 workers": (
                run_loop(tcfg, tmp / "r2", 2, resume=tmp / "ref/checkpoint-2"), 3),
            "resumed at step 5 (epoch 1.67, stage 1), 0 workers": (
                run_loop(tcfg, tmp / "r5", 0, resume=tmp / "ref/checkpoint-5"), 6),
            "resumed at step 4 (epoch 1.33), 2 workers": (
                run_loop(tcfg, tmp / "r4", 2, resume=tmp / "ref/checkpoint-4"), 5),
            "resumed at step 3 (a stage boundary), 2 workers": (
                run_loop(tcfg, tmp / "r3", 2, resume=tmp / "ref/checkpoint-3"), 4)}

    def agree(steps, first):
        return all(steps.get(s) == ref[s] for s in range(first, max(ref) + 1))

    # Two ranks, through accelerate's own resume path: the dataset keeps its epoch, and each rank serves
    # the rest of its shard of epoch 1, which together are the epoch's micro-batches after the skipped ones.
    ranks = {r: skip_loader_epochs(make(tcfg, per_step=ga), r) for r in (0, 1)}
    direct.set_draw_epoch(1)
    want = {r: [fingerprint(direct[p]["input_ids"], direct[p]["position_ids"]) for p in range(r + 4, 6, 2)]
            for r in (0, 1)}
    ranks_ok = all(ep == 1 and served == want[r] for r, (ep, served) in ranks.items())
    ok = len(ref) == 12 and not order_bad and not stage_bad and all(agree(r, k) for r, k in runs.values()) and ranks_ok
    detail = (f"12 steps of a 3-stage ramp, 3 steps an epoch, through the production trainer mixin: the uninterrupted "
              f"run is the dataset's own order at each epoch's stage ({order_bad} steps and {stage_bad} micro-batches "
              f"off); " + "; ".join(f"{name}: {'same' if agree(r, k) else 'DIFFERS'} from step {k}"
                                    for name, (r, k) in runs.items())
              + f"; two ranks through accelerate's skip loader at epoch 1: epochs {[ranks[r][0] for r in (0, 1)]}, "
              f"shards {'as expected' if ranks_ok else 'WRONG'}")

    class CountingStage(T.MixedMicrobatches):
        """The stage read from a counter of micro-batches served by this process."""
        served = 0

        def stage_of(self, epoch):
            return self.cfg.stage_of(self.served // self.n)

        def temporal_cursor(self, epoch):
            return super().temporal_cursor(self.served // self.n)

        def __getitem__(self, i):
            out = super().__getitem__(i)
            self.served += 1
            return out

    class HFShuffled(T.MixedMicrobatches):
        """The epoch's order left to HF's sampler."""
        orders_itself = False

    class WithSetEpoch(T.MixedMicrobatches):
        """The setter under the name accelerate calls."""
        def set_epoch(self, epoch):
            self.set_draw_epoch(epoch)

    rounding = run_loop(tcfg, tmp / "c_round", 0, callback="rounding", resume=tmp / "ref/checkpoint-2")
    counting = run_loop(tcfg, tmp / "c_count", 0, mixed_cls=CountingStage, resume=tmp / "ref/checkpoint-3")
    shuffled_ref = run_loop(tcfg, tmp / "c_shuf", 0, mixed_cls=HFShuffled)
    shuffled = run_loop(tcfg, tmp / "c_shuf5", 0, mixed_cls=HFShuffled, resume=tmp / "c_shuf/checkpoint-5")
    exposed = {r: skip_loader_epochs(make(tcfg, per_step=ga, mixed_cls=WithSetEpoch), r)[0] for r in (0, 1)}
    fired = {"SetEpochCallback's rounded epoch, resumed at step 2": not agree(rounding, 3),
             "the stage from a counter, resumed at step 3": not agree(counting, 4),
             "HF's shuffled sampler, resumed at step 5": any(shuffled.get(s) != shuffled_ref[s] for s in range(6, 13)),
             "a dataset with set_epoch, two ranks": all(e != 1 for e in exposed.values())}
    shutil.rmtree(tmp, ignore_errors=True)
    record("Y17 ramp and resume", ok, detail, all(fired.values()),
           ", ".join(f"{k}: {'differs' if v else 'SAME'}" for k, v in fired.items())
           + f" (the skip loader set its epoch to {sorted(set(exposed.values()))})")


CHECKS = {"Y1": y1, "Y3": y3, "Y4": y4, "Y5": y5, "Y6": y6, "Y7": y7, "Y8": y8, "Y9": y9, "Y14": y14,
          "Y16": y16, "Y17": y17}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(CHECKS, key=lambda k: int(k[1:])),
                    default=sorted(CHECKS, key=lambda k: int(k[1:])))
    args = ap.parse_args()
    for name in args.only:
        CHECKS[name]()
    if _FX:
        y3_audit()
    failed = [c for c, ok, ctrl in RESULTS if ok is False or ctrl is False]
    skipped = [c for c, ok, _ in RESULTS if ok is None]
    print(f"\n{len(RESULTS) - len(failed) - len(skipped)} passed, {len(failed)} failed, {len(skipped)} skipped"
          + (f": {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
