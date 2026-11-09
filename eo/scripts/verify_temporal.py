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
  Y2  (Step 6) Step 2's index checks, rerun: both indices load through load_temporal_index; the
      structural checks, recomputed from each root's metadata (val's, and train_meta's with
      train_coords' tok_index.parquet, in a child process: 9.3 GB), find no offence and equal
      temporal_build.json, and val_clean holds exactly temporal_rows_v1.json's locations; the token
      checks on val_clean (presence, statics, twins) give temporal_check.json's report, through
      temporal_index.token_report, which writes nothing; an untampered copy of the index loads
                                     | every structural control by its own offence, the wrong day
                                       rule, the split with a pair swapped; the token controls (one
                                       pair, 1% and every location swapped); the loader refuses another
                                       row order, a changed array, a missing, failed, partial or stale
                                       check and no build file; rows swapped with the hashes made
                                       consistent fail the token report
      ⚠ Train's token checks need g49's tokens: SKIPPED here, rerun in Step 8. The val_clean token
        checks read the static tokens and presence flags of all 2,176 locations, test side included,
        as Step 2's check did: integrity only; no model reads them, and Y3's audit counts the rows the
        dataset code and the parse-backs read.
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
  Y4  Every micro-batch, and the dry build through load_dataset_from_config with T360-T's arm as
      shipped (multimodal.split=val its one override, since Step 5), is the closed five keys;
      modality_ids labels every body and Date; labels follow the mask
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

Step 5 (continuation through pretrain_init, the arm, preflight; preflight's own checks have their
controls in eo/train/preflight_controls.py):

  Y18 arm_t360t_vanilla differs from arm_t360_vanilla in exactly its intended keys, each at its
      intended value, the temporal block naming every TemporalConfig field at the plan's value; the
      keys it restates or inherits as T360's hold; the builder's own refusals pass on it; its stop
      meets a save and an eval (HF does neither at stop_steps alone). The pilot's two other arms
      differ from it in exactly their own key. The data record pretrain.py writes is the one
      preflight compares, holds no schedule key, and refuses a changed design on a resume
                                     | a stray seed; stop_steps 1331 (a changed value, and a stop
                                       that meets no save); the block without year_drop; a stray key
                                       in each pilot arm; a resume with single_share 0.4, seed 43,
                                       no eval split, or the no-ramp arm's stages

Step 6 (the gate's own checks, 2026-11-09). Y10, Y11, Y12 and Y16 read the real T360-T, built once on
CPU at fp32 from the arm and loaded through pretrain_init's `continue` hook (~6 GB with Y10's two
other builds; 12 GB peak). Y11 and Y15 share one set of pretrain.py runs on a miniature T360-T. The whole gate
takes about 11 minutes on this workstation's CPU.

  Y10 Logits at shared positions are equal at max_length 1,048 and 2,793: T360-T built from the arm and
      with max_length=1048, the same extended base in both, fed a scene micro-batch and a two-date
      sequence cut to 1,048 (date blocks, lattice positions); and the whole two-date sequence at 2,793
      against the cut. RoPE has no position table: max_length only reaches max_position_embeddings
                                     | model_config.rope_theta 20000 moves them
  Y11 Continuation. The hook on the real model: 290 tensors loaded strictly, lm_head the only missing
      key, a date row the base's, the tie kept, every parameter training, the record naming the base,
      `continue` and T360's checkpoint, and accepted on a resume under the same init. Through
      pretrain.py on the miniature: a run stopped on a save (stop_steps a multiple of save_steps) and
      resumed with only stop_steps changed logs the uninterrupted run's learning_rate at every step of
      a schedule that changes it at every step (warmup 3, decay 3 of 6), and its loss within 1e-4
                                     | the record refuses another base and another variant; the resume
                                       with scheduler.pt removed (HF then loads neither scheduler nor
                                       optimizer, silently) changes the LR and the loss; pretrain.py
                                       refuses a base whose weights no longer hash to extension.json and
                                       a resume under another base's config, exit 1, nothing written
  Y12 The extended base reproduces T360's 5.0361 on the 4,416 held-out rows within 0.01 (measured
      +4.5e-09, where +0.001 was expected: the new rows sit at the mean row, whose logit lies far below the
      log-sum-exp, so they hold almost no softmax mass), read from the GPU run's artifact (eval_forgetting.py --model
      temporal, fp32, the GTX Titan X) and paired per token against T360's BEN-base reading (same rows,
      layout and worker-keyed modality orders), every token within one fp16 step (at least 1e-4 nats);
      live on CPU, the first two batches (8 Major TOM rows) reproduce the artifact's per-token CE within
      the same. SKIPPED while the artifact is missing
                                     | the table shifted by one row (every id reads its neighbour's) reads
                                       far off on those rows. Step 3's control 1, the rows inserted
                                       before PAD, cannot fire here: it moves only PAD, which these rows
                                       never score
  Y13 Each task's prompt (eo/eval/temporal.py, Step 12's four tasks at every target date, on dev
      locations) is the prefix of its sequence through the target's BO, assembled by
      TemporalLocations.sequence(draw=...) on the lattice, and equal to its context assembled alone;
      every block is its declared row's (tokens.npy), and nothing of the target's (modality, date) is in
      it, by declaration and by tokens; the layouts keep the defaults; without date blocks the positions
      are T360's; a test location is refused before any read
                                     | the target left in context; a prompt on sequential positions; a
                                       context one token short of the prompt
  Y15 Two real steps and a resume through pretrain.py, launched as train_eo.sh launches it, on a
      miniature T360-T (T360's arm, two layers of 96, extended to 87,664 through extend_checkpoint's own
      writer and identity check), CPU, fp32: pretrain_init.json and temporal_data.json written at
      launch, the latter the config's data settings; TemporalEpochCallback reports the draw epoch from
      global_step at every epoch begun (0, 1, 2; 1 and 2 after a resume at step 3), with HF's epoch
      length checked and unwrap_microbatch collating; stopped on a save and resumed with only stop_steps
      changed: accepted
                                     | a flipped init hash: exit 1, nothing trained or recorded; a resume
                                       with temporal.year_drop changed: exit 1, the run untouched; the
                                       callback refuses an epoch of another length and HF's default
                                       collator
  Y16 (generation half) On T360-T, a left-out-season prompt behind four date blocks (the lattice 16
      positions behind the token count): each KV-cached step with explicit positions
      (eo/generate/cached.py) has the logits of a full teacher-forced pass at the same positions; the
      sampler's explicit 0..L-1 path gives its default path's tokens bit for bit
                                     | the same cached steps on default positions

  ⚠ Y3's audit covers this process. Y15's pretrain.py runs read their rows in child processes: the
    dev locations and the Major TOM rows outside the held-out split (the dataset refuses a test
    location), and they evaluate nothing. The miniature's identity batch is the real one's less its
    row on a test location (row 84,920, a held-out row), topped up with a dev-side held-out row.

Step 7 (the evaluation tools: eo/eval/temporal.py's scoring, eo/scripts/eval_temporal.py, the frozen
layouts eo/eval/temporal_layouts_v1.json, 2026-11-09). Y20 reads the real T360-T base on CPU.

  Y19 The layout file is frozen (its hash pinned in eo/eval/temporal.FROZEN_LAYOUTS) and equal to its
      sources: every Step 0 layout to Step 0's own functions (t360_context, t360_best_layouts,
      t360_lattice's profile), every T360-T task layout to task_blocks (Step 6's defaults), the target
      dates to t360_context.date_of; T360's baselines are named for every task and target, layouts of
      the file without date blocks and inside T360's working range, and forecast's and gap fill's hold
      no block T360-T's prompt lacks; the date task's baseline is the prior
                                     | a Step 0 layout with two blocks swapped; the file edited by one
                                       block (refused by its pinned hash); a target in its own context; a
                                       date block in T360's forecast baseline
  Y20 The scorer's three paths, on T360-T's base, CPU, fp32, Step 0's first locations and target dates:
      the target path (run_pass, labels masked to the target block) gives Step 0's stored per-location
      CE (T360 itself, Titan X) within 1e-4; the profile path's block means (run_pass's fp16 per-token
      arrays) are within one fp16 step of fp32 block means taken directly; the date path's CE of Y and
      D and year distribution equal a direct log-softmax within 1e-4
                                     | the same units at the next target date; labels left on every
                                       body; a profile's blocks off by one; a date block without its
                                       year
  Y21 The wrong date (the date probe's settings from the file, and every date block moved, and a year
      moved) changes exactly the moved date's tokens, in input_ids and labels alike, and nothing else:
      positions, modality ids, mask and the prompt as the sequence's prefix
                                     | a mover that moves the first date block instead; one that also
                                       shifts the target's positions; one that leaves the labels; a
                                       wrong date on the date task's own target, and on a sequence
                                       without date blocks, refused
  Y23 The cache at the task prompts' length on the GTX Titan X, from eval_temporal.py cache's artifact:
      T360 on the gap-fill prompt without date blocks (2,382 tokens) and T360-T's base on it with them
      (2,398, lattice positions), every one of the 197 steps cached against a full pass, at fp32 within
      1e-3 and at fp16 within 5e-2 or 4 fp16 steps at the largest |logit| (eo/eval/ben.py), all finite;
      a timed batch of decodes at each precision. SKIPPED while the artifact is missing
                                     | decoded one position off (both models); on default positions
                                       behind date blocks
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
ARM = "arm_t360t_vanilla"        # T360-T's arm (Step 5): Y4's dry build composes it with split val
ARM_T360 = "arm_t360_vanilla"    # its parent, which Y18 holds it to

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


def compose(overrides, name: str = ARM):
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision/eo_terramesh"), version_base=None):
        return hydra_compose(config_name=name, overrides=list(overrides))


# The real arm, on val's root: its train path reads g49's root, where Step 8 builds it (Y3, Y4 there).
# Until Step 5 this was T360's arm with seven overrides, which are now the arm's own keys (Y18).
DRY = ("multimodal.split=val",)


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


def y16_dataset():
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
    epoch_cb = ds.epoch_callback(verbose=False) if callback == "temporal" else Rounding()
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


# ================================================================ Step 5: the arm and the data record
T360T_DIFF = {
    # arm_t360t_vanilla against arm_t360_vanilla: every key that differs, and the value it must hold
    "dataset": "terramesh_temporal", "max_length": 2793, "model_config.vocab_size": TV.TOTAL_VOCAB_SIZE,
    "precision": "bf16", "total_batch_size": 60, "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
    "pretrain_init.conversion": "${oc.env:MOR_EO_RCP,/data/enric}/runs/pretrain/temporal/base/" + EXT.name,
    "pretrain_init.variant": "continue", "pretrain_init.freeze_body": False, "pretrain_init.from_stage1": None,
    "learning_rate": 0.001, "num_train_steps": 10648, "stop_steps": 1332, "num_warmup_steps": 133,
    "lr_scheduler_kwargs.num_decay_steps": 1065, "save_steps": 333, "eval_steps": 333,
}
# Equal to T360's and asserted: the split and eval_on_start are restated in the arm (the first eval_single
# is read against T360's last, Step 8); T360's 4 workers key the single eval set's modality orders.
T360T_SAME = {"multimodal.split": "train", "eval_on_start": True, "dataloader_num_workers": 4,
              "mixed_precision": True, "multimodal.token_set": "clean", "multimodal.eval_split": "v1"}
PILOT = {"arm_t360t_lr3e4_vanilla": {"learning_rate": 0.0003},
         "arm_t360t_noramp_vanilla": {"temporal.stages": [{"dates": 4, "locations": 6, "epochs": None}]}}


def _flat(c, prefix=""):
    """Dotted keys of a config, interpolations unresolved (as verify_phase3._flat); lists stay whole."""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(c):
        c = OmegaConf.to_container(c, resolve=False)
    out = {}
    for k, v in c.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


def _diff(a, b) -> dict:
    fa, fb = _flat(a), _flat(b)
    return {k: fb.get(k, "<absent>") for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def _intended(diff: dict, want: dict) -> list:
    """What makes a measured diff other than the intended one: stray or missing keys, wrong values."""
    out = [f"stray {k}" for k in sorted(set(diff) - set(want))] + [f"missing {k}" for k in sorted(set(want) - set(diff))]
    return out + [f"{k} = {diff[k]!r}, not {want[k]!r}" for k in sorted(set(diff) & set(want)) if diff[k] != want[k]]


def y18():
    from omegaconf import OmegaConf

    from eo.data import temporal_dataset as T
    from eo.train.preflight import _load_config

    arm_t = "arm_t360t_vanilla"
    block = json.loads(json.dumps(T.TemporalConfig().record()))          # the plan's numbers, YAML's spelling
    want = {**T360T_DIFF, **_flat(block, "temporal")}
    t360, armc = compose([], name=ARM_T360), compose([], name=arm_t)
    problems = _intended(_diff(t360, armc), want)
    flat_t = _flat(armc)
    problems += [f"{k} = {flat_t.get(k)!r}, not {v!r}" for k, v in T360T_SAME.items() if flat_t.get(k) != v]
    tcfg = T.validate_config(armc)                     # the builder's own refusals pass on the arm as shipped
    fits = tcfg == T.TemporalConfig() and int(armc.max_length) == T.sequence_length(tcfg.max_dates())
    # HF saves and evaluates at multiples of save_steps/eval_steps and at max_steps, never at stop_steps:
    # the pilot's stop must meet both, or its last checkpoint and eval fall an epoch short (Step 9).
    on_save = lambda c: int(c.stop_steps) % int(c.save_steps) == 0 and int(c.stop_steps) % int(c.eval_steps) == 0
    fits = fits and on_save(armc)
    stray = _intended(_diff(t360, compose(["seed=43"], name=arm_t)), want)
    off = compose(["stop_steps=1331"], name=arm_t)
    value = _intended(_diff(t360, off), want)
    hole = _intended(_diff(t360, compose(["~temporal.year_drop"], name=arm_t)), want)
    fired = {"a stray seed": bool(stray), "stop_steps 1331 (same keys)": bool(value),
             "stop_steps 1331 meets no save": not on_save(off),
             "the block without year_drop": bool(hole)}
    record("Y18 arm", not problems and fits,
           f"{arm_t} differs from {ARM_T360} in exactly {len(want)} keys with their values ({len(T360T_DIFF)} plus "
           f"the {len(want) - len(T360T_DIFF)} of the temporal block, every TemporalConfig field at the plan's "
           f"value); restated or inherited as T360's: {sorted(T360T_SAME)}; validate_config passes, max_length "
           f"{armc.max_length} = the {tcfg.max_dates()}-date sequence; the stop at {armc.stop_steps} meets a save "
           f"and an eval" + (f"; {problems[:4]}" if problems else ""),
           all(fired.values()), ", ".join(f"{k}: {'caught' if v else 'MISSED'}" for k, v in fired.items()))

    pilot_problems, pilot_fired = [], {}
    for arm, keys in PILOT.items():
        c = compose([], name=arm)
        pilot_problems += [f"{arm}: {p}" for p in _intended(_diff(armc, c), keys)]
        T.validate_config(c)
        pilot_fired[f"{arm} with a stray single_share"] = bool(_intended(
            _diff(armc, compose(["temporal.single_share=0.4"], name=arm)), keys))
    record("Y18 pilot arms", not pilot_problems,
           "; ".join(f"{a} differs from {arm_t} in exactly {sorted(k)}" for a, k in PILOT.items())
           + (f"; {pilot_problems[:4]}" if pilot_problems else ""), all(pilot_fired.values()),
           ", ".join(f"{k}: {'caught' if v else 'MISSED'}" for k, v in pilot_fired.items()))

    # The record pretrain.py writes (from Hydra's DictConfig) is the one preflight compares (from its plain
    # dict), it holds no schedule key, and write_data_record refuses a changed design on a resume.
    path = REPO / "conf/pretrain_vision/eo_terramesh" / f"{arm_t}.yaml"
    run_form = T.data_settings(compose(["multimodal.split=val"], name=arm_t))
    preflight_form = T.data_settings(OmegaConf.create(_load_config(path, {"multimodal.split": "val"})))
    keys = set(_flat(run_form))
    schedule = {"learning_rate", "num_train_steps", "stop_steps", "num_warmup_steps", "save_steps", "eval_steps"}
    tmp = Path(tempfile.mkdtemp(prefix="y18_"))
    T.write_data_record(tmp, run_form)
    rewritten = refused(T.write_data_record, tmp, run_form)
    sched = T.data_settings(compose(["multimodal.split=val", "stop_steps=2662", "num_train_steps=5324",
                                     "learning_rate=0.0005", "num_warmup_steps=200"], name=arm_t))
    schedule_free = not refused(T.write_data_record, tmp, sched)
    ok = (run_form == preflight_form and not rewritten and schedule_free and not (keys & schedule)
          and set(run_form) == {"dataset", "temporal", "total_batch_size", "max_length", "seed", "multimodal"}
          and set(run_form["multimodal"]) == set(T.DATA_MULTIMODAL) and run_form["temporal"] == block)
    changed = {"single_share 0.4": ["multimodal.split=val", "temporal.single_share=0.4"],
               "seed 43": ["multimodal.split=val", "seed=43"],
               "the eval split": ["multimodal.split=val", "multimodal.eval_split=null"]}
    fired = {k: refused(T.write_data_record, tmp, T.data_settings(compose(ov, name=arm_t))) for k, ov in changed.items()}
    fired["the no-ramp arm"] = refused(T.write_data_record, tmp, T.data_settings(
        compose(["multimodal.split=val"], name="arm_t360t_noramp_vanilla")))
    shutil.rmtree(tmp)
    record("Y18 data record", ok, f"{len(keys)} settings, the same from pretrain.py's DictConfig and from preflight's "
           f"dict; no schedule key; written once, then accepted unchanged and with only the schedule changed",
           all(fired.values()), "a resume refused with " + ", ".join(
               f"{k}: {'yes' if v else 'NO'}" for k, v in fired.items()))


# ================================================================ Step 6: the gate's own checks
# ---------------------------------------------------------------- Y2
TRAIN_META = Path("/data/enric/data/TerraMesh/train_meta/train_metadata.parquet")
META = {"val_clean": None, "train": TRAIN_META}     # None: val's metadata, eval_split.DEFAULT_METADATA_PATH


def _refused(fn, *args, **kwargs) -> bool:
    """`refused`, plus the loader's FileNotFoundError for a missing file."""
    try:
        fn(*args, **kwargs)
    except (ValueError, RuntimeError, OSError):
        return True
    return False


def _structure(name: str) -> dict:
    """Step 2's structural checks on one root, recomputed from its metadata. Run in a child process:
    train's row table holds 8.9M rows and takes 9.3 GB, which the gate's later checks need back."""
    from eo.data import temporal_index as TI
    from eo.data.eval_split import DEFAULT_METADATA_PATH, _sha256
    from eo.data.temporal_split import row_dates
    root, meta = ROOTS[name], META[name]
    index = load_temporal_index(root)
    locations, dates = index.locations, np.asarray(index.dates)
    table = row_dates(root, meta)
    out = {"report": TI.structure_report(locations, dates, table),
           "controls": TI.structure_controls(locations, dates, table),
           "split": TI.split_report(locations, root, meta),
           "metadata_equal": _sha256(Path(meta or DEFAULT_METADATA_PATH)) == index.build["inputs"]["metadata"],
           "recorded": index.build["structure"]}
    if out["split"] is not None:            # the split check against its control: one pair of rows swapped
        bad = TI.split_report(TI.swap_rows(locations, 1), root, meta)
        out["split_control"] = not bad["passed"]
    return out


def _tampered_root(tmp: Path, name: str) -> Path:
    """val_clean's index in a root of its own: every entry symlinked, temporal/ copied."""
    r = tmp / name
    r.mkdir()
    for e in VAL.iterdir():
        if e.name != "temporal":
            (r / e.name).symlink_to(e)
    shutil.copytree(VAL / "temporal", r / "temporal")
    return r


def _rehash(root: Path) -> None:
    """Make temporal_build.json's array hashes consistent with the arrays, as a careful forger would."""
    from eo.data import temporal_index as TI
    from eo.data.eval_split import _sha256
    d = root / TI.INDEX_DIR
    build = json.loads((d / TI.BUILD_FILE).read_text())
    build["arrays"] = {n: _sha256(d / n) for n in build["arrays"]}
    (d / TI.BUILD_FILE).write_text(json.dumps(build))


def y2():
    import concurrent.futures as cf
    import multiprocessing as mp

    import pandas as pd

    from eo.data import temporal_index as TI
    rule, wrong = TI.date_rule_report(), TI.date_rule_report(TI._wrong_rule)
    with cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        st = {name: pool.submit(_structure, name).result() for name in ("val_clean", "train")}
    problems, fired = [], {"the wrong day rule": not wrong["passed"]}
    for name, s in st.items():
        rep = s["report"]
        if not rep["passed"] or rep != s["recorded"] or not s["metadata_equal"]:
            problems.append(f"{name}: {dict((k, v) for k, v in rep.items() if v)}, recorded equal "
                            f"{rep == s['recorded']}, metadata equal {s['metadata_equal']}")
        for k, c in s["controls"].items():
            fired[f"{name} {k}"] = c["fired"]
        if name == "val_clean":
            if s["split"] is None or not s["split"]["passed"]:
                problems.append(f"val_clean's index is not temporal_rows_v1.json's locations: {s['split']}")
            fired["val_clean split, a pair swapped"] = s.get("split_control", False)
        elif s["split"] is not None:
            problems.append("train's root claims val's split")
    if not rule["passed"]:
        problems.append(f"date rule: {rule}")
    v, t = st["val_clean"]["report"], st["train"]["report"]
    record("Y2 structure", not problems,
           f"recomputed from the metadata, no offence on either root and equal to temporal_build.json: val_clean "
           f"{v['locations']:,} locations over {v['rows']:,} rows, train {t['locations']:,} over {t['rows']:,} (the "
           f"metadata each was built from, by hash); val_clean holds exactly temporal_rows_v1.json's locations; "
           f"{rule['cases']} hand-worked dates pass the day rule" + (f"; {problems[:3]}" if problems else ""),
           all(fired.values()), f"{sum(fired.values())} of {len(fired)} controls caught by their own offence"
           + ("" if all(fired.values()) else f": MISSED {[k for k, x in fired.items() if not x]}"))

    # The token checks on val_clean, rerun without rewriting temporal_check.json, against the stored one.
    # They read the static tokens and presence flags of all 2,176 locations, test side included, as
    # Step 2's check did: integrity only, no model reads them (Y3's audit counts what the dataset code reads).
    rep = TI.token_report(VAL)
    stored = json.loads((VAL / TI.INDEX_DIR / TI.CHECK_FILE).read_text())
    keys = ("complete", "presence", "static", "twins", "controls", "controls_fired", "passed", "arrays", "inputs")
    same = all(rep[k] == stored[k] for k in keys)
    stat = rep["static"]["per_modality"]
    ctrl = rep["controls"]
    record("Y2 tokens (val_clean)", rep["passed"] and rep["complete"] and same,
           f"every date holds all six blocks; Coords differ in {stat['Coords']['tokens_differing']} of "
           f"{stat['Coords']['tokens_compared']:,} tokens, DEM in {stat['DEM']['tokens_differing']} of "
           f"{stat['DEM']['tokens_compared']:,} (limits {TI.STATIC_LIMITS}); {rep['twins']} twins; the same report as "
           f"temporal_check.json: {same}", rep["controls_fired"],
           ", ".join(f"{k}: {c['locations_over_limit']} of {c['locations_moved']} moved locations caught"
                     for k, c in ctrl.items()))
    skip("Y2 tokens (train)", "train's tokens are on g49 only: its token checks ran there in Step 2 "
         "(temporal_check.json, loaded above) and rerun with Step 8")

    # The loader refuses tampered copies, and accepts an untampered one.
    tmp = Path(tempfile.mkdtemp(prefix="y2_"))
    d = TI.INDEX_DIR
    same_root = _tampered_root(tmp, "same")
    accepted = not _refused(load_temporal_index, same_root)

    def tamper(name, how):
        r = _tampered_root(tmp, name)
        how(r)
        return r

    def order(r):
        t = pd.read_parquet(VAL / TI.INDEX_NAME)
        cols = [c for c in t.columns if c != "row"]
        t.loc[[0, 1], cols] = t.loc[[1, 0], cols].to_numpy()          # two rows' scenes exchanged
        (r / TI.INDEX_NAME).unlink()
        t.to_parquet(r / TI.INDEX_NAME, index=False)

    def array(r):
        loc = np.load(r / d / TI.LOCATIONS_FILE)
        loc[5, 2] += 1
        np.save(r / d / TI.LOCATIONS_FILE, loc)

    def check_key(**kv):
        def go(r):
            p = r / d / TI.CHECK_FILE
            doc = json.loads(p.read_text())
            doc.update(kv)
            p.write_text(json.dumps(doc))
        return go

    def gone(f):
        return lambda r: (r / d / f).unlink()

    def swapped(r):
        np.save(r / d / TI.LOCATIONS_FILE, TI.swap_rows(np.load(r / d / TI.LOCATIONS_FILE), 1))
        _rehash(r)

    cases = {"another row order": order, "a changed array": array, "no check file": gone(TI.CHECK_FILE),
             "a failed check": check_key(passed=False), "a partial check": check_key(complete=False),
             "a check of other arrays": check_key(arrays={TI.LOCATIONS_FILE: "0" * 64, TI.DATES_FILE: "0" * 64}),
             "no build file": gone(TI.BUILD_FILE), "rows swapped, hashes made consistent": swapped}
    roots = {k: tamper(k.replace(" ", "_").replace(",", ""), how) for k, how in cases.items()}
    refusals = {k: _refused(load_temporal_index, r) for k, r in roots.items()}
    # The forged copy passes every hash, so only the checks themselves can see it: the token report
    # fails and the structure counts the mixed locations, as check would (exit 1).
    forged = roots["rows swapped, hashes made consistent"]
    forged_tokens = TI.token_report(forged)
    forged_index = load_temporal_index(forged, require_checked=False)
    shutil.rmtree(tmp, ignore_errors=True)
    record("Y2 loader", accepted, f"an untampered copy of val_clean's index loads; {sum(refusals.values())} of "
           f"{len(refusals)} tampered copies refused", all(refusals.values()) and not forged_tokens["passed"],
           ", ".join(f"{k}: {'refused' if x else 'ACCEPTED'}" for k, x in refusals.items())
           + f"; the forged copy's token report: passed {forged_tokens['passed']}, "
           f"{forged_tokens['static']['locations_over_limit']} locations over the limit "
           f"({len(forged_index)} locations, hashes consistent)")


# ---------------------------------------------------------------- the real T360-T (Y10, Y11, Y12, Y16)
_T360: dict = {}
LOGIT_TOL = 1e-4          # fp32 round-off where two kernels block differently (extend_checkpoint.IDENTITY_TOL)
CACHE_TOL = 1e-3          # cached against full logits at fp32 (eo/eval/ben.py CACHE_TOL["none"])
CONTROL_MIN = 1.0         # a broken position or table is off by whole logits (tm_compare_gen's control floor)
CACHE_CONTROL_MIN = 100 * CACHE_TOL   # Y16's control must fail its check by two orders of magnitude


def t360t() -> SimpleNamespace:
    """T360-T as its arm builds it (CPU, fp32, 87,664 ids), loaded from the extended base through
    pretrain_init's `continue` hook, once per process: the model Y10, Y11, Y12 and Y16 read."""
    if not _T360:
        import time

        from eo.generate import conditional as G
        from eo.train import clevr_init as CI
        t0 = time.time()
        torch.manual_seed(0)
        model, cfg = G.build_model(f"eo_terramesh/{ARM}", None, device="cpu", overrides=list(DRY))
        probe = TV.YEAR_IDS[5]                        # the 2020 row: fresh before the hook, the base's after
        fresh = model.model.embed_tokens.weight[probe].detach().clone()
        rep = CI.pretrain_init(model, cfg)
        _T360.update(model=model.eval(), cfg=cfg, rep=rep, fresh=fresh, probe=probe, seconds=time.time() - t0)
    return SimpleNamespace(**_T360)


def _no_base(check: str) -> bool:
    if (EXT / "extension.json").is_file():
        return False
    skip(check, f"{EXT} has no extension.json: Step 3's extension has not run")
    return True


def _hidden(model, mb):
    with torch.no_grad():
        return model.model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"],
                           position_ids=mb["position_ids"], use_cache=False).last_hidden_state


def _logit_gap(m1, mb1, m2, mb2, n=None, chunk: int = 512) -> float:
    """Largest |logit difference| between two models' fp32 logits, at the real positions both inputs hold
    (the first n of them when given), through the lm_head in chunks: vocabulary-sized logits dominate
    memory on CPU."""
    h1, h2 = _hidden(m1, mb1), _hidden(m2, mb2)
    worst = 0.0
    with torch.no_grad():
        for b in range(h1.shape[0]):
            real = int(min(mb1["attention_mask"][b].sum(), mb2["attention_mask"][b].sum()))
            real = real if n is None else min(real, n)
            for s in range(0, real, chunk):
                e = min(real, s + chunk)
                worst = max(worst, float((m1.lm_head(h1[b, s:e]) - m2.lm_head(h2[b, s:e])).abs().max()))
    return worst


# ---------------------------------------------------------------- Y10
def y10():
    if _no_base("Y10 lengths"):
        return
    import gc

    from eo.finetune.extend import assert_tied
    from eo.finetune.init import strict_load
    from eo.generate import conditional as G
    f = fx()
    T = f.T
    m = t360t()
    A = m.model

    def build(extra):
        model, cfg = G.build_model(f"eo_terramesh/{ARM}", None, device="cpu", overrides=list(DRY) + list(extra))
        strict_load(model, EXT)
        assert_tied(model)
        return model.eval()

    tcfg = T.TemporalConfig()
    single = T.SingleScenes(f.base, f.index, f.majortom, tcfg, 42)
    scenes = T.stack_trim([single.sequence(int(r), 0) for r in f.majortom[100:102]])
    seq = T.TemporalLocations(f.base, f.index, f.dev, tcfg, 42).sequence(int(f.dev[7]), 0, 2)
    real = int(seq["attention_mask"].sum())
    full = {k: v[:real][None] for k, v in seq.items()}
    cut = {k: v[:1048][None] for k, v in seq.items()}
    B = build(["max_length=1048"])
    conf = {"rope_theta": (A.config.rope_theta, B.config.rope_theta),
            "rope_scaling": (A.config.rope_scaling, B.config.rope_scaling),
            "max_position_embeddings": (A.config.max_position_embeddings, B.config.max_position_embeddings)}
    gaps = {"a scene micro-batch": _logit_gap(A, scenes, B, scenes),
            "two dates cut to 1,048": _logit_gap(A, cut, B, cut),
            f"two dates whole ({real}) against cut": _logit_gap(A, full, B, cut, n=1048)}
    del B
    gc.collect()
    C = build(["+model_config.rope_theta=20000.0"])      # a key the arm does not set
    theta_c = C.config.rope_theta
    ctrl = _logit_gap(A, cut, C, cut)
    del C
    gc.collect()
    ok = (all(g <= LOGIT_TOL for g in gaps.values()) and conf["rope_theta"] == (10000.0, 10000.0)
          and conf["rope_scaling"] == (None, None) and conf["max_position_embeddings"] == (2793, 1048))
    record("Y10 lengths", ok, f"T360-T built from {ARM} (max_length 2,793) and with max_length=1048, the extended base "
           f"in both (rope_theta {conf['rope_theta'][0]:g}, rope_scaling null, max_position_embeddings "
           f"{conf['max_position_embeddings']}): largest fp32 logit difference at shared positions "
           + ", ".join(f"{k} {v:.1e}" for k, v in gaps.items()) + f" (tolerance {LOGIT_TOL:g}; "
           f"{int((cut['input_ids'] == TV.BO_DATE).sum())} date blocks in the cut, lattice positions)",
           ctrl > CONTROL_MIN, f"rope_theta {theta_c:g}: {ctrl:.2f}")


# ---------------------------------------------------------------- Y11
def y11():
    if _no_base("Y11 hook (real model)"):
        return
    from safetensors import safe_open

    from eo.train import clevr_init as CI
    m = t360t()
    rep, model = m.rep, m.model
    doc = json.loads((EXT / "extension.json").read_text())
    with safe_open(str(EXT / doc["weights_file"]), "pt") as fh:
        row = fh.get_tensor(TABLE)[m.probe]
    n_params = int(sum(p.numel() for p in model.parameters()))
    checks = {
        "290 tensors loaded strictly, lm_head the only missing key": rep["load"] == {
            "tensors": 290, "missing_allowed": ["lm_head.weight"]},
        "tied": model.lm_head.weight is model.model.embed_tokens.weight,
        "a date row is the base's, not the fresh init's": bool(torch.equal(model.model.embed_tokens.weight[m.probe], row))
        and not torch.equal(m.fresh, row),
        "every parameter trains": rep["trainable_params"] == rep["total_params"] == n_params
        and all(p.requires_grad for p in model.parameters()),
        "the record names the base and continue": (rep["variant"] == "continue" and rep["freeze_body"] is False
                                                   and rep["from_stage1"] is None and rep["lora"] is None
                                                   and rep["conversion_weights_sha256"] == doc["weights_sha256"]),
        "its source is T360's checkpoint": rep["source"]["weights_sha256"] == doc["source_weights_sha256"],
    }
    tmp = Path(tempfile.mkdtemp(prefix="y11_"))
    CI.write_record(tmp, rep)
    again = not refused(CI.write_record, tmp, rep)            # a resume under its own init
    refusals = {"another base": refused(CI.check_record, tmp, {**rep, "conversion_weights_sha256": "f" * 64}),
                "another variant": refused(CI.check_record, tmp, {**rep, "variant": "smollm"})}
    shutil.rmtree(tmp, ignore_errors=True)
    record("Y11 hook (real model)", all(checks.values()) and again,
           f"T360-T at {model.config.vocab_size:,} ids, {n_params:,} parameters, built and loaded on CPU in "
           f"{m.seconds:.0f} s through pretrain_init: " + "; ".join(k for k, v in checks.items() if v)
           + ("" if all(checks.values()) else f"; FAILED {[k for k, v in checks.items() if not v]}")
           + f"; its record accepted on a resume under the same init: {again}", all(refusals.values()),
           ", ".join(f"{k}: {'refused' if v else 'ACCEPTED'}" for k, v in refusals.items()))

    r = _runs()
    if r is None:
        return
    lr_u, lr_r, lr_c = (r.lr[k] for k in ("uninterrupted", "resumed", "no_scheduler"))
    loss_u, loss_r, loss_c = (r.loss[k] for k in ("uninterrupted", "resumed", "no_scheduler"))
    steps = sorted(lr_u)
    changes = all(lr_u[s] != lr_u[s + 1] for s in steps[:-1])
    same_lr = lr_r == lr_u
    loss_gap = max(abs(loss_r[s] - loss_u[s]) / abs(loss_u[s]) for s in steps)
    ctrl_lr = [s for s in steps if lr_c.get(s) != lr_u[s]]
    ctrl_loss = max(abs(loss_c[s] - loss_u[s]) / abs(loss_u[s]) for s in steps if s > r.stop)
    # (step stop + 1's loss is computed before its update, from the same weights, so only later steps can differ)
    record("Y11 resume keeps the LR curve", changes and same_lr and loss_gap <= 1e-4 and r.rc["resumed"] == 0,
           f"through pretrain.py on a miniature T360-T (CPU, fp32): {RUN_STEPS} steps uninterrupted against "
           f"{r.stop} steps stopped on a save, then resumed (only stop_steps changed): learning_rate at every step "
           f"{'equal' if same_lr else 'DIFFERENT'} ({', '.join(f'{lr_u[s]:.2e}' for s in steps)}: it changes at every "
           f"step, warmup and decay inside the run), loss within {loss_gap:.1e} relative",
           bool(ctrl_lr) and ctrl_loss > 1e-4,
           f"the resume with scheduler.pt removed (HF then loads neither scheduler nor optimizer, silently; "
           f"preflight refuses it): learning_rate differs at steps {ctrl_lr}, loss by {ctrl_loss:.1e}")
    record("Y11 refusals (pretrain.py)", r.rc["uninterrupted"] == 0 and r.records_name_base,
           "the runs from the miniature base record its weights' hash in pretrain_init.json",
           r.rc["flipped_hash"] == 1 and r.nothing["flipped_hash"] and r.rc["other_base_resume"] == 1
           and r.refused_by["other_base_resume"],
           f"a base whose weights no longer hash to its extension.json: exit {r.rc['flipped_hash']}, nothing written "
           f"({r.nothing['flipped_hash']}); a resume under another base's config (a second valid miniature base): "
           f"exit {r.rc['other_base_resume']}, refused by the run record ({r.refused_by['other_base_resume']}), "
           f"checkpoints unchanged ({r.nothing['other_base_resume']})")


# ---------------------------------------------------------------- Y11 and Y15: pretrain.py on a miniature
# T360's arm with these overrides is T360-T's shape in miniature: the same class, config and rope, two layers.
TINY = ("model_config.hidden_size=96", "model_config.intermediate_size=256", "model_config.num_hidden_layers=2",
        "model_config.num_attention_heads=6", "model_config.num_key_value_heads=2")
# One location or two scenes a micro-batch, two a step, two steps an epoch: CPU-sized, the ramp kept.
TINY_DATA = ("temporal.stages=[{dates:2,locations:1,epochs:1},{dates:3,locations:1,epochs:null}]",
             "temporal.scenes_per_microbatch=2", "temporal.steps_per_epoch=2", "total_batch_size=2")
RUN_STEPS, RUN_STOP = 6, 3      # warmup 3 and decay 3: the LR changes at every step; the first leg stops on a save
_RUNS: dict = {}


def _save_dir() -> Path:
    """MOR_SAVE_DIR as pretrain.py reads it (.env, unless the environment sets it). Not paths.SAVE_DIR,
    which is fixed when `paths` is first imported (verify_phase3's W20 found it)."""
    val = os.environ.get("MOR_SAVE_DIR")
    env_file = REPO / ".env"
    if val is None and env_file.is_file():
        for raw in env_file.read_text().splitlines():
            key, _, value = raw.strip().partition("=")
            if key.strip() == "MOR_SAVE_DIR":
                val = value.strip().strip("'\"")
    if not val or not val.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {val!r} is not under /data: refusing to write checkpoints")
    return Path(val)


def _identity_rows() -> list:
    """The real base's identity batch (extension.json), less its rows on val's temporal test side, topped
    up with dev-side held-out rows: the miniature's identity check reads no test row (Y3's audit)."""
    f = fx()
    real = json.loads((EXT / "extension.json").read_text())["identity"]["batch_rows"]
    keep = [int(r) for r in real if int(r) not in f.test_rows]
    more = [int(r) for r in np.sort(np.asarray(f.held)) if int(r) in f.dev_rows and int(r) not in keep]
    return keep + more[: len(real) - len(keep)]


def _snapshot(d: Path) -> dict:
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(d.rglob("*")) if p.is_file()}


def _history(d: Path) -> tuple:
    """{step: learning_rate}, {step: loss} from a run directory's trainer_state.json."""
    log = json.loads((d / "trainer_state.json").read_text())["log_history"]
    rows = [e for e in log if "learning_rate" in e and "loss" in e]
    return {e["step"]: e["learning_rate"] for e in rows}, {e["step"]: e["loss"] for e in rows}


def _runs():
    """The miniature's runs through pretrain.py, launched as train_eo.sh launches it, on CPU at fp32:
    uninterrupted; stopped on a save and resumed with only stop_steps changed; and four that must go wrong:
    the resume with scheduler.pt removed, with a changed temporal setting, under another base, and a fresh
    launch from a base whose weights no longer hash to its extension.json. Shared by Y11 and Y15."""
    if _RUNS:
        return SimpleNamespace(**_RUNS) if _RUNS.get("ok") else None
    import importlib.util
    import subprocess
    import time

    from safetensors.torch import load_file, save_file

    from eo.data import temporal_dataset as T
    from eo.generate import conditional as G
    from eo.train import clevr_init as CI
    _RUNS["ok"] = False
    if not (EXT / "extension.json").is_file():
        skip("Y11/Y15 runs", f"{EXT} has no extension.json")
        return None
    pretrain = _save_dir() / "pretrain"
    rel = Path("gate_y15") / time.strftime("%Y%m%d_%H%M%S")
    root = pretrain / rel
    base = root / "base"
    base.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("extend_checkpoint", REPO / "eo/scripts/extend_checkpoint.py")
    XC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(XC)
    torch.manual_seed(0)
    tiny, _ = G.build_model(f"eo_terramesh/{ARM_T360}", None, device="cpu", overrides=list(TINY))
    tiny.save_pretrained(base / "tiny", safe_serialization=True)
    del tiny
    rows = _identity_rows()
    docs = {name: XC.extend(f"eo_terramesh/{ARM_T360}", base / "tiny", base / name, seed=seed, layout="temporal",
                            device="cpu", overrides=list(TINY), batch_rows=rows)
            for name, seed in (("ext", 0), ("ext_other", 1))}
    shutil.copytree(base / "ext", base / "ext_flipped")
    state = load_file(str(base / "ext_flipped" / "model.safetensors"))
    state[TABLE][TV.BO_DATE, 0] += 1.0                        # extension.json still records the old hash
    save_file(state, str(base / "ext_flipped" / "model.safetensors"), metadata={"format": "pt"})

    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    common = list(TINY) + list(TINY_DATA) + [
        "multimodal.split=val", "precision=fp32", "wandb=false", "eval_on_start=false", "eval_steps=1000",
        "dataloader_num_workers=0", "logging_steps=1", f"num_train_steps={RUN_STEPS}", "num_warmup_steps=3",
        "lr_scheduler_kwargs.num_decay_steps=3", f"save_steps={RUN_STOP}"]
    rc, logs = {}, {}

    def run(name, out, extra, conversion="ext"):
        cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name",
               f"eo_terramesh/{ARM}", f"name={name}", f"output_dir={rel / out}", f"wandb_run_name={name}",
               f"hydra.run.dir={root / 'hydra' / name}", f"pretrain_init.conversion={base / conversion}",
               *common, *extra]
        r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1800)
        rc[name], logs[name] = r.returncode, r.stdout + r.stderr
        (root / f"{name}.log").write_text(logs[name])
        print(f"      [{name}] exit {r.returncode}", flush=True)

    run("uninterrupted", "u", [f"stop_steps={RUN_STEPS}"])
    run("leg1", "l", [f"stop_steps={RUN_STOP}"])
    for out in ("nosched", "changed", "otherbase"):                     # leg 1 as it stands, for the controls
        shutil.copytree(root / "l", root / out)
    (root / "nosched" / f"checkpoint-{RUN_STOP}" / "scheduler.pt").unlink()
    run("resumed", "l", [f"stop_steps={RUN_STEPS}", "resume_from_checkpoint=true"])
    run("no_scheduler", "nosched", [f"stop_steps={RUN_STEPS}", "resume_from_checkpoint=true"])
    before = {out: _snapshot(root / out) for out in ("changed", "otherbase")}
    run("changed_data", "changed", [f"stop_steps={RUN_STEPS}", "resume_from_checkpoint=true", "temporal.year_drop=0.2"])
    run("other_base_resume", "otherbase", [f"stop_steps={RUN_STEPS}", "resume_from_checkpoint=true"],
        conversion="ext_other")
    run("flipped_hash", "flipped", [f"stop_steps={RUN_STEPS}"], conversion="ext_flipped")

    hist = {k: _history(root / d) for k, d in (("uninterrupted", "u"), ("resumed", "l"), ("no_scheduler", "nosched"))
            if (root / d / "trainer_state.json").is_file()}
    flipped = root / "flipped"
    nothing = {"flipped_hash": not flipped.exists() or not any(flipped.glob("checkpoint-*"))
               and not (flipped / CI.RECORD).exists() and not (flipped / T.DATA_RECORD).exists(),
               "changed_data": _snapshot(root / "changed") == before["changed"],
               "other_base_resume": _snapshot(root / "otherbase") == before["otherbase"]}
    refused_by = {"changed_data": "records other data" in logs["changed_data"],
                  "other_base_resume": "records another init" in logs["other_base_resume"],
                  "flipped_hash": "does not hash to what extension.json recorded" in logs["flipped_hash"]}
    rec = {d: json.loads((root / d / CI.RECORD).read_text()) for d in ("u", "l") if (root / d / CI.RECORD).is_file()}
    data = json.loads((root / "l" / T.DATA_RECORD).read_text()) if (root / "l" / T.DATA_RECORD).is_file() else None
    want_data = T.data_settings(compose(common))
    epoch_lines = {k: [line for line in logs[k].splitlines() if line.startswith("TemporalEpochCallback:")]
                   for k in ("uninterrupted", "leg1", "resumed")}
    ok = all(k in hist for k in ("uninterrupted", "resumed", "no_scheduler")) and set(rec) == {"u", "l"}
    # Written at launch: before the first checkpoint of the uninterrupted run, and kept by its resume (leg 1's).
    first = root / "u" / f"checkpoint-{RUN_STOP}" / "trainer_state.json"
    at_launch = first.is_file() and all((root / "u" / f).stat().st_mtime < first.stat().st_mtime
                                        for f in (CI.RECORD, T.DATA_RECORD))
    _RUNS.update(ok=ok, root=root, rc=rc, logs=logs, stop=RUN_STOP, docs=docs, rows=rows, at_launch=at_launch,
                 lr={k: v[0] for k, v in hist.items()}, loss={k: v[1] for k, v in hist.items()},
                 nothing=nothing, refused_by=refused_by, rec=rec, data=data, want_data=want_data,
                 epoch_lines=epoch_lines,
                 records_name_base=all(r["conversion_weights_sha256"] == docs["ext"]["weights_sha256"]
                                       and r["variant"] == "continue" for r in rec.values()))
    if not ok:
        record("Y11/Y15 runs", False, f"a run that had to train did not: exits {rc}; logs in {root}")
        return None
    return SimpleNamespace(**_RUNS)


def _epoch_line_problems(lines: list, want: list) -> list:
    """The callback's own report (eo/data/temporal_dataset.TemporalEpochCallback), one line per epoch begun:
    (global_step, draw epoch), each with HF's epoch length checked and the micro-batch collator in use."""
    got = []
    for line in lines:
        words = line.replace(",", "").split()
        got.append((int(words[words.index("global_step") + 1]), int(words[words.index("epoch") + 1])))
    out = [] if got == want else [f"epochs {got}, not {want}"]
    tail = "HF's epoch 4 micro-batches / 2 accumulation = 2 steps; collator unwrap_microbatch"
    return out + [f"line without the checks: {line}" for line in lines if not line.endswith(tail)]


def y15():
    from transformers import TrainingArguments

    from eo.data import temporal_dataset as T
    r = _runs()
    if r is None:
        return
    rc = r.rc
    rec_l = r.rec["l"]
    probs = []
    if r.data != r.want_data:
        probs.append(f"temporal_data.json is not the config's data settings: "
                     f"{T.data_record_changes(r.want_data, r.data or {})[:3]}")
    if not (rec_l["load"]["missing_allowed"] == ["lm_head.weight"] and rec_l["trainable_params"] == rec_l["total_params"]):
        probs.append(f"pretrain_init.json: load {rec_l['load']}, trainable {rec_l['trainable_params']}")
    want = {"uninterrupted": [(0, 0), (2, 1), (4, 2)], "leg1": [(0, 0), (2, 1)], "resumed": [(3, 1), (4, 2)]}
    for k, w in want.items():
        probs += [f"{k}: {p}" for p in _epoch_line_problems(r.epoch_lines[k], w)]
    # The resumed leg's first epoch line is at global_step 3: it continued from its checkpoint (HF says so
    # only at INFO), and Y11 holds its learning rate and loss to the uninterrupted run's at every step.
    if not r.at_launch:
        probs.append("the records are not older than the run's first checkpoint")
    ok = (rc["leg1"] == 0 and rc["resumed"] == 0 and rc["uninterrupted"] == 0 and not probs
          and (r.root / "l" / f"checkpoint-{RUN_STEPS}").is_dir())
    # The callback's two refusals, which no config of a working run can reach (its controls):
    ds = make(small_cfg(), per_step=2)
    cb = ds.epoch_callback(verbose=False)
    args = TrainingArguments(output_dir=tempfile.mkdtemp(prefix="y15_"), use_cpu=True, report_to=[],
                             gradient_accumulation_steps=2, per_device_train_batch_size=1)
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    loaders = {"an epoch one micro-batch short": torch.utils.data.DataLoader(list(range(len(ds) - 1)), batch_size=1,
                                                                           collate_fn=T.unwrap_microbatch),
               "HF's default collator": torch.utils.data.DataLoader(list(range(len(ds))), batch_size=1)}
    fired = {k: refused(cb.on_epoch_begin, args, state, None, train_dataloader=ld) for k, ld in loaders.items()}
    accepted = not refused(cb.on_epoch_begin, args, state, None, train_dataloader=torch.utils.data.DataLoader(
        list(range(len(ds))), batch_size=1, collate_fn=T.unwrap_microbatch))
    shutil.rmtree(args.output_dir, ignore_errors=True)
    record("Y15 two real steps and a resume", ok and accepted,
           f"pretrain.py on the miniature (CPU, fp32), launched as train_eo.sh launches it: the run writes "
           f"pretrain_init.json (the base's hash, a strict load, every parameter training) and temporal_data.json "
           f"(the config's data settings) at launch; TemporalEpochCallback reports the draw epoch from global_step "
           f"at every epoch begun, with HF's epoch length checked and unwrap_microbatch collating "
           f"({sum(len(v) for v in r.epoch_lines.values())} lines over three runs); stopped on a save at step "
           f"{RUN_STOP} and resumed with only stop_steps changed: accepted, continued from step {RUN_STOP} to "
           f"{RUN_STEPS}" + (f"; {probs[:3]}" if probs else ""),
           rc["flipped_hash"] == 1 and r.nothing["flipped_hash"] and r.refused_by["flipped_hash"]
           and rc["changed_data"] == 1 and r.refused_by["changed_data"] and r.nothing["changed_data"]
           and all(fired.values()),
           f"a flipped init hash: exit {rc['flipped_hash']}, refused by verify_init ({r.refused_by['flipped_hash']}), "
           f"nothing trained or recorded ({r.nothing['flipped_hash']}); a resume with temporal.year_drop 0.2: exit "
           f"{rc['changed_data']}, refused by the data record ({r.refused_by['changed_data']}), the run directory "
           f"untouched ({r.nothing['changed_data']}); the callback itself: "
           + ", ".join(f"{k}: {'refused' if v else 'ACCEPTED'}" for k, v in fired.items()))


# ---------------------------------------------------------------- Y12
Y12_OUT = Path("/data/enric/reports/temporal/forgetting/t360t_base/extended_base")
Y12_REF = Path("/data/enric/reports/ben/forgetting/t360_base/extended_base")
Y12_TOL = 0.01
Y12_CMD = ("CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(python -m eo.train.gpu titanx) python "
           "eo/scripts/eval_forgetting.py --model temporal --arm-config eo_terramesh/arm_t360_vanilla --precision fp32 "
           "--tag t360t_base")


#: Per-token CE agrees across devices within one fp16 step (tokens_ce.npz stores fp16) at the larger value, and
#: never closer than this: the CPU and the Titan X differ by at most 3.0e-05 nats in fp32 (2026-11-09), and below
#: 6.1e-05 fp16 is subnormal, a fixed 6e-08 apart, so two steps there are 1.2e-07 nats.
TOKEN_FLOOR = 1e-4


def _fp16_step(x: np.ndarray) -> np.ndarray:
    """One rounding step of fp16 at |x|: tokens_ce.npz stores per-token CE as fp16."""
    return np.exp2(np.floor(np.log2(np.maximum(np.abs(x), 2.0 ** -14)))) * 2.0 ** -10


def _token_tol(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(_fp16_step(np.maximum(np.abs(x), np.abs(y))), TOKEN_FLOOR)


def y12():
    art = Y12_OUT / "forgetting.json"
    if not art.is_file():
        skip("Y12 forgetting reading", f"{art} missing: the GPU run writes it, on the GTX Titan X, about 30 min "
             f"in tmux: {Y12_CMD}")
        return
    if _no_base("Y12 forgetting reading"):
        return
    from unittest import mock

    from eo.eval import teacher_forced as TF
    from lm_dataset.load_dataset import load_eval_dataset_from_config
    doc, ref = json.loads(art.read_text()), json.loads((Y12_REF / "forgetting.json").read_text())
    ext = json.loads((EXT / "extension.json").read_text())
    a, r = np.load(Y12_OUT / "tokens_ce.npz"), np.load(Y12_REF / "tokens_ce.npz")
    same_rows = (np.array_equal(a["rows"], r["rows"]) and a["ce"].shape == r["ce"].shape
                 and np.array_equal(a["label_mod"], r["label_mod"]))
    lm = a["label_mod"] > 0 if same_rows else None
    d_mean = doc["ce_all_body_tokens"] - ref["ce_all_body_tokens"]
    ok = (doc["model"] == "temporal" and doc["checkpoint"] is None and doc["vocab"] == TV.TOTAL_VOCAB_SIZE
          and doc["loaded"]["conversion_weights_sha256"] == ext["weights_sha256"] and doc["loaded"]["variant"] == "continue"
          and doc["precision"] == ref["precision"] == "fp32" and doc["arm_config"] == ref["arm_config"]
          and doc["token_set"] == ref["token_set"] == "clean" and doc["n_rows"] == ref["n_rows"] == 4416
          and same_rows and abs(d_mean) <= Y12_TOL and "TITAN V" not in doc["gpu"])
    detail = (f"{doc['ce_all_body_tokens']:.4f} on the {doc['n_rows']:,} held-out rows ({doc['gpu']}, fp32, "
              f"{doc['seconds']:.0f} s), against T360's {ref['ce_all_body_tokens']:.4f} through its BEN base: "
              f"{d_mean:+.1e} (tolerance {Y12_TOL})")
    if same_rows:
        ca, cr = a["ce"].astype(np.float64)[lm], r["ce"].astype(np.float64)[lm]
        diff = ca - cr
        tol = _token_tol(ca, cr)
        ok = ok and not (np.abs(diff) > tol).any()
        per_mod = {n: doc["per_modality"][n]["ce"] - ref["per_modality"][n]["ce"] for n in ref["per_modality"]}
        detail += (f"; per token, paired on the same rows, layout and modality orders ({lm.sum():,} tokens): mean "
                   f"{diff.mean():+.1e}, {np.mean(diff == 0):.2%} equal in fp16, the rest up ({np.mean(diff > 0):.2%}) "
                   f"or down ({np.mean(diff < 0):.2%}) by fp16 steps, {int((np.abs(diff) > tol).sum())} beyond one step "
                   f"(floor {TOKEN_FLOOR:g} nats); "
                   f"per modality " + ", ".join(f"{n} {v:+.4f}" for n, v in per_mod.items()))
    else:
        detail += "; the rows, their order or their modalities differ from the reference"

    # Live, on CPU: the first two batches of the reference's loader (batch size 4, 4 workers, so batch k came
    # from worker k % 4, which keys the modality orders) through this gate's T360-T must give the artifact's
    # per-token CE; the same batches with the table shifted by one row must read far off. Major TOM rows only.
    m = t360t()
    A = m.model
    ds = load_eval_dataset_from_config(compose([], name=ARM_T360))
    batches = []
    for k in (0, 1):
        with mock.patch.object(torch.utils.data, "get_worker_info", lambda k=k: SimpleNamespace(id=k % 4)):
            batches.append(torch.utils.data.default_collate([ds[4 * k + i] for i in range(4)]))
    live = TF.run_pass(A, batches, device="cpu", keep_tokens=True, dtype=None)
    lt = live["tokens"]["ce"].astype(np.float64)
    keep = a["label_mod"][:8] > 0
    gap = np.abs(lt - a["ce"][:8].astype(np.float64))[keep]
    live_ok = (np.array_equal(np.asarray(ds.rows[:8]), a["rows"][:8])
               and np.array_equal(live["tokens"]["label_mod"], a["label_mod"][:8])
               and bool((gap <= _token_tol(lt, a["ce"][:8].astype(np.float64))[keep]).all()))
    # What the 108 rows over T360's 87,556 add to the softmax, on the first row: they sit at the mean row, so
    # their logit is the mean logit, far below the log-sum-exp, and their mass is negligible.
    one = {k: v[:1] for k, v in batches[0].items()}
    n_one = int(one["attention_mask"][0].sum())
    h = _hidden(A, one)[0]
    with torch.no_grad():
        mass = torch.cat([torch.softmax(A.lm_head(h[i: min(n_one, i + 512)]).float(), -1)[:, EV.TOTAL_VOCAB_SIZE:].sum(-1)
                          for i in range(0, n_one, 512)])
    W = A.model.embed_tokens.weight
    orig = W.detach().clone()
    with torch.no_grad():
        W.copy_(torch.cat([orig.mean(0, keepdim=True), orig[:-1]], 0))    # every id reads its neighbour's row
    try:
        shifted = TF.run_pass(A, batches, device="cpu", keep_tokens=False, dtype=None)
    finally:
        with torch.no_grad():
            W.copy_(orig)
    del orig
    rise = shifted["ce_all_body_tokens"] - live["ce_all_body_tokens"]
    record("Y12 forgetting reading", ok and live_ok, detail + f"; live on CPU, the first 8 rows: per-token CE equal "
           f"to the artifact's within one fp16 step or {TOKEN_FLOOR:g} nats ({live_ok}, largest gap {gap.max():.4f}); the "
           f"{TV.TOTAL_VOCAB_SIZE - EV.TOTAL_VOCAB_SIZE} rows over T360's hold {float(mass.mean()):.1e} of the "
           f"softmax per position on row {int(ds.rows[0])} (largest {float(mass.max()):.1e})",
           rise > CONTROL_MIN, f"the table shifted by one row: CE {shifted['ce_all_body_tokens']:.2f} on those rows, "
           f"against {live['ce_all_body_tokens']:.2f} ({rise:+.2f})")


# ---------------------------------------------------------------- Y13
def _prompt_problems(ts, rows4, reader, task: str, t: int, target: str) -> list:
    """Y13 on one task sequence: the prompt is the prefix, the lattice holds, every block is its declared
    row's, and nothing of the target's (modality, date) is in the prompt, by declaration and by tokens."""
    out = []
    s, e = ts.target
    full_ids, pos = ts.full["input_ids"].numpy(), ts.full["position_ids"].numpy()
    if any(not torch.equal(ts.prompt[k], ts.full[k][: s + 1]) for k in KEYS) or len(ts.prompt["input_ids"]) != s + 1:
        out.append("the prompt is not the sequence's prefix through the target's BO")
    out += lattice_problems(full_ids, pos)
    blocks = parse(full_ids)
    declared = list(ts.draw.blocks)
    if [b[0] for b in blocks] != [d.modality for d in declared] or blocks[-1][1] != s:
        return out + ["the token stream is not the declared layout, target last"]
    for d, (mod, a, b) in zip(declared, blocks):
        row = int(rows4[d.t])
        if mod == "Date":
            y, day = TV.decode_date_block(full_ids[a:b])
            if (y, day) != (reader.date(row)[0] if d.with_year else None, reader.date(row)[1]):
                out.append(f"the date block at {a} is not date {d.t}'s")
        elif not np.array_equal(full_ids[a:b], reader.block(mod, row)):
            out.append(f"the {mod} block at {a} is not date {d.t}'s row")
    tgt_row = int(rows4[t])
    for d, (mod, a, b) in zip(declared[:-1], blocks[:-1]):
        if (d.modality, d.t) == (target, t):
            out.append(f"the target ({target}, date {t}) is declared in the prompt at {a}")
        if mod == target and mod != "Date" and np.array_equal(full_ids[a:b], reader.block(mod, tgt_row)):
            out.append(f"the target's own block is in the prompt at {a}")
        if target == "Date" and mod == "Date" and TV.decode_date_block(full_ids[a:b])[1] == reader.date(tgt_row)[1]:
            out.append(f"a date block of the target date is in the prompt at {a}")
    # The layout's defaults (eo/eval/temporal.py): statics first, a date block first in every group, LULC in
    # the first context group, the target's group last with its date block first, one target block last.
    if [d.modality for d in declared[:2]] != ["Coords", "DEM"]:
        out.append("the statics are not first")
    groups = []
    for d in declared[2:]:
        if groups and groups[-1][0] == d.t:
            groups[-1][1].append(d.modality)
        else:
            groups.append((d.t, [d.modality]))
    if task == "date":
        if len(groups) != 1 or groups[0][1][-1] != "Date" or groups[0][1].count("Date") != 1:
            out.append(f"date estimation's one group is {groups}")
    else:
        if any(g[1][0] != "Date" for g in groups) or groups[-1][0] != t or [g[0] for g in groups[:-1]] != sorted(
                g[0] for g in groups[:-1]) or "LULC" not in groups[0][1] or sum(g[1].count("LULC") for g in groups) != 1:
            out.append(f"groups {[(g[0], g[1]) for g in groups]} break the defaults")
        want = ["Date"] + (["S1GRD"] if task == "gap_fill" else []) + [target]
        if groups[-1][1] != want:
            out.append(f"the target group is {groups[-1][1]}, not {want}")
    return out


def _alone_problems(temporal, loc, blocks, ts) -> list:
    """The context assembled alone, as a generation path builds it without the target, then the target's
    BO at the lattice's next position: it must be the prompt cut from the whole sequence."""
    from eo.data import temporal_dataset as T
    from eo.eval import temporal as ET
    ctx = list(blocks[:-1])
    alone = temporal.sequence(loc, 0, max(1, len(ET.forced_draw(loc, ctx).dates)), draw=ET.forced_draw(loc, ctx))
    n = int(alone["attention_mask"].sum())
    widths = [ET.block_width(b) for b in ctx] + [1]
    bo_pos = int(T.lattice_positions(widths, [b.modality == T.DATE for b in ctx] + [False], sum(widths))[-1])
    same = all(torch.equal(alone[k][:n], ts.prompt[k][:-1]) for k in ("input_ids", "position_ids", "modality_ids"))
    if not same or n != len(ts.prompt["input_ids"]) - 1 or int(ts.prompt["position_ids"][-1]) != bo_pos:
        return ["the context assembled alone is not the prompt's prefix, or the target's BO is off the lattice"]
    return []


def y13():
    f = fx()
    T = f.T
    from eo.eval import temporal as ET
    temporal = T.TemporalLocations(f.base, f.index, f.dev, T.TemporalConfig(), 42)
    locs = [int(x) for x in f.dev[3::67][:8]]
    inst = ET.task_instances()
    problems, n, lengths = [], 0, {}
    for loc in locs:
        rows4 = [int(r) for r in f.index.locations[loc]]
        for task, t, target in inst:
            blocks = ET.task_blocks(task, t, target)
            ts = ET.task_sequence(temporal, loc, blocks)
            problems += [f"{task} t{t} {target} loc {loc}: {p}"
                         for p in _prompt_problems(ts, rows4, f.reader, task, t, target)]
            problems += [f"{task} t{t} {target} loc {loc}: {p}" for p in _alone_problems(temporal, loc, blocks, ts)]
            lengths.setdefault(task, set()).add(len(ts.prompt["input_ids"]))
            n += 1
    # No date tokens (T360's row): the same blocks without date blocks, positions sequential as T360 trained.
    plain = ET.task_sequence(temporal, locs[0], ET.task_blocks("left_out", 1, "S2L2A", with_dates=False))
    seq_pos = np.array_equal(plain.full["position_ids"].numpy(), np.arange(len(plain.full["position_ids"])))
    test_loc = int(temporal.test_ids[0])
    read_before = len(READ_ROWS)
    test_refused = refused(ET.task_sequence, temporal, test_loc, ET.task_blocks("left_out", 1, "S2L2A"))
    record("Y13 task prompts", not problems and seq_pos and test_refused and len(READ_ROWS) == read_before,
           f"{n} task sequences ({len(inst)} instances of {', '.join(ET.TASKS)} at {len(locs)} dev locations) through "
           f"TemporalLocations.sequence(draw=...): every prompt is the prefix through the target's BO, on the "
           f"lattice, every block its declared row's, nothing of the target's (modality, date) in it; prompt lengths "
           f"{ {k: sorted(v) for k, v in lengths.items()} }; without date blocks the positions are T360's; a test "
           f"location is refused before any read" + (f"; {problems[:3]}" if problems else ""))

    def left_in(task, t, target):
        """The target block also in the first context group: the layout a careless builder would produce.
        For date estimation the date block also first in its group, LULC dropped to stay within 999 tokens."""
        b = ET.task_blocks(task, t, target)
        if task == "date":
            return b[:2] + [b[-1]] + [x for x in b[2:] if x.modality != "LULC"]
        return b[:3] + [b[-1]] + b[3:]

    flagged, total = 0, 0
    for loc in locs[:4]:
        rows4 = [int(r) for r in f.index.locations[loc]]
        for task, t, target in inst[::3]:
            ts = ET.task_sequence(temporal, loc, left_in(task, t, target))
            flagged += any("in the prompt" in p for p in _prompt_problems(ts, rows4, f.reader, task, t, target))
            total += 1
    # Prompts assembled with sequential positions: not the sequence's prefix once a date block has passed.
    ts = ET.task_sequence(temporal, locs[0], ET.task_blocks("left_out", 2, "NDVI"))
    seq = np.arange(len(ts.prompt["position_ids"]))           # what a default assembly would give
    sequential_caught = not np.array_equal(seq, ts.prompt["position_ids"].numpy())
    shifted = ET.task_blocks("left_out", 2, "NDVI")
    k = next(i for i, b in enumerate(shifted) if b.modality == "Date")
    off = list(shifted)
    off[k] = replace(off[k], with_year=False)                    # a context one token shorter than the prompt's
    alone_caught = bool(_alone_problems(temporal, locs[0], off, ts))
    record("Y13 controls", True, "the prompt checks against broken builders",
           flagged == total and sequential_caught and alone_caught,
           f"the target left in context: {flagged} of {total} prompts flagged; a prompt on sequential positions "
           f"differs from the prefix: {sequential_caught}; a context assembled with one year dropped is not the "
           f"prompt's prefix: {alone_caught}")


# ---------------------------------------------------------------- Y16, the generation half
def y16_generation():
    if _no_base("Y16 cached generation"):
        return
    from eo.eval import temporal as ET
    from eo.generate import cached as GC
    f = fx()
    T = f.T
    A = t360t().model
    temporal = T.TemporalLocations(f.base, f.index, f.dev, T.TemporalConfig(), 42)
    ts = ET.task_sequence(temporal, int(f.dev[11]), ET.task_blocks("left_out", 1, "S2L2A"))
    ids, pos = ts.prompt["input_ids"][None], ts.prompt["position_ids"][None]
    s, _ = ts.target_tokens
    cont = ts.full["input_ids"][s: s + 3][None]
    behind = int(ids.shape[1] - 1 - int(pos[0, -1]))
    good = GC.cache_check(A, ids, cont, position_ids=pos)
    bad = GC.cache_check(A, ids, cont, position_ids=pos, decode_positions=False)
    # The sampler's default path is the one every earlier result used: explicit 0..L-1 positions give the
    # same tokens and slot mass bit for bit on a T360 layout (a single scene, no date block).
    scene = f.T.SingleScenes(f.base, f.index, f.majortom, f.T.TemporalConfig(), 42).sequence(
        int(f.majortom[5]), 0, draw=f.T.SceneDraw(int(f.majortom[5]), ("S1RTC", "DEM", "S2L2A"), "absent", True))
    p = scene["input_ids"][: 2 * 198 + 1][None]
    kw = dict(temperature=1.0, top_p=0.8, seed=7, n_tokens=4)
    x0, m0 = GC.sample(A, p, "S2L2A", **kw)
    x1, m1 = GC.sample(A, p, "S2L2A", position_ids=torch.arange(p.shape[1])[None], **kw)
    same_default = bool(torch.equal(x0, x1) and torch.equal(m0, m1))
    record("Y16 cached generation", good["max_abs_logit_diff"] <= CACHE_TOL and behind == 16 and same_default,
           f"T360-T, fp32, CPU: a left-out-season prompt of {good['prompt_length']:,} tokens behind four date blocks "
           f"(the lattice {behind} positions behind the token count), {good['steps']} cached steps of the real target: "
           f"every step's logits equal a full teacher-forced pass at the same positions to "
           f"{good['max_abs_logit_diff']:.1e} (tolerance {CACHE_TOL:g}); the sampler with explicit 0..L-1 positions "
           f"gives the default path's tokens and slot mass bit for bit: {same_default}",
           bad["max_abs_logit_diff"] > CACHE_CONTROL_MIN,
           f"the same steps on default positions: {bad['max_abs_logit_diff']:.2f} (floor {CACHE_CONTROL_MIN:g}; per step "
           f"{', '.join(f'{v:.2f}' for v in bad['per_step'])})")


def y16():
    y16_dataset()
    y16_generation()


# ================================================================ Step 7: the evaluation tools
STEP0_REPORTS = Path("/data/enric/reports/temporal")
STEP0_CODE = REPO / "eo/experiments/temporal"
REPRO_TOL = 1e-4          # eval_temporal.REPRO_TOL; Step 0's own control reproduced stored CE to 1.5e-6
WORKING_RANGE = 1_600     # T360's working range in tokens (finding 4): its baselines stay inside it


def _layouts():
    from eo.eval import temporal as ET
    return ET.load_layouts(allow_unfrozen=ET.FROZEN_LAYOUTS.get(ET.LAYOUTS_TAG) is None)


def _step0_modules():
    if str(STEP0_CODE) not in sys.path:
        sys.path.insert(0, str(STEP0_CODE))
    import t360_best_layouts as BL
    import t360_context as C
    return C, BL


def layout_source_problems(doc) -> list:
    """The layout file against its sources: every Step 0 layout against Step 0's own functions
    (t360_context.layouts, its one_other, length, pairs and coords; t360_best_layouts.LAYOUTS; t360_lattice's
    profile list), every T360-T task layout against task_blocks, and the target-date rules against
    t360_context.date_of. Rows are a location's four, 100 + date."""
    from eo.eval import temporal as ET
    C, BL = _step0_modules()
    L, r, out = doc["layouts"], [100, 101, 102, 103], []

    def check(name, m, t, want):
        got = [(b.modality, r[b.t]) for b in ET.layout_blocks(L[name], m, t)]
        if got != [(k, x) for k, x in want] + ([(m, r[t])] if t is not None else []):
            out.append(f"{name} {m} t{t}")

    for m in C.TARGETS:
        for t in range(4):
            o = [x for x in range(4) if x != t]
            for nm, blocks in C.layouts(m, r, t):
                check(f"s0.{nm}", m, t, blocks)
            for k in range(3):
                check(f"s0.one_other_{k}", m, t, [(m, r[o[k]])])
            check("s0.one_near", m, t, [(m, r[C.near(t)])])
            for nm, fn in BL.LAYOUTS.items():
                check(f"s0.{nm}", m, t, fn(m, r, t))
    for t in range(4):
        tail = C.date_blocks(r, t) + C.others_at("S2L2A", r, t)
        for k in C.LENGTH_K:
            check(f"s0.length_k{k}", "S2L2A", t, tail[-k:])
        check("s0.pair_ndvi", "NDVI", t, [("S2L2A", r[t])])
        check("s0.pair_cross", "S2L2A", t, [("S2L2A", r[C.near(t)])])
        check("s0.pair_ndvi_none", "NDVI", t, [])
        check("s0.pair_cross_none", "S2L2A", t, [])
        c = ("Coords", r[t])
        for src, nm in ((("S2L2A", r[t]), "ndvi"), (("S2L2A", r[C.near(t)]), "cross")):
            tgt = "NDVI" if nm == "ndvi" else "S2L2A"
            check(f"s0.coords_{nm}_between", tgt, t, [src, c])
            check(f"s0.coords_{nm}_before", tgt, t, [c, src])
    # t360_lattice.profile builds its list inline: Coords, DEM, LULC of date 0, then the targets by date.
    check("s0.profile", "*", None, [("Coords", r[0]), ("DEM", r[0]), ("LULC", r[0])]
          + [(m, r[t]) for t in range(4) for m in C.TARGETS])
    for i in range(12):
        for j, m in enumerate(C.TARGETS):
            if ET.target_date(L["s0.none"], i, m) != C.date_of(i, j):
                out.append(f"target date of location {i}, {m}")
        for nm, m in (("s0.pair_ndvi", "NDVI"), ("s0.pair_cross", "S2L2A"), ("s0.length_k3", "S2L2A")):
            if ET.target_date(L[nm], i, m) != C.date_of(i, 0):
                out.append(f"target date of location {i}, {nm}")
    for name in ("t.left_out", "t.forecast", "t.gap_fill", "t.date"):
        e = L[name]
        for m in e["targets"]:
            for t in ([3] if name == "t.forecast" else range(4)):
                if ET.layout_blocks(e, m, t) != ET.task_blocks(e["task"], t, m):
                    out.append(f"{name} {m} t{t} is not task_blocks'")
    return out


def baseline_problems(doc) -> list:
    """T360's frozen baselines: named for every task's targets, layouts of the file, no date block, inside
    T360's working range, and for the baselines chosen in Step 7 (forecast, gap fill) no block that
    T360-T's prompt lacks. The left-out season's is Step 0's statics_same_mod, whose LULC comes from the
    near date's row where T360-T's prompt holds the first context date's (measured in Step 7)."""
    from eo.eval import temporal as ET
    L, B, out = doc["layouts"], doc.get("baselines") or {}, []
    for task in ("left_out", "forecast", "gap_fill"):
        tl = L[f"t.{task}"]
        for m in tl["targets"]:
            name = (B.get(task) or {}).get(m)
            if name not in L:
                out.append(f"{task} {m}: no baseline ({name})")
                continue
            e = L[name]
            for t, spec in e["blocks"][m].items():
                blocks = ET.parse_blocks(spec)
                n = sum(ET.block_width(b) for b in blocks)
                if any(b.modality == "Date" for b in blocks) or n > WORKING_RANGE:
                    out.append(f"{task} {m} t{t}: {name} holds a date block or {n} tokens")
                if task != "left_out":
                    have = {(b.modality, b.t) for b in ET.layout_blocks(tl, m, int(t))[:-1]}
                    extra = [b for b in blocks[:-1] if (b.modality, b.t) not in have]
                    if extra:
                        out.append(f"{task} {m} t{t}: {name} holds {extra}, which T360-T's prompt lacks")
    if B.get("date") != "prior (Step 12)":
        out.append(f"the date task's baseline is {B.get('date')!r}, not the prior")
    return out


def y19():
    """Step 7's layout file: frozen, equal to its sources, and T360's baselines inside the rules."""
    import copy
    from eo.eval import temporal as ET
    frozen = ET.FROZEN_LAYOUTS.get(ET.LAYOUTS_TAG)
    doc = _layouts()
    src = layout_source_problems(doc)
    base = baseline_problems(doc)
    ok = frozen is not None and doc["sha256"] == frozen and not src and not base and not ET.layout_problems(doc)
    # Controls: a frozen file edited by one block is refused; a Step 0 layout with two blocks swapped and a
    # layout holding its own target are flagged; a baseline with a date block, or past the range, is flagged.
    bad = copy.deepcopy({k: v for k, v in doc.items() if k != "sha256"})
    spec = bad["layouts"]["s0.statics_same_mod"]["blocks"]["S2L2A"]["2"].split()
    spec[3], spec[4] = spec[4], spec[3]
    bad["layouts"]["s0.statics_same_mod"]["blocks"]["S2L2A"]["2"] = " ".join(spec)
    swapped = any("s0.statics_same_mod S2L2A t2" in p for p in layout_source_problems(bad))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / ET.layouts_path().name
        p.write_text(json.dumps(bad, indent=1) + "\n")
        edited = frozen is not None and refused(ET.load_layouts, path=p)
    inside = copy.deepcopy(bad)
    inside["layouts"]["t.left_out"]["blocks"]["NDVI"]["1"] = "NDVI@1 " + inside["layouts"]["t.left_out"]["blocks"]["NDVI"]["1"]
    target_in = any("also in the context" in p for p in ET.layout_problems(inside))
    dated = copy.deepcopy(bad)
    name = (dated.get("baselines") or {}).get("forecast", {}) or {}
    name = name.get("S2L2A") if isinstance(name, dict) else None
    date_caught = False
    if name in dated["layouts"]:
        e = dated["layouts"][name]["blocks"]["S2L2A"]
        e["3"] = "Date@2 " + e["3"]
        date_caught = any("date block" in p for p in baseline_problems(dated))
    B = doc.get("baselines") or {}
    record("Y19 layouts", ok, f"temporal_layouts_{ET.LAYOUTS_TAG}.json, {len(doc['layouts'])} layouts, sha "
           f"{doc['sha256'][:12]} ({'frozen' if frozen else 'NOT FROZEN'}); every Step 0 layout equals Step 0's own "
           f"functions, every T360-T task layout equals task_blocks, the target dates t360_context.date_of "
           f"({len(src)} problems); T360's baselines: forecast {B.get('forecast')}, gap fill {B.get('gap_fill')}, "
           f"date {B.get('date')!r} ({len(base)} problems{': ' + str(base[:3]) if base else ''})"
           + (f"; {src[:3]}" if src else ""),
           swapped and edited and target_in and date_caught,
           f"a Step 0 layout with two blocks swapped flagged: {swapped}; the file edited by one block refused by its "
           f"pinned hash: {edited}; a target in its own context flagged: {target_in}; a date block in T360's "
           f"forecast baseline flagged: {date_caught}")


def _step0_unit(e, m, delta):
    return np.load(STEP0_REPORTS / e["step0_unit"].replace("{M}", m).replace("{k}", str(delta // 198 + 1)))


def y20():
    """The scorer's three paths on the real T360-T base, CPU, fp32: the target path reproduces Step 0's
    stored units, the profile path a direct per-token pass, the date path a direct log-softmax."""
    if _no_base("Y20 scorer"):
        return
    from eo.eval import temporal as ET
    f = fx()
    T = f.T
    A = t360t().model
    temporal = T.TemporalLocations(f.base, f.index, f.dev, T.TemporalConfig(), 42)
    L = _layouts()["layouts"]
    locs, _, _ = ET.step0_locations(f.index)
    n = 4

    def items(name, m, delta=0, shift=0, with_dates=True):
        e, out = L[name], []
        for i in range(n):
            t = ET.target_date(e, i, m)
            ts = ET.task_sequence(temporal, int(locs[i]), ET.layout_blocks(e, m, (t + shift) % 4, with_dates))
            out.append(ET.scored(ET.move_target(ts, delta) if delta else ts))
        return out

    units = [("s0.none", m, 0) for m in GROUP] + [("s0.statics_same_mod", "NDVI", 0), ("s0.same_date", "S1GRD", 0),
                                                  ("s0.pair_cross", "S2L2A", 198)]
    worst, flips, ctrl = 0.0, 0, []
    for name, m, delta in units:
        got = ET.score_target(A, items(name, m, delta), m, 4, dtype=None, device="cpu")
        ref = _step0_unit(L[name], m, delta)
        worst = max(worst, float(np.abs(got["ce"] - ref["ce"][:n]).max()))
        flips += int(np.abs(np.round((got["top1"] - ref["top1"][:n]) * 196)).sum())     # in tokens
    for name, m, delta in units[:2]:                 # control: the same units scored at the next target date
        got = ET.score_target(A, items(name, m, delta, shift=1), m, 4, dtype=None, device="cpu")
        ctrl.append(float(np.abs(got["ce"] - _step0_unit(L[name], m, delta)["ce"][:n]).mean()))
    unmasked = [ET.task_sequence(temporal, int(locs[0]), ET.layout_blocks(L["s0.statics_same_mod"], "NDVI", 2)).full]
    unmasked_refused = refused(ET.score_target, A, unmasked, "NDVI", 1, dtype=None, device="cpu")

    # The profile path: per-block means from run_pass's fp16 per-token arrays, against fp32 CE taken directly.
    prof = L["t.profile"]
    blocks = ET.layout_blocks(prof, "*", None)
    seqs = [ET.task_sequence(temporal, int(locs[i]), blocks).full for i in range(2)]
    pb = ET.score_blocks(A, seqs, blocks, 1, dtype=None, device="cpu")
    direct, step = np.empty_like(pb), np.empty_like(pb)
    with torch.no_grad():
        for i, s in enumerate(seqs):
            lg = A(input_ids=s["input_ids"][None], position_ids=s["position_ids"][None],
                   attention_mask=s["attention_mask"][None], use_cache=False).logits[0, :-1].float()
            c64 = torch.nn.functional.cross_entropy(lg, s["input_ids"][1:], reduction="none").double().numpy()
            for k, (st, b) in enumerate(zip(ET.block_starts(blocks), blocks)):
                body = c64[st:st + ET.block_width(b) - 2]
                direct[i, k], step[i, k] = body.mean(), _token_tol(body, body).max()
            del lg
    # Each stored token is within one fp16 step (or 1e-4) of fp32 (Y12's rule), so a block's mean is within the
    # largest step among its tokens: a 2-token date block can move by half a step at CE ~13.
    prof_gap = float(np.abs(pb - direct).max())
    prof_ratio = float((np.abs(pb - direct) / step).max())
    misaligned = refused(ET.score_blocks, A, seqs[:1], blocks[1:] + blocks[:1], 1, dtype=None, device="cpu")

    # The date path: CE of Y and D and the distributions over years and days, against a direct log-softmax.
    de = L["t.date"]
    ds = [ET.task_sequence(temporal, int(locs[i]), ET.layout_blocks(de, "Date", ET.target_date(de, i, "Date"))).full
          for i in range(n)]
    dr = ET.score_dates(A, ds, 4, dtype=None, device="cpu")
    with torch.no_grad():
        x = torch.stack([s["input_ids"] for s in ds])
        lg = A(input_ids=x, position_ids=torch.stack([s["position_ids"] for s in ds]), use_cache=False).logits.float()
        lsm = torch.log_softmax(lg[:, -4:-2], -1)
        want_y = -lsm[:, 0].gather(-1, x[:, -3:-2]).squeeze(-1)
        want_d = -lsm[:, 1].gather(-1, x[:, -2:-1]).squeeze(-1)
        py = torch.softmax(lg[:, -4, TV.YEAR_IDS], -1)
    date_gap = max(float(np.abs(dr["ce_y"] - want_y.numpy()).max()), float(np.abs(dr["ce_d"] - want_d.numpy()).max()),
                   float(np.abs(dr["p_year"] - py.numpy()).max()))
    year_less = []
    for i in range(1):
        b = ET.layout_blocks(de, "Date", 0)
        b[-1] = replace(b[-1], with_year=False)
        year_less.append(ET.task_sequence(temporal, int(locs[i]), b).full)
    yearless_refused = refused(ET.score_dates, A, year_less, 1, dtype=None, device="cpu")
    ok = worst <= REPRO_TOL and prof_ratio <= 1.0 and date_gap <= LOGIT_TOL
    record("Y20 scorer", ok, f"T360-T's base on CPU, fp32, Step 0's first {n} locations and target dates: the target "
           f"path (run_pass, labels masked to the target) gives Step 0's stored per-location CE (T360, Titan X) on "
           f"{len(units)} units within {worst:.1e} (tolerance {REPRO_TOL:g}), top-1 equal on all but {flips} tokens; the profile path's "
           f"block means, from run_pass's fp16 per-token arrays, equal fp32 block means taken directly within one fp16 "
           f"step of their tokens (largest gap {prof_gap:.1e}, {prof_ratio:.2f} of a step), on two whole chronological "
           f"sequences with date blocks; the date path's CE of Y and "
           f"D and its year distribution equal a direct log-softmax within {date_gap:.1e} (Y {dr['ce_y'].mean():.2f}, "
           f"D {dr['ce_d'].mean():.2f} nats on the untrained rows)",
           min(ctrl) > 100 * REPRO_TOL and unmasked_refused and misaligned and yearless_refused,
           f"the same units at the next target date: mean |dCE| {', '.join(f'{c:.3f}' for c in ctrl)}; labels left on "
           f"every body refused: {unmasked_refused}; a profile with its blocks off by one refused: {misaligned}; a "
           f"date block without its year refused: {yearless_refused}")


def move_problems(ts, moved, which: str, day_shift: int, year_shift: int) -> list:
    """What a wrong date may change (eo/eval/temporal.move_dates): the year and day tokens of the target
    date's own date block ('target') or of every date block before the target ('all'), in input_ids and
    labels alike, to the moved date; nothing else, anywhere."""
    from eo.eval import temporal as ET
    blocks = list(ts.draw.blocks)
    starts = ET.block_starts(blocks)
    k = starts.index(ts.target[0])
    dated = [i for i, b in enumerate(blocks[:k]) if b.modality == "Date"]
    chosen = [i for i in dated if blocks[i].t == blocks[k].t][-1:] if which == "target" else dated
    want = ts.full["input_ids"].clone()
    for i in chosen:
        s = starts[i]
        y, d = TV.decode_date_block(want[s:s + ET.block_width(blocks[i])].tolist())
        want[s:s + ET.block_width(blocks[i])] = torch.tensor(TV.date_block(y + year_shift, (d + day_shift) % N_DOY_BINS))
    out = []
    if not torch.equal(moved.full["input_ids"], want):
        out.append(f"input_ids differ from the moved date at {torch.nonzero(moved.full['input_ids'] != want).flatten()[:4].tolist()}")
    if not torch.equal(moved.full["labels"], want):
        out.append("labels do not follow the moved tokens")
    for key in ("position_ids", "modality_ids", "attention_mask"):
        if not torch.equal(moved.full[key], ts.full[key]):
            out.append(f"{key} changed")
    if any(not torch.equal(moved.prompt[key], moved.full[key][: ts.target[0] + 1]) for key in KEYS):
        out.append("the prompt is not the moved sequence's prefix")
    return out


def y21():
    """The wrong date and the date probe change the date tokens and nothing else."""
    from eo.eval import temporal as ET
    f = fx()
    T = f.T
    temporal = T.TemporalLocations(f.base, f.index, f.dev, T.TemporalConfig(), 42)
    doc = _layouts()
    L, p = doc["layouts"], doc["probe"]
    locs, _, _ = ET.step0_locations(f.index)
    problems, n, changed = [], 0, {"target": set(), "all": set(), "year": set()}
    for i in range(16):
        for m in GROUP:
            e = L[p["layout"]]
            t = ET.target_date(e, i, m)
            ts = ET.task_sequence(temporal, int(locs[i]), ET.layout_blocks(e, m, t))
            for key, which, ds, ys in (("target", p["which"], p["day_shift"], p["year_shift"]),
                                       ("all", "all", p["day_shift"], p["year_shift"]), ("year", "target", 0, -1)):
                mv = ET.move_dates(ts, which, ds, ys)
                problems += [f"loc {i} {m} {key}: {q}" for q in move_problems(ts, mv, which, ds, ys)]
                changed[key].add(int((mv.full["input_ids"] != ts.full["input_ids"]).sum()))
            n += 1

    def first_block(ts, which, ds, ys):             # moves the sequence's first date block, not the target's
        full = {k: v.clone() for k, v in ts.full.items()}
        s = next(st for st, b in zip(ET.block_starts(ts.draw.blocks), ts.draw.blocks) if b.modality == "Date")
        y, d = TV.decode_date_block(full["input_ids"][s:s + 4].tolist())
        full["input_ids"][s:s + 4] = full["labels"][s:s + 4] = torch.tensor(TV.date_block(y, (d + ds) % N_DOY_BINS))
        return ET._with(ts, full)

    def and_position(ts, which, ds, ys):            # the right date moved, and the target shifted one position
        return ET.move_target(ET.move_dates(ts, which, ds, ys), 1)

    def tokens_only(ts, which, ds, ys):             # the input moved, the labels left on the right date
        mv = ET.move_dates(ts, which, ds, ys)
        full = dict(mv.full, labels=ts.full["labels"].clone())
        return ET._with(ts, full)

    ts = ET.task_sequence(temporal, int(locs[3]), ET.layout_blocks(L[p["layout"]], "NDVI", ET.target_date(L[p["layout"]], 3, "NDVI")))
    caught = {fn.__name__: bool(move_problems(ts, fn(ts, "target", 36, 0), "target", 36, 0))
              for fn in (first_block, and_position, tokens_only)}
    de = L["t.date"]
    date_ts = ET.task_sequence(temporal, int(locs[0]), ET.layout_blocks(de, "Date", 0))
    target_refused = refused(ET.move_dates, date_ts, "target", 36, 0)
    nodate = ET.task_sequence(temporal, int(locs[0]), ET.layout_blocks(L[p["layout"]], "S2L2A", 0, with_dates=False))
    nodate_refused = refused(ET.move_dates, nodate, "target", 36, 0)
    record("Y21 wrong date", not problems and changed == {"target": {1}, "all": {4}, "year": {1}},
           f"{n} left-out-season sequences ({p['layout']}, 16 of Step 0's locations, every target): the probe "
           f"({p['which']}, day bin +{p['day_shift']} mod 73, year {p['year_shift']:+d}) changes exactly "
           f"{sorted(changed['target'])} token, every date block moved ('all') {sorted(changed['all'])}, a year moved "
           f"{sorted(changed['year'])}: the moved date's tokens in input_ids and labels, nothing else"
           + (f"; {problems[:3]}" if problems else ""),
           all(caught.values()) and target_refused and nodate_refused,
           f"broken movers flagged: {caught}; a wrong date on the date task's own target refused: {target_refused}; on "
           f"a sequence without date blocks refused: {nodate_refused}")


Y22_OUT = Path("/data/enric/reports/arm_eval/t360t_base_fp32/extended_base")
Y22_REF = Path("/data/enric/reports/arm_eval/t360_fp32/checkpoint-17441")
Y22_CMD = ("eval_checkpoint.py --arm-tag t360t_base_fp32 --arm-config eo_terramesh/arm_t360_vanilla --model-config "
           "eo_terramesh/arm_t360t_vanilla --run-dir <any> extended_base --precision fp32 --grid --grid-all-others, on "
           "the GTX Titan X (bash /data/enric/reports/temporal/step7/run_checkpoint.sh, about 2 h)")
ROW_TOL = 1e-4            # a row's mean CE: Y12's per-token floor, which a mean of tokens within it cannot exceed


def _row_gaps(a, r, key: str):
    """Rows and token counts equal, and |mean CE per row| differences, for one target-last or grid key."""
    same = np.array_equal(a[f"{key}|rows"], r[f"{key}|rows"]) and np.array_equal(a[f"{key}|n"], r[f"{key}|n"])
    if not same:
        return False, np.inf, 0
    n = np.maximum(r[f"{key}|n"], 1)
    gap = np.abs(a[f"{key}|ce_sum"] / n - r[f"{key}|ce_sum"] / n)
    return True, float(gap.max()) if len(gap) else 0.0, int(np.abs(a[f"{key}|top1"] - r[f"{key}|top1"]).sum())


def y22():
    """eval_checkpoint.py at 87,664 ids: the extended base through --model-config reproduces T360's stored fp32
    reading (per token, target-last per row, every grid cell per row) and adds the grid's "all" row."""
    if not (Y22_OUT / "grid_rows.npz").is_file():
        skip("Y22 eval_checkpoint at 87,664", f"{Y22_OUT} incomplete: the GPU run writes it: {Y22_CMD}")
        return
    if _no_base("Y22 eval_checkpoint at 87,664"):
        return
    from eo.eval import teacher_forced as TF
    from lm_dataset.load_dataset import load_eval_dataset_from_config
    doc, ref = json.loads((Y22_OUT / "metrics.json").read_text()), json.loads((Y22_REF / "metrics.json").read_text())
    ext = json.loads((EXT / "extension.json").read_text())
    meta_ok = (doc.get("vocab") == TV.TOTAL_VOCAB_SIZE and doc.get("model_config") == f"eo_terramesh/{ARM}"
               and (doc.get("loaded") or {}).get("variant") == "continue"
               and (doc.get("loaded") or {}).get("conversion_weights_sha256") == ext["weights_sha256"]
               and doc["precision"] == ref["precision"] == "fp32" and doc["arm_config"] == ref["arm_config"]
               and doc["token_set"] == ref["token_set"] == "clean" and doc["n_eval_rows"] == ref["n_eval_rows"] == 4416
               and "TITAN V" not in doc["gpu"])
    a, r = np.load(Y22_OUT / "tokens_router.npz"), np.load(Y22_REF / "tokens_router.npz")
    rows_ok = (np.array_equal(np.load(Y22_OUT / "eval_rows.npy"), np.load(Y22_REF / "eval_rows.npy"))
               and a["ce"].shape == r["ce"].shape and np.array_equal(a["label_mod"], r["label_mod"]))
    lm = a["label_mod"] > 0
    ca, cr = a["ce"].astype(np.float64)[lm], r["ce"].astype(np.float64)[lm]
    beyond = int((np.abs(ca - cr) > _token_tol(ca, cr)).sum()) if rows_ok else -1
    ta, tr = np.load(Y22_OUT / "target_last_rows.npz"), np.load(Y22_REF / "target_last_rows.npz")
    ga, gr = np.load(Y22_OUT / "grid_rows.npz"), np.load(Y22_REF / "grid_rows.npz")
    keys_t = sorted({k.split("|")[0] for k in tr.files})
    keys_g = sorted({k.split("|")[0] for k in gr.files})
    tl = [_row_gaps(ta, tr, k) for k in keys_t]
    gd = [_row_gaps(ga, gr, k) for k in keys_g]
    all_cells = sorted({k.split("|")[0] for k in ga.files if k.startswith("all->")})
    grid = json.loads((Y22_OUT / "grid.json").read_text())["cells"]
    p3 = [grid[c].get("p3_abs_diff") for c in all_cells if c in grid]
    worst_t = max(g for _, g, _ in tl)
    worst_g = max(g for _, g, _ in gd)
    ok = (meta_ok and rows_ok and beyond == 0 and all(s for s, _, _ in tl + gd) and worst_t <= ROW_TOL
          and worst_g <= ROW_TOL and len(all_cells) == 6 and all(x is not None and x <= 1e-6 for x in p3))
    # Live, on CPU: the first two batches of the held-out loader (batch 4, worker k % 4) through this gate's T360-T
    # give the artifact's per-token CE; with the table shifted by one row they read far off.
    A = t360t().model
    ds = load_eval_dataset_from_config(compose([], name=ARM_T360))
    from unittest import mock
    batches = []
    for k in (0, 1):
        with mock.patch.object(torch.utils.data, "get_worker_info", lambda k=k: SimpleNamespace(id=k % 4)):
            batches.append(torch.utils.data.default_collate([ds[4 * k + i] for i in range(4)]))
    live = TF.run_pass(A, batches, device="cpu", keep_tokens=True, dtype=None)
    lt = live["tokens"]["ce"].astype(np.float64)
    keep = a["label_mod"][:8] > 0
    gap = np.abs(lt - a["ce"][:8].astype(np.float64))[keep]
    live_ok = bool((gap <= _token_tol(lt, a["ce"][:8].astype(np.float64))[keep]).all())
    W = A.model.embed_tokens.weight
    orig = W.detach().clone()
    with torch.no_grad():
        W.copy_(torch.cat([orig.mean(0, keepdim=True), orig[:-1]], 0))
    try:
        shifted = TF.run_pass(A, batches, device="cpu", keep_tokens=False, dtype=None)
    finally:
        with torch.no_grad():
            W.copy_(orig)
    del orig
    rise = shifted["ce_all_body_tokens"] - live["ce_all_body_tokens"]
    record("Y22 eval_checkpoint at 87,664", ok and live_ok,
           f"the extended base at {doc.get('vocab')} ids through --model-config ({doc['gpu']}, fp32), against T360's "
           f"stored fp32 reading ({Y22_REF.parent.name}): per token, {int(lm.sum()):,} tokens of the 4,416 rows, "
           f"{beyond} beyond one fp16 step (floor {TOKEN_FLOOR:g}); target-last, {len(keys_t)} targets, rows and counts "
           f"equal, largest per-row gap {worst_t:.1e}; the grid, {len(keys_g)} cells, rows equal, largest per-row gap "
           f"{worst_g:.1e} (tolerance {ROW_TOL:g}); the 'all' row added for {len(all_cells)} targets, P3 within "
           f"{max(p3) if p3 else float('nan'):.1e}; live on CPU, the first 8 rows equal the artifact's tokens: {live_ok}",
           rise > CONTROL_MIN, f"the table shifted by one row: CE {shifted['ce_all_body_tokens']:.2f} against "
           f"{live['ce_all_body_tokens']:.2f} ({rise:+.2f})")


CACHE_ART = Path("/data/enric/reports/temporal/step7/t360t_base/extended_base/cache/cache_check.json")
CACHE_CMD = ("bash /data/enric/reports/temporal/step7/run_cache.sh  (eval_temporal.py cache --tag t360t_base on the "
             "GTX Titan X, about 20 min)")


def y23():
    """The cache at the task prompts' length on the GTX Titan X, read from the GPU run's artifact."""
    if not CACHE_ART.is_file():
        skip("Y23 cache at length", f"{CACHE_ART} missing: {CACHE_CMD}")
        return
    doc = json.loads(CACHE_ART.read_text())
    c = doc["checks"]
    need = ("t360/fp32", "t360/fp16", "t360t/fp32", "t360t/fp16")
    lengths = {k: c[k]["prompt_length"] for k in need if k in c}
    ok = (all(k in c and c[k]["passed"] and c[k]["finite"] and c[k]["steps"] >= 197 for k in need)
          and lengths.get("t360t/fp32", 0) >= 2398 and lengths.get("t360/fp32", 0) >= 2382
          and "TITAN X" in doc["gpu"] and all(doc["timing"].get(p, {}).get("finite") for p in ("fp32", "fp16")))
    ctrl = [c[k]["max_abs_logit_diff"] for k in ("t360/control_offset_1", "t360t/control_offset_1",
                                                 "t360t/control_default_positions") if k in c]
    record("Y23 cache at length", ok, f"{doc['gpu']}: " + "; ".join(
        f"{k} prompt {c[k]['prompt_length']}, {c[k]['steps']} steps: {c[k]['max_abs_logit_diff']:.1e} "
        f"({c[k]['max_ulp_diff']:.1f} steps at |logit| {c[k]['max_abs_logit']:.0f}), passed {c[k]['passed']}"
        for k in need if k in c) + f"; decode of {doc['timing'].get('fp32', {}).get('batch')} blocks: " + ", ".join(
        f"{p} {v['seconds_per_block']} s a block" for p, v in doc["timing"].items()),
        len(ctrl) == 3 and min(ctrl) > CACHE_CONTROL_MIN,
        f"decoded one position off (T360, T360-T) and on default positions behind date blocks: "
        f"{', '.join(f'{x:.2f}' for x in ctrl)} (floor {CACHE_CONTROL_MIN:g})")


CHECKS = {"Y1": y1, "Y2": y2, "Y3": y3, "Y4": y4, "Y5": y5, "Y6": y6, "Y7": y7, "Y8": y8, "Y9": y9, "Y10": y10,
          "Y11": y11, "Y12": y12, "Y13": y13, "Y14": y14, "Y15": y15, "Y16": y16, "Y17": y17, "Y18": y18,
          "Y19": y19, "Y20": y20, "Y21": y21, "Y22": y22, "Y23": y23}


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
    if _RUNS.get("root") is not None:
        if failed:
            print(f"(the miniature's runs are kept for inspection: {_RUNS['root']})")
        else:
            shutil.rmtree(_RUNS["root"], ignore_errors=True)
    print(f"\n{len(RESULTS) - len(failed) - len(skipped)} passed, {len(failed)} failed, {len(skipped)} skipped"
          + (f": {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
