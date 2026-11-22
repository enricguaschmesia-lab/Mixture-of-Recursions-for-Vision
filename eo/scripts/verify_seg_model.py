#!/usr/bin/env python
"""A2's segmentation gate, the model and data half (`.venv`; A2 Step 3.5). verify_segmentation.py (`a2seg`) runs
it as a subprocess and merges its results; it also runs on its own. Every check ships with a control that
must fail it.

    python eo/scripts/verify_seg_model.py [--only Z4,Z5] [--out results.json]
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<titanx> python eo/scripts/verify_seg_model.py   # Z8 on a card

  Z4  Map rows (eo/data/map_vocab.py, extend_checkpoint.py --layout map): the layout is the contiguous block
      after the base's last id (81,296 -> 161,665), slot sizes = the arms' codebooks, PAD a base id; the map
      base's extension.json passes verify_extension (identity passed, both controls fired, weights hash) and
      records this layout; live, in fp32 on CPU, the base's logits over its 81,296 ids on PASTIS-1 and
      PASTIS-7 prompts equal the map base's within 1e-4, and the head is tied
                                     | the rows inserted before the base's last id: layout refused, and live
                                       logits move; verify_extension at another vocab size refuses
  Z5  Sequences (eo/data/seg_dataset.py), every task x frames x arm: the five keys; the length the layout
      says; target last (the map block is the last real chunk, BO_MAP_<task> right before the first label);
      loss on the G codes and EO_MAP only, equal to the inputs there; modality ids (S2L2A 1, Date 9, map 10);
      codes inside the arm's slot; lattice positions (image blocks at 198 j, the map after them); a view's
      input rows and map row are that view's (read again from inputs.parquet); draws keyed on (seed, epoch,
      sample) only, one example per sample an epoch, a whole number of steps; in_pinned selects 1,200 / 255;
      a test split refused
                                     | the default labels (loss on every token) flagged; a map one view off
                                       flagged
      (Its date and dihedral-content halves read PASTIS-R and decode maps: verify_segmentation.py, a2seg.)
  Z8  KV cache: on the map base (fp32, on a card when one is visible), a cached decode equals full recomputation
      after 6 steps within max(4 rounding steps, 2 x the round-off floor: the same prompt recomputed with a
      trailing pad token), on PASTIS-1, a PASTIS-7 prompt (lattice positions) and CloudSen12; every generated
      code is in the arm's slot. (The 4-step rule alone, X15's, was set at fp16/bf16; fp32 reads 8-30 steps,
      the floor's size, on the Titan X and on CPU, 2026-11-19.)
                                     | the cache fed positions one off moves the logits past the tolerance
  Z9  Wrong input: a derangement (no sample keeps its own input; CloudSen12 keeps the crop index), and every
      val view's wrong input differs from its own
                                     | the view's own input through the same path is refused as "not wrong"
      Its functional half (wrong input scores near the prior) needs a fine-tune: SKIPPED until Step 4.0's.
  Z10 LoRA through pretrain.py (2 steps, CPU fp32, weight decay 0.1): every frozen tensor bit-identical to the
      map base, the map rows and the adapters moved, the trainable count = adapters + 80,369 x d; the epoch
      callback set draw epoch 0
                                     | the masked-gradient variant under AdamW moves the masked rows
  Z12 Overlays: each Step 4.0 overlay differs from r0_base in exactly seg.arm and max_length, max_length is
      the arm's sequence length, the vocab is the registry's, precision bf16
                                     | an overlay with a stray key (learning_rate) is flagged
  Z13 Step 4.0's follow-ups (2026-11-20, evening): r0_base_e40 = r0_base at 40 epochs; each follow-up overlay differs
      from it in exactly its seg keys and max_length; the aliases sit inside their parent slots and the recorded layout
      equals extension.json's; the 99 shift moves the map block and nothing else; the default layout is Step 4.0's;
      the nested layout puts every code 198 (j + 1) after its input patch's token as that patch's sub-cell j, positions
      rising, EO_MAP at 989, and its codes go back to raster order exactly
                                     | a code past the alias refused; patch-major order and positions one off
                                       flagged; nested on a 14 x 14 map refused
      Z13cache: Z8's rule at 6 steps under the nested and shifted layouts; at 200 steps (across the first block
      boundary) the nested drift within 3x the raster layout's own fp32 drift
                                     | the cache one position off
      Z13train: 2 steps of the nested layout through pretrain.py (CPU)
                                     | the nested layout on the aligned arm, refused by pretrain.py's dataset
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eo.data import map_vocab as MV            # noqa: E402
from eo.data import seg_dataset as SD          # noqa: E402

RESULTS: list = []
BASE_DIR = Path("/data/enric/runs/pretrain/temporal/base/b10cltrxnc_ext_temporal_nc_ckpt33000")
MAP_BASE = Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000")
BASE_ARM = "eo_terramesh/arm_b10ctrxnc_vanilla"
OVERLAYS = {arm: f"eo_seg/r0_pastis1_{arm}" for arm in MV.ARMS}
LAYOUTS = [("pastis", "p1"), ("pastis", "p7"), ("cloudsen12", "single")]
TOL = 1e-4


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def skip(name: str, why: str) -> None:
    RESULTS.append((name, "SKIP", why))
    print(f"[SKIP] {name}: {why}", flush=True)


def _compose(name: str, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


# ---------------------------------------------------------------- Z4
def z4() -> None:
    from eo.finetune import extend as X
    from eo.finetune.init import build_finetuned, verify_extension
    from eo.generate.conditional import build_model

    lay = MV.LAYOUT
    p = MV.layout_problems(lay)
    record("Z4 layout", not p and lay.base_vocab == 81_296 and lay.total == 161_665 and lay.pad_id == 81_187,
           f"base {lay.base_vocab}, slots {lay.slots}, EO_MAP {lay.eo_map}, BO_MAP {min(lay.bo_map.values())}.."
           f"{max(lay.bo_map.values())}, total {lay.total}, PAD {lay.pad_id}; problems {p}")
    sh = MV.build(lay.base_vocab - 1, lay.base)                  # every map id one lower: before the base's last
    bad = MV.MapLayout(base_vocab=lay.base_vocab, base=lay.base, slots=sh.slots, eo_map=sh.eo_map,
                       bo_map=sh.bo_map, total=sh.total)
    pb = MV.layout_problems(bad)
    record("Z4 control: rows inserted before the base's last id", bool(pb), f"refused: {pb[:2]}")

    if not (MAP_BASE / "extension.json").exists():
        skip("Z4 extension", f"{MAP_BASE} not built")
        return
    doc = verify_extension(MAP_BASE, MV.TOTAL_VOCAB_SIZE)
    ok = (doc["layout"] == "map" and doc["map_vocab"] == json.loads(json.dumps(lay.record()))
          and doc["control_insert_before_pad"]["inserted_at"] == lay.base_vocab - 1
          and doc["extension"]["vocab_before"] == lay.base_vocab)
    record("Z4 extension.json", ok, f"identity {doc['identity']['max_abs_diff']:.2e} ({doc['identity']['precision']}, "
           f"{doc['gpu']}), control 1 {doc['control_insert_before_pad']['max_abs_diff']:.3f} at id "
           f"{doc['control_insert_before_pad']['inserted_at']}, untied control fired, layout recorded = map_vocab's")
    try:
        verify_extension(MAP_BASE, MV.TOTAL_VOCAB_SIZE - 1, check_hash=False)
        record("Z4 control: another vocab size", False, "accepted")
    except RuntimeError as e:
        record("Z4 control: another vocab size", True, f"refused: {str(e)[:80]}")

    # live identity on segmentation prompts (ids < 81,188 only), fp32 CPU
    torch.manual_seed(0)
    base, _ = build_model(BASE_ARM, str(BASE_DIR), device="cpu", overrides=["model_config.vocab_size=81296"])
    base.float().eval()
    mp, _ = build_finetuned(OVERLAYS["aligned"], None, device="cpu", overrides=[f"finetune.init_from={MAP_BASE}"])
    mp.float().eval()
    X.assert_tied(mp)
    prompts = []
    for task, fr in (("pastis", "p1"), ("pastis", "p7")):
        ds = SD.SegDataset(SD.SegSpec(task, "aligned", fr, "val"))
        ids, pos = ds.prompt(int(ds.views.view.values[0]))
        prompts.append((ids[:-1], pos[:-1]))           # without BO_MAP: base ids only
    diffs, ctrl = [], []
    cm = copy.deepcopy(base)
    X.extend_vocab(cm, MV.N_NEW, seed=0, insert_before_pad=True, pad_id=lay.base_vocab - 1)
    with torch.no_grad():
        for ids, pos in prompts:
            a = base(input_ids=ids[None], position_ids=pos[None]).logits[0].float()
            b = mp(input_ids=ids[None], position_ids=pos[None]).logits[0].float()[:, : lay.base_vocab]
            c = cm(input_ids=ids[None], position_ids=pos[None]).logits[0].float()[:, : lay.base_vocab]
            diffs.append(float((a - b).abs().max()))
            ctrl.append(float((a - c).abs().max()))
    record("Z4 live identity", max(diffs) <= TOL and mp.config.vocab_size == lay.total,
           f"max |dlogit| over the base's {lay.base_vocab} ids: PASTIS-1 {diffs[0]:.2e}, PASTIS-7 {diffs[1]:.2e} "
           f"(tol {TOL}); tied head at {mp.config.vocab_size}")
    record("Z4 live control: inserted before the last id", min(ctrl) > TOL, f"max |dlogit| {ctrl}")


# ---------------------------------------------------------------- Z5
def sequence_problems(ds: SD.SegDataset, s: dict, view: int) -> list:
    out = []
    if set(s) != {"input_ids", "attention_mask", "labels", "position_ids", "modality_ids"}:
        out.append(f"keys {sorted(s)}")
    L = int(s["attention_mask"].sum())
    if L != ds.required:
        out.append(f"{L} real tokens, layout says {ds.required}")
    lab = s["labels"]
    tgt = (lab != -100).nonzero().flatten()
    G = ds.n_map
    if len(tgt) != G + 1:
        out.append(f"{len(tgt)} target tokens, not G + 1 = {G + 1}")
        return out
    if int(tgt[0]) != L - G - 1 or int(tgt[-1]) != L - 1 or not bool((tgt.diff() == 1).all()):
        out.append("the targets are not the last G + 1 real tokens")
    if not torch.equal(lab[tgt], s["input_ids"][tgt]):
        out.append("labels differ from the inputs at the targets")
    if int(s["input_ids"][tgt[0] - 1]) != ds.task_token:
        out.append("BO_MAP_<task> is not right before the first target")
    if int(s["input_ids"][tgt[-1]]) != ds.lay.eo_map:
        out.append("the last target is not EO_MAP")
    body = s["input_ids"][tgt[:-1]]
    if not bool(((body >= ds.map_off) & (body < ds.map_off + ds.map_size)).all()):
        out.append("a map id outside the arm's slot")
    if not torch.equal(torch.as_tensor(ds.lay.decode_block(ds.spec.arm, body.tolist())),
                       torch.as_tensor(np.asarray(ds.maps[view]).astype(np.int64))):
        out.append("the map body is not the view's map row")
    mid = s["modality_ids"]
    if not bool((mid[tgt[:-1]] == MV.MAP_MODALITY_ID).all()):
        out.append("map body modality id")
    # input blocks
    V = ds.V
    s2 = V.get_modality("S2L2A")
    bo = (s["input_ids"][:L] == s2.bo_id).nonzero().flatten()
    n_img = ds.in_rows.shape[1]
    if len(bo) != n_img:
        out.append(f"{len(bo)} S2L2A blocks, not {n_img}")
    pos = s["position_ids"]
    if [int(pos[b]) for b in bo] != [198 * j for j in range(n_img)]:
        out.append(f"image blocks at positions {[int(pos[b]) for b in bo]}, not 198 j")
    if int(pos[tgt[0] - 1]) != 198 * n_img:
        out.append(f"BO_MAP at position {int(pos[tgt[0] - 1])}, not {198 * n_img}")
    for j, b in enumerate(bo):
        body_in = s["input_ids"][b + 1:b + 197]
        if not bool((mid[b + 1:b + 197] == V.MODALITY_TO_ID["S2L2A"]).all()):
            out.append("S2L2A modality id")
        if not np.array_equal(body_in.numpy() - s2.codebook_offset, np.asarray(ds.s2[ds.in_rows[view, j]]).astype(np.int64)):
            out.append(f"image block {j} is not the view's input row")
    if ds.spec.frames == "p7":
        db = (s["input_ids"][:L] == V.BO_DATE).nonzero().flatten()
        if len(db) != 7 or not bool((mid[db + 1] == V.MODALITY_TO_ID["Date"]).all()):
            out.append("PASTIS-7 date blocks")
    return out


def z5() -> None:
    import pandas as pd
    for task, fr in LAYOUTS:
        for arm in MV.ARMS:
            for split in ("train", "val"):
                ds = SD.SegDataset(SD.SegSpec(task, arm, fr, split), batch=32)
                idx = list(range(0, len(ds), max(1, len(ds) // 6)))[:6]
                probs = []
                for i in idx:
                    v = ds.view_at(i)
                    probs += sequence_problems(ds, ds[i], v)
                record(f"Z5 sequences {task}/{fr}/{arm}/{split}", not probs,
                       f"{len(idx)} examples, {ds.required} tokens, G {ds.n_map}; problems {probs[:3]}")
    # controls on one layout
    ds = SD.SegDataset(SD.SegSpec("pastis", "short", "p7", "val"))
    v = int(ds.views.view.values[3])
    s = ds.sequence(v)
    s_bad = dict(s, labels=s["input_ids"].clone())
    record("Z5 control: loss on every token", bool(sequence_problems(ds, s_bad, v)),
           f"flagged: {sequence_problems(ds, s_bad, v)[:1]}")
    record("Z5 control: a map one view off", bool(sequence_problems(ds, s, v + 1)),
           f"flagged: {[p for p in sequence_problems(ds, s, v + 1) if 'map' in p][:1]}")
    # a view's rows, read again from inputs.parquet; map row = view
    inp = pd.read_parquet(SD.ROOT / "pastis" / "tok" / "inputs.parquet")
    ds = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p7", "train"), batch=32)
    bad = 0
    for i in range(0, len(ds), 37):
        v = ds.view_at(i, epoch=3)
        want = inp[(inp.view == v) & inp.frame.str.startswith("p7")].sort_values("frame").row.values
        bad += not np.array_equal(ds.in_rows[v], want)
    record("Z5 input rows = the view's (inputs.parquet)", bad == 0, f"{len(range(0, len(ds), 37))} draws, {bad} wrong")
    # draws
    a = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "train"), batch=32, seed=42)
    b = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "train"), batch=32, seed=42)
    e0 = [a.view_at(i, 0) for i in range(len(a))]
    e0b = [b.view_at(i, 0) for i in range(len(b))]
    e1 = [a.view_at(i, 1) for i in range(len(a))]
    smp = a.views.set_index("view").sample_idx
    one_each = len(set(smp.loc[e0])) == len(e0)
    ks = a.views.set_index("view").k.loc[e0].value_counts()
    record("Z5 draws", e0 == e0b and e0 != e1 and one_each and len(a) % 32 == 0 and len(ks) == 8,
           f"epoch 0 equal across instances, epoch 1 differs; one view per sample ({len(e0)} of {len(a.samples)} "
           f"samples, {len(a) // 32} steps of 32); dihedral k drawn {dict(sorted(ks.items()))}")
    n = {p: len(SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "train", in_pinned=p), batch=1).samples)
         for p in (None, True, False)}
    record("Z5 in_pinned", n == {None: 1455, True: 1200, False: 255}, f"{n}")
    try:
        SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "test"))
        record("Z5 test refused", False, "built")
    except PermissionError as e:
        record("Z5 test refused", True, str(e)[:70])


# ---------------------------------------------------------------- Z8
def z8() -> None:
    from eo.eval import seg as SEG
    from eo.finetune.init import build_finetuned
    if not (MAP_BASE / "extension.json").exists():
        skip("Z8", "map base not built")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_finetuned(OVERLAYS["aligned"], None, device=dev, overrides=[f"finetune.init_from={MAP_BASE}"])
    model.float().eval()
    where = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    for task, fr, arm, n in (("pastis", "p1", "aligned", 2), ("pastis", "p7", "long", 1), ("cloudsen12", "single", "short", 2)):
        ds = SD.SegDataset(SD.SegSpec(task, arm, fr, "val"))
        pr = [ds.prompt(int(v)) for v in ds.views.view.values[:n]]
        ids = torch.stack([p[0] for p in pr]).to(dev)
        pos = torch.stack([p[1] for p in pr]).to(dev)
        slot = ds.lay.slot(arm)
        r = SEG.cache_check(model, ids, pos, slot, 6)
        record(f"Z8 cache {task}/{fr}/{arm}", r["within"] and r["codes_in_slot"],
               f"{where} fp32: max |dlogit| {r['max_abs_logit_diff']:.2e} ({r['max_ulp_diff']:.1f} rounding steps at "
               f"|logit| <= {r['max_abs_logit']:.1f}); round-off floor {r['floor']:.2e}, tol {r['tol']:.2e} = max("
               f"{SEG.CACHE_ULPS} steps, {SEG.FLOOR_FACTOR:g} x floor); codes in slot {r['codes_in_slot']}")
        c = SEG.cache_check(model, ids, pos, slot, 6, cache_offset=1)
        record(f"Z8 control: cache positions one off ({task}/{fr})", not c["within"],
               f"max |dlogit| {c['max_abs_logit_diff']:.2e} against tol {c['tol']:.2e}")
        with torch.no_grad():
            free = model(input_ids=ids, position_ids=pos, logits_to_keep=1).logits[:, -1].argmax(-1)
        codes, _ = SEG.generate(model, ids, pos, slot, 8, greedy=False, seed=1)
        record(f"Z8 constrained {task}/{arm}", bool(((codes >= 0) & (codes < slot[1])).all()),
               f"8 sampled codes x {n} in 0..{slot[1] - 1}; unconstrained argmax ids {free.tolist()} "
               f"({'outside' if not bool(((free >= slot[0]) & (free < slot[0] + slot[1])).all()) else 'inside'} the slot)")


# ---------------------------------------------------------------- Z9
def z9() -> None:
    for task, fr, arm in (("pastis", "p1", "aligned"), ("pastis", "p7", "short"), ("cloudsen12", "single", "long")):
        ds = SD.SegDataset(SD.SegSpec(task, arm, fr, "val"))
        d = ds.derangement()
        vw = ds.views.set_index("view")
        fixed = sum(vw.sample_idx[v] == vw.sample_idx[w] for v, w in d.items())
        crop = (task != "cloudsen12") or all((vw.y0[v], vw.x0[v]) == (vw.y0[w], vw.x0[w]) for v, w in d.items())
        same = 0
        for v in ds.views.view.values[: min(len(ds.views), 400)]:
            own = torch.cat(ds.input_chunks(int(v), "full")[0])
            wr = torch.cat(ds.input_chunks(int(v), "wrong")[0])
            same += bool(torch.equal(own, wr))
        record(f"Z9 wrong input {task}/{fr}", fixed == 0 and same == 0 and crop,
               f"{len(d)} views: {fixed} keep their sample, {same} of {min(len(ds.views), 400)} get their own input; "
               f"crop index kept {crop}")
    ds = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "val"))
    ds._derange = {int(v): int(v) for v in ds.views.view.values}       # the view's own input through the path
    v = int(ds.views.view.values[0])
    own = torch.cat(ds.input_chunks(v, "full")[0])
    wr = torch.cat(ds.input_chunks(v, "wrong")[0])
    record("Z9 control: own input through the wrong path", bool(torch.equal(own, wr)),
           "identical input: the check above would flag it")
    skip("Z9 functional (wrong ~ prior)", "needs a fine-tuned model: Step 4.0's first R0 run")


# ---------------------------------------------------------------- Z10
def z10() -> None:
    import subprocess
    from eo.finetune import lora as LR
    from eo.finetune.init import _load_state, _weights
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        record("Z10", False, f"MOR_SAVE_DIR {save!r} is not under /data")
        return
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_z10"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name", OVERLAYS["aligned"],
           f"name={out.name}", f"output_dir={rel}", f"wandb_run_name={out.name}", "wandb=false",
           f"finetune.init_from={MAP_BASE}", "num_train_steps=2", "stop_steps=2", "num_warmup_steps=0",
           "lr_scheduler_kwargs.num_decay_steps=0", "save_steps=2", "eval_on_start=false", "eval_steps=1000",
           "total_batch_size=2", "per_device_train_batch_size=2", "logging_steps=1", "dataloader_num_workers=0",
           "precision=fp32", "weight_decay=0.1", f"hydra.run.dir={out}/hydra"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=3600)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pretrain.log").write_text(r.stdout + r.stderr)
    ck = out / "checkpoint-2"
    if r.returncode != 0 or not ck.is_dir():
        record("Z10 pretrain.py", False, f"exit {r.returncode}; log {out / 'pretrain.log'}: "
               + " | ".join((r.stdout + r.stderr).strip().splitlines()[-4:]))
        return
    after = _load_state(_weights(ck))
    before = _load_state(_weights(MAP_BASE))
    nb = MV.LAYOUT.base_vocab
    emb_b = before["model.embed_tokens.weight"]
    moved_frozen, checked = [], 0
    for k, v in after.items():
        if "lora_" in k or k.endswith("embed_tokens.new"):
            continue
        src = k.replace(".base.weight", ".weight")
        if k.endswith("embed_tokens.old"):
            ref = emb_b[:nb]
        else:
            ref = before.get(src)
        if ref is None:
            moved_frozen.append(f"{k}: no base tensor")
            continue
        checked += 1
        if not torch.equal(v, ref):
            moved_frozen.append(k)
    new_moved = not torch.equal(after["model.embed_tokens.new"], emb_b[nb:])
    lora_b = [k for k in after if k.endswith("lora_B")]
    adapters_moved = sum(bool(after[k].abs().sum() > 0) for k in lora_b)
    rep = json.loads((out / "finetune_init.json").read_text())
    d = emb_b.shape[1]
    want_tr = rep["lora"]["adapter_params"] + MV.N_NEW * d
    log = r.stdout + r.stderr
    cb = "SegEpochCallback: global_step 0 -> draw epoch 0" in log
    record("Z10 LoRA through pretrain.py", not moved_frozen and checked > 100 and new_moved and adapters_moved == len(lora_b)
           and rep["trainable_params"] == want_tr and cb,
           f"{checked} frozen tensors bit-identical to the map base (moved: {moved_frozen[:3]}); map rows moved "
           f"{new_moved}; {adapters_moved}/{len(lora_b)} adapters moved; trainable {rep['trainable_params']:,} = "
           f"adapters {rep['lora']['adapter_params']:,} + {MV.N_NEW:,} x {d}; epoch callback {cb}; "
           f"{time.time() - t0:.0f} s, CPU fp32, weight decay 0.1")
    # control: the old rows in the same Parameter, gradient masked, AdamW with decay -> they move
    w = torch.nn.Parameter(emb_b[nb - 64: nb + 64].clone())
    opt = torch.optim.AdamW([w], lr=1e-3, weight_decay=0.1)
    w0 = w.detach().clone()
    loss = (w * torch.randn_like(w)).sum()
    loss.backward()
    w.grad[:64] = 0
    opt.step()
    record("Z10 control: masked gradient under AdamW", not torch.equal(w.detach()[:64], w0[:64]),
           f"masked rows moved by up to {float((w.detach()[:64] - w0[:64]).abs().max()):.2e}")
    del LR


# ---------------------------------------------------------------- Z12
def z12() -> None:
    from omegaconf import OmegaConf
    from lm_dataset.modality_registry import get_registry
    base = _flat(OmegaConf.to_container(_compose("eo_seg/r0_base", ["seg.arm=x", "max_length=1"]), resolve=False))
    intended = {"seg.arm", "max_length"}
    for arm, name in OVERLAYS.items():
        c = _compose(name)
        f = _flat(OmegaConf.to_container(c, resolve=False))
        diff = {k for k in set(base) | set(f) if base.get(k) != f.get(k)}
        ok = (diff == intended and c.seg.arm == arm and c.max_length == SD.required_length("pastis", "p1", arm)
              and c.model_config.vocab_size == get_registry(c).TOTAL_VOCAB_SIZE and c.precision == "bf16"
              and c.finetune.regime == "lora" and float(c.learning_rate) == 1e-3)
        record(f"Z12 overlay {arm}", ok, f"differs from r0_base in {sorted(diff)}; max_length {c.max_length} = "
               f"required; vocab {c.model_config.vocab_size} = registry; {c.precision}, LoRA r{c.finetune.lora.r}, "
               f"LR {c.learning_rate}")
    f = _flat(OmegaConf.to_container(_compose(OVERLAYS["short"], ["learning_rate=3e-3"]), resolve=False))
    diff = {k for k in set(base) | set(f) if base.get(k) != f.get(k)}
    record("Z12 control: a stray key", diff != intended, f"flagged: differs in {sorted(diff)}")


# ---------------------------------------------------------------- Z13 (Step 4.0's follow-ups)
FOLLOWUPS = {"aligned_shift99": {"seg.arm": "aligned", "seg.map_shift": 99},
             "c4375_224n": {"seg.arm": "c4375_224", "seg.map_order": "nested"},
             "c1000_224an": {"seg.arm": "c1000_224a", "seg.map_order": "nested"},
             "c1000_224ar": {"seg.arm": "c1000_224a", "seg.map_order": "raster"},
             "b256_64000": {"seg.arm": "b256_64000"},
             "cosmos16_224": {"seg.arm": "cosmos16_224"}}          # follow-up 2 (2026-11-22): fine-tuned Cosmos


def nested_offset_problems(ds: SD.SegDataset, s: dict) -> list:
    """Every map code of a nested sequence sits 198 (j + 1) positions after its input patch's token, and is that
    patch's sub-cell j of the 28 x 28 map; positions only increase; EO_MAP right after the last code."""
    out = []
    lab = (s["labels"] != -100).nonzero().flatten()
    codes_idx, eo_idx = lab[:-1], lab[-1]
    pos = s["position_ids"]
    s2 = ds.V.get_modality("S2L2A")
    bo = int((s["input_ids"] == s2.bo_id).nonzero()[0])
    inner = SD.INPUT_SIDE * SD.INPUT_SIDE
    for t, i in enumerate(codes_idx.tolist()):
        j, k = divmod(t, inner)
        if int(pos[i]) - int(pos[bo + 1 + k]) != SD.BLOCK * (j + 1):
            out.append(f"code {t}: offset {int(pos[i]) - int(pos[bo + 1 + k])} from input patch {k}, not {SD.BLOCK * (j + 1)}")
            break
        r, c = divmod(k, SD.INPUT_SIDE)
        dy, dx = divmod(j, 2)
        if int(ds.seq_order[t]) != (2 * r + dy) * SD.NESTED_SIDE + (2 * c + dx):
            out.append(f"code {t} is not patch {k}'s sub-cell {j}")
            break
    p = pos[torch.cat([codes_idx, eo_idx.view(1)])]
    if not bool((p.diff() > 0).all()):
        out.append("map positions do not increase")
    if int(pos[eo_idx]) != int(pos[codes_idx[-1]]) + 1 or int(pos[eo_idx]) > 989:
        out.append(f"EO_MAP at {int(pos[eo_idx])}")
    return out


def z13() -> None:
    from omegaconf import OmegaConf
    from eo.finetune.init import _load_state  # noqa: F401  (import check: the eval path's loader)
    # (a) the overlays
    e40 = _flat(OmegaConf.to_container(_compose("eo_seg/r0_base_e40", ["seg.arm=x", "max_length=1"]), resolve=False))
    b = _flat(OmegaConf.to_container(_compose("eo_seg/r0_base", ["seg.arm=x", "max_length=1"]), resolve=False))
    d40 = {k for k in set(b) | set(e40) if b.get(k) != e40.get(k)}
    c40 = _compose("eo_seg/r0_base_e40", ["seg.arm=aligned", "max_length=396"])
    record("Z13 r0_base_e40 = r0_base at 40 epochs", d40 == {"num_train_steps", "stop_steps", "lr_scheduler_kwargs.num_decay_steps"}
           and c40.num_train_steps == 1480 and c40.stop_steps == 1480 and c40.lr_scheduler_kwargs.num_decay_steps == 296
           and c40.lr_scheduler_kwargs.decay_type == "linear" and c40.num_warmup_steps == 37 and c40.save_steps == 37,
           f"differs from r0_base in {sorted(d40)}; 1,480 steps, decay 296 ({c40.lr_scheduler_kwargs.decay_type}), warmup 37")
    for name, keys in FOLLOWUPS.items():
        c = _compose(f"eo_seg/r0_pastis1_{name}")
        f = _flat(OmegaConf.to_container(c, resolve=False))
        diff = {k for k in set(e40) | set(f) if e40.get(k) != f.get(k)}
        want = set(keys) | {"seg.arm", "max_length"}
        ok = (diff == want and all(f[k] == v for k, v in keys.items())
              and c.max_length == SD.required_length("pastis", "p1", str(c.seg.arm)) and c.precision == "bf16")
        record(f"Z13 overlay {name}", ok, f"differs from r0_base_e40 in {sorted(diff)}; max_length {c.max_length}")
    # (b) aliases: inside their parent slots; the recorded layout unchanged
    lay = MV.LAYOUT
    ext = json.loads((MAP_BASE / "extension.json").read_text())["map_vocab"]
    al = all(lay.slot(a)[0] == lay.slots[MV.ALIASES[a][0]][0] and lay.slot(a)[1] == MV.ALIASES[a][1] for a in MV.ALIASES)
    record("Z13 aliases", al and ext == json.loads(json.dumps(lay.record())),
           f"{ {a: lay.slot(a) for a in MV.ALIASES} }; layout record = extension.json's: {ext == json.loads(json.dumps(lay.record()))}")
    try:
        lay.map_block("c4375_224", "pastis", [4375] + [0] * 783)
        record("Z13 control: a code past the alias", False, "accepted")
    except ValueError as e:
        record("Z13 control: a code past the alias", True, f"refused: {str(e)[:60]}")
    # (c) the shift: inputs untouched, the map block 99 on, nothing else
    d0 = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "val"))
    d9 = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "val", map_shift=99))
    v = int(d0.views.view.values[5])
    s0, s9 = d0.sequence(v), d9.sequence(v)
    bo = int((s0["input_ids"] == d0.task_token).nonzero()[0])
    same = all(torch.equal(s0[k], s9[k]) for k in ("input_ids", "labels", "attention_mask", "modality_ids"))
    sh = (torch.equal(s0["position_ids"][:bo], s9["position_ids"][:bo])
          and torch.equal(s9["position_ids"][bo:] - s0["position_ids"][bo:], torch.full_like(s0["position_ids"][bo:], 99)))
    sp = d9.step_positions()
    record("Z13 shift 99", same and sh and int(d9.prompt(v)[1][-1]) == 297 and sp[0] == 298 and sp[-1] == 493,
           f"ids/labels/modalities equal, positions equal before BO_MAP and +99 from it; BO_MAP at "
           f"{int(d9.prompt(v)[1][-1])}, codes at {int(sp[0])}..{int(sp[-1])}")
    reg = all(torch.equal(d0.sequence(int(w))["position_ids"], torch.arange(d0.max_length))
              for w in d0.views.view.values[:20])
    record("Z13 regression: the default layout is Step 4.0's", reg,
           "PASTIS-1 positions = 0..L-1 on 20 views (the lattice without date blocks), as Step 4.0 trained")
    # (d) the nested layout
    for arm in ("c4375_224", "c1000_224a"):
        dn = SD.SegDataset(SD.SegSpec("pastis", arm, "p1", "train", map_order="nested"), batch=32)
        dr = SD.SegDataset(SD.SegSpec("pastis", arm, "p1", "train", map_order="raster"), batch=32)
        probs, rt = [], True
        for i in range(0, len(dn), max(1, len(dn) // 5))[:5]:
            w = dn.view_at(i)
            s = dn.sequence(w)
            probs += nested_offset_problems(dn, s)
            lab = (s["labels"] != -100).nonzero().flatten()[:-1]
            seq_codes = dn.lay.decode_block(arm, s["input_ids"][lab].tolist())
            rt &= np.array_equal(dn.to_raster(np.asarray(seq_codes)), np.asarray(dn.maps[w]).astype(np.int64))
            rt &= np.array_equal(dr.to_raster(np.asarray(dr.lay.decode_block(arm, dr.sequence(w)["input_ids"][
                (dr.sequence(w)["labels"] != -100).nonzero().flatten()[:-1]].tolist()))), np.asarray(dr.maps[w]).astype(np.int64))
            probs += [p for p in sequence_problems_nested(dn, s, w)]
        spn = dn.step_positions()
        record(f"Z13 nested {arm}", not probs and rt and int(spn[0]) == 199 and int(spn[195]) == 394
               and int(spn[196]) == 397 and int(spn[-1]) == 988,
               f"5 training views: every code 198 (j + 1) after its patch's token and its sub-cell j, positions rising, "
               f"EO_MAP at 989; sequence -> raster round trip {rt} (raster layout too); codes at 199..394, 397..592, "
               f"595..790, 793..988; problems {probs[:2]}")
    # controls on one view: patch-major order and positions one off are flagged
    dn = SD.SegDataset(SD.SegSpec("pastis", "c1000_224a", "p1", "val", map_order="nested"))
    w = int(dn.views.view.values[0])
    s = dn.sequence(w)
    pm = SD.SegDataset(SD.SegSpec("pastis", "c1000_224a", "p1", "val", map_order="nested"))
    pm.seq_order = np.asarray([(2 * (k // 14) + j // 2) * 28 + (2 * (k % 14) + j % 2) for k in range(196) for j in range(4)])
    record("Z13 control: patch-major order", bool(nested_offset_problems(pm, s)), f"flagged: {nested_offset_problems(pm, s)[:1]}")
    s_off = dict(s, position_ids=s["position_ids"].clone())
    lab = (s["labels"] != -100).nonzero().flatten()
    s_off["position_ids"][lab[:-1]] += 1
    record("Z13 control: map positions one off", bool(nested_offset_problems(dn, s_off)),
           f"flagged: {nested_offset_problems(dn, s_off)[:1]}")
    try:
        SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "val", map_order="nested"))
        record("Z13 control: nested on a 14 x 14 map", False, "built")
    except ValueError as e:
        record("Z13 control: nested on a 14 x 14 map", True, f"refused: {str(e)[:70]}")


def sequence_problems_nested(ds: SD.SegDataset, s: dict, view: int) -> list:
    """Z5's structural checks that do not assume raster order: five keys, length, labels = inputs on the G codes and
    EO_MAP only, BO_MAP right before, codes in the arm's slot, the map's modality id."""
    out = []
    if set(s) != {"input_ids", "attention_mask", "labels", "position_ids", "modality_ids"}:
        out.append(f"keys {sorted(s)}")
    L = int(s["attention_mask"].sum())
    lab = (s["labels"] != -100).nonzero().flatten()
    if L != ds.required or len(lab) != ds.n_map + 1 or int(lab[-1]) != L - 1:
        out.append(f"{L} tokens / {len(lab)} labels")
    if not torch.equal(s["labels"][lab], s["input_ids"][lab]):
        out.append("labels differ from inputs")
    if int(s["input_ids"][lab[0] - 1]) != ds.task_token or int(s["input_ids"][lab[-1]]) != ds.lay.eo_map:
        out.append("BO_MAP / EO_MAP")
    body = s["input_ids"][lab[:-1]]
    if not bool(((body >= ds.map_off) & (body < ds.map_off + ds.map_size)).all()):
        out.append("a code outside the arm's slot")
    if not bool((s["modality_ids"][lab[:-1]] == MV.MAP_MODALITY_ID).all()):
        out.append("map modality id")
    return out


def z13_cache() -> None:
    """Z8 under the follow-up layouts, on the map base (fp32): the cache follows the layout's step positions."""
    from eo.eval import seg as SEG
    from eo.finetune.init import build_finetuned
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    where = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    for name in ("c4375_224n", "aligned_shift99"):
        model, cfg = build_finetuned(f"eo_seg/r0_pastis1_{name}", None, device=dev, overrides=[f"finetune.init_from={MAP_BASE}"])
        model.float().eval()
        spec = SD.spec_from_cfg(cfg, "eval")
        ds = SD.SegDataset(spec)
        pr = [ds.prompt(int(v)) for v in ds.views.view.values[:2]]
        ids = torch.stack([p[0] for p in pr]).to(dev)
        pos = torch.stack([p[1] for p in pr]).to(dev)
        sp = ds.step_positions().to(dev)
        slot = ds.lay.slot(spec.arm)
        # Z8's rule (max(4 steps, 2 x floor)) is calibrated on 6-step decodes; it holds here at 6 steps
        r = SEG.cache_check(model, ids, pos, slot, 6, step_positions=sp)
        c = SEG.cache_check(model, ids, pos, slot, 6, cache_offset=1, step_positions=sp)
        record(f"Z13 cache {name}", r["within"] and r["codes_in_slot"],
               f"{where} fp32, 6 steps: max |dlogit| {r['max_abs_logit_diff']:.2e} (tol {r['tol']:.2e}, floor "
               f"{r['floor']:.2e}); codes in slot {r['codes_in_slot']}")
        record(f"Z13 control: cache positions one off ({name})", not c["within"],
               f"max |dlogit| {c['max_abs_logit_diff']:.2e} against tol {c['tol']:.2e}")
        if spec.map_order == "nested":
            # past the first block boundary (code 196) the cached decode drifts by fp32 round-off that grows with length
            # in EVERY layout (2026-11-20: 2.2e-3 at 100 steps in both nested and raster, 9.0e-3 / 5.1e-3 at 200). So at
            # 200 steps the nested drift is read against the raster layout's own, same model and prompts, and against
            # the position control, which a wrong schedule would look like.
            dr = SD.SegDataset(SD.SegSpec("pastis", spec.arm, "p1", "val", map_order="raster"))
            prr = [dr.prompt(int(v)) for v in dr.views.view.values[:2]]
            rr = SEG.cache_check(model, torch.stack([p[0] for p in prr]).to(dev), torch.stack([p[1] for p in prr]).to(dev),
                                 slot, 200, step_positions=dr.step_positions().to(dev))
            rn = SEG.cache_check(model, ids, pos, slot, 200, step_positions=sp)
            cn = SEG.cache_check(model, ids, pos, slot, 200, cache_offset=1, step_positions=sp)
            record(f"Z13 cache {name}, 200 steps (across a block boundary)",
                   rn["max_abs_logit_diff"] <= 3 * rr["max_abs_logit_diff"] and rn["codes_in_slot"]
                   and cn["max_abs_logit_diff"] > 100 * rn["max_abs_logit_diff"],
                   f"nested drift {rn['max_abs_logit_diff']:.2e} against the raster layout's {rr['max_abs_logit_diff']:.2e} "
                   f"(<= 3x) and the one-off control's {cn['max_abs_logit_diff']:.2e} (> 100x)")
        del model


def z13_train() -> None:
    """Two steps of the nested layout through pretrain.py (CPU, fp32): the run builds the nested dataset, trains with a
    finite loss and saves. Control: the same run with the nested layout on a 14 x 14 arm (aligned) is refused when the
    dataset is built, so the option really reaches the data. (Not a row-movement test: every map row of the tied table
    gets a gradient through the softmax, and Adam's first updates are about lr x sign(grad) whatever its size.)"""
    import re
    import subprocess
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        record("Z13 train", False, f"MOR_SAVE_DIR {save!r} is not under /data")
        return
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")

    def run(tag, extra):
        rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_z13{tag}"
        out = Path(save) / "pretrain" / rel
        cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name",
               "eo_seg/r0_pastis1_c4375_224n", f"name={out.name}", f"output_dir={rel}", f"wandb_run_name={out.name}",
               "wandb=false", f"finetune.init_from={MAP_BASE}", "num_train_steps=2", "stop_steps=2", "num_warmup_steps=0",
               "lr_scheduler_kwargs.num_decay_steps=0", "save_steps=2", "eval_on_start=false", "eval_steps=1000",
               "total_batch_size=2", "per_device_train_batch_size=2", "logging_steps=1", "dataloader_num_workers=0",
               "precision=fp32", f"hydra.run.dir={out}/hydra"] + extra
        r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=3600)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pretrain.log").write_text(r.stdout + r.stderr)
        return r, out

    r, out = run("", [])
    log = r.stdout + r.stderr
    losses = [float(x) for x in re.findall(r"'loss': ([0-9.eE+-]+|nan|inf)", log) if x not in ("nan", "inf")]
    nonfinite = bool(re.search(r"'loss': (nan|inf)", log))
    ok = r.returncode == 0 and (out / "checkpoint-2").is_dir() and "map_order nested" in log and len(losses) == 2 \
        and not nonfinite
    record("Z13 train (pretrain.py, nested)", ok,
           f"exit {r.returncode}; checkpoint-2 {(out / 'checkpoint-2').is_dir()}; the dataset says map_order nested "
           f"{'map_order nested' in log}; train losses {losses}; log {out / 'pretrain.log'}")
    c, cout = run("ctl", ["seg.arm=aligned", "max_length=396"])
    clog = c.stdout + c.stderr
    record("Z13 control: nested on the aligned arm through pretrain.py", c.returncode != 0 and "nested layout" in clog,
           f"exit {c.returncode}; refused by the dataset: {'nested layout' in clog}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = {"Z4": z4, "Z5": z5, "Z8": z8, "Z9": z9, "Z10": z10, "Z12": z12, "Z13": z13, "Z13cache": z13_cache,
              "Z13train": z13_train}
    t0 = time.time()
    for k, fn in checks.items():
        if a.only and k not in a.only.split(","):
            continue
        try:
            fn()
        except Exception as e:
            traceback.print_exc()
            record(k, False, f"raised {type(e).__name__}: {e}")
    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {n_fail} fail, "
          f"{sum(r[1] == 'SKIP' for r in RESULTS)} skipped ({time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).write_text(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in RESULTS], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
