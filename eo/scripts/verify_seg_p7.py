#!/usr/bin/env python
"""A2 Step 7.0's gate (docs/plans/a2_segmentation.md, Step 7, "7.0, PASTIS-7 on A2's current base"; 2026-11-25): the
overlay eo_seg/s70_p7 and the paths its run and its read use. Every check ships with a control that must fail it. A file
of its own, so Step 7.0 adds no lines to the gates other A2 work edits. Runs in `.venv` on CPU.

    CUDA_VISIBLE_DEVICES= python eo/scripts/verify_seg_p7.py [--only Z29a,Z29b,Z29c,Z29p] [--out results.json]

The PASTIS-7 layout itself is gated elsewhere and is not repeated here: Z5 (sequences of every arm at 1,605 tokens, image
blocks at 198 j, date blocks, the dates from metadata, shuffled_dates), Z8 (the cache on a PASTIS-7 prompt) and Z9 (the
wrong input) in verify_segmentation.py and verify_seg_model.py.

  Z29a  s70_p7 differs from r0_s41_lora16 in exactly seg.frames and max_length; max_length = required_length(pastis,
        p7, aligned) = 1,605; the arm is aligned
                                     | s70_p7 with a stray key (weight_decay): other keys differ
  Z29b  The run's datasets through the overlay: the training epoch is PASTIS-1's (the same pinned patches, 37 steps of
        32), val is the 482 fold-4 views, sequences of train and val pass verify_seg_model's sequence checker (1,605
        tokens, targets last, the view's map and input rows, seven date blocks, image blocks at 198 j, BO_MAP at 1,386),
        and every val view's seven dates are in date order
                                     | a PASTIS-1 sequence of the same view, checked as PASTIS-7: flagged
  Z29c  The read's date controls on fold-4 prompts: one_date is [BO_Date d EO_Date][the fourth date's S2 block][BO_MAP]
        with BO_MAP at 198, its date block the full prompt's fourth; shuffled_dates keeps the full prompt's positions and
        its seven S2 blocks token for token, and holds the same date blocks in another order; the full prompt is 1,408
        tokens ending at position 1,386
                                     | one_date's S2 block compared with the first date's row: unequal
  Z29p  s70_p7 through pretrain.py, 2 steps on CPU at fp32 from the map base: the dataset reports PASTIS-7 at 1,605
        tokens, finite losses, a save
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "scripts"))

from eo.data import seg_dataset as SD          # noqa: E402

RESULTS: list = []
MAP_BASE = Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000")
OVERLAY, PARENT = "eo_seg/s70_p7", "eo_seg/r0_s41_lora16"
KEYS = {"seg.frames", "max_length"}
P7_LEN, P7_PROMPT, P7_BO_MAP = 1605, 1408, 1386


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


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


def _diff(a, b) -> set:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


_DS: dict = {}


def ds(name: str, split: str):
    key = (name, split)
    if key not in _DS:
        _DS[key] = SD.build(_compose(name), split)
    return _DS[key]


# ---------------------------------------------------------------- Z29a
def z29a() -> None:
    from omegaconf import OmegaConf
    a = _flat(OmegaConf.to_container(_compose(PARENT), resolve=False))
    b = _flat(OmegaConf.to_container(_compose(OVERLAY), resolve=False))
    d = _diff(a, b)
    req = SD.required_length("pastis", "p7", "aligned")
    ok = (d == KEYS and int(b["max_length"]) == req == P7_LEN and b["seg.frames"] == "p7" and b["seg.arm"] == "aligned"
          and b["seg.task"] == "pastis")
    record("Z29a s70_p7 = r0_s41_lora16 but frames and length", ok,
           f"differs in {sorted(d)}; max_length {b['max_length']}, required_length {req}; task {b['seg.task']}, arm "
           f"{b['seg.arm']}, in_pinned {b['seg.in_pinned']}; LR {b['learning_rate']} in the file (3e-3 on the command line)")
    c = _flat(OmegaConf.to_container(_compose(OVERLAY, ["weight_decay=0.01"]), resolve=False))
    d2 = _diff(a, c)
    record("Z29a control: a stray key", d2 != KEYS, f"differs in {sorted(d2)}")


# ---------------------------------------------------------------- Z29b
def z29b() -> None:
    import verify_seg_model as VSM
    cfg = _compose(OVERLAY)
    tr, va, t1 = ds(OVERLAY, "train"), ds(OVERLAY, "val"), ds(PARENT, "train")
    tb = int(cfg.total_batch_size)
    spe = len(tr) // tb
    ok_len = len(tr) == len(t1) and len(tr) % tb == 0 and spe == 37 and len(va) == 482
    probs, n = [], 0
    for d in (tr, va):
        for i in list(range(0, len(d), max(1, len(d) // 5)))[:5]:
            v = d.view_at(i)
            s = d[i]
            n += 1
            p = VSM.sequence_problems(d, s, v)
            if int(s["position_ids"][P7_LEN - d.n_map - 2]) != P7_BO_MAP:
                p.append(f"BO_MAP at {int(s['position_ids'][P7_LEN - d.n_map - 2])}, not {P7_BO_MAP}")
            probs += p
    vv = va.views.view.values
    chrono = int(sum(bool((np.diff(va.in_dates[int(v)]) > 0).all()) for v in vv))
    record("Z29b the run's datasets", ok_len and not probs and chrono == len(vv),
           f"train {len(tr)} examples an epoch = {spe} steps of {tb} (PASTIS-1's: {len(t1)}); val {len(va)} views; "
           f"{n} sequences, problems {probs[:3]}; dates in order on {chrono} of {len(vv)} val views")
    # control: the same view's PASTIS-1 sequence read as PASTIS-7
    p1 = ds(PARENT, "val")
    v = int(vv[7])
    i1 = int(np.nonzero(p1.views.view.values == v)[0][0])
    bad = VSM.sequence_problems(va, p1[i1], v)
    record("Z29b control: a PASTIS-1 sequence checked as PASTIS-7", bool(bad), f"flagged: {bad[:2]}")


# ---------------------------------------------------------------- Z29c
def z29c() -> None:
    va = ds(OVERLAY, "val")
    V = va.V
    s2 = V.get_modality("S2L2A")
    g = SD.ONE_DATE_GROUP
    vv = [int(v) for v in va.views.view.values[:: max(1, len(va.views) // 6)][:6]]
    probs = []
    ctrl = 0
    for v in vv:
        idf, pf = va.prompt(v, "full")
        id1, p1 = va.prompt(v, "one_date")
        ids, ps = va.prompt(v, "shuffled_dates")
        if len(idf) != P7_PROMPT or int(pf[-1]) != P7_BO_MAP or int(idf[-1]) != va.task_token:
            probs.append(f"view {v}: full prompt {len(idf)} tokens ending at {int(pf[-1])}")
        bof = (idf == s2.bo_id).nonzero().flatten()
        dbf = (idf == V.BO_DATE).nonzero().flatten()
        # one_date
        bo1 = (id1 == s2.bo_id).nonzero().flatten()
        db1 = (id1 == V.BO_DATE).nonzero().flatten()
        if len(id1) != 202 or len(bo1) != 1 or len(db1) != 1 or int(p1[bo1[0]]) != 0 or int(p1[-1]) != 198 \
                or int(id1[-1]) != va.task_token:
            probs.append(f"view {v}: one_date prompt {len(id1)} tokens, BO_MAP at {int(p1[-1])}")
            continue
        body = (id1[bo1[0] + 1: bo1[0] + 197] - s2.codebook_offset).numpy()
        if not np.array_equal(body, np.asarray(va.s2[va.in_rows[v, g]]).astype(np.int64)):
            probs.append(f"view {v}: one_date's S2 is not date {g}'s row")
        if not torch.equal(id1[db1[0]: db1[0] + 3], idf[dbf[g]: dbf[g] + 3]):
            probs.append(f"view {v}: one_date's date block is not the full prompt's date {g}")
        ctrl += not np.array_equal(body, np.asarray(va.s2[va.in_rows[v, 0]]).astype(np.int64))
        # shuffled_dates
        if len(ids) != len(idf) or not torch.equal(ps, pf):
            probs.append(f"view {v}: shuffled_dates moves positions")
            continue
        bos = (ids == s2.bo_id).nonzero().flatten()
        if not torch.equal(bos, bof) or not all(torch.equal(ids[b: b + 198], idf[b: b + 198]) for b in bof):
            probs.append(f"view {v}: shuffled_dates changes an S2 block")
        dbs = (ids == V.BO_DATE).nonzero().flatten()
        Ds = [tuple(ids[b: b + 3].tolist()) for b in dbs]
        Df = [tuple(idf[b: b + 3].tolist()) for b in dbf]
        if sorted(Ds) != sorted(Df) or Ds == Df:
            probs.append(f"view {v}: shuffled_dates' date blocks are not the full prompt's in another order")
    record("Z29c the date controls' prompts", not probs,
           f"{len(vv)} fold-4 views: full {P7_PROMPT} tokens to BO_MAP at {P7_BO_MAP}; one_date 202 tokens, date {g}'s "
           f"block and image, BO_MAP at 198; shuffled_dates: same positions and S2 blocks, dates reordered; problems "
           f"{probs[:3]}")
    record("Z29c control: one_date's S2 against the first date's row", ctrl == len(vv),
           f"unequal on {ctrl} of {len(vv)} views")


# ---------------------------------------------------------------- Z29p
def z29p() -> None:
    import subprocess
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {save!r} is not under /data")
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_z29p"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    ov = {"name": out.name, "output_dir": rel, "wandb_run_name": out.name, "wandb": "false",
          "finetune.init_from": str(MAP_BASE), "num_train_steps": "2", "stop_steps": "2", "num_warmup_steps": "0",
          "lr_scheduler_kwargs.num_decay_steps": "0", "save_steps": "2", "eval_on_start": "false", "eval_steps": "1000",
          "total_batch_size": "2", "per_device_train_batch_size": "1", "logging_steps": "1",
          "dataloader_num_workers": "0", "precision": "fp32", "plateau_stop.enable": "false", "learning_rate": "3e-3",
          "seed": "42", "hydra.run.dir": f"{out}/hydra"}
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name", OVERLAY] + \
        [f"{k}={v}" for k, v in ov.items()]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    out.mkdir(parents=True, exist_ok=True)
    log = r.stdout + r.stderr
    (out / "pretrain.log").write_text(log)
    st = out / "trainer_state.json"
    hist = json.loads(st.read_text())["log_history"] if st.exists() else []
    losses = [h["loss"] for h in hist if "loss" in h]
    line = next((ln for ln in log.splitlines() if ln.startswith("seg train:")), "")
    ok = (r.returncode == 0 and (out / "checkpoint-2").is_dir() and "pastis p7 arm aligned" in line
          and f"{P7_LEN} tokens" in line and len(losses) == 2 and all(math.isfinite(x) for x in losses))
    record("Z29p s70_p7 through pretrain.py", ok, f"exit {r.returncode}; losses {losses}; {time.time() - t0:.0f} s, CPU "
           f"fp32; '{line[-120:]}'; log {out / 'pretrain.log'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = {"Z29a": z29a, "Z29b": z29b, "Z29c": z29c, "Z29p": z29p}
    want = a.only.split(",") if a.only else list(checks)
    t0 = time.time()
    for k in want:
        try:
            checks[k]()
        except Exception as e:
            traceback.print_exc()
            record(k, False, f"raised {type(e).__name__}: {e}")
    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {n_fail} fail ({time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in RESULTS], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
