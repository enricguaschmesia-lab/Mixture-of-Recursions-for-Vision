#!/usr/bin/env python
"""Gate for A2 Step 7.0d (docs/plans/a2_segmentation.md, Step 7, "7.0d, the date curriculum in one run at B1's budget"):
the date curriculum, its schedule, its pairing with B1 and the read. Every check has a control that fails it. `.venv`;
CPU (run it with CUDA_VISIBLE_DEVICES= ).

  Z32r  7.0c's gate (eo/scripts/verify_seg_p7k.py), rerun on this tree as the regression net: 20/20.
  Z32a  the overlay eo_seg/s70d_grow2 differs from s70_p7 in exactly seg.date_curriculum (2), num_train_steps and
        stop_steps (851), lr_scheduler_kwargs.num_decay_steps (185) and plateau_stop.enable (false); and from B1's own
        run config (its finetune_config.yaml) in those keys and run-identity keys only. Control: a stray key.
  Z32b  k(e) = min(7, 1 + e // 2) for e = 0..22; every training sequence of draw epoch e (items across the epoch, every
        stage boundary) equals 7.0c's D_k(e) sequence of the same view padded to 1,605 (gated by Z31b), and from epoch 12
        B1's all-seven sequence, all five keys. Control: epoch 2's sequences against D_1.
  Z32c  the order and the drawn views equal B1's in every epoch 0..22 (all 1,184 items: 37 steps of 32, the dataset
        cutting an epoch to whole steps). Control: seed 43.
  Z32d  evaluation ignores the curriculum: spec_from_cfg(.., "eval") (eval_seg.py's path) has none, and its sequences, the
        controls' included, equal s70_p7's. Control: a train view's epoch-0 sequence differs from its all-seven one.
  Z32e  the draw epoch reaches the dataloader's workers (the config's 4): set before the iterator, every sequence of the
        first batch holds k(e) date blocks at e = 1, 2, 11, 12. Control: set after the iterator, the workers keep the
        old count.
  Z32f  the LR schedule (HF's get_scheduler with the composed config) gives B1's logged learning rate at every logged
        step. Control: num_decay_steps 148.
  Z32g  SegEpochCallback writes date_curriculum.json, k(e) at global_step 37 e for e = 0..22, which the read's schedule
        rule accepts. Controls: no file without a curriculum; one entry moved fails the rule.
  Z32h  the refusals: a curriculum on val, on PASTIS-1, beside `dates`, of 0 or 1.5 epochs. Control: the valid spec builds.
  Z32i  the read's rules: lr_problems (B1 against itself; a value moved 1e-6), band_g on known pairs. Controls inside.
  Z32p  pretrain.py on CPU, 2 steps of s70d_grow2 with 2 workers: the curriculum dataset, epoch 0 at one date (the
        callback's record and a worker's print), losses finite. Control: seg.date_curriculum=null writes no record and
        trains on seven dates (another loss).
    python eo/scripts/verify_seg_p7d.py [--only Z32a,Z32b] [--out gate.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "scripts"))
sys.path.insert(0, str(REPO / "eo" / "experiments" / "segmentation"))

from eo.data import seg_dataset as SD              # noqa: E402

RESULTS: list = []
A2SEG = Path("/data/enric/runs/pretrain/a2_seg")
B1_BASE = A2SEG / "base/b10cltrxnct_ext_map_nc_ckpt2960"
B1_RUN_DIR = A2SEG / "p7b1_s42_20261126_084434"
PARENT, OVERLAY = "eo_seg/s70_p7", "eo_seg/s70d_grow2"
B1_CLI = ["learning_rate=3e-3", "seed=42"]
WANT = {"seg.date_curriculum": 2, "num_train_steps": 851, "stop_steps": 851, "lr_scheduler_kwargs.num_decay_steps": 185,
        "plateau_stop.enable": False}
RUN_KEYS = {"finetune.init_from", "gradient_accumulation_steps", "name", "output_dir", "seg.root", "tensorboard_dir",
            "wandb_run_id", "wandb_run_name"}
OUT = Path("/data/enric/reports/a2_seg/step70d")
TMP = OUT / "gate_tmp"


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
    return {k for k in set(a) | set(b) if (k in a) != (k in b) or a.get(k) != b.get(k)}


def latest(k: int):
    return tuple(range(7 - k, 7))


def k_of(e: int) -> int:
    return min(7, 1 + e // 2)


def _train_ds(cfg_name: str, overrides=(), seed: int = 42, **spec_kw):
    cfg = _compose(cfg_name, list(B1_CLI) + list(overrides))
    spec = SD.spec_from_cfg(cfg, "train")
    for k, v in spec_kw.items():
        setattr(spec, k, v)
    spec.__post_init__()
    return SD.SegDataset(spec, max_length=int(cfg.max_length), seed=seed, batch=int(cfg.total_batch_size)), cfg


def _dates_ds(k: int):
    """7.0c's D_k training dataset (B1's composed config with seg.dates), padded to 1,605 like the curriculum's."""
    cfg = _compose(PARENT, B1_CLI)
    spec = SD.spec_from_cfg(cfg, "train")
    spec.dates = SD.normalise_dates(latest(k))
    return SD.SegDataset(spec, max_length=1605, seed=42, batch=int(cfg.total_batch_size))


def _same(s, t) -> bool:
    return all(torch.equal(s[x], t[x]) for x in ("input_ids", "labels", "position_ids", "modality_ids", "attention_mask"))


# ------------------------------------------------------------------ Z32r
def z32r() -> None:
    out = TMP / "z32r_gate_p7k.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    t0 = time.time()
    r = subprocess.run([sys.executable, "eo/scripts/verify_seg_p7k.py", "--out", str(out)], cwd=str(REPO), env=env,
                       capture_output=True, text=True, timeout=7200)
    (TMP / "z32r_gate_p7k.log").write_text(r.stdout + r.stderr)
    res = json.loads(out.read_text()) if out.exists() else []
    n_pass = sum(x["status"] == "PASS" for x in res)
    record("Z32r 7.0c's gate on this tree", r.returncode == 0 and len(res) == 20 and n_pass == 20,
           f"exit {r.returncode}; {n_pass}/{len(res)} pass; {time.time() - t0:.0f} s; {out}")


# ------------------------------------------------------------------ Z32a
def z32a() -> None:
    import yaml
    from omegaconf import OmegaConf
    a = _flat(OmegaConf.to_container(_compose(PARENT, B1_CLI), resolve=False))
    g = _flat(OmegaConf.to_container(_compose(OVERLAY, B1_CLI), resolve=False))
    d = _diff(a, g)
    vals = {k: g.get(k) for k in WANT}
    b1 = _flat(yaml.safe_load((B1_RUN_DIR / "finetune_config.yaml").read_text()))
    d1 = _diff(b1, g) - RUN_KEYS
    ok = d == set(WANT) and vals == WANT and d1 == set(WANT)
    record("Z32a the overlay = s70_p7 (and B1's run config) but the curriculum and B1's realized schedule", ok,
           f"vs s70_p7 {sorted(d)} = {vals}; vs B1's finetune_config.yaml, run-identity keys aside: {sorted(d1)}")
    c = _flat(OmegaConf.to_container(_compose(OVERLAY, B1_CLI + ["weight_decay=0.01"]), resolve=False))
    d2 = _diff(a, c)
    record("Z32a control: a stray key", d2 != set(WANT), f"differs in {sorted(d2)}")


# ------------------------------------------------------------------ Z32b
EPOCHS_B = list(range(14)) + [22]
ITEMS = (0, 1, 7, 31, 600, 1183)                  # 1,183: an epoch's last item (37 x 32 = 1,184)


def z32b() -> None:
    G, _ = _train_ds(OVERLAY)
    full, _ = _train_ds(PARENT)
    sched = [len(G.groups_at(e)) for e in range(23)]
    want = [k_of(e) for e in range(23)]
    groups_ok = all(G.groups_at(e) == latest(k_of(e)) for e in range(23))
    dk = {k: _dates_ds(k) for k in range(1, 7)}
    bad, n = [], 0
    for e in EPOCHS_B:
        k = k_of(e)
        ref = full if k == 7 else dk[k]
        G.set_draw_epoch(e)
        ref.set_draw_epoch(e)
        for i in ITEMS:
            if G.view_at(i) != ref.view_at(i) or not _same(G[i], ref[i]):
                bad.append((e, i))
            n += 1
    record("Z32b the schedule and every sequence = 7.0c's D_k(e) (B1's from epoch 12)",
           sched == want and groups_ok and not bad and G.required == 1605 and G.max_length == 1605,
           f"k(e) {sched}; {n} sequences over epochs {EPOCHS_B[0]}..{EPOCHS_B[-1]}; mismatches {bad[:5]}")
    G.set_draw_epoch(2)
    dk[1].set_draw_epoch(2)
    differ = all(not _same(G[i], dk[1][i]) for i in ITEMS)
    record("Z32b control: epoch 2 against D_1", differ, f"every epoch-2 sequence differs from its one-date one: {differ}")


# ------------------------------------------------------------------ Z32c
def z32c() -> None:
    G, _ = _train_ds(OVERLAY)
    full, _ = _train_ds(PARENT)
    bad = []
    for e in range(23):
        if not np.array_equal(G.order(e), full.order(e)):
            bad.append((e, "order"))
        if [G.draw_view(int(s), e) for s in G.order(e)] != [full.draw_view(int(s), e) for s in full.order(e)]:
            bad.append((e, "views"))
    record("Z32c the order and the views = B1's in every epoch", not bad and len(G) == len(full) == 1184
           and G.steps_per_epoch() == 37, f"23 epochs x {len(G)} items, {G.steps_per_epoch()} steps an epoch; {bad[:4]}")
    other, _ = _train_ds(PARENT, seed=43)
    differs = not np.array_equal(other.order(0), full.order(0))
    record("Z32c control: seed 43", differs, f"seed 43's epoch-0 order differs: {differs}")


# ------------------------------------------------------------------ Z32d
def z32d() -> None:
    cg, cp = _compose(OVERLAY, B1_CLI), _compose(PARENT, B1_CLI)
    sg, sp = SD.spec_from_cfg(cg, "eval"), SD.spec_from_cfg(cp, "eval")
    bad = []
    if sg.date_curriculum is not None or sg.dates is not None or sg.split != "val":
        bad.append(("spec", sg.date_curriculum, sg.dates, sg.split))
    for var in ("full", "prior", "wrong", "one_date", "shuffled_dates", "wrong_early"):
        a, b = SD.spec_from_cfg(cg, "eval"), SD.spec_from_cfg(cp, "eval")
        a.variant = b.variant = var                       # eval_seg.py's generate path
        da, db = SD.SegDataset(a, max_length=None, seed=42), SD.SegDataset(b, max_length=None, seed=42)
        for v in [int(x) for x in da.views.view.values[[0, 100, 481]]]:
            if not _same(da.sequence(v), db.sequence(v)) or not torch.equal(da.prompt(v)[0], db.prompt(v)[0]):
                bad.append((var, v))
    record("Z32d evaluation ignores the curriculum (eval_seg.py's spec)", not bad,
           f"eval spec: date_curriculum {sg.date_curriculum}, dates {sg.dates}, split {sg.split}; 6 variants x 3 views "
           f"equal s70_p7's; {bad[:4]}")
    G, _ = _train_ds(OVERLAY)
    full, _ = _train_ds(PARENT)
    G.set_draw_epoch(0)
    full.set_draw_epoch(0)
    differs = not _same(G[5], full[5])
    record("Z32d control: a train view's epoch-0 sequence against its all-seven one", differs, f"differs: {differs}")


# ------------------------------------------------------------------ Z32e
def _date_blocks(batch, bo_date: int) -> list:
    return [int((row == bo_date).sum()) for row in batch["input_ids"]]


def z32e() -> None:
    from torch.utils.data import DataLoader, SequentialSampler
    G, cfg = _train_ds(OVERLAY)
    nw = int(cfg.dataloader_num_workers)
    bo_date = int(G._date_chunk(int(G.in_dates[0][0]))[0])
    dl = DataLoader(G, batch_size=int(cfg.per_device_train_batch_size), sampler=SequentialSampler(G), num_workers=nw)
    got, ok = {}, nw >= 2
    for e in (1, 2, 11, 12):
        G.set_draw_epoch(e)
        it = iter(dl)
        b = next(it)
        got[e] = _date_blocks(b, bo_date)
        ok &= got[e] == [k_of(e)] * len(got[e])
        del it
    record("Z32e the draw epoch reaches the workers", ok, f"{nw} workers; date blocks per sequence of the first batch: {got}")
    G.set_draw_epoch(1)
    it = iter(dl)
    G.set_draw_epoch(2)
    late = _date_blocks(next(it), bo_date)
    del it
    record("Z32e control: set after the iterator", late != [k_of(2)] * len(late),
           f"the workers kept epoch 1's count: {late} (epoch 2 wants {k_of(2)})")


# ------------------------------------------------------------------ Z32f
def _lrs(cfg, n: int) -> dict:
    """{step s: the LR update s used}, HF's scheduler built as Trainer.create_scheduler builds it."""
    from transformers import get_scheduler
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=float(cfg.learning_rate))
    sch = get_scheduler(str(cfg.lr_scheduler_type), optimizer=opt, num_warmup_steps=int(cfg.num_warmup_steps),
                        num_training_steps=int(cfg.num_train_steps),
                        scheduler_specific_kwargs=dict(cfg.lr_scheduler_kwargs))
    out = {}
    for s in range(1, n + 1):
        out[s] = sch.get_last_lr()[0]                  # HF logs the LR of the update it just made
        opt.step()
        sch.step()
    return out


def _hist(lrs: dict, steps) -> list:
    return [{"step": s, "loss": 0.0, "learning_rate": lrs[s]} for s in steps]


def z32f() -> None:
    import step70d_read as D
    b1 = json.loads((B1_RUN_DIR / "trainer_state.json").read_text())["log_history"]
    steps = [int(h["step"]) for h in b1 if "learning_rate" in h and "loss" in h]
    cfg = _compose(OVERLAY, B1_CLI)
    probs = D.lr_problems(_hist(_lrs(cfg, 851), steps), b1)
    record("Z32f the LR schedule = B1's logged one", not probs and len(steps) == 85,
           f"{len(steps)} logged steps ({steps[0]}..{steps[-1]}); {probs or 'equal within 1e-9'}")
    bad_cfg = _compose(OVERLAY, B1_CLI + ["lr_scheduler_kwargs.num_decay_steps=148"])
    probs2 = D.lr_problems(_hist(_lrs(bad_cfg, 851), steps), b1)
    record("Z32f control: num_decay_steps 148", bool(probs2), f"{probs2}")


# ------------------------------------------------------------------ Z32g
def _callback_record(ds, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    f = d / SD.CURRICULUM_RECORD
    if f.exists():
        f.unlink()
    cb = SD.SegEpochCallback(ds, 37, verbose=False)
    args = SimpleNamespace(output_dir=str(d), dataloader_persistent_workers=False, gradient_accumulation_steps=4)
    for e in range(23):
        cb.on_epoch_begin(args, SimpleNamespace(global_step=37 * e, is_world_process_zero=True), None)
    return json.loads(f.read_text()) if f.exists() else None


def z32g() -> None:
    import copy
    import step70d_read as D
    G, _ = _train_ds(OVERLAY)
    rec = _callback_record(G, TMP / "z32g_curriculum")
    probs = D.schedule_problems(rec)
    record("Z32g the callback's date record", rec is not None and not probs and len(rec["epochs"]) == 23,
           f"{len(rec['epochs']) if rec else 0} entries, k {[x['k'] for x in rec['epochs']] if rec else None}; {probs}")
    full, _ = _train_ds(PARENT)
    none = _callback_record(full, TMP / "z32g_plain")
    record("Z32g control: no curriculum, no record", none is None, f"record written: {none is not None}")
    moved = copy.deepcopy(rec)
    moved["epochs"][12]["k"], moved["epochs"][12]["groups"] = 6, list(range(1, 7))
    p2 = D.schedule_problems(moved)
    record("Z32g control: epoch 12 at six dates", bool(p2), f"{p2}")


# ------------------------------------------------------------------ Z32h
def z32h() -> None:
    base = dict(task="pastis", arm="aligned", frames="p7", split="train", in_pinned=True, date_curriculum=2)
    cases = {"val": dict(split="val", in_pinned=None), "PASTIS-1": dict(frames="p1"), "beside dates": dict(dates=(5, 6)),
             "0 epochs": dict(date_curriculum=0), "1.5 epochs": dict(date_curriculum=1.5)}
    refused = {}
    for name, kw in cases.items():
        try:
            SD.SegDataset(SD.SegSpec(**dict(base, **kw)), max_length=1605, batch=32)
            refused[name] = False
        except (ValueError, TypeError):
            refused[name] = True
    record("Z32h the refusals", all(refused.values()), f"{refused}")
    try:
        ok = len(SD.SegDataset(SD.SegSpec(**base), max_length=1605, batch=32)) == 1184
    except Exception as e:                                       # noqa: BLE001
        ok = False
        print(e)
    record("Z32h control: the valid spec builds", ok, f"built, 1,184 items an epoch: {ok}")


# ------------------------------------------------------------------ Z32i
def z32i() -> None:
    import step70d_read as D
    b1 = json.loads((B1_RUN_DIR / "trainer_state.json").read_text())["log_history"]
    same = D.lr_problems(b1, b1)
    moved = json.loads(json.dumps(b1))
    i = next(j for j, h in enumerate(moved) if "learning_rate" in h and "loss" in h and h["step"] == 700)
    moved[i]["learning_rate"] *= 1 + 1e-6
    diff = D.lr_problems(moved, b1)
    G = lambda d, lo, hi: {"delta": d, "lo95": lo, "hi95": hi}       # noqa: E731
    bands = [D.band_g(G(1.5, 0.2, 2.8))[:3], D.band_g(G(1.5, -0.1, 2.8))[:3], D.band_g(G(-1.3, -2.5, -0.1))[:3],
             D.band_g(G(0.3, -0.8, 1.4))[:3]]
    ks = [D.k_of(e) for e in (0, 1, 2, 11, 12, 22)]
    ok = not same and bool(diff) and bands == ["(a)", "(b)", "(c)", "(b)"] and ks == [1, 1, 2, 6, 7, 7]
    record("Z32i the read's rules", ok, f"B1 against itself {same}; a value moved 1e-6: {bool(diff)}; bands {bands}; "
           f"k(0, 1, 2, 11, 12, 22) {ks}")
    record("Z32i control: an interval reaching 0 is not a gain", bands[1] != "(a)", f"band {bands[1]}")


# ------------------------------------------------------------------ Z32p
def _run(tag: str, extra: dict) -> dict:
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {save!r} is not under /data")
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_z32p_{tag}"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    ov = {"name": out.name, "output_dir": rel, "wandb_run_name": out.name, "wandb": "false",
          "finetune.init_from": str(B1_BASE), "num_train_steps": "2", "stop_steps": "2", "num_warmup_steps": "0",
          "lr_scheduler_kwargs.num_decay_steps": "0", "save_steps": "2", "eval_on_start": "false", "eval_steps": "1000",
          "total_batch_size": "2", "per_device_train_batch_size": "1", "logging_steps": "1",
          "dataloader_num_workers": "2", "precision": "fp32", "learning_rate": "3e-3", "seed": "42",
          "hydra.run.dir": f"{out}/hydra"}                     # Z31p's overrides, with 2 workers
    ov.update(extra)
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name", OVERLAY] + \
        [f"{k}={v}" for k, v in ov.items()]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    out.mkdir(parents=True, exist_ok=True)
    log = r.stdout + r.stderr
    (out / "pretrain.log").write_text(log)
    st = out / "trainer_state.json"
    hist = json.loads(st.read_text())["log_history"] if st.exists() else []
    rec = out / SD.CURRICULUM_RECORD
    return {"rc": r.returncode, "out": out, "seconds": time.time() - t0, "log": log,
            "losses": [h["loss"] for h in hist if "loss" in h],
            "line": next((ln for ln in log.splitlines() if ln.startswith("seg train:")), ""),
            "record": json.loads(rec.read_text()) if rec.exists() else None}


def z32p() -> None:
    A = _run("G", {})
    rec = A["record"] or {}
    ep = rec.get("epochs") or []
    worker = [ln for ln in A["log"].splitlines() if ln.startswith("SegDataset date curriculum: draw epoch 0, item 0 "
                                                                   "holds the latest 1 date(s) [6]")]
    ok = (A["rc"] == 0 and "date_curriculum 2" in A["line"] and len(A["losses"]) == 2
          and all(math.isfinite(x) for x in A["losses"]) and rec.get("date_curriculum") == 2
          and [(x["global_step"], x["draw_epoch"], x["k"]) for x in ep] == [(0, 0, 1)] and len(worker) >= 1)
    record("Z32p s70d_grow2 through pretrain.py on CPU", ok,
           f"exit {A['rc']}; losses {A['losses']}; record {ep}; worker prints {len(worker)}; {A['seconds']:.0f} s; "
           f"'{A['line'][-90:]}'; log {A['out'] / 'pretrain.log'}")
    C = _run("plain", {"seg.date_curriculum": "null"})
    okc = (C["rc"] == 0 and C["record"] is None and "date_curriculum" not in C["line"] and len(C["losses"]) == 2
           and C["losses"][0] != (A["losses"][0] if A["losses"] else None))
    record("Z32p control: seg.date_curriculum=null", okc,
           f"exit {C['rc']}; record {C['record'] is not None}; losses {C['losses']} against {A['losses']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = {"Z32r": z32r, "Z32a": z32a, "Z32b": z32b, "Z32c": z32c, "Z32d": z32d, "Z32e": z32e, "Z32f": z32f,
              "Z32g": z32g, "Z32h": z32h, "Z32i": z32i, "Z32p": z32p}
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
