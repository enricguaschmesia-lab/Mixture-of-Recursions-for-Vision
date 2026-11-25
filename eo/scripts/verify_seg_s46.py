#!/usr/bin/env python
"""A2 Step 4.6's gate (docs/plans/a2_segmentation.md, Step 4, "4.6, the forgetting-bounded recipe (item 90)", item 7;
2026-11-24). Every check ships with a control that must fail it. A file of its own, so Step 4.6 adds no lines to the
gates other A2 work edits. Runs in `.venv` on CPU.

    python eo/scripts/verify_seg_s46.py [--only Z28a,Z28b,Z28c,Z28d,Z28e,Z28f,Z28p] [--out results.json]

The replay set's base rows come from the TerraMesh train root, which lives on g49 only. Here a stand-in root serves them:
val_clean's files behind a build.json that calls itself a train root (S46_SCRATCH/fake_train_root). The rows differ;
the code path is the run's.

  Z28a  A step's segmentation micro-batches are exactly R0's batch at that step and draw epoch (all five keys), for
        PASTIS-1 (steps 0, 5, 36 at draw epochs 0 and 3) and CloudSen12 (steps 0 and 124 at epoch 2)
                                     | the next step's R0 batch: it differs
  Z28b  A replay sequence: 990 tokens, five blocks of 198, each block (BO, body, EO) token-identical to
        TerraMeshTokenDataset's own chunk of that row and modality, every image modality of a Major TOM row once, labels
        on every token, positions 0..989, modality ids the blocks' own; its rows are Major TOM rows with all five blocks;
        the draw is a pure function of (seed, epoch, step, micro-batch)
                                     | two blocks of a sequence swapped: the identity fails
  Z28c  The base stream's share of a step's labelled tokens: RP-1/2 5,940 / 12,244 = 0.4851, RP-1/4 1,980 / 8,284 =
        0.2390, on every step read (0, 17, 36)
                                     | replay.base_microbatches 0: share 0
  Z28d  HF's epoch is the segmentation epoch: len / total_batch_size = R0's steps per epoch (PASTIS-1 37, CloudSen12
        125), and SegEpochCallback's own check accepts the loader
                                     | gradient accumulation one short of total_batch_size: the callback refuses
  Z28e  Each overlay differs from its parent (r0_s41_lora16, s52_cloudsen12) in exactly the replay block,
        total_batch_size and per_device_train_batch_size, and from the map base trains R0's 51,014,208 parameters
                                     | r0_s41_lora16 against s52_cloudsen12: other keys differ
  Z28f  The forgetting read's path (step44_map_mass.py --decompose --per-row, the screen's outputs): the map base
        reproduces 4.4's 5.067649 to 1e-5 and R0 PASTIS s42's kept save 4.4's 6.590123 to 1e-4; the per-row sums
        reproduce each file's aggregate to 1e-7 (the aggregate adds fp32 batch sums; the per-row sums are fp64)
                                     | the base over its first 128 rows only: moves by more than 1e-5
  Z28g  (tools/check_replay_root.py, on g49 beside the train root) the replay subset root's rows are the train root's:
        2,000 rows, every modality's presence and tokens equal the source row's; the draw reproduces the build record
                                     | the source row shifted by one: unequal
        With S46_BASE_ROOT=/data/enric/data/TerraMesh/replay_root_s46, Z28a-d and Z28p run on that subset root itself.
  Z28p  RP-1/2 through pretrain.py, 2 steps on CPU at fp32 on the stand-in root: the dataset reports the replay share,
        finite losses, a save
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS: list = []
MAP_BASE = Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000")
VAL_CLEAN = Path("/data/enric/data/TerraMesh/val_clean")
SCRATCH = Path(os.environ.get("S46_SCRATCH", "/data/enric/reports/a2_seg/step46/gate_root"))
F_DIR = Path("/data/enric/reports/a2_seg/step46/F")
OVERLAYS = {"s46_rp50_pastis": "r0_s41_lora16", "s46_rp25_pastis": "r0_s41_lora16",
            "s46_rp50_cloudsen12": "s52_cloudsen12"}
REPLAY_KEYS = {"replay.enable", "replay.seg_per_microbatch", "replay.base_microbatches", "replay.base_per_microbatch",
               "total_batch_size", "per_device_train_batch_size"}


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


def fake_train_root() -> Path:
    """val_clean's files behind a build.json that passes resolve_train_root (the gate's stand-in); or, with S46_BASE_ROOT
    set, that root (the replay subset root copied from g49, /data/enric/data/TerraMesh/replay_root_s46)."""
    if os.environ.get("S46_BASE_ROOT"):
        return Path(os.environ["S46_BASE_ROOT"])
    root = SCRATCH / "fake_train_root"
    if not (root / "build.json").exists():
        root.mkdir(parents=True, exist_ok=True)
        for p in VAL_CLEAN.iterdir():
            if not (root / p.name).exists():
                (root / p.name).symlink_to(p)
        (root / "build.json").write_text(json.dumps({"split": "train", "offset_free": True, "partial": False,
                                                     "note": "A2 Step 4.6 gate stand-in: val_clean's files"}))
    return root


_DS: dict = {}


def replay_ds(name: str, overrides=()):
    key = (name, tuple(overrides))
    if key not in _DS:
        from eo.data import seg_replay_dataset as SR
        _DS[key] = SR.build_train(_compose(f"eo_seg/{name}", overrides), base_root=fake_train_root())
    return _DS[key]


def r0_ds(parent: str):
    key = ("r0", parent)
    if key not in _DS:
        from eo.data import seg_dataset as SD
        _DS[key] = SD.build(_compose(f"eo_seg/{parent}"), "train")
    return _DS[key]


def z28a() -> None:
    import torch
    for name, cases in (("s46_rp50_pastis", ((0, 0), (5, 0), (36, 3))), ("s46_rp50_cloudsen12", ((0, 2), (124, 2)))):
        ds, r0 = replay_ds(name), r0_ds(OVERLAYS[name])
        bad, ctl = [], []
        for s, e in cases:
            ds.set_draw_epoch(e)
            r0.set_draw_epoch(e)
            mine = {k: torch.cat([ds[s * ds.M + j][k] for j in range(ds.S)]) for k in ("input_ids", "attention_mask",
                                                                                     "labels", "position_ids",
                                                                                     "modality_ids")}
            for nxt, store in ((s, bad), ((s + 1) % ds.steps, ctl)):
                ref = [r0[nxt * r0.batch + k] for k in range(r0.batch)]
                same = all(torch.equal(mine[k], torch.stack([x[k] for x in ref])) for k in mine)
                store.append(same)
        record(f"Z28a {name}: the segmentation micro-batches are R0's batch", all(bad) and len(bad) == len(cases),
               f"steps/epochs {cases}: equal {bad}; R0 batch {r0.batch}, {ds.S} micro-batches of "
               f"{ds.rc.seg_per_microbatch}")
        record(f"Z28a control {name}: the next step's batch differs", not any(ctl), f"equal {ctl}")


def z28b() -> None:
    import torch
    from eo.data import map_mix_dataset as MM
    ds = replay_ds("s46_rp50_pastis")
    base = ds.scenes.base
    V = base._V
    bo_to_mod = {V.get_modality(m).bo_id: m for m in base.active_modalities}
    probs, ctl_fires, n = [], [], 0
    for s, b, e in ((0, 0, 0), (0, 1, 0), (7, 1, 4), (36, 0, 9)):
        rows = ds.base_rows_at(s, b, e)
        again = ds.base_rows_at(s, b, e)
        other = ds.base_rows_at(s, b, e + 1)
        if not np.array_equal(rows, again) or np.array_equal(rows, other):
            probs.append(f"draw ({s},{b},{e}) not a pure function of its key, or epoch-independent")
        ds.set_draw_epoch(e)
        mb = ds[s * ds.M + ds.S + b]
        if tuple(mb["input_ids"].shape) != (ds.rc.base_per_microbatch, MM.L_BASE):
            probs.append(f"micro-batch shape {tuple(mb['input_ids'].shape)}")
        for r, row in enumerate(rows):
            n += 1
            ids = mb["input_ids"][r]
            if not torch.equal(mb["labels"][r], ids) or not torch.equal(mb["position_ids"][r], torch.arange(MM.L_BASE)) \
                    or int(mb["attention_mask"][r].sum()) != MM.L_BASE:
                probs.append(f"row {row}: labels, positions or mask")
            if len(base.present_modalities(int(row))) != MM.N_IMAGE:
                probs.append(f"row {row} lacks a block")
            seen = []
            for k in range(MM.N_IMAGE):
                blk = ids[k * MM.BLOCK:(k + 1) * MM.BLOCK]
                mod = bo_to_mod.get(int(blk[0]))
                seen.append(mod)
                if mod is None or not torch.equal(blk, base._load_chunk(mod, int(row))):
                    probs.append(f"row {row} block {k} ({mod}) not the dataset's chunk")
                mid = mb["modality_ids"][r][k * MM.BLOCK + 1]
                if mod is not None and int(mid) != int(V.MODALITY_TO_ID[mod]):
                    probs.append(f"row {row} block {k} modality id {int(mid)}")
            if sorted(seen) != sorted(base.present_modalities(int(row))):
                probs.append(f"row {row}: blocks {seen}")
            # control: two blocks swapped (the second block's content in the first block's place)
            sw = ids.clone()
            sw[:MM.BLOCK], sw[MM.BLOCK:2 * MM.BLOCK] = ids[MM.BLOCK:2 * MM.BLOCK].clone(), ids[:MM.BLOCK].clone()
            ctl_fires.append(not torch.equal(sw[:MM.BLOCK], base._load_chunk(seen[0], int(row))))
    corpus = __import__("pandas").read_parquet(fake_train_root() / "tok_index.parquet", columns=["row", "corpus"]) \
        .sort_values("row").corpus.values
    if not (corpus[ds.pool] == "majortom").all():
        probs.append("the pool holds a non-Major TOM row")
    record("Z28b replay sequences are the base dataset's blocks", not probs,
           f"{n} sequences, pool {len(ds.pool)} Major TOM rows; problems {probs[:4]}")
    record("Z28b control: two blocks swapped fail the identity", all(ctl_fires), f"fires on {sum(ctl_fires)} of {n}")


def z28c() -> None:
    want = {"s46_rp50_pastis": 5940 / (5940 + 6304), "s46_rp25_pastis": 1980 / (1980 + 6304)}
    for name, w in want.items():
        ds = replay_ds(name)
        got = [ds.labelled_share(s) for s in (0, 17, 36)]
        record(f"Z28c {name}: base share of the labelled tokens", all(abs(g - w) < 1e-12 for g in got),
               f"{[round(g, 4) for g in got]} against {w:.4f}")
    ds = replay_ds("s46_rp50_pastis", ["replay.base_microbatches=0", "total_batch_size=4"])
    record("Z28c control: no base micro-batch reads share 0", ds.labelled_share(0) == 0.0 and ds.M == 4,
           f"share {ds.labelled_share(0)}, {ds.M} micro-batches a step")


def z28d() -> None:
    from types import SimpleNamespace
    for name in OVERLAYS:
        ds, r0 = replay_ds(name), r0_ds(OVERLAYS[name])
        cfg = _compose(f"eo_seg/{name}")
        ok = len(ds) % int(cfg.total_batch_size) == 0 and len(ds) // int(cfg.total_batch_size) == r0.steps_per_epoch()
        cb = ds.epoch_callback(verbose=False)
        state = SimpleNamespace(global_step=3 * ds.steps, is_world_process_zero=True)
        args = SimpleNamespace(dataloader_persistent_workers=False, gradient_accumulation_steps=int(cfg.total_batch_size))
        try:
            cb.on_epoch_begin(args, state, None, train_dataloader=list(range(len(ds))))
            ok = ok and ds.draw_epoch == 3
            msg = "accepted"
        except RuntimeError as e:
            ok, msg = False, str(e)
        record(f"Z28d {name}: HF's epoch is the segmentation epoch", ok,
               f"{len(ds)} micro-batches / {cfg.total_batch_size} = {len(ds) / int(cfg.total_batch_size)} steps, R0's "
               f"{r0.steps_per_epoch()}; callback {msg}, draw epoch {ds.draw_epoch}")
        args.gradient_accumulation_steps -= 1
        try:
            cb.on_epoch_begin(args, state, None, train_dataloader=list(range(len(ds))))
            fired = False
        except RuntimeError:
            fired = True
        ds.set_draw_epoch(0)
        record(f"Z28d control {name}: accumulation one short is refused", fired, f"refused {fired}")


def z28e() -> None:
    from omegaconf import OmegaConf
    for name, parent in OVERLAYS.items():
        a = _flat(OmegaConf.to_container(_compose(f"eo_seg/{parent}"), resolve=False))
        b = _flat(OmegaConf.to_container(_compose(f"eo_seg/{name}"), resolve=False))
        d = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        record(f"Z28e {name}", d == REPLAY_KEYS, f"differs from {parent} in {sorted(d)}")
    a = _flat(OmegaConf.to_container(_compose("eo_seg/r0_s41_lora16"), resolve=False))
    b = _flat(OmegaConf.to_container(_compose("eo_seg/s52_cloudsen12"), resolve=False))
    d = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    record("Z28e control: r0_s41_lora16 against s52_cloudsen12", d != REPLAY_KEYS, f"differs in {sorted(d)}")
    from eo.finetune.init import init_from
    from eo.generate.conditional import build_model
    m, cfg = build_model("eo_seg/s46_rp50_pastis", None, device="cpu", overrides=[f"finetune.init_from={MAP_BASE}"])
    rep = init_from(m, cfg)
    record("Z28e s46_rp50_pastis trains R0's parameters", rep["trainable_params"] == 51_014_208,
           f"trainable {rep['trainable_params']:,} (R0: 51,014,208)")


def _agg(doc, first=None):
    rows = np.asarray(doc["per_row"]["rows"], np.float64)[:first]
    return rows[:, 1].sum() / rows[:, 0].sum(), rows


def z28f() -> None:
    want = {"map_base": 5.067648963092788, "r0p1s41_l16c_s42_20261122_123818__851": 6.590122991617845}
    tol = {"map_base": 1e-5, "r0p1s41_l16c_s42_20261122_123818__851": 1e-4}
    for name, w in want.items():
        doc = json.loads((F_DIR / f"{name}.json").read_text())
        got = doc["ce_image_tokens"]["ce_full"]
        pr, rows = _agg(doc)
        ok = abs(got - w) <= tol[name] and abs(pr - got) < 1e-7 and len(rows) == 256 and doc["rows"] == 256
        record(f"Z28f {name}", ok, f"ce_full {got:.7f} against 4.4's {w:.7f} (tol {tol[name]}); per-row "
               f"{pr:.10f}; {len(rows)} rows")
    doc = json.loads((F_DIR / "map_base.json").read_text())
    sub, _ = _agg(doc, 128)
    record("Z28f control: the base over 128 rows moves", abs(sub - want["map_base"]) > 1e-5,
           f"{sub:.7f} ({sub - want['map_base']:+.2e})")


def z28p() -> None:
    import subprocess
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {save!r} is not under /data")
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_z28p"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="",
               TERRAMESH_TRAIN_TOK_ROOT=str(fake_train_root()))
    ov = {"name": out.name, "output_dir": rel, "wandb_run_name": out.name, "wandb": "false",
          "finetune.init_from": str(MAP_BASE), "num_train_steps": "2", "stop_steps": "2", "num_warmup_steps": "0",
          "lr_scheduler_kwargs.num_decay_steps": "0", "save_steps": "2", "eval_on_start": "false", "eval_steps": "1000",
          "replay.seg_per_microbatch": "2", "replay.base_per_microbatch": "1", "total_batch_size": "4",
          "logging_steps": "1", "dataloader_num_workers": "0", "precision": "fp32", "plateau_stop.enable": "false",
          "learning_rate": "3e-3", "hydra.run.dir": f"{out}/hydra"}
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name",
           "eo_seg/s46_rp50_pastis"] + [f"{k}={v}" for k, v in ov.items()]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    out.mkdir(parents=True, exist_ok=True)
    log = r.stdout + r.stderr
    (out / "pretrain.log").write_text(log)
    st = out / "trainer_state.json"
    hist = json.loads(st.read_text())["log_history"] if st.exists() else []
    losses = [h["loss"] for h in hist if "loss" in h]
    ok = (r.returncode == 0 and (out / "checkpoint-2").is_dir() and "seg replay train" in log and len(losses) == 2
          and all(math.isfinite(x) for x in losses))
    line = next((ln for ln in log.splitlines() if "seg replay train" in ln), "")
    record("Z28p RP through pretrain.py", ok, f"exit {r.returncode}; losses {losses}; {time.time() - t0:.0f} s, CPU "
           f"fp32; '{line[-90:]}'; log {out / 'pretrain.log'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = {"Z28a": z28a, "Z28b": z28b, "Z28c": z28c, "Z28d": z28d, "Z28e": z28e, "Z28f": z28f, "Z28p": z28p}
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
