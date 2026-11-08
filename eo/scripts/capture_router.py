#!/usr/bin/env python
"""The router's full output at a checkpoint, for every row of a set (B0, Steps 1 and 3).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(python -m eo.train.gpu titanx) \
    python eo/scripts/capture_router.py --arm-tag a10c \
        --run-dir /data/enric/runs/pretrain/phase3/mor10c_20261012_013533 \
        --sets main,target_last --precisions fp16,fp32 untrained checkpoint-330 ...

Reads embedding + layer 0 + router and stops there (eo/routing/telemetry.capture_router): about
75-105 s for all 4,416 rows on the Titan X, against 1,814 s for a full run_pass there. Writes one
file per unit under <out-root>/<arm-tag>/<checkpoint>/, resumable by unit, atomically:

  main_<precision>.npz         logits, depth, input_mod, label_mod, rows: the trainer's held-out
                               loader, so it pairs with arm_eval/<tag>/<ck>/tokens_router.npz
  target_last_<precision>.npz  the same per target, keys "<T>|logits"..., on eval_checkpoint's rows
  train_<precision>.npz        --train-rows training rows drawn per corpus in proportion, in
                               random modality order as in training
  probe_<precision>.npz        --probe-rows held-out rows stratified by corpus (Step 3's replays)
  <set>_<precision>.json       card, seconds, shape, code version

⚠ --arm-config stays eo_terramesh/arm_a10c_mor for every seed: `seed` also sets the held-out
sequences, so this is what makes two seeds read one set of sequences (arm_a10cs2_mor.yaml says so).
⚠ `untrained` is seed 42's init whatever the tag: torch.manual_seed(42), as eval_checkpoint.
⚠ fp16 logits are stored as float16, exactly; bf16 and fp32 as float32.

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eo.data.eval_split import load_eval_rows, load_row_table, train_rows_from_eval  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402
from eo.generate import conditional as G  # noqa: E402
from eo.routing.telemetry import capture_router  # noqa: E402

ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
PRECISION = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}
TARGETS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
SETS = ("main", "target_last", "train", "probe")


def pack(res, precision, prefix=""):
    """Arrays to store; fp16 logits are float16 exactly (checked)."""
    out = {f"{prefix}{k}": v for k, v in res.items()}
    if precision == "fp16":
        lg = res["logits"]
        half = lg.astype(np.float16)
        if not np.array_equal(half.astype(np.float32), lg):
            raise RuntimeError("fp16 logits did not survive float16")
        out[f"{prefix}logits"] = half
    return out


def sample_rows(rows, corpus, n, seed=20261107):
    """n rows, each corpus in proportion. ⚠ Rows are sorted majortom-first: never rows[:n]."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    keep = []
    for c in ("majortom", "ssl4eos12"):
        have = rows[corpus[rows] == c]
        keep.append(rng.choice(have, size=int(round(n * len(have) / len(rows))), replace=False))
    return np.sort(np.concatenate(keep))


def subset(ds, rows):
    """The same dataset over other rows. A scene's sequence does not depend on the subset."""
    d = copy.copy(ds)
    d.rows = np.asarray(rows, dtype=np.int64)
    return d


def loader_for(ds, workers):
    return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=workers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-tag", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm-config", default="eo_terramesh/arm_a10c_mor")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--sets", default="main")
    ap.add_argument("--precisions", default="fp16")
    ap.add_argument("--out-root", default="/data/enric/reports/b0/router")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--train-rows", type=int, default=4416)
    ap.add_argument("--probe-rows", type=int, default=256)
    ap.add_argument("--target-last-rows", type=int, default=512)
    args = ap.parse_args()
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    precs = [p.strip() for p in args.precisions.split(",") if p.strip()]
    if not set(sets) <= set(SETS) or not set(precs) <= set(PRECISION):
        raise SystemExit(f"--sets from {SETS}, --precisions from {tuple(PRECISION)}")
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=...")
    gpu = torch.cuda.get_device_name(0)
    code = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2]).stdout.strip()

    for ck in args.checkpoints:
        out = Path(args.out_root) / args.arm_tag / ck
        out.mkdir(parents=True, exist_ok=True)
        todo = [(s, p) for s in sets for p in precs if not (out / f"{s}_{p}.npz").exists()]
        if not todo:
            print(f"[skip] {args.arm_tag} {ck}")
            continue
        torch.manual_seed(42)
        ck_path = None if ck == "untrained" else str(Path(args.run_dir) / ck)
        model, cfg = G.build_model(args.arm_config, ck_path)
        if int(cfg.dataloader_num_workers) != args.num_workers:
            raise SystemExit("--num-workers must equal the config's: modality orders follow it")
        ds, loader = TF.eval_loader(cfg, num_workers=args.num_workers)
        table = load_row_table(ROOT)
        corpus = table.corpus.values
        eval_rows = load_eval_rows(root_dir=ROOT)
        for s, p in todo:
            t0 = time.perf_counter()
            dt = PRECISION[p]
            if s == "main":
                res = capture_router(model, loader, dtype=dt)
                arrays = pack(res, p)
                arrays["rows"] = np.asarray(ds.rows)
            elif s == "train":
                rows = sample_rows(train_rows_from_eval(eval_rows, len(table)), corpus, args.train_rows)
                arrays = pack(capture_router(model, loader_for(subset(ds, rows), args.num_workers),
                                             dtype=dt), p)
                arrays["rows"] = rows
            elif s == "probe":
                rows = sample_rows(np.asarray(ds.rows), corpus, args.probe_rows)
                arrays = pack(capture_router(model, loader_for(subset(ds, rows), args.num_workers),
                                             dtype=dt), p)
                arrays["rows"] = rows
            else:
                arrays = {}
                for T in TARGETS:
                    rows = TF.stratified_target_rows(ds, eval_rows, corpus, T, args.target_last_rows)
                    dl = torch.utils.data.DataLoader(TF.TargetLast(ds, rows, T), batch_size=4,
                                                     num_workers=args.num_workers)
                    arrays.update(pack(capture_router(model, dl, dtype=dt), p, f"{T}|"))
                    arrays[f"{T}|rows"] = np.asarray(rows)
            tmp = out / f".{s}_{p}.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, out / f"{s}_{p}.npz")
            sec = round(time.perf_counter() - t0, 1)
            (out / f"{s}_{p}.json").write_text(json.dumps(
                {"arm_tag": args.arm_tag, "checkpoint": ck, "set": s, "precision": p, "gpu": gpu,
                 "seconds": sec, "code": code, "arm_config": args.arm_config}, indent=2) + "\n")
            print(f"{args.arm_tag} {ck} {s} {p}: {sec}s", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
