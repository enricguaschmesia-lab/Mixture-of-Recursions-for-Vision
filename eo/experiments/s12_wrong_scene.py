#!/usr/bin/env python
"""Step 12.1, built-in check 2's control: what does a genuinely broken prompt cost?

Check 2, as amended 2026-10-04 23:06 (PHASE3_PLAN.md 12.1): no one-to-one grid cell
is worse than the target-alone row (∅) by more than δ = 0.01 nats/token, i.e. no
cell's 95% CI of the paired gain CE(S->T) - CE(∅->T) has its lower bound above δ.
The check exists to catch wiring bugs. This measures whether it CAN, using the most
plausible silent one: the source tokens come from ANOTHER scene (an off-by-one, a
mis-joined row id) while the target is the right scene's.

For every one-to-one cell S -> T, at one checkpoint of one arm, on a stratified
subset of the grid's own rows (stratified_target_rows(..., T, --rows): the head of
the same corpus-alternating sequence the grid's 512 rows come from):
  none    the target alone, sources=[]                       (the grid's ∅ row)
  true    sources=[S] with the row's own S                   (the grid's cell)
  wrong   sources=[S] with S taken from a DONOR row of the same cell. Donors are a
          seeded derangement of the cell's rows: every row gets a different scene
          of the same population (corpus mix), and the layout -- positions, BO/EO,
          codebook offsets -- is token for token the true prompt's. Only the scene
          changes.
Per cell, paired by row: gain_true = true - none and gain_wrong = wrong - none,
each with a 95% percentile bootstrap over rows (10,000, seed fixed per cell), and
whether check 2 would fire on the wrong-scene prompt (CI lower bound > δ).

CONTROL, before anything is written: with every donor the row itself, `wrong` must
equal `true` exactly (same card, same batches) on two cells, one per corpus side.
It proves the substitution goes through the grid's own path.

⚠ δ is NOT revised from this after any grid number has been read (plan 12.1).
⚠ Gains are differences between passes on the SAME card, so the card does not enter
them; this runs on the Titan X while the grid runs on the TITAN V. Arm differences
are not computed here: wrong card for that, and the grid owns them.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(python -m eo.train.gpu titanx) \
      python eo/experiments/s12_wrong_scene.py run --arm-tag a10 \
        --arm-config eo_terramesh/arm_a10_mor --run-dir <run dir> --checkpoint checkpoint-3300
    python eo/experiments/s12_wrong_scene.py analyze

Outputs under /data/enric/reports/grid/calibration/. Runs in `.venv`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eo.data.eo_vocab import IMAGE_MODALITIES, MODALITIES, MODALITY_TO_ID, PAD_ID  # noqa: E402
from eo.data.eval_split import load_eval_rows, load_row_table  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402

ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
OUT = Path("/data/enric/reports/grid/calibration")
TARGETS, SOURCES = list(IMAGE_MODALITIES), list(MODALITIES)
HOLES = {"S1GRD->S1RTC", "S1RTC->S1GRD"}
ONE_TO_ONE = [f"{s}->{t}" for t in TARGETS for s in SOURCES if s != t and f"{s}->{t}" not in HOLES]
assert len(ONE_TO_ONE) == 34
DELTA = 0.01
CONTROL_CELLS = ["S2L2A->NDVI", "S1GRD->DEM"]     # S1GRD rows are ssl4eos12-only


class WrongScene(TF.TargetLast):
    """TargetLast(sources=[S]), except that S's tokens come from `donor[row]`."""

    def __init__(self, base, rows, target, source, donor):
        super().__init__(base, rows, target, sources=[source])
        self.source, self.donor = source, donor

    def __getitem__(self, i):
        from lm_dataset.sequence_assembly import assemble_sequence
        row = self.rows[i]
        order = [self.source, self.target]          # registry order puts every source first
        chunks = [self.base._load_chunk(self.source, self.donor[row]),
                  self.base._load_chunk(self.target, row)]
        return assemble_sequence(
            chunks=chunks, chunk_modality_ids=[MODALITY_TO_ID[m] for m in order],
            chunk_shufflable=[m in IMAGE_MODALITIES for m in order],
            max_length=self.base.max_length, pad_id=PAD_ID,
            rng=np.random.default_rng(0), shuffle_image_patches=False)


def derangement(rows, key, seed):
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    p = [rows[i] for i in rng.permutation(len(rows))]
    return {p[i]: p[(i + 1) % len(p)] for i in range(len(p))}


def per_row(model, ov, dset, T, workers):
    if ov is not None:
        TF.install_override(model, "router")
    dl = torch.utils.data.DataLoader(dset, batch_size=4, num_workers=workers,
                                     collate_fn=TF.trim_collate)
    r = TF.run_pass(model, dl, override=ov, keep_tokens=False, per_row_target=T)["rows"]
    return r["ce_sum"] / np.maximum(r["n"], 1)


def boot(x, key, seed, n_boot=10_000):
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"n": int(len(x)), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def run(args) -> int:
    from eo.generate import conditional as G
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU")
    gpu = torch.cuda.get_device_name(0)
    if args.expect_gpu not in gpu:
        raise SystemExit(f"CUDA sees {gpu!r}, expected {args.expect_gpu!r}")
    torch.manual_seed(42)
    model, cfg = G.build_model(args.arm_config, str(Path(args.run_dir) / args.checkpoint))
    ov = TF.install_override(model, "router")
    ds, _ = TF.eval_loader(cfg, num_workers=args.num_workers)
    eval_rows, corpus = load_eval_rows(root_dir=ROOT), load_row_table(ROOT).corpus.values
    out = Path(args.out) / args.arm_tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    R = {T: TF.stratified_target_rows(ds, eval_rows, corpus, T, args.rows) for T in TARGETS}
    # control: identity donors must reproduce the true-source pass exactly
    for key in CONTROL_CELLS:
        S, T = key.split("->")
        rows = [r for r in R[T] if S in ds.present_modalities(r)]
        a = per_row(model, ov, TF.TargetLast(ds, rows, T, sources=[S]), T, args.num_workers)
        b = per_row(model, ov, WrongScene(ds, rows, T, S, {r: r for r in rows}), T, args.num_workers)
        if not np.array_equal(a, b):
            raise SystemExit(f"CONTROL FAILED on {key}: identity donors differ from sources=[S] "
                             f"(max |d| {np.abs(a - b).max():.3e}); the substitution path is wrong")
        print(f"control {key}: identity donors == sources=[{S}] on {len(rows)} rows, exactly", flush=True)

    arrays, cells = {}, {}
    for T in TARGETS:
        none = per_row(model, ov, TF.TargetLast(ds, R[T], T, sources=[]), T, args.num_workers)
        z = dict(zip(R[T], none))
        for key in [k for k in ONE_TO_ONE if k.endswith(f"->{T}")]:
            S = key.split("->")[0]
            rows = [r for r in R[T] if S in ds.present_modalities(r)]
            donor = derangement(rows, key, args.seed)
            assert all(donor[r] != r for r in rows) and sorted(donor.values()) == sorted(rows)
            true = per_row(model, ov, TF.TargetLast(ds, rows, T, sources=[S]), T, args.num_workers)
            wrong = per_row(model, ov, WrongScene(ds, rows, T, S, donor), T, args.num_workers)
            base = np.array([z[r] for r in rows])
            for name, v in (("rows", np.asarray(rows)), ("donor", np.array([donor[r] for r in rows])),
                            ("none", base), ("true", true), ("wrong", wrong)):
                arrays[f"{key}|{name}"] = v
            cells[key] = {"n": len(rows), "none": float(base.mean()), "true": float(true.mean()),
                          "wrong": float(wrong.mean())}
            print(f"{args.arm_tag} {key:15} n {len(rows):3d}  gain true {true.mean() - base.mean():+.4f}"
                  f"  wrong {wrong.mean() - base.mean():+.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    doc = {"arm_tag": args.arm_tag, "arm_config": args.arm_config, "run_dir": args.run_dir,
           "checkpoint": args.checkpoint, "gpu": gpu, "rows_per_column": args.rows, "seed": args.seed,
           "control_cells": CONTROL_CELLS, "cells": cells, "seconds": round(time.perf_counter() - t0, 1)}
    tmp = out / "wrong_scene.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out / "wrong_scene_rows.npz")
    tj = out / "wrong_scene.json.tmp"
    tj.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tj, out / "wrong_scene.json")
    print(f"-> {out}", flush=True)
    return 0


def analyze(args) -> int:
    res, lines = {"delta": DELTA}, []
    root = Path(args.out)
    for d in sorted(p for p in root.iterdir() if (p / "wrong_scene.json").exists()):
        doc = json.loads((d / "wrong_scene.json").read_text())
        z = np.load(d / "wrong_scene_rows.npz")
        per = {}
        for key in ONE_TO_ONE:
            none = z[f"{key}|none"]
            gt = boot(z[f"{key}|true"] - none, f"true|{key}", args.seed)
            gw = boot(z[f"{key}|wrong"] - none, f"wrong|{key}", args.seed)
            per[key] = {"gain_true": gt, "gain_wrong": gw,
                        "fires_wrong": gw["lo"] > DELTA, "fires_true": gt["lo"] > DELTA,
                        "wrong_minus_true": gw["mean"] - gt["mean"]}
        fw = [k for k in ONE_TO_ONE if per[k]["fires_wrong"]]
        ft = [k for k in ONE_TO_ONE if per[k]["fires_true"]]
        gws = np.array([per[k]["gain_wrong"]["mean"] for k in ONE_TO_ONE])
        res[d.name] = {"checkpoint": doc["checkpoint"], "gpu": doc["gpu"], "cells": per,
                       "fires_on_wrong_scene": fw, "fires_on_true_source": ft,
                       "gain_wrong_median": float(np.median(gws)), "gain_wrong_min": float(gws.min())}
        lines += [f"{d.name} ({doc['checkpoint']}, {doc['gpu']}, {doc['rows_per_column']} rows/column):",
                  f"  {'cell':15} {'n':>4} {'gain true [95% CI]':>28} {'gain wrong [95% CI]':>28}  fires"]
        for k in ONE_TO_ONE:
            gt, gw = per[k]["gain_true"], per[k]["gain_wrong"]
            lines.append(f"  {k:15} {gt['n']:4d} {gt['mean']:+8.4f} [{gt['lo']:+.4f},{gt['hi']:+.4f}]"
                         f" {gw['mean']:+8.4f} [{gw['lo']:+.4f},{gw['hi']:+.4f}]  "
                         f"{'WRONG' if per[k]['fires_wrong'] else ''}{' TRUE' if per[k]['fires_true'] else ''}")
        lines += [f"  check 2 (δ = {DELTA}) fires on the wrong-scene prompt in {len(fw)}/34 cells; "
                  f"median gain_wrong {np.median(gws):+.4f}, smallest {gws.min():+.4f}",
                  f"  on the true sources it fires in {len(ft)}/34 cells{': ' + ', '.join(ft) if ft else ''}", ""]
    (root / "calibration.json").write_text(json.dumps(res, indent=1) + "\n")
    (root / "calibration.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm-tag", required=True)
    r.add_argument("--arm-config", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--checkpoint", default="checkpoint-3300")
    r.add_argument("--rows", type=int, default=128, help="per column, stratified")
    r.add_argument("--num-workers", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--expect-gpu", default="TITAN X")
    a = sub.add_parser("analyze")
    a.add_argument("--seed", type=int, default=0)
    for p in (r, a):
        p.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
