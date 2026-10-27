#!/usr/bin/env python
"""Teacher-forced per-checkpoint evaluation of one arm (accuracy curves + recursion).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(python -m eo.train.gpu titanv) \
    python eo/scripts/eval_checkpoint.py --arm-tag arm_a --arm-config eo_terramesh/arm_a_mor \
        --run-dir /data/enric/runs/pretrain/phase3/<run> untrained checkpoint-2000 ... \
        [--modes router,force1,force2,force3,perm_modality,perm_all]

Per checkpoint, writes /data/enric/reports/arm_eval/<arm-tag>/<ckpt>/:
  metrics.json         per-modality CE / top-1 / top-5 / slot mass, for every
                       mode, plus the target-last pass and the LOGGED eval loss
                       at that step for comparison
  tokens_router.npz    per-token CE, correctness, modality and (arm A) depth
  tokens_<mode>.npz    CE + correctness under each intervention
  grid.json            with --grid (Step 12.1): per source -> target cell, CE
                       (mean of per-row means, and token-weighted), top-1 and
                       the MoR depth mix; holes recorded as such
  grid_rows.npz        with --grid: per-row arrays per cell, for pairing arms

--grid runs one teacher-forced pass per cell over the target-last rows that
carry the cell's source (PHASE3_PLAN.md 12.1, 4.16), batches trimmed to their
longest sequence (trim_collate). --grid-all-others adds the "all" row
(sources=None, untrimmed), whose CE must equal target_last[T].ce from the same
card: P3 as a standing check, stored as `p3_abs_diff`. --modes '' skips the
main pass, e.g. for a grid-only run on an already-evaluated checkpoint.

⚠ THE CHECK THAT THIS PASS IS THE TRAINER'S PASS: `ce_batch_mean` must match
`eval_loss_<mod>` in trainer_state.json at the same step to within fp16
autocast noise. It is printed per modality and stored; a large gap means the
loader, the modality order or the attribution differs from training.

⚠ THE TOKENS ARE THE CONFIG'S: `multimodal.token_set` in --arm-config picks the
root (original | clean, lm_dataset/load_dataset.py) and metrics.json records it.
To score an original-token checkpoint on the cleaned tokens (the bridge, next
steps §3), pass the cleaned arm's config and a new --arm-tag: the model is the
same, and `logged_eval` then differs from the pass in S2L2A/NDVI by design.

⚠ `untrained` is a legitimate checkpoint name: a randomly initialised model
(seed 42), the step-0 point of every curve.

--force-depth D (the depth sweep, 2026-10-27; RANDOM-router arms only) puts
every token at D passes of the shared block for every pass of the run (main,
target-last, grid), D above the trained Nr included (TF.install_random_depth).
One D per --arm-tag directory: metrics.json records it, and a directory
holding another D is refused. Every pass checks that the depth it captured is
D at every position, and fails otherwise.
--gate-seed S (default 0, every sweep unit) seeds the forced run's random gate
draws; another S at the same D measures how much the draw alone moves a result.
A directory holding another S is refused.

Arm B works unchanged: it has no router, so it gets the accuracy metrics and
no depth, and any --modes other than `router` are refused for it.

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eo.data.eo_vocab import MODALITIES  # noqa: E402
from eo.data.eval_split import load_eval_rows, load_row_table  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402
from eo.generate import conditional as G  # noqa: E402

ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
TARGETS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]

# --grid (PHASE3_PLAN.md Step 12.1): the source x target grid of teacher-forced
# CE. Rows are the seven single sources plus "none" (the target alone) and, with
# --grid-all-others, "all" (sources=None: the target-last pass itself, untrimmed,
# so its CE must equal target_last[T].ce from the same card -- P3, every run).
GRID_TARGETS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
GRID_SOURCES = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]


def grid_keys(all_others: bool):
    srcs = GRID_SOURCES + ["none"] + (["all"] if all_others else [])
    return [f"{s}->{t}" for t in GRID_TARGETS for s in srcs if s != t]


def check_forced(depth_counts: np.ndarray, force_depth, where: str) -> None:
    """A forced pass must capture depth D at every counted position (columns 1..Nr).
    Empty counts pass (a "none" cell has no source tokens); the main pass's
    check, over every position, is the one that cannot be empty."""
    if force_depth is None:
        return
    tot = np.asarray(depth_counts).reshape(-1, np.asarray(depth_counts).shape[-1]).sum(0)
    if tot.sum() and tot[force_depth - 1] != tot.sum():
        raise SystemExit(f"{where}: forced depth {force_depth}, captured depth counts {tot.tolist()}")


def run_grid(model, ov, ds, eval_rows, corpus, out: Path, doc, args, rf=None) -> None:
    """One teacher-forced pass per cell; resumable per cell.

    Writes `grid.json` (per cell: rows, n, CE as the mean of per-row means and
    token-weighted, top-1, MoR depth mix) and `grid_rows.npz` (per-row arrays,
    for pairing the arms row by row). A cell whose source no row of the column
    carries (S1GRD <-> S1RTC) is recorded as a hole.
    """
    gpath, npath = out / "grid.json", out / "grid_rows.npz"
    grid = json.loads(gpath.read_text()) if gpath.exists() else {"cells": {}}
    arrays = dict(np.load(npath)) if npath.exists() else {}
    for T in GRID_TARGETS:
        col = [k for k in grid_keys(args.grid_all_others) if k.endswith(f"->{T}")
               and k not in grid["cells"]]
        if not col:
            continue
        R = TF.stratified_target_rows(ds, eval_rows, corpus, T, args.target_last_rows)
        for key in col:
            S = key.split("->")[0]
            if S == "all":
                rows, sources, collate = R, None, None
            elif S == "none":
                rows, sources, collate = R, [], TF.trim_collate
            else:
                rows = [r for r in R if S in ds.present_modalities(r)]
                sources, collate = [S], TF.trim_collate
            if not rows:
                grid["cells"][key] = {"source": S, "target": T, "hole": True, "n_rows": 0}
                continue
            t0 = time.perf_counter()
            if ov is not None:
                TF.install_override(model, "router")
            if rf is not None:
                rf.reseed(args.gate_seed)          # each cell reproducible on its own (resume)
            # The batch size only moves fp16 kernel choice (and, under --force-depth, which gate draw
            # lands on which token): trim_collate cuts each batch to its longest row. 16 runs a cell
            # 2.1-2.6x faster (2026-10-27, TITAN V); the "all" row keeps 4, as its P3 reference.
            bs = 4 if S == "all" else args.grid_batch_size
            dl = torch.utils.data.DataLoader(TF.TargetLast(ds, rows, T, sources=sources),
                                             batch_size=bs, num_workers=args.num_workers,
                                             collate_fn=collate)
            res = TF.run_pass(model, dl, override=ov, keep_tokens=False, per_row_target=T)
            pr = res["rows"]
            row_ce = pr["ce_sum"] / np.maximum(pr["n"], 1)
            cell = {"source": S, "target": T, "hole": False, "n_rows": len(rows),
                    "ce_row_mean": float(row_ce.mean()),
                    "ce_token_weighted": res["per_modality"][T]["ce"],
                    "top1": res["per_modality"][T]["top1"],
                    "top5": res["per_modality"][T]["top5"],
                    "n_tokens": res["per_modality"][T]["n_tokens"],
                    "trimmed": collate is not None, "batch_size": bs,
                    "seconds": round(time.perf_counter() - t0, 1)}
            arrays[f"{key}|rows"] = np.asarray(rows)
            arrays[f"{key}|n"] = pr["n"]
            arrays[f"{key}|ce_sum"] = pr["ce_sum"]
            arrays[f"{key}|top1"] = pr["top1"]
            for dk in ("d_pred", "d_tgt", "d_src"):
                if dk in pr:
                    check_forced(pr[dk], args.force_depth, f"{out.name} [grid {key}] {dk}")
                    arrays[f"{key}|{dk}"] = pr[dk]
                    tot = pr[dk].sum(0)
                    cell[f"{dk}_mix"] = (tot / max(tot.sum(), 1)).tolist()
                    cell[f"{dk}_mean"] = float((tot * np.arange(1, tot.size + 1)).sum() / max(tot.sum(), 1))
            if S == "all" and "target_last" in doc and T in doc["target_last"]:
                ref = doc["target_last"][T]["ce"]
                cell["target_last_ce"] = ref
                cell["p3_abs_diff"] = abs(cell["ce_token_weighted"] - ref)
            grid["cells"][key] = cell
            print(f"{out.name} [grid {key}] rows {len(rows)}  ce {cell['ce_row_mean']:.4f}  "
                  f"top1 {cell['top1']:.4f}  {cell['seconds']}s"
                  + (f"  P3 |d| {cell['p3_abs_diff']:.2e}" if "p3_abs_diff" in cell else ""))
        grid.update({"gpu": doc.get("gpu"), "checkpoint": doc.get("checkpoint"),
                     "arm_tag": doc.get("arm_tag"), "target_rows": args.target_last_rows})
        write_json(gpath, grid)
        tmp = npath.with_name(npath.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, npath)


def logged_eval(run_dir: Path, step: int):
    """eval_loss + eval_loss_<mod> at `step`. ⚠ They live in SEPARATE log_history
    entries at the same step (handover section 6 item 4) -- join on step."""
    f = run_dir / "trainer_state.json"
    if not f.exists():
        return None
    out = {}
    for e in json.loads(f.read_text())["log_history"]:
        if e.get("step") == step and any(k.startswith("eval_loss") for k in e):
            out.update({k: v for k, v in e.items() if k.startswith("eval_loss")})
    return out or None


def write_json(path: Path, doc) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(doc, indent=2) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-tag", required=True)
    ap.add_argument("--arm-config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--modes", default="router")
    ap.add_argument("--target-last-rows", type=int, default=512)
    ap.add_argument("--out-root", default="/data/enric/reports/arm_eval")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="must equal the training config's dataloader_num_workers")
    ap.add_argument("--grid", action="store_true",
                    help="also run Step 12's source x target grid (grid.json, grid_rows.npz)")
    ap.add_argument("--grid-all-others", action="store_true",
                    help="add the grid's 'all' row (sources=None, untrimmed): the P3 check")
    ap.add_argument("--grid-batch-size", type=int, default=4,
                    help="batch size of the grid's trimmed cells (default 4, as every grid before 2026-10-27)")
    ap.add_argument("--force-depth", type=int, default=None,
                    help="random-router arms: every token at this many passes (the depth sweep)")
    ap.add_argument("--gate-seed", type=int, default=0,
                    help="with --force-depth: the seed of the random gate draws (0 for every sweep unit; "
                         "another seed at the same depth measures how much the draw alone moves results)")
    args = ap.parse_args()
    if args.gate_seed != 0 and args.force_depth is None:
        raise SystemExit("--gate-seed applies only with --force-depth")

    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=...")
    gpu = torch.cuda.get_device_name(0)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    run_dir = Path(args.run_dir)

    for ck in args.checkpoints:
        out = Path(args.out_root) / args.arm_tag / ck
        out.mkdir(parents=True, exist_ok=True)
        mpath = out / "metrics.json"
        doc = json.loads(mpath.read_text()) if mpath.exists() else {}
        # Before the skip: a finished directory of another depth or gate draw must not pass as this run.
        if doc and doc.get("force_depth") != args.force_depth:
            raise SystemExit(f"{out} holds force_depth={doc.get('force_depth')!r} results; "
                             f"this run forces {args.force_depth!r}. Use another --arm-tag.")
        if doc and args.force_depth is not None and doc.get("gate_seed", 0) != args.gate_seed:
            raise SystemExit(f"{out} holds gate_seed={doc.get('gate_seed', 0)!r} results; "
                             f"this run draws with {args.gate_seed!r}. Use another --arm-tag.")
        todo = [m for m in modes if m not in doc.get("modes", {})]
        need_tl = "target_last" not in doc
        # ⚠ A checkpoint with every mode and target_last done must still run a
        #   requested grid; without this the --grid run silently does nothing.
        gpath = out / "grid.json"
        have = json.loads(gpath.read_text())["cells"] if gpath.exists() else {}
        need_grid = args.grid and any(k not in have for k in grid_keys(args.grid_all_others))
        if not todo and not need_tl and not need_grid:
            print(f"[skip] {ck}")
            continue

        torch.manual_seed(42)
        ck_path = None if ck == "untrained" else str(run_dir / ck)
        model, cfg = G.build_model(args.arm_config, ck_path)
        if int(cfg.dataloader_num_workers) != args.num_workers:
            raise SystemExit(f"--num-workers {args.num_workers} != config's "
                             f"{cfg.dataloader_num_workers}: modality orders would differ from training")
        rf = None
        if args.force_depth is not None:
            if any(m != "router" for m in modes):
                raise SystemExit("--force-depth runs the main pass as --modes router (or '')")
            rf = TF.install_random_depth(model, args.force_depth)
        ov = TF.install_override(model, "router")
        if ov is None and any(m != "router" for m in todo):
            raise SystemExit(f"{ck}: interventions need a router; this arm has none")
        step = 0 if ck == "untrained" else int(ck.split("-")[-1])
        # The tokens come from the config's multimodal.token_set (absent = original, every
        # run before B10c). Results resume per directory, so one directory holds one set.
        token_set = (cfg.get("multimodal") or {}).get("token_set", "original")
        if doc and doc.get("token_set", "original") != token_set:
            raise SystemExit(f"{out} holds token_set={doc.get('token_set', 'original')!r} results; "
                             f"this config reads {token_set!r}. Use another --arm-tag.")
        ds, loader = TF.eval_loader(cfg, num_workers=args.num_workers)
        doc.update({"arm_tag": args.arm_tag, "arm_config": args.arm_config, "checkpoint": ck,
                    "token_set": token_set, "token_root": str(ds.root_dir),
                    "step": step, "gpu": gpu, "n_eval_rows": len(ds),
                    "has_router": ov is not None,
                    "force_depth": args.force_depth,
                    "logged_eval": logged_eval(run_dir, step) if ck != "untrained" else None})
        if rf is not None:
            doc["trained_recursions"] = rf.n_trained
            doc["gate_seed"] = args.gate_seed
        doc.setdefault("modes", {})

        for mode in todo:
            t0 = time.perf_counter()
            if ov is not None:
                TF.install_override(model, mode, seed=0)
            if rf is not None:
                rf.reseed(args.gate_seed)
            res = TF.run_pass(model, loader, override=ov)
            tok = res.pop("tokens")
            if rf is not None and not (tok["depth"] == args.force_depth).all():
                raise SystemExit(f"{ck}: forced depth {args.force_depth}, captured "
                                 f"{np.unique(tok['depth'], return_counts=True)}")
            keep = tok if mode == "router" else {k: tok[k] for k in ("ce", "correct", "depth") if k in tok}
            np.savez_compressed(out / f"tokens_{mode}.npz", **keep)
            if mode == "router":
                np.save(out / "eval_rows.npy", np.asarray(ds.rows))
            res["seconds"] = round(time.perf_counter() - t0, 1)
            if "depth" in tok:
                v = tok["label_mod"] > 0 if "label_mod" in tok else None
                d = tok["depth"][v] if v is not None else tok["depth"].ravel()
                res["mean_depth_body"] = float(d.mean())
            doc["modes"][mode] = res
            write_json(mpath, doc)
            pm = res["per_modality"]
            print(f"{ck} [{mode}] {res['seconds']}s  CE(all) {res['ce_all_body_tokens']:.4f}"
                  + (f"  depth {res.get('mean_depth_body', 0):.3f}" if ov is not None or rf is not None else ""))
            for n in pm:
                lg = (doc["logged_eval"] or {}).get(f"eval_loss_{n}")
                print(f"   {n:7s} ce {pm[n]['ce']:.4f}  batch-mean {pm[n]['ce_batch_mean']:.4f}"
                      f"  logged {lg if lg is not None else '-':>7}  top1 {pm[n]['top1']:.4f}"
                      f"  top5 {pm[n]['top5']:.4f}  slot {pm[n]['slot_mass']:.4f}")

        if need_tl:
            if ov is not None:
                TF.install_override(model, "router")
            corpus = load_row_table(ROOT).corpus.values
            eval_rows = load_eval_rows(root_dir=ROOT)
            tl = {}
            for T in TARGETS:
                rows = TF.stratified_target_rows(ds, eval_rows, corpus, T, args.target_last_rows)
                dl = torch.utils.data.DataLoader(TF.TargetLast(ds, rows, T), batch_size=4,
                                                 num_workers=args.num_workers)
                if rf is not None:
                    rf.reseed(args.gate_seed)
                res = TF.run_pass(model, dl, override=ov, keep_tokens=ov is not None or rf is not None)
                r = {"n_rows": len(rows), **res["per_modality"][T]}
                if ov is not None or rf is not None:
                    t = res["tokens"]
                    sel = t["label_mod"] == TF.MODALITY_TO_ID[T]
                    r["mean_depth_target"] = float(t["depth"][sel].mean())
                    if rf is not None and not (t["depth"] == args.force_depth).all():
                        raise SystemExit(f"{ck} [target-last {T}]: forced depth {args.force_depth}, "
                                         f"captured {np.unique(t['depth'], return_counts=True)}")
                tl[T] = r
                print(f"{ck} [target-last {T}] rows {len(rows)}  ce {r['ce']:.4f}  "
                      f"top1 {r['top1']:.4f}  top5 {r['top5']:.4f}  slot {r['slot_mass']:.4f}")
            doc["target_last"] = tl
            write_json(mpath, doc)

        if need_grid:
            run_grid(model, ov, ds, load_eval_rows(root_dir=ROOT), load_row_table(ROOT).corpus.values,
                     out, doc, args, rf=rf)

        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
