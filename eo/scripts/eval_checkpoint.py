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
  target_last_rows.npz per-row arrays of the target-last pass, per target, in
                       grid_rows.npz's layout (2026-10-28: pairs all-to-one
                       across depths and arms)

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

--force-depth D (the depth sweep, 2026-10-27) puts every token at D passes of
the shared block for every pass of the run (main, target-last, grid), D above
the trained Nr included: TF.install_random_depth for a random router, and
(2026-10-28) TF.install_learned_depth for a learned one, whose gate stays the
router's own top probability. One D per --arm-tag directory: metrics.json
records it, and a directory holding another D is refused. Every pass checks
that the depth it captured is D at every position, and fails otherwise.
--gate-seed S (default 0, every sweep unit; random router only) seeds the forced
run's random gate draws; another S at the same D measures how much the draw
alone moves a result. A directory holding another S is refused.
--force-scope target (learned router, 2026-10-28; random router, B0 Step 6, 2026-11-08)
forces only the positions that predict the TARGET's tokens, in target-last and the
grid; every other position keeps the router's choice, so the context is computed
exactly as routed. For a random router "the router's choice" is each position's own
draw from the hook's generator (RandomDepthForce), the same draw at every depth, so
the context is identical across --force-depth. It has no main pass (any-to-any has
no single target): run it with --modes ''. Its check is on the predicting positions
alone. A directory holding another scope is refused.
--own-draw (random router, B0 Step 6) runs every position at its own draw from the
hook's generator: the paired "as routed" baseline of --force-scope target, with the
same --gate-seed. A random router's --table draws depth from the table and the gate
from the same generator, both seeded by --gate-seed, which must equal the spec's seed.
--skip-target-last runs no target-last pass (a main-pass-only unit).

--table SPEC.json:NAME with --modes table (B0 Step 5, 2026-11-08) runs the main pass with
each position's depth drawn from a per-modality table (DepthOverride's `table` mode):
SPEC.json holds {"tables": {NAME: {"mix": {"S2L2A": [p1, p2, p3], ..., "Coords": [...],
"none": null}, "gate": null | {"S2L2A": [g1, g2, g3], ...}, "seed": 0}}}. "none" is key 0,
the positions that predict no body token (null keeps the router's choice there); every
modality must have a row. tokens_table.npz keeps ce, correct, depth and the applied gate.
metrics.json records the table's hash, and a directory holding another table is refused.
--assign PATH.npz with --modes assign (B0 Step 5): every position at a depth given in advance,
`depth` (rows, positions) int, rows in the held-out loader's order, 0 keeping the router's choice
(DepthOverride's `assign` mode), e.g. a cross-fitted token-id table drawn offline. Its hash is
recorded and checked like a table's, and the pass must consume exactly every row.

--precision fp16|bf16|fp32 (2026-10-29) is the autocast of every pass: fp16 (the default, as
the trainer's eval and every result before), bf16, or fp32 (no autocast). An LLM-initialised
body runs close to fp16's range, and its LoRA arm keeps it frozen there, so the LLM-init
comparison is scored at fp32 for every arm. One precision per --arm-tag directory: metrics.json
records it, and a directory holding another is refused.

Arm B works unchanged: it has no router, so it gets the accuracy metrics and
no depth, and any --modes other than `router` are refused for it.

--model-config CONFIG (A1 Step 7, 2026-11-09) builds the model from CONFIG instead of --arm-config:
T360-T's arm (eo_terramesh/arm_t360t_vanilla) at its 87,664 ids, through eval_forgetting.build_temporal.
The checkpoint name `extended_base` loads that arm's pretrain_init base through the `continue` hook
(extension.json verified, strict load, tie): T360-T at step 0, which reads as T360 on these rows (A1
finding 24); any other name is a T360-T checkpoint under --run-dir, loaded strictly. The rows, layout,
token set and worker count stay --arm-config's (T360's single-date arm): the temporal arm's own eval set
is a dict of micro-batch sets, and its batches of 16 key other modality orders. So the reading pairs
token for token with T360's stored one at the same --arm-config, batch size and worker count.

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eo.data.eo_vocab import MODALITIES, MODALITY_TO_ID  # noqa: E402
from eo.data.eval_split import load_eval_rows, load_row_table  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402
from eo.generate import conditional as G  # noqa: E402
from eo.routing.telemetry import has_learned_router  # noqa: E402

ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
PRECISION = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}
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


def run_grid(model, ov, ds, eval_rows, corpus, out: Path, doc, args, rf=None,
             base: str = "router") -> None:
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
                TF.install_override(model, base)
                ov.only_mod = TF.MODALITY_TO_ID[T] if args.force_scope == "target" else None
            if rf is not None:
                rf.reseed(args.gate_seed)          # each cell reproducible on its own (resume)
                rf.only_mod = TF.MODALITY_TO_ID[T] if args.force_scope == "target" else None
            # The batch size only moves fp16 kernel choice (and, under --force-depth, which gate draw
            # lands on which token): trim_collate cuts each batch to its longest row. 16 runs a cell
            # 2.1-2.6x faster (2026-10-27, TITAN V); the "all" row keeps 4, as its P3 reference.
            bs = 4 if S == "all" else args.grid_batch_size
            dl = torch.utils.data.DataLoader(TF.TargetLast(ds, rows, T, sources=sources),
                                             batch_size=bs, num_workers=args.num_workers,
                                             collate_fn=collate)
            res = TF.run_pass(model, dl, override=ov, keep_tokens=False, per_row_target=T,
                              dtype=PRECISION[args.precision])
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
                    if args.force_scope == "all" or dk == "d_pred":
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


def load_table(spec: str) -> dict:
    """--table SPEC.json:NAME -> {mix, gate, seed, sha, name, spec}, keyed by modality id (0 = "none")."""
    path, _, name = spec.rpartition(":")
    if not path or not name:
        raise SystemExit(f"--table {spec!r}: expected SPEC.json:NAME")
    doc = json.loads(Path(path).read_text())
    if name not in doc.get("tables", {}):
        raise SystemExit(f"--table: {path} has no table {name!r} (has {sorted(doc.get('tables', {}))})")
    entry = doc["tables"][name]
    ids = {"none": 0, **MODALITY_TO_ID}

    def keyed(rows):
        if rows is None:
            return None
        bad = sorted(set(rows) - set(ids))
        if bad:
            raise SystemExit(f"--table {name}: unknown keys {bad}")
        return {ids[k]: v for k, v in rows.items()}

    sha = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
    return {"mix": keyed(entry["mix"]), "gate": keyed(entry.get("gate")), "seed": int(entry.get("seed", 0)),
            "sha": sha, "name": name, "spec": str(Path(path).resolve())}


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
    ap.add_argument("--skip-target-last", action="store_true",
                    help="no target-last pass: a main-pass-only unit, e.g. into its own --arm-tag while "
                         "another process writes the target-last of the same model and depth")
    ap.add_argument("--force-depth", type=int, default=None,
                    help="MoR arms: every token at this many passes (the depth sweep)")
    ap.add_argument("--gate-seed", type=int, default=0,
                    help="with --force-depth, random router: the seed of the random gate draws (0 for every "
                         "sweep unit; another seed at the same depth measures how much the draw alone moves results)")
    ap.add_argument("--force-scope", choices=("all", "target"), default="all",
                    help="with --force-depth, learned router: 'target' forces only the positions predicting the "
                         "target's tokens (target-last and grid; needs --modes '')")
    ap.add_argument("--model-config", default=None,
                    help="build the model from this config (T360-T's arm, 87,664 ids) instead of --arm-config, "
                         "which still gives the rows, layout, token set and workers; `extended_base` names its base")
    ap.add_argument("--precision", choices=tuple(PRECISION), default="fp16",
                    help="autocast of every pass: fp16 (default, as the trainer), bf16, or fp32 (none)")
    ap.add_argument("--table", default=None,
                    help="SPEC.json:NAME, with --modes table: depth drawn per position from a per-modality table")
    ap.add_argument("--assign", default=None,
                    help="PATH.npz holding `depth` (rows, positions), with --modes assign: depths given in advance")
    ap.add_argument("--own-draw", action="store_true",
                    help="random router: every position at its own draw from the hook's generator (the paired "
                         "baseline of --force-scope target); not with --force-depth")
    args = ap.parse_args()
    if args.own_draw and args.force_depth is not None:
        raise SystemExit("--own-draw is the unforced baseline: not with --force-depth")
    if args.gate_seed != 0 and args.force_depth is None and not args.own_draw and args.table is None:
        raise SystemExit("--gate-seed applies only with --force-depth, --own-draw or a random router's --table")
    if args.force_scope != "all" and args.force_depth is None:
        raise SystemExit("--force-scope applies only with --force-depth")

    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=...")
    gpu = torch.cuda.get_device_name(0)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    run_dir = Path(args.run_dir)
    table = None
    if args.table is not None or "table" in modes:
        if args.table is None or modes != ["table"]:
            raise SystemExit("--table goes with --modes table, and --modes table with --table, alone")
        if args.force_depth is not None:
            raise SystemExit("--table and --force-depth do not mix")
        table = load_table(args.table)
    assign = None
    if args.assign is not None or "assign" in modes:
        if args.assign is None or modes != ["assign"]:
            raise SystemExit("--assign goes with --modes assign, and --modes assign with --assign, alone")
        if args.force_depth is not None:
            raise SystemExit("--assign and --force-depth do not mix")
        arr = np.load(args.assign)["depth"]
        assign = {"depth": arr, "sha": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16],
                  "path": str(Path(args.assign).resolve())}

    for ck in args.checkpoints:
        out = Path(args.out_root) / args.arm_tag / ck
        out.mkdir(parents=True, exist_ok=True)
        mpath = out / "metrics.json"
        doc = json.loads(mpath.read_text()) if mpath.exists() else {}
        # Before the skip: a finished directory of another depth or gate draw must not pass as this run.
        if doc and doc.get("force_depth") != args.force_depth:
            raise SystemExit(f"{out} holds force_depth={doc.get('force_depth')!r} results; "
                             f"this run forces {args.force_depth!r}. Use another --arm-tag.")
        if doc and bool(doc.get("own_draw", False)) != bool(args.own_draw):
            raise SystemExit(f"{out} holds own_draw={doc.get('own_draw', False)!r} results; "
                             f"this run has own_draw={args.own_draw!r}. Use another --arm-tag.")
        if doc and (args.force_depth is not None or args.own_draw) and doc.get("gate_seed", 0) != args.gate_seed:
            raise SystemExit(f"{out} holds gate_seed={doc.get('gate_seed', 0)!r} results; "
                             f"this run draws with {args.gate_seed!r}. Use another --arm-tag.")
        if doc and doc.get("model_config") != args.model_config:
            raise SystemExit(f"{out} holds model_config={doc.get('model_config')!r} results; "
                             f"this run builds {args.model_config!r}. Use another --arm-tag.")
        if doc and doc.get("precision", "fp16") != args.precision:
            raise SystemExit(f"{out} holds precision={doc.get('precision', 'fp16')!r} results; "
                             f"this run scores at {args.precision!r}. Use another --arm-tag.")
        if doc and args.force_depth is not None and doc.get("force_scope", "all") != args.force_scope:
            raise SystemExit(f"{out} holds force_scope={doc.get('force_scope', 'all')!r} results; "
                             f"this run forces {args.force_scope!r}. Use another --arm-tag.")
        if doc and (doc.get("assign") or {}).get("sha") != (assign or {}).get("sha"):
            raise SystemExit(f"{out} holds assign={(doc.get('assign') or {}).get('path')!r} results; "
                             f"this run assigns {(assign or {}).get('path')!r}. Use another --arm-tag.")
        if doc and (doc.get("table") or {}).get("sha") != (table or {}).get("sha"):
            raise SystemExit(f"{out} holds table={(doc.get('table') or {}).get('name')!r} results; "
                             f"this run's table is {(table or {}).get('name')!r}. Use another --arm-tag.")
        todo = [m for m in modes if m not in doc.get("modes", {})]
        need_tl = "target_last" not in doc and not args.skip_target_last
        # ⚠ A checkpoint with every mode and target_last done must still run a
        #   requested grid; without this the --grid run silently does nothing.
        gpath = out / "grid.json"
        have = json.loads(gpath.read_text())["cells"] if gpath.exists() else {}
        need_grid = args.grid and any(k not in have for k in grid_keys(args.grid_all_others))
        if not todo and not need_tl and not need_grid:
            print(f"[skip] {ck}")
            continue

        torch.manual_seed(42)
        ck_path = None if ck in ("untrained", "extended_base") else str(run_dir / ck)
        loaded = None
        if args.model_config is not None:
            if ck == "untrained":
                raise SystemExit("--model-config reads a T360-T checkpoint or `extended_base`, not `untrained`")
            from eo.scripts.eval_forgetting import build_temporal
            model, mcfg, loaded = build_temporal(args.model_config, ck_path)
            _, cfg = G.build_model(args.arm_config, None, device="cpu")          # the rows' config only
            if cfg.dataset != "terramesh_multimodal":
                raise SystemExit(f"--arm-config {args.arm_config} builds {cfg.dataset}: name the single-date arm "
                                 f"whose held-out rows are scored (T360's, eo_terramesh/arm_t360_vanilla)")
            want, have = ((c.get("multimodal") or {}).get("token_set", "original") for c in (mcfg, cfg))
            if want != have:
                raise SystemExit(f"{args.model_config} trains on token set {want!r}, {args.arm_config} scores {have!r}")
        elif ck == "extended_base":
            raise SystemExit("`extended_base` is a --model-config checkpoint name")
        else:
            model, cfg = G.build_model(args.arm_config, ck_path)
        if int(cfg.dataloader_num_workers) != args.num_workers:
            raise SystemExit(f"--num-workers {args.num_workers} != config's "
                             f"{cfg.dataloader_num_workers}: modality orders would differ from training")
        rf, base, forced = None, "router", args.force_depth is not None
        if forced:
            if any(m != "router" for m in modes):
                raise SystemExit("--force-depth runs the main pass as --modes router (or '')")
            if args.force_scope == "target" and modes:
                raise SystemExit("--force-scope target has no main pass: run it with --modes ''")
            if has_learned_router(model):
                if args.gate_seed != 0:
                    raise SystemExit("--gate-seed: a learned router's gate is its own top probability, not a draw")
                TF.install_learned_depth(model, args.force_depth)
                base = f"force{args.force_depth}"
            else:
                rf = TF.install_random_depth(model, args.force_depth)
        elif args.own_draw or (table is not None and not has_learned_router(model)):
            if has_learned_router(model):
                raise SystemExit("--own-draw: a learned router has no draw")
            rf = TF.install_random_depth(model, None)
            rf.own_draw = bool(args.own_draw)
            if table is not None:
                if table["gate"] is not None:
                    raise SystemExit("a random router's gate is its draw: a gate table does not apply")
                if table["seed"] != args.gate_seed:
                    raise SystemExit(f"a random router's table draws with --gate-seed ({args.gate_seed}); "
                                     f"the spec's seed is {table['seed']}: make them equal")
                rf.set_table(table["mix"])
        ov = TF.install_override(model, base)
        if ov is None and rf is None and any(m != "router" for m in todo):
            raise SystemExit(f"{ck}: interventions need a router; this arm has none")
        step = 0 if ck in ("untrained", "extended_base") else int(ck.split("-")[-1])
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
                    "force_depth": args.force_depth, "precision": args.precision,
                    "model_config": args.model_config, "loaded": loaded, "vocab": int(model.config.vocab_size),
                    "logged_eval": logged_eval(run_dir, step) if ck not in ("untrained", "extended_base") else None})
        if rf is not None:
            doc["trained_recursions"] = rf.n_trained
            doc["gate_seed"] = args.gate_seed
        if args.own_draw:
            doc["own_draw"] = True
        elif forced:
            doc["trained_recursions"] = int(cfg.recursive.num_recursion)
        if forced:
            doc["force_scope"] = args.force_scope
        if table is not None:
            doc["table"] = {k: table[k] for k in ("name", "spec", "sha", "seed")}
        if assign is not None:
            if assign["depth"].shape[0] != len(ds):
                raise SystemExit(f"--assign holds {assign['depth'].shape[0]} rows, the held-out set {len(ds)}")
            doc["assign"] = {k: assign[k] for k in ("path", "sha")}
        doc.setdefault("modes", {})

        for mode in todo:
            t0 = time.perf_counter()
            if ov is not None:
                if mode == "table":
                    TF.install_override(model, "table", seed=table["seed"])
                    ov.set_table(table["mix"], table["gate"])
                elif mode == "assign":
                    TF.install_override(model, "assign")
                    ov.set_assignment(assign["depth"])
                else:
                    TF.install_override(model, base if mode == "router" else mode, seed=0)
            if rf is not None:
                rf.reseed(args.gate_seed)
            res = TF.run_pass(model, loader, override=ov, dtype=PRECISION[args.precision])
            if mode == "assign" and ov.row_ptr != len(ds):
                raise SystemExit(f"{ck}: the assign pass consumed {ov.row_ptr} rows of {len(ds)}")
            tok = res.pop("tokens")
            if forced and not (tok["depth"] == args.force_depth).all():
                raise SystemExit(f"{ck}: forced depth {args.force_depth}, captured "
                                 f"{np.unique(tok['depth'], return_counts=True)}")
            keep = tok if mode == "router" else {k: tok[k] for k in ("ce", "correct", "depth", "gate_applied")
                                                  if k in tok}
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
                TF.install_override(model, base)
            corpus = load_row_table(ROOT).corpus.values
            eval_rows = load_eval_rows(root_dir=ROOT)
            tl, tl_rows = {}, {}
            for T in TARGETS:
                rows = TF.stratified_target_rows(ds, eval_rows, corpus, T, args.target_last_rows)
                dl = torch.utils.data.DataLoader(TF.TargetLast(ds, rows, T), batch_size=4,
                                                 num_workers=args.num_workers)
                if rf is not None:
                    rf.reseed(args.gate_seed)
                    rf.only_mod = TF.MODALITY_TO_ID[T] if args.force_scope == "target" else None
                if ov is not None:
                    ov.only_mod = TF.MODALITY_TO_ID[T] if args.force_scope == "target" else None
                res = TF.run_pass(model, dl, override=ov, keep_tokens=ov is not None or rf is not None,
                                  per_row_target=T, dtype=PRECISION[args.precision])
                r = {"n_rows": len(rows), **res["per_modality"][T]}
                tl_rows[f"{T}|rows"] = np.asarray(rows)
                for k, v in res["rows"].items():
                    tl_rows[f"{T}|{k}"] = v
                if ov is not None or rf is not None:
                    t = res["tokens"]
                    sel = t["label_mod"] == TF.MODALITY_TO_ID[T]
                    r["mean_depth_target"] = float(t["depth"][sel].mean())
                    ctx = (t["label_mod"] > 0) & ~sel
                    if ctx.any():
                        r["mean_depth_context"] = float(t["depth"][ctx].mean())
                    chk = t["depth"][sel] if args.force_scope == "target" else t["depth"]
                    if forced and not (chk == args.force_depth).all():
                        raise SystemExit(f"{ck} [target-last {T}]: forced depth {args.force_depth}, "
                                         f"captured {np.unique(chk, return_counts=True)}")
                tl[T] = r
                print(f"{ck} [target-last {T}] rows {len(rows)}  ce {r['ce']:.4f}  "
                      f"top1 {r['top1']:.4f}  top5 {r['top5']:.4f}  slot {r['slot_mass']:.4f}")
            tmp = out / "target_last_rows.tmp.npz"
            np.savez(tmp, **tl_rows)
            os.replace(tmp, out / "target_last_rows.npz")
            doc["target_last"] = tl
            write_json(mpath, doc)
            if ov is not None:
                ov.only_mod = None
            if rf is not None:
                rf.only_mod = None

        if need_grid:
            run_grid(model, ov, ds, load_eval_rows(root_dir=ROOT), load_row_table(ROOT).corpus.values,
                     out, doc, args, rf=rf, base=base)

        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
