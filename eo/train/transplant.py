# eo/train/transplant.py
"""Paper B, criterion 3: continue a RANDOM-router run under a fixed depth table from one of its checkpoints (B1 Step 8).

WHAT IT DOES. Step 9's rows retrain the cell's random-router backbones from a checkpoint (1980) to 3300, each under one
table (`mor.token.depth_table`, eo/train/depth_table.py): learned tables, their permutations, a histogram-matched table,
uniform depth 2, and the no-op (the backbone's own uniform table at another draw seed). The model does not change (a
random router has no parameters to add), so the copy is the checkpoint as it is, every file bit for bit, in a new run
directory that `train_eo.sh --resume` continues under the table's config. Adam's moments, the LR schedule, the RNG states
and the data order carry on, so every row sees the same batches: the comparison is paired.

  every resume file     copied as it is; SHA-256 equal on both sides
  transplant.json       the table it starts (rows, gate, seed), the source, both configs, their difference, the hashes;
                        preflight's depth-table check accepts a random-router checkpoint under a table only with it

REFUSED: a source that is not a random-router MoR checkpoint (a learned router's, a table run's, a warm copy's) or is
missing a resume file; a source config that is not a random router or already has a table; a target config without a
table, or one that differs from the source in anything but `mor.token.depth_table`; a model whose keys or shapes do not
take the checkpoint strictly; an existing destination.

Checks with controls: gate W37 in eo/scripts/verify_phase3.py. Runs in `.venv` (torch, the MoR model).

    .venv/bin/python -m eo.train.transplant --source <run>/checkpoint-1980 \\
        --source-config eo_terramesh/arm_a10crb3trnc_random --target-config eo_terramesh/arm_a10crb3trnct2_random \\
        --dest <runs>/phase3/<new run id>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import torch

from eo.train import depth_table as DT
from eo.train.warm_start import COPIED, REQUIRED, ROUTER, build, compose, config_diff, sha256

TABLE_PREFIX = "mor.token.depth_table"


def check_configs(src_cfg, tgt_cfg) -> Dict:
    """The target continues the source exactly, under a table: the same everything but mor.token.depth_table."""
    if not src_cfg.mor.rand_router or not tgt_cfg.mor.rand_router:
        raise ValueError(f"both configs must be random routers (rand_router {src_cfg.mor.rand_router} -> "
                         f"{tgt_cfg.mor.rand_router}): a table replaces the random draw")
    if DT.read_block(src_cfg) is not None:
        raise ValueError("the source config already has a depth table: a transplant starts from a random-router run")
    table = DT.read_block(tgt_cfg)
    if table is None:
        raise ValueError("the target config has no mor.token.depth_table")
    problems = DT.config_problems(tgt_cfg)
    if problems:
        raise ValueError("the target config: " + "; ".join(problems))
    diff = config_diff(src_cfg, tgt_cfg)
    beyond = sorted(k for k in diff if not (k == TABLE_PREFIX or k.startswith(TABLE_PREFIX + ".")))
    if beyond:
        raise ValueError(f"the configs differ in {beyond} beyond the table: a transplant must continue the same run")
    return {"diff": sorted(diff), "table": table}


def make_copy(source: Path, source_config: str, target_config: str, dest: Path) -> Dict:
    """Write the transplant's copy (see the module docstring). Returns the record also written to transplant.json."""
    from eo.train import compute_target as CT
    source, dest = Path(source), Path(dest)
    missing = [f for f in REQUIRED if not (source / f).is_file()]
    if missing:
        raise RuntimeError(f"{source} is missing {missing}: not a resumable checkpoint")
    for f, why in ((DT.FILE, "a table run's"), (DT.TRANSPLANT, "a transplant's"), ("warm_start.json", "a warm copy's"),
                   (CT.FILE, "a compute-target run's")):
        if (source / f).is_file():
            raise RuntimeError(f"{source} carries {f}: it is {why} checkpoint, not a random-router run's")
    ck_dest = dest / source.name
    if ck_dest.exists():
        raise RuntimeError(f"{ck_dest} exists: a transplant is written once, never over another")
    src_cfg, tgt_cfg = compose(source_config), compose(target_config)
    chk = check_configs(src_cfg, tgt_cfg)
    state = torch.load(source / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if any(ROUTER in k for k in state) or not any(".block_list." in k for k in state):
        raise RuntimeError(f"{source}/pytorch_model.bin is not a random-router MoR checkpoint "
                           f"(mor_router present: {any(ROUTER in k for k in state)})")
    # the target model as pretrain.py builds it must take the checkpoint strictly, and get its values
    tgt_model, _ = build(target_config, int(tgt_cfg.seed))
    tgt_model.load_state_dict(state, strict=True)
    sd2 = tgt_model.state_dict()
    off = [k for k in state if not torch.equal(sd2[k], state[k])]
    if off:
        raise RuntimeError(f"{len(off)} tensors changed on the way in, e.g. {off[0]}")
    layers = [m for m in tgt_model.modules() if getattr(m, "eo_table", None) is not None]
    if len(layers) != 1 or DT.identity(layers[0].eo_table) != DT.identity(chk["table"]):
        raise RuntimeError(f"the target model holds {len(layers)} table layer(s), not one with the config's table")
    files = ["pytorch_model.bin", "optimizer.pt"] + [f for f in COPIED if (source / f).is_file()]
    ck_dest.mkdir(parents=True)
    for f in files:
        shutil.copy2(source / f, ck_dest / f)
    src_sha = {f: sha256(source / f) for f in files}
    dst_sha = {f: sha256(ck_dest / f) for f in files}
    if src_sha != dst_sha:
        raise RuntimeError(f"copies differ: {sorted(f for f in files if src_sha[f] != dst_sha[f])}")
    step = CT.checkpoint_step(ck_dest)
    rec = {"version": 1, "table": DT.identity(chk["table"]), "source": str(source), "source_config": source_config,
           "target_config": target_config, "global_step": step, "config_diff": chk["diff"],
           "tensors_identical": len(state), "files": files, "sha256": src_sha}
    (ck_dest / DT.TRANSPLANT).write_text(json.dumps(rec, indent=1) + "\n")
    (dest / DT.TRANSPLANT).write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="a random-router run's checkpoint-<s> (full: optimizer etc.)")
    ap.add_argument("--source-config", required=True, help="its arm config, e.g. eo_terramesh/arm_a10crb3trnc_random")
    ap.add_argument("--target-config", required=True, help="the same arm with mor.token.depth_table")
    ap.add_argument("--dest", type=Path, required=True, help="the new run directory (MOR_SAVE_DIR/pretrain/phase3/<id>)")
    a = ap.parse_args(argv)
    rec = make_copy(a.source, a.source_config, a.target_config, a.dest)
    print(json.dumps({k: v for k, v in rec.items() if k != "sha256"}, indent=1))
    print(f"wrote {a.dest / a.source.name}; resume with: train_eo.sh --resume --run-id {a.dest.name} "
          f"--config {a.target_config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
