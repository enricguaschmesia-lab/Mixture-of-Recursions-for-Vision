# eo/train/warm_start.py
"""Paper B intervention 4: continue a RANDOM-router run under a LEARNED router from one of its checkpoints (B1, 2026-11-15).

WHAT IT DOES. The factorial's arms C and D (Paper B plan, B2) are the cell's random-router runs continued from their
checkpoint-990 or -1980 with a learned router: C with the balancing loss, D with the compute target. Enric chose
(2026-11-15) to CONTINUE THE RUN EXACTLY rather than restart its optimizer: Adam's moments, the LR schedule and the
data order carry on, so C differs from A (the learned router from step 0) only in its routing history. And the PLAIN
gate from the switch (smoked first), and the v1 vocabulary like arm A (worklog §5).

A random-router checkpoint has no `mor_router` (the MoR layer builds it only when rand_router is false), so this writes
a COPY of the checkpoint into a new run directory that a learned-router run resumes from (`train_eo.sh --resume`):

  pytorch_model.bin   the source's tensors, every one bit for bit, plus the learned model's fresh router (its own
                      seeded init); the key sets must be exactly source + router
  optimizer.pt        the source's Adam state remapped BY PARAMETER NAME into the learned model's param groups
                      (MoRTrainer.create_optimizer's grouping: decay, no decay); the router's parameters get no state,
                      so torch's Adam initialises them (zero moments, step 0) at their first step
  scheduler.pt, scaler.pt, rng_state.pth, trainer_state.json, training_args.bin, config.json   copied as they are
  compute_target.json the controller's start (eo/train/compute_target.py), when the target config is the compute target
  warm_start.json     what was done and checked, with the source files' SHA-256

REFUSED: a source that is not a random-router MoR checkpoint, or is missing a resume file; target and source configs
that differ in anything but `mor.rand_router` (and, for the compute target, its balancing mode and block); a key or
shape that does not line up; an existing destination.

Checks with controls: gate W35 in eo/scripts/verify_phase3.py. Runs in `.venv` (torch, the MoR model).

    .venv/bin/python -m eo.train.warm_start --source <run>/checkpoint-1980 \\
        --source-config eo_terramesh/arm_a10crb3trnc_random --target-config eo_terramesh/arm_a10cb3trnc_mor \\
        --dest <runs>/phase3/<new run id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]
ROUTER = "mor_router"
#: Files a resume needs, copied as they are (pytorch_model.bin and optimizer.pt are rewritten).
COPIED = ("scheduler.pt", "scaler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin", "config.json",
          "generation_config.json")
REQUIRED = ("pytorch_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
#: The keys a target config may change against its source: the router, and the compute target's own.
ALLOWED_DIFF = {"mor.rand_router"}
COMPUTE_DIFF = {"mor.token.balancing", "mor.token.compute.target", "mor.token.compute.gain",
                "mor.token.compute.proportional", "mor.token.compute.price_init",
                "mor.token.compute.measure", "mor.token.compute.deadband"}     # the last two since 2026-11-16 (B1 Step 6)
RECORD = "warm_start.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def compose(name: str, overrides=()):
    from hydra import compose as _c, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return _c(config_name=name, overrides=list(overrides))


def _flat(cfg, prefix=""):
    from omegaconf import OmegaConf
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=False)
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


def config_diff(a, b) -> set:
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def check_configs(src_cfg, tgt_cfg) -> Dict:
    """The target continues the source: same everything but the router (and the compute target's own keys)."""
    from eo.train import compute_target as CT
    diff = config_diff(src_cfg, tgt_cfg)
    allowed = ALLOWED_DIFF | (COMPUTE_DIFF if CT.read_block(tgt_cfg) is not None else set())
    if not src_cfg.mor.rand_router or tgt_cfg.mor.rand_router:
        raise ValueError("the source config must be a random router and the target a learned one "
                         f"(rand_router {src_cfg.mor.rand_router} -> {tgt_cfg.mor.rand_router})")
    if not diff <= allowed or "mor.rand_router" not in diff:
        raise ValueError(f"the configs differ in {sorted(diff - allowed)} beyond the router "
                         f"(allowed: {sorted(allowed)}); a warm start must continue the same run")
    return {"diff": sorted(diff)}


def build(cfg_name: str, seed: int):
    """The model as pretrain.py builds it, on CPU in fp32 (eo.generate.conditional.build_model), seeded."""
    from eo.generate import conditional as G
    torch.manual_seed(seed)
    model, cfg = G.build_model(cfg_name, None, device="cpu")
    return model, cfg


def param_groups(model) -> List[List[str]]:
    """MoRTrainer.create_optimizer's two groups, as parameter names in named_parameters order."""
    from transformers import Trainer
    decay = set(Trainer.get_decay_parameter_names(None, model))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "mlp_router" not in n]
    return [[n for n, _ in named if n in decay], [n for n, _ in named if n not in decay]]


def remap_optimizer(src_opt: Dict, src_groups: List[List[str]], tgt_groups: List[List[str]],
                    tgt_shapes: Dict[str, torch.Size]) -> Tuple[Dict, Dict]:
    """The source optimizer's state dict re-indexed into the target's groups by name; router names get no state."""
    if len(src_opt["param_groups"]) != len(src_groups) or len(tgt_groups) != len(src_groups):
        raise RuntimeError(f"param groups: optimizer {len(src_opt['param_groups'])}, source {len(src_groups)}, "
                           f"target {len(tgt_groups)}")
    src_index, i = {}, 0
    for g, names in zip(src_opt["param_groups"], src_groups):
        if len(g["params"]) != len(names):
            raise RuntimeError(f"a source param group holds {len(g['params'])} tensors, the model's {len(names)}")
        for idx, n in zip(g["params"], names):
            src_index[n] = idx
            i += 1
    new_state, new_groups, j, added = {}, [], 0, []
    for g, names in zip(src_opt["param_groups"], tgt_groups):
        ng = {k: v for k, v in g.items() if k != "params"}
        ng["params"] = []
        for n in names:
            ng["params"].append(j)
            if n in src_index:
                st = src_opt["state"].get(src_index[n])
                if st is not None:
                    for k, v in st.items():
                        if torch.is_tensor(v) and v.dim() > 0 and v.shape != tgt_shapes[n]:
                            raise RuntimeError(f"{n}: optimizer {k} {tuple(v.shape)} against the model's "
                                               f"{tuple(tgt_shapes[n])}")
                    new_state[j] = st
            elif ROUTER in n:
                added.append(n)
            else:
                raise RuntimeError(f"{n} is in the target's optimizer groups but not the source's, and is not the router")
            j += 1
        new_groups.append(ng)
    missing = sorted(set(src_index) - {n for g in tgt_groups for n in g})
    if missing:
        raise RuntimeError(f"source parameters with no place in the target: {missing[:5]}")
    return {"state": new_state, "param_groups": new_groups}, {"router_params_without_state": added,
                                                              "tensors_with_state": len(new_state),
                                                              "tensors": j}


def make_copy(source: Path, source_config: str, target_config: str, dest: Path, router_seed: int = None) -> Dict:
    """Write the warm copy (see the module docstring). Returns the record also written to warm_start.json."""
    from eo.train import compute_target as CT
    source, dest = Path(source), Path(dest)
    missing = [f for f in REQUIRED if not (source / f).is_file()]
    if missing:
        raise RuntimeError(f"{source} is missing {missing}: not a resumable checkpoint")
    ck_dest = dest / source.name
    if ck_dest.exists():
        raise RuntimeError(f"{ck_dest} exists: a warm copy is written once, never over another")
    src_cfg, tgt_cfg = compose(source_config), compose(target_config)
    diff = check_configs(src_cfg, tgt_cfg)
    seed = int(tgt_cfg.seed) if router_seed is None else int(router_seed)
    state = torch.load(source / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if any(ROUTER in k for k in state) or not any(".block_list." in k for k in state):
        raise RuntimeError(f"{source}/pytorch_model.bin is not a random-router MoR checkpoint "
                           f"(mor_router present: {any(ROUTER in k for k in state)})")
    src_model, _ = build(source_config, seed)
    tgt_model, _ = build(target_config, seed)
    src_keys, tgt_sd = set(src_model.state_dict()), tgt_model.state_dict()
    router_keys = sorted(k for k in tgt_sd if ROUTER in k)
    if set(state) != src_keys:
        raise RuntimeError(f"the checkpoint's keys are not the source config's model: "
                           f"{sorted(set(state) ^ src_keys)[:5]}")
    if set(tgt_sd) != src_keys | set(router_keys) or not router_keys:
        raise RuntimeError(f"the target model is not the source plus a router: {sorted(set(tgt_sd) ^ src_keys)[:6]}")
    bad = [k for k in state if state[k].shape != tgt_sd[k].shape]
    if bad:
        raise RuntimeError(f"shapes differ: {bad[:5]}")
    new_sd = {k: state[k] for k in state}
    new_sd.update({k: tgt_sd[k].detach().clone() for k in router_keys})
    # strict load into the target proves the key set and shapes; the value check proves the values arrived
    tgt_model.load_state_dict(new_sd, strict=True)
    sd2 = tgt_model.state_dict()
    off = [k for k in state if not torch.equal(sd2[k], state[k])]
    if off:
        raise RuntimeError(f"{len(off)} source tensors changed on the way in, e.g. {off[0]}")
    src_opt = torch.load(source / "optimizer.pt", map_location="cpu", weights_only=True)
    new_opt, opt_rep = remap_optimizer(src_opt, param_groups(src_model), param_groups(tgt_model),
                                       {n: p.shape for n, p in tgt_model.named_parameters()})
    # the learned model's optimizer must take the remapped state, as HF will on the resume
    from torch.optim import AdamW
    groups = [{"params": [dict(tgt_model.named_parameters())[n] for n in names]} for names in param_groups(tgt_model)]
    AdamW(groups, lr=1e-3).load_state_dict(new_opt)

    ck_dest.mkdir(parents=True)
    torch.save(new_sd, ck_dest / "pytorch_model.bin")
    torch.save(new_opt, ck_dest / "optimizer.pt")
    for f in COPIED:
        if (source / f).is_file():
            shutil.copy2(source / f, ck_dest / f)
    step = CT.checkpoint_step(ck_dest)
    blk = CT.read_block(tgt_cfg)
    if blk is not None:
        CT.write_state(ck_dest, blk, CT.initial_state(blk), step,
                       origin=f"warm start from {source} (a random-router checkpoint)")
    rec = {"source": str(source), "source_config": source_config, "target_config": target_config,
           "global_step": step, "config_diff": diff["diff"], "router_seed": seed, "router_keys": router_keys,
           "router_params": int(sum(tgt_sd[k].numel() for k in router_keys)),
           "source_tensors_identical": len(state), "optimizer": opt_rep,
           "compute_target": blk, "gate": "plain (the learned router's own probability), Enric 2026-11-15",
           "source_sha256": {f: sha256(source / f) for f in ("pytorch_model.bin", "optimizer.pt")},
           "dest_sha256": {f: sha256(ck_dest / f) for f in ("pytorch_model.bin", "optimizer.pt")}}
    (ck_dest / RECORD).write_text(json.dumps(rec, indent=1) + "\n")
    (dest / RECORD).write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="a random-router run's checkpoint-<s> (full: optimizer etc.)")
    ap.add_argument("--source-config", required=True, help="its arm config, e.g. eo_terramesh/arm_a10crb3trnc_random")
    ap.add_argument("--target-config", required=True, help="the learned arm, e.g. eo_terramesh/arm_a10cb3trnc_mor")
    ap.add_argument("--dest", type=Path, required=True, help="the new run directory (MOR_SAVE_DIR/pretrain/phase3/<id>)")
    ap.add_argument("--router-seed", type=int, default=None, help="the fresh router's init seed (default: the target's seed)")
    a = ap.parse_args(argv)
    rec = make_copy(a.source, a.source_config, a.target_config, a.dest, a.router_seed)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("router_keys",)}, indent=1))
    print(f"wrote {a.dest / a.source.name}; resume with: train_eo.sh --resume --run-id {a.dest.name} "
          f"--config {a.target_config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
