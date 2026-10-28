# eo/train/preflight_controls.py
"""Prove every preflight check can FAIL. Rule 2, applied to the checks themselves.

A check that has never been observed to fail is not evidence of anything -- it
may be testing nothing at all. Phase 2 learned this the expensive way: an
import-time assert on the vocabulary layout looked like a guard and was
vacuous, because the layout is *derived* from the sizes it claimed to check.
Mutation is what exposed it.

So each check here gets a deliberately-broken environment that must trip it,
and the run dir also gets a positive control (`--force` must still be allowed),
because a check that fires on everything is as useless as one that never fires.

    python -m eo.train.preflight_controls

Exit code is 0 only if the baseline passes and every control fires. Step 9's
`verify_phase3.py` calls this as its preflight section rather than restating it.

⚠ `check_torch=False` throughout: these controls are about the CHECKS, and
initializing CUDA per case would be slow and would couple them to whichever GPU
happens to be free. The two checks that need real CUDA -- wrong card, and both
cards visible -- are exercised by `verify_phase3.py` in a subprocess.

Multi-GPU (2026-10-20, `--gpus N`): the visible-device count, the scope (RCP,
pretraining only), the batch and eval division, a resume at another GPU split or
per-device batch, `resolve_n` on fake inventories, and `distributed.check_launch`
(the DataParallel guard in pretrain.py), each with a case that must fire.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eo.train.preflight import PreflightContext, run_checks

REPO = Path(__file__).resolve().parent.parent.parent
CONFIG = REPO / "conf/pretrain_vision/eo_terramesh/terramesh_mor_token.yaml"

#: A known-good environment, independent of whatever the caller's shell says --
#: otherwise a control could "fire" because the ambient environment was already
#: broken, which proves nothing.
GOOD_ENV: Dict[str, str] = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "1",
    "MOR_SAVE_DIR": "/data/enric/runs",
    "TERRAMESH_TOK_ROOT": "/data/enric/data/TerraMesh/val",
    "HF_HOME": "/data/enric/hf",
    "WANDB_ENTITY": "enricguasch-epfl",
    "WANDB_PROJECT": "mor-eo-phase3",
    "WANDB_MODE": "online",
    "WANDB_DIR": "/data/enric/runs/wandb",
}


def _env(drop: Tuple[str, ...] = (), **override) -> Dict[str, str]:
    e = dict(GOOD_ENV)
    for k in drop:
        e.pop(k, None)
    e.update(override)
    return e


def _fake_checkpoint(parent: Path, arm: str, *, drop: Tuple[str, ...] = (),
                     step: int = 20, world: int = 1, ranks: Optional[Tuple[int, ...]] = None,
                     per_device: Optional[int] = 4) -> Path:
    """A minimal checkpoint with a real arm fingerprint, built from 2x2 tensors.

    Synthetic on purpose: the controls must be runnable on any machine, years
    from now, without depending on run artifacts that were cleaned up. The
    fingerprint only looks at key names and at whether two shared layers are the
    same tensor, so tiny tensors reproduce it exactly.
    """
    import torch
    d = parent / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)

    base_depth = (29 - 2) // 3          # middle_cycle, num_recursion=3 -> layers 1 and 10
    a = torch.eye(2)
    if arm == "random":                 # the MoR transform's names, no learned router (real: 264 keys)
        sd = {"model.embed_tokens.weight": torch.zeros(2, 2),
              "model.layers.1.block_list.0.0.self_attn.q_proj.weight": a}
    else:
        sd = {
            "model.embed_tokens.weight": torch.zeros(2, 2),
            "model.layers.1.self_attn.q_proj.weight": a,
            # recursive sharing means THE SAME TENSOR, not merely equal values
            f"model.layers.{1 + base_depth}.self_attn.q_proj.weight":
                a if arm in ("recursive", "mor") else torch.ones(2, 2),
        }
    if arm == "mor":
        sd["model.layers.1.mor_router.router.0.weight"] = torch.zeros(2, 2)
    torch.save(sd, d / "pytorch_model.bin")

    # One GPU writes rng_state.pth, N write rng_state_<rank>.pth; trainer_state.json
    # records the per-device batch HF restores on resume (default config: 4).
    rng = (["rng_state.pth"] if world == 1 else
           [f"rng_state_{r}.pth" for r in (ranks if ranks is not None else range(world))])
    state = json.dumps({"train_batch_size": per_device}) if per_device is not None else "{}"
    for f in ["optimizer.pt", "scheduler.pt", "trainer_state.json"] + rng:
        if f not in drop:
            (d / f).write_text(state if f == "trainer_state.json" else "{}")
    if "pytorch_model.bin" in drop:
        (d / "pytorch_model.bin").unlink()
    return d


def _fake_conversion(d: Path, variant: str = "clevr", drop_json: bool = False, **doc_over) -> Path:
    """A CLEVR conversion in miniature: three tensors, and a conversion.json that passed.

    Synthetic for the same reason as _fake_checkpoint. verify_conversion reads the record and
    the weights' hash, and check_handoff compares tensors by name, so 4x2 tensors suffice.
    """
    import torch
    from safetensors.torch import save_file
    from eo.train.clevr_init import sha256
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    save_file({"model.embed_tokens.weight": torch.randn(4, 2, generator=g) * 0.02,
               "model.layers.0.mlp.down_proj.weight": torch.randn(2, 2, generator=g),
               "model.norm.weight": torch.ones(2)}, str(d / "model.safetensors"))
    doc = {"variant": variant, "vocab_size": 87556, "weights_file": "model.safetensors",
           "weights_sha256": sha256(d / "model.safetensors"),
           "checks": {"body_identity": {"passed": True}}, "controls": {"body_identity:shuffled_body": {"fired": True}}}
    doc.update(doc_over)
    if not drop_json:
        (d / "conversion.json").write_text(json.dumps(doc))
    return d


def _fake_stage1(run: Path, conv: Path, *, step: int = 330, last: int = 330, move_body: bool = False,
                 move_table: bool = True, record: Optional[dict] = None) -> Path:
    """Stage 1's checkpoint in miniature, plus the run's pretrain_init.json (`record` overrides it)."""
    from safetensors.torch import load_file, save_file
    from eo.train.clevr_init import sha256
    s = load_file(str(conv / "model.safetensors"))
    if move_table:
        s["model.embed_tokens.weight"] = s["model.embed_tokens.weight"] + 1e-3
    if move_body:              # one ulp-scale step in one tensor: the comparison is bitwise
        s["model.layers.0.mlp.down_proj.weight"] = s["model.layers.0.mlp.down_proj.weight"] * (1 + 1e-6)
    ck = run / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    save_file(s, str(ck / "model.safetensors"))
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": last}))
    rec = {"conversion_weights_sha256": sha256(conv / "model.safetensors"), "variant": "clevr",
           "freeze_body": True, "from_stage1": None}
    rec.update(record or {})
    (run / "pretrain_init.json").write_text(json.dumps(rec))
    return ck


def _results(env, *, run_dir: Optional[Path] = None, config: Optional[Path] = None,
             allow_existing: bool = False, resuming: bool = False,
             overrides: Optional[dict] = None, check_torch: bool = False,
             min_free_gpu_mib: Optional[int] = None, gpu_alias: str = "titanv",
             n_gpus: int = 1) -> Dict[str, bool]:
    ctx = PreflightContext(
        gpu_alias=gpu_alias,
        run_dir=run_dir or (Path(tempfile.mkdtemp()) / "fresh"),
        config_path=CONFIG if config is None else config,
        allow_existing=allow_existing,
        resuming=resuming,
        overrides=overrides or {},
        env=env,
        check_torch=check_torch,
        min_free_gpu_mib=min_free_gpu_mib,
        n_gpus=n_gpus,
    )
    results = run_checks(ctx)
    out = _Res({r.name: r.ok for r in results})
    out.detail = {r.name: r.detail for r in results}
    return out


class _Res(dict):
    """{check name: ok}, plus `.detail` so a control can show WHY it fired."""
    detail: Dict[str, str]


def _broken_config(tmp: Path, **overrides) -> Path:
    """A copy of the shipped EO config with specific keys sabotaged."""
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    for dotted, value in overrides.items():
        keys = dotted.split("__")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    # ⚠ Numbered, not named by the keys alone: cases are built before any is
    # evaluated, so until 2026-10-11 the four eval-split cases (same key,
    # different values) all wrote one file and all tested the last value.
    n = len(list(tmp.glob("broken_*.yaml")))
    out = tmp / (f"broken_{n:03d}_" + "_".join(overrides) + ".yaml")
    out.write_text(yaml.safe_dump(cfg))
    return out


def _fake_train_root(d: Path, fault: str = "ok") -> Path:
    """A 4-row train root (2 majortom, 2 ssl4eos12) with val's real metadata.json, and one fault."""
    import numpy as np
    import pandas as pd
    from eo.terramesh_tok.contract import tok_dir_name
    from eo.data.eo_vocab import DEFAULT_ACTIVE_MODALITIES
    val = Path(GOOD_ENV["TERRAMESH_TOK_ROOT"])
    n = 4
    d.mkdir(parents=True)
    pd.DataFrame({"row": range(n), "stem": [f"majortom_train_{i}" for i in (1, 2)] + [f"ssl4eos12_train_{i}" for i in (3, 4)],
                  "source_shard": ["majortom_shard_000001.tar"] * 2 + ["ssl4eos12_shard_000001.tar"] * 2,
                  "corpus": ["majortom"] * 2 + ["ssl4eos12"] * 2}).to_parquet(d / "tok_index.parquet", index=False)
    for m in DEFAULT_ACTIVE_MODALITIES:
        md = d / tok_dir_name(m)
        md.mkdir()
        rows = n - 1 if fault == "short" and m == "DEM" else n
        np.save(md / "tokens.npy", np.zeros((rows, 3 if m == "Coords" else 196), dtype=np.uint16))
        p = np.ones(rows, dtype=bool)
        if m == "S1RTC":
            p[2:] = fault == "s1"                  # S1RTC on majortom only (both rows when s1 is broken)
        if m == "S1GRD":
            p[:2] = False
        np.save(md / "present.npy", p)
        meta = json.loads((val / tok_dir_name(m) / "metadata.json").read_text())
        if fault == "contract" and m == "NDVI":
            meta["standardization"] = {**meta["standardization"], "mean": [0.0]}
        (md / "metadata.json").write_text(json.dumps(meta))
    if fault != "nobuild":
        (d / "build.json").write_text(json.dumps({
            "split": "train", "offset_free": fault != "notfree", "partial": fault == "partial", "shards": 2}))
    return d


def _mutated_split(tmp: Path, name: str, mutate) -> Path:
    """The committed split with one field sabotaged, written beside it."""
    import json
    from eo.data.eval_split import default_split_path
    doc = json.loads(default_split_path().read_text())
    mutate(doc)
    out = tmp / f"eval_rows_{name}.json"
    out.write_text(json.dumps(doc))
    return out


def _stale_split(tmp: Path) -> Path:
    """Built against a different tok_index.parquet. The row indices still
    resolve, which is exactly why this must be caught by the hash."""
    return _mutated_split(tmp, "stale",
                          lambda d: d["inputs"].__setitem__("tok_index.parquet", "0" * 64))


def _duplicated_split(tmp: Path) -> Path:
    def m(d):
        d["eval_rows"] = d["eval_rows"] + [d["eval_rows"][0]]
        d["n_eval_rows"] = len(d["eval_rows"])
    return _mutated_split(tmp, "dup", m)


def _miscounted_split(tmp: Path) -> Path:
    return _mutated_split(tmp, "miscount",
                          lambda d: d.__setitem__("n_eval_rows", d["n_eval_rows"] + 1))


def main(argv=None) -> int:
    tmp = Path(tempfile.mkdtemp())
    occupied = tmp / "occupied"
    occupied.mkdir()
    (occupied / "checkpoint-1").touch()

    baseline = _results(GOOD_ENV)
    print("baseline (every check must pass):")
    bad = [k for k, v in baseline.items() if not v]
    print("  " + ("all pass" if not bad else f"BROKEN BASELINE -- failing: {bad}"))
    if bad:
        print("\nControls are meaningless against a broken baseline. Fix the environment first.")
        return 1

    # (label, target check, kwargs for _results)
    cases: List[Tuple[str, str, dict]] = [
        ("CUDA_DEVICE_ORDER unset",             "device order",       dict(env=_env(("CUDA_DEVICE_ORDER",)))),
        ("CUDA_DEVICE_ORDER=FASTEST_FIRST",     "device order",       dict(env=_env(CUDA_DEVICE_ORDER="FASTEST_FIRST"))),
        ("both GPUs visible",                   "visible devices",    dict(env=_env(CUDA_VISIBLE_DEVICES="0,1"))),
        ("CUDA_VISIBLE_DEVICES unset",          "visible devices",    dict(env=_env(("CUDA_VISIBLE_DEVICES",)))),
        ("--gpus 2, one card visible",          "visible devices",    dict(env=_env(CUDA_VISIBLE_DEVICES="0"), n_gpus=2)),
        ("--gpus 0",                            "multi-gpu",          dict(env=GOOD_ENV, n_gpus=0)),
        ("--gpus 2 on the workstation (titanv)", "multi-gpu",         dict(env=_env(CUDA_VISIBLE_DEVICES="0,1"), n_gpus=2)),
        # preprocess_config ROUNDS the accumulation: 250 / 4 -> 62 -> batch 248
        ("batch: 250 on one GPU x 4 (-> 248)",  "batch division",     dict(env=GOOD_ENV, overrides={"total_batch_size": 250})),
        ("MOR_SAVE_DIR unset",                  "MOR_SAVE_DIR",       dict(env=_env(("MOR_SAVE_DIR",)))),
        ("MOR_SAVE_DIR on /home",               "MOR_SAVE_DIR",       dict(env=_env(MOR_SAVE_DIR="/home/enric/results"))),
        ("MOR_SAVE_DIR relative",               "MOR_SAVE_DIR",       dict(env=_env(MOR_SAVE_DIR="results"))),
        ("TERRAMESH_TOK_ROOT unset",            "TERRAMESH_TOK_ROOT", dict(env=_env(("TERRAMESH_TOK_ROOT",)))),
        ("TERRAMESH_TOK_ROOT lacks _tok224",    "TERRAMESH_TOK_ROOT", dict(env=_env(TERRAMESH_TOK_ROOT="/data/enric/data/TerraMesh"))),
        ("HF_HOME unset",                       "HF_HOME",            dict(env=_env(("HF_HOME",)))),
        ("HF_HOME on /home",                    "HF_HOME",            dict(env=_env(HF_HOME="/home/enric/hf_cache"))),
        ("WANDB_ENTITY unset",                  "W&B",                dict(env=_env(("WANDB_ENTITY",)))),
        ("WANDB_PROJECT unset",                 "W&B",                dict(env=_env(("WANDB_PROJECT",)))),
        ("WANDB_DIR unset (leaks into repo)",   "W&B",                dict(env=_env(("WANDB_DIR",)))),
        ("WANDB_DIR on /home",                  "W&B",                dict(env=_env(WANDB_DIR="/home/enric/wandb"))),
        ("non-empty run dir, no --force",       "run dir",            dict(env=GOOD_ENV, run_dir=occupied)),
        ("config: CLEVR's vocab_size",          "config",             dict(env=GOOD_ENV, config=_broken_config(tmp, model_config__vocab_size=242271))),
        ("config: non-null deepspeed",          "config",             dict(env=GOOD_ENV, config=_broken_config(tmp, deepspeed="ds_configs/stage2.config"))),
        ("config: vision_eval enabled",         "config",             dict(env=GOOD_ENV, config=_broken_config(tmp, vision_eval__enable=True))),
        ("config: wrong dataset",               "config",             dict(env=GOOD_ENV, config=_broken_config(tmp, dataset="clevr_multimodal"))),
        ("config: unsupported precision",       "config",             dict(env=GOOD_ENV, config=_broken_config(tmp, precision="int8"))),
        ("config: missing file",                "config",             dict(env=GOOD_ENV, config=tmp / "nope.yaml")),
    ]

    # --- the held-out eval split (Step 1). A stale split is the worst of these
    # to miss: the row indices still resolve, so the run trains on rows it
    # reports as held out and every eval number is a training number. ---
    cases += [
        ("eval split: names a missing file", "eval split",
         dict(env=GOOD_ENV, config=_broken_config(tmp, multimodal__eval_split="does_not_exist.json"))),
        ("eval split: stale input hash", "eval split",
         dict(env=GOOD_ENV, config=_broken_config(tmp, multimodal__eval_split=str(_stale_split(tmp))))),
        ("eval split: duplicate rows", "eval split",
         dict(env=GOOD_ENV, config=_broken_config(tmp, multimodal__eval_split=str(_duplicated_split(tmp))))),
        ("eval split: count disagrees with list", "eval split",
         dict(env=GOOD_ENV, config=_broken_config(tmp, multimodal__eval_split=str(_miscounted_split(tmp))))),
        # ⚠ The workstation is shared with other projects. On 2026-09-23 an
        # unrelated job held 9.58 GiB of the TITAN V and a launch OOM'd at
        # step 1; nothing else in preflight would have caught it. Demanding
        # more memory than any card has proves the check can fail. Uses
        # nvidia-smi only, so it needs no CUDA context.
        ("gpu free memory: demand more than exists", "gpu free memory",
         dict(env=GOOD_ENV, check_torch=True, min_free_gpu_mib=10**7)),
    ]

    # --- resume checkpoint (Step 0.4). Every one of these was measured to be
    # SILENTLY ACCEPTED before this check existed: exit 0, "Continuing training
    # from global step N", and a model that was partly or wholly random. ---
    ARMS = {"vanilla": {"recursive.enable": False, "mor.enable": False},
            "recursive": {"recursive.enable": True, "mor.enable": False},
            "mor": {"recursive.enable": True, "mor.enable": True},
            "random": {"recursive.enable": True, "mor.enable": True, "mor.rand_router": True}}
    ck = {arm: (tmp / f"run_{arm}") for arm in ARMS}
    for arm, d in ck.items():
        _fake_checkpoint(d, arm)

    def resume_case(ckpt_arm, run_arm):
        return dict(env=GOOD_ENV, run_dir=ck[ckpt_arm], resuming=True,
                    overrides=ARMS[run_arm], allow_existing=True)

    for missing in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        d = tmp / f"run_missing_{missing}"
        _fake_checkpoint(d, "mor", drop=(missing,))
        cases.append((f"resume: {missing} missing", "resume checkpoint",
                      dict(env=GOOD_ENV, run_dir=d, resuming=True,
                           overrides=ARMS["mor"], allow_existing=True)))
    d = tmp / "run_no_weights"
    _fake_checkpoint(d, "mor", drop=("pytorch_model.bin",))
    cases.append(("resume: no model weights", "resume checkpoint",
                  dict(env=GOOD_ENV, run_dir=d, resuming=True,
                       overrides=ARMS["mor"], allow_existing=True)))
    d = tmp / "run_no_ckpt"; d.mkdir()
    (d / "stray.txt").write_text("x")
    cases.append(("resume: no checkpoint-* at all", "resume checkpoint",
                  dict(env=GOOD_ENV, run_dir=d, resuming=True,
                       overrides=ARMS["mor"], allow_existing=True)))
    for ckpt_arm, run_arm in [("vanilla", "mor"), ("recursive", "mor"), ("mor", "vanilla"),
                              ("mor", "recursive"), ("vanilla", "recursive"),
                              ("recursive", "vanilla"), ("random", "mor"), ("mor", "random"),
                              ("random", "vanilla"), ("vanilla", "random")]:
        cases.append((f"resume: {ckpt_arm} ckpt as {run_arm} run", "resume checkpoint",
                      resume_case(ckpt_arm, run_arm)))

    # --- remote site (RCP A100, 2026-10-11). Large outputs must live under the
    # private root named by MOR_EO_RCP, and that variable must never relax the
    # workstation's /data guard. ---
    rcp = tmp / "rcp"
    rcp_env = _env(MOR_EO_RCP=str(rcp), MOR_SAVE_DIR=str(rcp / "runs"), HF_HOME=str(rcp / "hf"),
                   WANDB_DIR=str(rcp / "runs/wandb"), WANDB_MODE="offline")
    rcp_base = _results(rcp_env, gpu_alias="a100")
    if not all(rcp_base.values()):
        print(f"  BROKEN a100 BASELINE -- failing: {[k for k, v in rcp_base.items() if not v]}")
        return 1
    cases += [
        ("a100: MOR_EO_RCP unset",              "MOR_SAVE_DIR", dict(env={k: v for k, v in rcp_env.items() if k != "MOR_EO_RCP"}, gpu_alias="a100")),
        ("a100: MOR_EO_RCP relative",           "MOR_SAVE_DIR", dict(env={**rcp_env, "MOR_EO_RCP": "rcp"}, gpu_alias="a100")),
        ("a100: MOR_SAVE_DIR outside RCP root", "MOR_SAVE_DIR", dict(env={**rcp_env, "MOR_SAVE_DIR": "/data/enric/runs"}, gpu_alias="a100")),
        ("a100: HF_HOME outside RCP root",      "HF_HOME",      dict(env={**rcp_env, "HF_HOME": "/data/enric/hf"}, gpu_alias="a100")),
        ("a100: WANDB_DIR outside RCP root",    "W&B",          dict(env={**rcp_env, "WANDB_DIR": "/data/enric/runs/wandb"}, gpu_alias="a100")),
        ("titanv: MOR_EO_RCP cannot relax /data", "MOR_SAVE_DIR", dict(env={**rcp_env}, gpu_alias="titanv")),
        ("MOR_SAVE_DIR on /data2 (not /data)",  "MOR_SAVE_DIR", dict(env=_env(MOR_SAVE_DIR="/data2/runs"))),
    ]

    # --- multi-GPU (2026-10-20, --gpus N): an RCP pod with N identical cards. Each
    # case is a way a DDP run trains a different recipe, or resumes wrongly, with
    # exit 0: a rounded batch, a padded eval, a resume that restores no RNG state or
    # changes the total batch, a fine-tune nobody verified across processes. ---
    rcp2 = {**rcp_env, "CUDA_VISIBLE_DEVICES": "0,1"}
    rcp3 = {**rcp_env, "CUDA_VISIBLE_DEVICES": "0,1,2"}
    on_a100 = dict(gpu_alias="a100")
    ben_cfg = REPO / "conf/pretrain_vision/eo_ben/b10_lora.yaml"
    resume_kw = dict(resuming=True, overrides=ARMS["vanilla"], allow_existing=True)
    ck2 = tmp / "run_world_2"
    _fake_checkpoint(ck2, "vanilla", world=2)
    ck_gap = tmp / "run_world_gap"
    _fake_checkpoint(ck_gap, "vanilla", world=2, ranks=(0, 2))
    ck_both = tmp / "run_world_both"
    _fake_checkpoint(ck_both, "vanilla")
    (ck_both / "checkpoint-20" / "rng_state_0.pth").write_text("{}")
    cases += [
        ("--gpus 2 on a BEN fine-tune",          "multi-gpu",         dict(env=rcp2, n_gpus=2, config=ben_cfg, **on_a100)),
        ("batch: 256 on 3 GPUs x 16 (-> 240)",   "batch division",    dict(env=rcp3, n_gpus=3, overrides={"per_device_train_batch_size": 16}, **on_a100)),
        ("batch: 256 on 3 GPUs x 4 (-> 252)",    "batch division",    dict(env=rcp3, n_gpus=3, **on_a100)),
        ("eval: 4416 rows over 2 GPUs x 5",      "batch division",    dict(env=rcp2, n_gpus=2, overrides={"per_device_eval_batch_size": 5}, **on_a100)),
        ("resume: 2-GPU ckpt on 1 GPU",          "resume checkpoint", dict(env=rcp_env, n_gpus=1, run_dir=ck2, **on_a100, **resume_kw)),
        ("resume: 1-GPU ckpt on 2 GPUs",         "resume checkpoint", dict(env=rcp2, n_gpus=2, run_dir=ck["vanilla"], **on_a100, **resume_kw)),
        ("resume: rng ranks 0, 2 (1 lost)",      "resume checkpoint", dict(env=rcp2, n_gpus=2, run_dir=ck_gap, **on_a100, **resume_kw)),
        ("resume: rng_state.pth and _0.pth",     "resume checkpoint", dict(env=rcp_env, n_gpus=1, run_dir=ck_both, **on_a100, **resume_kw)),
        ("resume: per-device 4 ckpt, config 16", "resume checkpoint",
         dict(env=GOOD_ENV, run_dir=ck["vanilla"], resuming=True, allow_existing=True,
              overrides={**ARMS["vanilla"], "per_device_train_batch_size": 16})),
    ]

    # --- token set (2026-10-11). The config must be the record of which tokens a
    # run saw, so a cleaned root is reachable only through multimodal.token_set,
    # and only once its build gate passed. Fake sites: a `val` symlink to the real
    # root, beside a cleaned root that is missing, failed, or empty. ---
    real_val = Path(GOOD_ENV["TERRAMESH_TOK_ROOT"])
    sites = {}
    for name, gate in (("nogate", None), ("failed", {"pass": False}), ("empty", {"pass": True})):
        site = tmp / f"tm_{name}"
        site.mkdir()
        (site / "val").symlink_to(real_val)
        if gate is not None:
            from eo.terramesh_tok.clean import CLEAN_ROOT_NAME, GATE_FILE
            (site / CLEAN_ROOT_NAME).mkdir()
            (site / CLEAN_ROOT_NAME / GATE_FILE).write_text(__import__("json").dumps(gate))
        sites[name] = str(site / "val")
    clean_cfg = _broken_config(tmp, multimodal__token_set="clean")
    cases += [
        ("token set: unknown value",            "token set", dict(env=GOOD_ENV, config=_broken_config(tmp, multimodal__token_set="cleaned"))),
        ("token set: root already val_clean",   "token set", dict(env=_env(TERRAMESH_TOK_ROOT=str(real_val.parent / "val_clean")))),
        ("token set: clean, root never built",  "token set", dict(env=_env(TERRAMESH_TOK_ROOT=sites["nogate"]), config=clean_cfg)),
        ("token set: clean, gate failed",       "token set", dict(env=_env(TERRAMESH_TOK_ROOT=sites["failed"]), config=clean_cfg)),
        ("token set: clean, gated but empty",   "token set", dict(env=_env(TERRAMESH_TOK_ROOT=sites["empty"]), config=clean_cfg)),
    ]
    arm_cfg = REPO / "conf/pretrain_vision/eo_terramesh"

    # --- the TRAIN split (2026-10-25): `multimodal.split: train` reads its own consolidated root,
    # gated by its build.json, row-aligned, S1 split exactly, under val's contract. Fake 4-row
    # roots (2 majortom + 2 ssl4eos12) with val's real metadata.json, then one fault each. ---
    train_cfg = _broken_config(tmp, multimodal__split="train", multimodal__token_set="clean")
    troots = {name: _fake_train_root(tmp / f"tr_{name}", fault=name) for name in (
        "ok", "nobuild", "notfree", "partial", "s1", "contract", "short")}
    tenv = {k: _env(TERRAMESH_TRAIN_TOK_ROOT=str(r)) for k, r in troots.items()}
    cases += [
        ("train root: TERRAMESH_TRAIN_TOK_ROOT unset", "train root", dict(env=GOOD_ENV, config=train_cfg)),
        ("train root: no build.json (not built)",      "train root", dict(env=tenv["nobuild"], config=train_cfg)),
        ("train root: offset scan did not pass",       "train root", dict(env=tenv["notfree"], config=train_cfg)),
        ("train root: partial, not allowed",           "train root", dict(env=tenv["partial"], config=train_cfg)),
        ("train root: paired with original val",       "train root",
         dict(env=tenv["ok"], config=_broken_config(tmp, multimodal__split="train"))),
        ("train root: S1RTC + S1GRD != rows",          "train root", dict(env=tenv["s1"], config=train_cfg)),
        ("train root: another tokenizer contract",     "train root", dict(env=tenv["contract"], config=train_cfg)),
        ("train root: arrays shorter than the index",  "train root", dict(env=tenv["short"], config=train_cfg)),
    ]

    # --- Phase 6: a BEN fine-tune starts from an EXTENDED checkpoint whose identity check
    # passed, and reads BEN tokens that exist for every row it uses (2026-10-12). Fake
    # extension dirs symlink the real weights and doctor extension.json; skipped (and said
    # so) where the real extended checkpoint or the BEN tokens are absent.
    ben_full = REPO / "conf/pretrain_vision/eo_ben/b10_full.yaml"
    ext = Path("/data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300")
    ben_cases, ben_pos = [], []
    if (ext / "extension.json").is_file() and Path("/data/enric/data/BigEarthNet/S2L2A_tok224/present.npy").is_file():
        import json as _json

        def fake_ext(name, mutate=None, drop_json=False):
            d = tmp / f"ext_{name}"
            d.mkdir()
            for f in ext.iterdir():
                if f.name != "extension.json":
                    (d / f.name).symlink_to(f)
            if not drop_json:
                doc = _json.loads((ext / "extension.json").read_text())
                if mutate:
                    mutate(doc)
                (d / "extension.json").write_text(_json.dumps(doc))
            return str(d)

        def ben(init, **kw):
            return dict(env=GOOD_ENV, config=ben_full, overrides={"finetune.init_from": init, **kw.pop("ov", {})}, **kw)
        ben_cases = [
            ("BEN: extension.json missing",      "init from", ben(fake_ext("nojson", drop_json=True))),
            ("BEN: weights changed after check", "init from", ben(fake_ext("sha", lambda d: d.__setitem__("weights_sha256", "0" * 64)))),
            ("BEN: identity check failed",       "init from", ben(fake_ext("ident", lambda d: d["identity"].__setitem__("passed", False)))),
            ("BEN: a control did not fire",      "init from", ben(fake_ext("ctrl", lambda d: d["control_untied_resize"].__setitem__("fired", False)))),
            ("BEN: no init_from at all",         "init from", dict(env=GOOD_ENV, config=ben_full)),
            ("BEN: --resume of a fine-tune",     "init from", ben(str(ext), resuming=True, allow_existing=True)),
            ("BEN: EO vocab size on a BEN run",  "config",    ben(str(ext), ov={"model_config.vocab_size": 87556})),
            ("BEN: eval role never tokenized",   "BEN data",  ben(str(ext), ov={"ben.eval_role": "sm_train"})),
            ("BEN: role not in the index",       "BEN data",  ben(str(ext), ov={"ben.train_role": "nope"})),
            ("BEN: scratch with an init_from",   "init from", dict(env=GOOD_ENV, config=REPO / "conf/pretrain_vision/eo_ben/b10_scratch.yaml",
                                                               overrides={"finetune.init_from": str(ext)})),
            # PHASE6_PLAN §2c (2026-10-18): L11b's pool must carry no label; L10 needs its generated S2
            ("BEN: distillation pool labelled",  "BEN data",  ben(str(ext), ov={"ben.unlab.role": "ft_train"})
                                                              | {"config": REPO / "conf/pretrain_vision/eo_ben/b10_lora_l11b.yaml"}),
            ("BEN: teacher blocks missing",      "BEN data",  ben(str(ext), ov={"ben.unlab.teacher": "teacher/nope"})
                                                              | {"config": REPO / "conf/pretrain_vision/eo_ben/b10_lora_l11b.yaml"}),
            ("BEN: S1G2 without generated S2",   "BEN data",  ben(str(ext), ov={"ben.gen_s2": "nope_gen"})
                                                              | {"config": REPO / "conf/pretrain_vision/eo_ben/b10_lora_l10.yaml"}),
        ]
        ben_pos = [
            (f"BEN: {c} from the extended B10", chk, _results(**ben(str(ext)) | {"config": REPO / f"conf/pretrain_vision/eo_ben/{c}.yaml"}))
            for c in ("b10_full", "b10_lora", "b10_head", "b10_lora_lr1e4") for chk in ("config", "BEN data", "init from")
        ] + [
            ("BEN: b10_scratch, no init_from", chk, _results(env=GOOD_ENV, config=REPO / "conf/pretrain_vision/eo_ben/b10_scratch.yaml"))
            for chk in ("config", "BEN data", "init from")
        ]
        # the L11b / L10 overlays pass once their artifacts exist (teacher blocks; generated S2)
        bn = Path("/data/enric/data/BigEarthNet")
        for c, art in (("b10_lora_l11b", bn / "teacher/l11b_b10lora1461_S1S2_K4.json"),
                       ("b10_lora_l10", bn / "S2L2A_gen_l10/metadata.json")):
            if art.is_file():
                ben_pos += [(f"BEN: {c} from the extended B10", chk,
                             _results(**ben(str(ext)) | {"config": REPO / f"conf/pretrain_vision/eo_ben/{c}.yaml"}))
                            for chk in ("config", "BEN data", "init from")]
            else:
                print(f"\n({c} positive controls skipped: {art} not built yet)")
    else:
        print("\n(Phase 6 controls skipped: no extended checkpoint or BEN tokens on this machine)")
    cases += ben_cases

    # --- the CLEVR-initialised vanilla (2026-10-27, eo/train/clevr_init.py). Each of these
    # trains happily: an unverified init, a stage 2 whose from_stage1 was forgotten (the
    # mandatory ??? reads as None through .get), a stage 2 from a stage 1 whose body moved
    # or that never finished, a resume under another init. Miniature fakes throughout. ---
    eo_cfg = REPO / "conf/pretrain_vision/eo_terramesh"
    s1_cfg, s2_cfg = eo_cfg / "arm_b10cv_stage1.yaml", eo_cfg / "arm_b10cv_vanilla.yaml"
    conv = _fake_conversion(tmp / "ci_conv")
    conv_shuf = _fake_conversion(tmp / "ci_conv_shuf", variant="shuffled")
    good_s1 = _fake_stage1(tmp / "ci_s1_good", conv)

    def ci(config, **ov):
        kw = {k: ov.pop(k) for k in ("run_dir", "resuming") if k in ov}
        if kw.get("resuming"):
            kw["allow_existing"] = True
        return dict(env=GOOD_ENV, config=config,
                    overrides={"pretrain_init.conversion": str(ov.pop("conversion", conv)), **ov}, **kw)
    stage2 = lambda ck, **ov: ci(s2_cfg, **{"pretrain_init.from_stage1": str(ck)}, **ov)
    ci_cases = [
        ("CLEVR init: on a MoR arm",              ci(s1_cfg, **{"recursive.enable": True, "mor.enable": True})),
        ("CLEVR init: conversion.json missing",   ci(s1_cfg, conversion=_fake_conversion(tmp / "ci_nojson", drop_json=True))),
        ("CLEVR init: weights changed after check", ci(s1_cfg, conversion=_fake_conversion(tmp / "ci_sha", weights_sha256="0" * 64))),
        ("CLEVR init: a conversion check failed", ci(s1_cfg, conversion=_fake_conversion(
            tmp / "ci_fail", checks={"body_identity": {"passed": False}}))),
        ("CLEVR init: variant is not the conversion's", ci(s1_cfg, conversion=conv_shuf)),
        ("CLEVR init: unknown key (typo of freeze_body)", ci(s1_cfg, **{"+pretrain_init.freeze": True})),
        ("CLEVR init: stage 2, from_stage1 forgotten", ci(s2_cfg)),
        ("CLEVR init: freeze_body with from_stage1", ci(s1_cfg, **{"pretrain_init.from_stage1": str(good_s1)})),
        ("handoff: stage 1 was not frozen",       stage2(_fake_stage1(tmp / "ci_s1_unfrozen", conv, move_body=True,
                                                                      record={"freeze_body": False}))),
        ("handoff: body moved (record says frozen)", stage2(_fake_stage1(tmp / "ci_s1_moved", conv, move_body=True))),
        ("handoff: table never trained",          stage2(_fake_stage1(tmp / "ci_s1_still", conv, move_table=False))),
        ("handoff: not stage 1's final checkpoint", stage2(_fake_stage1(tmp / "ci_s1_mid", conv, step=165))),
        ("handoff: the other variant's stage 1",  stage2(_fake_stage1(tmp / "ci_s1_shuf", conv, record={"variant": "shuffled"}))),
        ("handoff: a run with no record",         stage2(_fake_stage1(tmp / "ci_s1_norec", conv))),
        ("resume: run dir records another variant", ci(s1_cfg, run_dir=_fake_stage1(
            tmp / "ci_res_shuf", conv, record={"variant": "shuffled"}).parent, resuming=True)),
        ("resume: run dir has no record",         ci(s1_cfg, run_dir=tmp / "ci_res_none", resuming=True)),
        # The LLM init (2026-10-28) shares the hook: its stage 1 on a CLEVR conversion is refused.
        ("LLM init: stage 1 on a CLEVR conversion", ci(eo_cfg / "arm_b10cl_stage1.yaml")),
    ]
    (tmp / "ci_s1_norec" / "pretrain_init.json").unlink()
    (tmp / "ci_res_none").mkdir()
    cases += [(label, "pretrain init", kw) for label, kw in ci_cases]
    ci_pos = [
        ("CLEVR init: stage 1",                   "pretrain init", _results(**ci(s1_cfg))),
        ("CLEVR init: stage 2 from a good stage 1", "pretrain init", _results(**stage2(good_s1))),
        ("CLEVR init: resume stage 1 as itself",  "pretrain init", _results(**ci(s1_cfg, run_dir=good_s1.parent, resuming=True))),
    ]
    ci_pos.append(("LLM init: stage 1 on an LLM conversion", "pretrain init", _results(
        **ci(eo_cfg / "arm_b10cl_stage1.yaml", conversion=_fake_conversion(tmp / "ci_conv_llm", variant="smollm")))))
    for label, real, shipped in (
            ("CLEVR init", Path("/data/enric/weights/clevr_init/b10c_v1"),
             {"clevr_body": "arm_b10cv_stage1", "shuffled_body": "arm_b10cs_stage1"}),
            ("LLM init", Path("/data/enric/weights/smollm_init/b10c_v1"),
             {"smollm_body": "arm_b10cl_stage1", "smollm_shuffled_body": "arm_b10cls_stage1"})):
        if all((real / d / "conversion.json").is_file() for d in shipped):
            ci_pos += [(f"{label}: {c} as shipped", "pretrain init", _results(env=GOOD_ENV, config=eo_cfg / f"{c}.yaml"))
                       for c in shipped.values()]
        else:
            print(f"\n({label}: the shipped-config positives skipped, no conversion on this machine)")

    print("\nnegative controls (each must FIRE):")
    fired = 0
    for label, target, kw in cases:
        res = _results(**kw)
        ok = not res[target]
        fired += ok
        print(f"  {'FIRED ' if ok else 'MISSED'}  {label:<36} -> {target}: {res.detail[target][:90]}")

    print("\npositive controls (must still PASS):")
    positives = [
        ("non-empty run dir WITH --force", "run dir",
         _results(env=GOOD_ENV, run_dir=occupied, allow_existing=True)),
        ("WANDB_MODE=disabled needs no key", "W&B",
         _results(env=_env(("WANDB_DIR",), WANDB_MODE="disabled"))),
    ] + [
        # Each arm resumed as ITSELF must pass. Without these the check could be
        # "always fail", which is as useless as never failing -- and a name-only
        # arm fingerprint did exactly that to recursive-no-router on the first
        # attempt here.
        (f"resume: {arm} ckpt as {arm} run", "resume checkpoint", _results(**resume_case(arm, arm)))
        for arm in ARMS
    ] + [
        ("not resuming skips the checkpoint check", "resume checkpoint",
         _results(env=GOOD_ENV, run_dir=ck["mor"], resuming=False, allow_existing=True)),
        # A null split is legitimate -- it means "no held-out evaluation", the
        # Phase 2 behaviour. Without this the check could be "always fail".
        ("eval split: null is allowed (no eval)", "eval split",
         _results(env=GOOD_ENV, config=_broken_config(tmp, multimodal__eval_split=None))),
    ] + [
        (f"a100: {name} under the RCP root", name, rcp_base) for name in ("MOR_SAVE_DIR", "HF_HOME", "W&B")
    ] + [
        ("token set: clean resolves to val_clean", "token set", _results(env=GOOD_ENV, config=clean_cfg)),
        ("train root: a good root passes",         "train root", _results(env=tenv["ok"], config=train_cfg)),
        ("train root: partial, allowed (smoke)",   "train root",
         _results(env=tenv["partial"], config=_broken_config(
             tmp, multimodal__split="train", multimodal__token_set="clean",
             multimodal__allow_partial_train_root=True))),
        ("train root: n/a for a split-val run",    "train root", _results(env=GOOD_ENV)),
    ] + [
        ("--gpus 2, two cards visible",          "visible devices",   _results(env=rcp2, n_gpus=2, **on_a100)),
        ("--gpus 2 on a100 pretraining",         "multi-gpu",         _results(env=rcp2, n_gpus=2, **on_a100)),
        ("batch: 256 on 2 GPUs x 16, eval /8",   "batch division",
         _results(env=rcp2, n_gpus=2, overrides={"per_device_train_batch_size": 16}, **on_a100)),
        ("resume: 2-GPU ckpt as a 2-GPU run",    "resume checkpoint", _results(env=rcp2, n_gpus=2, run_dir=ck2, **on_a100, **resume_kw)),
    ] + [
        (f"token set: {arm} (Hydra overlay)", "token set",
         _results(env=GOOD_ENV, config=arm_cfg / f"{arm}.yaml"))
        for arm in ("arm_b10c_vanilla", "arm_a10c_mor")
    ] + ben_pos + ci_pos
    pos_ok = 0
    for label, target, res in positives:
        ok = res[target]
        pos_ok += ok
        print(f"  {'PASS  ' if ok else 'FAILED'}  {label:<36} -> {target}: {res.detail[target][:90]}")

    # --- GPU alias resolution on fake inventories (gpu.resolve, no nvidia-smi). The
    # compute capability is what disambiguates names: an A10 (sm_86) contains "A10"
    # but not "A100"; a card NAMED A100 at the wrong capability must still refuse. ---
    from eo.train.gpu import GpuInfo, resolve, resolve_n
    a100 = GpuInfo(0, "NVIDIA A100-SXM4-80GB", (8, 0), 81920)
    res_cases = [
        ("resolve a100 on an RCP pod",            [a100], "a100", 0),
        ("resolve titanv on the workstation",     [GpuInfo(0, "NVIDIA GeForce GTX TITAN X", (5, 2), 12288),
                                                   GpuInfo(1, "NVIDIA TITAN V", (7, 0), 12288)], "titanv", 1),
        ("a100 alias on an A10 (sm_86) refuses",  [GpuInfo(0, "NVIDIA A10", (8, 6), 24576)], "a100", None),
        ("a100 alias at the wrong capability",    [GpuInfo(0, "NVIDIA A100-SXM4-80GB", (7, 5), 81920)], "a100", None),
        # identical cards: any is right, so the lowest index (2026-10-20; refused before)
        ("a100 alias, two A100s: the lowest",     [GpuInfo(1, "NVIDIA A100-SXM4-80GB", (8, 0), 81920), a100], "a100", 0),
        ("a100 alias, A100 80 GB + 40 GB refuses", [a100, GpuInfo(1, "NVIDIA A100-SXM4-40GB", (8, 0), 40960)], "a100", None),
        ("a100 alias on the workstation refuses", [GpuInfo(1, "NVIDIA TITAN V", (7, 0), 12288)], "a100", None),
    ]
    print("\nalias resolution (fake inventories):")
    res_ok = 0
    for label, inv, alias, want in res_cases:
        try:
            got = resolve(alias, inv).index
        except RuntimeError:
            got = None
        ok = got == want
        res_ok += ok
        print(f"  {'OK    ' if ok else 'WRONG '}  {label:<36} -> {got}")

    # --- several cards (resolve_n, --gpus N): N of ONE card, lowest indices first ---
    a100_b, a100_c = (GpuInfo(i, "NVIDIA A100-SXM4-80GB", (8, 0), 81920) for i in (1, 2))
    ws = [GpuInfo(0, "NVIDIA GeForce GTX TITAN X", (5, 2), 12288), GpuInfo(1, "NVIDIA TITAN V", (7, 0), 12288)]
    n_cases = [
        ("a100 x2 on two A100s",                  [a100_b, a100], "a100", 2, [0, 1]),
        ("a100 x2 on three A100s",                [a100_c, a100, a100_b], "a100", 2, [0, 1]),
        ("a100 x2 on one A100 refuses",           [a100], "a100", 2, None),
        ("a100 x2 on 80 GB + 40 GB refuses",      [a100, GpuInfo(1, "NVIDIA A100-SXM4-40GB", (8, 0), 40960)], "a100", 2, None),
        ("titanv x2 on the workstation refuses",  ws, "titanv", 2, None),
    ]
    for label, inv, alias, n, want in n_cases:
        try:
            got = [g.index for g in resolve_n(alias, n, inv)]
        except RuntimeError:
            got = None
        ok = got == want
        res_ok += ok
        print(f"  {'OK    ' if ok else 'WRONG '}  {label:<36} -> {got}")
    res_total = len(res_cases) + len(n_cases)

    # --- the launch guard pretrain.py runs for every EO run (distributed.check_launch):
    # a process seeing more cards than its launch accounts for gets nn.DataParallel. ---
    from eo.train.distributed import check_launch
    launch_cases = [
        ("plain launch, 1 card",                  1, {}, True),
        ("plain launch, 2 cards (DataParallel)",  2, {"CUDA_VISIBLE_DEVICES": "0,1"}, False),
        ("torchrun 2, 2 cards each",              2, {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"}, True),
        ("torchrun 2, 4 cards each",              4, {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"}, False),
        ("torchrun 4 over two nodes",             2, {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "2"}, False),
    ]
    print("\nlaunch guard (distributed.check_launch, fake device counts):")
    launch_ok = 0
    for label, count, env, want in launch_cases:
        try:
            check_launch(count, env)
            got = True
        except RuntimeError:
            got = False
        ok = got == want
        launch_ok += ok
        print(f"  {'OK    ' if ok else 'WRONG '}  {label:<36} -> {'allowed' if got else 'refused'}")

    total_ok = (fired == len(cases) and pos_ok == len(positives) and res_ok == res_total
                and launch_ok == len(launch_cases))
    print(f"\n{fired}/{len(cases)} negative controls fired, "
          f"{pos_ok}/{len(positives)} positive controls passed, "
          f"{res_ok}/{res_total} alias resolutions right, "
          f"{launch_ok}/{len(launch_cases)} launch-guard cases right")
    print("PREFLIGHT CONTROLS PASSED" if total_ok else "PREFLIGHT CONTROLS FAILED")
    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
