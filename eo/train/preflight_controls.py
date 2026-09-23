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
"""
from __future__ import annotations

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
                     step: int = 20) -> Path:
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

    for f in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"):
        if f not in drop:
            (d / f).write_text("{}")
    if "pytorch_model.bin" in drop:
        (d / "pytorch_model.bin").unlink()
    return d


def _results(env, *, run_dir: Optional[Path] = None, config: Optional[Path] = None,
             allow_existing: bool = False, resuming: bool = False,
             overrides: Optional[dict] = None, check_torch: bool = False,
             min_free_gpu_mib: Optional[int] = None) -> Dict[str, bool]:
    ctx = PreflightContext(
        gpu_alias="titanv",
        run_dir=run_dir or (Path(tempfile.mkdtemp()) / "fresh"),
        config_path=CONFIG if config is None else config,
        allow_existing=allow_existing,
        resuming=resuming,
        overrides=overrides or {},
        env=env,
        check_torch=check_torch,
        min_free_gpu_mib=min_free_gpu_mib,
    )
    return {r.name: r.ok for r in run_checks(ctx)}


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
    out = tmp / ("broken_" + "_".join(overrides) + ".yaml")
    out.write_text(yaml.safe_dump(cfg))
    return out


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
        ("both GPUs visible",                   "one visible device", dict(env=_env(CUDA_VISIBLE_DEVICES="0,1"))),
        ("CUDA_VISIBLE_DEVICES unset",          "one visible device", dict(env=_env(("CUDA_VISIBLE_DEVICES",)))),
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
            "mor": {"recursive.enable": True, "mor.enable": True}}
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
                              ("recursive", "vanilla")]:
        cases.append((f"resume: {ckpt_arm} ckpt as {run_arm} run", "resume checkpoint",
                      resume_case(ckpt_arm, run_arm)))

    print("\nnegative controls (each must FIRE):")
    fired = 0
    for label, target, kw in cases:
        res = _results(**kw)
        ok = not res[target]
        fired += ok
        print(f"  {'FIRED ' if ok else 'MISSED'}  {label:<36} -> {target}")

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
    ]
    pos_ok = 0
    for label, target, res in positives:
        ok = res[target]
        pos_ok += ok
        print(f"  {'PASS  ' if ok else 'FAILED'}  {label:<36} -> {target}")

    total_ok = fired == len(cases) and pos_ok == len(positives)
    print(f"\n{fired}/{len(cases)} negative controls fired, "
          f"{pos_ok}/{len(positives)} positive controls passed")
    print("PREFLIGHT CONTROLS PASSED" if total_ok else "PREFLIGHT CONTROLS FAILED")
    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
