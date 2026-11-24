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

Coords (2026-11-12, `check_coords`): every model trained from now on is without
Coords (Enric, worklog §5). The base config still lists it, so the baselines run
it with the six image modalities (NO_COORDS). On the shipped configs, a new run on
a Coords arm, on the base config, on a null or empty list, with --force, or with
an exemption that gives no reason must fire; the Coords-free overlays, a resume,
an exemption with a reason and a BEN run must pass.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eo.data.eo_vocab import IMAGE_MODALITIES
from eo.train.preflight import PreflightContext, run_checks

REPO = Path(__file__).resolve().parent.parent.parent
CONFIG = REPO / "conf/pretrain_vision/eo_terramesh/terramesh_mor_token.yaml"

#: A known-good NEW run lists the six image modalities. Since 2026-11-12 (Enric, worklog §5)
#: check_coords refuses a new run on the base config's own list, which holds Coords.
NO_COORDS: Dict[str, object] = {"multimodal.active_modalities": list(IMAGE_MODALITIES)}

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
                 move_table: bool = True, record: Optional[dict] = None, lora_keys: bool = False) -> Path:
    """Stage 1's checkpoint in miniature, plus the run's pretrain_init.json (`record` overrides it).
    `lora_keys` adds an adapter pair, as a LoRA stage 2's checkpoints carry (2026-10-29)."""
    import torch
    from safetensors.torch import load_file, save_file
    from eo.train.clevr_init import sha256
    s = load_file(str(conv / "model.safetensors"))
    if move_table:
        s["model.embed_tokens.weight"] = s["model.embed_tokens.weight"] + 1e-3
    if move_body:              # one ulp-scale step in one tensor: the comparison is bitwise
        s["model.layers.0.mlp.down_proj.weight"] = s["model.layers.0.mlp.down_proj.weight"] * (1 + 1e-6)
    if lora_keys:
        s["model.layers.0.mlp.down_proj.lora_A"] = torch.zeros(1, 2)
        s["model.layers.0.mlp.down_proj.lora_B"] = torch.zeros(2, 1)
    ck = run / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    save_file(s, str(ck / "model.safetensors"))
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": last}))
    rec = {"conversion_weights_sha256": sha256(conv / "model.safetensors"), "variant": "clevr",
           "freeze_body": True, "from_stage1": None}
    rec.update(record or {})
    (run / "pretrain_init.json").write_text(json.dumps(rec))
    return ck


def _fake_extension(d: Path, vocab_after: int = 87664, drop_json: bool = False, change_weights: bool = False,
                    **doc_over) -> Path:
    """An extended base in miniature (A1 Step 5, variant `continue`): two tensors and an extension.json
    that passed, at the temporal vocabulary and the date layout eo/data/temporal_vocab.py has now.
    `change_weights` rewrites the weights after the record was written: a base changed since its check."""
    import torch
    from safetensors.torch import save_file
    from eo.data import temporal_vocab as TV
    from eo.train.clevr_init import sha256
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(1)
    w = {"model.embed_tokens.weight": torch.randn(4, 2, generator=g), "model.norm.weight": torch.ones(2)}
    save_file(w, str(d / "model.safetensors"))
    doc = {"source_checkpoint": "/nowhere/checkpoint-17441", "source_weights_sha256": "9" * 64,
           "extension": {"vocab_before": 87556, "vocab_after": vocab_after, "n_new": vocab_after - 87556},
           "layout": "temporal", "temporal_vocab": TV.layout_record(),
           "identity": {"passed": True, "max_abs_diff": 1.9e-05, "tol": 1e-4},
           "control_insert_before_pad": {"fired": True}, "control_untied_resize": {"fired": True},
           "weights_file": "model.safetensors", "weights_sha256": sha256(d / "model.safetensors"), **doc_over}
    if change_weights:
        w["model.norm.weight"] = w["model.norm.weight"] * 2
        save_file(w, str(d / "model.safetensors"))
    if not drop_json:
        (d / "extension.json").write_text(json.dumps(doc))
    return d


def _fake_continued(d: Path, vocab: int = 161665, *, drop_record: bool = False, change_weights: bool = False,
                    both: bool = False, cfg_vocab: Optional[int] = None, base_over=None, pi_over=None) -> Path:
    """A continued base in miniature (A2 Step 5.2, eo/finetune/init.verify_continued): two tensors, a config.json and
    a continued.json whose base passed its identity check at the same vocabulary (A2's map layout, 161,665 ids) and
    whose run continued from that base. `change_weights` rewrites the weights after the record (changed since
    packaging); `both` adds an extension.json beside it; `base_over` / `pi_over` mutate the held base extension.json /
    the run's pretrain_init.json."""
    import torch
    from safetensors.torch import save_file
    from eo.train.clevr_init import sha256
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(2)
    w = {"model.embed_tokens.weight": torch.randn(4, 2, generator=g), "model.norm.weight": torch.ones(2)}
    save_file(w, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"vocab_size": cfg_vocab or vocab}))
    base_sha = "6" * 64
    ext = {"extension": {"vocab_before": 81296, "vocab_after": vocab, "n_new": vocab - 81296}, "layout": "map",
           "identity": {"passed": True, "max_abs_diff": 3.1e-05, "tol": 1e-4},
           "control_insert_before_pad": {"fired": True}, "control_untied_resize": {"fired": True},
           "weights_file": "model.safetensors", "weights_sha256": base_sha}
    pi = {"variant": "continue", "lora": None, "freeze_body": False, "from_stage1": None,
          "conversion_weights_sha256": base_sha}
    if base_over:
        base_over(ext)
    if pi_over:
        pi_over(pi)
    doc = {"kind": "continued", "version": 1, "run": "r1_fake", "step": 4000,
           "source_checkpoint": "/nowhere/checkpoint-4000", "vocab_size": vocab,
           "weights_file": "model.safetensors", "weights_sha256": sha256(d / "model.safetensors"),
           "base": {"path": "/nowhere/base", "weights_sha256": base_sha, "extension": ext}, "pretrain_init": pi}
    if change_weights:
        w["model.norm.weight"] = w["model.norm.weight"] * 2
        save_file(w, str(d / "model.safetensors"))
    if not drop_record:
        (d / "continued.json").write_text(json.dumps(doc))
    if both:
        (d / "extension.json").write_text(json.dumps(ext))
    return d


def _fake_val_site(d: Path, temporal: Optional[Path] = None, tok_index: Optional[Path] = None,
                   mutate=None, drop_check: bool = False) -> Path:
    """`<d>/val` (the real val root, linked) beside a cleaned root `<d>/val_clean` that links every entry
    of the real val_clean except its location index (and, given `tok_index`, its tok_index.parquet).
    The index is a copy of `temporal` (val_clean's own by default), then `mutate(index dir)`, and
    `drop_check` removes its check file. Returns what TERRAMESH_TOK_ROOT names."""
    import shutil
    real = Path(GOOD_ENV["TERRAMESH_TOK_ROOT"])
    clean = real.parent / "val_clean"
    vc = d / "val_clean"
    vc.mkdir(parents=True)
    (d / "val").symlink_to(real)
    for f in clean.iterdir():
        if f.name == "temporal" or (tok_index is not None and f.name == "tok_index.parquet"):
            continue
        (vc / f.name).symlink_to(f)
    if tok_index is not None:
        (vc / "tok_index.parquet").symlink_to(tok_index)
    shutil.copytree(temporal or clean / "temporal", vc / "temporal")
    if mutate is not None:
        mutate(vc / "temporal")
    if drop_check:
        (vc / "temporal" / "temporal_check.json").unlink()
    return d / "val"


def _fake_temporal_train(d: Path, tok_index: Path, temporal: Path) -> Path:
    """A train root as resolve_train_root accepts it (build.json: train, offset-free, complete) that holds
    only `tok_index` and the index at `temporal`, both linked: enough for the index check alone."""
    d.mkdir(parents=True)
    (d / "build.json").write_text(json.dumps({"split": "train", "offset_free": True, "partial": False}))
    (d / "tok_index.parquet").symlink_to(tok_index)
    (d / "temporal").symlink_to(temporal)
    return d


def _swap_two_locations(index_dir: Path) -> None:
    """A tampered index: two locations' rows exchanged, the build file left as it was."""
    import numpy as np
    a = np.load(index_dir / "locations.npy")
    a[[0, 1]] = a[[1, 0]]
    np.save(index_dir / "locations.npy", a)


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

    baseline = _results(GOOD_ENV, overrides=NO_COORDS)
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
    rcp_base = _results(rcp_env, gpu_alias="a100", overrides=NO_COORDS)
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

    # --- remote site 2 (SCITAS Izar V100, 2026-11-08): the same rules under MOR_EO_IZAR,
    # and neither site's variable may stand in for the other's. ---
    izar = tmp / "izar"
    izar_env = _env(MOR_EO_IZAR=str(izar), MOR_SAVE_DIR=str(izar / "runs"), HF_HOME=str(izar / "hf"),
                    WANDB_DIR=str(izar / "runs/wandb"), WANDB_MODE="offline")
    izar_base = _results(izar_env, gpu_alias="v100", overrides=NO_COORDS)
    if not all(izar_base.values()):
        print(f"  BROKEN v100 BASELINE -- failing: {[k for k, v in izar_base.items() if not v]}")
        return 1
    cases += [
        ("v100: MOR_EO_IZAR unset",               "MOR_SAVE_DIR", dict(env={k: v for k, v in izar_env.items() if k != "MOR_EO_IZAR"}, gpu_alias="v100")),
        ("v100: MOR_EO_IZAR relative",            "MOR_SAVE_DIR", dict(env={**izar_env, "MOR_EO_IZAR": "izar"}, gpu_alias="v100")),
        ("v100: MOR_SAVE_DIR outside Izar root",  "MOR_SAVE_DIR", dict(env={**izar_env, "MOR_SAVE_DIR": "/data/enric/runs"}, gpu_alias="v100")),
        ("v100: HF_HOME outside Izar root",       "HF_HOME",      dict(env={**izar_env, "HF_HOME": "/data/enric/hf"}, gpu_alias="v100")),
        ("v100: WANDB_DIR outside Izar root",     "W&B",          dict(env={**izar_env, "WANDB_DIR": "/data/enric/runs/wandb"}, gpu_alias="v100")),
        ("v100: MOR_EO_RCP does not serve Izar",  "MOR_SAVE_DIR", dict(env={**{k: v for k, v in izar_env.items() if k != "MOR_EO_IZAR"}, "MOR_EO_RCP": str(izar)}, gpu_alias="v100")),
        ("a100: MOR_EO_IZAR does not serve RCP",  "MOR_SAVE_DIR", dict(env={**izar_env}, gpu_alias="a100")),
        ("titanv: MOR_EO_IZAR cannot relax /data", "MOR_SAVE_DIR", dict(env={**izar_env}, gpu_alias="titanv")),
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
    conv_llm = _fake_conversion(tmp / "ci_conv_llm", variant="smollm")
    llm_s1 = _fake_stage1(tmp / "ci_s1_llm", conv_llm, record={"variant": "smollm"})
    lora2 = lambda ck, **ov: ci(eo_cfg / "arm_b10cl_lora.yaml", conversion=conv_llm,
                                **{"pretrain_init.from_stage1": str(ck), **ov})
    lora_rec = {"variant": "smollm", "freeze_body": False, "from_stage1": str(llm_s1),
                "lora": {"r": 16, "alpha": 32.0}}
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
        # Its LoRA arm (2026-10-29): a malformed lora block, LoRA as a stage 1, and a resume whose
        # record says LoRA while the checkpoint has no adapters (HF would load it partially).
        ("LLM LoRA: lora with freeze_body",       lora2(llm_s1, **{"pretrain_init.freeze_body": True,
                                                                  "pretrain_init.from_stage1": None})),
        ("LLM LoRA: unknown key in lora",         lora2(llm_s1, **{"+pretrain_init.lora.dropout": 0.1})),
        ("LLM LoRA: rank 0",                      lora2(llm_s1, **{"pretrain_init.lora.r": 0})),
        ("LLM LoRA: from a CLEVR stage 1",        lora2(good_s1)),
        ("resume: LoRA record, plain checkpoint", lora2(llm_s1, run_dir=_fake_stage1(
            tmp / "ci_res_lora_plain", conv_llm, step=660, last=3300, record=lora_rec).parent, resuming=True)),
        ("resume: full-FT record under the LoRA config", lora2(llm_s1, run_dir=_fake_stage1(
            tmp / "ci_res_full", conv_llm, step=660, last=3300,
            record={**lora_rec, "lora": None}).parent, resuming=True)),
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
        **ci(eo_cfg / "arm_b10cl_stage1.yaml", conversion=conv_llm))))
    ci_pos += [
        ("LLM LoRA: stage 2 from an LLM stage 1", "pretrain init", _results(**lora2(llm_s1))),
        ("LLM LoRA: resume with adapters in the checkpoint", "pretrain init", _results(**lora2(
            llm_s1, run_dir=_fake_stage1(tmp / "ci_res_lora", conv_llm, step=660, last=3300, record=lora_rec,
                                         lora_keys=True).parent, resuming=True))),
    ]
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

    # --- B10cL0-tr-NC (2026-11-18, A0.3): SmolLM's body from step 0 without stage 1, (freeze_body false, from_stage1
    # null) on purpose. Each of these would train happily: a CLEVR or a v1-sized conversion under it, and a resume
    # whose run directory is a stage 2's, a stage 1's or has no record. ---
    l0_cfg = eo_cfg / "arm_b10cl0trxnc_vanilla.yaml"
    conv_llm_nc = _fake_conversion(tmp / "ci_conv_llm_nc", variant="smollm", vocab_size=81188)
    conv_clevr_nc = _fake_conversion(tmp / "ci_conv_clevr_nc", variant="clevr", vocab_size=81188)
    l0_rec = {"variant": "smollm", "freeze_body": False, "from_stage1": None}
    l0 = lambda **ov: ci(l0_cfg, conversion=ov.pop("conversion", conv_llm_nc), **ov)   # noqa: E731
    l0_run = lambda name, **rec: _fake_stage1(tmp / name, conv_llm_nc, step=660, last=33000, move_body=True,  # noqa: E731
                                              record={**l0_rec, **rec}).parent
    (tmp / "ci_l0_none").mkdir()
    cases += [(label, "pretrain init", kw) for label, kw in [
        ("no stage 1: on a CLEVR conversion",          l0(conversion=conv_clevr_nc)),
        ("no stage 1: an LLM conversion at v1's 87,556", l0(conversion=conv_llm)),
        ("no stage 1: resume of a stage 2's run",      l0(run_dir=l0_run("ci_l0_s2", from_stage1=str(llm_s1)),
                                                         resuming=True)),
        ("no stage 1: resume of a stage 1's run",      l0(run_dir=l0_run("ci_l0_s1", freeze_body=True), resuming=True)),
        ("no stage 1: resume, run dir has no record",  l0(run_dir=tmp / "ci_l0_none", resuming=True)),
    ]]
    ci_pos += [
        ("no stage 1: a new run on an LLM nc conversion", "pretrain init", _results(**l0())),
        ("no stage 1: resume as itself",               "pretrain init", _results(**l0(run_dir=l0_run("ci_l0_own"),
                                                                                       resuming=True))),
    ]
    if (Path("/data/enric/weights/smollm_init/b10c_nc_v1/smollm_body") / "conversion.json").is_file():
        ci_pos.append(("no stage 1: arm_b10cl0trxnc_vanilla as shipped", "pretrain init",
                       _results(env=GOOD_ENV, config=l0_cfg)))
    else:
        print("\n(no stage 1: the shipped-config positive skipped, no nc conversion on this machine)")

    # --- A2 Step 5.2 (2026-11-23): a segmentation fine-tune from a CONTINUED base, a training checkpoint of a run that
    # continued from a verified extended base (R1's), with its continued.json (eo/finetune/init.verify_continued). Each
    # of these would load weights the record does not vouch for: no record (a raw checkpoint), weights changed after
    # packaging, another vocabulary, a base whose identity check failed, whose control did not fire or which extended
    # to another size, a run that continued from other weights or under LoRA, both records at once. Miniature fakes;
    # R1's raw checkpoint, its packaged base and the map base add cases where they exist. ---
    seg_cfg = {t: REPO / f"conf/pretrain_vision/eo_seg/{c}.yaml"
               for t, c in (("PASTIS", "r0_s41_lora16"), ("CloudSen12", "s52_cloudsen12"))}

    def seg(init, task="PASTIS"):
        return dict(env=GOOD_ENV, config=seg_cfg[task], overrides={"finetune.init_from": str(init)})
    ct_ok = _fake_continued(tmp / "ct_ok")
    cases += [(label, "init from", kw) for label, kw in [
        ("continued: no record (a raw checkpoint)",      seg(_fake_continued(tmp / "ct_norec", drop_record=True))),
        ("continued: weights changed after packaging",   seg(_fake_continued(tmp / "ct_sha", change_weights=True))),
        ("continued: another vocabulary (81,296)",       seg(_fake_continued(tmp / "ct_vocab", vocab=81296))),
        ("continued: config.json at another vocabulary", seg(_fake_continued(tmp / "ct_cfgvocab", cfg_vocab=81296))),
        ("continued: the base's identity check failed",  seg(_fake_continued(
            tmp / "ct_ident", base_over=lambda e: e["identity"].__setitem__("passed", False)))),
        ("continued: a base control did not fire",       seg(_fake_continued(
            tmp / "ct_ctrl", base_over=lambda e: e["control_insert_before_pad"].__setitem__("fired", False)))),
        ("continued: the base extended to 81,296",       seg(_fake_continued(
            tmp / "ct_bvocab", base_over=lambda e: e["extension"].__setitem__("vocab_after", 81296)))),
        ("continued: the run continued from other weights", seg(_fake_continued(
            tmp / "ct_other", pi_over=lambda p: p.__setitem__("conversion_weights_sha256", "0" * 64)))),
        ("continued: a LoRA continuation",               seg(_fake_continued(
            tmp / "ct_lora", pi_over=lambda p: p.__setitem__("lora", {"r": 16, "alpha": 32.0})))),
        ("continued: both records in one directory",     seg(_fake_continued(tmp / "ct_both", both=True))),
    ]]
    a2seg = Path("/data/enric/runs/pretrain/a2_seg")
    r1_ck = a2seg / "r1aligned_n4000_20261122_193801/checkpoint-4000"
    if (r1_ck / "model.safetensors").is_file():
        cases.append(("continued: R1's raw checkpoint-4000", "init from", seg(r1_ck)))
    ci_pos += [(f"continued: a miniature continued base, {t}", "init from", _results(**seg(ct_ok, t))) for t in seg_cfg]
    for label, d, marker in (("continued: R1's packaged base", a2seg / "base/r1aligned_n4000_ckpt4000", "continued.json"),
                             ("extension: the map base", a2seg / "base/b10cltrxnc_ext_map_nc_ckpt33000", "extension.json")):
        if (d / marker).is_file():
            ci_pos += [(f"{label}, {t}", "init from", _results(**seg(d, t))) for t in seg_cfg]
        else:
            print(f"\n({label}: positives skipped, no {d} on this machine)")

    # --- the temporal stage (A1 Step 5, 2026-11-08). T360-T continues T360 from its extended base
    # (pretrain_init variant `continue`), reads whole four-date locations through a location index
    # on every root, and records the settings that define its data so that a resume cannot change
    # them. Each of these would otherwise train happily, or fail only on the pod after the model is
    # built. Miniature bases and copies of val_clean's index throughout; the real base, T360's own
    # checkpoint and BEN extension, and train_coords' index add cases where they exist. ---
    from omegaconf import OmegaConf
    from eo.data import temporal_dataset as TD
    from eo.data import temporal_vocab as TV
    from eo.train.clevr_init import sha256 as _sha
    from eo.train.preflight import _load_config
    t360t = eo_cfg / "arm_t360t_vanilla.yaml"
    val_clean = real_val.parent / "val_clean"
    train_coords = Path("/data/enric/data/TerraMesh/train_coords")
    have_train_index = (train_coords / "temporal" / "temporal_check.json").is_file()
    ext_ok = _fake_extension(tmp / "tx_ok")
    rcp2_ = {**rcp_env, "CUDA_VISIBLE_DEVICES": "0,1"}

    def tt(config=t360t, env=GOOD_ENV, **ov):
        """The temporal arm, split val, on the miniature base unless an override says otherwise."""
        kw = {k: ov.pop(k) for k in ("run_dir", "resuming", "n_gpus", "gpu_alias") if k in ov}
        if kw.get("resuming"):
            kw["allow_existing"] = True
        return dict(env=env, config=config,
                    overrides={"multimodal.split": "val", "pretrain_init.conversion": str(ext_ok), **ov}, **kw)

    def ext(name, **kw):
        return {"pretrain_init.conversion": str(_fake_extension(tmp / name, **kw))}

    def init_run(name, **rec):
        d = tmp / name
        d.mkdir()
        (d / "pretrain_init.json").write_text(json.dumps({
            "conversion_weights_sha256": _sha(ext_ok / "model.safetensors"), "variant": "continue",
            "freeze_body": False, "from_stage1": None, "lora": None, **rec}))
        return d

    def data_run(name, mutate=None):
        """A run directory holding the record pretrain.py writes for the temporal arm at launch."""
        d = tmp / name
        d.mkdir()
        s = TD.data_settings(OmegaConf.create(_load_config(t360t, tt()["overrides"])))
        if mutate is not None:
            mutate(s)
        (d / TD.DATA_RECORD).write_text(json.dumps(s))
        return d

    for d in ("tr_none", "td_none"):
        (tmp / d).mkdir()
        (tmp / d / "stray.txt").write_text("x")
    t_cases = [
        # continue: the base (clevr_init.verify_init, read_block) and the run record on --resume
        ("continue: another size (87,577 rows)",    "pretrain init", tt(**ext("tx_ben", vocab_after=87577))),
        ("continue: no extension.json",              "pretrain init", tt(**ext("tx_nojson", drop_json=True))),
        ("continue: weights changed after the check", "pretrain init", tt(**ext("tx_sha", change_weights=True))),
        ("continue: identity check failed",          "pretrain init", tt(**ext("tx_ident", identity={"passed": False}))),
        ("continue: another date layout, same size", "pretrain init",
         tt(**ext("tx_layout", temporal_vocab={**TV.layout_record(), "n_day_bins": 37}))),
        ("continue: with freeze_body",               "pretrain init", tt(**{"pretrain_init.freeze_body": True})),
        ("continue: with from_stage1",               "pretrain init", tt(**{"pretrain_init.from_stage1": str(tmp / "x")})),
        ("continue: with lora",                      "pretrain init",
         tt(**{"+pretrain_init.lora.r": 16, "+pretrain_init.lora.alpha": 32})),
        ("resume: the run records another base",     "pretrain init",
         tt(run_dir=init_run("tr_base", conversion_weights_sha256="f" * 64), resuming=True)),
        ("resume: the run records another variant",  "pretrain init",
         tt(run_dir=init_run("tr_variant", variant="smollm"), resuming=True)),
        ("resume: the run has no init record",       "pretrain init", tt(run_dir=tmp / "tr_none", resuming=True)),
        # the settings, through the builder's own validation, and their division over the GPUs
        ("config: temporal at EO's 87,556 ids",      "config",        tt(**{"model_config.vocab_size": 87556})),
        ("temporal: per-device train batch 2",       "temporal data", tt(per_device_train_batch_size=2)),
        ("temporal: per-device eval batch 16",       "temporal data", tt(per_device_eval_batch_size=16)),
        ("temporal: max_length 2195 (needs 2,793)",  "temporal data", tt(max_length=2195)),
        ("temporal: shuffled patches",               "temporal data", tt(**{"multimodal.shuffle_image_patches": True})),
        ("temporal: a fixed modality order",         "temporal data", tt(**{"multimodal.modality_order": "fixed"})),
        ("temporal: a misspelt key in the block",    "temporal data", tt(**{"+temporal.single_shares": 0.4})),
        ("temporal: date probabilities sum to 1.1",  "temporal data", tt(**{"temporal.date_probs.first": 0.9})),
        ("temporal: 544 dev locations / 6",          "temporal data", tt(**{"temporal.eval_locations_per_microbatch": 6})),
        ("temporal: 4,416 held-out rows / 10",       "temporal data", tt(**{"temporal.eval_scenes_per_microbatch": 10})),
        ("temporal: 3 GPUs (68 dev micro-batches)",  "temporal data",
         tt(env={**rcp_env, "CUDA_VISIBLE_DEVICES": "0,1,2"}, n_gpus=3, gpu_alias="a100")),
        ("temporal: total 61 on 2 GPUs",             "temporal data", tt(env=rcp2_, n_gpus=2, gpu_alias="a100", total_batch_size=61)),
        # a resume may not change the data
        ("resume: the record's single_share 0.4",    "temporal data",
         tt(run_dir=data_run("td_share", lambda s: s["temporal"].__setitem__("single_share", 0.4)), resuming=True)),
        ("resume: seed 43, the record's 42",         "temporal data", tt(run_dir=data_run("td_seed"), resuming=True, seed=43)),
        ("resume: the ramp arm as the no-ramp arm",  "temporal data",
         tt(config=eo_cfg / "arm_t360t_noramp_vanilla.yaml", run_dir=data_run("td_ramp"), resuming=True)),
        ("resume: no data record",                   "temporal data", tt(run_dir=tmp / "td_none", resuming=True)),
        # every root's index
        ("index: val, a tampered copy",              "temporal index",
         tt(env=_env(TERRAMESH_TOK_ROOT=str(_fake_val_site(tmp / "ti_tamper", mutate=_swap_two_locations))))),
        ("index: val, its check file missing",       "temporal index",
         tt(env=_env(TERRAMESH_TOK_ROOT=str(_fake_val_site(tmp / "ti_nocheck", drop_check=True))))),
        ("index: train, another root's index",       "temporal index",
         tt(env=_env(TERRAMESH_TRAIN_TOK_ROOT=str(_fake_temporal_train(
             tmp / "ti_tr_other", troots["ok"] / "tok_index.parquet", val_clean / "temporal"))),
            **{"multimodal.split": "train"})),
        ("index: train, val's root as train's",      "temporal index",
         tt(env=_env(TERRAMESH_TRAIN_TOK_ROOT=str(_fake_temporal_train(
             tmp / "ti_tr_val", val_clean / "tok_index.parquet", val_clean / "temporal"))),
            **{"multimodal.split": "train"})),
    ]
    real_base = Path("/data/enric/runs/pretrain/temporal/base/t360_ext_temporal_ckpt17441")
    t360_ckpt = Path("/data/enric/runs/pretrain/phase3/vanilla360t_20261026_040535/checkpoint-17441")
    t360_ben = Path("/data/enric/runs/pretrain/phase6/base/t360_ext_ckpt17441")
    if (t360_ben / "extension.json").is_file():
        t_cases.append(("continue: T360's BEN base (87,577 rows)", "pretrain init",
                        tt(**{"pretrain_init.conversion": str(t360_ben)})))
    if (t360_ckpt / "model.safetensors").is_file():
        t_cases.append(("continue: T360's own checkpoint", "pretrain init",
                        tt(**{"pretrain_init.conversion": str(t360_ckpt)})))
    if have_train_index:
        t_cases += [
            ("index: val, another root's index (train's)", "temporal index",
             tt(env=_env(TERRAMESH_TOK_ROOT=str(_fake_val_site(tmp / "ti_other", temporal=train_coords / "temporal"))))),
            ("index: val, its split not the root's",     "temporal index",
             tt(env=_env(TERRAMESH_TOK_ROOT=str(_fake_val_site(
                 tmp / "ti_split", temporal=train_coords / "temporal", tok_index=train_coords / "tok_index.parquet"))))),
        ]
    else:
        print(f"\n(temporal: the cases on train's index skipped, no {train_coords}/temporal on this machine)")
    cases += t_cases

    t_good = _results(**tt())
    t_pos = [
        ("continue: a miniature extended base",    "pretrain init",  t_good),
        ("temporal: the arm, split val, 1 GPU",    "temporal data",  t_good),
        ("index: val_clean as it is",              "temporal index", t_good),
        ("config: the temporal arm at 87,664",     "config",         t_good),
        ("continue: resume under its own record",  "pretrain init",
         _results(**tt(run_dir=init_run("tr_self"), resuming=True))),
        ("temporal: 2 GPUs (276 and 68 split)",    "temporal data",  _results(**tt(env=rcp2_, n_gpus=2, gpu_alias="a100"))),
        ("resume: the same data",                  "temporal data",  _results(**tt(run_dir=data_run("td_same"), resuming=True))),
        ("resume: only the schedule changed",      "temporal data",  _results(**tt(
            run_dir=data_run("td_sched"), resuming=True, stop_steps=2662, num_train_steps=5324,
            learning_rate=0.0005, num_warmup_steps=200))),
    ]
    if have_train_index:
        t_pos.append(("index: train, train_coords' index", "temporal index", _results(**tt(
            env=_env(TERRAMESH_TRAIN_TOK_ROOT=str(_fake_temporal_train(
                tmp / "ti_tr_ok", train_coords / "tok_index.parquet", train_coords / "temporal"))),
            **{"multimodal.split": "train"}))))
    if (real_base / "extension.json").is_file():
        for arm in ("arm_t360t_vanilla", "arm_t360t_lr3e4_vanilla", "arm_t360t_noramp_vanilla"):
            shipped = _results(env=GOOD_ENV, config=eo_cfg / f"{arm}.yaml", overrides={"multimodal.split": "val"})
            t_pos += [(f"{arm} as shipped, split val", chk, shipped)
                      for chk in ("config", "temporal index", "temporal data", "pretrain init")]
    else:
        print(f"\n(temporal: the shipped-arm positives skipped, no {real_base} on this machine)")

    # --- The Coords-free stage (2026-11-17): arm_b10cltrxnct_vanilla, the nc layout (temporal_vocab_nc, 81,296
    # ids) and the modality draw. A v1 base under it would load nowhere (another table size) but is refused before
    # the pod; so are v1's size, Coords under nc, and a malformed draw. ---
    nct = eo_cfg / "arm_b10cltrxnct_vanilla.yaml"
    nc_base = real_base.parent / "b10cltrxnc_ext_temporal_nc_ckpt33000"

    def ntt(**ov):
        return dict(env=GOOD_ENV, config=nct, overrides={"multimodal.split": "val", **ov})
    cases += [
        ("nc: the arm at v1's 87,664 ids",            "config",        ntt(**{"model_config.vocab_size": 87664})),
        ("nc continue: a v1 extended base",            "pretrain init", ntt(**{"pretrain_init.conversion": str(ext_ok)})),
        ("nc: Coords listed under the nc layout",      "coords",        ntt(**{"multimodal.active_modalities": [
            "S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]})),
        ("nc temporal: kits summing to 0.9",           "temporal data", ntt(**{"temporal.kits.all": 0.4})),
        ("nc temporal: statics_drop 1.5",              "temporal data", ntt(**{"temporal.statics_drop": 1.5})),
        ("nc temporal: max_length 2787 (needs 2,788)", "temporal data", ntt(max_length=2787)),
    ]
    if (nc_base / "extension.json").is_file():
        shipped = _results(**ntt())
        t_pos += [("arm_b10cltrxnct_vanilla as shipped, split val", chk, shipped)
                  for chk in ("config", "coords", "temporal index", "temporal data", "pretrain init")]
    else:
        print(f"\n(temporal: the nc arm's positives skipped, no {nc_base} on this machine)")
    ci_pos += t_pos

    # --- Coords (2026-11-12). Enric: every model trained from now on is without Coords (worklog §5). The
    # base config still lists it, so an overlay that forgets multimodal.active_modalities trains on Coords,
    # exit 0, and so does a null or empty list (the dataset's default holds Coords). A resume keeps the
    # modalities its run started with (the 10x extensions on RCP), an exemption gives its reason (worklog
    # §4 item 62), and a BEN run reads S2L2A and S1GRD whatever list it inherits. The shipped configs
    # throughout: the guard is about what the overlays compose to. ---
    a10c, a10cnc = eo_cfg / "arm_a10c_mor.yaml", eo_cfg / "arm_a10cnc_mor.yaml"
    exempt = lambda reason: {"+multimodal.coords_exempt": reason}
    reason = "worklog item 62: a control"     # as a Hydra override: no comma, no parenthesis
    resumed = dict(run_dir=ck["mor"], resuming=True, allow_existing=True)
    cases += [(label, "coords", kw) for label, kw in [
        ("coords: arm_a10c_mor, a new run",                dict(env=GOOD_ENV, config=a10c)),
        ("coords: the base config as it is",               dict(env=GOOD_ENV)),
        ("coords: arm_a10c_mor with --force (no --resume)", dict(env=GOOD_ENV, config=a10c, run_dir=occupied,
                                                                allow_existing=True)),
        ("coords: list null (the default holds Coords)",   dict(env=GOOD_ENV, config=_broken_config(
            tmp, multimodal__active_modalities=None))),
        ("coords: list empty (the default holds Coords)",  dict(env=GOOD_ENV, config=_broken_config(
            tmp, multimodal__active_modalities=[]))),
        ("coords: arm_a10cnc_mor with its list nulled",    dict(env=GOOD_ENV, config=a10cnc,
                                                                overrides={"multimodal.active_modalities": None})),
        ("coords: exemption, an empty reason",             dict(env=GOOD_ENV, config=a10c, overrides=exempt(""))),
        ("coords: exemption, a blank reason",              dict(env=GOOD_ENV, config=_broken_config(
            tmp, multimodal__coords_exempt="   "))),
        ("coords: exemption true, no reason",              dict(env=GOOD_ENV, config=a10c, overrides=exempt(True))),
        ("coords: exemption, empty reason, on a resume",   dict(env=GOOD_ENV, config=a10c, overrides=exempt(""),
                                                                **resumed)),
        ("coords: T360-T's arm, a new run",                tt()),
    ]]
    t_exempt = _results(**tt(**exempt(reason)))
    ci_pos += [(label, "coords", res) for label, res in [
        ("coords: arm_a10cnc_mor, a new run",              _results(env=GOOD_ENV, config=a10cnc)),
        ("coords: arm_a10c_mor resumed",                   _results(env=GOOD_ENV, config=a10c, **resumed)),
        ("coords: arm_a10c_mor exempt, with a reason",     _results(env=GOOD_ENV, config=a10c, overrides=exempt(reason))),
        ("coords: T360-T's arm exempt, with a reason",     t_exempt),
        ("coords: BEN b10c_lora (inherits the list)",      _results(env=GOOD_ENV, config=REPO / "conf/pretrain_vision/eo_ben/b10c_lora.yaml")),
    ] + [
        (f"coords: {arm} resumed (as on RCP)", _results(env=GOOD_ENV, config=eo_cfg / f"{arm}.yaml", run_dir=ck["vanilla"],
                                                        resuming=True, allow_existing=True))
        for arm in ("arm_b10ctr_x10_vanilla", "arm_b10c11tr_x10_vanilla")
    ] + [
        (f"coords: {arm}, a new run", _results(env=GOOD_ENV, config=eo_cfg / f"{arm}.yaml"))
        for arm in ("arm_b10cnc_vanilla", "arm_a10cb3trnc_mor", "arm_a10cb3trs2nc_mor",
                    "arm_a10crb3trnc_random", "arm_a10crb3trs2nc_random")
    ]]
    # An exemption is not data: temporal_data.json records five multimodal keys, so adding one to a run's
    # config leaves its data record as it was, on a new run and on a resume.
    ci_pos += [
        ("temporal data: T360-T exempt, a new run",        "temporal data", t_exempt),
        ("temporal data: exemption added on a resume",     "temporal data",
         _results(**tt(run_dir=data_run("td_exempt"), resuming=True, **exempt(reason)))),
    ]

    # --- the compute target (Paper B intervention 1, B1, 2026-11-15; check_compute_target). A broken block, a
    # random router, another router function or alpha, the max-depth warm-up, two GPUs; and on a resume, a checkpoint
    # without compute_target.json (the price would restart silently), with one written under another target or for
    # another step, and a balancing run resuming a compute-target checkpoint. ---
    from eo.train import compute_target as CTm
    # arm B's form since 2026-11-16 (B1 Step 6): the hard body mean with a deadband of 0.05
    ct_good = {"target": 1.8, "gain": 0.01, "proportional": 0.5, "price_init": 0.0, "measure": "hard", "deadband": 0.05}
    ct_soft = {**ct_good, "measure": "soft", "deadband": 0.0}
    ct_blk = CTm.read_block({"mor": {"token": {"balancing": "compute", "compute": ct_good}},
                             "recursive": {"num_recursion": 3}})

    def ct_cfg(**over):
        kw = {"mor__token__balancing": "compute", "mor__token__compute": dict(ct_good),
              "multimodal__active_modalities": list(IMAGE_MODALITIES)}
        kw.update(over)
        return _broken_config(tmp, **kw)

    def ct_ckpt(name, write=True, step=20, file_step=None, **blk_over):
        d = tmp / f"run_ct_{name}"
        c = _fake_checkpoint(d, "mor", step=step)
        (c / "trainer_state.json").write_text(json.dumps({"train_batch_size": 4, "global_step": step}))
        if write:
            CTm.write_state(c, {**ct_blk, **blk_over}, {"lam": 0.01, "e_last": -0.05},
                            step if file_step is None else file_step, origin="preflight control")
        return d

    def ct_ckpt_v1(name, step=20):
        """A compute_target.json as written before 2026-11-16 (version 1: no measure, no deadband)."""
        d = tmp / f"run_ct_{name}"
        c = _fake_checkpoint(d, "mor", step=step)
        (c / "trainer_state.json").write_text(json.dumps({"train_batch_size": 4, "global_step": step}))
        (c / CTm.FILE).write_text(json.dumps({"version": 1, "target": 1.8, "gain": 0.01, "proportional": 0.5,
                                              "price_init": 0.0, "lam": 0.01, "e_last": -0.05, "global_step": step}))
        return d

    def ct_ckpt_v2(name, step=20, **blk_over):
        """A compute_target.json as written before 2026-11-17 (version 2: no confidence)."""
        d = ct_ckpt(name, step=step, **blk_over)
        f = d / f"checkpoint-{step}" / CTm.FILE
        doc = json.loads(f.read_text())
        doc.pop("confidence", None)
        doc["version"] = 2
        f.write_text(json.dumps(doc))
        return d

    def ct_resume(run, config):
        return dict(env=GOOD_ENV, run_dir=run, resuming=True, allow_existing=True, config=config)

    ct_ok = ct_cfg()
    ct_soft_ok = ct_cfg(mor__token__compute=dict(ct_soft))
    cases += [(label, "compute target", kw) for label, kw in [
        ("compute: no mor.token.compute block",        dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute=None))),
        ("compute: gain missing",                      dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={
            k: v for k, v in ct_good.items() if k != "gain"}))),
        ("compute: a misspelt key (gian)",             dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={
            **ct_good, "gian": 0.01}))),
        ("compute: target 1.0 (the floor)",            dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={**ct_good, "target": 1.0}))),
        ("compute: target 3.0 (the ceiling)",          dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={**ct_good, "target": 3.0}))),
        ("compute: gain 0",                            dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={**ct_good, "gain": 0.0}))),
        ("compute: on a random router",                dict(env=GOOD_ENV, config=ct_cfg(mor__rand_router=True))),
        ("compute: a sigmoid router",                  dict(env=GOOD_ENV, config=ct_cfg(mor__token__router_func="sigmoid"))),
        ("compute: alpha 0.5",                         dict(env=GOOD_ENV, config=ct_cfg(mor__token__alpha=0.5))),
        ("compute: with bal_warmup_step",              dict(env=GOOD_ENV, config=ct_cfg(mor__token__bal_warmup_step=10))),
        ("compute: --gpus 2",                          dict(env=_env(CUDA_VISIBLE_DEVICES="0,1"), config=ct_ok, n_gpus=2)),
        ("compute: resume, no compute_target.json",    ct_resume(ct_ckpt("nofile", write=False), ct_ok)),
        ("compute: resume, the file's target is 2.0",  ct_resume(ct_ckpt("other_target", target=2.0), ct_ok)),
        ("compute: resume, the file is for step 10",   ct_resume(ct_ckpt("other_step", file_step=10), ct_ok)),
        ("compute: balancing run on a ct checkpoint",  ct_resume(ct_ckpt("as_balancing"),
                                                                 _broken_config(tmp, multimodal__active_modalities=list(IMAGE_MODALITIES)))),
        # B1 Step 6 (2026-11-16): the measure and the deadband
        ("compute: a misspelt measure (hrad)",         dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={**ct_good, "measure": "hrad"}))),
        ("compute: a 2026-11-15 block (no measure)",   dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={
            k: v for k, v in ct_good.items() if k not in ("measure", "deadband")}))),
        ("compute: deadband -0.01",                    dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={**ct_good, "deadband": -0.01}))),
        ("compute: deadband 0.1 (half a slot)",        dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={**ct_good, "deadband": 0.1}))),
        ("compute: resume, the file's measure is soft", ct_resume(ct_ckpt("other_measure", measure="soft"), ct_ok)),
        ("compute: resume, the file's deadband is 0",  ct_resume(ct_ckpt("other_band", deadband=0.0), ct_ok)),
        ("compute: resume, a version-1 file, hard run", ct_resume(ct_ckpt_v1("v1_hard"), ct_ok)),
        # B1 Step 7b (2026-11-17): the confidence term
        ("compute: confidence -0.01",                  dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={**ct_soft, "confidence": -0.01}))),
        ("compute: confidence 10.5",                   dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={**ct_soft, "confidence": 10.5}))),
        ("compute: a misspelt confidence (confidance)", dict(env=GOOD_ENV, config=ct_cfg(mor__token__compute={**ct_soft, "confidance": 0.1}))),
        ("compute: resume, the file's confidence is 0.1", ct_resume(ct_ckpt("other_conf", confidence=0.1), ct_ok)),
        ("compute: resume, a version-2 file, run at 0.1", ct_resume(ct_ckpt_v2("v2_conf", measure="soft", deadband=0.0),
                                                                    ct_cfg(mor__token__compute={**ct_soft, "confidence": 0.1}))),
    ]]
    ct_conf_ok = ct_cfg(mor__token__compute={**ct_soft, "confidence": 0.1})
    ci_pos += [(label, "compute target", res) for label, res in [
        ("compute: a good block, a new run",           _results(env=GOOD_ENV, config=ct_ok)),
        ("compute: resumed with its own file",         _results(**ct_resume(ct_ckpt("good"), ct_ok))),
        ("compute: n/a on a balancing run",            _results(env=GOOD_ENV, overrides=NO_COORDS)),
        ("compute: the soft measure, a new run",       _results(env=GOOD_ENV, config=ct_soft_ok)),
        ("compute: a version-1 file, soft run",        _results(**ct_resume(ct_ckpt_v1("v1_soft"), ct_soft_ok))),
        # B1 Step 7b (2026-11-17): the confidence term
        ("compute: confidence 0.1, a new run",         _results(env=GOOD_ENV, config=ct_conf_ok)),
        ("compute: confidence 0.1, resumed with its own file", _results(**ct_resume(
            ct_ckpt("good_conf", measure="soft", deadband=0.0, confidence=0.1), ct_conf_ok))),
        ("compute: a version-2 file, run without confidence", _results(**ct_resume(ct_ckpt_v2("v2_ok"), ct_ok))),
    ] + [
        (f"compute: {arm} as shipped (replays only)", _results(env=GOOD_ENV, config=eo_cfg / f"{arm}.yaml",
                                                               run_dir=ct_ckpt(f"ship_{arm}", **{
                                                                   "target": 2.0 if "ct20" in arm else 1.8,
                                                                   **({} if arm.endswith("18h_mor") else
                                                                      {"measure": "soft", "deadband": 0.0})}),
                                                               resuming=True, allow_existing=True))
        for arm in ("arm_a10cs2ct18_mor", "arm_a10cs2ct20_mor", "arm_a10cct18_mor", "arm_a10cct20_mor",
                    "arm_a10cs2ct18h_mor")
    ] + [
        (f"compute: {arm} as shipped, a new run (arm B)", _results(env=GOOD_ENV, config=eo_cfg / f"{arm}.yaml"))
        for arm in ("arm_a10cb3trncct18_mor", "arm_a10cb3trs2ncct18_mor",
                    "arm_a10cb3trncct18c_mor", "arm_a10cb3trs2ncct18c_mor")   # the last two: B1 Step 7b, 2026-11-17
    ]]

    # --- the depth table (B1 Step 8, 2026-11-18; check_depth_table). A broken block, a learned router, Coords active,
    # the max-depth warm-up; and on a resume, a checkpoint trained under another table, a table run resumed without its
    # table, a random-router checkpoint under a table without the transplant launcher's record, and a transplant record
    # of another table. ---
    from eo.train import depth_table as DTm
    third = [0.3333333333333333, 0.3333333333333333, 0.3333333333333334]
    dt_good = {"rows": {sl: [0.0, 1.0, 0.0] for sl in DTm.SLOTS}, "gate": "draw", "seed": 0}
    dt_other = {**dt_good, "rows": {sl: list(third) for sl in DTm.SLOTS}}

    def dt_cfg(table=None, **over):
        kw = {"mor__rand_router": True, "multimodal__active_modalities": list(IMAGE_MODALITIES)}
        if table is not None:
            kw["mor__token__depth_table"] = json.loads(json.dumps(table))
        kw.update(over)
        return _broken_config(tmp, **kw)

    def dt_rows(**rows):
        return {**dt_good, "rows": {**dt_good["rows"], **rows}}

    def dt_ckpt(name, record=None, transplant=None, step=20):
        d = tmp / f"run_dt_{name}"
        c = _fake_checkpoint(d, "random", step=step)
        (c / "trainer_state.json").write_text(json.dumps({"train_batch_size": 4, "global_step": step}))
        blk = lambda b: DTm.read_block({"mor": {"token": {"depth_table": b}}, "recursive": {"num_recursion": 3}})  # noqa: E731
        if record is not None:
            DTm.write_record(c, blk(record), step, "preflight control")
        if transplant is not None:
            (c / DTm.TRANSPLANT).write_text(json.dumps({"version": 1, "table": DTm.identity(blk(transplant))}))
        return d

    def dt_resume(run, config):
        return dict(env=GOOD_ENV, run_dir=run, resuming=True, allow_existing=True, config=config)

    dt_ok, dt_plain = dt_cfg(dt_good), dt_cfg()
    cases += [(label, "depth table", kw) for label, kw in [
        ("table: a missing slot (LULC)",             dict(env=GOOD_ENV, config=dt_cfg({**dt_good, "rows": {
            k: v for k, v in dt_good["rows"].items() if k != "LULC"}}))),
        ("table: an extra slot (Coords)",            dict(env=GOOD_ENV, config=dt_cfg(dt_rows(Coords=[0.0, 1.0, 0.0])))),
        ("table: a row summing to 0.9",              dict(env=GOOD_ENV, config=dt_cfg(dt_rows(DEM=[0.3, 0.3, 0.3])))),
        ("table: a row over 2 depths",               dict(env=GOOD_ENV, config=dt_cfg(dt_rows(DEM=[0.5, 0.5])))),
        ("table: gate soft",                         dict(env=GOOD_ENV, config=dt_cfg({**dt_good, "gate": "soft"}))),
        ("table: seed 1.5",                          dict(env=GOOD_ENV, config=dt_cfg({**dt_good, "seed": 1.5}))),
        ("table: on a learned router",               dict(env=GOOD_ENV, config=dt_cfg(dt_good, mor__rand_router=False))),
        ("table: Coords active",                     dict(env=GOOD_ENV, config=dt_cfg(
            dt_good, multimodal__active_modalities=list(IMAGE_MODALITIES) + ["Coords"]))),
        ("table: with bal_warmup_step",              dict(env=GOOD_ENV, config=dt_cfg(dt_good, mor__token__bal_warmup_step=10))),
        ("table: resume, another table's record",    dt_resume(dt_ckpt("other", record=dt_other), dt_ok)),
        ("table: resume, a table run under no table", dt_resume(dt_ckpt("as_plain", record=dt_good), dt_plain)),
        ("table: resume, a plain checkpoint, no transplant", dt_resume(dt_ckpt("plain"), dt_ok)),
        ("table: resume, another table's transplant", dt_resume(dt_ckpt("tr_other", transplant=dt_other), dt_ok)),
    ]]
    ci_pos += [(label, "depth table", res) for label, res in [
        ("table: a good block, a new run",           _results(env=GOOD_ENV, config=dt_ok)),
        ("table: resumed with its own record",       _results(**dt_resume(dt_ckpt("own", record=dt_good), dt_ok))),
        ("table: a transplant's first resume",       _results(**dt_resume(dt_ckpt("tr_own", transplant=dt_good), dt_ok))),
        ("table: n/a on a random router without one", _results(env=GOOD_ENV, config=dt_plain)),
        ("table: a plain checkpoint, no table",      _results(**dt_resume(dt_ckpt("plain_ok"), dt_plain))),
    ] + [
        (f"table: {arm} as shipped, a new run",      _results(env=GOOD_ENV, config=eo_cfg / f"{arm}.yaml"))
        for arm in ("arm_a10crb3trnct2_random", "arm_a10crb3trnct2g1_random", "arm_a10crb3trs2nct2_random",
                    "arm_a10crb3trs2nct2g1_random", "arm_a10crb3trnctu_random", "arm_a10crb3trnctu1_random")
    ]]

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
        (f"v100: {name} under the Izar root", name, izar_base) for name in ("MOR_SAVE_DIR", "HF_HOME", "W&B")
    ] + [
        ("--gpus 2 on a v100 node",              "multi-gpu",
         _results(env={**izar_env, "CUDA_VISIBLE_DEVICES": "0,1"}, n_gpus=2, gpu_alias="v100")),
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
    v100 = GpuInfo(0, "Tesla V100-PCIE-32GB", (7, 0), 32768)
    ws_both = [GpuInfo(0, "NVIDIA GeForce GTX TITAN X", (5, 2), 12288), GpuInfo(1, "NVIDIA TITAN V", (7, 0), 12288)]
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
        # V100 and TITAN V are both sm_70: only the name separates them, both ways.
        ("resolve v100 on an Izar node",          [v100], "v100", 0),
        ("v100 alias on the TITAN V refuses",     ws_both, "v100", None),
        ("titanv alias on a V100 refuses",        [v100], "titanv", None),
        ("v100 alias on an A100 refuses",         [a100], "v100", None),
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
        ("v100 x2 on a two-V100 node",            [v100, GpuInfo(1, "Tesla V100-PCIE-32GB", (7, 0), 32768)], "v100", 2, [0, 1]),
        ("v100 x2, 32 GB + 16 GB refuses",        [v100, GpuInfo(1, "Tesla V100-SXM2-16GB", (7, 0), 16384)], "v100", 2, None),
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
