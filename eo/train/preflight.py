# eo/train/preflight.py
"""Refuse to start a run whose environment is wrong, and say exactly why.

WHY THIS EXISTS. Every failure this module checks for is SILENT. None of them
raises; each one produces a run that trains happily and is worthless:

  * the wrong GPU              -- 2.4x slower, no error                (gpu.py)
  * both GPUs visible          -- nn.DataParallel across unequal cards, and
                                  gradient_accumulation_steps computed wrong
  * MOR_SAVE_DIR unset         -- paths.py falls back to <repo>/results, i.e.
                                  ~1.5 GB checkpoints on /home, which the
                                  project forbids for large outputs. (They would
                                  not be committed -- `results/` is gitignored --
                                  but they would be on the wrong filesystem, and
                                  invisible to anyone looking under /data.)
  * TERRAMESH_TOK_ROOT unset   -- points at a default that may not exist
  * a non-empty run directory  -- silently overwrites a previous arm
  * vocab_size drift           -- indexes past the embedding table, or carries
                                  dead rows and a wrong ln(V) loss baseline

They are cheap to check and expensive to discover on day three of a run, so the
launcher checks all of them before `pretrain.py` is reached.

DESIGN. Every check is a function returning a `CheckResult`, and `run_checks`
returns the list. That is deliberate: Step 9's gate imports this module and
feeds it deliberately-broken contexts to prove each check can FAIL, the same
discipline `verify_phase2.py` applies to the data path. A check that has never
been seen to fail is not evidence of anything.

Usage (normally via eo/scripts/train_eo.sh, which sets the environment first):

    python -m eo.train.preflight --gpu titanv --run-dir /data/enric/runs/... \
        --config conf/pretrain_vision/eo_terramesh/terramesh_mor_token.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from eo.train import gpu as gpu_mod

#: Refuse to start with less headroom than this on the checkpoint filesystem.
#: A checkpoint is ~360 MB of weights plus ~1.1 GB of Adam state; a run saving
#: every 2,000 steps over several days accumulates tens of GB.
MIN_FREE_GIB = 50

#: Large outputs belong on /data (CLAUDE.md). This is the load-bearing
#: guardrail now that the data is local rather than reachable only over SSH.
LARGE_OUTPUT_PREFIX = "/data"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


@dataclass
class PreflightContext:
    """Everything the checks read. Explicit so tests can construct a broken one."""
    gpu_alias: str
    run_dir: Path
    config_path: Optional[Path] = None
    allow_existing: bool = False          # --resume or --force
    env: dict = field(default_factory=lambda: dict(os.environ))
    check_torch: bool = True              # off in tests that must not init CUDA


# --------------------------------------------------------------------------
# Individual checks. Each returns a CheckResult; none raises for an expected
# failure, so one bad environment reports every problem at once rather than
# making the user rediscover them one launch at a time.
# --------------------------------------------------------------------------

def check_device_order(ctx: PreflightContext) -> CheckResult:
    got = ctx.env.get("CUDA_DEVICE_ORDER")
    ok = got == gpu_mod.DEVICE_ORDER
    return CheckResult(
        "device order", ok,
        f"CUDA_DEVICE_ORDER={got!r}",
        "" if ok else
        f"Must be {gpu_mod.DEVICE_ORDER!r}. Unpinned, CUDA enumerates fastest-first "
        f"while nvidia-smi uses PCI order, so the two disagree about which card is 0.",
    )


def check_single_visible_device(ctx: PreflightContext) -> CheckResult:
    raw = ctx.env.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw == "":
        return CheckResult("one visible device", False, "CUDA_VISIBLE_DEVICES is unset",
                           "Both cards visible => nn.DataParallel across a sm_70 and a "
                           "sm_52 card, and the wrong device count in preprocess_config.")
    n = len([p for p in raw.split(",") if p.strip() != ""])
    return CheckResult("one visible device", n == 1, f"CUDA_VISIBLE_DEVICES={raw!r}",
                       "" if n == 1 else "Exactly one device must be visible.")


def check_gpu_identity(ctx: PreflightContext) -> CheckResult:
    if not ctx.check_torch:
        return CheckResult("gpu identity", True, "skipped (check_torch=False)")
    try:
        info = gpu_mod.assert_selected(ctx.gpu_alias)
    except Exception as exc:
        return CheckResult("gpu identity", False, str(exc).splitlines()[0],
                           "\n".join(str(exc).splitlines()[1:]))
    cc = f"sm_{info.compute_capability[0]}{info.compute_capability[1]}"
    return CheckResult("gpu identity", True,
                       f"cuda:0 is {info.name} ({cc}, {info.memory_total_mib} MiB) "
                       f"= alias {ctx.gpu_alias!r}")


def check_save_dir(ctx: PreflightContext) -> CheckResult:
    raw = ctx.env.get("MOR_SAVE_DIR")
    if not raw:
        return CheckResult(
            "MOR_SAVE_DIR", False, "unset",
            "paths.py then falls back to <repo>/results: ~1.5 GB checkpoints on /home, "
            "which the project forbids for large outputs, and nowhere anyone looks.")
    p = Path(raw)
    if not p.is_absolute():
        return CheckResult("MOR_SAVE_DIR", False, f"{raw!r} is not absolute", "Use an absolute path.")
    if not str(p).startswith(LARGE_OUTPUT_PREFIX):
        return CheckResult("MOR_SAVE_DIR", False, f"{p} is not under {LARGE_OUTPUT_PREFIX}",
                           "Large outputs go on /data, never /home (CLAUDE.md).")
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".preflight_write_test"
        probe.touch(); probe.unlink()
    except OSError as exc:
        return CheckResult("MOR_SAVE_DIR", False, f"{p} is not writable: {exc}")
    return CheckResult("MOR_SAVE_DIR", True, str(p))


def check_data_root(ctx: PreflightContext) -> CheckResult:
    raw = ctx.env.get("TERRAMESH_TOK_ROOT")
    if not raw:
        return CheckResult("TERRAMESH_TOK_ROOT", False, "unset",
                           "Set it explicitly rather than relying on the dataloader default.")
    root = Path(raw)
    if not root.is_dir():
        return CheckResult("TERRAMESH_TOK_ROOT", False, f"{root} does not exist")
    index = root / "tok_index.parquet"
    if not index.is_file():
        return CheckResult("TERRAMESH_TOK_ROOT", False, f"{index} missing",
                           "That file defines the canonical row order every modality shares.")

    from eo.terramesh_tok.contract import CROP, tok_dir_name
    from eo.data.eo_vocab import DEFAULT_ACTIVE_MODALITIES
    missing = [m for m in DEFAULT_ACTIVE_MODALITIES if not (root / tok_dir_name(m)).is_dir()]
    if missing:
        return CheckResult("TERRAMESH_TOK_ROOT", False,
                           f"{root}: no artifact for {missing} at crop {CROP}",
                           "The crop is in the directory name; the dataloader also asserts "
                           "metadata.json's crop at load.")
    return CheckResult("TERRAMESH_TOK_ROOT", True, f"{root} (crop {CROP}, "
                                                   f"{len(DEFAULT_ACTIVE_MODALITIES)} modalities)")


def check_hf_home(ctx: PreflightContext) -> CheckResult:
    raw = ctx.env.get("HF_HOME")
    if not raw:
        return CheckResult("HF_HOME", False, "unset",
                           "paths.py falls back to <repo>/hf_cache, i.e. model weights on /home.")
    p = Path(raw)
    if not str(p).startswith(LARGE_OUTPUT_PREFIX):
        return CheckResult("HF_HOME", False, f"{p} is not under {LARGE_OUTPUT_PREFIX}")
    return CheckResult("HF_HOME", True, str(p))


def check_wandb(ctx: PreflightContext) -> CheckResult:
    entity = ctx.env.get("WANDB_ENTITY")
    project = ctx.env.get("WANDB_PROJECT")
    mode = ctx.env.get("WANDB_MODE", "online")
    if not entity:
        return CheckResult("W&B", False, "WANDB_ENTITY unset",
                           "The EO config interpolates ${oc.env:WANDB_ENTITY} with no default, "
                           "so Hydra fails at config resolution before anything else.")
    if not project:
        return CheckResult("W&B", False, "WANDB_PROJECT unset")
    if mode == "disabled":
        return CheckResult("W&B", True, "disabled")
    wandb_dir = ctx.env.get("WANDB_DIR")
    if wandb_dir and not str(Path(wandb_dir)).startswith(LARGE_OUTPUT_PREFIX):
        return CheckResult("W&B", False, f"WANDB_DIR={wandb_dir} is not under {LARGE_OUTPUT_PREFIX}")
    if not wandb_dir and mode != "disabled":
        return CheckResult(
            "W&B", False, "WANDB_DIR unset",
            "wandb then writes its local run data into the current working directory -- "
            "i.e. into the repo, on /home. Measured 2026-09-21.")
    if mode == "online":
        # Credential PRESENCE only. Never read, print or log the key itself.
        netrc = Path.home() / ".netrc"
        has_netrc = netrc.is_file() and "api.wandb.ai" in netrc.read_text()
        if not (has_netrc or ctx.env.get("WANDB_API_KEY")):
            return CheckResult("W&B", False, "online mode but no credentials found",
                               "Run `wandb login` (writes ~/.netrc, outside the repo). "
                               "Never put the key in a project file.")
    return CheckResult("W&B", True, f"{entity}/{project} (mode={mode})")


def check_run_dir(ctx: PreflightContext) -> CheckResult:
    p = ctx.run_dir
    if not p.exists() or not any(p.iterdir()):
        return CheckResult("run dir", True, f"{p} (new)")
    if ctx.allow_existing:
        return CheckResult("run dir", True, f"{p} (existing, allowed by --resume/--force)")
    return CheckResult(
        "run dir", False, f"{p} exists and is not empty",
        "Refusing to write into it. Overwriting an arm silently destroys days of "
        "training that cannot be recovered. Pass --resume to continue it, --force to "
        "overwrite deliberately, or use a new timestamp.")


def check_disk(ctx: PreflightContext) -> CheckResult:
    raw = ctx.env.get("MOR_SAVE_DIR")
    target = Path(raw) if raw else ctx.run_dir
    probe = target if target.exists() else target.parent
    try:
        free_gib = shutil.disk_usage(probe).free / 2**30
    except OSError as exc:
        return CheckResult("disk space", False, f"cannot stat {probe}: {exc}")
    ok = free_gib >= MIN_FREE_GIB
    return CheckResult("disk space", ok, f"{free_gib:.0f} GiB free on {probe}",
                       "" if ok else f"Want at least {MIN_FREE_GIB} GiB; a checkpoint is "
                                     f"~1.5 GB with optimizer state.")


def check_config(ctx: PreflightContext) -> CheckResult:
    if ctx.config_path is None:
        return CheckResult("config", True, "skipped (no --config given)")
    path = ctx.config_path
    if not path.is_file():
        return CheckResult("config", False, f"{path} does not exist")

    import yaml
    cfg = yaml.safe_load(path.read_text())
    from eo.data.eo_vocab import TOTAL_VOCAB_SIZE

    problems = []
    vocab = (cfg.get("model_config") or {}).get("vocab_size")
    if vocab != TOTAL_VOCAB_SIZE:
        problems.append(f"vocab_size {vocab} != registry {TOTAL_VOCAB_SIZE}")
    if cfg.get("dataset") != "terramesh_multimodal":
        problems.append(f"dataset {cfg.get('dataset')!r} is not the EO dataset")
    if cfg.get("deepspeed") is not None:
        problems.append("deepspeed must be null -- get_launcher_type() returns "
                        "'deepspeed' for plain `python`, so a non-null value is honoured")
    if (cfg.get("vision_eval") or {}).get("enable"):
        problems.append("vision_eval must be false on the EO path (CLEVR-only callback)")
    if cfg.get("precision") not in ("fp32", "fp16", "bf16"):
        problems.append(f"precision {cfg.get('precision')!r} is not supported")

    if problems:
        return CheckResult("config", False, f"{path.name}: " + "; ".join(problems))
    return CheckResult("config", True,
                       f"{path.name} (vocab {vocab}, precision {cfg.get('precision')}, "
                       f"max_length {cfg.get('max_length')})")


#: Evaluated in order. Environment checks come first so a misconfigured shell
#: fails before torch is imported and CUDA is initialized.
CHECKS: List[Callable[[PreflightContext], CheckResult]] = [
    check_device_order,
    check_single_visible_device,
    check_save_dir,
    check_data_root,
    check_hf_home,
    check_wandb,
    check_run_dir,
    check_disk,
    check_config,
    check_gpu_identity,      # last: the only one that touches CUDA
]


def run_checks(ctx: PreflightContext) -> List[CheckResult]:
    """Every check, in order. Never raises -- failures come back as results."""
    results = []
    for fn in CHECKS:
        try:
            results.append(fn(ctx))
        except Exception as exc:                      # a check itself is broken
            results.append(CheckResult(fn.__name__, False, f"check raised: {exc!r}"))
    return results


def format_results(results: List[CheckResult]) -> str:
    width = max(len(r.name) for r in results)
    lines = []
    for r in results:
        lines.append(f"  {'PASS' if r.ok else 'FAIL'}  {r.name:<{width}}  {r.detail}")
        if not r.ok and r.hint:
            lines.extend(f"        {h}" for h in r.hint.splitlines() if h.strip())
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Environment preflight for an EO training run.")
    ap.add_argument("--gpu", required=True, choices=sorted(gpu_mod.GPU_ALIASES),
                    help="GPU alias. Never an index -- see eo/train/gpu.py.")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--allow-existing", action="store_true",
                    help="Set by --resume/--force in the launcher.")
    ap.add_argument("--no-torch", action="store_true",
                    help="Skip the CUDA-touching check (for tests).")
    args = ap.parse_args(argv)

    ctx = PreflightContext(
        gpu_alias=args.gpu,
        run_dir=args.run_dir,
        config_path=args.config,
        allow_existing=args.allow_existing,
        check_torch=not args.no_torch,
    )
    results = run_checks(ctx)
    print("Preflight ".ljust(70, "-"))
    print(format_results(results))
    print("-" * 70)

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"PREFLIGHT FAILED -- {len(failed)} of {len(results)} checks: "
              f"{', '.join(r.name for r in failed)}")
        return 1
    print(f"preflight passed ({len(results)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
