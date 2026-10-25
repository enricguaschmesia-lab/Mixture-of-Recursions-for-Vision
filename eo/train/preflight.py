# eo/train/preflight.py
"""Refuse to start a run whose environment is wrong, and say exactly why.

WHY THIS EXISTS. Every failure this module checks for is SILENT. None of them
raises; each one produces a run that trains happily and is worthless:

  * the wrong GPU              -- 2.4x slower, no error                (gpu.py)
  * both GPUs visible          -- nn.DataParallel across unequal cards, and
                                  gradient_accumulation_steps computed wrong
                                  (with --gpus N: exactly N identical cards)
  * a batch that does not divide -- preprocess_config rounds the accumulation,
                                  so 256 on 3 GPUs trains at 240
  * a resume at another GPU split -- no RNG state restored, data resharded,
                                  or a different total batch
  * MOR_SAVE_DIR unset         -- paths.py falls back to <repo>/results, i.e.
                                  ~1.5 GB checkpoints on /home, which the
                                  project forbids for large outputs. (They would
                                  not be committed -- `results/` is gitignored --
                                  but they would be on the wrong filesystem, and
                                  invisible to anyone looking under /data.)
  * TERRAMESH_TOK_ROOT unset   -- points at a default that may not exist
  * token_set unresolvable     -- a cleaned root that was never gated, or a
                                  TERRAMESH_TOK_ROOT that already is one, so
                                  the config misreports which tokens the run saw
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
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from eo.train import gpu as gpu_mod
from eo.train.distributed import checkpoint_world_size

#: Repo root, for resolving repo-relative paths named in a config.
REPO = Path(__file__).resolve().parents[2]

#: Refuse to start with less headroom than this on the checkpoint filesystem.
#: A checkpoint is ~360 MB of weights plus ~1.1 GB of Adam state; a run saving
#: every 2,000 steps over several days accumulates tens of GB.
MIN_FREE_GIB = 50

#: Refuse to start if the selected GPU has less free memory than this. Arm A
#: peaks at 10,110 MiB (fp16, per_device 4, measured 2026-09-22), so this leaves
#: a small margin on a 12,288 MiB card.
#: ⚠ The workstation is NOT exclusively this project's. On 2026-09-23 an
#: unrelated job held 9.58 GiB of the TITAN V and a launch OOM'd at step 1.
#: Over a 6-day run that is the difference between a restart and a lost week.
MIN_FREE_GPU_MIB = 11000

#: Large outputs belong on /data (CLAUDE.md). This is the load-bearing
#: guardrail now that the data is local rather than reachable only over SSH.
#: A remote alias (gpu.py, `large_output_env`) replaces it with its private root.
LARGE_OUTPUT_PREFIX = "/data"


def large_output_root(ctx: "PreflightContext") -> Tuple[Optional[Path], str]:
    """(root large outputs must live under, error). The root is None on error.

    Workstation aliases: /data, whatever the environment says, so an RCP variable
    left in a local shell cannot relax the local guard. Remote aliases: the
    absolute path in the alias's `large_output_env`, which must be set.
    """
    spec = gpu_mod.GPU_ALIASES.get(ctx.gpu_alias.strip().lower())
    key = spec.large_output_env if spec else None
    if key is None:
        return Path(LARGE_OUTPUT_PREFIX), ""
    raw = ctx.env.get(key)
    if not raw or not Path(raw).is_absolute():
        return None, (f"{key} {'unset' if not raw else f'{raw!r} is not absolute'}: "
                      f"--gpu {ctx.gpu_alias} needs the site's private root there")
    return Path(raw), ""


def _under(p: Path, root: Path) -> bool:
    """Path containment, not a string prefix: /data2 is not under /data."""
    return p == root or root in p.parents


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


#: The datasets an EO run may use: TerraMesh pretraining, and the Phase 6 BEN fine-tune.
EO_DATASETS = ("terramesh_multimodal", "ben_multimodal")


@dataclass
class PreflightContext:
    """Everything the checks read. Explicit so tests can construct a broken one."""
    gpu_alias: str
    run_dir: Path
    config_path: Optional[Path] = None
    allow_existing: bool = False          # --resume or --force
    resuming: bool = False                # --resume specifically
    overrides: Dict[str, object] = field(default_factory=dict)
    env: dict = field(default_factory=lambda: dict(os.environ))
    check_torch: bool = True              # off in tests that must not init CUDA
    min_free_gpu_mib: Optional[int] = None   # None -> MIN_FREE_GPU_MIB
    n_gpus: int = 1                       # train_eo.sh --gpus; > 1 means a torchrun DDP launch


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


def check_visible_devices(ctx: PreflightContext) -> CheckResult:
    """Exactly as many devices visible as the run is launched for (1 unless --gpus N).

    More visible than launched for => nn.DataParallel (on the workstation across a
    sm_70 and a sm_52 card) and the wrong device count in preprocess_config.
    Fewer => a DDP launch whose ranks share a card.
    """
    raw = ctx.env.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw == "":
        return CheckResult("visible devices", False, "CUDA_VISIBLE_DEVICES is unset",
                           "Both cards visible => nn.DataParallel across a sm_70 and a "
                           "sm_52 card, and the wrong device count in preprocess_config.")
    n = len([p for p in raw.split(",") if p.strip() != ""])
    ok = n == ctx.n_gpus
    return CheckResult("visible devices", ok, f"CUDA_VISIBLE_DEVICES={raw!r} (--gpus {ctx.n_gpus})",
                       "" if ok else f"Exactly {ctx.n_gpus} device(s) must be visible.")


def check_multi_gpu(ctx: PreflightContext) -> CheckResult:
    """--gpus N > 1 only where it has been verified: TerraMesh pretraining, identical cards.

    The workstation's aliases each match ONE card (and its two cards differ, so DDP
    would run at the Titan X's pace). A BEN fine-tune adds per-epoch input draws and
    the distillation sampler, neither verified across processes; at ~40 A100-minutes
    a fine-tune gains nothing from more GPUs.
    """
    n = ctx.n_gpus
    if n < 1:
        return CheckResult("multi-gpu", False, f"--gpus {n}", "Must be at least 1.")
    if n == 1:
        return CheckResult("multi-gpu", True, "single GPU (the default)")
    spec = gpu_mod.GPU_ALIASES.get(ctx.gpu_alias.strip().lower())
    if spec is not None and spec.large_output_env is None:
        return CheckResult("multi-gpu", False, f"--gpus {n} on the workstation alias {ctx.gpu_alias!r}",
                           "The workstation has one card per alias, and its two cards differ. "
                           "Multi-GPU is for an RCP pod with identical cards (--gpu a100).")
    cfg = _load_config(ctx.config_path, ctx.overrides) if ctx.config_path and ctx.config_path.is_file() else {}
    if cfg.get("dataset") == "ben_multimodal" or (cfg.get("finetune") or {}).get("init_from"):
        return CheckResult("multi-gpu", False, f"--gpus {n} on a BEN fine-tune",
                           "Verified for TerraMesh pretraining only (eo/train/distributed.py).")
    return CheckResult("multi-gpu", True, f"{n} GPUs, torchrun DDP on one node")


def check_batch_division(ctx: PreflightContext) -> CheckResult:
    """The batch must split EXACTLY over the GPUs; the eval set too when N > 1.

    `util.config.preprocess_config` ROUNDS gradient_accumulation_steps =
    total_batch_size / (per_device x n_gpus), and only prints the batch that
    results. So 256 on 3 GPUs at per-device 16 trains at 240 (3 x 16 x 5), and at
    per-device 4 at 252: a different recipe from every single-GPU run, exit 0.

    With N > 1 the held-out rows must also split over N x per_device_eval_batch_size.
    Otherwise the distributed eval loader fills the last round by repeating rows;
    HF drops them from eval_loss, but the per-modality eval CE is accumulated inside
    compute_loss and would count them.
    """
    if ctx.config_path is None or not ctx.config_path.is_file():
        return CheckResult("batch division", True, "skipped (config missing; check_config reports it)")
    cfg = _load_config(ctx.config_path, ctx.overrides)
    n = max(ctx.n_gpus, 1)
    total, per = cfg.get("total_batch_size"), cfg.get("per_device_train_batch_size")
    notes = []
    if total is not None and per:
        if total % (per * n):
            actual = per * n * max(round(total / (per * n)), 0)
            return CheckResult(
                "batch division", False,
                f"total_batch_size {total} is not a multiple of per-device {per} x {n} GPU(s)",
                f"preprocess_config rounds the accumulation, so the run would train at batch "
                f"{actual}, not {total}. Pick a per-device batch or GPU count that divides it.")
        notes.append(f"{total} = {per} x {n} GPU(s) x {total // (per * n)}")
    if n > 1:
        ref = (cfg.get("multimodal") or {}).get("eval_split")
        if ref is not None and cfg.get("dataset") != "ben_multimodal":
            rows = _eval_row_count(ref, ctx.env.get("TERRAMESH_TOK_ROOT"))
            ev = cfg.get("per_device_eval_batch_size") or per
            if rows is None:
                return CheckResult("batch division", False, f"eval split {ref!r} unreadable",
                                   "check_eval_split says why.")
            if rows % (ev * n):
                return CheckResult(
                    "batch division", False,
                    f"{rows} held-out rows do not split over per-device eval {ev} x {n} GPUs",
                    "The distributed eval loader would repeat rows to fill the last round, and "
                    "the per-modality eval CE would count them.")
            notes.append(f"eval {rows} = {ev} x {n} x {rows // (ev * n)}")
    return CheckResult("batch division", True, "; ".join(notes) or "no total_batch_size (nothing rounded)")


def _eval_row_count(ref, root) -> Optional[int]:
    from eo.data.eval_split import default_split_path, load_eval_rows
    p = Path(str(ref))
    if not p.suffix:
        p = default_split_path(str(ref))
    elif not p.is_absolute():
        p = REPO / p
    try:
        return len(load_eval_rows(path=p, root_dir=root))
    except (FileNotFoundError, RuntimeError):
        return None


def check_gpu_identity(ctx: PreflightContext) -> CheckResult:
    if not ctx.check_torch:
        return CheckResult("gpu identity", True, "skipped (check_torch=False)")
    try:
        info = gpu_mod.assert_selected(ctx.gpu_alias, ctx.n_gpus)
    except Exception as exc:
        return CheckResult("gpu identity", False, str(exc).splitlines()[0],
                           "\n".join(str(exc).splitlines()[1:]))
    cc = f"sm_{info.compute_capability[0]}{info.compute_capability[1]}"
    which = "cuda:0 is" if ctx.n_gpus == 1 else f"cuda:0..{ctx.n_gpus - 1} are each"
    return CheckResult("gpu identity", True,
                       f"{which} {info.name} ({cc}, {info.memory_total_mib} MiB) "
                       f"= alias {ctx.gpu_alias!r}")


def check_gpu_free_memory(ctx: PreflightContext) -> CheckResult:
    """The selected GPU must be idle enough to hold the run.

    ⚠ THIS WORKSTATION IS NOT EXCLUSIVELY THIS PROJECT'S. Measured 2026-09-23:
    an unrelated job (`deepreefmap_fp32.py`) held 9.58 GiB of the TITAN V, and
    a launch into the remaining 2.2 GiB died with a CUDA OOM at the first step.
    Nothing else in preflight would have caught it -- `check_gpu_identity`
    reports the card's TOTAL memory and never its free memory.

    The consequence scales with the run: a 6-day arm that OOMs at step 1 costs
    a restart, but one that starts while a competing job is between allocations
    can die hours in, which is how a week disappears.

    Uses nvidia-smi rather than torch, so it runs before CUDA is initialised and
    can also name the processes that are in the way.
    """
    if not ctx.check_torch:
        return CheckResult("gpu free memory", True, "skipped (check_torch=False)")

    import subprocess
    want = ctx.min_free_gpu_mib if ctx.min_free_gpu_mib is not None else MIN_FREE_GPU_MIB
    free_by = []                                   # (index, free, total), every card the run uses
    try:
        for g in gpu_mod.resolve_n(ctx.gpu_alias, max(ctx.n_gpus, 1)):
            q = subprocess.run(
                ["nvidia-smi", f"--id={g.index}", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True)
            used, total = (int(x) for x in q.stdout.strip().split(","))
            free_by.append((g.index, total - used, total))
    except Exception as exc:                       # noqa: BLE001
        return CheckResult("gpu free memory", False, f"could not query nvidia-smi: {exc}")

    idx, free, total = min(free_by, key=lambda t: t[1])
    if free >= want:
        on = "" if len(free_by) == 1 else f" on the fullest of {len(free_by)} cards"
        return CheckResult("gpu free memory", True, f"{free} MiB free of {total}{on} (need {want})")

    who = ""
    try:
        procs = subprocess.run(
            ["nvidia-smi", f"--id={idx}", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if procs:
            who = " In use by: " + "; ".join(procs.splitlines())
    except Exception:                              # noqa: BLE001
        pass
    return CheckResult(
        "gpu free memory", False,
        f"only {free} MiB free of {total} on {ctx.gpu_alias} (need {want})",
        f"Another process is using this card.{who} "
        f"Wait for it, or launch on the other GPU with --gpu.")


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
    root, err = large_output_root(ctx)
    if root is None:
        return CheckResult("MOR_SAVE_DIR", False, err)
    if not _under(p, root):
        return CheckResult("MOR_SAVE_DIR", False, f"{p} is not under {root}",
                           "Large outputs go on /data, never /home (CLAUDE.md); on RCP, "
                           "under the private root.")
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
    root, err = large_output_root(ctx)
    if root is None:
        return CheckResult("HF_HOME", False, err)
    if not _under(p, root):
        return CheckResult("HF_HOME", False, f"{p} is not under {root}")
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
    root, err = large_output_root(ctx)
    if root is None:
        return CheckResult("W&B", False, err)
    if wandb_dir and not _under(Path(wandb_dir), root):
        return CheckResult("W&B", False, f"WANDB_DIR={wandb_dir} is not under {root}")
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

    cfg = _load_config(path, ctx.overrides)

    problems = []
    vocab = (cfg.get("model_config") or {}).get("vocab_size")
    if cfg.get("dataset") not in EO_DATASETS:
        problems.append(f"dataset {cfg.get('dataset')!r} is not an EO dataset {EO_DATASETS}")
    else:
        # The registry is the DATASET's: eo_vocab for TerraMesh, ben_vocab (eo_vocab + 21
        # appended ids) for a BigEarthNet fine-tune.
        from lm_dataset.modality_registry import get_registry
        expected = get_registry(cfg).TOTAL_VOCAB_SIZE
        if vocab != expected:
            problems.append(f"vocab_size {vocab} != registry {expected}")
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



def check_eval_split(ctx: PreflightContext) -> CheckResult:
    """The configured held-out split exists and still describes this data.

    ⚠ Why this is a launch-time check rather than a load-time one. The split is
    a list of integer row indices into tok_index.parquet's row order. If that
    file is ever regenerated the indices still resolve, still name real rows,
    and silently designate DIFFERENT scenes -- so a run would train on rows it
    reports as held out, and every eval number would be a training number. The
    artifact records SHA-256 of both parquet inputs; this verifies them before
    a multi-day run starts rather than after.

    A null eval_split is legitimate (the Phase 2 behaviour: no split, no
    evaluation) and passes with a note, because silently accepting "no
    evaluation" is a different failure from failing to find a split.
    """
    if ctx.config_path is None:
        return CheckResult("eval split", True, "skipped (no --config given)")
    path = ctx.config_path
    if not path.is_file():
        return CheckResult("eval split", True, "skipped (config missing; check_config reports it)")

    cfg = _load_config(path, ctx.overrides)
    if cfg.get("dataset") == "ben_multimodal":
        return CheckResult("eval split", True, "n/a: a BEN run evaluates on ben.eval_role ('BEN data')")
    ref = (cfg.get("multimodal") or {}).get("eval_split")
    if ref is None:
        return CheckResult("eval split", True,
                           "none configured -- this run has NO held-out evaluation")

    from eo.data.eval_split import default_split_path, load_eval_rows

    split_path = Path(str(ref))
    if not split_path.suffix:
        split_path = default_split_path(str(ref))
    elif not split_path.is_absolute():
        split_path = REPO / split_path

    root = ctx.env.get("TERRAMESH_TOK_ROOT")
    try:
        rows = load_eval_rows(path=split_path, root_dir=root)
    except FileNotFoundError:
        return CheckResult("eval split", False, f"{split_path} does not exist")
    except RuntimeError as e:
        return CheckResult("eval split", False, str(e).splitlines()[0])

    return CheckResult("eval split", True, f"{split_path.name}: {len(rows)} held-out rows")


def check_token_set(ctx: PreflightContext) -> CheckResult:
    """`multimodal.token_set` resolves to a usable token root (2026-10-11).

    The dataloader resolves it too (load_dataset._build_terramesh) and raises
    on the same faults, but only after the model is built; this says so before
    launch, and also checks that the resolved root carries every modality --
    a cleaned root is mostly symlinks, and a broken one would be found at load.
    """
    if ctx.config_path is None:
        return CheckResult("token set", True, "skipped (no --config given)")
    if not ctx.config_path.is_file():
        return CheckResult("token set", True, "skipped (config missing; check_config reports it)")
    raw = ctx.env.get("TERRAMESH_TOK_ROOT")
    if not raw:
        return CheckResult("token set", True, "skipped (TERRAMESH_TOK_ROOT unset; reported above)")

    from eo.terramesh_tok.clean import resolve_token_root
    from eo.terramesh_tok.contract import tok_dir_name
    from eo.data.eo_vocab import DEFAULT_ACTIVE_MODALITIES

    cfg = _load_config(ctx.config_path, ctx.overrides)
    if cfg.get("dataset") == "ben_multimodal":
        return CheckResult("token set", True, "n/a: a BEN run reads BigEarthNet tokens ('BEN data')")
    token_set = (cfg.get("multimodal") or {}).get("token_set", "original")
    try:
        root = resolve_token_root(raw, token_set)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return CheckResult("token set", False, str(e).splitlines()[0])
    missing = [m for m in DEFAULT_ACTIVE_MODALITIES
               if not (root / tok_dir_name(m) / "tokens.npy").is_file()]
    if not (root / "tok_index.parquet").is_file():
        missing.append("tok_index.parquet")
    if missing:
        return CheckResult("token set", False, f"{root}: missing {missing}")
    return CheckResult("token set", True, f"{token_set} -> {root}")


#: metadata.json fields that must agree between a train root and val's: the same tokenizer,
#: crop, grid, standardization and LULC encoding, i.e. the same contract.
CONTRACT_FIELDS = ("tokenizer", "codebook_size", "crop", "tokens_per_sample", "flatten_order",
                   "standardization", "lulc_one_hot", "nan_policy")


def check_train_root(ctx: PreflightContext) -> CheckResult:
    """`multimodal.split: train` reads its own consolidated root (2026-10-25).

    The dataloader resolves it too (train_split.resolve_train_root), but only after the model is
    built. This also checks what that cannot: every active modality is row-aligned with the
    index, S1RTC and S1GRD split the rows exactly, and each artifact records val's contract
    (tokenizer, crop, standardization...), so train and held-out tokens come from one tokenizer.
    """
    if ctx.config_path is None or not ctx.config_path.is_file():
        return CheckResult("train root", True, "skipped (config missing; check_config reports it)")
    cfg = _load_config(ctx.config_path, ctx.overrides)
    mm = cfg.get("multimodal") or {}
    if cfg.get("dataset") == "ben_multimodal" or mm.get("split", "val") != "train":
        return CheckResult("train root", True, "n/a: this run trains on the val split's complement")
    import numpy as np
    import pandas as pd
    from eo.terramesh_tok.clean import resolve_token_root
    from eo.terramesh_tok.contract import tok_dir_name
    from eo.terramesh_tok.train_split import resolve_train_root
    from eo.data.eo_vocab import DEFAULT_ACTIVE_MODALITIES
    token_set = mm.get("token_set", "original")
    try:
        root = resolve_train_root(ctx.env.get("TERRAMESH_TRAIN_TOK_ROOT"), token_set,
                                  allow_partial=bool(mm.get("allow_partial_train_root", False)))
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return CheckResult("train root", False, str(e).splitlines()[0])
    if not (root / "tok_index.parquet").is_file():
        return CheckResult("train root", False, f"{root}/tok_index.parquet missing")
    n = len(pd.read_parquet(root / "tok_index.parquet", columns=["row"]))
    present = {}
    for m in DEFAULT_ACTIVE_MODALITIES:
        d = root / tok_dir_name(m)
        if not all((d / f).is_file() for f in ("tokens.npy", "present.npy", "metadata.json")):
            return CheckResult("train root", False, f"{d}: tokens/present/metadata missing")
        t, p = np.load(d / "tokens.npy", mmap_mode="r"), np.load(d / "present.npy", mmap_mode="r")
        if t.shape[0] != n or p.shape[0] != n:
            return CheckResult("train root", False, f"{m}: {t.shape[0]} token rows / {p.shape[0]} "
                                                    f"presence rows for an index of {n}")
        present[m] = int(p.sum())
    s1 = present.get("S1RTC", 0) + present.get("S1GRD", 0)
    partial_mods = [m for m in DEFAULT_ACTIVE_MODALITIES if m not in ("S1RTC", "S1GRD") and present[m] != n]
    if s1 != n or partial_mods:
        return CheckResult("train root", False,
                           f"presence: S1RTC {present.get('S1RTC')} + S1GRD {present.get('S1GRD')} = {s1} "
                           f"for {n} rows; not every row in {partial_mods}")
    val = ctx.env.get("TERRAMESH_TOK_ROOT")
    if val:
        try:
            vroot = resolve_token_root(val, token_set)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            return CheckResult("train root", False, f"val root for the contract check: {e}")
        for m in DEFAULT_ACTIVE_MODALITIES:
            if m == "Coords":
                continue
            a = json.loads((root / tok_dir_name(m) / "metadata.json").read_text())
            b = json.loads((vroot / tok_dir_name(m) / "metadata.json").read_text())
            bad = [k for k in CONTRACT_FIELDS if a.get(k) != b.get(k)]
            if bad:
                return CheckResult("train root", False, f"{m}: {bad} differ from {vroot}",
                                   "The train tokens must come from val's tokenizer contract.")
    return CheckResult("train root", True, f"{root}: {n} rows (S1RTC {present.get('S1RTC')} / "
                                           f"S1GRD {present.get('S1GRD')}), val's contract")


def check_ben_data(ctx: PreflightContext) -> CheckResult:
    """A BEN fine-tune's data exists, is labelled, and was tokenized for every row it reads.

    The dataset raises on the same faults, but only after the model is built; and a role
    whose rows were never tokenized would otherwise read zero-filled token rows (0 is a
    valid id), which trains without an error.
    """
    if ctx.config_path is None or not ctx.config_path.is_file():
        return CheckResult("BEN data", True, "skipped (no config)")
    cfg = _load_config(ctx.config_path, ctx.overrides)
    if cfg.get("dataset") != "ben_multimodal":
        return CheckResult("BEN data", True, "n/a (not a BEN run)")
    import json as _json
    import numpy as _np
    import pandas as _pd
    from eo.data import ben_vocab as BV
    from eo.terramesh_tok.contract import CROP, TOKENS_PER_SAMPLE, tok_dir_name
    root = Path(ctx.env.get("BEN_ROOT", "/data/enric/data/BigEarthNet"))
    index = root / "index_v1.parquet"
    if not index.is_file():
        return CheckResult("BEN data", False, f"{index} missing", "eo/scripts/build_ben_index.py")
    idx = _pd.read_parquet(index)
    if "labels19" not in idx.columns:
        return CheckResult("BEN data", False, f"{index} has no labels", "build_ben_index.py labels")
    bcfg = cfg.get("ben") or {}
    roles = [bcfg.get("train_role", "ft_train")] + ([bcfg["eval_role"]] if bcfg.get("eval_role") else [])
    problems, counts = [], {}
    for role in roles:
        if role not in idx.columns:
            problems.append(f"no role {role!r} in the index")
            continue
        rows = _np.flatnonzero(idx[role].to_numpy(bool))
        counts[role] = len(rows)
        if not len(rows):
            problems.append(f"role {role} is empty")
        lab = idx.labels19.to_numpy()[rows]
        if (lab < 0).any():                    # 0 = an empty 19-class set: legitimate (SMARTIES keeps it)
            problems.append(f"{role}: {int((lab < 0).sum())} rows unlabelled (never packed)")
        for src, mod in BV.INPUT_MODALITIES.items():
            d = root / tok_dir_name(mod)
            if not (d / "present.npy").is_file() or not (d / "metadata.json").is_file():
                problems.append(f"{d} not tokenized")
                continue
            meta = _json.loads((d / "metadata.json").read_text())
            if meta.get("tokenizer_modality") != mod or int(meta.get("crop", -1)) != CROP \
                    or int(meta.get("tokens_per_sample", -1)) != TOKENS_PER_SAMPLE:
                problems.append(f"{d}: metadata {meta.get('tokenizer_modality')}/{meta.get('crop')} "
                                f"is not {mod}/{CROP}")
            pres = _np.load(d / "present.npy")
            if len(pres) != len(idx):
                problems.append(f"{d}: {len(pres)} rows, index has {len(idx)}")
            elif not pres[rows].all():
                problems.append(f"{role}: {int((~pres[rows]).sum())} rows not tokenized in {mod}")
    # L10 (PHASE6_PLAN §2c): a generated-S2 token dir covering every train and eval row
    conds = list(bcfg.get("conditions") or ["S1", "S2", "S1S2"])
    if "S1G2" in conds:
        g = bcfg.get("gen_s2")
        d = root / str(g)
        if not g or not (d / "present.npy").is_file() or not (d / "metadata.json").is_file():
            problems.append(f"condition S1G2 without a generated S2 dir ({d})")
        else:
            meta = _json.loads((d / "metadata.json").read_text())
            pres = _np.load(d / "present.npy")
            if not meta.get("generated") or meta.get("tokenizer_modality") != "S2L2A":
                problems.append(f"{d}: not a generated S2L2A dir")
            for role in roles:
                if role in idx.columns and not pres[idx[role].to_numpy(bool)].all():
                    problems.append(f"{d}: {int((~pres[idx[role].to_numpy(bool)]).sum())} {role} rows not generated")
    # L11b (PHASE6_PLAN §2c): the unlabelled pool carries NO true label, and the teacher covers it
    unl = bcfg.get("unlab")
    if unl:
        role = unl.get("role")
        if role not in idx.columns:
            problems.append(f"no unlabelled role {role!r} in the index")
        else:
            rows = _np.flatnonzero(idx[role].to_numpy(bool))
            counts[role] = len(rows)
            lab = idx.labels19.to_numpy()[rows]
            if (lab >= 0).any():
                problems.append(f"{role}: {int((lab >= 0).sum())} rows carry TRUE labels (a distillation pool must not)")
            t = root / str(unl.get("teacher"))
            if not t.with_suffix(".npz").is_file() or not t.with_suffix(".json").is_file():
                problems.append(f"teacher blocks {t}.npz/.json missing (eo/scripts/ben_teacher.py)")
            else:
                tz = _np.load(t.with_suffix(".npz"))
                tm = _json.loads(t.with_suffix(".json").read_text())
                if not _np.array_equal(_np.sort(tz["rows"]), rows):
                    problems.append(f"{t.name}: its rows are not {role}'s")
                if tm.get("plumbing") is None or not tm["plumbing"].get("passed") or not tm["plumbing"].get("control_fired"):
                    problems.append(f"{t.name}: no passing plumbing check")
            d = root / tok_dir_name(BV.INPUT_MODALITIES["S1"])
            if (d / "present.npy").is_file() and len(rows) and not _np.load(d / "present.npy")[rows].all():
                problems.append(f"{role}: rows not tokenized in S1GRD")
    if problems:
        return CheckResult("BEN data", False, "; ".join(problems[:3]))
    return CheckResult("BEN data", True, f"{root.name}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def check_init_from(ctx: PreflightContext) -> CheckResult:
    """A fine-tune starts from an EXTENDED checkpoint whose identity check passed (§4.9).

    Not a resume: the resume checks refuse a partial load, which loading across a vocabulary
    change would be. This verifies instead that the starting weights are exactly the
    ones extend_checkpoint.py proved identical to the pretrained model on every old id.
    """
    if ctx.config_path is None or not ctx.config_path.is_file():
        return CheckResult("init from", True, "skipped (no config)")
    cfg = _load_config(ctx.config_path, ctx.overrides)
    init = (cfg.get("finetune") or {}).get("init_from")
    regime = (cfg.get("finetune") or {}).get("regime")
    is_ben = cfg.get("dataset") == "ben_multimodal"
    if regime == "scratch":
        if init not in (None, "???"):
            return CheckResult("init from", False, f"regime scratch with init_from {init}",
                               "The from-scratch reference loads nothing; drop --init-from.")
        return CheckResult("init from", True, "regime scratch: random init, deliberately (PHASE6_PLAN L4)")
    if init in (None, "???"):
        if is_ben:
            return CheckResult("init from", False, "a BEN run has no finetune.init_from",
                               "Launch with train_eo.sh --init-from <extended checkpoint>.")
        return CheckResult("init from", True, "n/a (not a fine-tune)")
    if ctx.resuming:
        return CheckResult("init from", False, "--resume of a fine-tune is not supported",
                           "A fine-tune is short; rerun it. A resume would re-run init_from "
                           "and then load the run's checkpoint over it.")
    from eo.finetune.init import verify_extension
    vocab = (cfg.get("model_config") or {}).get("vocab_size")
    if vocab is None:
        return CheckResult("init from", False, "the config has no model_config.vocab_size")
    try:
        doc = verify_extension(Path(str(init)), int(vocab))
    except (RuntimeError, FileNotFoundError, KeyError) as e:
        return CheckResult("init from", False, str(e).splitlines()[0])
    return CheckResult("init from", True,
                       f"{Path(str(init)).name}: identity max |dlogit| {doc['identity']['max_abs_diff']:.1e}, "
                       f"controls fired, sha {doc['weights_sha256'][:10]}")


def _load_config(path: Path, overrides: Optional[Dict[str, object]] = None) -> dict:
    """The config the RUN will actually see, with CLI overrides applied.

    Without the overrides this validates a file nobody actually runs: every
    launch may pass `mor.enable=false`, `deepspeed=...` and so on, and a check
    that reads only the file would pass while the run does something else.

    ⚠ And without Hydra COMPOSITION it validates a file nobody runs either.
    The Phase 3 arm configs (`arm_a_mor`, `arm_b_vanilla`) are thin overlays --
    a `defaults:` list plus the two keys that define the arm -- so reading the
    raw YAML sees no `vocab_size`, no `dataset` and no `precision`, and
    preflight refused to launch a config that is in fact correct. Found
    2026-09-22 when the arm configs were introduced. Compose the way
    `pretrain.py` does, so this checks the run rather than the file.
    """
    import yaml
    raw = yaml.safe_load(path.read_text()) or {}

    if "defaults" in raw:
        cfg = _compose_config(path, overrides)
        if cfg is not None:
            return cfg
        # Composition failed: fall through, but strip `defaults` so the caller
        # reports the MISSING KEYS rather than silently passing a partial file.
        raw.pop("defaults", None)

    for dotted, value in (overrides or {}).items():
        node, keys = raw, dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return raw


def _compose_config(path: Path, overrides: Optional[Dict[str, object]] = None):
    """Resolve a config through Hydra exactly as pretrain.py will.

    Returns a plain dict, or None if Hydra is unavailable or composition fails
    (the caller then falls back and reports missing keys rather than passing).

    ⚠ `resolve=False`. The EO config interpolates `${oc.env:WANDB_ENTITY}`, and
    resolving here would raise in any shell that has not loaded `.env` -- which
    is precisely the shell preflight exists to diagnose. check_wandb reports
    that missing variable properly; this must not pre-empt it with a crash.
    """
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError:
        return None

    # conf/pretrain_vision/<group>/<name>.yaml -> config_dir is the directory
    # holding the file, which is also what its `defaults` entries resolve
    # against.
    # ⚠ Overrides arrive already parsed into Python values (the CLI runs them
    # through yaml.safe_load so bools and ints are real). Hydra's grammar wants
    # the ORIGINAL text back: Python's False renders as "False", which Hydra
    # reads as a string and then refuses to assign to a bool node. Render them
    # back to Hydra's spelling rather than str()-ing them.
    def _hydra_value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "null"
        return str(v)

    try:
        ov = [f"{k}={_hydra_value(v)}" for k, v in (overrides or {}).items()]
        # ⚠ must be ABSOLUTE; train_eo.sh passes a repo-relative path.
        # ⚠ Compose from conf/pretrain_vision with the group-qualified name, exactly as
        #   pretrain.py does (--config-path conf/pretrain_vision --config-name <group>/<name>),
        #   whenever the file lives there. Composing from the file's own directory resolves
        #   relative defaults identically, but cannot see another group: the Phase 6 configs
        #   (eo_ben/*) default to /eo_terramesh/arm_b10_vanilla, and composition failed
        #   (2026-10-12). Files elsewhere (the controls' temp copies) keep the old way.
        root = (REPO / "conf/pretrain_vision").resolve()
        rp = path.resolve()
        if root in rp.parents:
            cdir, cname = root, str(rp.relative_to(root).with_suffix(""))
        else:
            cdir, cname = rp.parent, path.stem
        with initialize_config_dir(config_dir=str(cdir), version_base=None):
            cfg = compose(config_name=cname, overrides=ov)
        return OmegaConf.to_container(cfg, resolve=False)
    except Exception as e:
        # ⚠ Never fail silently here. A silent fallback made this function look
        # like it worked while check_config reported nonsense about a config
        # that was actually fine.
        print(f"  (preflight: Hydra composition of {path.name} failed, falling back to "
              f"raw YAML -- {type(e).__name__}: {str(e).splitlines()[0][:120]})", file=sys.stderr)
        return None


def _latest_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpts = [d for d in run_dir.glob("checkpoint-*") if d.is_dir() and d.name[11:].isdigit()]
    return max(ckpts, key=lambda d: int(d.name[11:])) if ckpts else None


def _checkpoint_param_keys(ckpt: Path) -> Optional[List[str]]:
    """Parameter names in a checkpoint, without materializing the tensors."""
    st = ckpt / "model.safetensors"
    if st.is_file():
        from safetensors import safe_open
        with safe_open(str(st), framework="pt") as f:
            return list(f.keys())
    binp = ckpt / "pytorch_model.bin"
    if binp.is_file():
        import torch
        return list(torch.load(binp, map_location="cpu", mmap=True, weights_only=True).keys())
    return None


def _arm_name(recursive: bool, mor: bool, rand: bool = False) -> str:
    if mor:
        return "MoR (random router)" if rand else "MoR"
    return "recursive-no-router" if recursive else "vanilla"


def _checkpoint_arm(ckpt: Path, keys: List[str], cfg: dict) -> "tuple[bool, bool, bool]":
    """(recursive, mor, random router) as the checkpoint ON DISK actually is.

    Measured on real checkpoints from all three arms, 2026-09-21:

        arm                   format        tensors  lm_head  mor_router  layers 1/10/19
        vanilla               safetensors       263  absent   no          differ
        recursive-no-router   bin               264  present  no          IDENTICAL
        MoR                   bin               266  present  yes         n/a (block_list)

    Two things this had to learn the hard way:

      * `block_list` appears only under MoR. It is created by the MoR transform,
        not by `sharing_strategy`, so the recursive-no-router arm keeps ordinary
        `model.layers.N.*` names and is indistinguishable from vanilla BY NAME.
        A name-only fingerprint mislabels it as vanilla -- it did, on the first
        attempt here, and the positive control is what caught that.
      * What actually defines "recursive" is that the shared layers ARE THE SAME
        TENSOR. So test that directly: under `middle_cycle` with base_depth
        (L-2)//num_recursion, layer 1 and layer 1+base_depth are one tensor.

    (`lm_head.weight` present-vs-absent also separates them, because
    `sharing_strategy` unties the head -- worklog open item 26. That is a
    symptom of a bug rather than a definition, so it is not used as the test.)

    ⚠ `mor.rand_router=true` (Step 4's random-router control, 2026-10-11): the
    MoR transform runs, so the keys are `block_list` ones, but there is no
    `mor_router` (264 tensors against MoR's 266; measured on a built model).
    Without this branch it fingerprinted as vanilla and every resume was
    refused; treated as plain MoR, a learned-router run could resume from it as
    a partial load. Hence the third component.
    """
    if any("mor_router" in k for k in keys):
        return True, True, False   # MoR implies recursive sharing here
    if any(".block_list." in k for k in keys):
        return True, True, True    # the MoR transform without a learned router

    n_layers = (cfg.get("model_config") or {}).get("num_hidden_layers")
    n_rec = (cfg.get("recursive") or {}).get("num_recursion")
    if not n_layers or not n_rec or n_rec < 2:
        return False, False, False  # cannot tell; treat as non-recursive
    base_depth = (int(n_layers) - 2) // int(n_rec)
    a = f"model.layers.1.self_attn.q_proj.weight"
    b = f"model.layers.{1 + base_depth}.self_attn.q_proj.weight"
    if a not in keys or b not in keys:
        return False, False, False

    import torch
    st = ckpt / "model.safetensors"
    if st.is_file():
        from safetensors import safe_open
        with safe_open(str(st), framework="pt") as f:
            shared = torch.equal(f.get_tensor(a), f.get_tensor(b))
    else:
        sd = torch.load(ckpt / "pytorch_model.bin", map_location="cpu",
                        mmap=True, weights_only=True)
        shared = torch.equal(sd[a], sd[b])
    return bool(shared), False, False


def check_resume_checkpoint(ctx: PreflightContext) -> CheckResult:
    """On --resume, the checkpoint must be COMPLETE and match this arm.

    Both failure modes here are silent, and they compose into a third that is
    worse than either (all measured 2026-09-21, Phase 3 Step 0.4):

      * `optimizer.pt` missing    -> HF resumes with FRESH Adam moments. No
        warning, no error, exit 0. The loss curve looks continuous because the
        weights are right; only the second-moment estimates are gone.
      * architecture mismatch     -> the model load is a PARTIAL load. HF prints
        "missing keys"/"unexpected keys" as warnings and carries on. It is
        caught today only INCIDENTALLY, by the optimizer's param-group size
        check -- which is to say, only when optimizer.pt happens to be present.
      * both together             -> a run that prints "Continuing training from
        global step N", trains happily, and is a randomly-initialized model with
        the LR schedule fast-forwarded. Measured: loss 11.24 against
        ln(87,556) = 11.38, i.e. untrained, reported as a continuation.

    So this checks presence AND identity, because presence alone is what the
    optimizer's accidental guard already gave us.
    """
    if not ctx.resuming:
        return CheckResult("resume checkpoint", True, "not resuming")

    ckpt = _latest_checkpoint(ctx.run_dir)
    if ckpt is None:
        return CheckResult("resume checkpoint", False, f"no checkpoint-* under {ctx.run_dir}",
                           "Nothing to resume from. HF would start from scratch.")

    required = ["optimizer.pt", "scheduler.pt", "trainer_state.json"]
    missing = [f for f in required if not (ckpt / f).is_file()]
    world, why = checkpoint_world_size(ckpt)
    if world is None:
        missing.append(f"rng_state ({why})")
    if missing:
        return CheckResult(
            "resume checkpoint", False, f"{ckpt.name} is missing {missing}",
            "A checkpoint missing optimizer.pt resumes SILENTLY with fresh Adam moments "
            "-- no warning, exit 0. Missing scheduler/rng state loses the LR position and "
            "the data order.")

    # ⚠ The GPU split must be the one the checkpoint was written with. HF looks for
    # rng_state_<rank>.pth (N GPUs) or rng_state.pth (one), and a missing file is
    # logged at INFO and skipped; the data are also resharded. And on resume HF takes
    # the per-device batch from the checkpoint's trainer_state (`train_batch_size`)
    # while the accumulation comes from the new config, so a changed per-device batch
    # trains at a different total batch. All exit 0.
    if world != ctx.n_gpus:
        return CheckResult(
            "resume checkpoint", False,
            f"{ckpt.name} was written by {world} GPU(s); this run uses {ctx.n_gpus}",
            "Resume with the same --gpus. Another count restores no RNG state and reshards "
            "the data, silently.")
    import json
    cfg = _load_config(ctx.config_path, ctx.overrides) if ctx.config_path else {}
    try:
        saved_per = json.loads((ckpt / "trainer_state.json").read_text()).get("train_batch_size")
    except (ValueError, OSError):
        saved_per = None
    per = cfg.get("per_device_train_batch_size")
    if saved_per is not None and per is not None and int(saved_per) != int(per):
        return CheckResult(
            "resume checkpoint", False,
            f"{ckpt.name} trained at per-device batch {saved_per}; the config says {per}",
            "On resume HF keeps the checkpoint's per-device batch but the config's "
            "accumulation, so the total batch would change silently.")

    keys = _checkpoint_param_keys(ckpt)
    if keys is None:
        return CheckResult("resume checkpoint", False,
                           f"{ckpt.name} has neither pytorch_model.bin nor model.safetensors")

    want_recursive = bool((cfg.get("recursive") or {}).get("enable"))
    want_mor = bool((cfg.get("mor") or {}).get("enable"))
    want_rand = want_mor and bool((cfg.get("mor") or {}).get("rand_router"))
    got_recursive, got_mor, got_rand = _checkpoint_arm(ckpt, keys, cfg)

    if (got_recursive, got_mor, got_rand) != (want_recursive, want_mor, want_rand):
        return CheckResult(
            "resume checkpoint", False,
            f"{ckpt.name} is a {_arm_name(got_recursive, got_mor, got_rand)} checkpoint but this run "
            f"is {_arm_name(want_recursive, want_mor, want_rand)} "
            f"(checkpoint recursive={got_recursive}/mor={got_mor}/rand={got_rand}, "
            f"config recursive={want_recursive}/mor={want_mor}/rand={want_rand})",
            "Resuming across arms is a PARTIAL model load that HF reports as a warning. "
            "It is caught today only by the optimizer's param-group check, i.e. only when "
            "optimizer.pt is present. Do not rely on that.")

    step = ckpt.name[11:]
    fmt = "safetensors" if (ckpt / "model.safetensors").is_file() else "bin"
    return CheckResult("resume checkpoint", True,
                       f"{ckpt.name} complete ({fmt}, step {step}, {len(keys)} tensors, "
                       f"{world} GPU(s), per-device {saved_per})")


#: Evaluated in order. Environment checks come first so a misconfigured shell
#: fails before torch is imported and CUDA is initialized.
CHECKS: List[Callable[[PreflightContext], CheckResult]] = [
    check_device_order,
    check_visible_devices,
    check_multi_gpu,
    check_save_dir,
    check_data_root,
    check_hf_home,
    check_wandb,
    check_run_dir,
    check_disk,
    check_config,
    check_batch_division,
    check_eval_split,
    check_token_set,
    check_train_root,
    check_ben_data,
    check_init_from,
    check_gpu_free_memory,
    check_resume_checkpoint,
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
    ap.add_argument("--gpus", type=int, default=1,
                    help="Cards the run is launched for (train_eo.sh --gpus); > 1 is torchrun DDP.")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--allow-existing", action="store_true",
                    help="Set by --resume/--force in the launcher.")
    ap.add_argument("--resuming", action="store_true",
                    help="Set by --resume. Triggers the checkpoint completeness check.")
    ap.add_argument("--override", action="append", default=[], metavar="k.v=VALUE",
                    help="A Hydra override the run will use, so the config check "
                         "validates what actually runs. Repeatable.")
    ap.add_argument("--no-torch", action="store_true",
                    help="Skip the CUDA-touching check (for tests).")
    args = ap.parse_args(argv)

    import yaml as _yaml
    overrides = {}
    for item in args.override:
        if "=" not in item:
            continue
        k, _, v = item.partition("=")
        try:
            overrides[k] = _yaml.safe_load(v)     # gives real bools/ints, not strings
        except Exception:
            overrides[k] = v

    ctx = PreflightContext(
        gpu_alias=args.gpu,
        run_dir=args.run_dir,
        config_path=args.config,
        allow_existing=args.allow_existing,
        resuming=args.resuming,
        overrides=overrides,
        check_torch=not args.no_torch,
        n_gpus=args.gpus,
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
