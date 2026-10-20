# eo/train/distributed.py
"""Multi-GPU (DDP) rules for the EO training path (2026-10-20).

ONE GPU IS THE DEFAULT. Several GPUs are opt-in, `train_eo.sh --gpus N`, and only
on one node of identical cards (an RCP pod with `-g N`). The launcher then starts
`pretrain.py` under torchrun, one process per card; HF Trainer sees WORLD_SIZE and
wraps the model in DistributedDataParallel. Nothing about the model or the loss
changes: each process takes 1/N of every optimizer step's micro-batches, and the
gradients are averaged at the last one.

WHY torchrun AND NOT `accelerate launch`. `pretrain.py` keys `save_safetensors` on
the launcher, so under accelerate a vanilla model would save pytorch_model.bin
instead of model.safetensors: the same arm would checkpoint differently by GPU
count. torchrun sets no ACCELERATE_* variable, so everything pretrain.py derives
from the launcher is what a single-GPU run gets.

What this module holds is the part that is easy to get silently wrong:

  * `check_launch`: a process that sees more cards than the launch accounts for
    gets nn.DataParallel from HF (on the workstation, across a TITAN V and a
    Titan X) and a gradient_accumulation_steps computed from the wrong device
    count. Neither raises. pretrain.py calls this for every EO run.
  * `checkpoint_world_size`: HF saves the RNG state per process
    (`rng_state.pth` on one GPU, `rng_state_<rank>.pth` on N) and, on resume,
    a missing file is logged at INFO and skipped. Resuming at another GPU count
    therefore restores no RNG state and reshards the data, with exit 0.
    Preflight's resume check reads the count from the files.

Pure functions of their inputs, so the gate can feed them broken cases without
a GPU.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Optional, Tuple

_RANK_RNG = re.compile(r"^rng_state_(\d+)\.pth$")


def launch_world_size(env: Mapping[str, str]) -> int:
    """WORLD_SIZE as torchrun sets it; 1 for a plain `python` launch."""
    return int(env.get("WORLD_SIZE") or 1)


def check_launch(device_count: int, env: Mapping[str, str]) -> None:
    """Raise unless this process's view of the GPUs matches how it was launched.

    * plain launch: exactly one visible device (more means nn.DataParallel);
    * torchrun: every process sees all N cards of ONE node, WORLD_SIZE == N, and
      `util.config.preprocess_config` then computes the accumulation for N.
    """
    world = launch_world_size(env)
    vis = env.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if world == 1:
        if device_count > 1:
            raise RuntimeError(
                f"{device_count} CUDA devices visible to a single-process launch "
                f"(CUDA_VISIBLE_DEVICES={vis!r}). HF Trainer would wrap the model in "
                f"nn.DataParallel and compute gradient_accumulation_steps for "
                f"{device_count} devices, silently. Launch through eo/scripts/train_eo.sh "
                f"(--gpus N starts torchrun), or make one device visible.")
        return
    local = int(env.get("LOCAL_WORLD_SIZE") or world)
    if local != world:
        raise RuntimeError(
            f"WORLD_SIZE {world} spans more than one node (LOCAL_WORLD_SIZE {local}). "
            f"Multi-GPU is supported on one node only.")
    if device_count != world:
        raise RuntimeError(
            f"torchrun started {world} processes but each sees {device_count} CUDA devices "
            f"(CUDA_VISIBLE_DEVICES={vis!r}). preprocess_config divides the batch by the "
            f"device count, so the accumulation would be computed for {device_count}.")


def checkpoint_world_size(ckpt: Path) -> Tuple[Optional[int], str]:
    """(GPU count that wrote `ckpt`, problem). The count is None when unreadable.

    One GPU writes `rng_state.pth`; N write `rng_state_0.pth` .. `rng_state_{N-1}.pth`.
    """
    single = (ckpt / "rng_state.pth").is_file()
    ranks = sorted(int(m.group(1)) for p in ckpt.iterdir() if p.is_file()
                   for m in [_RANK_RNG.match(p.name)] if m)
    if single and ranks:
        return None, f"both rng_state.pth and rng_state_<rank>.pth ({len(ranks)}) present"
    if single:
        return 1, ""
    if not ranks:
        return None, "no rng_state file"
    if ranks != list(range(len(ranks))):
        return None, f"rng_state ranks {ranks} are not 0..{len(ranks) - 1}"
    return len(ranks), ""
