# eo/train/gpu.py
"""Resolve a GPU by NAME rather than by index, and prove the resolution.

WHY THIS EXISTS. This workstation has two heterogeneous cards, and the two
tools that name them DISAGREE about which one is "0":

    NVIDIA TITAN V         (Volta,   sm_70, fp16 tensor cores)   nvidia-smi 1
    NVIDIA GeForce TITAN X (Maxwell, sm_52, no tensor cores)     nvidia-smi 0

`nvidia-smi` enumerates in PCI bus order. CUDA defaults to
`CUDA_DEVICE_ORDER=FASTEST_FIRST`, which puts the TITAN V first -- so
`CUDA_VISIBLE_DEVICES=0` is the TITAN V while `nvidia-smi` calls it 1. Measured
2026-09-20. Picking the wrong one is a ~2.4x slowdown (5.30 vs 12.5 TFLOP/s
fp32) that produces no error and no warning; it just takes two and a half times
as long, which on a multi-day run is indistinguishable from "training is slow".

The fix is two-part and both parts matter:

  1. Pin `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the two tools agree from now on.
     After that, TITAN V is index 1 in BOTH -- matching `nvidia-smi`, which is
     what you type when you check on a run.
  2. Never write the index down anyway. Ask for `titanv`, resolve it here, and
     then ASSERT that the device CUDA actually handed us has the expected name
     and compute capability. An index is a fact about today's hardware; a name
     plus a compute capability is a fact about what the run needs.

Step 2 of the phase plan (fp16 via autocast) depends on sm_70 specifically, so
the capability assertion is not decoration -- it is the check that stops an
fp16 run from silently landing on a card with no fp16 tensor cores.

Imports torch lazily: `list_gpus`/`resolve` shell out to `nvidia-smi` and are
usable before CUDA is initialized, which is exactly when the selection is made.
"""
from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: The device order this project pins. Every launch sets it, so that an index
#: printed by `nvidia-smi` and an index in `CUDA_VISIBLE_DEVICES` mean the same
#: card. Do not "simplify" this away -- see the module docstring.
DEVICE_ORDER = "PCI_BUS_ID"


@dataclass(frozen=True)
class GpuSpec:
    """What an alias promises about the hardware behind it."""
    alias: str
    name_contains: str
    compute_capability: Tuple[int, int]
    note: str


#: Aliases are matched case-insensitively as substrings of the reported name and
#: are then confirmed against the compute capability. The capability is what
#: disambiguates near-identical marketing names -- a Maxwell TITAN X (sm_52) and
#: a Pascal TITAN Xp (sm_61) both contain "TITAN X".
GPU_ALIASES: Dict[str, GpuSpec] = {
    "titanv": GpuSpec(
        alias="titanv",
        name_contains="TITAN V",
        compute_capability=(7, 0),
        note="Volta. fp16 tensor cores (90.1 TFLOP/s fp16, 12.5 fp32). The training card.",
    ),
    "titanx": GpuSpec(
        alias="titanx",
        name_contains="GeForce GTX TITAN X",
        compute_capability=(5, 2),
        note="Maxwell. No tensor cores (5.30 TFLOP/s fp32, 4.44 fp16 -- fp16 is SLOWER here). "
             "Reserved for eval/generation so they never steal training time.",
    ),
}


@dataclass(frozen=True)
class GpuInfo:
    """One physical GPU, as `nvidia-smi` reports it (i.e. in PCI bus order)."""
    index: int
    name: str
    compute_capability: Tuple[int, int]
    memory_total_mib: int


def list_gpus() -> List[GpuInfo]:
    """Every visible GPU, in PCI bus order.

    Uses `nvidia-smi` rather than torch on purpose: this runs *before* the
    process decides which device to expose, and importing torch first would
    initialize CUDA against whatever the ambient environment happens to say.
    """
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,compute_cap,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    gpus: List[GpuInfo] = []
    for line in out.splitlines():
        idx, name, cap, mem = (f.strip() for f in line.split(","))
        major, _, minor = cap.partition(".")
        gpus.append(GpuInfo(
            index=int(idx),
            name=name,
            compute_capability=(int(major), int(minor)),
            memory_total_mib=int(float(mem)),
        ))
    return gpus


def resolve(alias: str, gpus: Optional[List[GpuInfo]] = None) -> GpuInfo:
    """The single GPU matching `alias`, or a hard error naming what is there.

    Raises rather than guessing on zero or multiple matches: silently picking
    one of two candidates is the class of bug this whole module exists to stop.
    """
    key = alias.strip().lower()
    if key not in GPU_ALIASES:
        raise KeyError(
            f"Unknown GPU alias {alias!r}. Known: {sorted(GPU_ALIASES)}.\n"
            f"  Aliases are defined in eo/train/gpu.py; add one there rather "
            f"than passing a raw index, which is what this module exists to avoid."
        )
    spec = GPU_ALIASES[key]
    if gpus is None:
        gpus = list_gpus()

    matches = [
        g for g in gpus
        if spec.name_contains.lower() in g.name.lower()
        and g.compute_capability == spec.compute_capability
    ]
    if len(matches) == 1:
        return matches[0]

    inventory = "\n".join(
        f"    nvidia-smi index {g.index}: {g.name} "
        f"(sm_{g.compute_capability[0]}{g.compute_capability[1]}, {g.memory_total_mib} MiB)"
        for g in gpus
    ) or "    (no GPUs reported by nvidia-smi)"
    raise RuntimeError(
        f"Alias {alias!r} matched {len(matches)} GPUs, expected exactly 1.\n"
        f"  Looking for a device whose name contains {spec.name_contains!r} at "
        f"compute capability sm_{spec.compute_capability[0]}{spec.compute_capability[1]}.\n"
        f"  Present:\n{inventory}\n"
        f"  If the hardware changed, update GPU_ALIASES in eo/train/gpu.py. Do not "
        f"fall back to an index -- the two index orderings disagree on this box."
    )


def assert_selected(alias: str) -> "GpuInfo":
    """Confirm, from inside the process, that CUDA handed us the right card.

    `resolve()` decides; this proves the decision survived `CUDA_VISIBLE_DEVICES`,
    the device ordering, and any ambient environment. Call it after CUDA is
    available and before anything expensive happens.

    Also asserts `device_count() == 1`. With both cards visible, HF `Trainer`
    wraps the model in `nn.DataParallel` across a sm_70 and a sm_52 card -- the
    slow one sets the pace -- and `preprocess_config` computes
    `gradient_accumulation_steps` from the wrong device count. Neither is an error.
    """
    import torch  # deliberately lazy: importing torch initializes CUDA

    spec = GPU_ALIASES[alias.strip().lower()]

    if not torch.cuda.is_available():
        raise RuntimeError("torch reports no CUDA device. Is CUDA_VISIBLE_DEVICES empty?")

    n = torch.cuda.device_count()
    if n != 1:
        raise RuntimeError(
            f"torch sees {n} CUDA devices; exactly 1 must be visible.\n"
            f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}\n"
            f"  With more than one visible, HF Trainer silently uses nn.DataParallel "
            f"across two unequal cards and gradient_accumulation_steps is computed "
            f"from the wrong device count. Both are silent."
        )

    props = torch.cuda.get_device_properties(0)
    got = (props.major, props.minor)
    if spec.name_contains.lower() not in props.name.lower() or got != spec.compute_capability:
        raise RuntimeError(
            f"Wrong GPU. Asked for {alias!r} "
            f"({spec.name_contains}, sm_{spec.compute_capability[0]}{spec.compute_capability[1]}); "
            f"cuda:0 is {props.name!r} (sm_{got[0]}{got[1]}).\n"
            f"  CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '<unset>')!r} "
            f"(must be {DEVICE_ORDER!r}), "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}"
        )

    return GpuInfo(
        index=int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1").split(",")[0]),
        name=props.name,
        compute_capability=got,
        memory_total_mib=props.total_memory // (1024 * 1024),
    )


def _main(argv=None) -> int:
    """`python -m eo.train.gpu titanv` -> the index to put in CUDA_VISIBLE_DEVICES.

    Exists so the shell launcher can resolve an alias without duplicating the
    matching rules in bash. Prints the index and nothing else, so it is safe to
    use in a command substitution; diagnostics go to stderr.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("alias", nargs="?", choices=sorted(GPU_ALIASES))
    ap.add_argument("--list", action="store_true", help="show every GPU and exit")
    args = ap.parse_args(argv)

    if args.list or not args.alias:
        for g in list_gpus():
            aliases = [a for a, s in GPU_ALIASES.items()
                       if s.name_contains.lower() in g.name.lower()
                       and s.compute_capability == g.compute_capability]
            cc = f"sm_{g.compute_capability[0]}{g.compute_capability[1]}"
            print(f"{g.index}  {g.name:<28} {cc:<7} {g.memory_total_mib:>6} MiB  "
                  f"{'/'.join(aliases) or '-'}", file=sys.stderr)
        return 0

    try:
        print(resolve(args.alias).index)
    except (KeyError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
