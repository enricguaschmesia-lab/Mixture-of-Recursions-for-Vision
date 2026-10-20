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

REMOTE ALIASES (2026-10-11). `a100` is EPFL RCP's Run:AI A100, used only through
the RCP job wrapper. A pod sees exactly one card, so resolution and the
assertions below work unchanged. What differs is WHERE large outputs may live:
there is no /data on RCP, and the private project root is named by the alias's
`large_output_env` (`MOR_EO_RCP`). Preflight enforces that root on RCP, and
/data on the workstation, so the remote setting can never relax the local guard.

SEVERAL GPUS (2026-10-20). `resolve_n(alias, n)` picks n cards of one alias for a
multi-GPU (DDP) run, and `assert_selected(alias, n)` proves the process sees exactly
those. The n cards must be the SAME card (name, capability and memory): DDP runs at
the slowest rank's pace, and a model split across an A100-80GB and an A100-40GB is
the RCP version of the workstation's TITAN V + Titan X. Among identical cards the
lowest PCI indices are taken, since any of them is right. On this workstation an
alias matches one card, so n > 1 always refuses here.
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
    #: None on the workstation (large outputs under /data). On a remote site, the
    #: env var naming the private root that large outputs must live under.
    large_output_env: Optional[str] = None


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
    "a100": GpuSpec(
        alias="a100",
        name_contains="A100",
        compute_capability=(8, 0),
        note="Ampere, EPFL RCP (Run:AI). fp16/bf16 tensor cores. Remote: launched only through "
             "the RCP job wrapper; large outputs under $MOR_EO_RCP, not /data.",
        large_output_env="MOR_EO_RCP",
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

    Raises rather than guessing when nothing matches, or when the matches are
    not all the same card: silently picking one of two DIFFERENT candidates is
    the class of bug this whole module exists to stop. Among identical cards
    (an RCP node's A100s) the lowest index is as good as any other.
    """
    return resolve_n(alias, 1, gpus)[0]


def _inventory(gpus: List[GpuInfo]) -> str:
    return "\n".join(
        f"    nvidia-smi index {g.index}: {g.name} "
        f"(sm_{g.compute_capability[0]}{g.compute_capability[1]}, {g.memory_total_mib} MiB)"
        for g in gpus
    ) or "    (no GPUs reported by nvidia-smi)"


def resolve_n(alias: str, n: int, gpus: Optional[List[GpuInfo]] = None) -> List[GpuInfo]:
    """`n` GPUs matching `alias`, all the same card, lowest indices first.

    Refuses when fewer than n cards match, and when the matching cards differ
    in name or memory (an A100-80GB next to an A100-40GB both contain "A100"
    at sm_80): DDP across unequal cards runs at the slow one's pace, silently.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
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

    matches = sorted(
        (g for g in gpus
         if spec.name_contains.lower() in g.name.lower()
         and g.compute_capability == spec.compute_capability),
        key=lambda g: g.index,
    )
    kinds = {(g.name, g.memory_total_mib) for g in matches}
    if len(matches) >= n and len(kinds) == 1:
        return matches[:n]

    why = (f"matched {len(matches)} GPUs, {n} needed" if len(matches) < n else
           f"matched {len(matches)} GPUs that are not the same card: {sorted(kinds)}")
    raise RuntimeError(
        f"Alias {alias!r} {why}.\n"
        f"  Looking for a device whose name contains {spec.name_contains!r} at "
        f"compute capability sm_{spec.compute_capability[0]}{spec.compute_capability[1]}.\n"
        f"  Present:\n{_inventory(gpus)}\n"
        f"  If the hardware changed, update GPU_ALIASES in eo/train/gpu.py. Do not "
        f"fall back to an index -- the two index orderings disagree on this box."
    )


def assert_selected(alias: str, n: int = 1) -> "GpuInfo":
    """Confirm, from inside the process, that CUDA handed us the right card(s).

    `resolve()` decides; this proves the decision survived `CUDA_VISIBLE_DEVICES`,
    the device ordering, and any ambient environment. Call it after CUDA is
    available and before anything expensive happens.

    Also asserts `device_count() == n` (1 unless the run is launched for DDP).
    With more cards visible than the run asked for, HF `Trainer` wraps the model
    in `nn.DataParallel` -- on the workstation across a sm_70 and a sm_52 card,
    the slow one setting the pace -- and `preprocess_config` computes
    `gradient_accumulation_steps` from the wrong device count. Neither is an error.
    With n > 1, EVERY visible device must be the alias's card, and all the same.

    Returns cuda:0's info (the n cards are identical).
    """
    import torch  # deliberately lazy: importing torch initializes CUDA

    spec = GPU_ALIASES[alias.strip().lower()]

    if not torch.cuda.is_available():
        raise RuntimeError("torch reports no CUDA device. Is CUDA_VISIBLE_DEVICES empty?")

    seen = torch.cuda.device_count()
    if seen != n:
        raise RuntimeError(
            f"torch sees {seen} CUDA devices; exactly {n} must be visible.\n"
            f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}\n"
            f"  With more visible than the run is launched for, HF Trainer silently uses "
            f"nn.DataParallel and gradient_accumulation_steps is computed from the wrong "
            f"device count. Both are silent."
        )

    kinds = set()
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        got = (props.major, props.minor)
        if spec.name_contains.lower() not in props.name.lower() or got != spec.compute_capability:
            raise RuntimeError(
                f"Wrong GPU. Asked for {alias!r} "
                f"({spec.name_contains}, sm_{spec.compute_capability[0]}{spec.compute_capability[1]}); "
                f"cuda:{i} is {props.name!r} (sm_{got[0]}{got[1]}).\n"
                f"  CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '<unset>')!r} "
                f"(must be {DEVICE_ORDER!r}), "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}"
            )
        kinds.add((props.name, props.total_memory))
    if len(kinds) != 1:
        raise RuntimeError(f"The {n} visible devices are not the same card: {sorted(kinds)}. "
                           f"DDP would run at the slowest one's pace.")

    props = torch.cuda.get_device_properties(0)
    return GpuInfo(
        index=int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1").split(",")[0]),
        name=props.name,
        compute_capability=(props.major, props.minor),
        memory_total_mib=props.total_memory // (1024 * 1024),
    )


def _main(argv=None) -> int:
    """`python -m eo.train.gpu titanv` -> the index to put in CUDA_VISIBLE_DEVICES.

    Exists so the shell launcher can resolve an alias without duplicating the
    matching rules in bash. Prints the index and nothing else, so it is safe to
    use in a command substitution; diagnostics go to stderr. With `--count N`
    it prints N comma-separated indices (`0,1`), ready for CUDA_VISIBLE_DEVICES.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("alias", nargs="?", choices=sorted(GPU_ALIASES))
    ap.add_argument("--count", type=int, default=1, help="how many cards (DDP); default 1")
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
        print(",".join(str(g.index) for g in resolve_n(args.alias, args.count)))
    except (KeyError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
