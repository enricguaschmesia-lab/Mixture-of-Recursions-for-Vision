# eo/train/fixed_eval.py
"""Held-out evals at a fixed list of steps, besides `eval_steps` (2026-11-14; `.venv`).

HF's Trainer evaluates every `eval_steps`, one interval for the whole run. A learning-curve comparison (the
random- against the LLM-initialised B10c-tr-x10-NC) wants its resolution where the curves move fastest, in the
first few hundred steps, without paying for that density over 33,000 steps. `fixed_eval_steps: "33,66,..."` in a
run's config adds evals at exactly those global steps; the upstream `fixed_save_steps` does the same for saves
(util/callback.ScalingLawsSaveCallback).

Runs after HF's DefaultFlowCallback in the callback list, so it only ever turns an eval ON. The list comes from
the config at every start, so a resume keeps it (HF restores eval_steps from trainer_state.json instead: CLAUDE.md,
Resume).
"""
from __future__ import annotations

from typing import Iterable, List, Union

from transformers import TrainerCallback


def parse_steps(spec: Union[str, Iterable[int], None]) -> List[int]:
    """'33,66,99' or [33, 66, 99] -> sorted unique positive ints. Refuses anything else."""
    if spec is None or spec == "":
        return []
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    steps = sorted({int(str(s).strip()) for s in items})
    if any(s <= 0 for s in steps):
        raise ValueError(f"fixed_eval_steps must be positive steps, got {spec!r}")
    return steps


class FixedEvalCallback(TrainerCallback):
    """Sets control.should_evaluate at each listed global step."""

    def __init__(self, steps: Union[str, Iterable[int]]):
        super().__init__()
        self.steps = set(parse_steps(steps))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            control.should_evaluate = True
        return control
