# eo/train/plateau_stop.py
"""Up to `num_train_steps`, stopped on the held-out CE with a decay at the plateau (A2 Step 4.1, 2026-11-22; `.venv`).

A2 Step 4.1's schedule (docs/plans/a2_segmentation.md, Step 4.1, item 2): a fine-tune runs "up to 50 epochs with early
stopping on fold-4 map CE". Under a warmup-stable-decay schedule, plain early stopping would end a run at the peak LR
and forfeit the decay, and a fixed horizon reads a fast LR at a peak-LR save long after it stopped improving (Step
4.0's aligned arm plateaued at epoch 26 of 40). So the plateau STARTS the decay:

  * an "epoch" is one eval interval (eval_steps, which must equal save_steps: every eval has a save);
  * after each epoch e >= patience + 1, with B(e) the lowest eval CE over epochs 1..e (step 0's eval never counts):
    the rule FIRES if B(e) >= (1 - rel) * B(e - patience): less than `rel` gained in `patience` epochs;
  * when it fires at step t0 = e * E, the LR goes linearly from the peak to 0 over D = E * max(min_decay,
    ceil(e * decay_frac)) steps, and training stops at t0 + D (a save and an eval fall there, since D is whole
    epochs). With decay_frac 0.25 the decay is the run's last 20%, as Step 4.0's schedule;
  * it fires only before the schedule's own decay starts (num_train_steps - num_decay_steps). If it never fires, the
    run is the plain schedule: the scheduled decay, then the end.

The decay replaces the scheduler's lambda in place (torch's LambdaLR reads `lr_lambdas` at every step), so the
optimizer, the scheduler object, the saves and the logged learning rate are HF's own. On the first step after the
fire the LR is still the peak (the scheduler stepped already), then (D - 1)/D, ..., 1/D on the last: the same shape as
the scheduled linear decay. A record goes to `<output_dir>/plateau_stop.json` (at the fire and at the end).

Config (pretrain.py adds the callback when `plateau_stop.enable`; an eval set is required):
  plateau_stop:
    enable: true
    patience: 3          # epochs
    rel: 0.001           # 0.1 % (1.0 fires at the first test, 0 only on no gain at all: the gate's smoke)
    decay_frac: 0.25
    min_decay: 2         # epochs
A restarted fine-tune starts again from step 0 (A2's job scripts never resume a fine-tune), so nothing here is
saved for a resume; a resume through this callback is refused.
Gate: eo/scripts/verify_seg_recipe.py (Z14), each check with a control.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

from transformers import TrainerCallback


def best_upto(history: Dict[int, float], e: int) -> float:
    """The lowest CE over epochs 1..e (epoch 0, the eval before training, never counts)."""
    vals = [v for k, v in history.items() if 1 <= k <= e]
    if not vals:
        raise ValueError(f"no eval in epochs 1..{e}")
    return min(vals)


def fires(history: Dict[int, float], e: int, patience: int, rel: float) -> bool:
    """The plateau rule at epoch e: B(e) >= (1 - rel) * B(e - patience), with B over epochs >= 1."""
    if e < patience + 1 or any(k not in history for k in range(1, e + 1)):
        return False
    return best_upto(history, e) >= (1.0 - rel) * best_upto(history, e - patience)


def decay_epochs(e: int, decay_frac: float, min_decay: int) -> int:
    return max(int(min_decay), int(math.ceil(e * float(decay_frac) - 1e-9)))


def linear_to_zero(t0: int, d: int, old):
    """The replacement lambda: the old one before t0, then the old factor at t0 (1 after the warmup) falling linearly
    to 0 at t0 + d."""
    f0 = float(old(t0))

    def lam(step: int) -> float:
        if step < t0:
            return old(step)
        return f0 * max(0.0, float(t0 + d - step) / float(d))
    return lam


class PlateauStopCallback(TrainerCallback):
    """See the module docstring. `cfg` is the run's config (reads `plateau_stop`)."""

    def __init__(self, cfg=None, *, patience: int = 3, rel: float = 1e-3, decay_frac: float = 0.25,
                 min_decay: int = 2, verbose: bool = True):
        super().__init__()
        ps = (cfg.get("plateau_stop") or {}) if cfg is not None else {}
        self.patience = int(ps.get("patience", patience))
        self.rel = float(ps.get("rel", rel))
        self.decay_frac = float(ps.get("decay_frac", decay_frac))
        self.min_decay = int(ps.get("min_decay", min_decay))
        if self.patience < 1 or not (0.0 <= self.rel <= 1.0) or self.decay_frac <= 0 or self.min_decay < 1:
            raise ValueError(f"plateau_stop: patience {self.patience}, rel {self.rel}, decay_frac {self.decay_frac}, "
                             f"min_decay {self.min_decay}")
        self.verbose = verbose
        self.history: Dict[int, float] = {}
        self.fired: Optional[Dict] = None
        self.E = self.max_steps = self.sched_decay_start = None
        self.output_dir: Optional[str] = None

    # ------------------------------------------------------------------ checks
    def on_train_begin(self, args, state, control, **kwargs):
        if str(getattr(args, "eval_strategy", "no")).split(".")[-1].lower() != "steps":
            raise ValueError("plateau_stop needs eval_strategy 'steps' (an eval set)")
        if int(args.eval_steps) != int(args.save_steps):
            raise ValueError(f"plateau_stop needs eval_steps == save_steps (every eval has a save), got "
                             f"{args.eval_steps} and {args.save_steps}")
        sched = str(getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type))
        if sched != "warmup_stable_decay":
            raise ValueError(f"plateau_stop replaces a warmup_stable_decay schedule's decay, not {sched!r}")
        if int(state.global_step) != 0:
            raise ValueError("plateau_stop does not resume: a fine-tune restarts from step 0")
        self.E = int(args.eval_steps)
        self.max_steps = int(state.max_steps)
        nd = int((args.lr_scheduler_kwargs or {}).get("num_decay_steps", 0))
        self.sched_decay_start = self.max_steps - nd
        self.output_dir = args.output_dir
        self._write()

    # ------------------------------------------------------------------ the rule
    def on_evaluate(self, args, state, control, metrics=None, lr_scheduler=None, **kwargs):
        step = int(state.global_step)
        if not metrics or "eval_loss" not in metrics or step == 0 or step % self.E:
            return control
        e = step // self.E
        self.history[e] = float(metrics["eval_loss"])
        if self.fired is None and step < self.sched_decay_start and fires(self.history, e, self.patience, self.rel):
            d = self.E * decay_epochs(e, self.decay_frac, self.min_decay)
            stop = min(step + d, self.max_steps)
            d = stop - step
            sch = getattr(lr_scheduler, "scheduler", lr_scheduler)      # accelerate's wrapper holds the LambdaLR
            if sch is None or not hasattr(sch, "lr_lambdas"):
                raise RuntimeError(f"plateau_stop: the scheduler {type(lr_scheduler).__name__} has no lr_lambdas")
            sch.lr_lambdas = [linear_to_zero(step, d, old) for old in sch.lr_lambdas]
            b, b0 = best_upto(self.history, e), best_upto(self.history, e - self.patience)
            self.fired = {"epoch": e, "step": step, "decay_steps": d, "decay_epochs": d / self.E, "stop_step": stop,
                          "best_ce": b, "best_ce_patience_earlier": b0, "relative_gain": 1.0 - b / b0}
            if self.verbose and getattr(state, "is_world_process_zero", True):
                print(f"PlateauStop: epoch {e} (step {step}): lowest CE {b:.6f} against {b0:.6f} {self.patience} "
                      f"epochs earlier (gain {100 * (1 - b / b0):.4f}% < {100 * self.rel:.3g}%) -> linear decay to 0 "
                      f"over {d} steps, stop at step {stop}", flush=True)
            self._write()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.fired is not None and int(state.global_step) >= self.fired["stop_step"]:
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._write(final_step=int(state.global_step))

    def _write(self, final_step: Optional[int] = None) -> None:
        if not self.output_dir or int(os.environ.get("RANK", "0")) != 0:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        doc = {"rule": "after epoch e >= patience+1: fire if min CE over 1..e >= (1 - rel) x min over 1..e-patience; "
                       "then linear decay to 0 over E x max(min_decay, ceil(e x decay_frac)) steps and stop "
                       "(eo/train/plateau_stop.py)",
               "patience": self.patience, "rel": self.rel, "decay_frac": self.decay_frac, "min_decay": self.min_decay,
               "steps_per_epoch": self.E, "max_steps": self.max_steps,
               "scheduled_decay_start": self.sched_decay_start, "fired": self.fired,
               "history": {str(k): v for k, v in sorted(self.history.items())}, "final_step": final_step}
        with open(os.path.join(self.output_dir, "plateau_stop.json"), "w") as fh:
            json.dump(doc, fh, indent=1)
