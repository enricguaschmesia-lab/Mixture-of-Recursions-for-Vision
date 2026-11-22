#!/usr/bin/env python
"""A2 Step 4.0: read the three R0 fine-tunes' training logs by the pre-registered protocol (docs/plans/
a2_segmentation.md, Step 4.0, items 1-3 and "How the protocol is read"; any env, json only).

For each arm, from the trainer_state.json of its LAST save (HF's log_history holds every log and eval):
  * the fold-4 map CE curve: `eval_loss` at step 0 and after every epoch (teacher-forced, labels on the G codes and
    EO_MAP only);
  * THE EXTENSION TEST: with N = max_steps, E = steps an epoch (= save_steps), D = the decay (the last 20%, N / 5):
    fires if CE(N - D) < 0.99 x CE(N - D - 3E) on ANY arm. At 740 steps: CE(592) against CE(481);
  * THE CHECKPOINT: the save with the lowest eval_loss, the earlier on a tie. Every save is an eval step (both every
    epoch), so the selection runs over the saves' evals, never step 0's (there is no checkpoint-0);
  * health: every train log's loss and grad_norm finite, the largest grad_norm, the last LR (0 at the end of the
    linear decay), and whether two successive evals are identical (frozen weights).

    python eo/experiments/segmentation/step4_select.py --run aligned=<dir> --run short=<dir> --run long=<dir> \\
        --out /data/enric/reports/a2_seg/step4/select.json
  <dir> holds the run's trainer_state.json (pulled from its last checkpoint) and lists the run's saves.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DECAY_FRACTION = 0.2
EXT_EPOCHS = 3
EXT_REL = 0.01


def read_arm(d: Path) -> dict:
    st = json.loads((d / "trainer_state.json").read_text())
    N, E = int(st["max_steps"]), int(st["save_steps"])
    if int(st["eval_steps"]) != E:
        raise RuntimeError(f"{d}: eval_steps {st['eval_steps']} != save_steps {E}")
    D = int(round(DECAY_FRACTION * N))
    evals = {int(h["step"]): float(h["eval_loss"]) for h in st["log_history"] if "eval_loss" in h}
    train = [h for h in st["log_history"] if "loss" in h and "eval_loss" not in h]
    a, b = N - D, N - D - EXT_EPOCHS * E
    if a not in evals or b not in evals:
        raise RuntimeError(f"{d}: no eval at {a} or {b} (evals at {sorted(evals)[:5]}...)")
    saves = [s for s in sorted(evals) if s > 0 and s % E == 0]
    best = min(saves, key=lambda s: (evals[s], s))           # lowest CE, the earlier on a tie
    gn = [float(h.get("grad_norm", float("nan"))) for h in train]
    finite = all(math.isfinite(float(h["loss"])) for h in train) and all(math.isfinite(g) for g in gn) \
        and all(math.isfinite(v) for v in evals.values())
    ev = [evals[s] for s in sorted(evals)]
    frozen = any(ev[i] == ev[i + 1] for i in range(len(ev) - 1))
    return {"dir": str(d), "max_steps": N, "steps_per_epoch": E, "decay_steps": D, "global_step": int(st["global_step"]),
            "eval_ce": {str(s): evals[s] for s in sorted(evals)},
            "extension_test": {"ce_last_before_decay": evals[a], "step": a, "ce_three_epochs_earlier": evals[b],
                               "step_earlier": b, "relative_drop": 1 - evals[a] / evals[b],
                               "fires": bool(evals[a] < (1 - EXT_REL) * evals[b])},
            "selected": {"step": best, "checkpoint": f"checkpoint-{best}", "eval_ce": evals[best],
                         "epoch": best // E},
            "final": {"step": max(saves), "eval_ce": evals[max(saves)]},
            "health": {"all_finite": bool(finite), "max_grad_norm": max(gn) if gn else None,
                       "median_grad_norm": sorted(gn)[len(gn) // 2] if gn else None,
                       "last_lr": float(train[-1].get("learning_rate", float("nan"))) if train else None,
                       "first_loss": float(train[0]["loss"]) if train else None,
                       "last_loss": float(train[-1]["loss"]) if train else None,
                       "successive_equal_evals": bool(frozen), "n_train_logs": len(train)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="arm=<dir with trainer_state.json>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for r in a.run:
        arm, d = r.split("=", 1)
        res[arm] = read_arm(Path(d))
    fires = [arm for arm, x in res.items() if x["extension_test"]["fires"]]
    out = {"protocol": "A2 Step 4.0, items 1-3 (docs/plans/a2_segmentation.md)", "arms": res,
           "extension_fires_on": fires, "extend": bool(fires)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for arm, x in res.items():
        t, s, h = x["extension_test"], x["selected"], x["health"]
        print(f"{arm:8s} CE(0) {x['eval_ce']['0']:.4f}  CE({t['step_earlier']}) {t['ce_three_epochs_earlier']:.4f} -> "
              f"CE({t['step']}) {t['ce_last_before_decay']:.4f} (drop {100 * t['relative_drop']:.2f}%, extension "
              f"{'FIRES' if t['fires'] else 'no'})  selected {s['checkpoint']} CE {s['eval_ce']:.4f}  final "
              f"{x['final']['eval_ce']:.4f}  finite {h['all_finite']}, max grad norm {h['max_grad_norm']:.3g}")
    print(f"extension: {'ALL ARMS RERUN AT TWICE THE EPOCHS (fires on ' + ', '.join(fires) + ')' if fires else 'not needed'}")


if __name__ == "__main__":
    main()
