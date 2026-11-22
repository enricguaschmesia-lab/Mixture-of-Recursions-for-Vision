#!/usr/bin/env python
"""A2 Step 4.1: read one recipe-search run by the pre-registered rule (docs/plans/a2_segmentation.md, Step 4.1, items
2, 3 and 7; json and yaml only, so it runs in the RCP job right after training and on the workstation).

From the run directory (pretrain.py's trainer_state.json, with HF's whole log_history; plateau_stop.json; the composed
finetune_config.yaml; finetune_init.json):
  * the fold-4 map CE curve: `eval_loss` at step 0 and after every epoch;
  * the plateau stop: where it fired and stopped (eo/train/plateau_stop.py), checked against the curve: the rule
    recomputed here from the evals must fire at the same epoch (or nowhere);
  * THE CHECKPOINT (item 3): the save with the lowest eval_loss, the earlier on a tie (every save is an eval step;
    step 0 has no save). It must exist as a directory;
  * health (item 7): every train loss, grad norm and eval finite, and the lowest CE below step 0's; otherwise FAILED;
  * beside it: the training loss logged nearest the selected step (the overfitting gap), the max and median grad norm,
    the trainable parameters, the regime, the LR and the seed.
Writes <out> (select41.json) and <run>/selected_checkpoint.txt (one line, "checkpoint-N", for the job script).

    python eo/experiments/segmentation/step41_select.py --run <run dir> [--out <run dir>/select41.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def fires_at(evals: dict, E: int, patience: int, rel: float, last_step: int):
    """eo/train/plateau_stop.fires, recomputed from the logged evals: the first epoch it fires at, or None."""
    ep = {s // E: v for s, v in evals.items() if s > 0 and s % E == 0}
    for e in sorted(ep):
        if e < patience + 1 or e * E >= last_step:
            continue
        if any(k not in ep for k in range(1, e + 1)):
            break
        b = min(ep[k] for k in range(1, e + 1))
        b0 = min(ep[k] for k in range(1, e - patience + 1))
        if b >= (1.0 - rel) * b0:
            return e
    return None


def read_run(d: Path) -> dict:
    st = json.loads((d / "trainer_state.json").read_text())
    cfg = yaml.safe_load((d / "finetune_config.yaml").read_text())
    init = json.loads((d / "finetune_init.json").read_text())
    ps = json.loads((d / "plateau_stop.json").read_text()) if (d / "plateau_stop.json").exists() else None
    E = int(st["save_steps"])
    if int(st["eval_steps"]) != E:
        raise RuntimeError(f"{d}: eval_steps {st['eval_steps']} != save_steps {E}")
    evals = {int(h["step"]): float(h["eval_loss"]) for h in st["log_history"] if "eval_loss" in h}
    train = [h for h in st["log_history"] if "loss" in h and "eval_loss" not in h]
    saves = [s for s in sorted(evals) if s > 0 and s % E == 0]
    best = min(saves, key=lambda s: (evals[s], s))
    gn = [float(h.get("grad_norm", float("nan"))) for h in train]
    finite = (all(math.isfinite(float(h["loss"])) for h in train) and all(math.isfinite(g) for g in gn)
              and all(math.isfinite(v) for v in evals.values()))
    failed = (not finite) or (0 in evals and evals[best] >= evals[0])
    near = min(train, key=lambda h: abs(int(h["step"]) - best)) if train else None
    pc = cfg.get("plateau_stop") or {}
    sched_start = int(cfg["num_train_steps"]) - int(cfg["lr_scheduler_kwargs"]["num_decay_steps"])
    again = fires_at(evals, E, int(pc.get("patience", 3)), float(pc.get("rel", 1e-3)), sched_start) if pc.get("enable") else None
    fired = (ps or {}).get("fired")
    agree = (fired is None and again is None) or (fired is not None and fired.get("epoch") == again)
    ck = d / f"checkpoint-{best}"
    return {"dir": str(d), "regime": cfg["finetune"]["regime"],
            "lora": cfg["finetune"].get("lora") if cfg["finetune"]["regime"] == "lora" else None,
            "learning_rate": float(cfg["learning_rate"]), "seed": int(cfg.get("seed", 42)),
            "trainable_params": int(init["trainable_params"]), "steps_per_epoch": E,
            "max_steps": int(st["max_steps"]), "global_step": int(st["global_step"]),
            "eval_ce": {str(s): evals[s] for s in sorted(evals)},
            "plateau_stop": {"fired": fired, "recomputed_fire_epoch": again, "agrees": bool(agree),
                             "stopped_at_step": int(st["global_step"])},
            "selected": {"step": best, "checkpoint": ck.name, "eval_ce": evals[best], "epoch": best // E,
                         "exists": ck.is_dir(),
                         "in_decay": bool(best > (fired["step"] if fired else sched_start)),
                         "train_loss_near": float(near["loss"]) if near else None,
                         "train_loss_step": int(near["step"]) if near else None},
            "final": {"step": max(saves), "eval_ce": evals[max(saves)]},
            "health": {"all_finite": bool(finite), "failed": bool(failed),
                       "ce_step0": evals.get(0), "max_grad_norm": max(gn) if gn else None,
                       "median_grad_norm": sorted(gn)[len(gn) // 2] if gn else None,
                       "first_loss": float(train[0]["loss"]) if train else None,
                       "last_loss": float(train[-1]["loss"]) if train else None,
                       "last_lr": float(train[-1].get("learning_rate", float("nan"))) if train else None,
                       "n_train_logs": len(train)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.run)
    r = read_run(d)
    out = Path(a.out) if a.out else d / "select41.json"
    out.write_text(json.dumps(r, indent=1) + "\n")
    (d / "selected_checkpoint.txt").write_text(r["selected"]["checkpoint"] + "\n")
    p, s, h = r["plateau_stop"], r["selected"], r["health"]
    fz = p["fired"]
    print(f"{d.name}: {r['regime']} LR {r['learning_rate']:g} seed {r['seed']}: CE(0) {h['ce_step0']:.4f}; plateau "
          f"{'fired at epoch %d, stopped at step %d' % (fz['epoch'], p['stopped_at_step']) if fz else 'never fired, ran to step %d' % p['stopped_at_step']} "
          f"(recomputed: {p['recomputed_fire_epoch']}, agrees {p['agrees']}); selected {s['checkpoint']} (epoch {s['epoch']}) "
          f"CE {s['eval_ce']:.4f}, train loss there {s['train_loss_near']}; finite {h['all_finite']}, failed {h['failed']}, "
          f"max grad norm {h['max_grad_norm']:.3g}", flush=True)
    if not s["exists"]:
        raise SystemExit(f"the selected {s['checkpoint']} is not on disk")
    if not p["agrees"]:
        raise SystemExit("the plateau stop's record and the rule recomputed from the evals disagree")


if __name__ == "__main__":
    main()
