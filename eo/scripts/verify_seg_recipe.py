#!/usr/bin/env python
"""A2 Step 4.1's gate: the recipe search's schedule and overlays (`.venv`, CPU; 2026-11-22). Every check ships with a
control that must fail it. A separate file from verify_seg_model.py so that Step 4.1 adds no lines to the gate other
A2 work edits.

    python eo/scripts/verify_seg_recipe.py [--only Z14,Z15] [--out results.json]
    python eo/scripts/verify_seg_recipe.py --only Z14run --fire-dir <run> --nofire-dir <run>     # the RCP smoke's runs

  Z14 The plateau stop (eo/train/plateau_stop.py) in HF's loop order, on a real warmup_stable_decay LambdaLR
      (warmup 37, 1,850 steps, linear decay over the last 370) with synthetic eval CE every 37 steps:
      * Step 4.0's own aligned curve (epochs 1-31 as logged, then flat): fires at epoch 29 and not before; the LR
        falls linearly from the peak to 0 over 8 epochs (296 steps) and training stops at step 1,369 exactly;
                                     | a curve gaining 0.2 % an epoch never fires: the LR is the plain schedule's at
                                       every step, and nothing stops the run before 1,850
      * the threshold: 0.09 % gained over three epochs fires at epoch 4;
                                     | 0.11 % does not
      * a curve that only flattens at epoch 37 cannot fire (the first test it could fail falls on the scheduled
        decay's start, step 1,480), so the run is the plain schedule; a fire at epoch 39 stops at 49 x 37
      * the refusals: eval_steps != save_steps, a cosine schedule, a resume
  Z14run The RCP smoke through pretrain.py (the job path; a2s41_cell.sh): a run forced to fire (rel 1.0, an epoch of
      2 steps, warmup 2) fires at step 8; its steps 9-12 use 1, 3/4, 1/2, 1/4 of the peak (HF logs the LR a step used);
      it stops at step 12 with an eval and a save there; plateau_stop.json and select41.json record it
                                     | its control (rel 0: fires only on no gain at all) runs to its stop_steps on
                                       the plain schedule's LR
  Z15 Overlays: r0_s41 differs from r0_pastis1_aligned in exactly the schedule keys and the plateau_stop block (50
      epochs x 37 steps, the last 20 % decaying, eval = save = one epoch); r0_s41_lora16 equals r0_s41;
      r0_s41_lora64 differs in exactly finetune.lora.{r, alpha} = (64, 128); r0_s41_full in exactly finetune.regime;
      each regime built from the map base trains what the plan says (51,014,208 / 65,179,200 / 195,782,400) and
      LoRA leaves every base row frozen
                                     | a stray key (weight_decay) is flagged; LoRA r64 at alpha 32 is flagged
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS: list = []
MAP_BASE = Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000")
STEP40_SELECT = Path("/data/enric/reports/a2_seg/step4/select.json")
E, N, WARM, DECAY = 37, 1850, 37, 370
PEAK = 1e-3


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def _compose(name: str, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


# ---------------------------------------------------------------- Z14 (unit)
def simulate(ce_of_epoch, *, rel=1e-3, patience=3, eval_steps=E, save_steps=E, sched="warmup_stable_decay",
             start_step=0):
    """HF's loop order on a real WSD LambdaLR: optimizer.step, scheduler.step, global_step += 1, on_step_end, then
    (at an eval step) on_evaluate. Returns (lr after each step, the step training stopped at or None, callback)."""
    import torch
    from transformers import TrainerControl, get_scheduler
    from eo.train.plateau_stop import PlateauStopCallback
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=PEAK)
    sch = get_scheduler("warmup_stable_decay", opt, num_warmup_steps=WARM, num_training_steps=N,
                        scheduler_specific_kwargs={"num_decay_steps": DECAY, "decay_type": "linear"})
    tmp = tempfile.mkdtemp(prefix="z14_")
    args = SimpleNamespace(eval_strategy="steps", eval_steps=eval_steps, save_steps=save_steps,
                           lr_scheduler_type=sched, lr_scheduler_kwargs={"num_decay_steps": DECAY}, output_dir=tmp)
    state = SimpleNamespace(global_step=start_step, max_steps=N, is_world_process_zero=True)
    ctl = TrainerControl()
    cb = PlateauStopCallback(None, patience=patience, rel=rel, verbose=False)
    cb.on_train_begin(args, state, ctl)
    lrs, stopped = {}, None
    for step in range(1, N + 1):
        opt.step()
        sch.step()
        state.global_step = step
        cb.on_step_end(args, state, ctl, lr_scheduler=sch)
        if step % eval_steps == 0:
            cb.on_evaluate(args, state, ctl, metrics={"eval_loss": float(ce_of_epoch(step // eval_steps))},
                           lr_scheduler=sch)
        lrs[step] = sch.get_last_lr()[0]
        if ctl.should_training_stop:
            stopped = step
            break
    return lrs, stopped, cb, tmp


def plain_lr(step: int) -> float:
    """The WSD schedule's LR after `step` scheduler steps (warmup 37, stable, linear decay over the last 370)."""
    if step < WARM:
        return PEAK * step / WARM
    if step < N - DECAY:
        return PEAK
    return PEAK * max(0.0, 1.0 - (step - (N - DECAY)) / DECAY)


def z14() -> None:
    from eo.train.plateau_stop import fires
    sel = json.loads(STEP40_SELECT.read_text())["arms"]["aligned"]["eval_ce"]
    real = {int(k) // E: float(v) for k, v in sel.items() if 0 < int(k) <= 31 * E}
    flat = lambda e: real[e] if e <= 31 else real[31]                     # noqa: E731
    first = [e for e in range(4, 32) if fires(real, e, 3, 1e-3)]
    lrs, stopped, cb, tmp = simulate(flat)
    f = cb.fired or {}
    t0, d = f.get("step"), f.get("decay_steps")
    want_lr = {s: (plain_lr(s) if s < (t0 or 0) else PEAK * max(0.0, ((t0 or 0) + (d or 1) - s) / (d or 1)))
               for s in lrs}
    lr_err = max(abs(lrs[s] - want_lr[s]) for s in lrs)
    rec = json.loads((Path(tmp) / "plateau_stop.json").read_text())
    ok = (first[:1] == [29] and f.get("epoch") == 29 and d == 8 * E and stopped == 29 * E + 8 * E == 1369
          and lr_err < 1e-12 and rec["fired"]["stop_step"] == 1369 and abs(lrs[1369 - 1] - PEAK / d) < 1e-12)
    record("Z14 Step 4.0's aligned curve", ok,
           f"the rule first fires at epoch {first[:1]} (CE {real[29]:.6f}; lowest {f.get('best_ce'):.6f} against "
           f"{f.get('best_ce_patience_earlier'):.6f}); decay {d} steps; stopped at step {stopped}; max |LR - the "
           f"intended LR| {lr_err:.1e} over {len(lrs)} steps; last decay step's LR {lrs[stopped - 1]:.3e} = peak/{d}")

    lrs, stopped, cb, _ = simulate(lambda e: 10.0 * 0.998 ** e)
    err = max(abs(lrs[s] - plain_lr(s)) for s in lrs)
    record("Z14 control: 0.2 % an epoch never fires", cb.fired is None and stopped is None and len(lrs) == N
           and err < 1e-12, f"fired {cb.fired}; stopped {stopped}; ran {len(lrs)} steps; max |LR - plain| {err:.1e}")

    g_fire, g_keep = 1 - (1 - 0.0009) ** (1 / 3), 1 - (1 - 0.0011) ** (1 / 3)
    _, s1, c1, _ = simulate(lambda e: 8.0 * (1 - g_fire) ** e)
    record("Z14 threshold: 0.09 % in three epochs fires", c1.fired is not None and c1.fired["epoch"] == 4
           and s1 == 4 * E + 2 * E, f"fired at epoch {c1.fired and c1.fired['epoch']}, stopped at {s1} (4 x 37 + 2 x 37)")
    _, s2, c2, _ = simulate(lambda e: 8.0 * (1 - g_keep) ** e)
    record("Z14 control: 0.11 % in three epochs does not", c2.fired is None and s2 is None,
           f"fired {c2.fired}; stopped {s2}")

    late = lambda e: 10.0 * 0.995 ** min(e, 37)                           # noqa: E731
    lrs, stopped, cb, _ = simulate(late)
    err = max(abs(lrs[s] - plain_lr(s)) for s in lrs)
    l39 = lambda e: 10.0 * 0.995 ** min(e, 36)                            # noqa: E731
    _, s39, c39, _ = simulate(l39)
    record("Z14 no fire on the scheduled decay; a fire at 39 stops at 49", cb.fired is None and stopped is None
           and err < 1e-12 and c39.fired is not None and c39.fired["epoch"] == 39 and s39 == 49 * E,
           f"flat from 37: fired {cb.fired}, max |LR - plain| {err:.1e}; flat from 36: fired at epoch "
           f"{c39.fired and c39.fired['epoch']}, decay {c39.fired and c39.fired['decay_epochs']} epochs, stopped at {s39}")

    refused = []
    for kw in ({"save_steps": 2 * E}, {"sched": "cosine"}, {"start_step": 37}):
        try:
            simulate(flat, **kw)
        except ValueError as e:
            refused.append(str(e).split(",")[0][:60])
    record("Z14 refusals", len(refused) == 3, f"refused {len(refused)}/3: {refused}")


# ---------------------------------------------------------------- Z14run (the RCP smoke)
def _state(d: Path):
    st = json.loads((d / "trainer_state.json").read_text()) if (d / "trainer_state.json").exists() else None
    if st is None:
        cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        st = json.loads((cks[-1] / "trainer_state.json").read_text())
    return st


def _plain_factor(cfg: dict):
    """The run's own WSD factor after `s` scheduler steps, from its finetune_config.yaml."""
    w, n = int(cfg["num_warmup_steps"]), int(cfg["num_train_steps"])
    nd = int(cfg["lr_scheduler_kwargs"]["num_decay_steps"])
    return lambda s: (s / w if s < w else 1.0 if s < n - nd else max(0.0, 1.0 - (s - (n - nd)) / nd))


def z14run(fire_dir: Path, nofire_dir: Path) -> None:
    """HF logs, at global step s, the LR that step USED: the scheduler's factor after s - 1 steps (transformers 4.52.4
    reads it before lr_scheduler.step()). So after a fire at t0 the logged LR is factor(t0) x (d - k + 1) / d at step
    t0 + k: the peak on the first step after the fire, 1/d on the last. Checkpoint existence is read from the run's
    select41.json, written on RCP where the checkpoints are."""
    import yaml
    st = _state(fire_dir)
    cfg = yaml.safe_load((fire_dir / "finetune_config.yaml").read_text())
    base, pf = float(cfg["learning_rate"]), _plain_factor(cfg)
    ps = json.loads((fire_dir / "plateau_stop.json").read_text())
    sel = json.loads((fire_dir / "select41.json").read_text())
    lr = {int(h["step"]): float(h["learning_rate"]) for h in st["log_history"] if "learning_rate" in h}
    ev = sorted(int(h["step"]) for h in st["log_history"] if "eval_loss" in h)
    fz = ps.get("fired") or {}
    t0, d, stop = fz.get("step"), fz.get("decay_steps"), fz.get("stop_step")

    def used(s):                      # the factor step s used = the intended lambda at s - 1
        u = s - 1
        if t0 is None or u < t0:
            return pf(u)
        return pf(t0) * max(0.0, (t0 + d - u) / d)
    err = max(abs(lr[s] - base * used(s)) / base for s in lr)
    ok = (t0 == 8 and d == 4 and stop == 12 and int(st["global_step"]) == 12 and sel["selected"]["checkpoint"] ==
          "checkpoint-12" and sel["selected"]["exists"] and 12 in ev and err < 1e-6 and ps.get("final_step") == 12
          and sel["plateau_stop"]["agrees"])
    record("Z14run forced fire through pretrain.py", ok,
           f"fired at step {t0}, decay {d}, stop {stop}; global_step {st['global_step']}; selected "
           f"{sel['selected']['checkpoint']} (exists on RCP {sel['selected']['exists']}); evals at {ev}; LR used by steps "
           f"{t0}..{stop}: {[round(lr.get(s, float('nan')) / base, 4) for s in range(t0 or 0, (stop or 0) + 1)]} x {base:g} "
           f"(max relative error against the intended schedule {err:.1e}, every logged step); the select script's "
           f"recomputed fire agrees {sel['plateau_stop']['agrees']}")
    st2 = _state(nofire_dir)
    cfg2 = yaml.safe_load((nofire_dir / "finetune_config.yaml").read_text())
    base2, pf2 = float(cfg2["learning_rate"]), _plain_factor(cfg2)
    ps2 = json.loads((nofire_dir / "plateau_stop.json").read_text())
    lr2 = {int(h["step"]): float(h["learning_rate"]) for h in st2["log_history"] if "learning_rate" in h}
    err2 = max(abs(lr2[s] - base2 * pf2(s - 1)) / base2 for s in lr2)
    record("Z14run control: rel 0 never fires", ps2.get("fired") is None and int(st2["global_step"])
           == int(cfg2["stop_steps"]) and err2 < 1e-6,
           f"fired {ps2.get('fired')}; ran to step {st2['global_step']} (stop_steps {cfg2['stop_steps']}); every logged "
           f"LR within {err2:.1e} of the plain schedule's; history {ps2.get('history')}")


# ---------------------------------------------------------------- Z15
def z15() -> None:
    from omegaconf import OmegaConf
    base = _flat(OmegaConf.to_container(_compose("eo_seg/r0_pastis1_aligned"), resolve=False))
    s41c = _compose("eo_seg/r0_s41")
    s41 = _flat(OmegaConf.to_container(s41c, resolve=False))
    diff = {k for k in set(base) | set(s41) if base.get(k) != s41.get(k)}
    want = {"num_train_steps", "stop_steps", "lr_scheduler_kwargs.num_decay_steps", "plateau_stop.enable",
            "plateau_stop.patience", "plateau_stop.rel", "plateau_stop.decay_frac", "plateau_stop.min_decay"}
    spe = 1200 // int(s41c.total_batch_size)
    ok = (diff == want and s41c.num_train_steps == 50 * spe == s41c.stop_steps and spe == 37
          and s41c.lr_scheduler_kwargs.num_decay_steps == s41c.num_train_steps // 5 and s41c.eval_steps == spe
          and s41c.save_steps == spe and s41c.num_warmup_steps == spe and s41c.eval_on_start
          and s41c.lr_scheduler_kwargs.decay_type == "linear" and s41c.seg.in_pinned is True)
    record("Z15 r0_s41", ok, f"differs from r0_pastis1_aligned in {sorted(diff)}; {s41c.num_train_steps} steps = 50 x "
           f"{spe}; decay {s41c.lr_scheduler_kwargs.num_decay_steps} ({s41c.lr_scheduler_kwargs.decay_type}); eval "
           f"{s41c.eval_steps}, save {s41c.save_steps}, warmup {s41c.num_warmup_steps}; in_pinned {s41c.seg.in_pinned}")
    intended = {"lora16": set(), "lora64": {"finetune.lora.r", "finetune.lora.alpha"}, "full": {"finetune.regime"}}
    for reg, keys in intended.items():
        c = _compose(f"eo_seg/r0_s41_{reg}")
        f = _flat(OmegaConf.to_container(c, resolve=False))
        d = {k for k in set(s41) | set(f) if s41.get(k) != f.get(k)}
        ok = d == keys and (reg != "lora64" or (c.finetune.lora.r == 64 and c.finetune.lora.alpha == 128)) \
            and (reg != "full" or c.finetune.regime == "full")
        record(f"Z15 r0_s41_{reg}", ok, f"differs from r0_s41 in {sorted(d)}; regime {c.finetune.regime}, r "
               f"{c.finetune.lora.r}, alpha {c.finetune.lora.alpha}")
    f = _flat(OmegaConf.to_container(_compose("eo_seg/r0_s41_lora64", ["weight_decay=0.1"]), resolve=False))
    d = {k for k in set(s41) | set(f) if s41.get(k) != f.get(k)}
    c = _compose("eo_seg/r0_s41_lora64", ["finetune.lora.alpha=32"])
    record("Z15 controls: a stray key; r64 at alpha 32", d != intended["lora64"] and not (
        c.finetune.lora.r == 64 and c.finetune.lora.alpha == 128), f"stray: differs in {sorted(d)}; alpha 32 flagged")

    import torch
    from eo.finetune.init import init_from
    from eo.generate.conditional import build_model
    want_n = {"lora16": 51_014_208, "lora64": 65_179_200, "full": 195_782_400}
    for reg, n in want_n.items():
        model, cfg = build_model(f"eo_seg/r0_s41_{reg}", None, device="cpu",
                                 overrides=[f"finetune.init_from={MAP_BASE}"])
        rep = init_from(model, cfg)
        frozen_rows = True
        if reg != "full":
            emb = model.get_input_embeddings()
            frozen_rows = (not emb.old.requires_grad) and emb.new.requires_grad and emb.old.shape[0] == 81_296
        record(f"Z15 {reg} trains what the plan says", rep["trainable_params"] == n and frozen_rows,
               f"trainable {rep['trainable_params']:,} (plan {n:,}) of {rep['total_params']:,}; base rows frozen "
               f"{frozen_rows if reg != 'full' else 'n/a (full)'}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fire-dir", default=None)
    ap.add_argument("--nofire-dir", default=None)
    a = ap.parse_args()
    checks = {"Z14": z14, "Z15": z15}
    if a.fire_dir and a.nofire_dir:
        checks["Z14run"] = lambda: z14run(Path(a.fire_dir), Path(a.nofire_dir))
    t0 = time.time()
    for k, fn in checks.items():
        if a.only and k not in a.only.split(","):
            continue
        try:
            fn()
        except Exception as e:
            traceback.print_exc()
            record(k, False, f"raised {type(e).__name__}: {e}")
    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {n_fail} fail ({time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).write_text(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in RESULTS], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
