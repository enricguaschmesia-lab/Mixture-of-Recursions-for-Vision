"""Fine-tune Cosmos Tokenize1-DI16x16 on palette-shuffled renders (A2 Step 1 candidate (c); Enric, 2026-11-22).

Everything but the network is agent B's recipe for the frozen pick `b224_64000_long`, imported from step1_B_train:
the data (`Data`: the `fit` maps only, 12 PASTIS / 10 CloudSen12 224-px crops / 10 LULC per step, half the PASTIS
maps as 2 x 2 mosaics, a random dihedral transform, a random injective class -> colour assignment per map), the
loss (`native_nll`: the true colour's log-probability pooled to the native grid, the read's own quantity), the
EMA read on `monitor` every 2,500 steps (never `val`), the best-monitor and last exports, and the warmup-stable-
decay schedule whose decay starts when the monitor mean gains under `--plateau-gain` over the last quarter of the
run, or at `--max-stable`. The network is step1_D_tok.MapTokD, started from Cosmos's released weights
(`--init`, the extract of model.pt's `network`) with the nearest-colour palette head.

What differs from B's run, and why:
  - a fine-tune: peak LR 1e-4 (B: 5e-4 from scratch), warmup 1,000, weight decay 0.01 on conv weights (B: 0.05);
  - the palette head (128 numbers, started as a sharp nearest-colour rule) at 10x the LR without weight decay, so its
    scale can recalibrate (the smoke, 2026-11-22: training NLL 5.9 on PASTIS at step 30, the misclassified pixels
    at NLL 30-50);
  - the stable phase is bounded at 10k-40k steps (B's aligned run: 45k-100k at 11.5 steps/s); Cosmos is about 4x
    the compute per map, and a pretrained start should need fewer steps;
  - precision: bf16 autocast on the A100s (the project's rule), fp32 allowed for local smoke tests.

Resumable from `ckpt_last.pt` (model, EMA, optimizer, schedule, every RNG state, curves); `done.json` ends it.

    python -X faulthandler -u -m eo.experiments.segmentation.step1_D_train --run d224_cosmos16 \
        --init /path/cosmos_tokenize1_di16x16_network.pt --precision bf16 --data-gpu --expect-gpu A100
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from eo.experiments.segmentation import step1_D_tok as T
from eo.experiments.segmentation.step1_B_train import (CS_CROP, Data, ema_update, lr_at, make_probe, native_nll,
                                                       plateau, plot, save_atomic)
from eo.maps import harness as H
from eo.maps import palette as P
from eo.maps import sets

OUT_ROOT = Path("/data/enric/reports/a2_seg/step1/D")


def monitor_read(ema: torch.nn.Module, run: str, step: int, batch: int) -> dict:
    """mIoU on every dataset's `monitor` set through the harness (nearest and vote), on the EMA."""
    tok = T.DTokenizer(ema, name=f"D_{run}_s{step}", weights=None, sha=f"in_training:{run}:{step}", batch=batch)
    out = {}
    for d in sets.DATASETS:
        r = H.sweep(tok, d, "monitor", [{"rule": "nearest"}, {"rule": "vote"}], cache=False)
        out[d] = {k: {m: r[k][m] for m in ("mIoU", "boundary_acc", "interior_acc", "pixel_acc", "iou_per_class")}
                  for k in r}
    out["mean_nearest"] = float(np.mean([out[d]["nearest"]["mIoU"] for d in sets.DATASETS]))
    out["mean_vote"] = float(np.mean([out[d]["vote"]["mIoU"] for d in sets.DATASETS]))
    return out


@torch.no_grad()
def act_probe(model: torch.nn.Module, probe: torch.Tensor) -> dict:
    """Max |output| of every residual, attention and resampling block, and of the RGB and the logits, on the probe in
    fp32 (B's fp16 run died of a decoder conv passing 65,504; this tracks growth)."""
    acts, hooks = {}, []
    for name, m in model.named_modules():
        if isinstance(m, (T.ResnetBlock, T.AttnBlock, T.Upsample, T.Downsample)) or name.endswith(("conv_out", "conv_in")):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, k=name: acts.__setitem__(k, float(out.detach().abs().max()))))
    try:
        logits, _, _ = model(probe)
    finally:
        for h in hooks:
            h.remove()
    top = sorted(acts.items(), key=lambda kv: -kv[1])[:3]
    return {"max": top[0][1], "top": top, "max_logit": float(logits.abs().max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", required=True, help="Cosmos weights (step1_D_tok.extract_cosmos's file, or model.pt)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--head-scale", type=float, default=T.DEFAULT["head_scale"])
    ap.add_argument("--head", choices=("rgb", "logits"), default="rgb",
                    help="rgb: Cosmos's RGB output and a 1 x 1 palette head; logits: the last conv widened to the logits")
    ap.add_argument("--comp", default="12,10,10", help="maps per step: pastis,cloudsen12,lulc")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="the palette head's LR / the body's (no decay)")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--boundary-weight", type=float, default=0.0)
    ap.add_argument("--pastis-mosaic", type=float, default=0.5, help="share of PASTIS maps drawn as 2x2 mosaics")
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--min-stable", type=int, default=10000)
    ap.add_argument("--max-stable", type=int, default=40000)
    ap.add_argument("--plateau-gain", type=float, default=0.0015)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--min-decay", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=0, help="hard stop (smoke tests); 0 = none")
    ap.add_argument("--force-decay", action="store_true", help="start the decay at the next read")
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--data-gpu", action="store_true",
                    help="copy every fit pool to the GPU once and draw there (RCP: no random reads over NFS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    a = ap.parse_args()

    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    out = Path(a.out_root) / a.run
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print(f"{out}/done.json exists: nothing to do", flush=True)
        return
    if (out / "dead.json").exists():
        raise SystemExit(f"{out}/dead.json exists: the run died; not resuming it ({(out / 'dead.json').read_text()})")
    torch.backends.cudnn.benchmark = True

    ints = lambda s: [int(v) for v in s.split(",")]
    cfg = dict(T.DEFAULT, input_size=a.size, head_scale=a.head_scale)
    if a.head == "rgb":
        cfg.pop("head")                    # the first run's config has no "head" key: an "rgb" model, resumable as it was
    else:
        cfg["head"] = a.head
    comp = dict(zip(sets.DATASETS, ints(a.comp)))
    B = sum(comp.values())
    data_cfg = {"comp": comp, "seed": a.seed, "boundary_weight": a.boundary_weight, "cs_crop": CS_CROP,
                "dihedral": True, "assignment": "palette.shuffled_assignment per map",
                "pastis_mosaic": a.pastis_mosaic}
    if a.data_gpu:
        data_cfg["draw"] = "fit pools on the GPU, GPU generator"
    run_cfg = {"precision": a.precision}
    use_amp = a.precision == "bf16"

    torch.manual_seed(a.seed)
    model, init = T.from_cosmos(a.init, cfg)
    model = model.to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    is_head = lambda n: n.startswith(("head.", "decoder.conv_out.") if a.head == "logits" else ("head.",))
    decay_p = [p for n, p in model.named_parameters() if p.ndim >= 2 and not is_head(n)]
    other_p = [p for n, p in model.named_parameters() if p.ndim < 2 and not is_head(n)]
    head_p = [p for n, p in model.named_parameters() if is_head(n)]
    assert len(decay_p) + len(other_p) + len(head_p) == len(list(model.parameters()))
    # the head (128 numbers; with --head logits, the widened last conv) starts as a sharp nearest-colour rule, logits up
    # to ~400 on Cosmos's RGB: at the body's LR Adam would need ~1e5 steps to recalibrate it, so it trains at
    # head_lr_mult x the LR, without weight decay
    opt = torch.optim.AdamW([{"params": decay_p, "weight_decay": a.wd, "lr_mult": 1.0},
                             {"params": other_p, "weight_decay": 0.0, "lr_mult": 1.0},
                             {"params": head_p, "weight_decay": 0.0, "lr_mult": a.head_lr_mult}],
                            lr=a.lr, betas=(0.9, 0.99))
    data = Data(comp, a.seed, dev, a.pastis_mosaic, on_gpu=a.data_gpu)
    n_par = sum(p.numel() for p in model.parameters())

    step, sched = 0, {"decay_start": None, "decay_steps": None}
    curves = {"run": a.run, "config": cfg, "data": data_cfg, "args": vars(a), "params": n_par, "gpu": gpu,
              "init": init, "run_cfg": run_cfg, "train": [], "evals": [], "wall": []}
    best = {"mean_nearest": -1.0, "step": None}
    ck_path = out / "ckpt_last.pt"
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck["config"] != cfg or ck["data_cfg"] != data_cfg or ck["run_cfg"] != run_cfg:
            raise SystemExit(f"resume refused: config, data or precision differ from {ck_path}")
        if ck["init"]["init_sha256"] != init["init_sha256"]:
            raise SystemExit(f"resume refused: the init weights differ from {ck_path}'s")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, sched, curves, best = ck["step"], ck["sched"], ck["curves"], ck["best"]
        data.load_state(ck["data"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        curves["args_resumed"] = curves.get("args_resumed", []) + [{"step": step, "args": vars(a)}]
        print(f"resumed from {ck_path} at step {step} (sched {sched}, best {best})", flush=True)
    else:
        print(f"run {a.run}: {n_par:,} params from {init['init_weights']} (sha256 {init['init_sha256'][:12]}), codes "
              f"{model.fsq.n_codes}, grid {a.size // cfg['factor']}, batch {B} {comp}, fit {data.n_fit}, gpu {gpu}",
              flush=True)
    (out / "config.json").write_text(json.dumps({"config": cfg, "data": data_cfg, "args": vars(a), "params": n_par,
                                                 "init": init, "run_cfg": run_cfg}, indent=1))

    def checkpoint():
        save_atomic({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                     "step": step, "sched": sched, "curves": curves, "best": best, "data": data.state(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                     "config": cfg, "data_cfg": data_cfg, "run_cfg": run_cfg, "init": init}, ck_path)
        (out / "curves.json").write_text(json.dumps(curves))

    def evaluate():
        nonlocal best
        t0 = time.time()
        mon = monitor_read(ema, a.run, step, a.eval_batch)
        torch.cuda.empty_cache()
        hist = codes_hist.cpu().numpy()
        p = hist[hist > 0] / max(hist.sum(), 1)
        ev = {"step": step, "lr": lr_at(step, a, sched), "monitor": mon, "seconds": time.time() - t0,
              "train_codes": {"distinct": int((hist > 0).sum()), "entropy_bits": float(-(p * np.log2(p)).sum()),
                              "tokens": int(hist.sum())}}
        curves["wall_s"] = curves.get("wall_s", 0.0) + (time.time() - t0)
        curves["evals"].append(ev)
        line = " ".join(f"{d} {mon[d]['nearest']['mIoU']:.4f}/{mon[d]['vote']['mIoU']:.4f}" for d in sets.DATASETS)
        print(f"EVAL step {step} monitor nearest/vote: {line} mean {mon['mean_nearest']:.4f} | codes "
              f"{ev['train_codes']['distinct']} {ev['train_codes']['entropy_bits']:.2f} bits | {ev['seconds']:.0f} s",
              flush=True)
        if mon["mean_nearest"] > best["mean_nearest"]:
            best = {"mean_nearest": mon["mean_nearest"], "step": step,
                    "per_dataset": {d: mon[d]["nearest"]["mIoU"] for d in sets.DATASETS}}
            sha = T.export(out / "best.pt", ema, {"run": a.run, "step": step, "monitor": mon, "selected_on": "monitor",
                                                  "init": init})
            best["sha256"] = sha
            print(f"  best -> {out / 'best.pt'} (sha256 {sha[:12]})", flush=True)
        codes_hist.zero_()
        ev["act"] = act_probe(model, probe)
        print(f"  act max {ev['act']['max']:.0f} at {ev['act']['top'][0][0]}, max |logit| {ev['act']['max_logit']:.1f}",
              flush=True)
        plot(curves, out / "curves.png")

    codes_hist = torch.zeros(model.fsq.n_codes, dtype=torch.long, device=dev)
    probe = make_probe(a.size, dev)
    if step == 0 and not curves["evals"]:
        model.eval()
        evaluate()                                     # step 0: Cosmos as released, with the nearest-colour head
        model.train()
        checkpoint()
    win = {d: [] for d in sets.DATASETS}
    win_gn, t_last, t_ck = [], time.time(), time.time()
    n_skip = 0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    while True:
        if a.max_steps and step >= a.max_steps:
            break
        if sched["decay_start"] is not None and step >= sched["decay_start"] + sched["decay_steps"]:
            break
        lr = lr_at(step, a, sched)
        for g_ in opt.param_groups:
            g_["lr"] = lr * g_["lr_mult"]
        maps = data.draw()
        xs, order = [], [d for d in sets.DATASETS if d in maps]
        for d in order:
            task = P.TASKS[d]
            assign = P.shuffled_assignment(len(maps[d]), task.n_classes, task.ignore, generator=data.gc, device="cpu")
            xs.append(P.render(maps[d], assign, a.size))
        x = torch.cat(xs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, idx, cidx = model(x)
        losses, o = {}, 0
        for d in order:
            n = len(maps[d])
            losses[d] = native_nll(logits[o:o + n].float(), cidx[o:o + n], maps[d], P.TASKS[d].ignore, a.boundary_weight)
            o += n
        loss = sum(comp[d] / B * losses[d] for d in order)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        if not bool(torch.isfinite(gnorm)):
            opt.zero_grad(set_to_none=True)            # never step on a nan/inf gradient
            n_skip += 1
        else:
            opt.step()
        ema_update(ema, model, min(a.ema, (1 + step) / (10 + step)))
        codes_hist += torch.bincount(idx.flatten(), minlength=model.fsq.n_codes)
        step += 1
        for d in order:
            win[d].append(losses[d].detach())
        win_gn.append(gnorm.detach())
        if step % 100 == 0 or (a.max_steps and step == a.max_steps):
            now = time.time()
            nsteps = len(win_gn)
            curves["wall_s"] = curves.get("wall_s", 0.0) + (now - t_last)
            rec = {"step": step, "lr": lr, "loss": {d: float(torch.stack(win[d]).mean()) for d in order},
                   "grad_norm": float(torch.stack(win_gn).float().mean()), "skipped": n_skip,
                   "steps_per_s": nsteps / (now - t_last), "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            curves["train"].append(rec)
            if n_skip > 20:                            # never train on silently (CLAUDE.md)
                (out / "dead.json").write_text(json.dumps({"step": step, "record": rec}, indent=1))
                (out / "curves.json").write_text(json.dumps(curves))
                raise SystemExit(f"DEAD at step {step}: {rec} (ckpt_last.pt is the last healthy save)")
            n_skip = 0
            if step % 500 == 0 or step <= 1000 or a.max_steps:
                print(f"step {step} lr {lr:.2e} " + " ".join(f"{d} {v:.4f}" for d, v in rec["loss"].items())
                      + f" gn {rec['grad_norm']:.3f} {rec['steps_per_s']:.2f} it/s mem {rec['max_mem_gb']:.2f} GB"
                      f" skipped {rec['skipped']}", flush=True)
            win = {d: [] for d in sets.DATASETS}
            win_gn, t_last = [], now
        if step % a.eval_every == 0:
            model.eval()
            evaluate()
            model.train()
            if sched["decay_start"] is None and (a.force_decay or plateau(curves["evals"], step, a)
                                                 or step >= a.max_stable):
                sched = {"decay_start": step, "decay_steps": max(a.min_decay, int(a.decay_frac * step)),
                         "reason": "forced" if a.force_decay else ("plateau" if step < a.max_stable else "max_stable")}
                print(f"DECAY from step {step} over {sched['decay_steps']} steps ({sched['reason']})", flush=True)
            checkpoint()
            t_ck = time.time()
            t_last = time.time()
        elif time.time() - t_ck > a.ckpt_minutes * 60:
            checkpoint()
            t_ck = time.time()
    model.eval()
    if not curves["evals"] or curves["evals"][-1]["step"] != step:
        evaluate()
    checkpoint()
    last_sha = T.export(out / "last.pt", ema, {"run": a.run, "step": step, "selected_on": "last", "init": init})
    finished = sched["decay_start"] is not None and step >= sched["decay_start"] + sched["decay_steps"]
    done = {"run": a.run, "steps": step, "finished_schedule": finished, "best": best, "sched": sched,
            "last": str(out / "last.pt"), "last_sha256": last_sha, "best_path": str(out / "best.pt"),
            "params": n_par, "gpu": gpu, "init": init, "wall_hours": curves.get("wall_s", 0.0) / 3600}
    (out / ("done.json" if finished else "stopped.json")).write_text(json.dumps(done, indent=1))
    print("DONE" if finished else "STOPPED", json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
