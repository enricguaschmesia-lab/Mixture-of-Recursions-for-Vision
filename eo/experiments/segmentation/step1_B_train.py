"""Train agent B's f16 map tokenizer (A2 Step 1) on palette-shuffled renders of the `fit` maps.

Data: `eo.maps.sets.training_maps()`, the `fit` part of each pool only. Each step draws a fixed number of
maps per dataset (PASTIS 128 px maps; CloudSen12 random 224-px crops of 512-px tiles; LULC 224-px maps),
applies a random dihedral transform, gives every map its own random injective class -> colour assignment
(`palette.shuffled_assignment`: colours 1..31, void / no data on white) and renders it with
`palette.render` (nearest upsampling to the input size), exactly as the harness renders a read.

Loss: the read's own quantity. The log-probability of each input pixel's true colour (log-softmax over
the 32 universal colours) is pooled to the native grid by `eo.maps.decode.to_native` (each native pixel
averages exactly the input-size pixels the render copied it to) and averaged over the scored native
pixels (void and no data dropped); optional extra weight on native boundary pixels (`--boundary-weight`).
Per dataset the loss is a mean over its pixels; datasets are weighted by their share of the batch.

Schedule: warmup, a constant learning rate, then a cosine decay to zero (warmup-stable-decay). The decay
starts when the monitor's mean nearest-rule mIoU over the three datasets gains under `--plateau-gain`
(0.003) over the last quarter of the run (each end smoothed over two reads), or at `--max-stable` steps.
The monitor (`harness.sweep(tok, d, "monitor", ...)`, never `val`) is read every `--eval-every` steps
on an EMA of the weights. The best monitor checkpoint and the last one are kept.

Resumable: `ckpt_last.pt` (model, EMA, optimizer, scaler, schedule, every RNG state, curves) is written
every `--ckpt-minutes` and at every read; a relaunch with the same out dir resumes from it. `done.json`
marks a finished run (a relaunch then exits at once).

    python -X faulthandler -u -m eo.experiments.segmentation.step1_B_train --run b224_4375 --size 224 \
        --levels 7,5,5,5,5 --expect-gpu "TITAN V"
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eo.experiments.segmentation import step1_B_tok as T
from eo.maps import decode as D
from eo.maps import harness as H
from eo.maps import palette as P
from eo.maps import scoring as S
from eo.maps import sets

OUT_ROOT = Path("/data/enric/reports/a2_seg/step1/B")
CS_CROP = 224


def dihedral(m: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """(B, H, W) maps, k (B,) in 0..7: rotation by 90 * (k % 4), then a flip when k >= 4."""
    out = torch.empty_like(m)
    for t in range(8):
        sel = k == t
        if bool(sel.any()):
            x = torch.rot90(m[sel], t % 4, (1, 2))
            out[sel] = torch.flip(x, (2,)) if t >= 4 else x
    return out


class Data:
    """Random training maps from the `fit` parts only: PASTIS on the GPU, CloudSen12 and LULC drawn from
    the memory-mapped pools, or (on_gpu, for RCP: no random reads over NFS) every fit pool copied to the
    GPU once and drawn there with the GPU generator."""

    def __init__(self, comp: dict, seed: int, device: str, p_mosaic: float = 0.0, on_gpu: bool = False):
        self.p_mosaic, self.on_gpu = p_mosaic, on_gpu
        tm = sets.training_maps()
        pp, pf = tm["pastis"]
        self.pastis = torch.as_tensor(np.ascontiguousarray(pp[pf]), device=device)
        self.cs_pool, self.cs_fit = tm["cloudsen12"]
        self.lu_pool, self.lu_fit = tm["lulc"]
        self.n_fit = {"pastis": len(pf), "cloudsen12": len(self.cs_fit), "lulc": len(self.lu_fit)}
        mon = {d: set(sets.subset_indices(d, "monitor").tolist()) for d in sets.DATASETS}
        assert not mon["pastis"] & set(pf.tolist()) and not mon["cloudsen12"] & set(self.cs_fit.tolist()) \
            and not mon["lulc"] & set(self.lu_fit.tolist()), "a monitor map is in fit"
        if on_gpu:
            self.cs_gpu = torch.as_tensor(np.ascontiguousarray(self.cs_pool[self.cs_fit]), device=device)
            self.lu_gpu = torch.as_tensor(np.ascontiguousarray(self.lu_pool[self.lu_fit]), device=device)
            assert len(self.cs_gpu) == len(self.cs_fit) and len(self.lu_gpu) == len(self.lu_fit)
        self.comp, self.device = comp, device
        self.rng = np.random.default_rng(seed)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed)
        self.gc = torch.Generator()                  # assignments: palette.render takes them on the CPU
        self.gc.manual_seed(seed + 1)

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "g": self.g.get_state(), "gc": self.gc.get_state()}

    def load_state(self, s: dict) -> None:
        self.rng.bit_generator.state = s["rng"]
        self.g.set_state(s["g"])
        self.gc.set_state(s["gc"])

    def mosaic(self, n: int) -> torch.Tensor:
        """n PASTIS maps, each a random 128-px window of four random fit maps tiled 2 x 2 (native scale)."""
        dv = self.device
        i = torch.randint(0, len(self.pastis), (n, 4), device=dv, generator=self.g)
        m = self.pastis[i]                                                  # (n, 4, 128, 128)
        big = torch.cat([torch.cat([m[:, 0], m[:, 1]], -1), torch.cat([m[:, 2], m[:, 3]], -1)], -2)
        y = torch.randint(0, 129, (n,), device=dv, generator=self.g)
        x = torch.randint(0, 129, (n,), device=dv, generator=self.g)
        ar = torch.arange(128, device=dv)
        rows, cols = (y[:, None] + ar)[:, :, None], (x[:, None] + ar)[:, None, :]
        return big[torch.arange(n, device=dv)[:, None, None], rows, cols]

    def draw(self) -> dict:
        out = {}
        n = self.comp.get("pastis", 0)
        if n:
            n_m = int(round(n * self.p_mosaic))
            i = torch.randint(0, len(self.pastis), (n - n_m,), device=self.device, generator=self.g)
            out["pastis"] = self.pastis[i] if not n_m else torch.cat([self.pastis[i], self.mosaic(n_m)])
        n = self.comp.get("cloudsen12", 0)
        if n and self.on_gpu:
            dv = self.device
            ii = torch.randperm(len(self.cs_gpu), device=dv, generator=self.g)[:n]
            ys = torch.randint(0, 512 - CS_CROP + 1, (n,), device=dv, generator=self.g)
            xs = torch.randint(0, 512 - CS_CROP + 1, (n,), device=dv, generator=self.g)
            ar = torch.arange(CS_CROP, device=dv)
            out["cloudsen12"] = self.cs_gpu[ii[:, None, None], (ys[:, None] + ar)[:, :, None],
                                            (xs[:, None] + ar)[:, None, :]]
        elif n:
            ii = self.rng.choice(self.cs_fit, n, replace=False)
            ys = self.rng.integers(0, 512 - CS_CROP + 1, n)
            xs = self.rng.integers(0, 512 - CS_CROP + 1, n)
            crops = np.stack([self.cs_pool[i, y:y + CS_CROP, x:x + CS_CROP] for i, y, x in zip(ii, ys, xs)])
            out["cloudsen12"] = torch.as_tensor(crops).to(self.device, non_blocking=True)
        n = self.comp.get("lulc", 0)
        if n and self.on_gpu:
            out["lulc"] = self.lu_gpu[torch.randperm(len(self.lu_gpu), device=self.device, generator=self.g)[:n]]
        elif n:
            ii = np.sort(self.rng.choice(self.lu_fit, n, replace=False))
            out["lulc"] = torch.as_tensor(np.ascontiguousarray(self.lu_pool[ii])).to(self.device, non_blocking=True)
        for d in out:
            k = torch.randint(0, 8, (len(out[d]),), device=self.device, generator=self.g)
            out[d] = dihedral(out[d], k).long()
        return out


def native_nll(logits: torch.Tensor, cidx: torch.Tensor, maps: torch.Tensor, ignore: int | None,
               bw: float) -> torch.Tensor:
    """NLL of the true colour on the native grid: log-softmax at the input size, the true colour's
    log-probability pooled by decode.to_native (the harness's pooling), void dropped."""
    lp = F.log_softmax(logits, 1).gather(1, cidx[:, None]).float()
    pooled = D.to_native(lp, maps.shape[-1])[:, 0]
    w = torch.ones_like(pooled) if ignore is None else (maps != ignore).float()
    if bw:
        w = w * (1 + bw * S.boundary(maps).float())
    return -(pooled * w).sum() / w.sum().clamp(min=1)


def lr_at(step: int, a, sched: dict) -> float:
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    ds = sched.get("decay_start")
    if ds is None or step < ds:
        return a.lr
    t = min(1.0, (step - ds) / max(sched["decay_steps"], 1))
    return a.lr * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ep = [p for p in ema.parameters()]
    mp = [p.detach() for p in model.parameters()]
    torch._foreach_mul_(ep, decay)
    torch._foreach_add_(ep, mp, alpha=1 - decay)


def monitor_read(ema: torch.nn.Module, run: str, step: int, batch: int) -> dict:
    """mIoU on every dataset's `monitor` set through the harness (nearest and vote), on the EMA."""
    tok = T.BTokenizer(ema, name=f"B_{run}_s{step}", weights=None, sha=f"in_training:{run}:{step}", batch=batch)
    out = {}
    for d in sets.DATASETS:
        r = H.sweep(tok, d, "monitor", [{"rule": "nearest"}, {"rule": "vote"}], cache=False)
        out[d] = {k: {m: r[k][m] for m in ("mIoU", "boundary_acc", "interior_acc", "pixel_acc", "iou_per_class")}
                  for k in r}
    out["mean_nearest"] = float(np.mean([out[d]["nearest"]["mIoU"] for d in sets.DATASETS]))
    out["mean_vote"] = float(np.mean([out[d]["vote"]["mIoU"] for d in sets.DATASETS]))
    return out


def make_probe(size: int, device: str) -> torch.Tensor:
    """A fixed batch for the activation probe: the first 4 monitor maps of each dataset (CloudSen12: the
    tile's top-left 224-px crop) rendered with each task's fixed assignment. Activation statistics only."""
    xs = []
    for d in sets.DATASETS:
        lab = torch.as_tensor(np.ascontiguousarray(sets.load(d, "monitor")[0][:4]), device=device).long()
        if d == "cloudsen12":
            lab = lab[:, :CS_CROP, :CS_CROP]
        xs.append(P.render(lab, P.TASKS[d].assign, size))
    return torch.cat(xs)


@torch.no_grad()
def act_probe(model: torch.nn.Module, probe: torch.Tensor) -> dict:
    """Max |output| of every top-level stage of the encoder's and decoder's conv stacks on the probe, in fp32.
    The fp16 run on the TITAN V died (2026-11-18) when the decoder's up-sampling conv at f2 passed 65,504
    (3,467 at step 5,000, 78,874 at step 6,367); this tracks that growth."""
    acts, hooks = {}, []
    for part in ("enc", "dec"):
        for i, m in enumerate(getattr(model, part).convs):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, k=f"{part}.convs.{i}": acts.__setitem__(k, float(out.detach().abs().max()))))
    try:
        logits, _, _ = model(probe)
    finally:
        for h in hooks:
            h.remove()
    top = sorted(acts.items(), key=lambda kv: -kv[1])[:3]
    return {"max": top[0][1], "top": top, "max_logit": float(logits.abs().max())}


def plateau(evals: list, step: int, a) -> bool:
    """True when the mean monitor mIoU gained under a.plateau_gain over the last quarter of the run
    (each end the mean of two reads)."""
    if step < a.min_stable or len(evals) < 4:
        return False
    s = np.array([e["step"] for e in evals])
    v = np.array([e["monitor"]["mean_nearest"] for e in evals])
    now = v[-2:].mean()
    j = int(np.argmin(np.abs(s - 0.75 * step)))
    j = max(1, min(j, len(v) - 3))
    before = v[j - 1:j + 1].mean()
    return bool(now - before < a.plateau_gain)


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def plot(curves: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tr = curves["train"]
    if tr:
        st = [r["step"] for r in tr]
        for d in sets.DATASETS:
            ax[0].plot(st, [r["loss"][d] for r in tr], label=d, lw=0.8)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("native NLL (train, 100-step mean)")
        ax[0].legend()
    ev = curves["evals"]
    if ev:
        st = [e["step"] for e in ev]
        for d in sets.DATASETS:
            ax[1].plot(st, [e["monitor"][d]["nearest"]["mIoU"] for e in ev], marker=".", label=f"{d} nearest")
        ax[1].plot(st, [e["monitor"]["mean_nearest"] for e in ev], "k-", marker="o", label="mean")
        ax[1].axhline(0.85, color="C0", ls=":", lw=0.8)
        ax[1].axhline(0.95, color="C1", ls=":", lw=0.8)
        ax[1].set_xlabel("step")
        ax[1].set_ylabel("monitor mIoU (EMA)")
        ax[1].legend(fontsize=7)
    fig.suptitle(curves["run"])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--levels", default="7,5,5,5,5")
    ap.add_argument("--widths", default="64,128,256,512")
    ap.add_argument("--enc-blocks", default="1,2,2,1")
    ap.add_argument("--dec-blocks", default="2,2,2,1")
    ap.add_argument("--attn-dim", type=int, default=512)
    ap.add_argument("--enc-attn", type=int, default=4)
    ap.add_argument("--dec-attn", type=int, default=6)
    ap.add_argument("--comp", default="12,10,10", help="maps per step: pastis,cloudsen12,lulc")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--boundary-weight", type=float, default=0.0)
    ap.add_argument("--pastis-mosaic", type=float, default=0.0, help="share of PASTIS maps drawn as 2x2 mosaics")
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--min-stable", type=int, default=30000)
    ap.add_argument("--max-stable", type=int, default=150000)
    ap.add_argument("--plateau-gain", type=float, default=0.003)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--min-decay", type=int, default=8000)
    ap.add_argument("--max-steps", type=int, default=0, help="hard stop (smoke tests); 0 = none")
    ap.add_argument("--force-decay", action="store_true", help="start the decay at the next read")
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile the training forward (not the EMA)")
    ap.add_argument("--precision", choices=("fp16", "bf16"), default="fp16",
                    help="autocast dtype: fp16 with a grad scaler (TITAN V), or bf16 without one, the loss in "
                         "fp32 (A100s: the project's rule)")
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
    cfg = dict(T.DEFAULT, input_size=a.size, levels=ints(a.levels), widths=ints(a.widths),
               enc_blocks=ints(a.enc_blocks), dec_blocks=ints(a.dec_blocks), attn_dim=a.attn_dim,
               enc_attn=a.enc_attn, dec_attn=a.dec_attn)
    comp = dict(zip(sets.DATASETS, ints(a.comp)))
    B = sum(comp.values())
    data_cfg = {"comp": comp, "seed": a.seed, "boundary_weight": a.boundary_weight, "cs_crop": CS_CROP,
                "dihedral": True, "assignment": "palette.shuffled_assignment per map",
                "pastis_mosaic": a.pastis_mosaic}
    if a.data_gpu:                         # only when set: a run started without it keeps its data_cfg
        data_cfg["draw"] = "fit pools on the GPU, GPU generator"
    run_cfg = {"precision": a.precision}   # a checkpoint without one is an fp16 run
    adt = torch.bfloat16 if a.precision == "bf16" else torch.float16

    torch.manual_seed(a.seed)
    model = T.MapTok(cfg).to(dev)
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    decay_p = [p for n, p in model.named_parameters() if p.ndim >= 2 and "pos" not in n]
    other_p = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay_p, "weight_decay": a.wd}, {"params": other_p, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=a.precision == "fp16")
    data = Data(comp, a.seed, dev, a.pastis_mosaic, on_gpu=a.data_gpu)
    n_par = sum(p.numel() for p in model.parameters())

    step, sched = 0, {"decay_start": None, "decay_steps": None}
    curves = {"run": a.run, "config": cfg, "data": data_cfg, "args": vars(a), "params": n_par, "gpu": gpu,
              "train": [], "evals": [], "wall": []}
    if a.precision != "fp16":
        curves["run_cfg"] = run_cfg
    best = {"mean_nearest": -1.0, "step": None}
    ck_path = out / "ckpt_last.pt"
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck["config"] != cfg or ck["data_cfg"] != data_cfg:
            raise SystemExit(f"resume refused: config or data differ from {ck_path}")
        if ck.get("run_cfg", {"precision": "fp16"}) != run_cfg:
            raise SystemExit(f"resume refused: precision differs from {ck_path} ({ck.get('run_cfg')})")
        if a.precision == "fp16" and not ck["scaler"]:
            raise SystemExit(f"resume refused: {ck_path} has no grad scaler state")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, sched, curves, best = ck["step"], ck["sched"], ck["curves"], ck["best"]
        data.load_state(ck["data"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        curves["args_resumed"] = curves.get("args_resumed", []) + [{"step": step, "args": vars(a)}]
        print(f"resumed from {ck_path} at step {step} (sched {sched}, best {best})", flush=True)
    else:
        print(f"run {a.run}: {n_par:,} params, codes {model.fsq.n_codes}, grid {a.size // 16}, batch {B} {comp}, "
              f"fit {data.n_fit}, gpu {gpu}", flush=True)
    (out / "config.json").write_text(json.dumps({"config": cfg, "data": data_cfg, "args": vars(a), "params": n_par}, indent=1))

    def checkpoint():
        save_atomic({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict(), "step": step, "sched": sched, "curves": curves, "best": best,
                     "data": data.state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                     "config": cfg, "data_cfg": data_cfg, "run_cfg": run_cfg}, ck_path)
        (out / "curves.json").write_text(json.dumps(curves))

    def evaluate():
        nonlocal best
        t0 = time.time()
        mon = monitor_read(ema, a.run, step, a.eval_batch)
        torch.cuda.empty_cache()
        hist = codes_hist.cpu().numpy()
        p = hist[hist > 0] / max(hist.sum(), 1)
        curves["wall_s"] = curves.get("wall_s", 0.0) + (time.time() - t0)
        ev = {"step": step, "lr": lr_at(step, a, sched), "monitor": mon, "seconds": time.time() - t0,
              "train_codes": {"distinct": int((hist > 0).sum()), "entropy_bits": float(-(p * np.log2(p)).sum()),
                              "tokens": int(hist.sum())}}
        curves["evals"].append(ev)
        line = " ".join(f"{d} {mon[d]['nearest']['mIoU']:.4f}/{mon[d]['vote']['mIoU']:.4f}" for d in sets.DATASETS)
        print(f"EVAL step {step} monitor nearest/vote: {line} mean {mon['mean_nearest']:.4f} | codes "
              f"{ev['train_codes']['distinct']} {ev['train_codes']['entropy_bits']:.2f} bits | {ev['seconds']:.0f} s",
              flush=True)
        if mon["mean_nearest"] > best["mean_nearest"]:
            best = {"mean_nearest": mon["mean_nearest"], "step": step,
                    "per_dataset": {d: mon[d]["nearest"]["mIoU"] for d in sets.DATASETS}}
            sha = T.export(out / "best.pt", ema, {"run": a.run, "step": step, "monitor": mon, "selected_on": "monitor"})
            best["sha256"] = sha
            print(f"  best -> {out / 'best.pt'} (sha256 {sha[:12]})", flush=True)
        codes_hist.zero_()
        if a.precision != "fp16":                       # runs started before the probe keep their records
            ev["act"] = act_probe(model, probe)
            print(f"  act max {ev['act']['max']:.0f} at {ev['act']['top'][0][0]}, max |logit| "
                  f"{ev['act']['max_logit']:.1f}", flush=True)
        plot(curves, out / "curves.png")

    fwd = torch.compile(model) if a.compile else model
    codes_hist = torch.zeros(model.fsq.n_codes, dtype=torch.long, device=dev)
    probe = make_probe(a.size, dev) if a.precision != "fp16" else None
    win = {d: [] for d in sets.DATASETS}
    win_gn, t_last, t_ck = [], time.time(), time.time()
    n_skip = 0                                         # bf16 steps skipped for a non-finite gradient (window)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    while True:
        if a.max_steps and step >= a.max_steps:
            break
        if sched["decay_start"] is not None and step >= sched["decay_start"] + sched["decay_steps"]:
            break
        lr = lr_at(step, a, sched)
        for g_ in opt.param_groups:
            g_["lr"] = lr
        maps = data.draw()
        xs, order = [], [d for d in sets.DATASETS if d in maps]
        for d in order:
            task = P.TASKS[d]
            assign = P.shuffled_assignment(len(maps[d]), task.n_classes, task.ignore, generator=data.gc, device="cpu")
            xs.append(P.render(maps[d], assign, a.size))
        x = torch.cat(xs)
        if a.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=adt):
            logits, idx, cidx = fwd(x, adt)
        losses, o = {}, 0
        for d in order:
            n = len(maps[d])
            lg = logits[o:o + n] if a.precision == "fp16" else logits[o:o + n].float()
            losses[d] = native_nll(lg, cidx[o:o + n], maps[d], P.TASKS[d].ignore, a.boundary_weight)
            o += n
        loss = sum(comp[d] / B * losses[d] for d in order)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        if a.precision != "fp16" and not bool(torch.isfinite(gnorm)):
            opt.zero_grad(set_to_none=True)            # no scaler in bf16: never step on a nan/inf gradient
            n_skip += 1
        else:
            scaler.step(opt)
            scaler.update()
        ema_update(ema, model, min(a.ema, (1 + step) / (10 + step)))
        codes_hist += torch.bincount(idx.flatten(), minlength=model.fsq.n_codes)
        step += 1
        for d in order:
            win[d].append(losses[d].detach())
        win_gn.append(gnorm.detach())
        if step % 100 == 0:
            now = time.time()
            curves["wall_s"] = curves.get("wall_s", 0.0) + (now - t_last)
            rec = {"step": step, "lr": lr, "loss": {d: float(torch.stack(win[d]).mean()) for d in order},
                   "grad_norm": float(torch.stack(win_gn).float().mean()), "scale": scaler.get_scale(),
                   "steps_per_s": 100 / (now - t_last), "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            if a.precision != "fp16":
                rec["skipped"] = n_skip
            curves["train"].append(rec)
            dead = (a.precision == "fp16" and scaler.get_scale() < 1.0) or n_skip > 20
            if dead:                                   # never train on silently (CLAUDE.md: fp16 killed T360)
                (out / "dead.json").write_text(json.dumps({"step": step, "record": rec}, indent=1))
                (out / "curves.json").write_text(json.dumps(curves))
                raise SystemExit(f"DEAD at step {step}: {rec} (ckpt_last.pt is the last healthy save)")
            n_skip = 0
            if step % 500 == 0:
                print(f"step {step} lr {lr:.2e} " + " ".join(f"{d} {v:.4f}" for d, v in rec["loss"].items())
                      + f" gn {rec['grad_norm']:.3f} scale {rec['scale']:.0f} {rec['steps_per_s']:.2f} it/s "
                      f"mem {rec['max_mem_gb']:.2f} GB" + (f" skipped {rec['skipped']}" if "skipped" in rec else ""),
                      flush=True)
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
    last_sha = T.export(out / "last.pt", ema, {"run": a.run, "step": step, "selected_on": "last"})
    finished = sched["decay_start"] is not None and step >= sched["decay_start"] + sched["decay_steps"]
    done = {"run": a.run, "steps": step, "finished_schedule": finished, "best": best, "sched": sched,
            "last": str(out / "last.pt"), "last_sha256": last_sha, "best_path": str(out / "best.pt"),
            "params": n_par, "gpu": gpu, "wall_hours": curves.get("wall_s", 0.0) / 3600}
    (out / ("done.json" if finished else "stopped.json")).write_text(json.dumps(done, indent=1))
    print("DONE" if finished else "STOPPED", json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
