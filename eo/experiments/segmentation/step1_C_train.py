"""Train agent C's f8 map tokenizer (A2 Step 1, long slot) on palette-shuffled renders of the `fit` maps.

Data (eo.maps.sets.training_maps(), the `fit` part of each pool only):
  - PASTIS folds 1-3 less monitor: 1,355 maps at 128 px, on the GPU;
  - CloudSen12 train less monitor: 3,800 tiles at 512 px, on the GPU; a random 224-px crop per draw
    (the read's crop size; upsampled to `size` by the render);
  - LULC pool less monitor: 61,928 maps at 224 px, memory-mapped, fetched in chunks.
Every step draws a fixed number of maps per dataset (`--mix`), applies a random dihedral transform at
the native grid (and, for PASTIS, an optional whole-pixel translation with void fill), a random
injective class -> colour assignment per map (`palette.shuffled_assignment`, void / no data on white)
and renders with `palette.render` (nearest) at `size`.

Loss: the harness's read quantity. log_softmax over the 32 palette logits at `size`, pooled to the
native grid by `eo.maps.decode.to_native` (the adjoint of the render), NLL against each map's own
colour targets; void and no-data pixels ignored. The three datasets' losses are averaged with equal
weight (`--loss-w`); an optional boundary weight (`--boundary-w`) up-weights native boundary pixels.

Monitor: every `--eval-every` steps the EMA weights are read through `eo.maps.harness.sweep(tok, d,
"monitor", [nearest], cache=False)` on the three monitor sets, with code use (distinct codes, entropy)
on the monitor tokens. Schedule: warmup, a constant LR, then a linear cooldown to zero that starts when
the mean monitor mIoU has flattened (its running maximum gained under `--flat-gain` over the last
quarter of the steps, after `--min-steps`), at `--deadline`, at `--max-steps` minus the cooldown, or
when the file STOP_COOLDOWN appears in the run directory. The best checkpoint (by the monitor mean) and
the last are kept as loadable tokenizer files (tok_best.pt, tok_last.pt); ckpt_last.pt is resumable
(model, EMA, optimizer, RNG states, schedule, history) and is rewritten at every evaluation.

    python -X faulthandler -m eo.experiments.segmentation.step1_C_train --out <run_dir> [--levels 8,5,5,5]

Runs in the `a2seg` conda env, on the GTX Titan X (fp32), or on RCP's A100s in the `mor-eo` env
(`--expect-gpu A100 --amp bf16 --lulc-gpu`: bf16 autocast for the forward pass, the loss and FSQ in
fp32, the LULC fit maps held on the GPU instead of read from the NFS memory map). A run whose
done.json exists exits at once (a recreated RCP pod restarts the command from the top), and any other
restart resumes from ckpt_last.pt. The monitor reads the EMA weights in fp32 (the harness turns TF32
off for encode and decode), as the final read does.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eo.maps import decode as D
from eo.maps import harness as H
from eo.maps import palette as P
from eo.maps import scoring as S
from eo.maps import sets
from eo.experiments.segmentation.step1_C_tok import MapAE8, StepCTokenizer, n_params, save_weights

DATASETS = ("pastis", "cloudsen12", "lulc")
CROP = 224                                   # CloudSen12's training crops: the read's crop size
IGNORE = -100


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dihedral_perms(n: int, device) -> torch.Tensor:
    """(8, n*n): source flat index of every output pixel under the 8 dihedral transforms."""
    base = torch.arange(n * n, device=device).view(n, n)
    out = []
    for t in range(8):
        x = torch.rot90(base, t % 4, (0, 1))
        out.append((torch.flip(x, (1,)) if t >= 4 else x).reshape(-1))
    return torch.stack(out)


class Sampler:
    """Draws one training batch: per dataset a fixed number of maps, transformed, shuffled, rendered."""

    def __init__(self, args, dev):
        self.dev, self.size = dev, args.size
        self.counts = {d: int(v) for d, v in (kv.split(":") for kv in args.mix.split(","))}
        assert set(self.counts) == set(DATASETS), self.counts
        tm = sets.training_maps()
        pool, fit = tm["pastis"]
        self.pastis = torch.as_tensor(np.asarray(pool[fit])).to(dev)              # (1355, 128, 128) u8
        pool, fit = tm["cloudsen12"]
        self.cloud = torch.as_tensor(np.asarray(pool[fit])).to(dev)               # (3800, 512, 512) u8
        self.lulc_pool, self.lulc_fit = tm["lulc"]                                 # memmap, (61928,)
        self.lulc_all = None
        if getattr(args, "lulc_gpu", False):                                       # (61928, 224, 224) u8, 3.1 GB
            self.lulc_all = torch.as_tensor(np.asarray(self.lulc_pool[self.lulc_fit])).to(dev)
        self.n_fit = {"pastis": len(self.pastis), "cloudsen12": len(self.cloud), "lulc": len(self.lulc_fit)}
        self.perms = {n: dihedral_perms(n, dev) for n in (128, CROP)}
        self.g = torch.Generator(device=dev).manual_seed(args.seed)               # transforms, crops
        self.gc = torch.Generator().manual_seed(args.seed + 1)                    # colour assignments
        self.rng = np.random.default_rng(args.seed + 2)                           # LULC rows
        self.shift, self.shift_p = args.pastis_shift, args.pastis_shift_p
        self.chunk = args.lulc_chunk
        self.lulc_buf, self.lulc_pos = None, 0

    def state(self) -> dict:
        return {"g": self.g.get_state(), "gc": self.gc.get_state(), "rng": self.rng.bit_generator.state,
                "lulc_buf": None if self.lulc_buf is None else self.lulc_buf.cpu(), "lulc_pos": self.lulc_pos}

    def load_state(self, st: dict) -> None:
        self.g.set_state(st["g"])
        self.gc.set_state(st["gc"])
        self.rng.bit_generator.state = st["rng"]
        self.lulc_buf = None if st["lulc_buf"] is None else st["lulc_buf"].to(self.dev)
        self.lulc_pos = st["lulc_pos"]

    def _lulc(self, k: int) -> torch.Tensor:
        """k LULC maps; fetched from the memory map `chunk` steps at a time (one host sync per chunk), or
        drawn on the GPU when the fit maps are held there (--lulc-gpu)."""
        if self.lulc_all is not None:
            i = torch.randint(0, len(self.lulc_all), (k,), generator=self.g, device=self.dev)
            return self.lulc_all[i]
        if self.lulc_buf is None or self.lulc_pos + k > len(self.lulc_buf):
            rows = np.sort(self.rng.choice(self.lulc_fit, size=k * self.chunk, replace=True))
            buf = torch.as_tensor(np.asarray(self.lulc_pool[rows]))
            perm = torch.as_tensor(self.rng.permutation(len(rows)))
            self.lulc_buf, self.lulc_pos = buf[perm].to(self.dev), 0
        out = self.lulc_buf[self.lulc_pos:self.lulc_pos + k]
        self.lulc_pos += k
        return out

    def _dihedral(self, lab: torch.Tensor) -> torch.Tensor:
        B, n, _ = lab.shape
        t = torch.randint(0, 8, (B,), generator=self.g, device=self.dev)
        return torch.gather(lab.reshape(B, -1), 1, self.perms[n][t]).view(B, n, n)

    def _translate(self, lab: torch.Tensor, void: int) -> torch.Tensor:
        """PASTIS: a whole-native-pixel shift of up to `shift` px with probability `shift_p`, the
        uncovered band filled with void (ignored in the loss, white in the render)."""
        if self.shift <= 0 or self.shift_p <= 0:
            return lab
        B, n, _ = lab.shape
        on = (torch.rand(B, generator=self.g, device=self.dev) < self.shift_p).long()
        dy = torch.randint(-self.shift, self.shift + 1, (B,), generator=self.g, device=self.dev) * on
        dx = torch.randint(-self.shift, self.shift + 1, (B,), generator=self.g, device=self.dev) * on
        ar = torch.arange(n, device=self.dev)
        yy = ar[None, :, None] - dy[:, None, None]
        xx = ar[None, None, :] - dx[:, None, None]
        ok = (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
        b = torch.arange(B, device=self.dev)[:, None, None]
        src = lab[b, yy.clamp(0, n - 1), xx.clamp(0, n - 1)]
        return torch.where(ok, src, torch.full_like(src, void))

    def draw(self):
        """-> x (B, 3, size, size) in [0, 1]; per dataset (name, k, native, colour targets with IGNORE,
        native labels)."""
        xs, parts = [], []
        for d in DATASETS:
            k = self.counts[d]
            if k == 0:
                continue
            task = P.TASKS[d]
            if d == "pastis":
                i = torch.randint(0, self.n_fit[d], (k,), generator=self.g, device=self.dev)
                lab = self._translate(self.pastis[i].long(), task.ignore)
            elif d == "cloudsen12":
                i = torch.randint(0, self.n_fit[d], (k,), generator=self.g, device=self.dev)
                y0 = torch.randint(0, 512 - CROP + 1, (k,), generator=self.g, device=self.dev)
                x0 = torch.randint(0, 512 - CROP + 1, (k,), generator=self.g, device=self.dev)
                ar = torch.arange(CROP, device=self.dev)
                lab = self.cloud[i[:, None, None], (y0[:, None] + ar)[:, :, None],
                                 (x0[:, None] + ar)[:, None, :]].long()
            else:
                lab = self._lulc(k).long()
            lab = self._dihedral(lab)
            assign = P.shuffled_assignment(k, task.n_classes, task.ignore, generator=self.gc)   # CPU
            xs.append(P.render(lab, assign, self.size))
            a = assign.to(self.dev)
            tgt = torch.gather(a, 1, lab.reshape(k, -1)).view_as(lab)
            if task.ignore is not None:
                tgt = torch.where(lab == task.ignore, torch.full_like(tgt, IGNORE), tgt)
            parts.append((d, k, lab.shape[-1], tgt, lab))
        return torch.cat(xs), parts

    def class_freqs(self) -> dict:
        """Per dataset, each class's share of the fit maps' scored pixels (the ignored class counts 0).
        LULC from all fit maps when they are on the GPU, else from 5,000 of them (a fixed draw)."""
        out = {}
        for d in DATASETS:
            task = P.TASKS[d]
            n = task.n_classes
            cnt = torch.zeros(n, dtype=torch.float64)
            if d == "pastis":
                src = [self.pastis]
            elif d == "cloudsen12":
                src = [self.cloud[i:i + 200] for i in range(0, len(self.cloud), 200)]
            elif self.lulc_all is not None:
                src = [self.lulc_all[i:i + 2000] for i in range(0, len(self.lulc_all), 2000)]
            else:
                rows = np.sort(np.random.default_rng(0).choice(self.lulc_fit, size=5000, replace=False))
                src = [torch.as_tensor(np.asarray(self.lulc_pool[rows[i:i + 500]]))
                       for i in range(0, len(rows), 500)]
            for s in src:
                cnt += torch.bincount(s.reshape(-1).long(), minlength=n)[:n].double().cpu()
            if task.ignore is not None:
                cnt[task.ignore] = 0
            out[d] = cnt / cnt.sum()
        return out


def class_weights(freqs: dict, gamma: float) -> dict:
    """w_c = share_c ** -gamma per dataset, normalised so that the pixel-weighted mean weight over the fit
    maps is 1. mIoU's sensitivity to one pixel of class c goes as 1 / (pixels of c), so gamma = 1 matches
    it and gamma = 0 is the plain NLL. The tokenizer never sees the class (shuffled palettes): the weight
    only tells it which regions the metric counts most."""
    out = {}
    for d, f in freqs.items():
        w = torch.where(f > 0, f.clamp(min=1e-12) ** (-gamma), torch.zeros_like(f))
        out[d] = (w / (f * w).sum()).float()
    return out


def losses(logits: torch.Tensor, parts, boundary_w: float, cw: dict | None = None):
    """Per dataset: NLL of the pooled log-probabilities at the native grid, and pixel accuracy.
    cw: optional per-class weights (class_weights), indexed by each pixel's true class."""
    logp = F.log_softmax(logits.float(), 1)
    out, off = {}, 0
    for d, k, native, tgt, lab in parts:
        lp = D.to_native(logp[off:off + k], native)
        off += k
        nll = F.nll_loss(lp, tgt, ignore_index=IGNORE, reduction="none")          # (k, n, n)
        valid = (tgt != IGNORE).float()
        w = valid
        if boundary_w > 0:
            w = valid * (1 + boundary_w * S.boundary(lab).float())
        if cw is not None:
            w = w * cw[d][lab]
        loss = (nll * w).sum() / w.sum().clamp(min=1)
        acc = ((lp.argmax(1) == tgt).float() * valid).sum() / valid.sum().clamp(min=1)
        out[d] = (loss, acc)
    return out


@torch.no_grad()
def monitor(ema: MapAE8, run: str, step: int, batch: int) -> dict:
    tok = StepCTokenizer(ema, sha=f"live-{run}-{step}", name=f"stepC_live_{run}_s{step}", batch=batch)
    res, codes = {}, {}
    for d in DATASETS:
        r = H.sweep(tok, d, "monitor", [{"rule": "nearest"}], cache=False)["nearest"]
        res[d] = {k: r[k] for k in ("mIoU", "pixel_acc", "boundary_acc", "interior_acc")}
        idx = H.tokens(tok, d, "monitor", cache=False)
        cs = S.code_stats(idx, None, tok.n_codes)
        codes[d] = {"distinct": cs["read_distinct"], "entropy_bits": round(cs["read_entropy_bits"], 4)}
    ema.train(False)
    return {"monitor": res, "mean": float(np.mean([res[d]["mIoU"] for d in DATASETS])), "codes": codes}


def lr_at(step: int, args, cool) -> float:
    lr = args.lr * min(1.0, (step + 1) / args.warmup)
    if cool is not None:
        start, length = cool
        lr = args.lr * max(0.0, 1 - (step - start) / length)
    return lr


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run directory under /data/enric/reports/a2_seg/step1/C/")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--levels", default="8,5,5,5")
    ap.add_argument("--c1", type=int, default=128)
    ap.add_argument("--c2", type=int, default=256)
    ap.add_argument("--blocks", default="2,4,6,2", help="n1_enc,n2_enc,n2_dec,n1_dec")
    ap.add_argument("--mix", default="pastis:12,cloudsen12:10,lulc:10", help="maps per step and dataset")
    ap.add_argument("--loss-w", default="1,1,1", help="pastis,cloudsen12,lulc loss weights")
    ap.add_argument("--boundary-w", type=float, default=0.0)
    ap.add_argument("--class-gamma", type=float, default=0.0,
                    help="weight each pixel by its true class's share of the fit maps ** -gamma (0: off)")
    ap.add_argument("--pastis-shift", type=int, default=32)
    ap.add_argument("--pastis-shift-p", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--max-steps", type=int, default=150000)
    ap.add_argument("--min-steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--flat-gain", type=float, default=0.003)
    ap.add_argument("--cool-frac", type=float, default=0.15)
    ap.add_argument("--cool-min", type=int, default=4000)
    ap.add_argument("--deadline", default=None, help="'YYYY-mm-dd HH:MM': start the cooldown at the first eval after it")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--lulc-chunk", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-gpu", default="TITAN X")
    ap.add_argument("--amp", default="none", choices=("none", "bf16"),
                    help="bf16: autocast the forward pass (the A100 rule); the loss and FSQ stay fp32")
    ap.add_argument("--lulc-gpu", action="store_true", help="hold the LULC fit maps on the GPU (3.1 GB)")
    ap.add_argument("--exit-at", type=int, default=None,
                    help="smoke only: exit right after the evaluation and save at this step, without "
                         "done.json (a simulated preemption)")
    a = ap.parse_args()

    out = Path(a.out)
    if (out / "done.json").exists():
        print(f"{now()} {out / 'done.json'} exists: the run is done, nothing to do", flush=True)
        return
    name = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in name:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {name}")
    if torch.cuda.device_count() > 1 or "TITAN" in (a.expect_gpu or ""):   # the workstation: by name, PCI order
        assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "set CUDA_DEVICE_ORDER=PCI_BUS_ID"
    if a.exit_at is not None:
        assert a.exit_at % a.eval_every == 0, "--exit-at must be a multiple of --eval-every"
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    out.mkdir(parents=True, exist_ok=True)
    run = out.name
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if a.amp == "bf16" else contextlib.nullcontext
    n1e, n2e, n2d, n1d = (int(v) for v in a.blocks.split(","))
    cfg = {"size": a.size, "levels": [int(v) for v in a.levels.split(",")], "unshuffle": 4, "c1": a.c1,
           "c2": a.c2, "n1_enc": n1e, "n2_enc": n2e, "n2_dec": n2d, "n1_dec": n1d, "groups": 32}
    lw = dict(zip(DATASETS, (float(v) for v in a.loss_w.split(","))))

    torch.manual_seed(a.seed)
    model = MapAE8(cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (decay if p.ndim == 4 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.99))
    sampler = Sampler(a, dev)
    cw = None
    if a.class_gamma > 0:
        freqs = sampler.class_freqs()
        cw = {d: v.to(dev) for d, v in class_weights(freqs, a.class_gamma).items()}
        print(f"{now()} class weights (gamma {a.class_gamma}): " + "; ".join(
            f"{d} " + " ".join(f"{P.TASKS[d].class_names[c][:10]} {freqs[d][c]:.4f}->{float(cw[d][c]):.2f}"
                               for c in range(P.TASKS[d].n_classes) if freqs[d][c] > 0) for d in DATASETS),
            flush=True)
    step, cool, history, best, t_prev = 0, None, [], {"step": -1, "mean": -1.0}, 0.0
    ck = out / "ckpt_last.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        assert st["config"] == cfg, (st["config"], cfg)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sampler.load_state(st["sampler"])
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state(st["cuda_rng"])
        step, cool, history, best, t_prev = st["step"], st["cool"], st["history"], st["best"], st["elapsed"]
        print(f"{now()} resumed from {ck} at step {step} (cooldown {cool})", flush=True)
    else:
        (out / "config.json").write_text(json.dumps({"config": cfg, "args": vars(a), "gpu": name,
                                                     "params": n_params(model), "started": now(),
                                                     "torch": torch.__version__}, indent=1))
    print(f"{now()} run {run} on {name}: cfg {cfg}, params {n_params(model):,}, codes {model.fsq.n_codes}, "
          f"mix {sampler.counts}, fit sizes {sampler.n_fit}", flush=True)

    deadline = dt.datetime.strptime(a.deadline, "%Y-%m-%d %H:%M") if a.deadline else None
    t0 = time.time() - t_prev
    acc_loss = {d: torch.zeros((), device=dev) for d in DATASETS}
    acc_acc = {d: torch.zeros((), device=dev) for d in DATASETS}
    acc_gn, n_acc, t_log = torch.zeros((), device=dev), 0, time.time()
    log_f = open(out / "train_log.jsonl", "a")
    end = None if cool is None else cool[0] + cool[1]

    def evaluate_and_save(final: bool = False):
        nonlocal best, cool, end
        t_e = time.time()
        m = monitor(ema, run, step, a.eval_batch)
        m.update({"step": step, "time_h": (time.time() - t0) / 3600, "lr": lr_at(step, a, cool),
                  "eval_s": time.time() - t_e, "when": now(), "cooldown": cool})
        history.append(m)
        line = " ".join(f"{d} {m['monitor'][d]['mIoU']:.4f}" for d in DATASETS)
        cds = " ".join(f"{d} {m['codes'][d]['distinct']}/{m['codes'][d]['entropy_bits']:.2f}b" for d in DATASETS)
        print(f"{now()} EVAL step {step}: mean {m['mean']:.4f} | {line} | codes {cds} | {m['eval_s']:.0f} s",
              flush=True)
        if m["mean"] > best["mean"]:
            best = {"step": step, "mean": m["mean"], "monitor": m["monitor"]}
            sha = save_weights(ema, out / "tok_best.pt", {"run": run, "step": step, "monitor": m["monitor"],
                                                           "mean": m["mean"], "codes": m["codes"]})
            best["sha256"] = sha
            print(f"{now()}   new best (tok_best.pt sha256 {sha[:16]})", flush=True)
        # flattening: the running max of the monitor mean over the last quarter of the steps
        if cool is None and not final:
            reason = None
            if step >= a.min_steps:
                runmax = lambda s_: max([h["mean"] for h in history if h["step"] <= s_] or [-1])
                gain = runmax(step) - runmax(0.75 * step)
                print(f"{now()}   gain over the last quarter ({int(0.75 * step)}..{step}): {100 * gain:.2f} pts",
                      flush=True)
                if gain < a.flat_gain:
                    reason = f"flat: +{100 * gain:.2f} pts over the last quarter"
            if deadline and dt.datetime.now() >= deadline:
                reason = f"deadline {a.deadline}"
            if (out / "STOP_COOLDOWN").exists():
                reason = "STOP_COOLDOWN"
            length = max(a.cool_min, int(a.cool_frac * step))
            if step + length >= a.max_steps:
                reason = reason or "max steps"
            if reason:
                cool = (step, length)
                end = step + length
                history[-1]["cooldown_started"] = reason
                print(f"{now()}   COOLDOWN from step {step} for {length} steps ({reason})", flush=True)
        state = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                 "sampler": sampler.state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(), "step": step, "cool": cool, "history": history,
                 "best": best, "elapsed": time.time() - t0, "config": cfg, "args": vars(a)}
        atomic_save(state, ck)
        (out / "history.json").write_text(json.dumps(history, indent=1))

    if step == 0 and not history:
        evaluate_and_save()
    model.train()
    while end is None or step < end:
        lr = lr_at(step, a, cool)
        for gp in opt.param_groups:
            gp["lr"] = lr
        x, parts = sampler.draw()
        with autocast():
            logits, _ = model(x)
        ls = losses(logits, parts, a.boundary_w, cw)
        loss = sum(lw[d] * ls[d][0] for d in ls) / sum(lw[d] for d in ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        dcy = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            ep, mp = list(ema.parameters()), list(model.parameters())
            torch._foreach_mul_(ep, dcy)
            torch._foreach_add_(ep, mp, alpha=1 - dcy)
        step += 1
        for d in ls:
            acc_loss[d] += ls[d][0].detach()
            acc_acc[d] += ls[d][1].detach()
        acc_gn += gn.detach()
        n_acc += 1
        if step % a.log_every == 0:
            rec = {"step": step, "lr": lr, "grad_norm": float(acc_gn) / n_acc,
                   "loss": {d: float(acc_loss[d]) / n_acc for d in ls},
                   "acc": {d: float(acc_acc[d]) / n_acc for d in ls},
                   "s_per_step": (time.time() - t_log) / n_acc, "elapsed_h": (time.time() - t0) / 3600}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if not math.isfinite(rec["grad_norm"]) or not all(math.isfinite(v) for v in rec["loss"].values()):
                raise SystemExit(f"non-finite loss or grad norm at step {step}: {rec}")
            if step % (a.log_every * 10) == 0 or step <= 1000:
                print(f"{now()} step {step} lr {lr:.2e} gn {rec['grad_norm']:.3f} loss "
                      + " ".join(f"{d} {rec['loss'][d]:.4f}" for d in ls) + " | acc "
                      + " ".join(f"{d} {rec['acc'][d]:.4f}" for d in ls)
                      + f" | {rec['s_per_step']:.3f} s/step, {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                      flush=True)
            for d in DATASETS:
                acc_loss[d].zero_()
                acc_acc[d].zero_()
            acc_gn.zero_()
            n_acc, t_log = 0, time.time()
        if step % a.eval_every == 0 or (end is not None and step >= end):
            evaluate_and_save(final=end is not None and step >= end)
            model.train()
            if a.exit_at is not None and step == a.exit_at and not (end is not None and step >= end):
                log_f.close()
                print(f"{now()} EXIT-AT step {step} (simulated preemption: no done.json)", flush=True)
                sys.exit(0)
    save_weights(ema, out / "tok_last.pt", {"run": run, "step": step, "monitor": history[-1]["monitor"],
                                            "mean": history[-1]["mean"], "codes": history[-1]["codes"]})
    done = {"run": run, "steps": step, "hours": (time.time() - t0) / 3600, "best": best,
            "last": {k: history[-1][k] for k in ("step", "mean", "monitor", "codes")}, "finished": now(),
            "gpu": name}
    (out / "done.json").write_text(json.dumps(done, indent=1))
    print(f"{now()} DONE {json.dumps(done)}", flush=True)


if __name__ == "__main__":
    main()
