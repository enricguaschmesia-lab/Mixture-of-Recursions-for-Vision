"""Throughput and memory of agent B's training step under a few implementation variants (one-off).

    python -m eo.experiments.segmentation.step1_B_bench --variants base,cl,cl_compile --batch 32
Each variant: a fresh model, 10 warmup steps (plus compilation), 40 timed steps on random renders.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from eo.experiments.segmentation import step1_B_tok as T
from eo.experiments.segmentation.step1_B_train import Data
from eo.maps import decode as D
from eo.maps import palette as P
from eo.maps import sets


def run(variant: str, batch: int, size: int, steps: int = 40) -> dict:
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    model = T.MapTok(dict(T.DEFAULT, input_size=size)).cuda()
    cl = "cl" in variant
    if cl:
        model = model.to(memory_format=torch.channels_last)
    fwd = torch.compile(model) if "compile" in variant else model
    fp32_loss = "fp32loss" in variant
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    comp = {"pastis": batch * 3 // 8, "cloudsen12": batch * 5 // 16, "lulc": batch - batch * 3 // 8 - batch * 5 // 16}
    data = Data(comp, 0, "cuda")
    torch.cuda.reset_peak_memory_stats()
    t0, tw = None, time.time()
    for s in range(10 + steps):
        if s == 10:
            torch.cuda.synchronize()
            t0 = time.time()
            warm = t0 - tw
        maps = data.draw()
        xs = []
        for d in sets.DATASETS:
            t = P.TASKS[d]
            xs.append(P.render(maps[d], P.shuffled_assignment(len(maps[d]), t.n_classes, t.ignore, generator=data.gc), size))
        x = torch.cat(xs)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, idx, cidx = fwd(x, torch.float16)
        o, loss = 0, 0.0
        for d in sets.DATASETS:
            n = len(maps[d])
            lg = logits[o:o + n].float() if fp32_loss else logits[o:o + n]
            lp = F.log_softmax(lg, 1).gather(1, cidx[o:o + n, None]).float()
            pooled = D.to_native(lp, maps[d].shape[-1])[:, 0]
            w = torch.ones_like(pooled) if P.TASKS[d].ignore is None else (maps[d] != P.TASKS[d].ignore).float()
            loss = loss + n / batch * (-(pooled * w).sum() / w.sum())
            o += n
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"variant": variant, "batch": batch, "size": size, "warmup_s": warm, "it_s": steps / dt, "img_s": steps * batch / dt,
            "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "loss": float(loss)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,fp32loss,cl,cl_compile")
    ap.add_argument("--batch", default="32")
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()
    print(torch.cuda.get_device_name(0), flush=True)
    for v in a.variants.split(","):
        b, v = (int(v.split("@")[1]), v.split("@")[0]) if "@" in v else (int(a.batch), v)
        r = run(v, b, a.size)
        print(json.dumps(r), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
