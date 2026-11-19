"""Agent C's smoke test on one RCP A100 (A2 Step 1, the f8 map tokenizer): the card, the speed, bf16, a
simulated preemption and its resume, the done-file exit, and determinism of the trained tokenizer.

    python -X faulthandler -u -m eo.experiments.segmentation.step1_C_smoke --out <dir> --size 256 \
        --levels 8,5,5,5 --expect-gpu A100

Checks, each printed PASS / FAIL, all written to <dir>/smoke.json:
  1. the card's name;
  2. speed: one training step exactly as step1_C_train takes it (draw, forward, the native-grid NLL,
     backward, clip, AdamW), at 1x and 3x the Titan X run's batch (32 maps), in bf16 autocast with
     NCHW and channels-last tensors, and in fp32 (TF32); s/step, maps/s, peak memory;
  3. step1_C_train with --exit-at (a simulated preemption: exit 0, no done.json, ckpt_last.pt at that
     step), then the same command again (it resumes at that step and finishes: done.json), then once
     more (it must exit at once: the run is done);
  4. the trained tokenizer, loaded by step1_C_tok.load, encodes the monitor maps twice bit-identically
     at its fixed batch shape (and decodes twice identically), through the harness's own path.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eo.maps import harness as H
from eo.maps.tokenizers import run_fixed
from eo.experiments.segmentation import step1_C_tok as T
from eo.experiments.segmentation.step1_C_train import Sampler, losses, DATASETS


def bench(size: int, levels: str, out: dict) -> None:
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    ns = SimpleNamespace(size=size, mix="pastis:12,cloudsen12:10,lulc:10", seed=0, pastis_shift=32,
                         pastis_shift_p=0.5, lulc_chunk=64, lulc_gpu=True)
    sampler = Sampler(ns, "cuda")
    out["data_load_s"] = round(time.time() - t0, 1)
    print(f"data on the GPU in {out['data_load_s']} s: {sampler.n_fit}", flush=True)
    cfg = {"size": size, "levels": [int(v) for v in levels.split(",")]}
    rows = []
    for scale in (1, 3):
        sampler.counts = {"pastis": 12 * scale, "cloudsen12": 10 * scale, "lulc": 10 * scale}
        for mode in ("bf16", "bf16_cl", "fp32_tf32"):
            torch.manual_seed(0)
            model = T.MapAE8(cfg).cuda().train()
            cl = mode.endswith("_cl")
            if cl:
                model = model.to(memory_format=torch.channels_last)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if mode.startswith("bf16") \
                else contextlib.nullcontext
            torch.cuda.reset_peak_memory_stats()

            def one():
                x, parts = sampler.draw()
                if cl:
                    x = x.contiguous(memory_format=torch.channels_last)
                with ac():
                    logits, _ = model(x)
                ls = losses(logits, parts, 0.0)
                loss = sum(v[0] for v in ls.values()) / len(ls)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                return loss
            for _ in range(5):
                one()
            torch.cuda.synchronize()
            t1 = time.time()
            n = 20
            for _ in range(n):
                loss = one()
            torch.cuda.synchronize()
            s = (time.time() - t1) / n
            r = {"maps_per_step": 32 * scale, "mode": mode, "s_per_step": round(s, 4),
                 "maps_per_s": round(32 * scale / s, 1), "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 "loss": round(float(loss), 4)}
            rows.append(r)
            print("bench", json.dumps(r), flush=True)
            del model, opt
            torch.cuda.empty_cache()
    out["bench"] = rows
    del sampler
    torch.cuda.empty_cache()


def train_cmd(run: Path, size: int, levels: str, gpu: str) -> list[str]:
    return [sys.executable, "-X", "faulthandler", "-u", "-m", "eo.experiments.segmentation.step1_C_train",
            "--out", str(run), "--size", str(size), "--levels", levels,
            "--mix", "pastis:36,cloudsen12:30,lulc:30", "--amp", "bf16", "--lulc-gpu", "--expect-gpu", gpu,
            "--max-steps", "600", "--min-steps", "200", "--eval-every", "200", "--warmup", "100",
            "--cool-min", "100", "--cool-frac", "0.25", "--log-every", "50"]


def resume_test(run: Path, size: int, levels: str, gpu: str, out: dict) -> bool:
    cmd = train_cmd(run, size, levels, gpu)
    ok = True
    t0 = time.time()
    r1 = subprocess.run(cmd + ["--exit-at", "200"])
    st = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    a = {"rc": r1.returncode, "done_json": (run / "done.json").exists(), "ckpt_step": st["step"],
         "s": round(time.time() - t0, 1)}
    good = a["rc"] == 0 and not a["done_json"] and a["ckpt_step"] == 200
    print(f"{'PASS' if good else 'FAIL'} simulated preemption: {a}", flush=True)
    ok &= good
    t0 = time.time()
    r2 = subprocess.run(cmd)
    done = json.loads((run / "done.json").read_text()) if (run / "done.json").exists() else None
    hist = json.loads((run / "history.json").read_text())
    b = {"rc": r2.returncode, "done": done is not None, "steps": done and done["steps"],
         "history_steps": [h["step"] for h in hist], "s": round(time.time() - t0, 1)}
    good = b["rc"] == 0 and b["done"] and b["history_steps"][:2] == [0, 200] and b["steps"] > 200
    print(f"{'PASS' if good else 'FAIL'} resume to the end: {b}", flush=True)
    ok &= good
    t0 = time.time()
    r3 = subprocess.run(cmd, capture_output=True, text=True)
    c = {"rc": r3.returncode, "said": r3.stdout.strip()[-200:], "s": round(time.time() - t0, 1)}
    good = c["rc"] == 0 and "nothing to do" in r3.stdout
    print(f"{'PASS' if good else 'FAIL'} a done run exits at once: {c}", flush=True)
    ok &= good
    out["resume"] = {"preempted": a, "resumed": b, "done_again": c,
                     "monitor": [{"step": h["step"], "mean": round(h["mean"], 4),
                                  **{d: round(h["monitor"][d]["mIoU"], 4) for d in DATASETS}} for h in hist]}
    return ok


def determinism(tok_path: Path, out: dict) -> bool:
    tok = T.load(str(tok_path))
    ok = True
    res = {}
    for d in DATASETS:
        i1 = H.tokens(tok, d, "monitor", cache=False)
        i2 = H.tokens(tok, d, "monitor", cache=False)
        y1 = run_fixed(tok.decode, torch.as_tensor(i1[:64]).cuda().long(), tok.batch)
        y2 = run_fixed(tok.decode, torch.as_tensor(i1[:64]).cuda().long(), tok.batch)
        res[d] = {"crops": int(len(i1)), "codes_identical": bool(np.array_equal(i1, i2)),
                  "decoded_identical": bool(torch.equal(y1, y2)), "distinct": int(len(np.unique(i1)))}
        ok &= res[d]["codes_identical"] and res[d]["decoded_identical"]
    print(f"{'PASS' if ok else 'FAIL'} determinism at batch {tok.batch}: {json.dumps(res)}", flush=True)
    out["determinism"] = res
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--levels", default="8,5,5,5")
    ap.add_argument("--expect-gpu", default="A100")
    ap.add_argument("--skip-bench", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = torch.cuda.get_device_name(0)
    res = {"gpu": name, "torch": torch.__version__, "numpy": np.__version__, "size": a.size, "levels": a.levels}
    ok = a.expect_gpu in name
    print(f"{'PASS' if ok else 'FAIL'} card {name} (torch {torch.__version__}, numpy {np.__version__})", flush=True)
    if not a.skip_bench:
        bench(a.size, a.levels, res)
    ok &= resume_test(out_dir / "run", a.size, a.levels, a.expect_gpu, res)
    ok &= determinism(out_dir / "run" / "tok_best.pt", res)
    res["pass"] = bool(ok)
    (out_dir / "smoke.json").write_text(json.dumps(res, indent=1))
    print("SMOKE", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
