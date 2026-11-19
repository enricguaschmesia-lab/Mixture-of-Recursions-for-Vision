"""Agent B's smoke test on one RCP A100 (A2 Step 1): speed, a short run, its resume, and the read's code path.

    python -X faulthandler -u -m eo.experiments.segmentation.step1_B_rcp_smoke --out-root <$L/a2seg/out/b> \
        [--bench 224:32,224:64,224:128,224:64:cl,256:32,256:64]

  1. bench: the training step as step1_B_train runs it (bf16 autocast, fp32 loss on the native grid, clip,
     AdamW, EMA), data drawn on the GPU, at a few (size, batch, channels-last) settings: it/s, img/s, peak
     memory. Speed only; nothing is kept.
  2. a 300-step run (`step1_B_train --precision bf16 --data-gpu`, monitor read at 300), then the same command
     with --max-steps 400: it must resume from ckpt_last.pt at step 300 and end at 400.
  3. the read's code path on the smoke's best.pt (`step1_B_read --split monitor --read --controls`): never
     `val` in a smoke.
Each part prints PASS or FAIL; the last line is SMOKE PASS or SMOKE FAIL.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from eo.experiments.segmentation import step1_B_tok as T
from eo.experiments.segmentation import step1_B_train as TR
from eo.maps import palette as P
from eo.maps import sets


def bench(data: TR.Data, size: int, batch: int, cl: bool, steps: int = 30) -> dict:
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    model = T.MapTok(dict(T.DEFAULT, input_size=size)).cuda()
    if cl:
        model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_p = round(batch * 12 / 32)
    n_c = round(batch * 10 / 32)
    data.comp = {"pastis": n_p, "cloudsen12": n_c, "lulc": batch - n_p - n_c}
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for s in range(8 + steps):
        if s == 8:
            torch.cuda.synchronize()
            t0 = time.time()
        maps = data.draw()
        xs = []
        for d in sets.DATASETS:
            t = P.TASKS[d]
            xs.append(P.render(maps[d], P.shuffled_assignment(len(maps[d]), t.n_classes, t.ignore,
                                                              generator=data.gc), size))
        x = torch.cat(xs)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, idx, cidx = model(x, torch.bfloat16)
        o, loss = 0, 0.0
        for d in sets.DATASETS:
            n = len(maps[d])
            loss = loss + n / batch * TR.native_nll(logits[o:o + n].float(), cidx[o:o + n], maps[d],
                                                    P.TASKS[d].ignore, 0.0)
            o += n
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        TR.ema_update(ema, model, 0.999)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"size": size, "batch": batch, "channels_last": cl, "it_s": steps / dt, "img_s": steps * batch / dt,
            "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "loss": float(loss)}


def sh(cmd: list[str]) -> tuple[int, str]:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout[-6000:], flush=True)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bench", default="224:32,224:64,224:128,224:64:cl,256:32,256:64")
    ap.add_argument("--run", default="smoke_a100")
    a = ap.parse_args()
    ok = True
    gpu = torch.cuda.get_device_name(0)
    print(f"torch {torch.__version__}, card {gpu}", flush=True)
    ok &= "A100" in gpu
    t0 = time.time()
    data = TR.Data({"pastis": 12, "cloudsen12": 10, "lulc": 10}, 0, "cuda", 0.5, on_gpu=True)
    print(f"fit pools on the GPU in {time.time() - t0:.0f} s: pastis {tuple(data.pastis.shape)}, cloudsen12 "
          f"{tuple(data.cs_gpu.shape)}, lulc {tuple(data.lu_gpu.shape)}; "
          f"{torch.cuda.memory_allocated() / 2 ** 30:.2f} GB", flush=True)
    for spec in a.bench.split(","):
        f = spec.split(":")
        r = bench(data, int(f[0]), int(f[1]), len(f) > 2 and f[2] == "cl")
        print("BENCH", json.dumps(r), flush=True)
        torch.cuda.empty_cache()
    del data
    torch.cuda.empty_cache()

    py = [sys.executable, "-X", "faulthandler", "-u", "-m"]
    train = py + ["eo.experiments.segmentation.step1_B_train", "--run", a.run, "--out-root", a.out_root,
                  "--precision", "bf16", "--data-gpu", "--expect-gpu", "A100", "--warmup", "100",
                  "--eval-every", "300", "--pastis-mosaic", "0.5"]
    run_dir = Path(a.out_root) / a.run
    rc, txt = sh(train + ["--max-steps", "300"])
    st = json.loads((run_dir / "stopped.json").read_text()) if (run_dir / "stopped.json").exists() else {}
    good = rc == 0 and st.get("steps") == 300 and "EVAL step 300" in txt
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} 300-step run (rc {rc}, steps {st.get('steps')})", flush=True)
    rc, txt = sh(train + ["--max-steps", "400"])
    st = json.loads((run_dir / "stopped.json").read_text()) if (run_dir / "stopped.json").exists() else {}
    good = rc == 0 and "resumed from" in txt and "at step 300" in txt and st.get("steps") == 400
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} resume 300 -> 400 (rc {rc}, steps {st.get('steps')})", flush=True)

    rc, txt = sh(py + ["eo.experiments.segmentation.step1_B_read", "--ckpt", str(run_dir / "best.pt"),
                       "--out", str(run_dir / "read_monitor_smoke.json"), "--split", "monitor", "--read",
                       "--controls", "--expect-gpu", "A100"])
    res = json.loads((run_dir / "read_monitor_smoke.json").read_text()) if rc == 0 else {}
    det = res.get("controls", {}).get("determinism", {})
    good = rc == 0 and bool(det) and all(v["identical_twice"] and not v["control_one_map_changed_identical"]
                                         for v in det.values())
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} read code path on monitor with controls (rc {rc}); determinism "
          f"{ {d: (v['identical_twice'], v['control_tokens_differing']) for d, v in det.items()} }", flush=True)
    print("SMOKE", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
