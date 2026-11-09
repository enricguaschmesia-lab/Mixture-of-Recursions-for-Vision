#!/usr/bin/env python
"""A1 Step 8: T360-T on one A100 at the temporal stage's micro-batch sizes (`mor-eo` on RCP).

For each setting (dates per sequence x locations per micro-batch) it measures peak memory, seconds per
micro-batch by stream, the optimizer step, and how long the main process waits on data, in the
Trainer's own pattern: a step first fetches ALL its micro-batches (Trainer.get_batch_samples, which
counts num_items_in_batch over the step), then runs them. With 4 workers and torch's prefetch of 2
per worker, only 8 micro-batches can be ready when a step of 30 begins.

Everything is the run's own code path:
  - the model: T360-T built as pretrain.py builds it (eo.generate.conditional.build_model), loaded
    from the extended base through pretrain_init's `continue` hook;
  - the step: accelerate's Accelerator(mixed_precision="bf16"), which is what HF's Trainer prepares
    (autocast around forward, fp32 outputs), the stock loss with num_items_in_batch, then the per-
    modality CE under no_grad exactly as EOTrainer computes it (PerModalityLossMixin._per_modality_ce),
    backward, clipping at max_grad_norm, AdamW with the arm's betas and weight decay;
  - the data: the arm's own training set (load_dataset_from_config) on train's root, with one stage
    overridden per setting, read as HF reads it (SequentialSampler, batch 1, unwrap_microbatch,
    dataloader_num_workers, pinned memory).

One GPU only: DDP adds its gradient buckets (about one fp32 copy of the gradients, ~1.6 GB) on top.

    python eo/experiments/temporal/a100_bench.py --out OUT.json [--settings 2x10 3x7 4x6 4x7 4x8]
           [--steps 4] [--accum 30] [--split train]

`--test` checks the logic on CPU (CUDA_VISIBLE_DEVICES=): two layers, no base, 2 scenes a micro-batch.

Settings run in the order given, smallest first, so an out-of-memory error at the end leaves the
others measured. Step 1 of each setting is warm-up and is not counted.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from util.env import load_dotenv  # noqa: E402

load_dotenv()        # HF_HOME and HF_HUB_OFFLINE, as pretrain.py reads them; the shell's values win (RCP: rcp_job.sh)

ARM = "eo_terramesh/arm_t360t_vanilla"
GIB = 2 ** 30
CUDA = torch.cuda.is_available()


def sync():
    if CUDA:
        torch.cuda.synchronize()


def reset_peak():
    if CUDA:
        torch.cuda.reset_peak_memory_stats()


def peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / GIB if CUDA else 0.0


def compose(overrides):
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return hydra_compose(config_name=ARM, overrides=list(overrides))


def stage_override(dates: int, locations: int) -> str:
    return f"temporal.stages=[{{dates:{dates},locations:{locations},epochs:null}}]"


def smi() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:                                     # noqa: BLE001
        return f"nvidia-smi failed: {exc!r}"


def build_cost(ds, n: int = 8) -> dict:
    """Seconds to build one micro-batch of each stream in this process, cold then warm (no workers)."""
    a = ds.assignment(0)
    out = {}
    for stream, pick in (("temporal", (a >= 0).nonzero()[0]), ("single", (a < 0).nonzero()[0])):
        ts = []
        for i in pick[-n:]:                      # from the epoch's end: indices the timed loop does not read
            t0 = time.perf_counter()
            ds[int(i)]
            ts.append(time.perf_counter() - t0)
        out[stream] = {"seconds": [round(t, 4) for t in ts], "median": round(statistics.median(ts), 4)}
    return out


def run_setting(dates, locations, args, model, optimizer, acc, stub, base_overrides):
    from lm_dataset.load_dataset import load_dataset_from_config
    from util.trainer_pt import PerModalityLossMixin

    cfg = compose(base_overrides + [stage_override(dates, locations)])
    t0 = time.time()
    ds = load_dataset_from_config(cfg)
    rec = {"dates": dates, "locations": locations, "build_set_s": round(time.time() - t0, 1),
           "token_share": ds.token_share(0), "build_cost": build_cost(ds)}
    loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=torch.utils.data.SequentialSampler(ds),
                                         num_workers=int(cfg.dataloader_num_workers), collate_fn=ds.collate_fn,
                                         pin_memory=True)
    streams = ["temporal" if a >= 0 else "single" for a in ds.assignment(0)]
    it = iter(loader)
    steps, pos, oom = [], 0, None
    reset_peak()
    for s in range(args.steps):
        t_step = time.perf_counter()
        batch = []
        for _ in range(args.accum):                       # get_batch_samples: the whole step first
            batch.append(next(it))
        wait = time.perf_counter() - t_step
        n_items = sum(int(mb["labels"].ne(-100).sum()) for mb in batch)
        n_items_t = torch.tensor(n_items, device=acc.device)
        mb_times, kinds, shapes, peaks = [], [], [], []
        try:
            for mb in batch:
                kind = streams[pos]
                pos += 1
                sync()
                t0 = time.perf_counter()
                reset_peak()
                inputs = {k: v.to(acc.device, non_blocking=True) for k, v in mb.items()}
                modality_ids = inputs.pop("modality_ids")
                labels = inputs["labels"]
                outputs = model(**inputs, num_items_in_batch=n_items_t)
                loss = outputs["loss"]
                PerModalityLossMixin._per_modality_ce(stub, outputs, labels, modality_ids)
                del outputs
                acc.backward(loss)
                sync()
                mb_times.append(time.perf_counter() - t0)
                peaks.append(peak_gib())
                kinds.append(kind)
                shapes.append(tuple(mb["input_ids"].shape))
            sync()
            t0 = time.perf_counter()
            reset_peak()
            acc.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sync()
            opt = time.perf_counter() - t0
            opt_peak = peak_gib()
        except torch.cuda.OutOfMemoryError as exc:
            oom = {"step": s + 1, "micro_batch": len(mb_times) + 1, "stream": streams[pos - 1],
                   "shape": tuple(batch[len(mb_times)]["input_ids"].shape),
                   "message": str(exc).splitlines()[0][:300], "smi": smi()}
            break
        steps.append({"step": s + 1, "wait_s": round(wait, 3), "wall_s": round(time.perf_counter() - t_step, 3),
                      "opt_s": round(opt, 3), "opt_peak_gib": round(opt_peak, 2), "n_items": n_items,
                      "micro_batches": [{"stream": k, "shape": sh, "s": round(t, 4), "peak_gib": round(p, 2)}
                                        for k, sh, t, p in zip(kinds, shapes, mb_times, peaks)]})
        print(f"  {dates}x{locations} step {s + 1}: wait {wait:.2f} s, wall {steps[-1]['wall_s']:.1f} s, "
              f"peak {max(peaks):.1f} GiB, optimizer {opt:.2f} s", flush=True)
    del it, loader
    rec["oom"] = oom
    rec["steps"] = steps
    counted = steps[1:]
    if counted:
        by = {k: [m["s"] for st in counted for m in st["micro_batches"] if m["stream"] == k] for k in ("temporal", "single")}
        pk = {k: [m["peak_gib"] for st in steps for m in st["micro_batches"] if m["stream"] == k] for k in by}
        rec["summary"] = {
            "temporal_mb_s": round(statistics.median(by["temporal"]), 4) if by["temporal"] else None,
            "single_mb_s": round(statistics.median(by["single"]), 4) if by["single"] else None,
            "n_temporal": len(by["temporal"]), "n_single": len(by["single"]),
            "peak_temporal_gib": round(max(pk["temporal"]), 2) if pk["temporal"] else None,
            "peak_single_gib": round(max(pk["single"]), 2) if pk["single"] else None,
            "peak_optimizer_gib": round(max(st["opt_peak_gib"] for st in steps), 2),
            "wait_s_per_step": [st["wait_s"] for st in counted],
            "wall_s_per_step": [st["wall_s"] for st in counted],
            "wait_share": round(sum(st["wait_s"] for st in counted) / sum(st["wall_s"] for st in counted), 4),
            "opt_s": round(statistics.median(st["opt_s"] for st in counted), 3),
            "max_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2) if CUDA else None,
            "smi": smi()}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", default=["2x10", "3x7", "4x6", "4x7", "4x8"])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--accum", type=int, default=30, help="micro-batches per step on one rank (the pilot's 30)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--lr", type=float, default=1e-5, help="any: only the timing is read")
    ap.add_argument("--test", action="store_true", help="logic only, on CPU: 2 layers, no base, 2 scenes a micro-batch")
    args = ap.parse_args()

    from accelerate import Accelerator

    from eo.generate import conditional as G
    from eo.train import clevr_init as CI
    from lm_dataset.modality_registry import get_id_to_modality

    assert CUDA or args.test, "a GPU is needed (or --test, on CPU)"
    base_overrides = [f"multimodal.split={args.split}"]
    if args.test:
        base_overrides += ["model_config.num_hidden_layers=2", "temporal.scenes_per_microbatch=2"]
    t0 = time.time()
    model, cfg = G.build_model(ARM, None, device="cpu", overrides=base_overrides)
    init = {} if args.test else CI.pretrain_init(model, cfg)
    load_s = time.time() - t0
    args.max_grad_norm = float(cfg.max_grad_norm)
    # bf16 on a CPU without native bf16 is emulated and very slow: --test on CPU runs fp32 (logic only).
    acc = Accelerator(mixed_precision="bf16" if CUDA else "no", cpu=not CUDA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
                                  weight_decay=float(cfg.weight_decay))
    model.train()
    model, optimizer = acc.prepare(model, optimizer)

    class Stub:
        id_to_modality = get_id_to_modality(cfg)

    doc = {"script": "eo/experiments/temporal/a100_bench.py", "argv": sys.argv[1:],
           "gpu": torch.cuda.get_device_name(0) if CUDA else "cpu",
           "torch": torch.__version__, "host": os.uname().nodename, "attn": cfg.get("attn_implementation"),
           "accum": args.accum, "steps": args.steps, "split": args.split,
           "init": {k: init.get(k) for k in ("conversion", "variant", "conversion_weights_sha256")},
           "model_load_s": round(load_s, 1), "parameters": sum(p.numel() for p in model.parameters()),
           "train_root": os.environ.get("TERRAMESH_TRAIN_TOK_ROOT"), "settings": []}
    print(f"T360-T on {doc['gpu']}: {doc['parameters']:,} parameters, loaded in {load_s:.0f} s; "
          f"{doc['attn']}; accum {args.accum}", flush=True)
    for spec in args.settings:
        d, l = (int(x) for x in spec.split("x"))
        print(f"setting {d} dates x {l} locations", flush=True)
        rec = run_setting(d, l, args, model, optimizer, acc, Stub(), base_overrides)
        doc["settings"].append(rec)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=1))
        s = rec.get("summary") or {}
        print(f"=> {d}x{l}: oom {bool(rec['oom'])}, temporal {s.get('temporal_mb_s')} s, single {s.get('single_mb_s')} s, "
              f"peak {s.get('peak_temporal_gib')} / {s.get('peak_single_gib')} GiB, reserved {s.get('max_reserved_gib')}, "
              f"wait share {s.get('wait_share')}", flush=True)
        if rec["oom"]:
            print(f"   OOM: {rec['oom']}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            if CUDA:
                torch.cuda.empty_cache()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
