#!/usr/bin/env python
"""A2's generated-map evaluation, the CLI (A2 Step 3.3; eo/eval/seg.py). Two subcommands, two environments:

  generate (`.venv`, GTX Titan X): a model's maps for one task's val views under a variant, K samples or greedy
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<titanx> python eo/scripts/eval_seg.py generate \\
          --config eo_seg/r0_pastis1_aligned --checkpoint <run>/checkpoint-900 \\
          --variant full --k 8 --temperature 1.0 --out /data/enric/reports/a2_seg/<...>/full_k8
      `--checkpoint base` reads the map-extended base itself (finetune.init_from, its identity verified).

  score (`a2seg`, one card; the TITAN V only under the watchdog unit): decode, vote, GEO-Bench-2's mIoU
      python eo/scripts/eval_seg.py score --codes <dir from generate> [--out score.json]
      python eo/scripts/eval_seg.py score --ceiling --task pastis --arm aligned --frames p1 --out ceiling.json
      `--ceiling` scores the stored TRUE map tokens of the val views through the same path (the bound).

  cachecheck (`.venv`, one card): gate Z8 on a fine-tune's checkpoint at the precision it generates in (A2 Step
      4.0's protocol, item 4: bf16 on an A100). A cached decode equals full recomputation after --steps greedy
      tokens within eo/eval/seg.CACHE_ULPS rounding steps of that precision (X15's rule, set at fp16/bf16), on
      the first --n val prompts of the config's task and arm; every code lands in the arm's slot. Its control,
      the cache fed positions one off, must land past the same rule. Exit 0 only if the check passes AND the
      control fires.
      python eo/scripts/eval_seg.py cachecheck --config eo_seg/r0_pastis1_long --checkpoint <run>/checkpoint-N \\
          --precision bf16 --expect-gpu A100 --out <dir>/z8_bf16.json

All three refuse a test split. The generate record names the config, checkpoint, variant, decoding and git commit;
score reads it and writes beside it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def cmd_generate(a) -> None:
    import torch
    from eo.data import seg_dataset as SD
    from eo.eval import seg as SEG
    from eo.finetune.init import build_finetuned

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    ov = list(a.override)
    if a.init_from:
        ov.append(f"finetune.init_from={a.init_from}")
    model, cfg = build_finetuned(a.config, None if a.checkpoint == "base" else a.checkpoint, device=dev,
                                 overrides=ov)
    model.float() if a.precision == "fp32" else None
    autocast = {"fp32": None, "bf16": torch.bfloat16}[a.precision]
    spec = SD.spec_from_cfg(cfg, "eval")
    spec.variant = a.variant
    ds = SD.SegDataset(spec, max_length=None, seed=int(cfg.get("seed", 42)))
    views = ds.views.view.values if a.limit is None else ds.views.view.values[: a.limit]
    t0 = time.time()
    out = SEG.generate_views(model, ds, views, a.variant, a.k, a.greedy, a.temperature, a.seed, a.batch, dev,
                             autocast=autocast, log_every=10)
    rec = {"config": a.config, "checkpoint": a.checkpoint, "init_from": str(cfg.finetune.init_from),
           "overrides": ov, "task": spec.task, "arm": spec.arm, "frames": spec.frames, "split": spec.split,
           "map_order": spec.map_order, "map_shift": spec.map_shift, "codes_order": "raster",
           "variant": a.variant, "k": a.k, "greedy": a.greedy, "temperature": a.temperature, "seed": a.seed,
           "precision": a.precision, "views": [int(v) for v in views], "seconds": time.time() - t0,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "fork_commit": _commit(),
           "slot_mass_mean": float(out["slot_mass"].mean())}
    SEG.write_codes(Path(a.out), out, rec)
    print(f"wrote {a.out}: {len(views)} views x K {a.k}, slot mass {rec['slot_mass_mean']:.4f}, "
          f"{rec['seconds']:.0f} s", flush=True)


def cmd_score(a) -> None:
    import torch
    from eo.data import seg_dataset as SD
    from eo.eval import seg as SEG

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    if a.ceiling:
        ds = SD.SegDataset(SD.SegSpec(a.task, a.arm, a.frames, "val"))
        views = ds.views
        codes = np.asarray(ds.maps[views.view.values])[:, None]
        rec = {"ceiling": True, "task": a.task, "arm": a.arm, "frames": a.frames}
    else:
        d = Path(a.codes)
        rec = json.loads((d / "record.json").read_text())
        codes = np.load(d / "codes.npy")
        ds = SD.SegDataset(SD.SegSpec(rec["task"], rec["arm"], rec["frames"], rec["split"]))
        views = ds.views.set_index("view").loc[rec["views"]].reset_index()
    t0 = time.time()
    res = SEG.score(rec["task"], rec["arm"], codes, views, device=dev)
    res.update({"source": rec, "seconds": time.time() - t0,
                "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "fork_commit": _commit()})
    out = Path(a.out) if a.out else Path(a.codes) / "score.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    r = res["result"]
    print(f"{rec['task']} {rec['arm']} {'ceiling' if a.ceiling else rec['variant']}: mIoU {r['mIoU']:.4f}, pixel "
          f"{r['pixel_acc']:.4f}, predicted-class entropy {res['pred_entropy_nats']:.3f} of "
          f"{res['pred_entropy_max_nats']:.3f}, top share {res['pred_top_share']:.3f} -> {out}", flush=True)


def cmd_cachecheck(a) -> int:
    import torch
    from eo.data import seg_dataset as SD
    from eo.eval import seg as SEG
    from eo.finetune.init import build_finetuned

    if not torch.cuda.is_available():
        raise SystemExit("cachecheck reads the card the maps are generated on: no CUDA device")
    if a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    model, cfg = build_finetuned(a.config, None if a.checkpoint == "base" else a.checkpoint, device="cuda",
                                 overrides=list(a.override))
    model.float()                                        # fp32 weights; bf16 runs under autocast, as training did
    autocast = {"fp32": None, "bf16": torch.bfloat16}[a.precision]
    spec = SD.spec_from_cfg(cfg, "eval")
    ds = SD.SegDataset(spec, max_length=None, seed=int(cfg.get("seed", 42)))
    views = [int(v) for v in ds.views.view.values[: a.n]]
    pr = [ds.prompt(v) for v in views]
    ids = torch.stack([p[0] for p in pr]).cuda()
    pos = torch.stack([p[1] for p in pr]).cuda()
    slot = ds.lay.slot(spec.arm)
    sp = ds.step_positions().cuda()                      # the layout's code positions (seg_dataset MAP LAYOUTS)
    t0 = time.time()
    r = SEG.cache_check(model, ids, pos, slot, a.steps, autocast=autocast, step_positions=sp)
    c = SEG.cache_check(model, ids, pos, slot, a.steps, autocast=autocast, cache_offset=1, step_positions=sp)
    passed = bool(r["max_ulp_diff"] <= SEG.CACHE_ULPS and r["codes_in_slot"])
    fires = bool(c["max_ulp_diff"] > SEG.CACHE_ULPS)
    rec = {"config": a.config, "checkpoint": a.checkpoint, "overrides": list(a.override), "task": spec.task,
           "arm": spec.arm, "frames": spec.frames, "split": spec.split, "views": views, "precision": a.precision,
           "map_order": spec.map_order, "map_shift": spec.map_shift, "step_positions_head": sp[:8].tolist(),
           "rule": f"cached = recomputed within {SEG.CACHE_ULPS} rounding steps of {a.precision}",
           "check": r, "control_positions_one_off": c, "pass": passed, "control_fires": fires,
           "seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0), "fork_commit": _commit()}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"Z8 {spec.task} {spec.arm} {a.precision} on {rec['gpu']}: max |dlogit| {r['max_abs_logit_diff']:.3e} = "
          f"{r['max_ulp_diff']:.2f} rounding steps (rule {SEG.CACHE_ULPS}; |logit| <= {r['max_abs_logit']:.1f}; floor "
          f"{r['floor']:.2e}); codes in slot {r['codes_in_slot']} -> {'PASS' if passed else 'FAIL'}; control "
          f"{c['max_ulp_diff']:.1f} steps -> {'fires' if fires else 'DOES NOT FIRE'} -> {out}", flush=True)
    return 0 if (passed and fires) else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--config", required=True)
    g.add_argument("--checkpoint", required=True, help="a fine-tune's checkpoint dir, or 'base'")
    g.add_argument("--init-from", default=None)
    g.add_argument("--override", action="append", default=[])
    g.add_argument("--variant", default="full")
    g.add_argument("--k", type=int, default=1)
    g.add_argument("--greedy", action="store_true")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch", type=int, default=16)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    g.add_argument("--expect-gpu", default=None)
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--codes", default=None)
    s.add_argument("--ceiling", action="store_true")
    s.add_argument("--task", default=None)
    s.add_argument("--arm", default=None)
    s.add_argument("--frames", default="p1")
    s.add_argument("--expect-gpu", default=None)
    s.add_argument("--out", default=None)
    c = sub.add_parser("cachecheck")
    c.add_argument("--config", required=True)
    c.add_argument("--checkpoint", required=True, help="a fine-tune's checkpoint dir, or 'base'")
    c.add_argument("--override", action="append", default=[])
    c.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    c.add_argument("--n", type=int, default=8)
    c.add_argument("--steps", type=int, default=6)
    c.add_argument("--expect-gpu", default=None)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "generate":
        if a.greedy and a.k != 1:
            raise SystemExit("greedy is one sample: --k 1")
        cmd_generate(a)
    elif a.cmd == "cachecheck":
        sys.exit(cmd_cachecheck(a))
    else:
        if not a.ceiling and not a.codes:
            raise SystemExit("score needs --codes or --ceiling")
        cmd_score(a)


if __name__ == "__main__":
    main()
