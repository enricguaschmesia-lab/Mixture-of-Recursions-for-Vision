#!/usr/bin/env python
"""Forgetting, teacher-forced half (PHASE6_PLAN Step 6.1, D6.8). `.venv`, GPU.

TerraMesh's 4,416 held-out rows, built from the PRETRAINING arm's config (so the sequences,
modality orders and token set are exactly those of B10's own evaluation), scored by:

    --model pretrained   the original checkpoint (87,556 ids)          reference
    --model extended     the extended base, before any fine-tune       "before" (plan 2.4)
    --model finetuned    a BEN fine-tune checkpoint                    "after"

The extended base, not the original, is "before": its softmax has 21 more terms, so its CE
differs slightly from the original's even though every old logit is identical.

    python eo/scripts/eval_forgetting.py --model extended --ben-config eo_ben/b10_full --tag b10_ext
    python eo/scripts/eval_forgetting.py --model finetuned --ben-config eo_ben/b10_full \\
        --checkpoint <run>/checkpoint-1461 --tag b10_full

Writes <out-root>/<tag>/<checkpoint>/forgetting.json and tokens_ce.npz (per-token CE, so the
before/after difference can be paired by token and bootstrapped over rows).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.eval import teacher_forced as TF  # noqa: E402
from eo.finetune.init import build_finetuned  # noqa: E402
from eo.generate import conditional as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pretrained", "extended", "finetuned"], required=True)
    ap.add_argument("--arm-config", default="eo_terramesh/arm_b10_vanilla",
                    help="the PRETRAINING arm: its held-out split, token set and worker count")
    ap.add_argument("--pretrained", default="/data/enric/runs/pretrain/phase3/vanilla10_20261002_133848/checkpoint-3300")
    ap.add_argument("--ben-config", default="eo_ben/b10_full")
    ap.add_argument("--init-from", default="/data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("/data/enric/reports/ben/forgetting"))
    a = ap.parse_args()

    gpu = torch.cuda.get_device_name(0)
    if "TITAN V" in gpu:
        sys.exit("refusing the TITAN V (stalls under load, worklog 2026-10-12)")
    _, arm_cfg = G.build_model(a.arm_config, None, device="cpu")          # config only
    if a.model == "pretrained":
        model, _ = G.build_model(a.arm_config, a.pretrained)
        ck = a.pretrained
    else:
        ov = [f"finetune.init_from={a.init_from}"]
        ck = None if a.model == "extended" else a.checkpoint
        if a.model == "finetuned" and not ck:
            sys.exit("--model finetuned needs --checkpoint")
        model, _ = build_finetuned(a.ben_config, ck, overrides=ov)
    name = "pretrained" if a.model == "pretrained" else ("extended_base" if ck is None else Path(ck).name)
    out = a.out_root / a.tag / name
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    ds, loader = TF.eval_loader(arm_cfg, num_workers=int(arm_cfg.dataloader_num_workers))
    t0 = time.time()
    res = TF.run_pass(model, loader, keep_tokens=True)
    tok = res.pop("tokens")
    np.savez_compressed(out / "tokens_ce.npz", ce=tok["ce"], label_mod=tok["label_mod"],
                        rows=np.asarray(ds.rows))
    doc = {"model": a.model, "checkpoint": ck, "ben_config": a.ben_config if a.model != "pretrained" else None,
           "arm_config": a.arm_config, "token_set": (arm_cfg.get("multimodal") or {}).get("token_set", "original"),
           "n_rows": len(ds), "vocab": int(model.config.vocab_size), "gpu": gpu,
           "seconds": round(time.time() - t0, 1),
           "fork_commit": subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                                          "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           **res}
    (out / "forgetting.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{a.tag}/{name}: CE(all) {res['ce_all_body_tokens']:.4f}  " +
          "  ".join(f"{n} {v['ce']:.4f}" for n, v in res["per_modality"].items()))


if __name__ == "__main__":
    main()
