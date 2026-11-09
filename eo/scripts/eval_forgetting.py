#!/usr/bin/env python
"""Forgetting, teacher-forced half (PHASE6_PLAN Step 6.1, D6.8). `.venv`, GPU.

TerraMesh's 4,416 held-out rows, built from the PRETRAINING arm's config (so the sequences,
modality orders and token set are exactly those of B10's own evaluation), scored by:

    --model pretrained   the original checkpoint (87,556 ids)          reference
    --model extended     the extended base, before any fine-tune       "before" (plan 2.4)
    --model finetuned    a BEN fine-tune checkpoint                    "after"
    --model temporal     T360-T at the temporal arm's 87,664 ids        the temporal stage (A1)

The extended base, not the original, is "before": its softmax has 21 more terms, so its CE
differs slightly from the original's even though every old logit is identical.

    python eo/scripts/eval_forgetting.py --model extended --ben-config eo_ben/b10_full --tag b10_ext
    python eo/scripts/eval_forgetting.py --model finetuned --ben-config eo_ben/b10_full \\
        --checkpoint <run>/checkpoint-1461 --tag b10_full

Writes <out-root>/<tag>/<checkpoint>/forgetting.json and tokens_ce.npz (per-token CE, so the
before/after difference can be paired by token and bootstrapped over rows).

--precision fp16|bf16|fp32 (2026-11-02) is the autocast of the pass: fp16, the default, as every
result before it (B10's); fp32 for T360, whose own pretraining overflowed fp16. Before and after
must be scored at the same precision: a tag directory refuses a second precision.

THE TEMPORAL STAGE (A1, 2026-11-09; Step 7's item, pulled forward for gate check Y12). `--model
temporal` builds T360-T from `--temporal-config` (default arm_t360t_vanilla) at its vocabulary,
87,664 ids, which build_finetuned cannot (it builds at BEN's 87,577). Without --checkpoint it loads
the temporal arm's extended base through pretrain_init's `continue` hook (extension.json verified,
date layout, strict load, values, tie: eo/train/clevr_init.py), so this is that base's "before";
with one, a T360-T checkpoint is loaded strictly over the same build. The rows, layout and worker
count are still `--arm-config`'s, which must be the single-date arm whose rows are scored (T360's,
eo_terramesh/arm_t360_vanilla): the temporal arm's own eval set is a dict of two micro-batch sets,
and T360's in-training batches of 16 key other modality orders than this script's batches of 4.
So a temporal reading pairs token for token with T360's BEN-base reading at the same arm-config.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(python -m eo.train.gpu titanx) \\
      python eo/scripts/eval_forgetting.py --model temporal --arm-config eo_terramesh/arm_t360_vanilla \\
        --precision fp32 --tag t360t_base

Its default --out-root is /data/enric/reports/temporal/forgetting.
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


def build_temporal(config_name: str, checkpoint, device: str = "cuda"):
    """T360-T at the temporal arm's vocabulary, for eval (module docstring, THE TEMPORAL STAGE).
    Returns (model, cfg, what was loaded)."""
    from eo.finetune.extend import assert_tied
    from eo.finetune.init import strict_load
    from eo.train import clevr_init as CI
    model, cfg = G.build_model(config_name, None, device="cpu")
    if checkpoint is None:
        rep = CI.pretrain_init(model, cfg)                    # verify_init, strict load, values, tie
        loaded = {k: rep[k] for k in ("conversion", "conversion_weights_sha256", "variant", "load", "source")}
    else:
        loaded = {"checkpoint": str(checkpoint), "load": strict_load(model, Path(checkpoint))}
        assert_tied(model)
    model.to(device).eval()
    return model, cfg, loaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pretrained", "extended", "finetuned", "temporal"], required=True)
    ap.add_argument("--arm-config", default="eo_terramesh/arm_b10_vanilla",
                    help="the PRETRAINING arm: its held-out split, token set and worker count")
    ap.add_argument("--pretrained", default="/data/enric/runs/pretrain/phase3/vanilla10_20261002_133848/checkpoint-3300")
    ap.add_argument("--ben-config", default="eo_ben/b10_full")
    ap.add_argument("--init-from", default="/data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300")
    ap.add_argument("--temporal-config", default="eo_terramesh/arm_t360t_vanilla",
                    help="--model temporal: T360-T's arm; without --checkpoint its pretrain_init base is scored")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default /data/enric/reports/ben/forgetting, or .../temporal/forgetting for --model temporal")
    ap.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    a = ap.parse_args()
    if a.out_root is None:
        a.out_root = Path("/data/enric/reports/temporal/forgetting" if a.model == "temporal"
                          else "/data/enric/reports/ben/forgetting")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[a.precision]
    for prev in (a.out_root / a.tag).glob("*/forgetting.json"):          # one precision per tag
        got = json.loads(prev.read_text()).get("precision", "fp16")
        if got != a.precision:
            sys.exit(f"{prev} was scored at {got}; this run scores at {a.precision}. Use another --tag.")

    gpu = torch.cuda.get_device_name(0)
    if "TITAN V" in gpu:
        sys.exit("refusing the TITAN V (stalls under load, worklog 2026-10-12)")
    _, arm_cfg = G.build_model(a.arm_config, None, device="cpu")          # config only
    loaded = None
    if a.model == "temporal":
        if arm_cfg.dataset != "terramesh_multimodal":
            sys.exit(f"--arm-config {a.arm_config} builds {arm_cfg.dataset}: name the single-date arm whose "
                     f"held-out rows are scored (T360's, eo_terramesh/arm_t360_vanilla)")
        model, t_cfg, loaded = build_temporal(a.temporal_config, a.checkpoint)
        want, have = ((c.get("multimodal") or {}).get("token_set", "original") for c in (t_cfg, arm_cfg))
        if want != have:
            sys.exit(f"{a.temporal_config} trains on token set {want!r}, {a.arm_config} scores {have!r}")
        ck = a.checkpoint
    elif a.model == "pretrained":
        model, _ = G.build_model(a.arm_config, a.pretrained)
        ck = a.pretrained
    else:
        ov = [f"finetune.init_from={a.init_from}"]
        ck = None if a.model == "extended" else a.checkpoint
        if a.model == "finetuned" and not ck:
            sys.exit("--model finetuned needs --checkpoint")
        model, _ = build_finetuned(a.ben_config, ck, overrides=ov)
    name = "pretrained" if a.model == "pretrained" else ("extended_base" if ck is None else Path(ck).name)
    if a.model == "temporal" and ck is not None and name == "extended_base":
        sys.exit("a checkpoint directory named extended_base would overwrite the base's reading")
    out = a.out_root / a.tag / name
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    ds, loader = TF.eval_loader(arm_cfg, num_workers=int(arm_cfg.dataloader_num_workers))
    t0 = time.time()
    res = TF.run_pass(model, loader, keep_tokens=True, dtype=dtype)
    tok = res.pop("tokens")
    np.savez_compressed(out / "tokens_ce.npz", ce=tok["ce"], label_mod=tok["label_mod"],
                        rows=np.asarray(ds.rows))
    doc = {"model": a.model, "checkpoint": ck,
           "ben_config": a.ben_config if a.model in ("extended", "finetuned") else None,
           "temporal_config": a.temporal_config if a.model == "temporal" else None, "loaded": loaded,
           "arm_config": a.arm_config, "token_set": (arm_cfg.get("multimodal") or {}).get("token_set", "original"),
           "n_rows": len(ds), "vocab": int(model.config.vocab_size), "gpu": gpu, "precision": a.precision,
           "seconds": round(time.time() - t0, 1),
           "fork_commit": subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                                          "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           **res}
    (out / "forgetting.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{a.tag}/{name}: CE(all) {res['ce_all_body_tokens']:.4f}  " +
          "  ".join(f"{n} {v['ce']:.4f}" for n, v in res["per_modality"].items()))


if __name__ == "__main__":
    main()
