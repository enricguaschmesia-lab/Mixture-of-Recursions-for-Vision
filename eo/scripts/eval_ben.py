#!/usr/bin/env python
"""Generate BEN label blocks for one checkpoint, per input condition (Phase 6 Step 5, `.venv`).

    python eo/scripts/eval_ben.py --config eo_ben/b10_full --checkpoint <run>/checkpoint-487 \\
        --role ft_valsel --tag b10_full --out-root /data/enric/reports/ben
    # the extended base (no fine-tune): --checkpoint none
    # the wrong-input control: --wrong-input (inputs from another patch, truth unchanged)

Writes <out-root>/<tag>/<checkpoint>/<role>/<condition>[_wrong].npz with rows, truth, the
greedy set, the Rao-Blackwellized marginals over K sampled blocks, and the first-step slot
mass; plus a metrics JSON beside each. Resumable per (condition, wrong) unit. Metrics against
the class prior and thresholds chosen on val are computed by ben_report.py, not here.

Before any number: `cache_check` compares KV-cached against uncached decoding on real
prompts and refuses above its tolerance.
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
from eo.ben import labels as L  # noqa: E402
from eo.data.ben_dataset import BenTokenDataset, CONDITIONS, load_index  # noqa: E402
from eo.eval import ben as E  # noqa: E402
from eo.finetune.init import build_finetuned  # noqa: E402

CACHE_TOL, derangement = E.CACHE_TOL, E.derangement   # moved to eo/eval/ben.py (2026-10-13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir, or 'none' = the extended base")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--role", default="ft_valsel")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--wrong-input", action="store_true")
    ap.add_argument("--k", type=int, default=8, help="sampled blocks per patch for the marginals")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N patches of a fixed shuffle")
    ap.add_argument("--autocast", choices=["none", "fp16", "bf16"], default="none")
    ap.add_argument("--out-root", type=Path, default=Path("/data/enric/reports/ben"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default="/data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300",
                    help="the extended base (finetune.init_from); used by --checkpoint none")
    a = ap.parse_args()

    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    if "TITAN V" in gpu:
        sys.exit("refusing the TITAN V: it stalls within seconds under load (worklog 2026-10-12)")
    ck = None if a.checkpoint.lower() == "none" else a.checkpoint
    ck_name = "extended_base" if ck is None else Path(ck).name
    out = a.out_root / a.tag / ck_name / a.role
    out.mkdir(parents=True, exist_ok=True)
    ac = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[a.autocast]

    model, cfg = build_finetuned(a.config, ck, device=dev, overrides=[f"finetune.init_from={a.init_from}"])
    idx = load_index()
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse",
                             "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    for cond in [c for c in a.conditions.split(",") if c]:
        name = cond + ("_wrong" if a.wrong_input else "")
        npz = out / f"{name}.npz"
        if npz.exists():
            print(f"[skip] {npz}")
            continue
        ds = BenTokenDataset(role=a.role, condition=cond, index=idx)
        pos = np.arange(len(ds))
        if a.limit:
            pos = np.sort(np.random.default_rng(a.seed).permutation(len(ds))[: a.limit])
        rows = ds.rows[pos]
        inp_rows = rows[derangement(len(rows), a.seed + 1)] if a.wrong_input else rows

        # Cache check on this condition's real prompts, before any number is kept.
        probe = torch.stack([ds.prompt(int(p), cond, input_row=int(r))
                             for p, r in zip(pos[:4], inp_rows[:4])]).to(dev)
        cc = E.cache_check(model, probe, autocast=ac)
        if cc["max_abs_logit_diff"] > CACHE_TOL[a.autocast]:
            sys.exit(f"KV cache disagrees with uncached decoding: {cc}")

        t0 = time.time()
        acc = {k: [] for k in ("greedy", "marginal", "sample_freq", "slot_mass")}
        steps = []
        for b0 in range(0, len(pos), a.batch):
            bp, br = pos[b0:b0 + a.batch], inp_rows[b0:b0 + a.batch]
            prompts = torch.stack([ds.prompt(int(p), cond, input_row=int(r)) for p, r in zip(bp, br)]).to(dev)
            r = E.decode_labels(model, prompts, k_samples=a.k, greedy=True,
                                seed=a.seed * 1_000_003 + b0, autocast=ac)
            for k in acc:
                acc[k].append(r[k])
            steps.append(r["greedy_steps"])
            if (b0 // a.batch) % 50 == 0:
                print(f"  {cond}{' wrong' if a.wrong_input else ''}: {b0 + len(bp)}/{len(pos)} "
                      f"{(b0 + len(bp)) / (time.time() - t0):.1f} patches/s", flush=True)
        res = {k: np.concatenate(v) for k, v in acc.items()}
        truth = np.stack([np.isin(np.arange(L.N_CLASSES), ds.classes(int(rw))) for rw in rows])
        np.savez_compressed(npz, rows=rows, input_rows=inp_rows, truth=truth, **res)
        m = {"condition": cond, "wrong_input": a.wrong_input, "n": int(len(rows)), "k": a.k,
             **{f"greedy_{k}": v for k, v in E.f1_scores(truth, res["greedy"]).items() if k != "per_class_f1"},
             **{k: v for k, v in E.map_scores(truth, res["marginal"]).items() if k != "per_class_ap"},
             "mean_slot_mass": float(res["slot_mass"].mean()),
             "mean_labels_true": float(truth.sum(1).mean()), "mean_labels_greedy": float(res["greedy"].sum(1).mean()),
             "seconds": round(time.time() - t0, 1), "patches_per_s": round(len(rows) / (time.time() - t0), 2),
             "cache_check": cc, "config": a.config, "checkpoint": ck, "gpu": gpu, "autocast": a.autocast,
             "fork_commit": commit, "limit": a.limit}
        (out / f"{name}.json").write_text(json.dumps(m, indent=2) + "\n")
        print(f"{a.tag} {ck_name} {a.role} {name}: micro-F1 {m['greedy_micro_f1']:.4f}  "
              f"micro-mAP {m['micro_map']:.4f}  macro-mAP {m['macro_map']:.4f}  "
              f"slot {m['mean_slot_mass']:.3f}  ({m['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
