#!/usr/bin/env python
"""A2 Step 4.4, beside forgetting: how much of a fine-tune's softmax sits on the map rows at TerraMesh positions (`.venv`).

Plan: docs/plans/a2_segmentation.md, Step 4, "Steps 4.3 and 4.4's design and reading", item 6. The fine-tuned softmax
holds the 80,369 map rows (ids 81,296..161,664), which LoRA's fine-tune trains. At every scored position of the first
--rows of TerraMesh's held-out rows (eval_forgetting.py's rows: --arm-config's held-out split, in its order), the mean
softmax mass on those rows. The share of forgetting that is mass moved onto map rows is at most -log(1 - mass) per token.
fp32, TF32 off.

--decompose (added 2026-11-23, after forgetting read +1.3 to +5.7 nats): also, per target modality, the CE of the label
three ways at the same positions: over the full softmax, renormalised over the base's rows (ids < the first map row), and
renormalised over the label's own codebook slot. The first minus the second is what the mass on the map rows costs; the
second minus the third, the mass on other base rows; the third is what is left of the base's within-slot prediction.

--per-row (added 2026-11-24, A2 Step 4.6; needs --decompose): also each row's sums over its image tokens (token count,
full-softmax CE, base-rows CE, own-slot CE), in row order, so that F = CE after - before is paired by row and
bootstrapped over rows (eo/experiments/segmentation/step46_read.py). The aggregates are unchanged by it.

    python eo/experiments/segmentation/step44_map_mass.py --ben-config eo_seg/r0_s41_lora16 \
        --checkpoint <run>/checkpoint-N --out <json>          (no --checkpoint: the map base)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

MAP_BASE = "/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ben-config", default="eo_seg/r0_s41_lora16")
    ap.add_argument("--init-from", default=MAP_BASE)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--arm-config", default="eo_terramesh/arm_b10ctrxnc_vanilla")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--expect-gpu", default="GTX TITAN X")
    ap.add_argument("--decompose", action="store_true")
    ap.add_argument("--per-row", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.per_row and not a.decompose:
        raise SystemExit("--per-row needs --decompose")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu not in gpu or torch.cuda.device_count() != 1:
        raise SystemExit(f"CUDA sees {gpu!r} ({torch.cuda.device_count()} cards), expected one {a.expect_gpu!r}")
    from eo.data import map_vocab as MV
    from eo.eval import teacher_forced as TF
    from eo.finetune.init import build_finetuned
    from eo.generate import conditional as G
    model, _ = build_finetuned(a.ben_config, a.checkpoint, overrides=[f"finetune.init_from={a.init_from}"])
    V = int(model.config.vocab_size)
    lo = V - MV.N_NEW                                          # the first map-layout row: the base's vocabulary size
    _, arm_cfg = G.build_model(a.arm_config, None, device="cpu")
    ds, loader = TF.eval_loader(arm_cfg, num_workers=0, batch_size=a.batch)
    mass_sum, n, rows, t0 = 0.0, 0, 0, time.time()
    per_row = []
    if a.decompose:
        from eo.data import eo_vocab_nc as VNC
        slots = {m: (i.codebook_offset, i.codebook_offset + i.codebook_size) for m, i in VNC.MODALITIES.items()}
        acc = {m: {"n": 0, "full": 0.0, "base": 0.0, "slot": 0.0} for m in slots}
    rowsums = []                                                       # --per-row: [n, full, base, slot] per row
    with torch.no_grad():
        for b in loader:
            if rows >= a.rows:
                break
            ids, pos, lab = b["input_ids"].cuda(), b["position_ids"].cuda(), b["labels"].cuda()
            logits = model(input_ids=ids, attention_mask=b["attention_mask"].cuda(), position_ids=pos).logits.float()
            p = torch.softmax(logits[:, :-1], -1)[..., lo:].sum(-1)            # predicting token t + 1
            m = lab[:, 1:] != -100
            for r in range(len(ids)):
                per_row.append(float(p[r][m[r]].mean()))
            if a.decompose:
                lg, y = logits[:, :-1], lab[:, 1:]
                lse_all = torch.logsumexp(lg, -1)
                lse_base = torch.logsumexp(lg[..., :lo], -1)
                rs = torch.zeros(len(ids), 4, dtype=torch.float64, device=lg.device)
                for mod, (s0, s1) in slots.items():
                    sel = (y >= s0) & (y < s1)
                    if not sel.any():
                        continue
                    z = lg[sel]                                        # (k, V): positions whose label is in mod's slot
                    zy = z.gather(1, y[sel].unsqueeze(1)).squeeze(1)
                    acc[mod]["n"] += int(sel.sum())
                    acc[mod]["full"] += float((lse_all[sel] - zy).sum())
                    acc[mod]["base"] += float((lse_base[sel] - zy).sum())
                    ce_slot = torch.logsumexp(z[:, s0:s1], -1) - zy
                    acc[mod]["slot"] += float(ce_slot.sum())
                    if a.per_row:
                        r_idx = sel.nonzero()[:, 0]
                        for j, v in enumerate((torch.ones_like(zy), lse_all[sel] - zy, lse_base[sel] - zy, ce_slot)):
                            rs[:, j].index_add_(0, r_idx, v.double())
                    del z
                if a.per_row:
                    rowsums.extend(rs.cpu().tolist())
            mass_sum += float(p[m].sum())
            n += int(m.sum())
            rows += len(ids)
            del logits, p
    doc = {"checkpoint": a.checkpoint or "map base", "ben_config": a.ben_config, "arm_config": a.arm_config,
           "rows": rows, "positions": n, "map_rows": [lo, V - 1], "mean_map_mass": mass_sum / n,
           "max_row_mean": max(per_row), "bound_nats": float(-np.log1p(-mass_sum / n)), "gpu": gpu,
           "precision": "fp32", "seconds": round(time.time() - t0, 1)}
    if a.decompose:
        doc["ce_by_modality"] = {mod: {"n": v["n"], "ce_full": v["full"] / v["n"], "ce_base_rows": v["base"] / v["n"],
                                       "ce_own_slot": v["slot"] / v["n"]} for mod, v in acc.items() if v["n"]}
        tot = {k: sum(v[k] for v in acc.values()) for k in ("n", "full", "base", "slot")}
        doc["ce_image_tokens"] = {"n": tot["n"], "ce_full": tot["full"] / tot["n"], "ce_base_rows": tot["base"] / tot["n"],
                                  "ce_own_slot": tot["slot"] / tot["n"]}
    if a.per_row:
        doc["per_row"] = {"columns": ["n", "ce_full_sum", "ce_base_rows_sum", "ce_own_slot_sum"], "rows": rowsums}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
