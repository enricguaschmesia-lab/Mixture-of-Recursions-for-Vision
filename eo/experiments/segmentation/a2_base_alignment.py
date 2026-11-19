#!/usr/bin/env python
"""A2's base: does B10cL-tr-x10-NC align a target block to its source by position, as T360 does? (`.venv`, GPU, fp32)

A2 moves from T360 to B10cL-tr-x10-NC's temporal-layout extension (2026-11-19, Enric). Decision 7 and the
grid-aligned map arm rest on T360's exact cross-block alignment at 198-token offsets (A1 Step 0, findings 4
and 14), never measured on this model. This reads it on both models, same rows, same code:

  one source block, S2L2A, then the target block of the same scene (LULC, the nearest thing to a map, and
  NDVI); the target's position_ids moved by d = 0..21 and 99 (t360_context.Run.target_ce), paired per row.
  References: the target alone (no source) and a source from another row (wrong source).
  Read: the gain the source gives (none - d0), and the rise at d as a share of that gain; A1's reading is the
  worst share over d >= 3.

Rows: 256 of eval_rows_v1 (seeded) where S2L2A, NDVI and LULC are present in the cleaned set.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python -X faulthandler -u \\
        eo/experiments/segmentation/a2_base_alignment.py --model b10cl --expect-gpu "GTX TITAN X"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "experiments" / "temporal"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import t360_context as TC  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402

MODELS = {
    "t360": ("eo_terramesh/arm_t360_vanilla", "/data/enric/runs/pretrain/phase3/vanilla360t_20261026_040535/checkpoint-17441"),
    "b10cl": ("eo_terramesh/arm_b10ctrxnc_vanilla", "/data/enric/runs/pretrain/phase3/vanilla10cltrxnc_20261114_185037/checkpoint-33000"),
}
OUT = Path("/data/enric/reports/a2_seg/step3/base_alignment")
N, SEED = 256, 20261119
DELTAS = tuple(range(22)) + (99,)
TARGETS = ("LULC", "NDVI")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--expect-gpu", default="TITAN X")
    a = ap.parse_args()
    from eo.generate import conditional as G
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu not in gpu or torch.cuda.device_count() != 1:
        raise SystemExit(f"CUDA sees {gpu!r} ({torch.cuda.device_count()} cards), expected one {a.expect_gpu!r}")
    arm, ck = MODELS[a.model]
    model, cfg = G.build_model(arm, ck)
    if (cfg.get("multimodal") or {}).get("token_set") != "clean":
        raise SystemExit("both models are read on the cleaned tokens")
    assert next(model.parameters()).dtype == torch.float32
    ds, _ = TF.eval_loader(cfg, num_workers=0)
    held = json.loads((REPO / "eo/data/eval_rows_v1.json").read_text())["eval_rows"]
    pres = {m: np.load(Path(ds.root_dir) / f"{m}_tok224" / "present.npy") for m in ("S2L2A",) + TARGETS}
    pool = np.array([r for r in held if all(pres[m][r] for m in pres)])
    rows = np.sort(np.random.default_rng(SEED).permutation(pool)[:N])
    wrong = np.roll(rows, 1)
    out = OUT / a.model
    run = TC.Run(model, ds, "cuda", out)
    res = {"model": a.model, "checkpoint": ck, "arm": arm, "gpu": gpu, "precision": "fp32", "n": int(len(rows)),
           "rows_sha": __import__("hashlib").sha256(rows.tobytes()).hexdigest()[:16], "targets": {}}
    for m in TARGETS:
        none, _ = run.unit(f"{m}_none", lambda: run.run_seqs([run.seq([], (m, r)) for r in rows]))
        wr, _ = run.unit(f"{m}_wrong", lambda: run.run_seqs([run.seq([("S2L2A", w)], (m, r)) for r, w in zip(rows, wrong)]))
        seqs = [run.seq([("S2L2A", r)], (m, r)) for r in rows]
        ce = {d: run.unit(f"{m}_d{d}", lambda d=d: run.run_seqs(seqs, delta=d))[0] for d in DELTAS}
        gain = float(none.mean() - ce[0].mean())
        share = {d: float((ce[d] - ce[0]).mean() / gain) for d in DELTAS}
        res["targets"][m] = {"ce_none": float(none.mean()), "ce_wrong_source": float(wr.mean()),
                             "ce": {d: float(ce[d].mean()) for d in DELTAS}, "gain": gain, "share": share,
                             "worst_share_d3_21": max(share[d] for d in range(3, 22)), "share_d1": share[1],
                             "wrong_source_share_of_gain_kept": float((none.mean() - wr.mean()) / gain)}
        t = res["targets"][m]
        print(f"{a.model} {m}: none {t['ce_none']:.4f} wrong {t['ce_wrong_source']:.4f} d0 {t['ce'][0]:.4f} "
              f"gain {gain:.4f} | share d1 {share[1]:.2f}, worst d3-21 {t['worst_share_d3_21']:.2f}, d99 {share[99]:.2f}", flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
