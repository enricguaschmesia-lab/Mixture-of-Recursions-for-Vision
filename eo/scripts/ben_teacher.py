#!/usr/bin/env python
"""L11b.3 (PHASE6_PLAN §2c): a teacher's sampled label blocks on the unlabelled pool. `.venv`, GPU.

    python eo/scripts/ben_teacher.py --config eo_ben/b10_lora --checkpoint <run>/checkpoint-1461 \\
        --condition S1S2 --role ft_unlab --k 4 --name teacher/l11b_b10lora1461_S1S2_K4 \\
        --check-ref <reports>/ben/b10_lora/checkpoint-1461/test_a/S1S2.npz

Writes <BEN_ROOT>/<name>.npz (rows, masks (N, K) int32 = the K sampled label sets as 19-bit masks,
the teacher's Rao-Blackwellized marginals (N, 19) float16, slot_mass) and <name>.json (provenance,
the plumbing check, block statistics). The student reads them through BenTokenDataset(teacher=).

SAMPLING is Step 5's constrained decoder (`eo/eval/ben.decode_labels`) at T = 1, K paths per patch,
seed `seed * 1_000_003 + b0` per batch at global offset b0, as eval_ben. The teacher never sees a
label: the pool's rows carry none (the prompts-only dataset reads tokens only).

PLUMBING CHECK (`--check-ref`, before any block is kept): the same decoder with return_samples, on
test_a's first --check-n rows at the reference's batch and seed (K = 16, condition S1S2), must
reproduce eval_ben's stored sample frequencies and marginals for this checkpoint, and its
per-sample sets must average to them. Control: another patch's S2 must not reproduce them. Bit
identity needs the reference's card (A100); --local-smoke only warns elsewhere.

Resumable: one part file per --part rows under <name>.parts/, skipped if present; merged at the end.
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
from eo.data.ben_dataset import BenTokenDataset, DEFAULT_ROOT, CONDITIONS, load_index  # noqa: E402
from eo.eval import ben as E  # noqa: E402
from eo.finetune.init import build_finetuned  # noqa: E402

BITS = (1 << np.arange(L.N_CLASSES)).astype(np.int64)


def to_masks(sets: np.ndarray) -> np.ndarray:
    """(..., 19) bool -> (...) int32 19-bit masks (labels19's encoding, labels.to_mask)."""
    return (sets.astype(np.int64) * BITS).sum(-1).astype(np.int32)


def plumbing(model, idx, ref_path: Path, n: int, ac, local_smoke: bool) -> dict:
    ref = np.load(ref_path)
    ds = BenTokenDataset(role="test_a", condition="S1S2", index=idx)
    assert (ref["rows"][:n] == ds.rows[:n]).all(), "the reference was written for other rows"
    p_real = torch.stack([ds.prompt(i, "S1S2") for i in range(n)]).cuda()
    p_ctl = torch.stack([ds.prompt(i, "S1S2", input_row=int(r)) for i, r in enumerate(np.roll(ds.rows[:n], 1))]).cuda()
    res = E.decode_labels(model, p_real, k_samples=16, greedy=True, seed=0, autocast=ac, return_samples=True)
    ctl = E.decode_labels(model, p_ctl, k_samples=16, greedy=True, seed=0, autocast=ac)
    pl = {"ref": str(ref_path), "n": n, "batch": n, "k": 16,
          "samples_mean_equals_sample_freq": bool(np.array_equal(res["samples"].mean(1), res["sample_freq"])),
          "sample_freq_equal_ref": bool(np.array_equal(res["sample_freq"], ref["sample_freq"][:n])),
          "marginal_max_abs_diff": float(np.abs(res["marginal"] - ref["marginal"][:n]).max()),
          "greedy_equal": bool((res["greedy"] == ref["greedy"][:n]).all()),
          "control_marginal_max_abs_diff": float(np.abs(ctl["marginal"] - ref["marginal"][:n]).max()),
          "control_sample_freq_equal_ref": bool(np.array_equal(ctl["sample_freq"], ref["sample_freq"][:n]))}
    pl["passed"] = (pl["samples_mean_equals_sample_freq"] and pl["sample_freq_equal_ref"] and pl["greedy_equal"]
                    and pl["marginal_max_abs_diff"] < 1e-4)
    pl["control_fired"] = pl["control_marginal_max_abs_diff"] > 0.05 and not pl["control_sample_freq_equal_ref"]
    print("plumbing:", json.dumps(pl), flush=True)
    if not (pl["passed"] and pl["control_fired"]):
        if not local_smoke:
            sys.exit("plumbing check failed")
        print("WARNING: plumbing check failed; --local-smoke (not the reference's card)", flush=True)
    return pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--condition", choices=CONDITIONS, required=True, help="the teacher's inputs")
    ap.add_argument("--role", default="ft_unlab")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--name", required=True, help="output path under BEN_ROOT, without suffix")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--part", type=int, default=16384, help="rows per resumable part file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--autocast", choices=["none", "fp16"], default="fp16")
    ap.add_argument("--limit", type=int, default=None, help="smoke: the first N rows only (writes <name>_limitN)")
    ap.add_argument("--check-ref", type=Path, default=None, help="eval_ben's test_a S1S2.npz of this checkpoint")
    ap.add_argument("--check-n", type=int, default=128)
    ap.add_argument("--local-smoke", action="store_true")
    ap.add_argument("--init-from", default="/data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300")
    a = ap.parse_args()

    gpu = torch.cuda.get_device_name(0)
    if "TITAN V" in gpu:
        sys.exit("refusing the TITAN V: it stalls within seconds under load (worklog 2026-10-12)")
    root = Path(DEFAULT_ROOT)
    name = a.name + (f"_limit{a.limit}" if a.limit else "")
    out = root / name
    parts = out.parent / (out.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    ac = {"none": None, "fp16": torch.float16}[a.autocast]
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    idx = load_index()
    model, _ = build_finetuned(a.config, a.checkpoint, device="cuda", overrides=[f"finetune.init_from={a.init_from}"])
    model.eval()
    pl_file = parts / "plumbing.json"
    if a.check_ref is not None and not pl_file.exists():
        pl_file.write_text(json.dumps(plumbing(model, idx, a.check_ref, a.check_n, ac, a.local_smoke), indent=2) + "\n")
    pl = json.loads(pl_file.read_text()) if pl_file.exists() else None
    if pl is None and not a.local_smoke:
        sys.exit("no plumbing check: pass --check-ref (or --local-smoke for a smoke)")

    ds = BenTokenDataset(role=a.role, condition=a.condition, index=idx, prompts_only=True)
    lab = idx.labels19.to_numpy()[ds.rows]
    if a.role == "ft_unlab" and (lab >= 0).any():
        sys.exit(f"{int((lab >= 0).sum())} rows of the pool are labelled")
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    probe = torch.stack([ds.prompt(i, a.condition) for i in range(4)]).cuda()
    cc = E.cache_check(model, probe, autocast=ac)
    if cc["max_abs_logit_diff"] > E.CACHE_TOL[a.autocast]:
        sys.exit(f"KV cache disagrees with uncached decoding: {cc}")

    t0 = time.time()
    for p0 in range(0, n, a.part):
        pf = parts / f"part_{p0 // a.part:04d}.npz"
        if pf.exists():
            continue
        acc = {"masks": [], "marginal": [], "slot_mass": []}
        for b0 in range(p0, min(p0 + a.part, n), a.batch):
            bp = np.arange(b0, min(b0 + a.batch, p0 + a.part, n))
            prompts = torch.stack([ds.prompt(int(p), a.condition) for p in bp]).cuda()
            r = E.decode_labels(model, prompts, k_samples=a.k, greedy=False, seed=a.seed * 1_000_003 + b0,
                                autocast=ac, return_samples=True)
            acc["masks"].append(to_masks(r["samples"]))
            acc["marginal"].append(r["marginal"].astype(np.float16))
            acc["slot_mass"].append(r["slot_mass"].astype(np.float32))
        tmp = pf.with_suffix(".tmp.npz")
        np.savez(tmp, pos=np.arange(p0, min(p0 + a.part, n)), **{k: np.concatenate(v) for k, v in acc.items()})
        tmp.replace(pf)
        done = min(p0 + a.part, n)
        print(f"  {done}/{n}  {done / (time.time() - t0):.1f} patches/s  {(time.time() - t0) / 60:.1f} min", flush=True)

    z = [np.load(parts / f"part_{i:04d}.npz") for i in range((n + a.part - 1) // a.part)]
    pos = np.concatenate([q["pos"] for q in z])
    assert (pos == np.arange(n)).all(), "parts do not tile the rows"
    masks = np.concatenate([q["masks"] for q in z])
    marg = np.concatenate([q["marginal"] for q in z])
    sm = np.concatenate([q["slot_mass"] for q in z])
    rows = ds.rows[:n]
    np.savez_compressed(out.with_suffix(".npz"), rows=rows, masks=masks, marginal=marg, slot_mass=sm)
    sets = (masks[..., None].astype(np.int64) & BITS) > 0                      # (N, K, 19)
    ft = idx.labels19.to_numpy()[idx.ft_train.to_numpy(bool)]
    ft_freq = ((ft[:, None] & BITS) > 0).mean(0)
    doc = {"config": a.config, "checkpoint": a.checkpoint, "condition": a.condition, "role": a.role,
           "k": a.k, "n": int(n), "temperature": 1.0, "seed": a.seed, "batch": a.batch, "autocast": a.autocast,
           "sampler": "eo.eval.ben.decode_labels (constrained, canonical order), return_samples",
           "plumbing": pl, "cache_check": cc, "gpu": gpu, "fork_commit": commit,
           "seconds": round(time.time() - t0, 1),
           "stats": {"mean_labels_per_block": float(sets.sum(-1).mean()),
                     "empty_blocks": float((masks == 0).mean()),
                     "blocks_differing_within_patch": float((masks != masks[:, :1]).any(1).mean()),
                     "mean_slot_mass": float(sm.mean()),
                     "class_freq_blocks": dict(zip(L.SHORT, sets.mean((0, 1)).round(4).tolist())),
                     "class_freq_ft_train": dict(zip(L.SHORT, ft_freq.round(4).tolist()))},
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.with_suffix(".json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "stats"} | {"stats": {k: v for k, v in doc["stats"].items()
                                                                              if not k.startswith("class_freq")}}, indent=1))


if __name__ == "__main__":
    main()
