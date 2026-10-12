#!/usr/bin/env python
"""Extend a pretrained EO checkpoint by the 21 BEN ids, with its identity control (D6.4, `.venv`).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<titanv> python eo/scripts/extend_checkpoint.py \\
        --arm-config eo_terramesh/arm_b10_vanilla \\
        --checkpoint /data/enric/runs/pretrain/phase3/vanilla10_20261002_133848/checkpoint-3300 \\
        --out /data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300

Writes the extended weights (same format convention as training: safetensors unless the
model shares weights) plus `extension.json`, the identity artifact preflight's
`check_init_from` requires before any fine-tune may start from this directory:

  * identity: logits over ids 0..87,555 on a fixed, stratified TerraMesh held-out batch,
    before vs after, fp32. Must be within IDENTITY_TOL.
  * control 1 (must FAIL the identity): the BEN ids inserted before PAD.
  * control 2 (must FAIL the tie check): an untied resize (input side only, as HF's
    resize does on the recursive class).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.data import ben_vocab as BV  # noqa: E402
from eo.data import eo_vocab as EV  # noqa: E402
from eo.finetune import extend as X  # noqa: E402
from eo.generate import conditional as G  # noqa: E402

IDENTITY_TOL = 1e-4      # fp32 GEMM round-off is ~1e-6 at these logit scales; a moved row is O(1)
N_ROWS_PER_CORPUS = 2    # stratified: majortom (S1RTC) AND ssl4eos12 (S1GRD) rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_file(d: Path) -> Path:
    for n in ("model.safetensors", "pytorch_model.bin"):
        if (d / n).exists():
            return d / n
    raise FileNotFoundError(f"no weights in {d}")


def fixed_batch(cfg):
    """A stratified held-out TerraMesh batch, built exactly as evaluation builds it."""
    from eo.data.eval_split import load_row_table
    from eo.eval.teacher_forced import eval_loader
    ds, _ = eval_loader(cfg, num_workers=0)
    corpus = load_row_table(ds.root_dir).corpus.values
    pos = []
    for c in ("majortom", "ssl4eos12"):
        cand = [i for i, r in enumerate(ds.rows) if corpus[r] == c]
        pos += cand[:: max(1, len(cand) // N_ROWS_PER_CORPUS)][:N_ROWS_PER_CORPUS]
    items = [ds[i] for i in pos]
    batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
    return batch, [int(ds.rows[i]) for i in pos]


def per_row_logits(model, batch, device):
    out = []
    for i in range(batch["input_ids"].shape[0]):
        b = {k: v[i:i + 1] for k, v in batch.items()}
        out.append(X.logits_on(model, b, device).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-config", required=True)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f"{a.out} exists and is not empty")
    device = "cuda"
    gpu = torch.cuda.get_device_name(0)

    model, cfg = G.build_model(a.arm_config, str(a.checkpoint), device=device)
    model.float()
    X.assert_tied(model)
    batch, rows = fixed_batch(cfg)
    before = per_row_logits(model, batch, device)
    v_old = EV.TOTAL_VOCAB_SIZE
    assert before.shape[-1] == v_old

    # Control 1: insert before PAD. Must FAIL identity.
    ctrl = copy.deepcopy(model)
    X.extend_vocab(ctrl, BV.N_NEW, seed=a.seed, insert_before_pad=True, pad_id=EV.PAD_ID)
    rep_c1 = X.identity_report(before, per_row_logits(ctrl, batch, device),
                               batch["attention_mask"], v_old)
    del ctrl
    # Control 2: an untied resize. Must FAIL the tie assertion.
    ctrl = copy.deepcopy(model)
    old = ctrl.model.embed_tokens.weight.data
    e = torch.nn.Embedding(old.shape[0] + BV.N_NEW, old.shape[1], device=old.device)
    e.weight.data[: old.shape[0]] = old
    ctrl.model.embed_tokens = e
    ctrl.config.vocab_size = old.shape[0] + BV.N_NEW
    try:
        X.assert_tied(ctrl)
        c2_fired = False
    except AssertionError as err:
        c2_fired, c2_msg = True, str(err)
    del ctrl
    torch.cuda.empty_cache()

    # The real extension.
    rec = X.extend_vocab(model, BV.N_NEW, seed=a.seed)
    after = per_row_logits(model, batch, device)
    assert after.shape[-1] == BV.TOTAL_VOCAB_SIZE
    rep = X.identity_report(before, after, batch["attention_mask"], v_old)
    new_logit_mean = float(after[batch["attention_mask"].bool()][:, v_old:].mean())
    old_logit_mean = float(after[batch["attention_mask"].bool()][:, :v_old].mean())

    ok = rep["max_abs_diff"] <= IDENTITY_TOL
    c1_fired = rep_c1["max_abs_diff"] > IDENTITY_TOL
    print(f"identity: max |dlogit| {rep['max_abs_diff']:.3e} (tol {IDENTITY_TOL}) -> {'PASS' if ok else 'FAIL'}")
    print(f"control 1 (insert before PAD): max |dlogit| {rep_c1['max_abs_diff']:.3e}, worst id "
          f"{rep_c1['worst_id']} -> {'fired' if c1_fired else 'DID NOT FIRE'}")
    print(f"control 2 (untied resize): {'fired: ' + c2_msg if c2_fired else 'DID NOT FIRE'}")
    print(f"new-row logit mean {new_logit_mean:.3f} vs old-row mean {old_logit_mean:.3f}")
    if not (ok and c1_fired and c2_fired):
        sys.exit("extension identity check failed; nothing saved")

    a.out.mkdir(parents=True, exist_ok=True)
    shares = bool(cfg.recursive.get("enable")) or bool(cfg.get("mor", {}).get("enable"))
    model.save_pretrained(a.out, safe_serialization=not shares)
    wf = weights_file(a.out)
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse",
                             "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    doc = {
        "source_checkpoint": str(a.checkpoint),
        "source_weights_sha256": sha256(weights_file(a.checkpoint)),
        "arm_config": a.arm_config,
        "token_set": (cfg.get("multimodal") or {}).get("token_set", "original"),
        "extension": rec,
        "ben_vocab": {"base": BV.BASE, "bo_ben": BV.BO_BEN, "eo_ben": BV.EO_BEN,
                      "total": BV.TOTAL_VOCAB_SIZE, "pad_id": BV.PAD_ID},
        "identity": {**rep, "tol": IDENTITY_TOL, "passed": ok, "batch_rows": rows,
                     "precision": "fp32"},
        "control_insert_before_pad": {**rep_c1, "fired": c1_fired},
        "control_untied_resize": {"fired": c2_fired},
        "new_row_logit_mean": new_logit_mean, "old_row_logit_mean": old_logit_mean,
        "weights_file": wf.name, "weights_sha256": sha256(wf),
        "tied": True, "fork_commit": commit, "gpu": gpu,
    }
    (a.out / "extension.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"saved {wf} ({doc['weights_sha256'][:12]}) + extension.json")


if __name__ == "__main__":
    main()
