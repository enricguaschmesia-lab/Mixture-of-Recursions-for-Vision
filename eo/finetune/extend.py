"""Grow a pretrained EO checkpoint's vocabulary by the BEN ids (PHASE6_PLAN Step 2.3/2.4, D6.4).

⚠ NOT `resize_token_embeddings`. On the recursive class HF's resize finds no
`get_output_embeddings` (worklog §5 item 26), resizes the input side only and silently
leaves `lm_head` at the old size and untied. The resize here is explicit and identical for
every class that has `.model.embed_tokens` and `.lm_head`: build one new embedding, point
`lm_head.weight` at THE SAME Parameter, and assert identity (`is`), not equality.

New rows start at the mean of the existing rows plus small noise (HF's `mean_resizing`
idea): their logits then start in the bulk of the distribution rather than at an extreme.

The identity control (2.4): on a fixed TerraMesh held-out batch, the logits over ids
0..87,555 must be unchanged by the extension, in fp32. Hidden states cannot move (no input
contains a new id) and the old output rows are copied, so any difference above GEMM
round-off means a row moved. `insert_before_pad=True` builds the broken layout the check
must reject (BEN ids inserted before PAD, which shifts PAD).
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def embedding_and_head(model):
    emb = model.model.embed_tokens
    head = model.lm_head
    return emb, head


def assert_tied(model) -> None:
    emb, head = embedding_and_head(model)
    if head.weight is not emb.weight:
        raise AssertionError("lm_head.weight is not embed_tokens.weight (the tie is broken)")
    if head.weight.shape[0] != model.config.vocab_size:
        raise AssertionError(f"vocab {head.weight.shape[0]} != config {model.config.vocab_size}")


@torch.no_grad()
def extend_vocab(model, n_new: int, seed: int = 0, noise: float = 0.01,
                 insert_before_pad: bool = False, pad_id: int = None) -> Dict:
    """In place. Returns a record of what was done (for the identity artifact)."""
    assert_tied(model)
    emb, _ = embedding_and_head(model)
    old = emb.weight.data
    v_old, d = old.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    mean = old.float().mean(0, keepdim=True).cpu()
    std = old.float().std(0, keepdim=True).cpu()
    new_rows = (mean + noise * std * torch.randn(n_new, d, generator=g)).to(old.dtype)

    if insert_before_pad:
        # THE BROKEN CONTROL: BEN rows placed before PAD, so PAD (and anything after it)
        # moves to a new id. Never used for a real checkpoint.
        assert pad_id is not None
        w = torch.cat([old[:pad_id].cpu(), new_rows, old[pad_id:].cpu()], 0)
    else:
        w = torch.cat([old.cpu(), new_rows], 0)

    new_emb = nn.Embedding(v_old + n_new, d, padding_idx=emb.padding_idx,
                           device=old.device, dtype=old.dtype)
    new_emb.weight.data.copy_(w.to(old.device))
    model.model.embed_tokens = new_emb
    new_head = nn.Linear(d, v_old + n_new, bias=False, device=old.device, dtype=old.dtype)
    new_head.weight = new_emb.weight               # THE tie: one Parameter, two modules
    model.lm_head = new_head
    model.config.vocab_size = v_old + n_new
    if hasattr(model, "vocab_size"):
        model.vocab_size = v_old + n_new
    assert_tied(model)
    return {"vocab_before": int(v_old), "vocab_after": int(v_old + n_new), "n_new": int(n_new),
            "init": "mean + noise*std*N(0,1)", "noise": noise, "seed": seed,
            "insert_before_pad": bool(insert_before_pad)}


@torch.no_grad()
def logits_on(model, batch: Dict[str, torch.Tensor], device) -> torch.Tensor:
    """fp32 logits for a TerraMesh batch (no autocast)."""
    out = model(input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                position_ids=batch["position_ids"].to(device))
    return out.logits.float()


def identity_report(before: torch.Tensor, after: torch.Tensor, attention_mask: torch.Tensor,
                    v_old: int) -> Dict:
    """Max |Δlogit| over ids < v_old at real (non-pad) positions, and where it peaks."""
    m = attention_mask.bool().to(before.device)
    a, b = before[m][:, :v_old], after[m][:, :v_old]
    diff = (a - b).abs()
    col = diff.max(0).values
    return {"positions": int(m.sum()), "ids": int(v_old),
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "worst_id": int(col.argmax()), "ids_with_diff_gt_1e-3": int((col > 1e-3).sum()),
            "logit_scale": float(a.abs().mean())}
