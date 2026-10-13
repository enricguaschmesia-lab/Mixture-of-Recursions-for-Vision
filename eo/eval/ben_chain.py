"""Chained generation on BigEarthNet (Phase 6 Step 5b, D6.11). `.venv`.

Impute the missing input with a generator, then classify from both inputs:

    S1toS2:  [BO_S1 s1 EO_S1 BO_S2] -> 196 ids in S2L2A's slot -> EO_S2 appended
    S2toS1:  [BO_S2 s2 EO_S2 BO_S1] -> 196 ids in S1GRD's slot -> EO_S1 appended

The classifier then reads Step 5's fixed S1+S2 layout (the S2L2A block, the S1GRD block,
BO_BEN), with the generated block in place of the missing real one, so the ceiling is the
real pair on the same rows (PHASE6_PLAN Step 5b).

GENERATION is slot-masked: only the target's 15,360 codebook ids are allowed, exactly
TOKENS_PER_SAMPLE of them are drawn, and EO is appended. It samples at temperature T, with a
KV cache. The cache is vanilla-only, as in eval/ben.py. `gen_cache_check` compares cached
against uncached logits on a real continuation; with shift=1 (the control) it compares each
cached step with the PREVIOUS uncached position, which must disagree.
The unconstrained softmax mass on the target slot is recorded at every step (`slot_mass`), so
the mask is never doing all the work unseen.

FIRST-LEG FIDELITY (`teacher_forced`): the CE of the real target tokens given the other
modality, against the CE given nothing (`[BO_target]` alone), over the model's full
vocabulary (Phase 3's metric), and the top-1 accuracy within the slot.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from eo.data import ben_vocab as V
from eo.eval import ben as E
from eo.terramesh_tok.contract import TOKENS_PER_SAMPLE

DIRECTIONS = {"S1toS2": ("S1", "S2"), "S2toS1": ("S2", "S1")}     # (given, generated)
LAYOUT = ("S2", "S1")                                              # Step 5's fixed order


def slot(src: str):
    info = V.get_modality(V.INPUT_MODALITIES[src])
    return info.codebook_offset, info.codebook_size, info.bo_id, info.eo_id


def gen_prompt(ds, row: int, given: str, target: str) -> torch.Tensor:
    return torch.cat([ds.chunk(given, row), torch.tensor([slot(target)[2]])])


def block(target: str, raw) -> torch.Tensor:
    """A generated body (raw codebook indices) as `BO body EO` in global ids."""
    off, _, bo, eo = slot(target)
    body = torch.as_tensor(np.asarray(raw, dtype=np.int64)) + off               # int64 FIRST
    return torch.cat([torch.tensor([bo]), body, torch.tensor([eo])])


def cls_prompt(blocks: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Classifier prompt from the blocks present, in the fixed layout, then BO_BEN."""
    return torch.cat([blocks[s] for s in LAYOUT if s in blocks] + [torch.tensor([V.BO_BEN])])


def _ctx(autocast):
    return torch.autocast("cuda", dtype=autocast) if autocast is not None else E._null()


@torch.no_grad()
def generate_blocks(model, prompts: torch.Tensor, target: str, *, n_samples: int, temperature: float,
                    seed: int, autocast: Optional[torch.dtype] = None) -> Dict[str, np.ndarray]:
    """prompts: (B, L) equal-length, ending in BO_target. Returns tokens (B, n, 196) raw codebook
    indices and slot_mass (B, n): the mean unconstrained softmax mass on the slot over the steps."""
    off, size, _, _ = slot(target)
    B, dev = prompts.shape[0], prompts.device
    with _ctx(autocast):
        out = model(input_ids=prompts, use_cache=True, logits_to_keep=1)
    cache = E._expand_cache(out.past_key_values, n_samples)
    logits = out.logits[:, -1, :].float().repeat_interleave(n_samples, 0)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    toks, mass = [], torch.zeros(B * n_samples, device=dev)
    for t in range(TOKENS_PER_SAMPLE):
        mass += torch.softmax(logits, -1)[:, off:off + size].sum(-1)
        p = torch.softmax(logits[:, off:off + size] / max(temperature, 1e-6), -1)
        x = torch.multinomial(p, 1, generator=g).squeeze(1)
        toks.append(x)
        if t == TOKENS_PER_SAMPLE - 1:
            break
        with _ctx(autocast):
            o = model(input_ids=(x + off).unsqueeze(1), past_key_values=cache, use_cache=True)
        cache, logits = o.past_key_values, o.logits[:, -1, :].float()
    tok = torch.stack(toks, 1).view(B, n_samples, TOKENS_PER_SAMPLE)
    return {"tokens": tok.cpu().numpy().astype(np.uint16),
            "slot_mass": (mass / TOKENS_PER_SAMPLE).view(B, n_samples).cpu().numpy()}


@torch.no_grad()
def gen_cache_check(model, prompts: torch.Tensor, cont_raw: torch.Tensor, target: str, *,
                    n_steps: int = 4, shift: int = 0, autocast: Optional[torch.dtype] = None) -> float:
    """Max |cached - uncached| logit over the target slot while feeding the real continuation
    `cont_raw` (B, >= n_steps raw indices). shift=1 compares cached step t with uncached t - 1:
    the control, which must disagree."""
    off, size, _, _ = slot(target)
    cont = cont_raw[:, :n_steps].to(prompts.device).long() + off
    with _ctx(autocast):
        out = model(input_ids=prompts, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        cached, full = [], [out.logits[:, -1, off:off + size].float()]
        seq = prompts
        for t in range(n_steps):
            ids = cont[:, t:t + 1]
            o = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            cached.append(o.logits[:, -1, off:off + size].float())
            seq = torch.cat([seq, ids], 1)
            full.append(model(input_ids=seq, use_cache=False, logits_to_keep=1).logits[:, -1, off:off + size].float())
    # full[0] is the prompt's own logits; cached[t] corresponds to full[t + 1]
    return max(float((cached[t] - full[t + 1 - shift]).abs().max()) for t in range(n_steps))


@torch.no_grad()
def teacher_forced(model, prompts: torch.Tensor, truth_raw: torch.Tensor, target: str, *,
                   autocast: Optional[torch.dtype] = None) -> Dict[str, np.ndarray]:
    """Per-sequence mean CE (nats, full vocabulary) of the 196 real target tokens after `prompts`
    (ending in BO_target), and the top-1 accuracy within the target slot."""
    off, size, _, _ = slot(target)
    tgt = truth_raw.to(prompts.device).long()
    seq = torch.cat([prompts, tgt + off], 1)
    with _ctx(autocast):
        lg = model(input_ids=seq, logits_to_keep=TOKENS_PER_SAMPLE + 1).logits[:, :TOKENS_PER_SAMPLE, :].float()
    ce = F.cross_entropy(lg.transpose(1, 2), tgt + off, reduction="none").mean(1)
    acc = (lg[..., off:off + size].argmax(-1) == tgt).float().mean(1)
    return {"ce": ce.cpu().numpy(), "acc": acc.cpu().numpy()}
