# eo/generate/cached.py
"""KV-cached generation of one block, with explicit position_ids for the temporal stage (`.venv`).

WHY POSITIONS MUST BE PASSED. HF's Llama takes the position of a token fed against a cache from the
cache's length (`cache_position`). That is right for every layout before the temporal stage, whose
positions are 0..L-1. The temporal stage's positions are the LATTICE (A1 decision 9,
eo/data/temporal_dataset.py): after a date block the counter goes back to where the block began, so
behind k date blocks the lattice position is the cache length minus 3-4 tokens per block. A cached
decode on default positions would place every generated token k blocks' worth too far, and Step 0
measured that one position of misalignment costs more than a source's whole gain. So a decode under
the lattice passes the prompt's own position_ids, and for the t-th generated token the prompt's last
position + 1 + t: positions are sequential inside a block, and what is generated is one block.

`position_ids=None` is the model's default, the code path of every result before the temporal stage:
`sample` is eo/experiments/generation/tm_compare_gen.py's sampler, moved here unchanged on that path
(tm_compare_gen imports it), and eo/eval/ben.py's cache_check has the same optional argument.

Prompts in one call must have one length: the layouts of one task do, when every date block keeps
its year (eo/eval/temporal.py). Gate check Y16 (eo/scripts/verify_temporal.py) holds `cache_check` on
T360-T with a prompt behind four date blocks, against the same decode on default positions.

Runs in `.venv`; never imports terratorch.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from eo.terramesh_tok.contract import TOKENS_PER_SAMPLE


def slot(target: str) -> Tuple[int, int]:
    """(offset, size) of a modality's codebook slot, by name (the EO slots and the date rows' registry)."""
    from eo.data import temporal_vocab as TV
    info = TV.get_modality(target)
    return info.codebook_offset, info.codebook_size


def next_positions(position_ids: torch.Tensor, t: int) -> torch.Tensor:
    """(B, 1) position of the t-th generated token (t = 0 is the first after the prompt)."""
    return position_ids[:, -1:] + 1 + int(t)


def _pos(position_ids: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {} if position_ids is None else {"position_ids": position_ids}


@torch.no_grad()
def sample(model, prompts, target, *, temperature, top_p, seed, position_ids=None,
           n_tokens: int = TOKENS_PER_SAMPLE, slot_range: Optional[Tuple[int, int]] = None):
    """(B, L) prompts ending in BO_target -> (B, n_tokens) raw codebook indices, and the mean
    unconstrained softmax mass on the slot. ben_chain.generate_blocks plus top-p over the slot
    (HF's rule: a token is kept while the mass before it is below top_p). The slot sort is
    15,360 wide, never vocabulary-sized; run it on the Titan X.

    `position_ids` (B, L): the prompt's positions; generated tokens follow at +1 each (module
    docstring). None: the model's default positions, as every result before the temporal stage."""
    off, size = slot(target) if slot_range is None else slot_range
    out = model(input_ids=prompts, use_cache=True, logits_to_keep=1, **_pos(position_ids))
    cache, logits = out.past_key_values, out.logits[:, -1, :].float()
    g = torch.Generator(device=prompts.device).manual_seed(int(seed))
    toks, mass = [], torch.zeros(len(prompts), device=prompts.device)
    for t in range(n_tokens):
        mass += torch.softmax(logits, -1)[:, off:off + size].sum(-1)
        p = torch.softmax(logits[:, off:off + size] / temperature, -1)
        if top_p < 1.0:
            sp, si = torch.sort(p, -1, descending=True)
            sp = sp * ((sp.cumsum(-1) - sp) < top_p)
            p = torch.zeros_like(p).scatter(-1, si, sp)
            p = p / p.sum(-1, keepdim=True)
        x = torch.multinomial(p, 1, generator=g).squeeze(1)
        toks.append(x)
        if t == n_tokens - 1:
            break
        step = None if position_ids is None else next_positions(position_ids, t)
        o = model(input_ids=(x + off).unsqueeze(1), past_key_values=cache, use_cache=True, **_pos(step))
        cache, logits = o.past_key_values, o.logits[:, -1, :].float()
    return torch.stack(toks, 1), mass / n_tokens


@torch.no_grad()
def cache_check(model, prompts: torch.Tensor, continuation: torch.Tensor, *,
                position_ids: Optional[torch.Tensor] = None, decode_positions: bool = True,
                autocast: Optional[torch.dtype] = None, decode_offset: int = 0) -> Dict:
    """Cached decoding against full teacher-forced passes, feeding a known continuation (B, n global
    ids): after each cached step, the whole sequence so far through the model without a cache, and the
    last position's logits compared over the whole vocabulary. fp32 unless `autocast` (A1 Step 7): then
    both sides run under torch.autocast at that dtype, as a decode at that precision would.

    The full passes use `position_ids` (the prompt's, then +1 per token) when given, else the default.
    The cached decode uses the same, unless `decode_positions=False`: the model's default positions
    against lattice references, the control that must disagree. `decode_offset` (a control for any
    layout) moves the cached steps' positions by that much; it needs `position_ids`.

    Returns the largest |difference| and |logit| (NaN-safe: reduced in torch, so a non-finite step
    shows), and the difference in rounding steps of the autocast dtype at the largest |logit|
    (`max_ulp_diff`, eo/eval/ben.py's CACHE_ULPS rule)."""
    from eo.eval.ben import rounding_step
    n = int(continuation.shape[1])
    lattice = position_ids is not None
    if decode_offset and not lattice:
        raise ValueError("decode_offset moves explicit positions: pass position_ids")
    ctx = torch.autocast(prompts.device.type, dtype=autocast) if autocast is not None else _NullCtx()
    diffs, mags = [], []
    with ctx:
        out = model(input_ids=prompts, use_cache=True, logits_to_keep=1,
                    **_pos(position_ids if decode_positions else None))
        cache = out.past_key_values
        seq, pos = prompts, position_ids
        for t in range(n):
            ids = continuation[:, t:t + 1]
            step = next_positions(position_ids, t) if lattice else None
            fed = None if not decode_positions else (step + int(decode_offset) if lattice else None)
            o = model(input_ids=ids, past_key_values=cache, use_cache=True, **_pos(fed))
            cache = o.past_key_values
            seq = torch.cat([seq, ids], 1)
            pos = torch.cat([pos, step], 1) if lattice else None
            full = model(input_ids=seq, use_cache=False, logits_to_keep=1, **_pos(pos)).logits[:, -1, :].float()
            a = o.logits[:, -1, :].float()
            diffs.append((a - full).abs().max())
            mags.append(torch.maximum(a.abs(), full.abs()).max())
    d, m = torch.stack(diffs), torch.stack(mags)
    worst, big = d.max(), m.max()
    return {"max_abs_logit_diff": float(worst), "per_step": [float(x) for x in d], "max_abs_logit": float(big),
            "max_ulp_diff": float(worst / rounding_step(big, autocast or torch.float32)),
            "finite": bool(torch.isfinite(d).all() and torch.isfinite(m).all()), "steps": n,
            "batch": int(prompts.shape[0]), "prompt_length": int(prompts.shape[1]),
            "autocast": None if autocast is None else str(autocast).replace("torch.", ""),
            "positions": ("explicit" if lattice and decode_positions else "default")
                         + (f", decode offset {int(decode_offset)}" if decode_offset else "")}


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
