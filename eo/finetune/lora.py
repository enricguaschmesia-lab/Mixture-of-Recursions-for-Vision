"""Hand-rolled LoRA for the EO models (PHASE6_PLAN §4.6, §4.7). `.venv`; no `peft`.

Why not `peft`: it is not installed, adding it edits upstream's lockfile (rule 9), and it
wraps modules BY NAME, which on MoR's aliased shared block is one more thing to verify
rather than rely on. Here every wrapped module is found by identity, so an aliased module
gets ONE adapter, shared wherever the base module is shared.

What trains under LoRA, and nothing else:
  * a low-rank adapter (A: r x in, B: out x r, B = 0 at init) on every attention and MLP
    projection: q, k, v, o, gate, up, down;
  * the 21 NEW vocabulary rows, as their own Parameter (`SplitEmbedding.new`).

⚠ WHY THE NEW ROWS ARE A SEPARATE PARAMETER. Masking the gradient of the old rows does
not freeze them under AdamW: decoupled weight decay shrinks every row of a parameter in
the decay group whether its gradient is zero or not. `SplitEmbedding` keeps the 87,556
pretraining rows in a frozen Parameter (requires_grad=False, so the optimizer never sees
it) and the 21 new rows in a trainable one. The output projection reads the SAME two
Parameters (`SplitLMHead`), so the input/output tie survives the split.
Gate X6 checks that every frozen tensor is bit-identical after training, and its control
shows the masked-gradient + AdamW variant moving them.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r, self.scale = int(r), float(alpha) / float(r)
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A)


class SplitEmbedding(nn.Module):
    """Embedding over [old (frozen) ; new (trainable)] without materializing the concat."""

    def __init__(self, weight: torch.Tensor, n_old: int, padding_idx=None):
        super().__init__()
        self.n_old = int(n_old)
        self.old = nn.Parameter(weight[:n_old].detach().clone(), requires_grad=False)
        self.new = nn.Parameter(weight[n_old:].detach().clone(), requires_grad=True)
        self.padding_idx = padding_idx

    @property
    def num_embeddings(self) -> int:
        return self.n_old + self.new.shape[0]

    @property
    def weight(self) -> torch.Tensor:            # for code that reads .weight (reports only)
        return torch.cat([self.old, self.new], 0)

    def forward(self, ids):
        is_new = ids >= self.n_old
        e_old = F.embedding(ids.clamp(max=self.n_old - 1), self.old)
        if not bool(is_new.any()):
            return e_old
        e_new = F.embedding((ids - self.n_old).clamp(min=0), self.new)
        return torch.where(is_new.unsqueeze(-1), e_new, e_old)


class SplitLMHead(nn.Module):
    """Tied output projection over the SAME two Parameters as `SplitEmbedding`."""

    def __init__(self, emb: SplitEmbedding):
        super().__init__()
        object.__setattr__(self, "_emb", emb)    # a reference, not a submodule: no duplicate keys

    @property
    def weight(self) -> torch.Tensor:
        return self._emb.weight

    def forward(self, h):
        return F.linear(h, torch.cat([self._emb.old, self._emb.new], 0))


def _unique_linears(model) -> List[nn.Linear]:
    """Every target projection, ONCE per distinct module object (aliases share an adapter)."""
    seen, out = set(), []
    for name, mod in model.named_modules(remove_duplicate=False):
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGETS and id(mod) not in seen:
            seen.add(id(mod))
            out.append(mod)
    return out


def apply_lora(model, r: int, alpha: float, n_old_vocab: int) -> Dict:
    """In place. Freezes everything, then adds adapters and the split vocabulary."""
    if getattr(model.config, "_lora_applied", False):
        raise RuntimeError("LoRA already applied")
    for p in model.parameters():
        p.requires_grad_(False)

    targets = _unique_linears(model)
    wrapped = {id(m): LoRALinear(m, r, alpha) for m in targets}
    n_sites = 0
    for parent in list(model.modules()):
        for cname, child in list(parent.named_children()):
            if id(child) in wrapped:
                setattr(parent, cname, wrapped[id(child)])
                n_sites += 1

    emb = model.model.embed_tokens
    head = model.lm_head
    if head.weight is not emb.weight:
        raise AssertionError("LoRA expects a tied model (lm_head.weight is embed_tokens.weight)")
    split = SplitEmbedding(emb.weight.data, n_old_vocab, padding_idx=emb.padding_idx)
    model.model.embed_tokens = split
    model.lm_head = SplitLMHead(split)
    model.config._lora_applied = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapters = sum(p.numel() for m in wrapped.values() for p in (m.lora_A, m.lora_B))
    return {"rank": r, "alpha": alpha, "targets": list(TARGETS),
            "unique_projections": len(targets), "wrap_sites": n_sites,
            "adapter_params": int(adapters), "new_row_params": int(split.new.numel()),
            "trainable_params": int(trainable),
            "total_params": int(sum(p.numel() for p in model.parameters()))}


def frozen_state(model) -> Dict[str, torch.Tensor]:
    """Every non-trainable tensor, by name: what X6 must find bit-identical after training."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
