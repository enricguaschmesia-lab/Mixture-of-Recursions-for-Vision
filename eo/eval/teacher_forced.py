# eo/eval/teacher_forced.py
"""Per-checkpoint teacher-forced evaluation, and recursion-depth interventions.

WHAT THIS ADDS OVER THE TRAINING-TIME EVAL. The trainer logs only per-modality
cross-entropy, and only as a mean of per-batch means. This pass recomputes that
number (so it can be checked against `trainer_state.json`) and adds what the
trainer cannot keep under `prediction_loss_only=True`:

  top-1 / top-5    next-token accuracy per modality -- the "accuracy" axis of
                   the per-checkpoint curves, identical in definition for both
                   arms.
  slot mass        probability the model puts on the RIGHT modality's codebook
                   slot. The teacher-forced analogue of Step 6's off-slot rate.
  per-token arrays CE, correctness and (arm A) the router's depth for every
                   held-out token, so depth can be read against difficulty.

⚠ THE MAIN PASS MIRRORS THE TRAINER'S EVAL LOADER EXACTLY: sequential order,
batch 4, 4 workers. Modality order is drawn from `_get_rng(seed, worker_id,
row)`, and PyTorch hands batch i to worker i % num_workers, so the same loader
settings reproduce the same orders the logged eval saw. Change either and the
per-modality numbers move by more than the effects being measured (a modality
placed first has no cross-modal context).

⚠ TARGET-LAST is the generation setting, teacher-forced: every other modality
as context in registry order, then the target. It is what Step 6/7 sample
from, so it is the low-noise companion of the D3.11 generation metrics.

⚠ DEPTH ALIGNMENT. The logits at position t predict token t+1, and they are
computed from the hidden state of token t -- so the depth that bought a
prediction is depth[t], not depth[t+1]. Per-token arrays here are stored in
that shifted frame.

INTERVENTIONS (arm A only). `DepthOverride` wraps the router and rewrites its
logits so the argmax lands on a chosen depth, by SWAPPING the chosen logit with
the router's own maximum. The softmax probability at the new depth is then
exactly the router's original top probability, so the gate that scales the
recursed update is unchanged and ONLY the number of passes moves. Modes:

  router          as trained (control: must equal a pass with no wrapper)
  force1/2/3      every token at 1, 2 or 3 passes
  perm_modality   the router's depths shuffled among tokens of the SAME
                  modality in the same row: identical per-row, per-modality
                  compute, token-level assignment destroyed
  perm_all        shuffled among all real tokens of the row: identical per-row
                  compute, modality-level assignment destroyed too

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from eo.data.eo_vocab import ID_TO_MODALITY, MODALITIES, MODALITY_TO_ID, PAD_ID
from eo.routing.telemetry import find_mor_layers

MODES = ("router", "force1", "force2", "force3", "perm_modality", "perm_all", "skip")


# ------------------------------------------------------------------ data

def eval_loader(cfg, num_workers: int = 4, batch_size: int = 4):
    """The held-out loader, built exactly as HF's evaluate() builds it."""
    from lm_dataset.load_dataset import load_eval_dataset_from_config
    ds = load_eval_dataset_from_config(cfg)
    if ds is None:
        raise ValueError("this config names no multimodal.eval_split")
    return ds, torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class TargetLast(torch.utils.data.Dataset):
    """Rows re-assembled with every other present modality first, `target` last.

    `sources` (Step 12.0) restricts the context:
      None   every present non-target modality -- today's target-last pass and
             D3.11's prompt layout, unchanged (gate W12 holds it token-identical);
      [S..]  only those, each only if the row carries it, still in registry order;
      []     the target alone: the grid's unconditional row. In distribution,
             because training puts a random modality first in every sequence.
    """

    def __init__(self, base, rows: Sequence[int], target: str,
                 sources: Optional[Sequence[str]] = None):
        self.base, self.rows, self.target = base, [int(r) for r in rows], target
        if sources is not None:
            bad = [s for s in sources if s not in MODALITY_TO_ID or s == target]
            if bad:
                raise ValueError(f"sources {bad}: not a modality, or the target itself")
        self.sources = None if sources is None else set(sources)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from eo.data.eo_vocab import IMAGE_MODALITIES
        from lm_dataset.sequence_assembly import assemble_sequence
        row = self.rows[i]
        present = self.base.present_modalities(row)
        order = [m for m in MODALITIES if m in present and m != self.target
                 and (self.sources is None or m in self.sources)] + [self.target]
        return assemble_sequence(
            chunks=[self.base._load_chunk(m, row) for m in order],
            chunk_modality_ids=[MODALITY_TO_ID[m] for m in order],
            chunk_shufflable=[m in IMAGE_MODALITIES for m in order],
            max_length=self.base.max_length, pad_id=PAD_ID,
            rng=np.random.default_rng(0), shuffle_image_patches=False)


def trim_collate(items):
    """Stack TargetLast items, then cut the batch to its longest real sequence.

    `assemble_sequence` right-pads every row to `max_length` (1,048), so a
    one-source grid prompt (~400 tokens) would pay for 1,048. Attention is
    causal and the padding is trailing, so cutting it changes no real token's
    logits (up to fp16 kernel selection). Used for the grid's subset cells only;
    the `sources=None` cell keeps the default collate, so it reproduces the
    target-last pass exactly.
    """
    batch = torch.utils.data.default_collate(items)
    keep = int(batch["attention_mask"].sum(1).max())
    return {k: v[:, :keep] for k, v in batch.items()}


def stratified_target_rows(base, eval_rows, corpus, target: str, n: int) -> List[int]:
    """Up to n held-out rows carrying `target`, alternating corpora.

    ⚠ The eval rows are sorted majortom-first and majortom never carries S1GRD;
    an unstratified head would silently contain no S1GRD (worklog section 2).
    """
    ok = [int(r) for r in eval_rows if target in base.present_modalities(int(r))]
    by = {c: [r for r in ok if corpus[r] == c] for c in ("majortom", "ssl4eos12")}
    out, i = [], 0
    while len(out) < n and (i < len(by["majortom"]) or i < len(by["ssl4eos12"])):
        for c in ("majortom", "ssl4eos12"):
            if i < len(by[c]) and len(out) < n:
                out.append(by[c][i])
        i += 1
    return sorted(out)


# ------------------------------------------------------------------ routing

class DepthOverride(torch.nn.Module):
    """Wraps `mor_router`; see the module docstring for the modes."""

    MARGIN = 1e-4

    def __init__(self, inner: torch.nn.Module, mode: str = "router", seed: int = 0,
                 alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.inner, self.mode = inner, mode
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.mod_ids: Optional[torch.Tensor] = None      # (B, L), set per batch
        self.att: Optional[torch.Tensor] = None
        self.last_top_prob: Optional[torch.Tensor] = None

    def forward(self, x):
        w = self.inner(x)
        self.last_top_prob = torch.softmax(w.float(), -1).amax(-1).detach()
        if self.mode in ("router", "skip"):        # skip is applied at the layer; see run_pass
            return w
        # The router's ACTUAL choice, computed exactly as the MoR layer does it
        # (softmax * alpha, then topk). ⚠ Not argmax: 3.4% of tokens carry an
        # exact fp16 tie between their top two logits (measured, checkpoint-
        # 16500), and argmax and topk break ties differently.
        a = torch.topk(F.softmax(w, dim=-1) * self.alpha, 1, dim=-1,
                       sorted=False).indices.squeeze(-1)    # (B, L)
        if self.mode.startswith("force"):
            tgt = torch.full_like(a, int(self.mode[-1]) - 1)
        else:
            tgt = a.clone()
            groups = self.mod_ids if self.mode == "perm_modality" else self.att.long()
            for b in range(a.shape[0]):
                for g in torch.unique(groups[b]).tolist():
                    if self.mode == "perm_all" and g == 0:
                        continue                            # padding stays put
                    idx = torch.nonzero(groups[b] == g).squeeze(1)
                    p = torch.randperm(idx.numel(), generator=self.gen).to(idx.device)
                    tgt[b, idx] = a[b, idx[p]]
        # Swap in fp32, and give the target a margin so it wins UNIQUELY: an
        # exact tie would hand the choice back to topk's tie-breaking. The
        # margin moves the gate by ~1e-4 relative.
        w32 = w.float()
        top = w32.gather(-1, a.unsqueeze(-1))
        at_tgt = w32.gather(-1, tgt.unsqueeze(-1))
        w2 = w32.clone()
        w2.scatter_(-1, a.unsqueeze(-1), at_tgt)
        w2.scatter_(-1, tgt.unsqueeze(-1), top + self.MARGIN)
        return w2


def install_override(model, mode: str, seed: int = 0) -> Optional[DepthOverride]:
    """Wrap arm A's router in place (idempotent). None for a model without one."""
    layers = [m for m in find_mor_layers(model) if hasattr(m, "mor_router")]
    if not layers:
        return None
    layer = layers[0]
    if not isinstance(layer.mor_router, DepthOverride):
        layer.mor_router = DepthOverride(layer.mor_router, mode, seed,
                                         alpha=float(layer.cfg.mor.token.get("alpha", 1.0)))
    layer.mor_router.mode = mode
    layer.mor_router.gen.manual_seed(seed)
    return layer.mor_router


# ------------------------------------------------------------------ the pass

def _slot_bounds():
    return {mid: (MODALITIES[n].codebook_offset,
                  MODALITIES[n].codebook_offset + MODALITIES[n].codebook_size)
            for mid, n in ID_TO_MODALITY.items()}


@torch.no_grad()
def run_pass(model, loader, *, device: str = "cuda", override: Optional[DepthOverride] = None,
             keep_tokens: bool = True, per_row_target: Optional[str] = None) -> Dict:
    """One teacher-forced pass. Returns per-modality aggregates (+ per-token arrays).

    Aggregates are token-weighted. `ce_batch_mean` is the trainer's definition
    (mean over batches of the per-batch mean), reported so the logged
    `eval_loss_<mod>` can be reproduced.

    `per_row_target` (Step 12.0) adds `out["rows"]`, reduced per row inside the
    loop, so it works with `trim_collate`'s varying lengths, where
    `keep_tokens` cannot: per row, the number of `per_row_target` tokens
    predicted, their summed CE and top-1 hits, and (MoR) the depth counts
    1/2/3 at the positions PREDICTING target tokens (`d_pred`, the frame of
    `mean_depth_target`), ON target tokens (`d_tgt`) and on every other
    modality's tokens (`d_src`).
    """
    model.eval()
    mor = [m for m in find_mor_layers(model)]
    grabbed: List[torch.Tensor] = []
    handle = None
    skip = override is not None and override.mode == "skip"

    def hook(_m, inputs, out):
        grabbed.append(out.token_expert_indices.detach().to(torch.int8).cpu())
        if skip:
            # ZERO passes: the recursed stack becomes the identity. The control
            # that says how much the shared blocks contribute at all, as opposed
            # to how much the 2nd and 3rd passes add (force1 vs force3).
            out.hidden_state = inputs[0]
        return out

    if mor:
        handle = mor[0].register_forward_hook(hook)
    bounds = _slot_bounds()
    names = dict(ID_TO_MODALITY)
    acc = {n: dict(n=0, ce=0.0, top1=0, top5=0, slot=0.0, bm_sum=0.0, bm_n=0) for n in names.values()}
    toks = {k: [] for k in ("ce", "correct", "rank_ok5", "label_mod", "input_mod", "depth", "gate", "label")}
    rows = {k: [] for k in ("n", "ce_sum", "top1", "d_pred", "d_tgt", "d_src")}
    tid = MODALITY_TO_ID[per_row_target] if per_row_target is not None else None
    try:
        for batch in loader:
            batch = dict(batch)                    # never mutate the caller's batch
            mids = batch.pop("modality_ids")
            labels = batch.pop("labels")
            inputs = {k: v.to(device) for k, v in batch.items()}
            if override is not None:
                override.mod_ids = mids.to(device)
                override.att = inputs["attention_mask"]
            grabbed.clear()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**inputs).logits
            lg = logits[:, :-1].float()
            lab = labels[:, 1:].to(device)
            lmod = mids[:, 1:].to(device)
            valid = lab.ne(-100) & lmod.gt(0)
            safe = lab.clamp_min(0)
            ce = F.cross_entropy(lg.transpose(1, 2), safe, reduction="none")
            top5 = lg.topk(5, dim=-1).indices
            c1 = top5[..., 0].eq(safe)
            c5 = top5.eq(safe.unsqueeze(-1)).any(-1)
            lse = torch.logsumexp(lg, -1)
            slot = torch.zeros_like(ce)
            for mid, (lo, hi) in bounds.items():
                m = valid & lmod.eq(mid)
                if m.any():
                    slot[m] = torch.exp(torch.logsumexp(lg[..., lo:hi][m], -1) - lse[m])
            for mid, n in names.items():
                m = valid & lmod.eq(mid)
                k = int(m.sum())
                if not k:
                    continue
                a = acc[n]
                a["n"] += k
                a["ce"] += float(ce[m].sum()); a["top1"] += int(c1[m].sum())
                a["top5"] += int(c5[m].sum()); a["slot"] += float(slot[m].sum())
                a["bm_sum"] += float(ce[m].mean()); a["bm_n"] += 1
            if tid is not None:
                m = valid & lmod.eq(tid)
                rows["n"].append(m.sum(1).cpu())
                rows["ce_sum"].append((ce * m).sum(1).double().cpu())
                rows["top1"].append((c1 & m).sum(1).cpu())
                if mor:
                    d = grabbed[-1][:, :-1].to(device).long() + 1
                    imod = mids[:, :-1].to(device)
                    for key, sel in (("d_pred", m), ("d_tgt", imod.eq(tid)),
                                     ("d_src", imod.gt(0) & imod.ne(tid))):
                        rows[key].append(torch.stack([(sel & d.eq(k)).sum(1) for k in (1, 2, 3)], 1).cpu())
            if keep_tokens:
                toks["ce"].append(torch.where(valid, ce, torch.zeros_like(ce)).half().cpu())
                toks["correct"].append((c1 & valid).cpu())
                toks["rank_ok5"].append((c5 & valid).cpu())
                toks["label_mod"].append(torch.where(valid, lmod, torch.zeros_like(lmod)).to(torch.int8).cpu())
                toks["input_mod"].append(mids[:, :-1].to(torch.int8))
                toks["label"].append(torch.where(valid, lab, torch.full_like(lab, -1)).to(torch.int32).cpu())
                if mor:
                    toks["depth"].append((grabbed[-1][:, :-1] + 1).to(torch.int8))
                    if override is not None and override.last_top_prob is not None:
                        toks["gate"].append(override.last_top_prob[:, :-1].half().cpu())
            del logits, lg
    finally:
        if handle is not None:
            handle.remove()

    per_mod = {}
    tot_n = tot_ce = 0
    for n, a in acc.items():
        if not a["n"]:
            continue
        per_mod[n] = {"n_tokens": a["n"], "ce": a["ce"] / a["n"],
                      "ce_batch_mean": a["bm_sum"] / a["bm_n"],
                      "top1": a["top1"] / a["n"], "top5": a["top5"] / a["n"],
                      "slot_mass": a["slot"] / a["n"]}
        tot_n += a["n"]; tot_ce += a["ce"]
    out = {"per_modality": per_mod, "ce_all_body_tokens": tot_ce / max(tot_n, 1)}
    if keep_tokens:
        out["tokens"] = {k: torch.cat(v).numpy() for k, v in toks.items() if v}
    if tid is not None:
        out["rows"] = {k: torch.cat(v).numpy() for k, v in rows.items() if v}
    return out
