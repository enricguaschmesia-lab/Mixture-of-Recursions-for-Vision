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
  table           (B0 Step 5, 2026-11-08) each position's depth DRAWN from a
                  per-modality table T[key][d], key = the modality of the token
                  it predicts (the `label_mod` frame, as `only_mod`), from this
                  wrapper's own generator; `set_table` gives the rows. A key
                  mapped to None keeps the router's choice (key 0: positions
                  that predict no body token). Through the same swap, so the
                  gate stays the router's top probability; an optional gate
                  table G[key][d] sets the gate too. "Is the router a
                  per-modality table?": compute follows the table's histogram
  assign          (B0 Step 5) each position at a depth given in advance, one
                  array per row (`set_assignment`, rows in loader order, 0 =
                  the router's choice), e.g. a cross-fitted token-id table
                  drawn offline. Through the same swap, so the gate stays

DEPTH SWEEP (random router, 2026-10-27). `install_random_depth` puts every
token of a RANDOM-router model at one depth d, including depths above the Nr
it was trained with (the tied shared block applied more times). The random
router draws r ~ U(0,1)^Nr per token: depth = argmax(r) + 1, and the gate on
the recursed update is max(r) * alpha. For iid draws the argmax is independent
of the max, so forcing d and drawing the gate as the max of Nr uniforms is
exactly the training-time computation conditional on depth d.

DEPTH SWEEP, LEARNED ROUTER (2026-10-28). `install_learned_depth` does the same
for a LEARNED router through `DepthOverride`'s logit swap, so the gate stays the
router's own top probability, as in force1/2/3. Above Nr the block list is
extended as for the random router, and the router's logits are padded with -inf
up to the new depth (the padded slots get probability 0, so the swapped-in
maximum keeps its probability). `DepthOverride.only_mod` restricts the forcing to
the positions that PREDICT one modality's body tokens (mods[t+1] == m, the frame
of `mean_depth_target`); every other position keeps the router's choice. In a
target-last row the context precedes the target, so under causal attention the
context is computed exactly as routed and only the target's depth moves.

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from eo.data.eo_vocab import ID_TO_MODALITY, MODALITIES, MODALITY_TO_ID, PAD_ID
from eo.routing.telemetry import find_mor_layers, has_learned_router

MODES =("router", "force1", "force2", "force3", "perm_modality", "perm_all", "skip", "table", "assign")


def check_mode(mode: str) -> None:
    """MODES, plus force<d> for any d >= 1 (above Nr through install_learned_depth)."""
    if mode not in MODES and not re.fullmatch(r"force[1-9][0-9]*", mode):
        raise ValueError(f"mode must be one of {MODES} or force<d>, got {mode!r}")


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
        check_mode(mode)
        self.inner, self.mode = inner, mode
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.mod_ids: Optional[torch.Tensor] = None      # (B, L), set per batch
        self.att: Optional[torch.Tensor] = None
        self.last_top_prob: Optional[torch.Tensor] = None
        self.n_out: Optional[int] = None     # pad the logits to this many depths (install_learned_depth)
        self.only_mod: Optional[int] = None  # force only the positions predicting this modality
        self.table_cdf: Optional[Dict[int, Optional[torch.Tensor]]] = None   # `table` mode, set_table
        self.table_gate: Optional[torch.Tensor] = None   # (8, depths), NaN = the router's own gate
        self.last_gate: Optional[torch.Tensor] = None    # the gate applied in `table` mode
        self.assignment: Optional[torch.Tensor] = None   # `assign` mode: (rows, positions) depth, 0 = router
        self.row_ptr = 0                                 # `assign` mode: the next row in loader order

    def set_assignment(self, depth) -> None:
        """The `assign` mode's depths: (rows, positions) in 1..n, 0 keeping the router's choice, rows in
        the order the loader serves them (a sequential loader: dataset positions 0, 1, ...). Position t
        is the router's position t. Restarts at the first row; a pass that runs past the last row, or a
        batch longer than the array's rows, is refused."""
        a = torch.as_tensor(np.asarray(depth), dtype=torch.long)
        if a.dim() != 2 or int(a.min()) < 0:
            raise ValueError(f"assignment must be (rows, positions) with depths >= 0, got {tuple(a.shape)}")
        self.assignment, self.row_ptr = a, 0

    def set_table(self, mix: Dict[int, Optional[Sequence[float]]],
                  gate: Optional[Dict[int, Optional[Sequence[float]]]] = None) -> None:
        """The `table` mode's rows, keyed by modality id (eo_vocab), plus key 0.

        `mix[k]`: probabilities of depths 1..n for the positions that PREDICT a token of
        modality k (key 0: they predict BO, EO or padding); None keeps the router's choice
        there. ⚠ Every key must be present, so a table that forgot a modality is refused here
        rather than silently routed. `gate[k][d-1]`, optional: the gate of a key-k position
        sent to depth d (None for a key: the router's own top probability, as without it).
        Rows are checked against the router's depth count at the first forward.
        """
        need = {0} | set(MODALITY_TO_ID.values())
        if set(mix) != need:
            raise ValueError(f"table keys {sorted(mix)}: need exactly {sorted(need)} "
                             f"(missing {sorted(need - set(mix))}, extra {sorted(set(mix) - need)})")
        cdf = {}
        for k, row in mix.items():
            if row is None:
                cdf[k] = None
                continue
            p = torch.tensor([float(v) for v in row], dtype=torch.float64)
            if (p < 0).any() or not abs(float(p.sum()) - 1.0) < 1e-6:
                raise ValueError(f"table row {k}: {row} is not a probability vector")
            c = torch.cumsum(p / p.sum(), 0)
            c[-1] = 1.0
            cdf[k] = c
        lens = {c.numel() for c in cdf.values() if c is not None}
        if len(lens) > 1:
            raise ValueError(f"table rows of different lengths {sorted(lens)}")
        g = None
        if gate is not None:
            if not set(gate) <= need:
                raise ValueError(f"gate keys {sorted(gate)} outside {sorted(need)}")
            n = lens.pop() if lens else 0
            g = torch.full((max(need) + 1, n), float("nan"), dtype=torch.float32)
            for k, row in gate.items():
                if row is None:
                    continue
                if mix[k] is None:
                    raise ValueError(f"gate row {k} for a key that keeps the router's choice")
                r = torch.tensor([float(v) for v in row], dtype=torch.float32)
                if r.numel() != n or bool(((r < 1.0 / n) | (r > 1.0)).any()):
                    raise ValueError(f"gate row {k}: {row} must hold {n} values in [1/{n}, 1]")
                g[k] = r
        self.table_cdf, self.table_gate = cdf, g

    def forward(self, x):
        w = self.inner(x)
        self.last_top_prob = torch.softmax(w.float(), -1).amax(-1).detach()
        self.last_gate = None
        if self.mode in ("router", "skip"):        # skip is applied at the layer; see run_pass
            return w
        # The router's ACTUAL choice, computed exactly as the MoR layer does it
        # (softmax * alpha, then topk). ⚠ Not argmax: 3.4% of tokens carry an
        # exact fp16 tie between their top two logits (measured, checkpoint-
        # 16500), and argmax and topk break ties differently.
        a = torch.topk(F.softmax(w, dim=-1) * self.alpha, 1, dim=-1,
                       sorted=False).indices.squeeze(-1)    # (B, L)
        key = None
        if self.mode == "table":
            if self.table_cdf is None:
                raise ValueError("table mode: call set_table() first")
            n = self.n_out or w.shape[-1]
            key = torch.zeros_like(a)
            key[:, :-1] = self.mod_ids[:, 1:]                      # the label_mod frame
            known = torch.tensor(sorted(self.table_cdf), device=key.device)
            if not bool(torch.isin(key, known).all()):
                raise ValueError(f"table mode: keys {torch.unique(key).tolist()} outside the table")
            # One draw per position from this wrapper's own generator (CPU, so a pass is
            # reproducible whatever the card), whether or not its key is sampled.
            u = torch.rand(a.shape, generator=self.gen, dtype=torch.float64).to(a.device)
            tgt = a.clone()
            for k, c in self.table_cdf.items():
                if c is None:
                    continue
                if c.numel() != n:
                    raise ValueError(f"table row {k} has {c.numel()} depths, the router {n}")
                sel = key == k
                if bool(sel.any()):
                    tgt[sel] = torch.searchsorted(c.to(a.device), u[sel].contiguous(),
                                                  right=True).clamp_max(n - 1)
        elif self.mode == "assign":
            if self.assignment is None:
                raise ValueError("assign mode: call set_assignment() first")
            B, L = a.shape
            if self.row_ptr + B > self.assignment.shape[0]:
                raise ValueError(f"assign mode: rows {self.row_ptr}..{self.row_ptr + B} beyond the "
                                 f"assignment's {self.assignment.shape[0]}")
            n = self.n_out or w.shape[-1]
            blk = self.assignment[self.row_ptr:self.row_ptr + B, :L].to(a.device)
            if int(blk.max()) > n:
                raise ValueError(f"assign mode: depth {int(blk.max())} above the router's {n}")
            full = torch.zeros_like(a)
            full[:, :blk.shape[1]] = blk
            tgt = torch.where(full > 0, full - 1, a)
            self.row_ptr += B
        elif self.mode.startswith("force"):
            d = int(self.mode[5:])
            if d > (self.n_out or w.shape[-1]):
                raise ValueError(f"{self.mode} above the router's {w.shape[-1]} depths: "
                                 f"use install_learned_depth, which extends the blocks")
            tgt = torch.full_like(a, d - 1)
            if self.only_mod is not None:
                pred = torch.zeros_like(a, dtype=torch.bool)
                pred[:, :-1] = self.mod_ids[:, 1:] == self.only_mod
                tgt = torch.where(pred, tgt, a)
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
        # margin moves the gate by ~1e-4 relative. Padded slots (depth > Nr) hold
        # -inf, so the slot the maximum leaves gets probability 0.
        w32 = w.float()
        if self.n_out is not None and self.n_out > w32.shape[-1]:
            w32 = F.pad(w32, (0, self.n_out - w32.shape[-1]), value=float("-inf"))
        top = w32.gather(-1, a.unsqueeze(-1))
        at_tgt = w32.gather(-1, tgt.unsqueeze(-1))
        w2 = w32.clone()
        w2.scatter_(-1, a.unsqueeze(-1), at_tgt)
        w2.scatter_(-1, tgt.unsqueeze(-1), top + self.MARGIN)
        if key is not None and self.table_gate is not None:
            # The gate from the table: logits whose softmax puts G at the drawn depth and
            # shares the rest evenly, with the same margin so the drawn depth wins uniquely
            # (G >= 1/n always, so it is the maximum). NaN rows keep the swap's logits.
            n = w2.shape[-1]
            g = self.table_gate.to(w2.device)[key, tgt]
            has = ~torch.isnan(g)
            lp = torch.log(((1 - g) / (n - 1)).clamp_min(1e-30)).unsqueeze(-1).expand_as(w2).clone()
            lp.scatter_(-1, tgt.unsqueeze(-1), (torch.log(g.clamp_min(1e-30)) + self.MARGIN).unsqueeze(-1))
            w2 = torch.where(has.unsqueeze(-1), lp, w2)
        if key is not None:
            self.last_gate = torch.softmax(w2, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).detach()
        return w2


def install_override(model, mode: str, seed: int = 0) -> Optional[DepthOverride]:
    """Wrap arm A's router in place (idempotent). None for a model without one."""
    layers = [m for m in find_mor_layers(model) if hasattr(m, "mor_router")]
    if not layers:
        return None
    layer = layers[0]
    check_mode(mode)
    if not isinstance(layer.mor_router, DepthOverride):
        layer.mor_router = DepthOverride(layer.mor_router, mode, seed,
                                         alpha=float(layer.cfg.mor.token.get("alpha", 1.0)))
    layer.mor_router.mode = mode
    layer.mor_router.gen.manual_seed(seed)
    return layer.mor_router


class RandomDepthForce:
    """The random router's draw, rewritten so every token makes `depth` passes.

    Called by the MoR layer on its fresh draw r (B, L, num_recursion). The
    draw is ignored: the gate is drawn from this object's own generator, as
    the max of `n_trained` uniforms, so it does not depend on how many blocks
    the layer holds (an extended block list leaves depth <= Nr unchanged,
    exactly). `depth=None` is the pass-through control: r is returned as is.

    B0 Step 6 (2026-11-08) adds three uses, all drawing every position from this
    object's generator, so that what is not forced is identical across them:
      only_mod   target scope: the positions that PREDICT this modality (mods[t+1],
                 run_pass hands over `mod_ids`) at `depth`; every other position at
                 its own draw (depth argmax, gate max of the same uniforms), the same
                 draw whatever `depth` is. The training-time computation, conditional
                 on the target's depth.
      own_draw   depth None: every position at its own draw from this generator. The
                 paired "as routed" baseline of the target-scope units.
      set_table  depth per position from a per-modality table (key 0: positions that
                 predict no body token; None keeps the position's own draw), from a
                 second generator, so the gates are the forced passes' own: a table
                 one-hot at d everywhere is force<d> bit for bit.
    """

    def __init__(self, depth: Optional[int], n_trained: int, seed: int = 0):
        self.depth, self.n_trained, self.seed = depth, n_trained, seed
        self.gen: Optional[torch.Generator] = None
        self.only_mod: Optional[int] = None
        self.own_draw = False
        self.mod_ids: Optional[torch.Tensor] = None
        self.table_cdf: Optional[Dict[int, Optional[torch.Tensor]]] = None
        self.gen_table: Optional[torch.Generator] = None
        self.last_out: Optional[torch.Tensor] = None

    def reseed(self, seed: Optional[int] = None) -> None:
        self.seed = self.seed if seed is None else seed
        self.gen = None                                 # rebuilt on the draw's device
        self.gen_table = None

    def set_table(self, mix: Dict[int, Optional[Sequence[float]]]) -> None:
        """Per-modality rows, as DepthOverride.set_table (every key 0..7 present, None = own draw)."""
        need = {0} | set(MODALITY_TO_ID.values())
        if set(mix) != need:
            raise ValueError(f"table keys {sorted(mix)}: need exactly {sorted(need)}")
        cdf = {}
        for k, row in mix.items():
            if row is None:
                cdf[k] = None
                continue
            pr = torch.tensor([float(v) for v in row], dtype=torch.float64)
            if (pr < 0).any() or not abs(float(pr.sum()) - 1.0) < 1e-6:
                raise ValueError(f"table row {k}: {row} is not a probability vector")
            c = torch.cumsum(pr / pr.sum(), 0)
            c[-1] = 1.0
            cdf[k] = c
        self.table_cdf, self.gen_table = cdf, None

    def _pred(self, like: torch.Tensor) -> torch.Tensor:
        if self.mod_ids is None:
            raise ValueError("RandomDepthForce: target scope and tables need mod_ids (run_pass sets them)")
        pred = torch.zeros(like.shape[:2], dtype=torch.bool, device=like.device)
        pred[:, :-1] = self.mod_ids[:, 1:].to(like.device) == self.only_mod
        return pred

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        table = self.table_cdf is not None
        if self.depth is None and not self.own_draw and not table:
            return r
        if self.gen is None or self.gen.device != r.device:
            self.gen = torch.Generator(device=r.device).manual_seed(self.seed)
        draws = torch.rand(*r.shape[:2], self.n_trained, generator=self.gen, device=r.device)
        w = draws.amax(-1)
        out = torch.zeros_like(r)
        if self.depth is not None and self.only_mod is None and not table:
            # The forced depth must win uniquely: a zero gate would tie with the zeros.
            out[..., self.depth - 1] = w.clamp_min(1e-4).to(r.dtype)
            self.last_out = out
            return out
        dep = draws.argmax(-1)                          # each position's own draw, 0..n_trained-1
        if table:
            if self.mod_ids is None:
                raise ValueError("RandomDepthForce: a table needs mod_ids (run_pass sets them)")
            key = torch.zeros(r.shape[:2], dtype=torch.long, device=r.device)
            key[:, :-1] = self.mod_ids[:, 1:].to(r.device)
            if self.gen_table is None:
                self.gen_table = torch.Generator(device="cpu").manual_seed(self.seed + 7919)
            u = torch.rand(tuple(key.shape), generator=self.gen_table, dtype=torch.float64).to(r.device)
            for k, c in self.table_cdf.items():
                if c is None:
                    continue
                if c.numel() > r.shape[-1]:
                    raise ValueError(f"table row {k} has {c.numel()} depths, the layer {r.shape[-1]}")
                sel = key == k
                if bool(sel.any()):
                    dep[sel] = torch.searchsorted(c.to(r.device), u[sel].contiguous(), right=True).clamp_max(c.numel() - 1)
        if self.depth is not None:                      # target scope
            dep = torch.where(self._pred(r), torch.full_like(dep, self.depth - 1), dep)
        out.scatter_(-1, dep.unsqueeze(-1), w.clamp_min(1e-4).to(r.dtype).unsqueeze(-1))
        self.last_out = out
        return out


def install_random_depth(model, depth: Optional[int], seed: int = 0,
                         n_blocks: Optional[int] = None) -> RandomDepthForce:
    """Put every token of a RANDOM-router model at `depth` passes (None: the
    trained random draw, through the hook). Idempotent; call again to change.

    Depth above the trained Nr applies the shared block more times than in
    training: the block list is extended with the last trained block, whose
    weights are tied to every other (checked here, tensor by tensor). `n_blocks`
    (default max(Nr, depth)) exists for the control that extending the list
    leaves a depth <= Nr unchanged.

    ⚠ Turns the model's KV cache off (`config.use_cache`). The cache is keyed
    by layer index, and the repeated block reuses the last block's layers, so a
    cached forward appends the extra pass's keys to that block's slot and the
    attention shapes break (measured). A teacher-forced pass has no use for
    the cache, and turning it off leaves the logits bit-identical (controlled
    in the smoke test, worklog 2026-10-27).
    """
    layers = find_mor_layers(model)
    if not layers or has_learned_router(model):
        raise ValueError("install_random_depth needs a MoR model with a RANDOM router "
                         "(mor.rand_router=true); a learned router takes install_learned_depth")
    layer = layers[0]
    n, _ = _extend_blocks(model, layer, depth, n_blocks)
    layer.rand_override = RandomDepthForce(depth, n, seed)
    return layer.rand_override


def install_learned_depth(model, depth: Optional[int],
                          n_blocks: Optional[int] = None) -> DepthOverride:
    """Put every token of a LEARNED-router model at `depth` passes (None: as routed,
    with the block list still extended to `n_blocks`, the pass-through control).
    Idempotent; call again to change. The gate stays the router's own top
    probability (DepthOverride's swap); above Nr the blocks are extended as in
    install_random_depth (same checks, KV cache off) and the router's logits are
    padded. Set `.only_mod` on the returned override to force one modality's
    predicting positions only.
    """
    layers = find_mor_layers(model)
    if not layers or not has_learned_router(model):
        raise ValueError("install_learned_depth needs a MoR model with a LEARNED router; "
                         "a random router takes install_random_depth")
    layer = layers[0]
    _, nb = _extend_blocks(model, layer, depth, n_blocks)
    ov = install_override(model, "router" if depth is None else f"force{depth}")
    ov.n_out, ov.only_mod = nb, None
    return ov


def _extend_blocks(model, layer, depth: Optional[int], n_blocks: Optional[int]):
    """The block list extended with the last trained block to max(Nr, depth) blocks
    (or `n_blocks`), after checking that the trained blocks are tied tensor by
    tensor. Turns the KV cache off (see install_random_depth). Returns (Nr, blocks)."""
    trained = layer.__dict__.get("_eo_trained_blocks")
    if trained is None:
        trained = list(layer.block_list)                # a plain list: not registered as a module
        ref = [p.data_ptr() for p in trained[0].parameters()]
        for blk in trained[1:]:
            if [p.data_ptr() for p in blk.parameters()] != ref:
                raise ValueError("the recursion blocks do not share weights; cannot extend them")
        layer.__dict__["_eo_trained_blocks"] = trained
    n = len(trained)
    if depth is not None and not 1 <= depth:
        raise ValueError(f"depth must be >= 1, got {depth}")
    nb = n_blocks or max(n, depth or n)
    if nb < max(n, depth or n):
        raise ValueError(f"n_blocks {nb} < max(trained {n}, depth {depth})")
    layer.block_list = torch.nn.ModuleList(trained + [trained[-1]] * (nb - n))
    layer.num_recursion = nb
    for c in {id(model.config): model.config, id(model.model.config): model.model.config}.values():
        c.use_cache = False
    return n, nb


# ------------------------------------------------------------------ the pass

def _slot_bounds():
    return {mid: (MODALITIES[n].codebook_offset,
                  MODALITIES[n].codebook_offset + MODALITIES[n].codebook_size)
            for mid, n in ID_TO_MODALITY.items()}


@torch.no_grad()
def run_pass(model, loader, *, device: str = "cuda", override: Optional[DepthOverride] = None,
             keep_tokens: bool = True, per_row_target: Optional[str] = None,
             dtype: Optional[torch.dtype] = torch.float16) -> Dict:
    """One teacher-forced pass. Returns per-modality aggregates (+ per-token arrays).

    Aggregates are token-weighted. `ce_batch_mean` is the trainer's definition
    (mean over batches of the per-batch mean), reported so the logged
    `eval_loss_<mod>` can be reproduced.

    `per_row_target` (Step 12.0) adds `out["rows"]`, reduced per row inside the
    loop, so it works with `trim_collate`'s varying lengths, where
    `keep_tokens` cannot: per row, the number of `per_row_target` tokens
    predicted, their summed CE and top-1 hits, and (MoR) the depth counts
    1..num_recursion at the positions PREDICTING target tokens (`d_pred`, the frame of
    `mean_depth_target`), ON target tokens (`d_tgt`) and on every other
    modality's tokens (`d_src`).

    `dtype` is the autocast dtype: fp16 as the trainer's eval (the default, every result before
    2026-10-29), bf16, or None for none (fp32). ⚠ An LLM-initialised body runs close to fp16's
    range (SmolLM's layer-11 activation, CLAUDE.md), and a frozen one stays there.
    """
    model.eval()
    mor = [m for m in find_mor_layers(model)]
    rand_hook = getattr(mor[0], "rand_override", None) if mor else None   # RandomDepthForce, if installed
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
    toks = {k: [] for k in ("ce", "correct", "rank_ok5", "label_mod", "input_mod", "depth", "gate", "label",
                            "gate_applied")}
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
            if rand_hook is not None:
                rand_hook.mod_ids = mids.to(device)
            grabbed.clear()
            with torch.autocast("cuda", dtype=dtype or torch.float16, enabled=dtype is not None):
                logits = model(**inputs).logits
            lg = logits[:, :-1].float()
            lab = labels[:, 1:].to(device)
            lmod = mids[:, 1:].to(device)
            valid = lab.ne(-100) & lmod.gt(0)
            safe = lab.clamp_min(0)
            ce = F.cross_entropy(lg.transpose(1, 2), safe, reduction="none")
            # ⚠ Rank, NOT torch.topk (2026-10-04). topk over the 87,556-way vocabulary
            #   DEADLOCKS on the TITAN V: deterministically on random logits, and
            #   intermittently on model logits (four eval_checkpoint runs hung with the
            #   card at idle power), while the GTX Titan X runs the same call in 0.06 s.
            #   Every evaluation hang of 2026-10-04 traced here. A label is top-k iff
            #   fewer than k logits are STRICTLY above its own -- the same as topk's
            #   membership except for exact ties at the k-th place (measured in the
            #   worklog, 2026-10-04).
            n_above = (lg > lg.gather(-1, safe.unsqueeze(-1))).sum(-1)
            c1 = n_above.eq(0)
            c5 = n_above.lt(5)
            lse = torch.logsumexp(lg, -1)
            # Masked arithmetic rather than boolean-mask indexing, so the host syncs
            # once per batch instead of ~35 times (2026-10-04; checked against the
            # indexed form to <= 3e-8 relative). The evaluation hangs that first
            # surfaced at a sync in this block were the topk above.
            slot = torch.zeros_like(ce)
            for mid, (lo, hi) in bounds.items():
                part = torch.exp(torch.logsumexp(lg[..., lo:hi], -1) - lse)
                slot = torch.where(valid & lmod.eq(mid), part, slot)
            mid_t = torch.tensor(list(names), device=lmod.device).view(-1, 1, 1)
            M = (valid.unsqueeze(0) & lmod.unsqueeze(0).eq(mid_t)).double()      # (K, B, L)
            per = torch.stack([M.sum((1, 2)), (M * ce.double()).sum((1, 2)),
                               (M * c1.double()).sum((1, 2)), (M * c5.double()).sum((1, 2)),
                               (M * slot.double()).sum((1, 2))], 1).cpu().numpy()
            for j, n in enumerate(names.values()):
                k = int(per[j, 0])
                if not k:
                    continue
                a = acc[n]
                a["n"] += k
                a["ce"] += float(per[j, 1]); a["top1"] += int(round(per[j, 2]))
                a["top5"] += int(round(per[j, 3])); a["slot"] += float(per[j, 4])
                a["bm_sum"] += float(per[j, 1]) / k; a["bm_n"] += 1
            if tid is not None:
                m = valid & lmod.eq(tid)
                rows["n"].append(m.sum(1).cpu())
                rows["ce_sum"].append((ce * m).sum(1).double().cpu())
                rows["top1"].append((c1 & m).sum(1).cpu())
                if mor:
                    d = grabbed[-1][:, :-1].to(device).long() + 1
                    imod = mids[:, :-1].to(device)
                    depths = range(1, int(mor[0].num_recursion) + 1)    # 1..3 for every trained arm
                    for key, sel in (("d_pred", m), ("d_tgt", imod.eq(tid)),
                                     ("d_src", imod.gt(0) & imod.ne(tid))):
                        rows[key].append(torch.stack([(sel & d.eq(k)).sum(1) for k in depths], 1).cpu())
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
                    # `table` mode: the gate actually applied (the router's own unless a gate table
                    # sets it); `gate` above stays the router's top probability in every mode
                    if override is not None and override.last_gate is not None:
                        toks["gate_applied"].append(override.last_gate[:, :-1].half().cpu())
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
