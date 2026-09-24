# eo/routing/telemetry.py
"""Capturing and reading MoR recursion depth (Phase 3 D3.9).

WHAT THIS ANSWERS. Arm A shares 9 middle transformer blocks and applies them
1, 2 or 3 times per token; a router picks how many. The research question is
whether that choice tracks CONTENT -- modality, scene complexity -- or whether
it is really a function of POSITION in the sequence, which is what killed
expert-choice routing on CLEVR. The aggregate `balancing_entropy` logged during
training cannot tell those apart: a router keyed purely on position produces
exactly the same entropy curve as one keyed on content.

HOW DEPTH IS DEFINED. `MoRLlamaDecoderLayer.forward` returns
`token_expert_indices`, shape (batch, seq_len), values in {0, 1, 2}. From
`model/mor_model/token_choice_router.py:21`: "A token assigned to depth d is
processed by blocks 0..d and then exits." So the token makes **d + 1** passes
through the shared 9-block stack. We store d + 1 and call it `depth`.

⚠ RANDOM MODALITY ORDER IS THE POINT, not an inherited default. Training
shuffles modality order per sample, and this pass keeps that. With a FIXED
order every modality would sit at the same sequence positions in every sample,
so "S1GRD is routed deeper" and "tokens at positions 400-600 are routed deeper"
would be the same statement and could never be separated. Random order
decorrelates the two, which is what makes `depth_by_position` interpretable
beside `depth_by_modality`.

⚠ DEPTH IS SHAPED BY THE BALANCING LOSS, NOT BY CONTENT ALONE (plan 8.5). The
run trains with `mor.token.balancing: loss`, `coeff: 0.1`, `bal_warmup_step: 0`,
which actively pushes the depth marginal towards uniform. Any content-driven
separation found here happens DESPITE that pressure; a null result is
correspondingly harder to read. Every figure must carry these numbers.

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from eo.data.eo_vocab import ID_TO_MODALITY, MODALITY_TO_ID
from eo.terramesh_tok import contract as C

#: Transformer layers outside the recursed stack: one before, one after.
#: Verified against the checkpoint: model.layers.0 and model.layers.2 carry
#: their own weights, model.layers.1 is the MoR wrapper holding block_list.
N_OUTER_LAYERS = 2


def find_mor_layers(model) -> List[torch.nn.Module]:
    """Every recursion layer. Empty means this is not a MoR model.

    ⚠ Checked by STRUCTURE, not by name. `CLAUDE.md`: a recursive-but-unrouted
    model keeps ordinary `model.layers.N.*` names and looks exactly like
    vanilla, so a name test would silently accept the wrong arm.

    ⚠ The marker is `block_list`, NOT `mor_router`. `CLAUDE.md` records that
    `block_list` is created by the MoR transform and by nothing else, which is
    exactly the property wanted here. `mor_router` is the wrong marker because
    `mor.rand_router=true` builds the same recursion structure with NO router
    module at all -- and that configuration is Step 8's own control, so keying
    on the router would make the control unrunnable. Found by running it.
    """
    inner = getattr(getattr(model, "model", model), "layers", None)
    if inner is None:
        return []
    return [m for m in inner if hasattr(m, "block_list")]


def has_learned_router(model) -> bool:
    """True when depth comes from a trained router rather than from `torch.rand`."""
    return any(hasattr(m, "mor_router") for m in find_mor_layers(model))


def describe_geometry(model) -> Dict[str, int]:
    """Recursion geometry, read off the built model rather than the config."""
    layers = find_mor_layers(model)
    if not layers:
        raise ValueError(
            "no MoR router found. This is arm B (vanilla) or a recursive-without-"
            "router model; neither has a depth to report."
        )
    if len(layers) != 1:
        raise ValueError(f"expected exactly one MoR layer, found {len(layers)}")
    bl = layers[0].block_list
    n_recursion = len(bl)
    blocks_per_step = len(bl[0])
    return {
        "n_recursion": n_recursion,
        "blocks_per_recursion_step": blocks_per_step,
        "learned_router": has_learned_router(model),
        "outer_layers": N_OUTER_LAYERS,
        # what arm B applies to every token, for the compute comparison
        "layers_if_all_max_depth": N_OUTER_LAYERS + blocks_per_step * n_recursion,
    }


@dataclass
class Capture:
    """Raw per-token telemetry over a set of held-out rows."""
    depth: np.ndarray          # (n_rows, seq_len) int8, 1..n_recursion; 0 where masked
    modality_ids: np.ndarray   # (n_rows, seq_len) int8, 0 = pad/BO/EO
    rows: np.ndarray           # (n_rows,) int64, row index into tok_index.parquet
    geometry: Dict[str, int]

    @property
    def valid(self) -> np.ndarray:
        """Body tokens only: excludes padding and the BO/EO markers."""
        return self.modality_ids > 0


@torch.no_grad()
def capture_depths(model, dataset, rows: Sequence[int], *, batch_size: int = 4,
                   device: str = "cuda") -> Capture:
    """Run held-out rows through the model and record each token's depth.

    ⚠ OFFLINE, and not a hook on the training run's `evaluate()` (plan 8.6).
    The hooks would fire there, but `prediction_loss_only=True` means nothing
    but scalars survives, and paying for capture on every eval pass for the
    whole phase to build a figure made once is the wrong trade.

    ⚠ `modality_ids` is popped before the forward. The batch dict is a closed
    five-key set and `MoRTrainer.compute_loss` pops it for exactly this reason;
    passing it through would hit `forward()` as an unexpected kwarg.
    """
    geom = describe_geometry(model)
    layer = find_mor_layers(model)[0]

    grabbed: List[torch.Tensor] = []

    def hook(_module, _inputs, output):
        idx = getattr(output, "token_expert_indices", None)
        if idx is None:
            raise RuntimeError(
                "the MoR layer returned no token_expert_indices; the router "
                "did not run and any depth read from this pass is fiction.")
        grabbed.append(idx.detach().to(torch.int16).cpu())

    handle = layer.register_forward_hook(hook)
    model.eval()
    try:
        depths, mods = [], []
        for i in range(0, len(rows), batch_size):
            chunk = [dataset[int(r)] for r in rows[i:i + batch_size]]
            batch = {k: torch.stack([c[k] for c in chunk]) for k in chunk[0]}
            mod_ids = batch.pop("modality_ids")
            batch.pop("labels", None)
            grabbed.clear()
            model(**{k: v.to(device) for k, v in batch.items()})
            if len(grabbed) != 1:
                raise RuntimeError(
                    f"expected 1 router capture per forward, got {len(grabbed)}")
            depths.append(grabbed[0].numpy())
            mods.append(mod_ids.numpy())
    finally:
        handle.remove()

    depth = np.concatenate(depths).astype(np.int16) + 1      # d -> passes
    modality_ids = np.concatenate(mods).astype(np.int16)
    depth = np.where(modality_ids > 0, depth, 0).astype(np.int8)
    return Capture(depth=depth, modality_ids=modality_ids.astype(np.int8),
                   rows=np.asarray(rows, dtype=np.int64), geometry=geom)


# ------------------------------------------------------------------ analysis

def depth_by_modality(cap: Capture) -> Dict[str, Dict]:
    """Depth histogram, mean and share per modality (plan 8.2)."""
    n_rec = cap.geometry["n_recursion"]
    out = {}
    for name, mid in MODALITY_TO_ID.items():
        sel = cap.modality_ids == mid
        n = int(sel.sum())
        if n == 0:
            continue                      # absent from this sample; 8.7's trap
        d = cap.depth[sel]
        hist = np.bincount(d, minlength=n_rec + 1)[1:]
        out[name] = {
            "n_tokens": n,
            "mean_depth": round(float(d.mean()), 4),
            "hist": [int(x) for x in hist],
            "share": [round(float(x), 4) for x in hist / n],
        }
    return out


def depth_by_position(cap: Capture, n_bins: int = 16) -> Dict:
    """Mean depth against absolute position, averaged over modalities.

    The companion to `depth_by_modality`. If depth varied with position and not
    with content, this curve would be structured and the modality table flat.
    Only interpretable because modality order is randomised per sample.
    """
    pos = np.broadcast_to(np.arange(cap.depth.shape[1]), cap.depth.shape)
    v = cap.valid
    p, d = pos[v], cap.depth[v]
    edges = np.linspace(0, cap.depth.shape[1], n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    means, counts = [], []
    for b in range(n_bins):
        m = idx == b
        counts.append(int(m.sum()))
        means.append(round(float(d[m].mean()), 4) if m.any() else None)
    return {"bin_edges": [float(e) for e in edges], "mean_depth": means,
            "n_tokens": counts}


def variance_explained(cap: Capture, n_bins: int = 16) -> Dict:
    """How much of the depth variance modality explains, against position.

    ⚠ This is the number the research question turns on, and it is a
    DESCRIPTIVE decomposition, not a causal one -- modality and position are
    decorrelated by the random ordering but not orthogonalised, so the two
    shares need not sum to anything in particular.
    """
    v = cap.valid
    d = cap.depth[v].astype(np.float64)
    total = d.var()
    if total == 0:
        return {"total_variance": 0.0, "by_modality": 0.0, "by_position": 0.0}

    def eta_sq(groups):
        num = 0.0
        for g in np.unique(groups):
            m = groups == g
            num += m.sum() * (d[m].mean() - d.mean()) ** 2
        return float(num / (len(d) * total))

    mod = cap.modality_ids[v]
    pos = np.broadcast_to(np.arange(cap.depth.shape[1]), cap.depth.shape)[v]
    pbin = np.clip(pos * n_bins // cap.depth.shape[1], 0, n_bins - 1)
    return {"total_variance": round(float(total), 6),
            "by_modality": round(eta_sq(mod), 6),
            "by_position": round(eta_sq(pbin), 6)}


def spatial_maps(cap: Capture) -> Dict[str, np.ndarray]:
    """Mean 14x14 depth map per image modality (plan 8.3).

    Valid only because `shuffle_image_patches` is false, so a modality's body
    is its patch grid in row-major order: token k <-> patch (k//14, k%14), per
    `contract.FLATTEN_ORDER`.
    """
    g, n_tok = C.GRID, C.TOKENS_PER_SAMPLE
    acc: Dict[str, List[np.ndarray]] = {}
    for r in range(cap.depth.shape[0]):
        ids, dep = cap.modality_ids[r], cap.depth[r]
        for name, mid in MODALITY_TO_ID.items():
            where = np.flatnonzero(ids == mid)
            if where.size != n_tok:
                continue                  # Coords (3 tokens), or absent
            if where[-1] - where[0] != n_tok - 1:
                raise RuntimeError(f"row {cap.rows[r]}: {name} body is not contiguous")
            acc.setdefault(name, []).append(dep[where].reshape(g, g))
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def compute_accounting(cap: Capture) -> Dict:
    """Layer applications per token: the honest compute figure (plan 8.5a).

    ⚠ WHY THIS EXISTS. The Trainer's `total_flos` is exactly
    `6 x tokens x non_embedding_parameters` -- a PARAMETER proxy, blind to
    recursion and routing. And `1 - 39.61/102.66 = 61.4%` reproduces the
    published "MoR ~61% lower FLOPs" figure exactly, which is evidence that the
    claim is that same parameter ratio. Parameters are storage. Compute is how
    many layers each token is actually pushed through, which is what this
    counts.

    ⚠ WHAT IT EXCLUDES, deliberately: embeddings and `lm_head` (identical in
    both arms and tied), and attention's quadratic term. The quadratic term
    FAVOURS MoR -- deeper recursion steps run over fewer selected tokens -- so
    the linear count below is a CONSERVATIVE estimate of MoR's saving.
    """
    g = cap.geometry
    outer, per_step, n_rec = g["outer_layers"], g["blocks_per_recursion_step"], g["n_recursion"]
    d = cap.depth[cap.valid].astype(np.float64)
    per_token = outer + per_step * d
    vanilla = outer + per_step * n_rec
    return {
        "layers_per_token_mor": round(float(per_token.mean()), 4),
        "layers_per_token_vanilla": int(vanilla),
        "ratio_mor_over_vanilla": round(float(per_token.mean() / vanilla), 4),
        "compute_saving": round(1 - float(per_token.mean() / vanilla), 4),
        "mean_depth": round(float(d.mean()), 4),
        "note": ("layer applications per body token; excludes embeddings, lm_head "
                 "and attention's quadratic term, which favours MoR"),
    }

def _eta_sq(vals: np.ndarray, groups: np.ndarray) -> float:
    """Share of `vals` variance lying between groups rather than within them."""
    total = vals.var()
    if total == 0:
        return 0.0
    mean = vals.mean()
    num = 0.0
    for g in np.unique(groups):
        sel = vals[groups == g]
        num += sel.size * (sel.mean() - mean) ** 2
    return float(num / (vals.size * total))


def within_modality_decomposition(cap: Capture) -> Dict[str, Dict]:
    """Inside ONE modality, does depth follow the scene or the patch position?

    ⚠ THIS IS THE QUESTION `depth_by_modality` CANNOT ANSWER. Learning that the
    router sends every S1RTC token deep tells us it reads the modality tag; it
    says nothing about whether it reads the image. Splitting the remaining
    variance two ways does:

      SCENE    -> depth differs between held-out scenes: the router is
                  responding to what is in the picture.
      POSITION -> depth differs by where the patch sits in the 14x14 grid,
                  the same way in every scene: layout, not content. This is
                  the CLEVR expert-choice failure mode, scoped to one modality.

    ⚠ A non-flat 14x14 mean map is NOT by itself evidence of content. Averaging
    over scenes turns genuine per-scene variation into position-wise means that
    only LOOK positional, which is why position is measured against scene here
    rather than eyeballed off the heatmap.

    ⚠ Scene-level attribution is coarser than "depth tracks patch complexity".
    It shows some scenes are routed deeper than others; whatever within-scene
    patch-to-patch variation remains is the residual and is not attributed here.

    ⚠⚠ `by_scene` MUST BE READ AGAINST THE UNTRAINED MODEL, NEVER AGAINST ZERO.
    A randomly-initialised router applied to real hidden states already scores
    **0.19-0.31** here, because any function of the input varies with the input.
    The near-zero reference is `mor.rand_router=true` (~0.005), which draws
    depth from `torch.rand` and cannot see the input at all -- a different
    control answering a different question. Measured 2026-09-24 at
    checkpoint-4000: only LULC (0.42 vs 0.22 untrained) and weakly DEM
    (0.37 vs 0.31) beat the untrained baseline; S2L2A, S1GRD, S1RTC and NDVI
    all score BELOW it, because a near-deterministic per-modality policy
    removes within-modality variation by construction. Read against zero, the
    same numbers say the opposite. This was misread once before it was caught.
    """
    n_tok = C.TOKENS_PER_SAMPLE
    out: Dict[str, Dict] = {}
    for name, mid in MODALITY_TO_ID.items():
        bodies, scenes = [], []
        for r in range(cap.depth.shape[0]):
            w = np.flatnonzero(cap.modality_ids[r] == mid)
            if w.size == n_tok:
                bodies.append(cap.depth[r][w])
                scenes.append(np.full(n_tok, r))
        if len(bodies) < 5:
            continue
        v = np.concatenate(bodies).astype(np.float64)
        scene = np.concatenate(scenes)
        pos = np.tile(np.arange(n_tok), len(bodies))

        # Does the mean 14x14 map hold more spread than sampling noise allows?
        stack = np.stack(bodies).astype(np.float64)
        floor = stack.std() / np.sqrt(len(bodies))
        map_sd = stack.mean(0).std()
        out[name] = {
            "n_scenes": len(bodies),
            "depth_sd": round(float(v.std()), 4),
            "by_scene": round(_eta_sq(v, scene), 4),
            "by_patch_position": round(_eta_sq(v, pos), 4),
            "spatial_map_sd": round(float(map_sd), 4),
            "spatial_noise_floor": round(float(floor), 4),
            "spatial_signal_to_noise": round(float(map_sd / floor), 2) if floor else None,
        }
    return out
