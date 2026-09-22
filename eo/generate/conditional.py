# eo/generate/conditional.py
"""Conditional cross-modal generation on the EO vocabulary (Phase 3 D3.7).

WHAT THIS DOES. Take a HELD-OUT scene, feed its other modalities as context in
a FIXED order, emit `<BO_target>`, and generate until `<EO_target>`. That tests
cross-modal prediction -- the thing that makes this a multimodal model --
rather than unconditional sampling, which at this budget would mostly show
texture (plan Step 6.2).

⚠ WHY NOT `infer.py`. It imports `MODALITIES` and `PAD_ID` from
`lm_dataset.multimodal_vocab_shared_caption_scene_desc` at MODULE SCOPE, builds
prompts from `tok_rgb@256` and `caption`, and decodes through Cosmos on a
square `patch_grid_size`. Pointed at an EO checkpoint it would construct
prompts in the wrong id space and generate confidently meaningless tokens --
the same class of bug as Phase 2's `trainer_pt.py` registry import, and just as
quiet. The model-construction ORDER is reused from it verbatim (build -> share
-> MoR-transform -> load_checkpoint), because getting that order wrong yields a
model that loads with warnings and generates noise.

⚠ FIXED CONTEXT ORDER, even though training shuffles. A varying prompt layout
is one more thing to control for between the arms (plan Step 6.2). The order is
the registry order, which is also the order `eo_vocab.MODALITIES` declares.

⚠ OFF-SLOT RATE IS THE HEADLINE METRIC, and it needs no decoder. Only the
target modality's slot is valid at a given position -- 15,360 ids for the image
modalities, 4,375 for LULC, 6,366 for Coords, out of 87,556. An UNTRAINED model
therefore sits at chance:

    image modalities   1 - 15360/87556 = 82.5% off-slot
    LULC               1 -  4375/87556 = 95.0% off-slot
    Coords             1 -  6366/87556 = 92.7% off-slot

That makes it a control as well as a metric: if a trained model does not beat
those numbers, nothing has been learned about slot structure. LULC is the most
sensitive of the four, which is part of why it leads (plan section 4.5).

Runs in the repo `.venv`, NOT the `mor` conda env -- it never imports
terratorch. Decoding generated ids to pixels is Step 7 and is a SEPARATE
process in the `mor` env; this module's output is raw ids on disk.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from eo.data.eo_vocab import MODALITIES, PAD_ID, TOTAL_VOCAB_SIZE, get_modality

#: Registry order. Context modalities are laid out in this order, target excluded.
FIXED_ORDER: List[str] = list(MODALITIES)


def build_model(config_name: str, checkpoint: Optional[str], device: str = "cuda"):
    """Build an EO model and load a checkpoint, in the order `infer.py` uses.

    ⚠ The order matters: build -> sharing_strategy -> MoR transform ->
    load_checkpoint. Applying the MoR transform after loading, or skipping
    sharing, produces a model that loads with warnings and generates noise.

    ⚠ `config_name` is composed through HYDRA, not read as YAML. The arm
    configs are thin overlays (`defaults: [terramesh_mor_token, _self_]`), so
    parsing the file directly yields a config with no `vocab_size`, no
    `dataset` and no `precision` -- the trap that made preflight refuse a
    correct config in Step 4.

    ⚠ `checkpoint=None` is legitimate and useful: it gives an UNTRAINED model,
    whose off-slot rate is the chance-level control described in the module
    docstring.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from model.sharing_strategy import SHARING_STRATEGY
    from model.util import load_checkpoint, load_model_from_config
    from util.config import preprocess_config

    repo = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo / "conf/pretrain_vision"), version_base=None):
        cfg = compose(config_name=config_name)
    with open_dict(cfg):
        cfg.wandb = False
        cfg.tensorboard = False
        cfg.num_train_steps = 1
        cfg.stop_steps = 1
    cfg = preprocess_config(cfg)

    model = load_model_from_config(cfg)
    if cfg.recursive.get("enable"):
        model, _ = SHARING_STRATEGY[cfg.model](cfg, model)
    if "mor" in cfg and cfg.mor.get("enable"):
        if cfg.mor.type == "expert":
            model.transform_layer_to_mor_expert(cfg)
        else:
            model.transform_layer_to_mor_token(cfg)

    if checkpoint:
        # Handles BOTH formats: arm A writes pytorch_model.bin, arm B
        # model.safetensors (save_safetensors is keyed on weight sharing).
        model = load_checkpoint(model, checkpoint)

    model.to(device).eval()
    return model, cfg


def build_prompt(dataset, row: int, target: str) -> Dict[str, torch.Tensor]:
    """Context modalities in FIXED order, then `<BO_target>`.

    Returns the prompt ids plus the ground-truth target body, so the caller can
    score against it without re-reading the artifact.

    ⚠ Only modalities actually PRESENT in this row contribute. S1GRD and S1RTC
    are exact complements, so every row carries one of them and never both;
    emitting an absent modality as an empty BO/EO pair would teach the prompt a
    structure the training data never had.
    """
    present = set(dataset.present_modalities(row))
    if target not in present:
        raise ValueError(f"row {row} does not carry {target!r}; it has {sorted(present)}")

    chunks: List[torch.Tensor] = []
    context: List[str] = []
    for name in FIXED_ORDER:
        if name == target or name not in present:
            continue
        chunks.append(dataset._load_chunk(name, row))   # [BO, body, EO], offset applied
        context.append(name)

    info = get_modality(target)
    truth = dataset._load_chunk(target, row)[1:-1]      # strip BO/EO -> the body
    prompt = torch.cat(chunks + [torch.tensor([info.bo_id], dtype=torch.long)])
    return {"prompt": prompt, "truth": truth, "context": context}


def _slot_mask(target: str, device) -> torch.Tensor:
    """Boolean mask over the vocabulary: True where an id is INVALID for `target`.

    Used only for the slot-masked decoding arm of 6.3. The unconstrained arm
    must never see this -- the gap between the two is the informative part.
    """
    info = get_modality(target)
    bad = torch.ones(TOTAL_VOCAB_SIZE, dtype=torch.bool, device=device)
    bad[info.codebook_offset: info.codebook_offset + info.codebook_size] = False
    bad[info.eo_id] = False          # the model is allowed, and expected, to stop
    return bad


@torch.no_grad()
def generate(model, prompts: Sequence[torch.Tensor], target: str, *,
             device: str = "cuda", max_new_tokens: Optional[int] = None,
             do_sample: bool = True, temperature: float = 1.0, top_k: int = 0,
             slot_masked: bool = False, use_cache: bool = False,
             seed: Optional[int] = 42) -> List[torch.Tensor]:
    """Generate the target modality for a batch of prompts.

    ⚠ Prompts are LEFT-padded. They differ in length only when a row's context
    differs, but a right-padded batch would put PAD between the prompt and the
    first generated token, which is a layout the model never saw in training.

    ⚠ `use_cache` is NOT IMPLEMENTED and raises rather than being ignored.
    This loop re-runs the full prefix every step, which is O(L^2) per scene but
    unambiguously correct. A KV cache is plan S3.c, still optional: under
    recursion `RecursiveDynamicCache` is easy to get subtly wrong, and a wrong
    cache produces plausible tokens rather than an error. Accepting the flag
    and silently ignoring it would be exactly the kind of dead parameter this
    project keeps finding -- so it fails loudly instead. Measured cost without
    it: ~6 s/scene for a 196-token modality on the TITAN V, which is
    affordable (see the worklog), so this is a deliberate deferral, not a gap.
    """
    from eo.data.eo_vocab import PAD_ID as PAD
    if use_cache:
        raise NotImplementedError(
            "use_cache is not implemented for EO generation. It is plan S3.c and "
            "needs a validated RecursiveDynamicCache; a wrong cache yields plausible "
            "tokens, not an error. Without it generation costs ~6 s/scene, which the "
            "Step 6 measurements show is affordable."
        )
    info = get_modality(target)
    if max_new_tokens is None:
        max_new_tokens = info.tokens_per_sample + 1     # + the EO marker

    if seed is not None:
        torch.manual_seed(seed)

    lengths = [len(p) for p in prompts]
    width = max(lengths)
    ids = torch.full((len(prompts), width), PAD, dtype=torch.long)
    att = torch.zeros((len(prompts), width), dtype=torch.long)
    for i, p in enumerate(prompts):                      # LEFT pad
        ids[i, width - len(p):] = p
        att[i, width - len(p):] = 1
    ids, att = ids.to(device), att.to(device)

    bad = _slot_mask(target, device) if slot_masked else None
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    out: List[List[int]] = [[] for _ in prompts]

    for _ in range(max_new_tokens):
        logits = model(input_ids=ids, attention_mask=att).logits[:, -1, :]
        if bad is not None:
            logits = logits.masked_fill(bad, float("-inf"))
        if do_sample:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            if top_k:
                v, _ = torch.topk(probs, top_k)
                probs = probs.masked_fill(probs < v[:, [-1]], 0.0)
                probs = probs / probs.sum(-1, keepdim=True)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
        else:
            nxt = logits.argmax(-1)

        for i, t in enumerate(nxt.tolist()):
            if not done[i]:
                out[i].append(t)
        done |= nxt.eq(info.eo_id)
        if bool(done.all()):
            break
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        att = torch.cat([att, torch.ones_like(nxt).unsqueeze(1)], dim=1)

    return [torch.tensor(o, dtype=torch.long) for o in out]


def score(generated: Sequence[torch.Tensor], truths: Sequence[torch.Tensor],
          target: str) -> Dict[str, object]:
    """Off-slot rate, BO/EO well-formedness, codebook usage and entropy (6.3, 6.4).

    Every number here is computed WITHOUT decoding, which is the point: it is
    available on day one, it is directly comparable between the arms, and it
    does not inherit the DiVAE decoder's [-1,1] clamp.
    """
    info = get_modality(target)
    lo, hi = info.codebook_offset, info.codebook_offset + info.codebook_size
    expect = info.tokens_per_sample

    n_tok = n_off = 0
    stopped_exactly = stopped_early = never_stopped = 0
    emitted_pad = emitted_other_bo_eo = 0
    local_counts = np.zeros(info.codebook_size, dtype=np.int64)
    truth_counts = np.zeros(info.codebook_size, dtype=np.int64)
    exact_matches, compared = 0, 0

    bo_eo_ids = set()
    for m in MODALITIES.values():
        bo_eo_ids.add(m.bo_id); bo_eo_ids.add(m.eo_id)

    for gen, truth in zip(generated, truths):
        g = gen.tolist()
        body = g[:-1] if g and g[-1] == info.eo_id else g
        if g and g[-1] == info.eo_id:
            if len(body) == expect:
                stopped_exactly += 1
            else:
                stopped_early += 1
        else:
            never_stopped += 1

        for t in body:
            n_tok += 1
            if lo <= t < hi:
                local_counts[t - lo] += 1
            else:
                n_off += 1
                if t == PAD_ID:
                    emitted_pad += 1
                elif t in bo_eo_ids:
                    emitted_other_bo_eo += 1

        tl = truth.tolist()
        for t in tl:
            if lo <= t < hi:
                truth_counts[t - lo] += 1
        # ⚠ Compare over the OVERLAP, not only when the lengths match. An
        # untrained model never emits EO, so its body runs to max_new_tokens
        # (197) against a 196-token truth and a length-equality guard would
        # report accuracy as nan -- exactly when the baseline is most wanted.
        k = min(len(body), len(tl))
        if k:
            compared += k
            exact_matches += sum(1 for a, b in zip(body[:k], tl[:k]) if a == b)

    def ent(counts):
        tot = counts.sum()
        if tot == 0:
            return 0.0
        p = counts[counts > 0] / tot
        return float(-(p * np.log(p)).sum())

    n = len(generated)
    return {
        "target": target,
        "n_scenes": n,
        "tokens_emitted": n_tok,
        "off_slot_rate": (n_off / n_tok) if n_tok else float("nan"),
        "off_slot_chance": 1.0 - info.codebook_size / TOTAL_VOCAB_SIZE,
        "emitted_pad": emitted_pad,
        "emitted_foreign_bo_eo": emitted_other_bo_eo,
        "stopped_at_exact_length": stopped_exactly,
        "stopped_early_or_late": stopped_early,
        "never_emitted_eo": never_stopped,
        "codes_used": int((local_counts > 0).sum()),
        "codes_used_truth": int((truth_counts > 0).sum()),
        "codebook_size": info.codebook_size,
        "entropy_nats": ent(local_counts),
        "entropy_nats_truth": ent(truth_counts),
        "token_accuracy": (exact_matches / compared) if compared else float("nan"),
    }


def save(out_dir, target: str, rows, generated, stats: Dict, provenance: Dict):
    """Raw ids + provenance on disk (D3.7). Large outputs live on /data."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    width = max((len(g) for g in generated), default=0)
    arr = np.full((len(generated), width), PAD_ID, dtype=np.int32)
    for i, g in enumerate(generated):
        arr[i, : len(g)] = g.numpy()
    with open(out / f"generated_{target}.npy", "wb") as fh:
        np.save(fh, arr)
    np.save(out / f"rows_{target}.npy", np.asarray(rows, dtype=np.int64))
    (out / f"stats_{target}.json").write_text(
        json.dumps({"stats": stats, "provenance": provenance}, indent=2) + "\n",
        encoding="utf-8",
    )
    return out
