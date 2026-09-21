# lm_dataset/sequence_assembly.py
"""Shared sequence finalization for multimodal token datasets.

Lifted verbatim out of MultimodalTokenizedDataset.__getitem__ so that the CLEVR
dataset and the TerraMesh EO dataset (eo/mor_data/terramesh_token_dataset.py)
cannot drift apart in how they pad, mask, truncate and position tokens. Phases 3–4
compare EO routing behaviour against the CLEVR results, and a subtle difference
here would confound that comparison at the source.

Deliberately registry-agnostic: the caller resolves modality ids and decides
which chunks are shufflable, so the two datasets can use different unified
vocabularies. The only things this module knows about are chunk boundaries and
the five-key output contract.

⚠ The returned dict's keys are a closed set. MoRTrainer.compute_loss pops
'modality_ids' and forwards everything else into model(**inputs), so an extra
key here becomes an unexpected keyword argument at the model's forward().
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def assemble_sequence(
    chunks: Sequence[torch.Tensor],
    chunk_modality_ids: Sequence[int],
    chunk_shufflable: Sequence[bool],
    max_length: int,
    pad_id: int,
    rng: np.random.Generator,
    shuffle_image_patches: bool = False,
) -> Dict[str, torch.Tensor]:
    """Concatenate BO/EO-wrapped modality chunks into one fixed-length sample.

    Args:
        chunks: per-modality token tensors, each already wrapped as
            [BO, body..., EO] and already shifted into the unified vocabulary.
            Order is the final sequence order; the caller applies any shuffling.
        chunk_modality_ids: per-chunk modality id for per-modality loss logging.
            0 is reserved for "no modality" and is what BO/EO/padding get.
        chunk_shufflable: per-chunk flag -- True if the chunk's body is an image
            patch grid whose patches may be permuted when shuffle_image_patches
            is set. Text bodies must be False.
        max_length: output length. Sequences longer than this are truncated,
            shorter ones padded with pad_id.
        pad_id: unified-vocabulary padding id.
        rng: per-sample generator. Consumed only when shuffle_image_patches is
            set; callers relying on reproducible streams must pass the same
            generator they used for ordering decisions, at the same point.
        shuffle_image_patches: permute image-patch bodies in place.

    Returns:
        dict with exactly input_ids, attention_mask, labels, position_ids,
        modality_ids -- each (max_length,) torch.long.
    """
    if not (len(chunks) == len(chunk_modality_ids) == len(chunk_shufflable)):
        raise ValueError(
            f"chunks, chunk_modality_ids and chunk_shufflable must align: "
            f"got {len(chunks)}, {len(chunk_modality_ids)}, {len(chunk_shufflable)}"
        )
    if len(chunks) == 0:
        raise ValueError("assemble_sequence needs at least one chunk.")

    # Track which intra-sequence ranges are image-token bodies (exclude BO/EO),
    # so we can permute them while keeping their canonical positions in RoPE.
    # Also build a per-token modality id (0 = pad/BO/EO, >0 = body of a modality)
    # used downstream for per-modality loss logging.
    body_ranges: List[Tuple[int, int]] = []
    per_token_modality_ids: List[torch.Tensor] = []
    cursor = 0
    for chunk, mid, shufflable in zip(chunks, chunk_modality_ids, chunk_shufflable):
        ids = torch.zeros(chunk.shape[0], dtype=torch.long)
        if chunk.shape[0] > 2:
            ids[1:-1] = mid
        per_token_modality_ids.append(ids)
        if shuffle_image_patches and shufflable and chunk.shape[0] > 2:
            body_ranges.append((cursor + 1, cursor + chunk.shape[0] - 1))
        cursor += chunk.shape[0]
    seq = torch.cat(list(chunks), dim=0)
    seq_modality_ids = torch.cat(per_token_modality_ids, dim=0)

    # Truncate or pad to max_length.
    if seq.shape[0] > max_length:
        seq = seq[:max_length]
        seq_modality_ids = seq_modality_ids[:max_length]
        body_ranges = [(s, min(e, max_length)) for s, e in body_ranges if s < max_length]
        body_ranges = [(s, e) for s, e in body_ranges if e - s >= 2]

    input_ids = torch.full((max_length,), pad_id, dtype=torch.long)
    input_ids[: seq.shape[0]] = seq
    modality_ids = torch.zeros(max_length, dtype=torch.long)
    modality_ids[: seq_modality_ids.shape[0]] = seq_modality_ids

    attention_mask = torch.zeros(max_length, dtype=torch.long)
    attention_mask[: seq.shape[0]] = 1

    position_ids = torch.arange(max_length, dtype=torch.long)

    # Permute image-patch bodies in-place; position_ids carries the pre-shuffle
    # index so RoPE encodes canonical raster position, not shuffled sequence position.
    for s, e in body_ranges:
        sigma = torch.from_numpy(rng.permutation(e - s)).long() + s
        input_ids[s:e] = input_ids[sigma]
        position_ids[s:e] = sigma

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'position_ids': position_ids,
        'modality_ids': modality_ids,
    }
