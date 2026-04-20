# lm_dataset/multimodal_vocab.py
"""
Unified vocabulary layout for multimodal tokenized data.

Token IDs are allocated in three zones:
  1. Per-modality codebooks (each shifted by a cumulative offset) we decided to allocate a fixed-size codebook for each modality.
  # We belive that this is good since in this way we can add new modalities without changing the vocabolary structure. 
  2. Per-modality special tokens: <BO_mod>, <EO_mod>
  3. Global special tokens: <PAD>

All five CLEVR-relevant modalities have slots reserved regardless of which
are active in a given run, so you can add modalities later without
shifting the token IDs of already-trained modalities.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ModalityInfo:
    name: str
    codebook_size: int
    file_ext: str
    data_type: str
    codebook_group: str           # NEW: modalities in the same group share codebook_offset
    codebook_offset: int = 0
    bo_id: int = 0
    eo_id: int = 0


# (name, codebook_size, file_ext, data_type, codebook_group)
# Modalities with the same `codebook_group` share a codebook.
# The first entry in a group defines the group's codebook_size; others must set 0.
_MODALITY_REGISTRY: List[Tuple[str, int, str, str, str]] = [
    ('tok_rgb@256',    64000, '.npy',  'tokens', 'rgb'),
    ('tok_depth@256',  64000, '.npy',  'tokens', 'depth'),
    ('tok_normal@256', 64000, '.npy',  'tokens', 'normal'),
    ('caption',        50260, '.json', 'text',   'text'),   # defines the 'text' codebook
    ('scene_desc',     0,     '.json', 'text',   'text'),   # shares it
]


def build_vocab() -> Tuple[Dict[str, ModalityInfo], int, int]:
    """
    Build the unified vocab layout.

    Layout:
      [0, sum_unique_codebooks)               per-group codebooks
      [sum_unique_codebooks, +2M)             BO/EO pairs, 2 per modality
      sum_unique_codebooks + 2M               <PAD>
      total_vocab_size = sum_unique_codebooks + 2M + 1

    Modalities in the same `codebook_group` share a codebook_offset, so their
    token IDs collide by design (e.g., GPT-2 id 1212 means the same embedding
    row whether it came from a caption or a scene description). They still
    get distinct BO/EO delimiters.
    """
    modalities: Dict[str, ModalityInfo] = {}

    # First pass: assign codebook_offset per group.
    group_offsets: Dict[str, int] = {}
    group_sizes: Dict[str, int] = {}
    running_offset = 0
    for name, codebook_size, file_ext, data_type, group in _MODALITY_REGISTRY:
        if group not in group_offsets:
            # First modality in this group defines the codebook.
            if codebook_size <= 0:
                raise ValueError(
                    f"Modality '{name}' is the first in group '{group}' "
                    f"and must declare a positive codebook_size."
                )
            group_offsets[group] = running_offset
            group_sizes[group] = codebook_size
            running_offset += codebook_size
        else:
            # Subsequent modalities in the group must not re-declare a size.
            if codebook_size != 0:
                raise ValueError(
                    f"Modality '{name}' shares group '{group}' with a prior modality; "
                    f"it must declare codebook_size=0, got {codebook_size}."
                )

        modalities[name] = ModalityInfo(
            name=name,
            codebook_size=group_sizes[group],
            file_ext=file_ext,
            data_type=data_type,
            codebook_group=group,
            codebook_offset=group_offsets[group],
        )

    sum_unique_codebooks = running_offset

    # Second pass: assign BO/EO per modality (always unique).
    for i, name in enumerate(modalities):
        modalities[name].bo_id = sum_unique_codebooks + 2 * i
        modalities[name].eo_id = sum_unique_codebooks + 2 * i + 1

    num_modalities = len(modalities)
    pad_id = sum_unique_codebooks + 2 * num_modalities
    total_vocab_size = pad_id + 1

    return modalities, pad_id, total_vocab_size


MODALITIES, PAD_ID, TOTAL_VOCAB_SIZE = build_vocab()


def get_modality(name: str) -> ModalityInfo:
    if name not in MODALITIES:
        raise KeyError(
            f"Unknown modality '{name}'. Known: {list(MODALITIES.keys())}. "
            f"To add a new modality, append to _MODALITY_REGISTRY in multimodal_vocab.py."
        )
    return MODALITIES[name]