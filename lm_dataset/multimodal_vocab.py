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
    codebook_size: int          # size of the modality's own vocab
    file_ext: str               # .npy or .json
    data_type: str              # 'tokens' or 'text'
    codebook_offset: int = 0    # start of this modality's codebook in the unified vocab
    bo_id: int = 0              # <BO_mod> token id
    eo_id: int = 0              # <EO_mod> token id


# Order here is permanent. Do not reorder.
# Adding new modalities: APPEND only.
_MODALITY_REGISTRY: List[Tuple[str, int, str, str]] = [
    # (name, codebook_size, file_ext, data_type)
    ('tok_rgb@256',    64000, '.npy',  'tokens'),
    ('tok_depth@256',  64000, '.npy',  'tokens'),
    ('tok_normal@256', 64000, '.npy',  'tokens'),
    ('caption',        50264, '.json', 'text'),   # GPT-2 BPE + specials, rounded
    ('scene_desc',     50264, '.json', 'text'),   # same tokenizer, separate slot
]


def build_vocab() -> Tuple[Dict[str, ModalityInfo], int, int]:
    """
    Build the full vocab layout over all registered modalities.
    
    Layout:
      total_vocab_size = sum_codebooks + 2*M + 1
      We have this number since we have the sum_codebooks, the BOS and EOS for modality, we have also the PAD token.
    Returns:
        modalities: dict mapping modality name -> ModalityInfo (populated)
        pad_id: id of the <PAD> token
        total_vocab_size: full vocab size
    """
    modalities: Dict[str, ModalityInfo] = {}
    offset = 0
    for name, codebook_size, file_ext, data_type in _MODALITY_REGISTRY:
        modalities[name] = ModalityInfo(
            name=name,
            codebook_size=codebook_size,
            file_ext=file_ext,
            data_type=data_type,
            codebook_offset=offset,
        )
        offset += codebook_size

    sum_codebooks = offset
    for i, name in enumerate(modalities):
        modalities[name].bo_id = sum_codebooks + 2 * i
        modalities[name].eo_id = sum_codebooks + 2 * i + 1

    num_modalities = len(modalities)
    pad_id = sum_codebooks + 2 * num_modalities
    total_vocab_size = pad_id + 1

    return modalities, pad_id, total_vocab_size


# Module-level singletons — build once, use everywhere.
MODALITIES, PAD_ID, TOTAL_VOCAB_SIZE = build_vocab()


def get_modality(name: str) -> ModalityInfo:
    if name not in MODALITIES:
        raise KeyError(
            f"Unknown modality '{name}'. Known: {list(MODALITIES.keys())}. "
            f"To add a new modality, append to _MODALITY_REGISTRY in multimodal_vocab.py."
        )
    return MODALITIES[name]