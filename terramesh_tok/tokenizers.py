"""Building and caching the six TerraMind FSQ tokenizers."""
from __future__ import annotations
import os
import functools

os.environ.setdefault("HF_HOME", "/data/enric/hf")

import torch

from . import contract as C

DEVICE = "cuda:0"   # torch cuda:0 == TITAN V (sm_70). nvidia-smi numbers it 1.


@functools.lru_cache(maxsize=8)
def build(modality: str, device: str = DEVICE):
    """Load a pretrained tokenizer, eval mode, on device. Cached per modality."""
    from terratorch.registry import FULL_MODEL_REGISTRY
    tok = FULL_MODEL_REGISTRY.build(C.TOKENIZER[modality], pretrained=True)
    tok = tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


@torch.no_grad()
def encode(tok, x: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B, H_q, W_q) int64 token grid."""
    _, _, tokens = tok.encode(x)
    return tokens


@torch.no_grad()
def decode(tok, tokens: torch.Tensor, timesteps: int = 50,
           seed: int | None = 0) -> torch.Tensor:
    """Token grid -> reconstruction. LULC's ViT decoder ignores `timesteps`."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        return tok.decode_tokens(tokens, timesteps=timesteps)
    except TypeError:
        return tok.decode_tokens(tokens)
