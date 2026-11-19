"""Agent C's dedicated RGB-map tokenizer at f8 (A2 Step 1, candidate family (b), the long slot).

An FSQ autoencoder for label maps rendered with the universal palette (eo.maps.palette.UNIVERSAL):
  - input: the RGB render at `size` px (224: 28 x 28 = 784 tokens; 256: 32 x 32 = 1,024 tokens), in [0, 1];
  - encoder: a lossless space-to-depth stem (PixelUnshuffle(4): 224 -> 56, 48 channels), residual conv
    blocks at 1/4 (c1 channels), a stride-2 conv to 1/8, residual blocks at 1/8 (c2 channels), and a 1 x 1
    projection to the FSQ dimensions;
  - FSQ (Mentzer et al., straight-through, computed in fp32): levels 8,5,5,5 (1,000 codes) or 7,5,5,5,5
    (4,375 codes);
  - decoder: mirrored, ending in 32 palette logits per pixel at the input size through a PixelShuffle(4)
    head (`output = "logits"`, over UNIVERSAL's 32 colours).
Step 0's MapAE (`rgb_map_ae.py`, f16, 13.3M parameters) is the starting point; this one drops the
full-resolution stages (on the GTX Titan X they cost the most and carry nothing a lossless 4 x 4
space-to-depth does not) and spends the compute at 1/8 and 1/4, where the context that lets a token
say "same colour as my neighbour" lives.

`load(checkpoint, device)` rebuilds a frozen tokenizer from a weights file written by
`step1_C_train.py` (`tok_best.pt`, `tok_last.pt`): {"config", "state_dict", "meta"}. Encode and decode
run in fp32; through `eo.maps.tokenizers.run_fixed` (deterministic cuDNN, a fixed batch shape) the codes
are bit-reproducible.

Runs in the `a2seg` conda env.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eo.maps.palette import N_COLOURS
from eo.maps.tokenizers import MapTokenizer, sha256 as file_sha256

DEFAULT = {
    "size": 224,               # input px; tokens = (size / 8) ** 2
    "levels": [8, 5, 5, 5],    # FSQ levels: 1,000 codes
    "unshuffle": 4,            # space-to-depth factor of the stem; the factor is unshuffle * 2 = 8
    "c1": 128,                 # channels at 1/4
    "c2": 256,                 # channels at 1/8
    "n1_enc": 2, "n2_enc": 4,  # residual blocks at 1/4 and 1/8 in the encoder
    "n2_dec": 6, "n1_dec": 2,  # ... and in the decoder
    "groups": 32,
}


class FSQ(nn.Module):
    """Finite scalar quantization, straight-through; Step 0's (`rgb_map_ae.FSQ`) plus the inverse
    (indices -> quantized values) that decoding from codes needs. Always fp32."""

    def __init__(self, levels):
        super().__init__()
        levels = [int(v) for v in levels]
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32), persistent=False)
        basis = np.concatenate([[1], np.cumprod(levels[:-1])]).astype(np.int64)
        self.register_buffer("basis", torch.tensor(basis), persistent=False)
        self.n_codes = int(np.prod(levels))
        self.dim = len(levels)

    def forward(self, z: torch.Tensor):
        """z (B, d, h, w) -> (quantized values in [-1, 1], indices (B, h, w) int64)."""
        L = self.levels.view(1, -1, 1, 1)
        half_l = (L - 1) * (1 + 1e-3) / 2
        offset = torch.where(L % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        zb = torch.tanh(z.float() + shift) * half_l - offset
        q = zb + (torch.round(zb) - zb).detach()            # integers in [-L//2, (L-1)//2]
        hw = torch.div(L, 2, rounding_mode="floor")
        idx = ((q.detach() + hw).round().long() * self.basis.view(1, -1, 1, 1)).sum(1)
        return q / hw, idx

    def indices_to_q(self, idx: torch.Tensor) -> torch.Tensor:
        """indices (B, h, w) -> quantized values (B, d, h, w), exactly as forward produced them."""
        L = self.levels.long().view(1, -1, 1, 1)
        digits = (idx.long()[:, None] // self.basis.view(1, -1, 1, 1)) % L
        hw = torch.div(L, 2, rounding_mode="floor")
        return (digits - hw).float() / hw.float()


class ResBlock(nn.Module):
    def __init__(self, c: int, groups: int):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.c2.weight)                      # each block starts as the identity
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return x + self.c2(F.silu(self.n2(h)))


class MapAE8(nn.Module):
    """f8 FSQ autoencoder: RGB render (B, 3, S, S) in [0, 1] -> codes (B, S/8, S/8) -> 32 palette logits
    (B, 32, S, S)."""

    def __init__(self, cfg: dict):
        super().__init__()
        cfg = {**DEFAULT, **cfg}
        self.cfg = cfg
        u, c1, c2, g = cfg["unshuffle"], cfg["c1"], cfg["c2"], cfg["groups"]
        self.fsq = FSQ(cfg["levels"])
        d = self.fsq.dim
        self.enc = nn.Sequential(
            nn.PixelUnshuffle(u), nn.Conv2d(3 * u * u, c1, 3, padding=1),
            *[ResBlock(c1, g) for _ in range(cfg["n1_enc"])],
            nn.Conv2d(c1, c2, 4, stride=2, padding=1),
            *[ResBlock(c2, g) for _ in range(cfg["n2_enc"])],
            nn.GroupNorm(g, c2), nn.SiLU(), nn.Conv2d(c2, d, 1))
        self.dec = nn.Sequential(
            nn.Conv2d(d, c2, 3, padding=1),
            *[ResBlock(c2, g) for _ in range(cfg["n2_dec"])],
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c1, 3, padding=1),
            *[ResBlock(c1, g) for _ in range(cfg["n1_dec"])],
            nn.GroupNorm(g, c1), nn.SiLU(), nn.Conv2d(c1, N_COLOURS * u * u, 1), nn.PixelShuffle(u))

    @property
    def factor(self) -> int:
        return self.cfg["unshuffle"] * 2

    def encode(self, x01: torch.Tensor):
        """(B, 3, S, S) in [0, 1] -> (quantized values (B, d, h, w), indices (B, h, w) int64)."""
        return self.fsq(self.enc(x01.float() * 2 - 1))

    def decode_q(self, q: torch.Tensor) -> torch.Tensor:
        return self.dec(q.float())

    def decode_idx(self, idx: torch.Tensor) -> torch.Tensor:
        return self.decode_q(self.fsq.indices_to_q(idx))

    def forward(self, x01: torch.Tensor):
        q, idx = self.encode(x01)
        return self.decode_q(q), idx


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_sha256(state_dict: dict) -> str:
    """A hash of the weights themselves (for a live model that has no file yet)."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


class StepCTokenizer(MapTokenizer):
    """The MapTokenizer interface around a MapAE8 (eo/maps/tokenizers.py)."""

    output = "logits"

    def __init__(self, model: MapAE8, weights: str | None = None, sha: str | None = None,
                 batch: int = 32, device: str = "cuda", name: str | None = None, meta: dict | None = None):
        self.model = model.to(device).eval()
        self.cfg = dict(model.cfg)
        self.device = device
        self.input_size = int(self.cfg["size"])
        self.n_codes = model.fsq.n_codes
        self.batch = int(batch)
        self.grid = self.input_size // model.factor
        self.weights = weights
        self.sha = sha if sha is not None else (file_sha256(weights) if weights else
                                                 state_sha256(model.state_dict()))
        self.meta = meta or {}
        levels = "".join(str(v) for v in self.cfg["levels"])
        self.name = name or f"stepC_f8_{self.input_size}_L{levels}_{self.sha[:10]}"

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x.to(self.device).float())[1]

    @torch.no_grad()
    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model.decode_idx(idx.to(self.device).long())

    def record(self) -> dict:
        return {"kind": "stepC_map_ae8", "weights": self.weights, "sha256": {"weights": self.sha},
                "config": self.cfg, "input_size": self.input_size, "grid": [self.grid, self.grid],
                "tokens": self.grid ** 2, "n_codes": self.n_codes, "fsq_levels": list(self.cfg["levels"]),
                "output": "logits over palette.UNIVERSAL (32)", "dtype": "float32", "batch": self.batch,
                "cudnn_deterministic": True, "params": n_params(self.model),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "meta": self.meta,
                "loader": (f"eo.experiments.segmentation.step1_C_tok.load('{self.weights}', 'cuda')"
                           if self.weights else None)}


def save_weights(model: MapAE8, path: str | Path, meta: dict) -> str:
    """Write a loadable tokenizer file atomically; returns its sha256."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"config": dict(model.cfg), "state_dict": sd, "meta": meta}, tmp)
    tmp.replace(path)
    file_sha256.cache_clear()
    return file_sha256(str(path))


def load(checkpoint: str, device: str = "cuda", batch: int = 32) -> StepCTokenizer:
    """Rebuild a frozen tokenizer from a weights file written by step1_C_train.py."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MapAE8(blob["config"])
    model.load_state_dict(blob["state_dict"])
    return StepCTokenizer(model, weights=str(Path(checkpoint).resolve()), batch=batch, device=device,
                          meta=blob.get("meta", {}))


def untrained(cfg: dict, seed: int = 0, device: str = "cuda", batch: int = 32) -> StepCTokenizer:
    """A randomly initialised tokenizer of the same design (the untrained control). Note: every
    residual block starts as the identity (its last conv is zero-initialised), as in training."""
    torch.manual_seed(seed)
    model = MapAE8(cfg)
    levels = "".join(str(v) for v in model.cfg["levels"])
    return StepCTokenizer(model, batch=batch, device=device,
                          name=f"stepC_untrained_{model.cfg['size']}_L{levels}_seed{seed}",
                          meta={"untrained_seed": seed, "config": json.loads(json.dumps(model.cfg))})
