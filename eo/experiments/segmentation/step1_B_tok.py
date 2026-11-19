"""A2 Step 1, agent B: a dedicated RGB-map tokenizer at f16 (196 tokens at 224 px, 256 at 256 px).

The MapTokenizer behind `load(checkpoint, device)`. Trained by `step1_B_train.py` on palette-shuffled
renders of the `fit` maps (eo.maps.sets); read only through `eo.maps.harness`.

Design (Step 0's `rgb_map_ae.py`, made larger and coordinated):
  - input: the RGB render. The universal palette lies exactly on a 6-level RGB grid
    (eo.maps.palette: GRID_LEVELS), so every rendered pixel is mapped to its palette index exactly
    (round to the grid, then a 216-entry table; a pixel off the palette takes the palette colour
    nearest its grid point), and the encoder sees a 32-channel one-hot;
  - encoder: a 2x2 patch stem, residual conv stages at f2, f4, f8, f16 (widths 64, 128, 256, 512),
    then transformer blocks over the 14 x 14 (or 16 x 16) grid so the codes can coordinate, and a
    projection to the FSQ dimensions (4 for 8*5*5*5 = 1,000 codes, 5 for 7*5*5*5*5 = 4,375);
  - FSQ (Mentzer et al.), straight-through, in fp32;
  - decoder: transformer blocks over the grid, residual conv stages back to f2, and a pixel-shuffle
    head: 32 palette logits per pixel over palette.UNIVERSAL at the input size (output = "logits").
Encode and decode run in fp32 with the math attention kernel, under the harness's run_fixed
(deterministic cuDNN, fixed batch shape).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from eo.maps.palette import GRID_LEVELS, N_COLOURS, UNIVERSAL
from eo.maps.tokenizers import MapTokenizer, sha256

CODE_PATH = Path(__file__).resolve()
FACTOR = 16

DEFAULT = dict(input_size=224, levels=[7, 5, 5, 5, 5], stem=2, widths=[64, 128, 256, 512],
               enc_blocks=[1, 2, 2, 1], dec_blocks=[2, 2, 2, 1], attn_dim=512, heads=8, enc_attn=4,
               dec_attn=6, mlp_ratio=4)
ARCH_KEYS = tuple(DEFAULT)


def colour_lut() -> torch.Tensor:
    """(216,) palette index of every point of the 6-level RGB grid (key r*36 + g*6 + b): itself for a
    palette colour, else the palette colour nearest the grid point."""
    lv = np.arange(GRID_LEVELS) / (GRID_LEVELS - 1)
    grid = np.stack(np.meshgrid(lv, lv, lv, indexing="ij"), -1).reshape(-1, 3)
    d = ((grid[:, None] - np.asarray(UNIVERSAL)[None]) ** 2).sum(-1)
    lut = d.argmin(1)
    assert all(d[i, lut[i]] < 1e-12 for i in range(len(grid)) if d[i].min() < 1e-12)
    return torch.as_tensor(lut, dtype=torch.long)


def rgb_to_index(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """(B, 3, S, S) in [0, 1] -> (B, S, S) palette indices (exact for a render)."""
    q = torch.round(x.float().clamp(0, 1) * (GRID_LEVELS - 1)).long()
    return lut[(q[:, 0] * GRID_LEVELS + q[:, 1]) * GRID_LEVELS + q[:, 2]]


class FSQ(nn.Module):
    """Finite scalar quantization, straight-through (Step 0's, plus codes_to_q for decoding)."""

    def __init__(self, levels):
        super().__init__()
        levels = [int(v) for v in levels]
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32), persistent=False)
        basis = np.concatenate([[1], np.cumprod(levels[:-1])]).astype(np.int64)
        self.register_buffer("basis", torch.tensor(basis), persistent=False)
        self.n_codes, self.dim = int(np.prod(levels)), len(levels)

    def forward(self, z: torch.Tensor):                         # z (B, d, h, w) -> (q in [-1, 1], idx)
        z = z.float()
        L = self.levels.view(1, -1, 1, 1)
        half_l = (L - 1) * (1 + 1e-3) / 2
        offset = torch.where(L % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        zb = torch.tanh(z + shift) * half_l - offset
        r = torch.round(zb)
        q = zb + (r - zb).detach()
        hw = torch.div(L, 2, rounding_mode="floor")
        idx = ((r.detach() + hw).long() * self.basis.view(1, -1, 1, 1)).sum(1)
        return q / hw, idx

    def codes_to_q(self, idx: torch.Tensor) -> torch.Tensor:    # (B, h, w) -> (B, d, h, w)
        L = self.levels.long().view(1, -1, 1, 1)
        hw = L // 2
        digits = (idx.long()[:, None] // self.basis.view(1, -1, 1, 1)) % L
        return (digits - hw).float() / hw.float()


def gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c // 4), c)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.n1, self.c1 = gn(c), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = gn(c), nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class Block(nn.Module):
    """Pre-norm transformer block over the token grid."""

    def __init__(self, d: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.heads = heads
        self.n1, self.qkv, self.proj = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.fc1, self.fc2 = nn.LayerNorm(d), nn.Linear(d, mlp_ratio * d), nn.Linear(mlp_ratio * d, d)
        for m in (self.proj, self.fc2):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


def _check(cfg: dict) -> int:
    w = cfg["widths"]
    assert cfg["stem"] * 2 ** (len(w) - 1) == FACTOR, "the model must be f16"
    assert cfg["input_size"] % FACTOR == 0
    assert len(cfg["enc_blocks"]) == len(cfg["dec_blocks"]) == len(w)
    return cfg["input_size"] // FACTOR


class Encoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        g = _check(cfg)
        w, nb, d = cfg["widths"], cfg["enc_blocks"], cfg["attn_dim"]
        self.stem = nn.Conv2d(N_COLOURS, w[0], cfg["stem"], stride=cfg["stem"])
        layers: list[nn.Module] = []
        for i in range(len(w)):
            layers += [ResBlock(w[i]) for _ in range(nb[i])]
            if i < len(w) - 1:
                layers.append(nn.Conv2d(w[i], w[i + 1], 4, stride=2, padding=1))
        layers += [gn(w[-1]), nn.SiLU(), nn.Conv2d(w[-1], d, 1)]
        self.convs = nn.Sequential(*layers)
        self.pos = nn.Parameter(torch.randn(1, g * g, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg["heads"], cfg["mlp_ratio"]) for _ in range(cfg["enc_attn"])])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, len(cfg["levels"]))

    def forward(self, onehot):
        h = self.convs(self.stem(onehot))
        B, _, gh, gw = h.shape
        t = h.flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            t = b(t)
        return self.out(self.norm(t)).transpose(1, 2).reshape(B, -1, gh, gw)


class Decoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        g = _check(cfg)
        w, nb, d, s = cfg["widths"], cfg["dec_blocks"], cfg["attn_dim"], cfg["stem"]
        self.inp = nn.Linear(len(cfg["levels"]), d)
        self.pos = nn.Parameter(torch.randn(1, g * g, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg["heads"], cfg["mlp_ratio"]) for _ in range(cfg["dec_attn"])])
        self.norm = nn.LayerNorm(d)
        layers: list[nn.Module] = [nn.Conv2d(d, w[-1], 1)]
        for i in reversed(range(len(w))):
            layers += [ResBlock(w[i]) for _ in range(nb[i])]
            if i > 0:
                layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(w[i], w[i - 1], 3, padding=1)]
        layers += [gn(w[0]), nn.SiLU(), nn.Conv2d(w[0], N_COLOURS * s * s, 3, padding=1), nn.PixelShuffle(s)]
        self.convs = nn.Sequential(*layers)

    def forward(self, q):
        B, _, gh, gw = q.shape
        t = self.inp(q.flatten(2).transpose(1, 2)) + self.pos
        for b in self.blocks:
            t = b(t)
        h = self.norm(t).transpose(1, 2).reshape(B, -1, gh, gw)
        return self.convs(h)


class MapTok(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = {k: cfg[k] for k in ARCH_KEYS}
        self.enc, self.fsq, self.dec = Encoder(self.cfg), FSQ(self.cfg["levels"]), Decoder(self.cfg)
        self.register_buffer("lut", colour_lut(), persistent=False)

    def onehot(self, x: torch.Tensor, dtype=torch.float32):
        """RGB (B, 3, S, S) -> (one-hot (B, 32, S, S) in `dtype`, palette indices (B, S, S))."""
        idx = rgb_to_index(x, self.lut)
        oh = torch.zeros((idx.shape[0], N_COLOURS, *idx.shape[1:]), dtype=dtype, device=idx.device)
        oh.scatter_(1, idx[:, None], 1.0)
        if self.enc.stem.weight.is_contiguous(memory_format=torch.channels_last):
            oh = oh.contiguous(memory_format=torch.channels_last)
        return oh, idx

    def forward(self, x: torch.Tensor, oh_dtype=torch.float32):
        """Training path: (logits (B, 32, S, S), codes (B, h, w), palette indices of x (B, S, S))."""
        oh, cidx = self.onehot(x, oh_dtype)
        q, idx = self.fsq(self.enc(oh))
        return self.dec(q), idx, cidx


class BTokenizer(MapTokenizer):
    """Agent B's map tokenizer behind the harness interface: fp32, math attention, fixed batch."""
    output = "logits"

    def __init__(self, model: MapTok, name: str, weights: str | None = None, sha: str | None = None,
                 batch: int = 16, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.name, self.weights, self.sha, self.batch, self.device = name, weights, sha, batch, device
        self.input_size = model.cfg["input_size"]
        self.n_codes = model.fsq.n_codes
        self.grid = self.input_size // FACTOR

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with sdpa_kernel(SDPBackend.MATH):
            oh, _ = self.model.onehot(x.to(self.device))
            _, idx = self.model.fsq(self.model.enc(oh))
        return idx

    @torch.no_grad()
    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        with sdpa_kernel(SDPBackend.MATH):
            return self.model.dec(self.model.fsq.codes_to_q(idx.to(self.device))).float()

    def record(self) -> dict:
        cfg = self.model.cfg
        return {"kind": "step1_B_maptok", "weights": self.weights,
                "sha256": {"weights": self.sha or "unsaved"},
                "config": cfg, "input_size": self.input_size, "grid": [self.grid, self.grid],
                "tokens": self.grid ** 2, "n_codes": self.n_codes, "fsq_levels": list(cfg["levels"]),
                "output": "logits", "palette": "universal (eo.maps.palette.UNIVERSAL)", "dtype": "float32",
                "attention_kernel": "math", "cudnn_deterministic": True, "batch": self.batch,
                "params": sum(p.numel() for p in self.model.parameters()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "code": str(CODE_PATH), "code_sha256": sha256(str(CODE_PATH)),
                "loader": (f"eo.experiments.segmentation.step1_B_tok.load('{self.weights}', 'cuda')"
                           if self.weights else None)}


def export(path: str | Path, model: MapTok, meta: dict | None = None) -> str:
    """Write a candidate's weights file ({"config", "state_dict", "meta"}) and return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"config": dict(model.cfg), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "meta": meta or {}}, tmp)
    tmp.replace(path)
    sha256.cache_clear()
    return sha256(str(path.resolve()))


def load(checkpoint: str, device: str = "cuda", batch: int = 16) -> BTokenizer:
    """The tokenizer stored in `checkpoint` (a file written by `export`)."""
    path = str(Path(checkpoint).resolve())
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = MapTok(ck["config"])
    model.load_state_dict(ck["state_dict"])
    sha = sha256(path)
    name = f"B_f16_{ck['config']['input_size']}_{model.fsq.n_codes}_{sha[:10]}"
    return BTokenizer(model, name, weights=path, sha=sha, batch=batch, device=device)
