"""The map-tokenizer interface, and NVIDIA Cosmos's discrete image tokenizers behind it (A2 Step 1).

A candidate tokenizer implements `MapTokenizer`:
  name, input_size (px), n_codes, output ("rgb" or "logits"), batch (the fixed batch shape),
  encode(x)  x (B, 3, S, S) float32 in [0, 1] on the device -> codes (B, h, w) int64
  decode(i)  codes -> RGB (B, 3, S, S) in [0, 1], or palette logits (B, 32, S, S) over UNIVERSAL
  record()   what Step 4.0 freezes: weights and their sha256, the code that loads it, its settings
A dedicated tokenizer (agents B, C) ships a module with `load(checkpoint, device) -> MapTokenizer`.

Cosmos: the JITs of `nvidia/Cosmos-Tokenize1-DI{8x8,16x16}-360p` and `nvidia/Cosmos-0.1-Tokenizer-
DI{8x8,16x16}`, loaded as cosmos_tokenizer's ImageTokenizer loads them (torch.jit.load, eval, then cast),
without the package. They run in bf16 only: their traced graphs carry bf16 constants (Step 0,
finding 10). On the TITAN V bf16 is emulated.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch

HF = Path("/data/enric/hf/hub")
COSMOS = {
    "t1_di8": ("models--nvidia--Cosmos-Tokenize1-DI8x8-360p", 8),
    "t1_di16": ("models--nvidia--Cosmos-Tokenize1-DI16x16-360p", 16),
    "c01_di8": ("models--nvidia--Cosmos-0.1-Tokenizer-DI8x8", 8),
    "c01_di16": ("models--nvidia--Cosmos-0.1-Tokenizer-DI16x16", 16),
}
COSMOS_CODES = 64000                       # FSQ levels 8, 8, 8, 5, 5, 5


@lru_cache(maxsize=None)
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class MapTokenizer:
    name: str
    input_size: int
    n_codes: int
    output: str = "rgb"
    batch: int = 16

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def record(self) -> dict:
        raise NotImplementedError


class CosmosDI(MapTokenizer):
    """One Cosmos discrete image tokenizer at one input size."""

    def __init__(self, key: str, input_size: int, device: str = "cuda", batch: int = 16):
        # TorchScript's profiling executor runs a graph's first call unoptimized and later calls fused:
        # on the TITAN V (bf16) the first call flips 4.6% of the codes of later calls (measured
        # 2026-11-18). With graph optimization off, every call is the same graph: four calls agreed bit
        # for bit, at the same speed. The codes depend on this setting, so `record` carries it.
        torch._C._set_graph_executor_optimize(False)
        repo, factor = COSMOS[key]
        snap = sorted((HF / repo / "snapshots").iterdir())
        assert len(snap) == 1, snap
        self.snapshot = snap[0]
        self.key, self.factor = key, factor
        self.name = f"{key}_{input_size}"
        self.input_size, self.n_codes, self.batch, self.device = input_size, COSMOS_CODES, batch, device
        self.grid = input_size // factor
        self.output = "rgb"
        self.enc = torch.jit.load(str(self.snapshot / "encoder.jit"), map_location=device).eval().to(device)
        self.dec = torch.jit.load(str(self.snapshot / "decoder.jit"), map_location=device).eval().to(device)
        self.enc, self.dec = self.enc.to(torch.bfloat16), self.dec.to(torch.bfloat16)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self.enc((x * 2 - 1).to(self.device, torch.bfloat16))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.long()

    @torch.no_grad()
    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        y = self.dec(idx.to(self.device).long()).float().clamp(-1, 1)
        return (y + 1) / 2

    def record(self) -> dict:
        files = {f: str((self.snapshot / f).resolve()) for f in ("encoder.jit", "decoder.jit")}
        return {"kind": "cosmos_di", "key": self.key, "hf_repo": COSMOS[self.key][0].replace("models--", "").replace("--", "/"),
                "snapshot": self.snapshot.name, "weights": files,
                "sha256": {f: sha256(p) for f, p in files.items()},
                "input_size": self.input_size, "grid": [self.grid, self.grid], "tokens": self.grid ** 2,
                "n_codes": self.n_codes, "fsq_levels": [8, 8, 8, 5, 5, 5], "dtype": "bfloat16",
                "torchscript_graph_executor_optimize": False, "cudnn_deterministic": True,
                "batch": self.batch, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "loader": f"eo.maps.tokenizers.CosmosDI('{self.key}', {self.input_size})"}


class Identity(MapTokenizer):
    """A perfect tokenizer for the gate: the 'code' of a pixel is its nearest universal colour at the
    input size, and decoding paints it back. Through the harness it reads 1.0 wherever no resampling
    is involved, and otherwise measures the floor that resampling alone sets."""

    def __init__(self, input_size: int, device: str = "cuda"):
        import numpy as np
        from eo.maps.palette import UNIVERSAL
        self.name, self.input_size, self.n_codes, self.batch, self.output = \
            f"identity_{input_size}", input_size, len(UNIVERSAL), 16, "rgb"
        self.pal = torch.as_tensor(np.asarray(UNIVERSAL), dtype=torch.float32, device=device)

    def encode(self, x):
        d = ((x.float()[:, None] - self.pal[None, :, :, None, None]) ** 2).sum(2)
        return d.argmin(1)

    def decode(self, idx):
        return self.pal[idx.long()].permute(0, 3, 1, 2)

    def record(self):
        return {"kind": "identity", "sha256": {"none": "identity"}, "input_size": self.input_size}


def cosmos(name: str, device: str = "cuda") -> CosmosDI:
    """'t1_di8_256' -> CosmosDI('t1_di8', 256)."""
    key, size = name.rsplit("_", 1)
    return CosmosDI(key, int(size), device)


@contextmanager
def deterministic():
    """cuDNN in deterministic mode for the duration. Measured 2026-11-18 on the TITAN V: with the
    defaults, Cosmos Tokenize1-DI8x8's first call on a batch flips 4.6% of the codes of later calls on
    the same batch (bf16, emulated), and the decoded RGB differs by up to 0.043; in deterministic mode
    three calls agree bit for bit."""
    old = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
           torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    # TF32 off as well: on an A100 cuDNN's convolutions default to TF32, which would make an fp32
    # tokenizer's codes depend on the card's precision (no effect on the TITAN V or GTX Titan X).
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = old


def run_fixed(fn, x: torch.Tensor, batch: int) -> torch.Tensor:
    """Apply fn to x in batches of exactly `batch` rows (the tail padded with zeros, then dropped), with
    cuDNN deterministic: tokens are reproducible only at a fixed batch shape and a fixed algorithm."""
    outs = []
    with deterministic(), torch.no_grad():
        for s in range(0, len(x), batch):
            xb = x[s:s + batch]
            n = len(xb)
            if n < batch:
                pad = torch.zeros((batch - n, *xb.shape[1:]), dtype=xb.dtype, device=xb.device)
                xb = torch.cat([xb, pad])
            outs.append(fn(xb)[:n])
    return torch.cat(outs)
