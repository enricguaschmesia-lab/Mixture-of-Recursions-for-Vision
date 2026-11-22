"""A2 Step 1, candidate (c): NVIDIA Cosmos Tokenize1-DI16x16 fine-tuned on palette-shuffled map renders.

Enric, 2026-11-22: "Launch a test of the fine tuned cosmos tokenizer you think will perform best." Step 1 never
trained (c) (plan Step 1.1; worklog §4, item 81): the dedicated tokenizers (b) passed in both bands. This module is
that candidate, at f16 and 224 px, so 196 tokens on the inputs' 14 x 14 grid with Cosmos's own 64,000 codes: the
aligned arm's geometry and codebook size, so it can stand in for the frozen pick without moving anything else.

The network is Cosmos's discrete image tokenizer, rebuilt in plain PyTorch from NVIDIA's code and started from the
released weights:
  - code: github.com/NVIDIA/Cosmos-Tokenizer at 3584ae752ce8ebdbe06a420bf60d7513c0e878cc (Apache License 2.0),
    cosmos_tokenizer/modules/{layers2d,patching,quantizers,utils}.py, adapted (see "Adapted" below);
  - weights: model.pt's `network` of nvidia/Cosmos-Tokenize1-DI16x16-360p (revision 5f511e03), identical to the
    released JITs' (max |difference| 0.0 on every tensor, checked 2026-11-22). They are bf16 values; we train in fp32
    master weights.
  - DI16x16 as released: a one-level Haar patcher (f2; the repo's default config says 4, the weights say 2: the
    first conv takes 12 channels), residual conv stages at f2, f4, f8 (widths 256, 512, 512, 2 blocks each), each
    followed by a stride-2 conv, a mid block with self-attention at the 14 x 14 grid, a 3 x 3 conv to 256
    channels, a 1 x 1 conv to 6 dims, FSQ 8,8,8,5,5,5 = 64,000 codes; the decoder mirrors it (3 blocks per stage,
    each followed by a nearest x2 upsampling and a conv) back to RGB through the inverse Haar transform.
    86.55M parameters.
One module is added: `head`, a 1 x 1 conv from the decoder's RGB (in [-1, 1]) to the 32 palette logits over
palette.UNIVERSAL. It starts as a scaled nearest-colour rule: logit_k = s (2 y.c_k - |c_k|^2), which is
-s |y - c_k|^2 up to a per-pixel constant, so its argmax is the universal colour nearest the decoded RGB. So the
tokenizer's output is "logits", as B's and C's are, and it trains on their loss (step1_D_train.py).

Adapted from Cosmos: GroupNorm with one group (the released graphs' setting, not the repo's default 32), no loguru,
no dropout modules (0.0 in the config), no attention in the down and up stages (none at this config:
attn_resolutions [32] never matches), the Haar transforms only; FSQ as agent B's (step1_B_tok.FSQ,
copied here so the frozen loader is one file): the same bound, rounding and index basis as Cosmos's FSQuantizer,
computed in fp32. The input is the RGB render mapped to [-1, 1], as Cosmos takes it.
Encode and decode run in fp32 under the harness's run_fixed (deterministic cuDNN, TF32 off, fixed batch shape).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eo.maps.palette import GRID_LEVELS, N_COLOURS, UNIVERSAL
from eo.maps.tokenizers import MapTokenizer, sha256

CODE_PATH = Path(__file__).resolve()
COSMOS_CODE = ("github.com/NVIDIA/Cosmos-Tokenizer 3584ae752ce8ebdbe06a420bf60d7513c0e878cc (Apache-2.0): "
               "cosmos_tokenizer/modules/{layers2d,patching,quantizers,utils}.py")
COSMOS_WEIGHTS = {16: ("nvidia/Cosmos-Tokenize1-DI16x16-360p", "5f511e03859e5f4caebecfd860357ddc9ce30706")}
HAAR = (0.7071067811865476, 0.7071067811865476)

DEFAULT = dict(factor=16, input_size=224, levels=[8, 8, 8, 5, 5, 5], channels=128, channels_mult=[2, 4, 4],
               num_res_blocks=2, z_channels=256, patch_size=2, head_scale=25.0, head="rgb")
ARCH_KEYS = tuple(DEFAULT)
#: head "rgb": Cosmos's decoder as released (3 x p^2 channels -> RGB) and a 1 x 1 palette head on the RGB.
#: head "logits" (added 2026-11-22, 13:15): the decoder's last conv widened to 32 x p^2 channels, so the inverse Haar
#: transform yields the 32 palette logits directly (as B's and C's decoders do), with no RGB bottleneck; it starts as
#: the "rgb" head folded into that conv (fold_head), the same function. Its last conv and the inverse Haar run in fp32.
#: A config without "head" is an "rgb" model (the first run, d224_cosmos16).


# ----------------------------------------------------------------------------------------------- palette input
def colour_lut() -> torch.Tensor:
    """(216,) palette index of every point of the 6-level RGB grid (as step1_B_tok.colour_lut)."""
    lv = np.arange(GRID_LEVELS) / (GRID_LEVELS - 1)
    grid = np.stack(np.meshgrid(lv, lv, lv, indexing="ij"), -1).reshape(-1, 3)
    d = ((grid[:, None] - np.asarray(UNIVERSAL)[None]) ** 2).sum(-1)
    return torch.as_tensor(d.argmin(1), dtype=torch.long)


def rgb_to_index(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """(B, 3, S, S) in [0, 1] -> (B, S, S) palette indices (exact for a render; as step1_B_tok.rgb_to_index)."""
    q = torch.round(x.float().clamp(0, 1) * (GRID_LEVELS - 1)).long()
    return lut[(q[:, 0] * GRID_LEVELS + q[:, 1]) * GRID_LEVELS + q[:, 2]]


# ----------------------------------------------------------------------------------------------- Cosmos layers
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def normalize(c: int) -> nn.GroupNorm:
    """GroupNorm with ONE group, as the released Tokenize1 graphs run it (`torch.group_norm(x, 1, ...)` in the JITs'
    code); the repo's `Normalize` defaults to 32 groups, which reads the same weights wrongly (2026-11-22: codes 1.4%
    equal to the JIT's, as far off as a broken patcher)."""
    return nn.GroupNorm(num_groups=1, num_channels=c, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))


class Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class ResnetBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1, self.conv1 = normalize(cin), nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1)
        self.norm2, self.conv2 = normalize(cout), nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = nn.Conv2d(cin, cout, kernel_size=1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.norm = normalize(c)
        self.q, self.k, self.v = (nn.Conv2d(c, c, kernel_size=1) for _ in range(3))
        self.proj_out = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        w_ = torch.bmm(q, k.reshape(b, c, h * w)) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)
        h_ = torch.bmm(v.reshape(b, c, h * w), w_.permute(0, 2, 1)).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class Patcher(nn.Module):
    """Cosmos's 2D Haar patcher (patch_method "haar"): log2(patch) levels of a rescaled DWT on every band."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.levels = int(round(math.log2(patch_size)))
        self.register_buffer("wavelets", torch.tensor(HAAR), persistent=False)
        self.register_buffer("_arange", torch.arange(2), persistent=False)

    def _dwt(self, x):
        dtype, h = x.dtype, self.wavelets
        n, g = h.shape[0], x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)
        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode="reflect").to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))
        return torch.cat([xll, xlh, xhl, xhh], dim=1) / 2

    def forward(self, x):
        for _ in range(self.levels):
            x = self._dwt(x)
        return x


class UnPatcher(nn.Module):
    """Cosmos's inverse of Patcher (the rescaled inverse DWT, log2(patch) levels)."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.levels = int(round(math.log2(patch_size)))
        self.register_buffer("wavelets", torch.tensor(HAAR), persistent=False)
        self.register_buffer("_arange", torch.arange(2), persistent=False)

    def _idwt(self, x):
        dtype, h = x.dtype, self.wavelets
        n, g = h.shape[0], x.shape[1] // 4
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1]).to(dtype)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)
        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)
        ct = F.conv_transpose2d
        yl = ct(xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)) + \
            ct(xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yh = ct(xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)) + \
            ct(xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        y = ct(yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)) + \
            ct(yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2))
        return y * 2

    def forward(self, x):
        for _ in range(self.levels):
            x = self._idwt(x)
        return x


def _n_resample(cfg: dict) -> int:
    return int(math.log2(cfg["factor"])) - int(math.log2(cfg["patch_size"]))


class Encoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ch, mult, nb = cfg["channels"], cfg["channels_mult"], cfg["num_res_blocks"]
        self.num_resolutions, self.num_res_blocks, self.num_downsamples = len(mult), nb, _n_resample(cfg)
        assert self.num_downsamples <= self.num_resolutions
        self.patcher = Patcher(cfg["patch_size"])
        self.conv_in = nn.Conv2d(3 * cfg["patch_size"] ** 2, ch, kernel_size=3, stride=1, padding=1)
        in_mult = (1,) + tuple(mult)
        self.down = nn.ModuleList()
        for i in range(self.num_resolutions):
            block, cin, cout = nn.ModuleList(), ch * in_mult[i], ch * mult[i]
            for _ in range(nb):
                block.append(ResnetBlock(cin, cout))
                cin = cout
            down = nn.Module()
            down.block = block
            if i < self.num_downsamples:
                down.downsample = Downsample(cin)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1, self.mid.attn_1, self.mid.block_2 = ResnetBlock(cin, cin), AttnBlock(cin), ResnetBlock(cin, cin)
        self.norm_out = normalize(cin)
        self.conv_out = nn.Conv2d(cin, cfg["z_channels"], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(self.patcher(x))
        for i in range(self.num_resolutions):
            for j in range(self.num_res_blocks):
                h = self.down[i].block[j](h)
            if i < self.num_downsamples:
                h = self.down[i].downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(swish(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ch, mult, nb = cfg["channels"], cfg["channels_mult"], cfg["num_res_blocks"]
        self.num_resolutions, self.num_res_blocks, self.num_upsamples = len(mult), nb, _n_resample(cfg)
        assert self.num_upsamples <= self.num_resolutions
        self.unpatcher = UnPatcher(cfg["patch_size"])
        cin = ch * mult[-1]
        self.conv_in = nn.Conv2d(cfg["z_channels"], cin, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1, self.mid.attn_1, self.mid.block_2 = ResnetBlock(cin, cin), AttnBlock(cin), ResnetBlock(cin, cin)
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            block, cout = nn.ModuleList(), ch * mult[i]
            for _ in range(nb + 1):
                block.append(ResnetBlock(cin, cout))
                cin = cout
            up = nn.Module()
            up.block = block
            if i >= self.num_resolutions - self.num_upsamples:
                up.upsample = Upsample(cin)
            self.up.insert(0, up)
        self.norm_out = normalize(cin)
        self.logits = cfg.get("head", "rgb") == "logits"
        out = N_COLOURS if self.logits else 3
        self.conv_out = nn.Conv2d(cin, out * cfg["patch_size"] ** 2, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = self.up[i].block[j](h)
            if i >= self.num_resolutions - self.num_upsamples:
                h = self.up[i].upsample(h)
        h = swish(self.norm_out(h))
        if self.logits:                     # logits reach ~100: the last conv and the inverse Haar in fp32
            with torch.autocast(device_type=h.device.type, enabled=False):
                return self.unpatcher(self.conv_out(h.float()))
        return self.unpatcher(self.conv_out(h))


class FSQ(nn.Module):
    """Finite scalar quantization, straight-through, in fp32 (step1_B_tok.FSQ; Cosmos's FSQuantizer convention:
    bound = tanh(z + atanh(offset / half_l)) * half_l - offset, half_l = (L - 1)(1 + 1e-3) / 2, index =
    sum_i (round(bound_i) + L_i // 2) * prod_{j < i} L_j)."""

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


def init_head(head: nn.Conv2d, scale: float) -> None:
    """logit_k = scale * (2 y . c_k - |c_k|^2), c_k = universal colour k in [-1, 1]: -scale * |y - c_k|^2 up to a
    per-pixel constant, so the argmax is the nearest universal colour."""
    c = torch.as_tensor(np.asarray(UNIVERSAL), dtype=torch.float32) * 2 - 1          # (32, 3)
    with torch.no_grad():
        head.weight.copy_((2 * scale * c)[:, :, None, None])
        head.bias.copy_(-scale * (c ** 2).sum(1))


def fold_head(w0: torch.Tensor, b0: torch.Tensor, scale: float, patch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The "rgb" head folded into the decoder's last conv: (w0 (3 p^2, C, 3, 3), b0) -> (w1 (32 p^2, C, 3, 3), b1) such that
    inverse_haar(conv(h; w1, b1)) = head(inverse_haar(conv(h; w0, b0))) exactly (both maps are linear). One Haar level:
    channels are band-major (UnPatcher chunks them into LL, LH, HL, HH), and a constant v in the LL band comes out as v,
    so the head's bias goes to the LL band."""
    assert patch == 2, "fold_head handles one Haar level"
    c = torch.as_tensor(np.asarray(UNIVERSAL), dtype=torch.float32) * 2 - 1           # (32, 3) in [-1, 1]
    A, a = 2 * scale * c, -scale * (c ** 2).sum(1)
    w = w0.float().view(4, 3, *w0.shape[1:])
    w1 = torch.einsum("kc,bc...->bk...", A, w).reshape(4 * N_COLOURS, *w0.shape[1:])
    b1 = torch.einsum("kc,bc->bk", A, b0.float().view(4, 3))
    b1[0] += a
    return w1.contiguous(), b1.reshape(-1).contiguous()


class MapTokD(nn.Module):
    """Cosmos DI (encoder, quant_conv, FSQ, post_quant_conv, decoder) plus the palette head."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = {k: cfg.get(k, DEFAULT[k]) if k == "head" else cfg[k] for k in ARCH_KEYS}
        assert self.cfg["head"] in ("rgb", "logits"), self.cfg["head"]
        assert self.cfg["input_size"] % self.cfg["factor"] == 0
        nd = len(self.cfg["levels"])
        self.encoder, self.decoder = Encoder(self.cfg), Decoder(self.cfg)
        self.quant_conv = nn.Conv2d(self.cfg["z_channels"], nd, 1)
        self.post_quant_conv = nn.Conv2d(nd, self.cfg["z_channels"], 1)
        self.fsq = FSQ(self.cfg["levels"])
        if self.cfg["head"] == "rgb":
            self.head = nn.Conv2d(3, N_COLOURS, 1)
            init_head(self.head, self.cfg["head_scale"])
        self.register_buffer("lut", colour_lut(), persistent=False)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """RGB (B, 3, S, S) in [0, 1] -> the pre-quantization latent (B, 6, h, w)."""
        return self.quant_conv(self.encoder(x * 2 - 1))

    def encode_q(self, x: torch.Tensor):
        return self.fsq(self.latent(x))

    def rgb(self, q: torch.Tensor) -> torch.Tensor:
        """Quantized codes (B, 6, h, w) -> Cosmos's decoded RGB in [-1, 1] (unclamped); the "logits" head: the logits."""
        return self.decoder(self.post_quant_conv(q))

    def decode_q(self, q: torch.Tensor) -> torch.Tensor:
        """Palette logits, the head always in fp32: its logits reach ~100, where bf16's step is 0.5."""
        y = self.rgb(q)
        if self.cfg["head"] == "logits":
            return y.float()
        with torch.autocast(device_type=y.device.type, enabled=False):
            return self.head(y.float())

    def forward(self, x: torch.Tensor, oh_dtype=None):
        """Training path, as step1_B_tok.MapTok: (logits (B, 32, S, S), codes (B, h, w), palette indices of x)."""
        q, idx = self.encode_q(x)
        return self.decode_q(q), idx, rgb_to_index(x, self.lut)


def cosmos_state(path: str | Path) -> tuple[dict, dict]:
    """The `network` weights of a Cosmos Tokenize1 model.pt (or of a file written by `extract_cosmos`), as fp32."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if "network" in ck and isinstance(ck["network"], dict):              # an extract
        return {k: v.float() for k, v in ck["network"].items()}, ck.get("meta", {})
    m = ck["model"]
    net = {k[len("network."):]: v.float() for k, v in m.items() if k.startswith("network.")}
    return net, {"iteration": ck.get("iteration")}


def from_cosmos(path: str | Path, cfg: dict | None = None) -> tuple[MapTokD, dict]:
    """A MapTokD with Cosmos's released weights and the nearest-colour head. Strict: every Cosmos tensor is used, and
    the head's two tensors are the only ones not loaded ("rgb"), or the decoder's last conv is the head folded into
    Cosmos's ("logits")."""
    model = MapTokD(dict(DEFAULT, **(cfg or {})))
    net, meta = cosmos_state(path)
    want = ["head.bias", "head.weight"]
    if model.cfg["head"] == "logits":
        net = dict(net)
        net["decoder.conv_out.weight"], net["decoder.conv_out.bias"] = fold_head(
            net["decoder.conv_out.weight"], net["decoder.conv_out.bias"], model.cfg["head_scale"], model.cfg["patch_size"])
        want = []
    missing, unexpected = model.load_state_dict(net, strict=False)
    if unexpected or sorted(missing) != want:
        raise RuntimeError(f"Cosmos weights do not fit: missing {missing}, unexpected {unexpected}")
    return model, {"init_weights": str(Path(path).resolve()), "init_sha256": sha256(str(Path(path).resolve())),
                   "init_tensors": len(net), "init_meta": meta}


class DTokenizer(MapTokenizer):
    """The fine-tuned Cosmos behind the harness interface: fp32, fixed batch (run_fixed)."""
    output = "logits"

    def __init__(self, model: MapTokD, name: str, weights: str | None = None, sha: str | None = None,
                 batch: int = 16, device: str = "cuda"):
        self.model = model.to(device).float().eval()
        self.name, self.weights, self.sha, self.batch, self.device = name, weights, sha, batch, device
        self.input_size = model.cfg["input_size"]
        self.n_codes = model.fsq.n_codes
        self.grid = self.input_size // model.cfg["factor"]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, idx = self.model.encode_q(x.to(self.device).float())
        return idx

    @torch.no_grad()
    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model.decode_q(self.model.fsq.codes_to_q(idx.to(self.device))).float()

    def record(self) -> dict:
        cfg = self.model.cfg
        return {"kind": "step1_D_cosmos_ft", "weights": self.weights, "sha256": {"weights": self.sha or "unsaved"},
                "config": cfg, "input_size": self.input_size, "grid": [self.grid, self.grid],
                "tokens": self.grid ** 2, "n_codes": self.n_codes, "fsq_levels": list(cfg["levels"]),
                "output": "logits", "palette": "universal (eo.maps.palette.UNIVERSAL)", "dtype": "float32",
                "attention": "Cosmos AttnBlock (bmm, softmax)", "cudnn_deterministic": True, "batch": self.batch,
                "params": sum(p.numel() for p in self.model.parameters()), "cosmos_code": COSMOS_CODE,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "code": str(CODE_PATH), "code_sha256": sha256(str(CODE_PATH)),
                "loader": (f"eo.experiments.segmentation.step1_D_tok.load('{self.weights}', 'cuda')"
                           if self.weights else None)}


def export(path: str | Path, model: MapTokD, meta: dict | None = None) -> str:
    """Write a candidate's weights file ({"config", "state_dict", "meta"}) and return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"config": dict(model.cfg), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "meta": meta or {}}, tmp)
    tmp.replace(path)
    sha256.cache_clear()
    return sha256(str(path.resolve()))


def load(checkpoint: str, device: str = "cuda", batch: int = 16) -> DTokenizer:
    """The tokenizer stored in `checkpoint` (a file written by `export`)."""
    path = str(Path(checkpoint).resolve())
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = MapTokD(ck["config"])
    model.load_state_dict(ck["state_dict"])
    sha = sha256(path)
    name = f"D_cosmos{ck['config']['factor']}_{ck['config']['input_size']}_{model.fsq.n_codes}_{sha[:10]}"
    return DTokenizer(model, name, weights=path, sha=sha, batch=batch, device=device)


def extract_cosmos(model_pt: str | Path, out: str | Path, repo: str, revision: str) -> str:
    """Write model.pt's `network` weights alone (bf16 as released) with their provenance; return the sha256."""
    ck = torch.load(str(model_pt), map_location="cpu", weights_only=False)
    net = {k[len("network."):]: v for k, v in ck["model"].items() if k.startswith("network.")}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save({"network": net, "meta": {"hf_repo": repo, "revision": revision, "source": str(model_pt),
                                         "source_sha256": sha256(str(Path(model_pt).resolve())),
                                         "iteration": ck.get("iteration")}}, tmp)
    tmp.replace(out)
    sha256.cache_clear()
    return sha256(str(out.resolve()))
