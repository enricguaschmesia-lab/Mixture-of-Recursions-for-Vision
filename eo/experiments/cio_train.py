#!/usr/bin/env python
"""Continuous-I/O pilot: small decoders that read and write TerraMesh as tokens OR as pixels. (`.venv`)

Question (Enric, 2026-10-28): can our decoder-only model ingest continuous patches and emit
continuous outputs, with no per-modality tokenizer? This pilot answers the two halves on the
same rows (cio_data.py), with the same backbone: a LLaMA decoder from the `transformers` class
our model uses (LlamaModel, sdpa, 1-D RoPE), shrunk to d=384, 8 layers, 6 heads, 2 KV heads.

One sequence = one (source, target) pair over S2L2A, S1RTC, DEM, LULC (12 ordered pairs, one
pair per batch, drawn at random):

    [BO_s] s_1 ... s_196 [EO_s] [BO_t] t_1 ... t_195          (394 positions, causal)

The hidden state at [BO_t] predicts t_1, at t_j predicts t_{j+1}; only the target is scored.

MASKED CONTEXT (added 2026-10-28 after the first run). Plain teacher forcing let the model ignore
the source: at step 2,000 of lin_in, held-out loss with the true source was within 0.006 nats of
a source from another row, and identical across the three sources of each target, because the
target's own previous patches predict the next one well. So each training sequence replaces a
random fraction r ~ U(0, 1) of its teacher-forced target inputs with one learned [MASK] embedding
(MAR's masked autoregression; the literature's remedy for exposure bias as well). Evaluation reads
both ends: r = 0 (autoregressive, full context) and r = 1 (every target input masked, so each
target patch is predicted from the source and its position alone: a dense prediction).

VARIANTS (what changes; everything else is identical)
  input side, target = FSQ tokens with cross-entropy (the current design's output):
    tok_in    source as FSQ token ids -> embedding table             (what our model does now)
    lin_in    source as raw patches -> per-modality Linear(16*16*C -> d) + LayerNorm   (Fuyu/EVE)
    linraw_in lin_in WITHOUT the LayerNorm (added 2026-10-28): the LayerNorm rescales each patch
              embedding, which can erase a near-constant patch's level -- and a DEM patch is
              almost only its level (one PCA component holds > 99.9% of its variance).
    linrel_in linraw_in on a level + relief view of each band (added 2026-10-28, after linraw_in
              did not fix DEM): the band's patch mean, and its de-meaned patch divided by that
              band's typical within-patch spread (train rows). z-scored with dataset statistics, a
              DEM patch's relief is ~0.01 z beside a level of ~1 z, invisible to a linear map.
    band_in   source as raw patches -> ONE modality-agnostic embedder: every channel's 16x16
              patch through a shared Linear(256 -> d), FiLM-ed by a learned channel embedding,
              pooled over channels by attention with a learned query (Panopticon/DOFA-like).
              A new band needs a channel embedding, not a new tokenizer.
    bandrop_in  band_in trained with random band dropout on multi-band sources (S2L2A, S1RTC):
              per batch keep each band with probability p ~ U(0.25, 1), at least one. Tests
              whether one embedder can then take any band subset (PASTIS's 10 bands, RGB+NIR...).
  output side, source = raw patches as in lin_in, target = raw patches (teacher-forced as raw
  patches through the same per-modality Linear):
    reg_out   MLP head -> the 16*16*C patch, MSE in z units                    (AIM/EarthPT)
    flow_out  the same MLP trunk as a per-token flow-matching head (MAR-style, AdaLN on the
              hidden state and t), x-prediction with the v-loss (JiT), 4 noise draws per token;
              sampled with a 32-step Euler ODE
    flowdct_out  the flow head of flow_out, but generating each band's --dct-keep lowest 2-D DCT
              frequencies (u+v order) instead of its 256 pixels: a FIXED, untrained, band-agnostic
              transform that brings the head's dimension down (S2L2A 3,072 -> 12 x keep). The
              teacher-forced target inputs are the same low-passed patches the model will feed
              back at generation; scores are against the full-resolution truth.
    LULC is categorical in all three: the head emits 16*16*10 logits, per-pixel cross-entropy.

    python eo/experiments/cio_train.py train --variant lin_in --gpu titanv
    python eo/experiments/cio_train.py eval  --run <dir>              # teacher-forced, all 12 pairs
    python eo/experiments/cio_train.py rollout --run <dir> --pairs S1RTC:S2L2A ...   # AR generation

Runs land in /data/enric/experiments/continuous_io/runs/<variant>_s<seed>/. A run resumes from
its last checkpoint (every 1,000 steps), so a watchdog may kill and restart it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ROOT = Path("/data/enric/experiments/continuous_io")
DATA = ROOT / "data"
MODS = ("S2L2A", "S1RTC", "DEM", "LULC")
NCH = {"S2L2A": 12, "S1RTC": 2, "DEM": 1, "LULC": 10}
CODEBOOK = {"S2L2A": 15360, "S1RTC": 15360, "DEM": 15360, "LULC": 4375}
P, G = 16, 14
NP = G * G
DIM = {m: NCH[m] * P * P for m in MODS}
PAIRS = [(s, t) for s in MODS for t in MODS if s != t]
CONT = ("S2L2A", "S1RTC", "DEM")          # continuous-valued modalities; LULC is categorical
VARIANTS = ("tok_in", "lin_in", "linraw_in", "linrel_in", "band_in", "bandrop_in", "reg_out", "flow_out",
            "flowdct_out")


# ----------------------------------------------------------------------------- data
def patchify(x: torch.Tensor) -> torch.Tensor:
    """(B, C, 224, 224) -> (B, 196, C*256): row-major patches, each flattened (C, 16, 16)."""
    B, C = x.shape[:2]
    return x.reshape(B, C, G, P, G, P).permute(0, 2, 4, 1, 3, 5).reshape(B, NP, C * P * P)


def unpatchify(p: torch.Tensor, C: int) -> torch.Tensor:
    B = p.shape[0]
    return p.reshape(B, G, G, C, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, C, G * P, G * P)


def pixels(m: str, a: torch.Tensor) -> torch.Tensor:
    """A stored batch -> (B, C, 224, 224) float: z-scores, or LULC one-hot over 10 classes."""
    if m == "LULC":
        return F.one_hot(a.long(), NCH["LULC"]).permute(0, 3, 1, 2).float()
    return a.float()


class Arrays:
    def __init__(self, split: str):
        self.x = {m: np.load(DATA / f"{split}_{m}.npy", mmap_mode="r") for m in MODS}
        self.tok = np.load(DATA / f"{split}_tok.npy", mmap_mode="r")
        self.n = len(self.tok)

    def batch(self, idx, s: str, t: str) -> dict:
        idx = np.sort(np.asarray(idx))
        out = {"src": s, "tgt": t, "rows": torch.from_numpy(idx)}
        for role, m in (("s", s), ("t", t)):
            out[f"x{role}"] = torch.from_numpy(np.ascontiguousarray(self.x[m][idx]))
            out[f"k{role}"] = torch.from_numpy(self.tok[idx, MODS.index(m)].astype(np.int64))
        return out


class Stream(IterableDataset):
    """Endless random batches, one (source, target) pair per batch. Seeded per worker."""

    def __init__(self, batch: int, seed: int, step0: int):
        self.batch, self.seed, self.step0 = batch, seed, step0

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        rng = np.random.default_rng([self.seed, wid, self.step0])
        arr = Arrays("train")
        while True:
            s, t = PAIRS[rng.integers(len(PAIRS))]
            yield arr.batch(rng.choice(arr.n, self.batch, replace=False), s, t)


# ----------------------------------------------------------------------------- model
class DCTCodec(nn.Module):
    """Per band: orthonormal 16x16 DCT-II, keep the `keep` lowest frequencies (u+v, then u)."""

    def __init__(self, keep: int):
        super().__init__()
        u, x = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
        m = np.cos(np.pi * (2 * x + 1) * u / (2 * P)) * np.sqrt(2.0 / P)
        m[0] /= np.sqrt(2.0)
        uu, vv = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
        order = np.lexsort((uu.ravel(), (uu + vv).ravel()))[:keep]
        self.keep = keep
        self.register_buffer("M", torch.from_numpy(m).float(), persistent=False)
        self.register_buffer("order", torch.from_numpy(order).long(), persistent=False)

    def encode(self, p, C):
        """(..., C*256) patches -> (..., C*keep) coefficients."""
        sh = p.shape[:-1]
        x = p.float().reshape(-1, C, P, P)
        c = torch.einsum("ux,ncxy,vy->ncuv", self.M, x, self.M).reshape(-1, C, P * P)
        return c[:, :, self.order].reshape(*sh, C * self.keep)

    def decode(self, c, C):
        sh = c.shape[:-1]
        full = torch.zeros(c.reshape(-1, C, self.keep).shape[0], C, P * P, device=c.device)
        full[:, :, self.order] = c.float().reshape(-1, C, self.keep)
        x = torch.einsum("ux,ncuv,vy->ncxy", self.M, full.reshape(-1, C, P, P), self.M)
        return x.reshape(*sh, C * P * P)

    def lowpass(self, p, C):
        return self.decode(self.encode(p, C), C)


class ResBlock(nn.Module):
    """MAR's SimpleMLPAdaLN block: LN -> AdaLN shift/scale -> MLP -> gated residual."""

    def __init__(self, w: int, cond: bool):
        super().__init__()
        self.ln = nn.LayerNorm(w, elementwise_affine=not cond, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(w, 3 * w)) if cond else None
        if cond:
            nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c=None):
        if self.ada is None:
            return x + self.mlp(self.ln(x))
        shift, scale, gate = self.ada(c).chunk(3, -1)
        return x + gate * self.mlp(self.ln(x) * (1 + scale) + shift)


class MLPHead(nn.Module):
    """Hidden state -> one patch (regression values, or LULC per-pixel logits)."""

    def __init__(self, d: int, out: int, w: int = 768, n: int = 3):
        super().__init__()
        self.inp = nn.Linear(d, w)
        self.blocks = nn.ModuleList([ResBlock(w, cond=False) for _ in range(n)])
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, out))

    def forward(self, h):
        x = self.inp(h)
        for b in self.blocks:
            x = b(x)
        return self.out(x)


def t_embed(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = (t.float() * 1000.0)[..., None] * f
    return torch.cat([a.cos(), a.sin()], -1)


class FlowHead(nn.Module):
    """Per-token flow matching (MAR's diffusion head, rectified flow, JiT's x-prediction).

    x_t = t x + (1 - t) e;  the net predicts x;  loss on v = (x_hat - x_t) / (1 - t) vs x - e.
    """

    def __init__(self, d: int, dim: int, w: int = 768, n: int = 3):
        super().__init__()
        self.inp = nn.Linear(dim, w)
        self.cond = nn.Linear(d, w)
        self.temb = nn.Sequential(nn.Linear(256, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList([ResBlock(w, cond=True) for _ in range(n)])
        self.fln = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.fada = nn.Sequential(nn.SiLU(), nn.Linear(w, 2 * w))
        self.out = nn.Linear(w, dim)
        nn.init.zeros_(self.fada[1].weight); nn.init.zeros_(self.fada[1].bias)

    def forward(self, xt, t, h):
        c = self.cond(h) + self.temb(t_embed(t))
        x = self.inp(xt)
        for b in self.blocks:
            x = b(x, c)
        shift, scale = self.fada(c).chunk(2, -1)
        return self.out(self.fln(x) * (1 + scale) + shift)

    def loss(self, x, h, draws: int = 4, max_rows: int = 25088):
        """x: (N, dim) targets, h: (N, d). Mean v-loss over `draws` noise draws per token,
        capped at max_rows (token, draw) pairs drawn at random: 4 draws of a 64 x 196 batch of
        3,072-dim S2L2A patches do not fit a 12 GB card, 2 do."""
        x = x.repeat(draws, 1); h = h.repeat(draws, 1)
        if x.shape[0] > max_rows:
            keep = torch.randperm(x.shape[0], device=x.device)[:max_rows]
            x, h = x[keep], h[keep]
        t = torch.sigmoid(torch.randn(x.shape[0], device=x.device))      # logit-normal(0, 1)
        e = torch.randn_like(x)
        xt = t[:, None] * x + (1 - t[:, None]) * e
        xh = self(xt, t, h).float()
        v_hat = (xh - xt) / (1 - t[:, None]).clamp_min(0.05)
        return ((v_hat - (x - e)) ** 2).mean()

    @torch.no_grad()
    def sample(self, h, steps: int = 32, gen: torch.Generator | None = None):
        x = torch.randn(h.shape[0], self.out.out_features, device=h.device, generator=gen)
        ts = torch.linspace(0, 1, steps + 1, device=h.device)
        for i in range(steps):
            t = ts[i].expand(h.shape[0])
            xh = self(x, t, h).float()
            if i == steps - 1:
                return xh
            v = (xh - x) / (1 - ts[i]).clamp_min(0.05)
            x = x + (ts[i + 1] - ts[i]) * v
        return x


class BandEmbed(nn.Module):
    """Modality-agnostic patch embedding: any set of channels -> one d-vector per patch.

    Each channel's 16x16 patch goes through ONE shared Linear(256 -> d), is FiLM-ed by its
    channel embedding, and the channels are pooled by multi-head attention with a learned
    query. Nothing in it is specific to a modality except the channel embedding table.
    """

    def __init__(self, d: int, n_channels: int, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(P * P, d)
        self.chan = nn.Embedding(n_channels, d)
        self.film = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, x, chan_ids):
        """x: (B, 196, C*256) as patchify() makes it; chan_ids: (C,) long."""
        B, N, _ = x.shape
        C = len(chan_ids)
        u = self.proj(x.reshape(B * N, C, P * P))                        # (BN, C, d)
        e = self.chan(chan_ids)[None]                                     # (1, C, d)
        g, b = self.film(e).chunk(2, -1)
        u = u * (1 + g) + b + e
        q = self.query.expand(B * N, 1, -1)
        pooled, _ = self.attn(q, u, u, need_weights=False)
        return self.ln(pooled.reshape(B, N, -1))


CHAN_BASE = {m: sum(NCH[k] for k in MODS[:i]) for i, m in enumerate(MODS)}


class CIOModel(nn.Module):
    def __init__(self, variant: str, d=384, layers=8, heads=6, kv=2, ffn=1024, dct_keep: int = 16):
        super().__init__()
        from transformers import LlamaConfig, LlamaModel
        assert variant in VARIANTS, variant
        self.variant = variant
        cfg = LlamaConfig(hidden_size=d, intermediate_size=ffn, num_hidden_layers=layers,
                          num_attention_heads=heads, num_key_value_heads=kv, vocab_size=8,
                          max_position_embeddings=512, rope_theta=10000.0,
                          tie_word_embeddings=False, attn_implementation="sdpa")
        self.backbone = LlamaModel(cfg)          # its own embed_tokens is unused (inputs_embeds)
        self.special = nn.Embedding(2 * len(MODS), d)        # BO/EO per modality
        self.mask_emb = nn.Parameter(torch.randn(d))         # replaces masked target inputs
        self.out_kind = ("tok" if variant in ("tok_in", "lin_in", "linraw_in", "linrel_in", "band_in", "bandrop_in")
                         else variant[:-4])
        if variant == "tok_in" or self.out_kind == "tok":
            self.tok = nn.ModuleDict({m: nn.Embedding(CODEBOOK[m], d) for m in MODS})
        self.dct = DCTCodec(dct_keep) if variant == "flowdct_out" else None
        if self.dct is not None:
            # per-coefficient scale (the DC term is 16x a patch mean): the flow head sees unit-
            # variance coefficients. From the first 256 train rows; persistent, so a loaded run
            # uses the values it was trained with.
            tr = Arrays("train")
            for m in CONT:
                y = patchify(torch.from_numpy(tr.x[m][:256].astype(np.float32)))
                sd = self.dct.encode(y, NCH[m]).reshape(-1, NCH[m] * dct_keep).std(0)
                self.register_buffer(f"dct_scale_{m}", sd.clamp_min(1e-3))
        if variant in ("lin_in", "reg_out", "flow_out", "flowdct_out"):
            self.lin = nn.ModuleDict({m: nn.Sequential(nn.Linear(DIM[m], d), nn.LayerNorm(d))
                                      for m in MODS})
        if variant == "linraw_in":
            self.lin = nn.ModuleDict({m: nn.Linear(DIM[m], d) for m in MODS})
        if variant == "linrel_in":
            self.lin = nn.ModuleDict({m: nn.Linear(DIM[m] + (NCH[m] if m in CONT else 0), d) for m in MODS})
            tr = Arrays("train")
            for m in CONT:          # per-band std of the within-patch deviations, first 256 train rows
                x = patchify(torch.from_numpy(tr.x[m][:256].astype(np.float32))).reshape(-1, NP, NCH[m], P * P)
                sd = (x - x.mean(-1, keepdim=True)).std(dim=(0, 1, 3))
                self.register_buffer(f"rel_scale_{m}", sd.clamp_min(1e-4))
        if variant in ("band_in", "bandrop_in"):
            self.band = BandEmbed(d, sum(NCH.values()))
        if self.out_kind == "tok":
            self.head = nn.ModuleDict({m: nn.Linear(d, CODEBOOK[m], bias=False) for m in MODS})
            for h in self.head.values():
                nn.init.normal_(h.weight, std=0.02)
        elif self.out_kind == "reg":
            self.head = nn.ModuleDict({m: MLPHead(d, DIM[m]) for m in MODS})
        else:
            fdim = (lambda m: NCH[m] * dct_keep) if self.dct is not None else (lambda m: DIM[m])
            self.head = nn.ModuleDict({m: (FlowHead(d, fdim(m)) if m in CONT else MLPHead(d, DIM[m]))
                                       for m in MODS})

    def input_view(self, t, y):
        """What the model is fed for target patches y: the patches, or their DCT low-pass."""
        if self.dct is not None and t in CONT:
            return self.dct.lowpass(y, NCH[t])
        return y

    def sample_patches(self, t, h, gen=None):
        """A continuous target patch per hidden state: regression mean or a flow sample."""
        if self.out_kind == "reg":
            return self.head[t](h).float()
        x = self.head[t].sample(h, gen=gen)
        if self.dct is None:
            return x
        return self.dct.decode(x * getattr(self, f"dct_scale_{t}"), NCH[t])

    def embed_source(self, b, dev):
        s = b["src"]
        if self.variant == "tok_in":
            return self.tok[s](b["ks"].to(dev))
        xs = b["xs"].to(dev)
        if self.variant in ("band_in", "bandrop_in"):
            # src_bands: evaluate on a SUBSET of the source's channels; only this embedder can take
            # one, since it pools over whatever channels arrive. bandrop_in trains on subsets.
            keep = getattr(self, "src_bands", None) if s == "S2L2A" else None
            if self.training and self.variant == "bandrop_in" and s in ("S2L2A", "S1RTC"):
                p_keep = float(torch.empty(1).uniform_(0.25, 1.0))
                sel = torch.rand(NCH[s]) < p_keep
                if not sel.any():
                    sel[torch.randint(NCH[s], (1,))] = True
                keep = sel.nonzero()[:, 0].tolist()
            ch = torch.arange(NCH[s], device=dev) if keep is None else torch.as_tensor(keep, device=dev)
            x = patchify(pixels(s, xs)[:, ch])
            return self.band(x, ch + CHAN_BASE[s])
        x = patchify(pixels(s, xs))
        if self.variant == "linrel_in" and s in CONT:
            B = x.shape[0]
            x = x.reshape(B, NP, NCH[s], P * P)
            mu = x.mean(-1, keepdim=True)
            rel = (x - mu) / getattr(self, f"rel_scale_{s}").reshape(1, 1, -1, 1)
            x = torch.cat([mu.reshape(B, NP, -1), rel.reshape(B, NP, -1)], -1)
        return self.lin[s](x)

    def embed_target_inputs(self, t, tgt_vals, dev):
        """The 195 teacher-forced target inputs: token ids (tok out) or patches (B, 195, dim)."""
        if self.out_kind == "tok":
            return self.tok[t](tgt_vals)
        return self.lin[t](tgt_vals)

    def hidden(self, b, tgt_in, dev, mask_rate=0.0):
        """mask_rate: a float, or (B,) per-sequence rates; each target input is masked w.p. rate."""
        s, t = b["src"], b["tgt"]
        B = tgt_in.shape[0]
        rate = torch.as_tensor(mask_rate, dtype=torch.float32, device=tgt_in.device)
        if tgt_in.shape[1] and bool((rate > 0).any()):
            m = torch.rand(B, tgt_in.shape[1], device=tgt_in.device) < rate.reshape(-1, 1)
            tgt_in = torch.where(m[..., None], self.mask_emb.to(tgt_in.dtype), tgt_in)
        sp = self.special.weight
        bo = lambda m: sp[2 * MODS.index(m)].expand(B, 1, -1)
        eo = lambda m: sp[2 * MODS.index(m) + 1].expand(B, 1, -1)
        seq = torch.cat([bo(s), self.embed_source(b, dev), eo(s), bo(t), tgt_in], 1)
        out = self.backbone(inputs_embeds=seq).last_hidden_state
        return out[:, NP + 2:]                    # positions [BO_t] .. t_195 -> predict t_1..t_196

    def target(self, b, dev):
        """Ground-truth target: token ids (B,196) or patches (B,196,dim)."""
        if self.out_kind == "tok":
            return b["kt"].to(dev)
        return patchify(pixels(b["tgt"], b["xt"].to(dev)))

    def loss(self, b, dev, mask_rate=None):
        """mask_rate None: training, r ~ U(0, 1) per sequence."""
        t = b["tgt"]
        y = self.target(b, dev)
        if mask_rate is None:
            mask_rate = torch.rand(y.shape[0], device=dev)
        yin = y if self.out_kind == "tok" else self.input_view(t, y)
        h = self.hidden(b, self.embed_target_inputs(t, yin[:, :-1], dev), dev, mask_rate)
        if self.out_kind == "tok":
            logits = self.head[t](h).float()
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if t == "LULC":
            return lulc_ce(self.head[t](h).float(), b["xt"].to(dev))
        if self.out_kind == "reg":
            return F.mse_loss(self.head[t](h).float(), y)
        if self.dct is not None:
            y = self.dct.encode(y, NCH[t]) / getattr(self, f"dct_scale_{t}")
        return self.head[t].loss(y.reshape(-1, y.shape[-1]), h.reshape(-1, h.shape[-1]))


def lulc_ce(logits, classes):
    """logits (B, 196, 2560) laid out as patchify() lays out the one-hot; classes (B, 224, 224)."""
    lg = unpatchify(logits, NCH["LULC"])                       # (B, 10, 224, 224)
    return F.cross_entropy(lg, classes.long())


# ----------------------------------------------------------------------------- train
def run_dir(variant: str, seed: int) -> Path:
    return ROOT / "runs" / f"{variant}_s{seed}"


def check_gpu(alias: str) -> str:
    name = torch.cuda.get_device_name(0)
    want = {"titanv": "TITAN V", "titanx": "TITAN X"}[alias]
    if want not in name:
        raise SystemExit(f"expected a {want}, CUDA gave {name!r}")
    return name


def cmd_train(a) -> int:
    torch.manual_seed(a.seed)
    out = run_dir(a.variant, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    gpu = check_gpu(a.gpu)
    dev = torch.device("cuda:0")
    fp16 = a.gpu == "titanv"                 # Volta has fp16 tensor cores; Maxwell does not
    model = CIOModel(a.variant, dct_keep=a.dct_keep).to(dev)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "special" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "special" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    lr_at = lambda s: a.lr * min(1.0, (s + 1) / a.warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step, log = 0, []
    ck = out / "last.pt"
    if ck.exists():
        # to CPU: the RNG states must stay CPU ByteTensors, and load_state_dict moves the model
        # and optimizer tensors to their parameters' device (2026-10-28: map_location=dev broke
        # the first watchdog retry with "RNG state must be a torch.ByteTensor")
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"]); step, log = st["step"], st["log"]
        torch.set_rng_state(st["rng"]); torch.cuda.set_rng_state(st["cuda_rng"])
        print(f"resumed at step {step}", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    cfg = {"variant": a.variant, "seed": a.seed, "steps": a.steps, "batch": a.batch, "lr": a.lr,
           "dct_keep": a.dct_keep,
           "warmup": a.warmup, "gpu": gpu, "fp16": fp16, "params": n_par, "backbone_params": n_bb,
           "data": json.loads((DATA / "meta.json").read_text())["created"]}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg), flush=True)
    dl = DataLoader(Stream(a.batch, a.seed, step), batch_size=None, num_workers=a.workers,
                    prefetch_factor=4, persistent_workers=True)
    it = iter(dl)
    test = Arrays("test")
    model.train()
    t0, acc = time.time(), {}
    while step < a.steps:
        b = next(it)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        n_b, loss = len(b["rows"]), 0.0
        for i in range(0, n_b, a.micro or n_b):       # micro-batches: same step, less memory
            sub = {k: (v[i:i + (a.micro or n_b)] if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                part = model.loss(sub, dev) * (len(sub["rows"]) / n_b)
            scaler.scale(part).backward()
            loss += part.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        key = f"{b['src']}>{b['tgt']}"
        acc.setdefault(key, []).append(loss)
        if step % 100 == 0:
            rate = 100 / (time.time() - t0); t0 = time.time()
            mean = {k: float(np.mean(v)) for k, v in acc.items()}
            log.append({"step": step, "lr": lr_at(step), "gn": float(gn), "loss": mean})
            print(f"step {step} lr {lr_at(step):.2e} gn {float(gn):.2f} {rate:.2f} it/s "
                  + " ".join(f"{k} {v:.3f}" for k, v in sorted(mean.items())), flush=True)
            acc = {}
        if step % a.eval_every == 0 or step == a.steps:
            log.append({"step": step, "heldout": heldout_loss(model, test, dev, fp16)})
            print(f"step {step} held-out " + " ".join(
                f"{k} {v:.3f}" for k, v in sorted(log[-1]["heldout"].items())), flush=True)
            model.train()
        if step % 1000 == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "log": log,
                        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
                       out / "last.tmp")
            os.replace(out / "last.tmp", ck)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "train_log.json").write_text(json.dumps(log))
    print("done", flush=True)
    return 0


@torch.no_grad()
def heldout_loss(model, test: "Arrays", dev, fp16: bool, n: int = 128) -> dict:
    """The training loss on the first n held-out rows, per pair, at mask rate 0 (AR) and 1
    (source only). The RNG is forked, so noise draws here do not move the training stream."""
    model.eval()
    out = {}
    with torch.random.fork_rng(devices=[dev]):
        torch.manual_seed(0)
        for s, t in PAIRS:
            for r in (0.0, 1.0):
                vals = []
                for i in range(0, n, 64):
                    b = test.batch(np.arange(i, min(n, i + 64)), s, t)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                        vals.append(model.loss(b, dev, mask_rate=r).item())
                out[f"{s}>{t}@{int(r)}"] = float(np.mean(vals))
    return out


# ----------------------------------------------------------------------------- evaluation
def load_run(path: Path, dev):
    cfg = json.loads((path / "config.json").read_text())
    model = CIOModel(cfg["variant"], dct_keep=cfg.get("dct_keep", 16)).to(dev).eval()
    model.load_state_dict(torch.load(path / "model.pt", map_location=dev, weights_only=True))
    return cfg, model


def hf_energy(x: torch.Tensor) -> torch.Tensor:
    """Share of spectral power above 1/4 of Nyquist, per image and channel: (B, C)."""
    f = torch.fft.rfft2(x.float() - x.float().mean((-2, -1), keepdim=True), norm="ortho")
    pw = f.abs() ** 2
    fy = torch.fft.fftfreq(x.shape[-2], device=x.device).abs()[:, None]
    fx = torch.fft.rfftfreq(x.shape[-1], device=x.device)[None, :]
    hi = (torch.maximum(fy, fx) > 0.125).float()
    return (pw * hi).sum((-2, -1)), pw.sum((-2, -1))


def lulc_scores(pred, true, conf):
    """Accumulate a 10x10 confusion matrix from predicted and true class maps."""
    idx = true.reshape(-1).long() * 10 + pred.reshape(-1).long()
    conf += torch.bincount(idx, minlength=100).reshape(10, 10).cpu()


def miou(conf):
    conf = conf.double()
    inter = conf.diag()
    union = conf.sum(0) + conf.sum(1) - inter
    present = conf.sum(1) > 0
    return float((inter[present] / union[present].clamp_min(1)).mean()), float(inter.sum() / conf.sum())


@torch.no_grad()
def cmd_eval(a) -> int:
    """Teacher-forced metrics on the held-out rows, every pair. Per-row values saved for pairing."""
    dev = torch.device("cuda:0")
    check_gpu(a.gpu)
    path = Path(a.run)
    cfg, model = load_run(path, dev)
    if a.src_bands:
        assert cfg["variant"] in ("band_in", "bandrop_in"), "--src-bands needs the band-agnostic embedder"
        model.src_bands = [int(x) for x in a.src_bands.split(",")]
    arr = Arrays("test")
    n = min(a.n, arr.n)
    res, per_row = {}, {}
    gen = torch.Generator(device=dev).manual_seed(0)
    for s, t in PAIRS:
        if a.src_bands and s != "S2L2A":
            continue
        key = f"{s}>{t}"
        rows_ce, rows_acc, rows_mse, rows_es, rows_mse_ens = [], [], [], [], []
        conf = torch.zeros(10, 10, dtype=torch.long)
        for i in range(0, n, a.batch):
            b = arr.batch(np.arange(i, min(n, i + a.batch)), s, t)
            if a.shuffle_source:    # the control: every row gets the next row's source
                b["xs"], b["ks"] = b["xs"].roll(1, 0), b["ks"].roll(1, 0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.gpu == "titanv"):
                y = model.target(b, dev)
                yin = y if model.out_kind == "tok" else model.input_view(t, y)
                h = model.hidden(b, model.embed_target_inputs(t, yin[:, :-1], dev), dev, a.mask_rate).float()
            if model.out_kind == "tok":
                lg = model.head[t](h).float()
                ce = F.cross_entropy(lg.transpose(1, 2), y, reduction="none")      # (B, 196)
                rows_ce.append(ce.mean(1).cpu()); rows_acc.append((lg.argmax(-1) == y).float().mean(1).cpu())
                continue
            if t == "LULC":
                lg = unpatchify(model.head[t](h).float(), 10)
                rows_ce.append(F.cross_entropy(lg, b["xt"].to(dev).long(), reduction="none").mean((1, 2)).cpu())
                pred = lg.argmax(1)
                rows_acc.append((pred == b["xt"].to(dev).long()).float().mean((1, 2)).cpu())
                lulc_scores(pred, b["xt"].to(dev), conf)
                continue
            if model.out_kind == "reg":
                mu = model.head[t](h).float()
                err = ((mu - y) ** 2).mean(-1)                                     # (B, 196)
                rows_mse.append(err.mean(1).cpu()); rows_mse_ens.append(err.mean(1).cpu())
                rows_es.append((mu - y).norm(dim=-1).mean(1).cpu())
            else:
                B = h.shape[0]
                hs = h.reshape(-1, h.shape[-1])
                xs = torch.stack([model.sample_patches(t, hs, gen).reshape(B, NP, -1)
                                  for _ in range(a.k)])                            # (K, B, 196, dim)
                rows_mse.append(((xs[0] - y) ** 2).mean((-1, -2)).cpu())
                rows_mse_ens.append(((xs.mean(0) - y) ** 2).mean((-1, -2)).cpu())
                d_obs = (xs - y[None]).norm(dim=-1).mean(0)                        # (B, 196)
                d_pair = torch.stack([(xs[k] - xs[l]).norm(dim=-1)
                                      for k in range(a.k) for l in range(a.k) if k != l]).mean(0)
                rows_es.append((d_obs - 0.5 * d_pair).mean(1).cpu())
        r = {}
        if rows_ce:
            r["ce"] = float(torch.cat(rows_ce).mean()); per_row[f"{key}:ce"] = torch.cat(rows_ce).tolist()
        if rows_acc:
            r["acc"] = float(torch.cat(rows_acc).mean()); per_row[f"{key}:acc"] = torch.cat(rows_acc).tolist()
        if rows_mse:
            r["mse_one_sample"] = float(torch.cat(rows_mse).mean())
            r["mse_mean_pred"] = float(torch.cat(rows_mse_ens).mean())
            r["energy_score"] = float(torch.cat(rows_es).mean())
            per_row[f"{key}:es"] = torch.cat(rows_es).tolist()
            per_row[f"{key}:mse"] = torch.cat(rows_mse_ens).tolist()
        if conf.sum() > 0:
            r["miou"], r["pix_acc"] = miou(conf)
        res[key] = r
        print(key, json.dumps(r), flush=True)
    tag = ("tf" if a.mask_rate == 0 else f"mask{a.mask_rate:g}") + ("_shufsrc" if a.shuffle_source else "") \
        + (f"_bands{a.src_bands.replace(',', '-')}" if a.src_bands else "")
    (path / f"eval_{tag}.json").write_text(json.dumps({"n": n, "k": a.k, "mask_rate": a.mask_rate,
                                                        "results": res}, indent=1))
    (path / f"eval_{tag}_rows.json").write_text(json.dumps(per_row))
    return 0


@torch.no_grad()
def cmd_rollout(a) -> int:
    """Autoregressive generation of the whole target, given the source. No KV cache: each step
    re-runs the prefix (394 positions), which is cheap at this size. Saves the generated target
    (token ids, or z-score patches) so tokens can be decoded to pixels in the `mor` env."""
    dev = torch.device("cuda:0")
    check_gpu(a.gpu)
    path = Path(a.run)
    cfg, model = load_run(path, dev)
    arr = Arrays("test")
    n = min(a.n, arr.n)
    outdir = path / "rollout"
    outdir.mkdir(exist_ok=True)
    for spec in a.pairs:
        s, t = spec.split(":")
        for k in range(a.k):
            f = outdir / f"{s}_{t}_k{k}_{a.mode}{'_par' if a.parallel else ''}.npy"
            if f.exists():
                print("[skip]", f.name); continue
            gen = torch.Generator(device=dev).manual_seed(1000 + k)
            outs = []
            t0 = time.time()
            for i in range(0, n, a.batch):
                b = arr.batch(np.arange(i, min(n, i + a.batch)), s, t)
                B = len(b["rows"])
                if a.parallel:          # every target input masked: one pass, all 196 at once
                    outs.append(parallel_predict(model, b, t, a.mode, gen, dev, a.gpu == "titanv"))
                    continue
                if model.out_kind == "tok":
                    cur = torch.zeros(B, 0, dtype=torch.long, device=dev)
                else:
                    cur = torch.zeros(B, 0, DIM[t], device=dev)
                for j in range(NP):
                    tin = cur if j < NP else cur[:, :-1]
                    with torch.autocast("cuda", dtype=torch.float16, enabled=a.gpu == "titanv"):
                        emb = model.embed_target_inputs(t, tin, dev) if j > 0 else \
                            torch.zeros(B, 0, model.special.weight.shape[1], device=dev)
                        h = model.hidden(b, emb, dev)[:, j].float()
                    if model.out_kind == "tok":
                        lg = model.head[t](h).float()
                        nxt = lg.argmax(-1) if a.mode == "greedy" else \
                            torch.multinomial(torch.softmax(lg, -1), 1, generator=gen)[:, 0]
                        cur = torch.cat([cur, nxt[:, None]], 1)
                    elif t == "LULC":
                        lg = model.head[t](h).float().reshape(B, 10, P * P)
                        if a.mode == "greedy":
                            cls = lg.argmax(1)
                        else:
                            cls = torch.multinomial(torch.softmax(lg.transpose(1, 2).reshape(-1, 10), -1), 1,
                                                    generator=gen).reshape(B, P * P)
                        cur = torch.cat([cur, F.one_hot(cls, 10).transpose(1, 2).reshape(B, 1, -1).float()], 1)
                    elif model.out_kind == "reg":
                        cur = torch.cat([cur, model.head[t](h).float()[:, None]], 1)
                    else:
                        cur = torch.cat([cur, model.sample_patches(t, h, gen)[:, None]], 1)
                outs.append(cur.cpu().numpy().astype(np.int64 if model.out_kind == "tok" else np.float16))
            out = np.concatenate(outs)
            if model.out_kind != "tok" and t == "LULC" and not a.parallel:
                out = unpatchify(torch.from_numpy(out.astype(np.float32)), 10).argmax(1).numpy().astype(np.uint8)
            np.save(f, out)
            print(f"{spec} k{k} {a.mode}: {out.shape} in {time.time() - t0:.0f} s", flush=True)
    return 0


@torch.no_grad()
def parallel_predict(model, b, t, mode, gen, dev, fp16):
    """All 196 target patches from the source alone (mask rate 1). Tokens: argmax or a sample
    per patch; regression: the mean; flow: one independent sample per patch; LULC (continuous
    variants): per-pixel argmax, returned as a (B, 224, 224) class map."""
    y = model.target(b, dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        h = model.hidden(b, model.embed_target_inputs(t, y[:, :-1], dev), dev, 1.0).float()
    B = h.shape[0]
    if model.out_kind == "tok":
        lg = model.head[t](h).float()
        if mode == "greedy":
            return lg.argmax(-1).cpu().numpy()
        pr = torch.softmax(lg, -1).reshape(-1, lg.shape[-1])
        return torch.multinomial(pr, 1, generator=gen).reshape(B, NP).cpu().numpy()
    if t == "LULC":
        return unpatchify(model.head[t](h).float(), 10).argmax(1).cpu().numpy().astype(np.uint8)
    if model.out_kind == "reg":
        return model.head[t](h).float().cpu().numpy().astype(np.float16)
    xs = model.sample_patches(t, h.reshape(-1, h.shape[-1]), gen).reshape(B, NP, -1)
    return xs.cpu().numpy().astype(np.float16)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    tr = sp.add_parser("train")
    tr.add_argument("--variant", choices=VARIANTS, required=True)
    tr.add_argument("--gpu", choices=("titanv", "titanx"), required=True)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--steps", type=int, default=8000)
    tr.add_argument("--batch", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--warmup", type=int, default=500)
    tr.add_argument("--workers", type=int, default=2)
    tr.add_argument("--eval-every", type=int, default=1000)
    tr.add_argument("--dct-keep", type=int, default=16, help="flowdct_out: DCT coefficients per band")
    tr.add_argument("--micro", type=int, default=0, help="micro-batch size (0 = the whole batch)")
    ev = sp.add_parser("eval")
    ev.add_argument("--run", required=True)
    ev.add_argument("--gpu", choices=("titanv", "titanx"), required=True)
    ev.add_argument("--n", type=int, default=1024)
    ev.add_argument("--batch", type=int, default=32)
    ev.add_argument("--k", type=int, default=4)
    ev.add_argument("--mask-rate", type=float, default=0.0)
    ev.add_argument("--shuffle-source", action="store_true", help="control: another row's source")
    ev.add_argument("--src-bands", default=None, help="band_in only: S2L2A channel indices, e.g. 1,2,3,7")
    ro = sp.add_parser("rollout")
    ro.add_argument("--run", required=True)
    ro.add_argument("--gpu", choices=("titanv", "titanx"), required=True)
    ro.add_argument("--pairs", nargs="+", required=True)
    ro.add_argument("--n", type=int, default=256)
    ro.add_argument("--batch", type=int, default=64)
    ro.add_argument("--k", type=int, default=2)
    ro.add_argument("--mode", choices=("sample", "greedy"), default="sample")
    ro.add_argument("--parallel", action="store_true", help="mask every target input: one pass")
    a = p.parse_args()
    return {"train": cmd_train, "eval": cmd_eval, "rollout": cmd_rollout}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
