"""A small dedicated tokenizer for label maps rendered as RGB (Paper A, A2 Step 0, short test).

Question: can a map tokenizer trained for the job beat Cosmos's round-trip ceiling on PASTIS at
the same token grid, and with how many codes?

Design (palette-agnostic within one universal palette):
  - a universal palette of K maximally distinct colours (greedy farthest point, as `distinct`);
  - every training map gets a random injective class -> palette-colour assignment, so the
    tokenizer learns geometry and K colours, never one dataset's classes;
  - input: the RGB render at 224 px (nearest upsampling from the native grid);
  - encoder: strided convolutions down to 14 x 14 (f16) or 28 x 28 (f8), FSQ bottleneck;
  - decoder: mirrored, ending in K logits per pixel (cross-entropy), so decoding needs no snap.

Train: PASTIS folds 1-3 (+ TerraMesh val LULC maps), flips and 90-degree rotations.
Select on fold 4, report on fold 5 (GEO-Bench-2's test fold), with PASTIS classes mapped to the
first 19 universal colours. Void (19) is rendered as the last colour and ignored in the loss
and the score.

Runs in `.venv` on one card (TITAN V at fp16 autocast; see rgb_map_ceiling.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "rgb_map_ceiling", Path(__file__).with_name("rgb_map_ceiling.py"))
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)


class FSQ(nn.Module):
    """Finite scalar quantization (Mentzer et al.), straight-through."""

    def __init__(self, levels):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        basis = np.concatenate([[1], np.cumprod(levels[:-1])]).astype(np.int64)
        self.register_buffer("basis", torch.tensor(basis))
        self.n_codes = int(np.prod(levels))

    def forward(self, z):                                   # z: (B, d, h, w)
        L = self.levels.view(1, -1, 1, 1)
        half_l = (L - 1) * (1 + 1e-3) / 2
        offset = torch.where(L % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        zb = torch.tanh(z.float() + shift) * half_l - offset
        q = zb + (torch.round(zb) - zb).detach()            # integers in [-L//2, (L-1)//2]
        hw = torch.div(L, 2, rounding_mode="floor")
        idx = ((q.detach() + hw).round().long() * self.basis.view(1, -1, 1, 1)).sum(1)
        return q / hw, idx


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class MapAE(nn.Module):
    def __init__(self, K: int, n_down: int, levels, width: int = 64):
        super().__init__()
        chs = [width * min(2 ** i, 4) for i in range(n_down + 1)]
        enc = [nn.Conv2d(3, chs[0], 3, padding=1), block(chs[0], chs[0])]
        for i in range(n_down):
            enc += [nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1), block(chs[i + 1], chs[i + 1])]
        enc += [nn.Conv2d(chs[-1], len(levels), 1)]
        self.enc = nn.Sequential(*enc)
        self.fsq = FSQ(levels)
        dec = [nn.Conv2d(len(levels), chs[-1], 3, padding=1), block(chs[-1], chs[-1])]
        for i in reversed(range(n_down)):
            dec += [nn.ConvTranspose2d(chs[i + 1], chs[i], 4, stride=2, padding=1), block(chs[i], chs[i])]
        dec += [nn.Conv2d(chs[0], K, 1)]
        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        q, idx = self.fsq(self.enc(x))
        return self.dec(q), idx


def render(lab: torch.Tensor, colours: torch.Tensor) -> torch.Tensor:
    """lab (B,H,W) of universal-palette indices -> RGB in [-1,1] at 224 px, (B,3,224,224)."""
    x = colours[lab].permute(0, 3, 1, 2)
    if x.shape[-1] != 224:
        x = F.interpolate(x, size=(224, 224), mode="nearest")
    return x * 2 - 1


def to_native(logits: torch.Tensor, native: int) -> torch.Tensor:
    return F.interpolate(logits, size=(native, native), mode="area") if logits.shape[-1] != native else logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=24, help="universal palette size")
    ap.add_argument("--down", type=int, default=4, help="4: 14x14 tokens, 3: 28x28")
    ap.add_argument("--levels", default="8,5,5,5")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lulc", action="store_true", help="add TerraMesh val LULC maps to training")
    ap.add_argument("--lulc-npy", default="/data/enric/reports/a2_seg/data/lulc_val600.npy")
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in name:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {name}")

    K = a.K
    g = rmc._greedy(K, 11, "lab")                          # [black, white, ...]: K distinct colours
    colours = torch.tensor(np.concatenate([g[:1], g[2:], g[1:2]]), dtype=torch.float32, device=dev)

    # PASTIS maps: class c in 0..18 -> colour c, void (19) -> colour K-1, ignored in loss/score
    tr = torch.from_numpy(rmc.load_pastis(rmc.pastis_ids([1, 2, 3])))
    va = torch.from_numpy(rmc.load_pastis(rmc.pastis_ids([4])))
    te = torch.from_numpy(rmc.load_pastis(rmc.pastis_ids([5])))
    lulc = torch.from_numpy(np.load(a.lulc_npy).astype(np.int64)) if a.lulc else None

    levels = [int(v) for v in a.levels.split(",")]
    model = MapAE(K, a.down, levels, a.width).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    scaler = torch.cuda.amp.GradScaler()
    print(f"MapAE K={K} down={a.down} levels={levels} codes={model.fsq.n_codes} params={n_par:,}",
          flush=True)

    # All maps on the GPU at 128 px; LULC (224 px) resampled once, nearest. A source id per map
    # says which class is ignored there (PASTIS void 19, LULC no data 0).
    pool = [tr]
    ign_of = [torch.full((len(tr),), 19)]
    if lulc is not None:
        lr = F.interpolate(lulc[:, None].float(), size=(128, 128), mode="nearest")[:, 0].long()
        pool.append(lr)
        ign_of.append(torch.full((len(lr),), 0))
    pool = torch.cat(pool).to(dev)
    ign_of = torch.cat(ign_of).to(dev)
    p_lulc = 0.3 if lulc is not None else 0.0
    n_tr = len(tr)
    g = torch.Generator(device=dev).manual_seed(0)

    def draw_batch():
        """Random maps with random injective class -> universal colour assignments, flips and
        90-degree rotations; vectorised on the GPU."""
        B = a.batch
        from_lulc = torch.rand(B, device=dev, generator=g) < p_lulc
        i_tr = torch.randint(0, n_tr, (B,), device=dev, generator=g)
        i_lu = (torch.randint(0, max(len(pool) - n_tr, 1), (B,), device=dev, generator=g) + n_tr)
        idx = torch.where(from_lulc, i_lu, i_tr) if p_lulc > 0 else i_tr
        m = pool[idx]                                              # (B,128,128), class ids < 20
        ig = ign_of[idx]
        perm = torch.argsort(torch.rand(B, K - 1, device=dev, generator=g), 1)[:, :20]
        perm = perm.clone()
        perm[torch.arange(B, device=dev), ig] = K - 1             # ignored class -> last colour
        m = torch.gather(perm, 1, m.view(B, -1)).view(B, 128, 128)
        k = torch.randint(0, 8, (B,), device=dev, generator=g)
        out = torch.empty_like(m)
        for t in range(8):
            sel = k == t
            if sel.any():
                x = torch.rot90(m[sel], t % 4, (1, 2))
                out[sel] = torch.flip(x, (2,)) if t >= 4 else x
        return out

    t0 = time.time()
    model.train()
    for step in range(a.steps):
        lab = draw_batch()                                   # (B,128,128) universal indices
        x = render(lab, colours)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(x)
        logits = to_native(logits.float(), 128)
        loss = F.cross_entropy(logits, lab, ignore_index=K - 1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)

    @torch.no_grad()
    def evaluate(maps: torch.Tensor) -> tuple[dict, np.ndarray]:
        model.eval()
        lab_u = maps.clone()
        lab_u[maps == 19] = K - 1                            # PASTIS class c -> colour c
        preds, idxs = [], []
        for s in range(0, len(maps), 32):
            x = render(lab_u[s:s + 32].to(dev), colours)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, idx = model(x)
            logits = to_native(logits.float(), 128)
            logits[:, 19:] = -1e9                            # only PASTIS classes 0..18
            preds.append(logits.argmax(1).cpu().numpy())
            idxs.append(idx.cpu().numpy())
        pred = np.concatenate(preds)
        r = rmc.scores(maps.numpy(), pred, 20, 19)
        return r, np.concatenate(idxs)

    rv, _ = evaluate(va)
    rt, idx_te = evaluate(te)
    _, idx_tr = evaluate(tr)
    stats = {
        "test_distinct_codes": int(len(np.unique(idx_te))),
        "train_distinct_codes": int(len(np.unique(idx_tr))),
        "unseen_token_share": float(1 - np.isin(idx_te.ravel(), np.unique(idx_tr)).mean()),
    }
    _, ct = np.unique(idx_te, return_counts=True)
    p = ct / ct.sum()
    stats["code_entropy_bits"] = float(-(p * np.log2(p)).sum())
    out = {"args": vars(a), "params": n_par, "codes": model.fsq.n_codes,
           "grid": list(idx_te.shape[1:]), "val": {k: rv[k] for k in ("mIoU", "pixel_acc", "boundary_acc", "interior_acc")},
           "test": {k: rt[k] for k in ("mIoU", "pixel_acc", "boundary_acc", "interior_acc", "iou_per_class")},
           "code_stats": stats, "seconds": time.time() - t0}
    print(json.dumps({k: out[k] for k in ("grid", "codes", "val", "code_stats")}), flush=True)
    print(f"TEST mIoU {rt['mIoU']:.4f} acc {rt['pixel_acc']:.4f} boundary {rt['boundary_acc']:.4f} "
          f"interior {rt['interior_acc']:.4f}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    torch.save(model.state_dict(), Path(a.out).with_suffix(".pt"))


if __name__ == "__main__":
    main()
