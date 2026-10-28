#!/usr/bin/env python
"""Continuous-I/O pilot: FSQ token grids -> pixels with TerraMind's own decoders. (`mor` env)

Two jobs, both on the held-out rows cio_data.py wrote:
  roundtrip   the GROUND-TRUTH tokens of the first --n rows, decoded: the tokenizer's ceiling on
              exactly the rows the pilot's generations are scored on;
  rollouts    the token grids cio_train.py rollout generated (token-output variants), decoded,
              so they can be scored in pixels beside the continuous-output variants.

    python eo/experiments/cio_decode.py roundtrip --n 256
    python eo/experiments/cio_decode.py rollouts --runs tok_in_s0 lin_in_s0

Decoder settings are the project's (12.6b, D3.11): DiVAE with the clamp off
(thresholding=False, clip_sample=False), 50 steps, seed 0 re-applied per batch; LULC's ViT decoder
is deterministic, its 10 output channels are argmax-ed to classes. Output: z-score float16
(N, C, 224, 224) for S2L2A/S1RTC/DEM, uint8 classes (N, 224, 224) for LULC, next to the input
with a `_decoded` suffix. Run on the GTX Titan X (CLAUDE.md: long GPU jobs off the TITAN V).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eo.terramesh_tok import tokenizers as TK                  # noqa: E402
from eo.experiments import clamp_test as CT                     # noqa: E402

ROOT = Path("/data/enric/experiments/continuous_io")
MODS = ("S2L2A", "S1RTC", "DEM", "LULC")
STEPS, SEED, BATCH = 50, 0, 8


@torch.no_grad()
def decode(mod: str, ids: np.ndarray) -> np.ndarray:
    tok = TK.build(mod)
    sched = None if mod == "LULC" else CT.scheduler_like(tok, **CT.VARIANTS["noclip"])
    out = []
    for i in range(0, len(ids), BATCH):
        g = torch.from_numpy(ids[i:i + BATCH].reshape(-1, 14, 14).astype(np.int64)).to(TK.DEVICE)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        if mod == "LULC":
            x = tok.decode_tokens(g).float()
            out.append(x.argmax(1).cpu().numpy().astype(np.uint8))
        else:
            x = tok.decode_tokens(g, timesteps=STEPS, scheduler=sched).float()
            out.append(x.cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def check_gpu():
    name = torch.cuda.get_device_name(0)
    if "TITAN X" not in name:
        raise SystemExit(f"run this on the GTX Titan X, CUDA gave {name!r}")


def cmd_roundtrip(a) -> int:
    check_gpu()
    tok = np.load(ROOT / "data/test_tok.npy")[: a.n]
    out = ROOT / "roundtrip"
    out.mkdir(exist_ok=True)
    for k, m in enumerate(MODS):
        f = out / f"{m}_gt_n{a.n}_decoded.npy"
        if f.exists():
            print("[skip]", f.name); continue
        t0 = time.time()
        np.save(f, decode(m, tok[:, k]))
        print(f"{m}: {a.n} rows in {time.time() - t0:.0f} s", flush=True)
    return 0


def cmd_rollouts(a) -> int:
    check_gpu()
    for run in a.runs:
        cfg = json.loads((ROOT / "runs" / run / "config.json").read_text())
        assert cfg["variant"] in ("tok_in", "lin_in", "linraw_in", "linrel_in", "band_in", "bandrop_in"), f"{run} does not emit tokens"
        for f in sorted((ROOT / "runs" / run / "rollout").glob("*_k*_*.npy")):
            if f.stem.endswith("_decoded"):
                continue
            dst = f.with_name(f.stem + "_decoded.npy")
            if dst.exists():
                print("[skip]", run, dst.name); continue
            tgt = f.stem.split("_")[1]
            t0 = time.time()
            np.save(dst, decode(tgt, np.load(f)))
            print(f"{run} {f.stem}: decoded in {time.time() - t0:.0f} s", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("roundtrip"); r.add_argument("--n", type=int, default=256)
    o = sp.add_parser("rollouts"); o.add_argument("--runs", nargs="+", required=True)
    a = p.parse_args()
    return {"roundtrip": cmd_roundtrip, "rollouts": cmd_rollouts}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
