#!/usr/bin/env python
"""Which SmolLM-135M layer can go? Next-token loss on English text with one layer removed (`.venv`, CPU).

B10c has 29 layers and SmolLM-135M 30, so the LLM init (eo/train/smollm_init.py) keeps 29 of them.
This measures, for each layer, SmolLM's text loss without it (the ShortGPT / Gromov et al. 2024
criterion); smollm_init.DROPPED_LAYER is the cheapest. 2026-10-28: layer 5 (+0.060 nats on 16,384
tokens of prose), the last layer +1.45.

    CUDA_VISIBLE_DEVICES= .venv/bin/python -I eo/experiments/smollm_layer_drop.py \\
        --source /data/enric/weights/smollm_135m/1d461723eec654e65efdc40cf49301c89c0c92f4 \\
        --n-seq 16 <text files...>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from eo.train import smollm_init as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("text", nargs="+", type=Path)
    a = ap.parse_args()
    S.load_smollm(a.source)                          # the pinned weights, or refuse
    ids = S.text_ids(a.source, a.text, seq_len=1024, n_seq=a.n_seq)
    print(f"tokens {ids.numel()} from {[str(p) for p in a.text]}")
    m = S.reference(a.source, None)
    layers = m.model.layers

    def loss(keep):
        m.model.layers = torch.nn.ModuleList([layers[i] for i in keep])
        m.config.num_hidden_layers = len(keep)
        return S.text_loss(m, ids)

    full = loss(range(S.SOURCE_LAYERS))
    print(f"full {S.SOURCE_LAYERS} layers: {full:.3f}")
    res = []
    for d in range(S.SOURCE_LAYERS):
        v = loss(S.layer_map(d))
        res.append((d, v))
        print(f"drop {d:2d}: {v:.3f} (+{v - full:.3f})", flush=True)
    print("cheapest:", [(d, round(v - full, 3)) for d, v in sorted(res, key=lambda r: r[1])[:5]])


if __name__ == "__main__":
    main()
