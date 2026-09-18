#!/usr/bin/env python
"""Regenerate gate_equivalence.py's single-sample reference tokens at the
current contract crop.

WHY THIS EXISTS. gate_equivalence.py's criterion 3 asserts that the production
batched path (batch 32, tail-padded) reproduces the SINGLE-SAMPLE path to within
the measured FSQ bin-flip rate. That needs a single-sample reference to compare
against. Step 3 produced one as a side effect of its round-trip experiment
(eo/experiments/s35_roundtrip.py -> /data/enric/figures/step3/tokens.json), at
the 256 crop that was in force then.

Those reference tokens are NOT reusable at 224. Changing the crop changes the
patch grid, so the ViT interpolates its position embeddings to a different size
and every token changes -- see contract.py's CROP comment. Run criterion 3
against the 256 reference at a 224 contract and it fails 100%, which would look
like a contract divergence and is not one.

So the reference is regenerated per crop, into tokens<CROP>.json. This script is
the reproducible way to do that, rather than re-running the whole Step 3
round-trip experiment (which also decodes, computes SSIM and writes figures --
minutes of GPU for three numbers we already have).

The SAMPLES are selected by exactly the rule s35_roundtrip.py used, so the
reference covers the same three deliberately-diverse scenes per modality
(equatorial / high-latitude / heavily-clouded, and three ssl4eos12 scenes for
S1GRD). Verified 2026-09-18 to reproduce Step 3's stems exactly.

Runs in the `mor` env.

    python eo/scripts/make_gate_reference.py            # writes tokens224.json
    python eo/scripts/make_gate_reference.py --dry-run  # print, write nothing
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import warnings

os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T

FIG = pathlib.Path("/data/enric/figures/step3")
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
MODS = ["DEM", "NDVI", "LULC", "S1RTC", "S1GRD", "S2L2A"]


def samples():
    """The Step 3.5 selection rule, verbatim (s35_roundtrip.py:26-29).

    majortom: one equatorial, one high-latitude, one maximally clouded.
    ssl4eos12: fixed positional picks, used for S1GRD which covers that corpus.
    """
    mt = META[META.tar.str.startswith("majortom")]
    ss = META[META.tar.str.startswith("ssl4eos12")]
    pick = [mt[mt.center_lat.abs() < 8].iloc[0],
            mt[mt.center_lat.abs() > 55].iloc[0],
            mt.sort_values("cloud_cover", ascending=False).iloc[0]]
    pick_ss = [ss.iloc[0], ss.iloc[400], ss.iloc[3000]]
    to = lambda rows: [(r.tar, r.zarr[: -len(".zarr.zip")]) for r in rows]
    return to(pick), to(pick_ss)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mt, ss = samples()
    out = FIG / f"tokens{C.CROP}.json"
    print(f"=== single-sample reference at crop {C.CROP} "
          f"({C.TOKENS_PER_SAMPLE} tokens/sample) -> {out} ===", flush=True)

    tokens = {}
    for mod in MODS:
        tok = T.build(mod)
        for tar, stem in (ss if mod == "S1GRD" else mt):
            _, arr = tio.read_sample(mod, tar, stem)
            # Batch of 1: this IS the single-sample path criterion 3 compares
            # the batched path against. Do not pad it to 32 -- that would make
            # the reference and the thing under test the same computation, and
            # the check would pass by construction.
            x, info = P.prepare(arr, mod, device=T.DEVICE)
            assert info["crop"] == C.CROP and info["crop_off"] == C.CROP_OFF
            f = P.flatten_tokens(T.encode(tok, x))[0].cpu().numpy()
            assert f.shape == (C.TOKENS_PER_SAMPLE,), \
                f"{mod}/{stem}: {f.shape}, expected {(C.TOKENS_PER_SAMPLE,)}"
            assert 0 <= int(f.min()) and int(f.max()) < C.CODEBOOK[mod], \
                f"{mod}/{stem}: tokens outside [0,{C.CODEBOOK[mod]})"
            tokens.setdefault(mod, {})[stem] = f.astype(int).tolist()
            print(f"  {mod:6s} {stem:26s} {len(f):4d} tokens  "
                  f"range [{int(f.min())},{int(f.max())}]  "
                  f"{len(np.unique(f)):3d} distinct", flush=True)

    if args.dry_run:
        print("dry run: nothing written")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(tokens, fh)
    os.replace(tmp, out)
    print(f"\nwrote {out} "
          f"({sum(len(v) for v in tokens.values())} samples, "
          f"{C.TOKENS_PER_SAMPLE} tokens each)")


if __name__ == "__main__":
    main()
