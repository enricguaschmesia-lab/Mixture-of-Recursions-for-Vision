#!/usr/bin/env python
"""Phase 2 Step 1.4 -- bring-up before scaling the 224 re-tokenization.

CLAUDE.md rule 2: any transformation applied to a whole dataset is validated on
a handful of samples first, AND against a deliberately-broken control that
proves the check can fail. A check that cannot fail is not evidence.

What this validates, per modality, on a small sample through the PRODUCTION
path (batch 32, tail-padded, exactly as tokenize_terramesh.py will run it):

  A. shape is (N, TOKENS_PER_SAMPLE) = (N, 196) at the 224 contract
  B. dtype is uint16 and every token is inside its own codebook
  C. no constant token sequence -- the NaN-poisoning signature
  D. prepare() actually applied the contract crop (crop=224, crop_off=20)

THE CONTROL (the point of this script). The same samples are re-encoded with
crop_off=0 -- the correct 224x224 window size, shifted 20 px off centre. The
control must produce DIFFERENT tokens, or checks A-D are blind to a wrong crop
offset and the full run would be unverified.

The criterion is conditioned on the input actually changing, which is not
pedantry -- it is load-bearing for LULC. A scene that is a single land-cover
class throughout (measured: 7 of the first 40 majortom scenes are, e.g. all
class 9) has a bitwise IDENTICAL one-hot input at offset 0 and offset 20, so
identical tokens are the only correct output and demanding a difference would
be demanding the tokenizer be non-deterministic. So: every row whose prepared
input changed MUST produce different tokens, every row whose input did not
change MUST produce identical tokens (a free determinism check), and the
control is vacuous unless some row's input changed at all.

Note what the control does NOT do: it does not produce out-of-range tokens, a
crash, or a constant sequence. It produces 196 perfectly plausible in-range
tokens of the wrong part of the scene. That is precisely the failure mode rule 2
exists for -- a wrong preprocessing contract yields plausible-looking,
meaningless output that surfaces weeks later as an unexplained training failure.
Only a differential check catches it.

Runs in the `mor` env.

    python eo/scripts/bringup_224.py
    python eo/scripts/bringup_224.py --n 16 --modality DEM
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import warnings

os.environ.setdefault("HF_HOME", "/data/enric/hf")
warnings.filterwarnings("ignore")

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terramesh_tok import contract as C, io as tio, preprocess as P, tokenizers as T

# First shard of the corpus each modality actually covers (S1RTC = majortom,
# S1GRD = ssl4eos12; see tokenize_terramesh.MOD_SHARDS).
FIRST_SHARD = {m: "majortom_shard_000001.tar" for m in C.MODALITIES}
FIRST_SHARD["S1GRD"] = "ssl4eos12_shard_000001.tar"

BATCH = 32
fail = 0


def bad(msg: str) -> None:
    global fail
    fail += 1
    print(f"    FAIL: {msg}")


@torch.no_grad()
def encode_batched(mod, tok, arrs, *, crop_off=None):
    """The production path: fixed batch shape, tail padded, padding sliced off.

    crop_off is threaded through to prepare() so the control runs through
    exactly this code path and differs in the crop window alone.
    """
    out, buf, infos = [], [], []

    prepared = []

    def flush():
        if not buf:
            return
        n = len(buf)
        b = torch.cat(buf)
        if n < BATCH:
            b = torch.cat([b, b[-1:].repeat(BATCH - n, *([1] * (b.ndim - 1)))])
        g = T.encode(tok, b.to(T.DEVICE))
        out.append(P.flatten_tokens(g).cpu().numpy()[:n].astype(np.uint16))
        buf.clear()

    for a in arrs:
        x, info = P.prepare(a, mod, device="cpu", crop_off=crop_off)
        buf.append(x)
        prepared.append(x.numpy().copy())
        infos.append(info)
        if len(buf) >= BATCH:
            flush()
    flush()
    return np.concatenate(out), infos, np.concatenate(prepared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="samples per modality")
    ap.add_argument("--modality", nargs="*", default=C.MODALITIES)
    args = ap.parse_args()

    print(f"=== Step 1.4 bring-up: crop {C.CROP}, grid {C.GRID}x{C.GRID}, "
          f"{C.TOKENS_PER_SAMPLE} tokens/sample, batch {BATCH} ===")
    print(f"    contract offset {C.CROP_OFF} px/side; "
          f"control shifts the window to offset 0\n")

    for mod in args.modality:
        shard = FIRST_SHARD[mod]
        arrs = [a for _, a in tio.iter_shard(mod, shard, limit=args.n)]
        tok = T.build(mod)

        toks, infos, x_prod = encode_batched(mod, tok, arrs)
        ctrl, _, x_ctrl = encode_batched(mod, tok, arrs, crop_off=0)

        n = len(arrs)
        # A. shape
        if toks.shape != (n, C.TOKENS_PER_SAMPLE):
            bad(f"{mod}: shape {toks.shape}, expected {(n, C.TOKENS_PER_SAMPLE)}")
        # B. dtype + range
        if toks.dtype != np.uint16:
            bad(f"{mod}: dtype {toks.dtype}")
        lo, hi = int(toks.min()), int(toks.max())
        if lo < 0 or hi >= C.CODEBOOK[mod]:
            bad(f"{mod}: tokens outside [0,{C.CODEBOOK[mod]}) -- got [{lo},{hi}]")
        # C. NaN-poisoning signature
        const = (toks == toks[:, :1]).all(axis=1)
        if const.any():
            bad(f"{mod}: {int(const.sum())} constant token sequence(s)")
        # D. the contract crop was actually applied
        crops = {i["crop"] for i in infos}
        offs = {i["crop_off"] for i in infos}
        if crops != {C.CROP} or offs != {C.CROP_OFF}:
            bad(f"{mod}: prepare() used crop={crops}, off={offs}, "
                f"expected {{{C.CROP}}}/{{{C.CROP_OFF}}}")

        # THE CONTROL -- conditioned on the input actually changing.
        axes = tuple(range(1, x_prod.ndim))
        input_changed = ~np.all(x_prod == x_ctrl, axis=axes)   # per row
        tokens_changed = ~np.all(toks == ctrl, axis=1)
        n_changed = int(input_changed.sum())

        missed = int((input_changed & ~tokens_changed).sum())   # blind to the crop
        spurious = int((~input_changed & tokens_changed).sum())  # non-determinism
        rate = float((toks != ctrl).mean())
        ctrl_in_range = 0 <= int(ctrl.min()) and int(ctrl.max()) < C.CODEBOOK[mod]
        control_fired = n_changed > 0 and missed == 0 and spurious == 0

        print(f"  {mod:6s} {n:3d} samples  shape {str(toks.shape):12s} "
              f"tok[{lo:5d},{hi:6d}]  {len(np.unique(toks)):4d} distinct")
        print(f"         control (crop_off=0): input changed on {n_changed}/{n} rows, "
              f"{100*rate:5.1f}% of tokens differ, in-range={ctrl_in_range}")
        print(f"         -> {'CONTROL FIRED' if control_fired else 'CONTROL DID NOT FIRE'}"
              f"  (missed {missed}, spurious {spurious})")
        if n_changed == 0:
            bad(f"{mod}: control is vacuous -- the off-centre window produced an "
                f"identical input on every row, so nothing was tested")
        if missed:
            bad(f"{mod}: {missed} row(s) changed input but produced identical "
                f"tokens -- these checks cannot detect a wrong crop offset")
        if spurious:
            bad(f"{mod}: {spurious} row(s) had identical input but different "
                f"tokens -- the encode path is non-deterministic at fixed batch shape")
        if n_changed < n:
            print(f"         note: {n - n_changed} row(s) are spatially uniform over "
                  f"both windows, so identical tokens there are correct, not a miss")
        if not ctrl_in_range:
            print(f"         note: control tokens left the codebook, so this "
                  f"modality would also have been caught by the range check")

    print()
    print("BRING-UP PASSED -- safe to scale to the full run"
          if not fail else f"BRING-UP FAILED ({fail} check(s)) -- do not scale")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
