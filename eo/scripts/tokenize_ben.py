#!/usr/bin/env python
"""Tokenize BigEarthNet S1/S2 with the TerraMind tokenizers (PHASE6_PLAN Step 1.4, `mor` env, GPU).

    python eo/scripts/tokenize_ben.py --modality S2L2A --roles ft_train,ft_valsel,test_a,test_c
    python eo/scripts/tokenize_ben.py --modality S1GRD --roles ft_train,ft_valsel,test_a,test_c

Output, in the TerraMesh format, rows = index_v1.parquet's rows (ALL of them; only the
requested roles are filled, `present.npy` says which):

    /data/enric/data/BigEarthNet/<MOD>_tok224/tokens.npy    (N, 196) uint16
                                             /present.npy   (N,) bool
                                             /metadata.json provenance

`--modality` names the TOKENIZER (S2L2A for S2; S1GRD, or S1RTC for the 1.3 comparison, for
S1), so the directory says which slot the ids belong to.

Contract: eo/ben/preprocess.py (bilinear, one interpolation to 224, v1 stats), batch 32 with
tail padding (tokens are bit-reproducible only at a fixed batch shape; the pad rows are
sliced off). Resumable and atomic per chunk: `present.npy` is rewritten (tmp + replace) only
after the chunk's tokens are flushed, so a crash leaves rows either done or not done.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/enric/hf")

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eo.ben import layout, preprocess as BP  # noqa: E402
from eo.terramesh_tok import contract as C, preprocess as P, tokenizers as T  # noqa: E402

BATCH = 32
CHUNK = 4096
MANIFEST = Path("/data/enric/weights/tokenizer_manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["S2L2A", "S1GRD", "S1RTC"], required=True)
    ap.add_argument("--roles", required=True, help="comma-separated index columns, e.g. ft_train,test_a")
    ap.add_argument("--limit", type=int, default=None, help="dry run: only the first N rows")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--packed", type=Path, default=None, help="read this pack (default layout.PACKED; L8: layout.PACKED_L8)")
    a = ap.parse_args()

    gpu = torch.cuda.get_device_name(0)
    idx = pd.read_parquet(layout.INDEX)
    n = len(idx)
    roles = [r.strip() for r in a.roles.split(",") if r.strip()]
    sel = np.zeros(n, bool)
    for r in roles:
        sel |= idx[r].to_numpy(bool)
    rows = np.flatnonzero(sel)
    if a.limit:
        rows = rows[: a.limit]
    src = "S2" if a.modality == "S2L2A" else "S1"
    packed, r2p = layout.load_packed(src, n, a.packed or layout.PACKED)
    if a.packed is not None:                 # an extra pack: the role's rows that it holds
        rows = rows[r2p[rows] >= 0]
    if (r2p[rows] < 0).any():
        sys.exit(f"{int((r2p[rows] < 0).sum())} selected rows are not packed ({src})")

    out = layout.ROOT / C.tok_dir_name(a.modality)
    out.mkdir(parents=True, exist_ok=True)
    tpath, ppath = out / "tokens.npy", out / "present.npy"
    if tpath.exists():
        tokens = np.load(tpath, mmap_mode="r+")
        assert tokens.shape == (n, C.TOKENS_PER_SAMPLE) and tokens.dtype == np.uint16
    else:
        tokens = np.lib.format.open_memmap(tpath, mode="w+", dtype=np.uint16,
                                           shape=(n, C.TOKENS_PER_SAMPLE))
    present = np.load(ppath) if ppath.exists() else np.zeros(n, bool)
    todo = rows[~present[rows]]
    todo = todo[np.argsort(r2p[todo], kind="stable")]     # read the packed arrays in order
    print(f"{a.modality}: {len(rows)} selected, {len(rows) - len(todo)} already done, "
          f"{len(todo)} to go, on {gpu}", flush=True)

    tok = T.build(a.modality, a.device)
    n_nonfinite, t0 = 0, time.time()
    for c0 in range(0, len(todo), CHUNK):
        chunk = todo[c0:c0 + CHUNK]
        for b0 in range(0, len(chunk), BATCH):
            br = chunk[b0:b0 + BATCH]
            pad = BATCH - len(br)
            brp = np.concatenate([br, np.repeat(br[:1], pad)]) if pad else br   # tail padding
            pp = r2p[brp]
            if src == "S2":
                x = BP.s2_batch(packed["S2_10m"][pp], packed["S2_20m"][pp], packed["S2_60m"][pp], a.device)
            else:
                x, nb = BP.s1_batch(packed["S1"][pp], a.device, tokenizer=a.modality)
                n_nonfinite += nb
            t = P.flatten_tokens(T.encode(tok, x)).cpu().numpy()
            assert t.shape == (BATCH, C.TOKENS_PER_SAMPLE)
            assert t.min() >= 0 and t.max() < C.CODEBOOK[a.modality], (t.min(), t.max())
            tokens[br] = t[: len(br)].astype(np.uint16)
        tokens.flush()
        present[chunk] = True
        with open(str(ppath) + ".tmp", "wb") as fh:
            np.save(fh, present)
        os.replace(str(ppath) + ".tmp", ppath)
        done = c0 + len(chunk)
        dt = time.time() - t0
        print(f"  {done}/{len(todo)}  {done / dt:.0f} rows/s  {dt / 60:.1f} min", flush=True)

    manifest = json.loads(MANIFEST.read_text()).get(C.TOKENIZER[a.modality], {})
    tk = np.asarray(tokens[present])
    meta_path = out / "metadata.json"
    prev = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = {
        "dataset": "BigEarthNet v1 (Zenodo 12687186)", "index": str(layout.INDEX),
        "n_rows": n, "n_present": int(present.sum()),
        "roles_requested": sorted(set(prev.get("roles_requested", [])) | set(roles)),
        "tokenizer_modality": a.modality, "tokenizer": C.TOKENIZER[a.modality],
        "tokenizer_revision": manifest.get("revision"),
        "source": src, "crop": C.CROP, "tokens_per_sample": C.TOKENS_PER_SAMPLE,
        "resample": "bilinear, one interpolation from native size to 224, align_corners=False",
        "stats": "v1_pretraining (contract.V1_TOK_MEAN/STD)",
        "nan_policy": C.NAN_POLICY, "nonfinite_filled_this_run": n_nonfinite,
        "batch": BATCH, "tail_padding": True, "flatten_order": C.FLATTEN_ORDER,
        "token_min": int(tk.min()), "token_max": int(tk.max()),
        "codes_used": int(len(np.unique(tk))), "codebook": C.CODEBOOK[a.modality],
        "gpu": gpu, "fork_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip(),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(str(meta_path) + ".tmp", "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(str(meta_path) + ".tmp", meta_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
