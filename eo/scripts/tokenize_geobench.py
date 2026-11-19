#!/usr/bin/env python
"""Tokenize GEO-Bench-2's S2 inputs with TerraMind's S2L2A tokenizer (A2 Step 2.3-2.4, `a2seg` env, GPU).

    python eo/scripts/tokenize_geobench.py --task pastis --frames p1 --splits train,val --expect-gpu "GTX TITAN X"
    python eo/scripts/tokenize_geobench.py --task cloudsen12 --splits train,val,test --expect-gpu "GTX TITAN X"

One row per row of <task root>/inputs.parquet (eo/geobench/views.py): PASTIS one row per (view, frame),
CloudSen12 one per view. The contract is eo/geobench/s2.py (B01/B09 imputed by s2.CHOSEN_IMPUTATION for
PASTIS, one bilinear resample to 224, v1 stats; CloudSen12 cut at 224, border filled), the dihedral
variant applied to the native raster first. Batch 32, the tail padded by repeating the first row, as
tokenize_ben.py does (tokens are bit-reproducible only at a fixed batch shape). Output:

    <task root>/S2L2A_tok224/tokens.npy    (n_inputs, 196) uint16
                            /present.npy   (n_inputs,) bool
                            /metadata.json

Resumable and atomic per chunk. TF32 is turned off after terratorch's import (CLAUDE.md).
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
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eo.geobench import s2 as G, sources as S, views as V  # noqa: E402
from eo.terramesh_tok import contract as C, preprocess as P, tokenizers as T  # noqa: E402

BATCH, CHUNK = 32, 512


def build_inputs(task: str, rows, views, method: str | None, device: str = "cuda") -> tuple[torch.Tensor, int]:
    """Encoder inputs (n, 12, 224, 224) for input rows `rows` (DataFrame slice, sample-ordered)."""
    vv = views.set_index("view")
    xs, n_nodata, cache = [], 0, {}
    for r in rows.itertuples():
        v = vv.loc[r.view]
        if v.sample_idx not in cache:
            cache.clear()
            cache[v.sample_idx] = (S.pastis_sample(v.sample_idx, ("s2",))["s2"] if task == "pastis"
                                   else S.cloudsen12_sample(v.sample_idx)[0])
        a = cache[v.sample_idx]
        if task == "pastis":
            xs.append(G.dihedral(a[r.t], v.k))                               # (10, 128, 128)
        else:
            c = int(v["size"])
            xs.append(G.dihedral(a[:, v.y0:v.y0 + c, v.x0:v.x0 + c], v.k))   # (14, 224, 224)
    x = np.stack(xs)
    if task == "pastis":
        return G.pastis_input(x, method, device), 0
    z, n_nodata = G.cloudsen12_input(x, device)
    return z, n_nodata


def encode_fixed(tok, x: torch.Tensor) -> np.ndarray:
    out = []
    for s in range(0, len(x), BATCH):
        xb = x[s:s + BATCH]
        n = len(xb)
        if n < BATCH:
            xb = torch.cat([xb, xb[:1].expand(BATCH - n, *xb.shape[1:])])
        out.append(P.flatten_tokens(T.encode(tok, xb)).cpu().numpy()[:n])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pastis", "cloudsen12"], required=True)
    ap.add_argument("--frames", default="p1", help="PASTIS: p1, p7 or p1,p7")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--method", default=None, help="B01/B09 imputation (default s2.CHOSEN_IMPUTATION)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    method = a.method or G.CHOSEN_IMPUTATION
    if a.task == "pastis" and method is None:
        raise SystemExit("no imputation chosen yet (eo/geobench/s2.py CHOSEN_IMPUTATION)")
    V.write(a.task)
    idx = V.load(a.task)
    views, inputs = idx["views"], idx["inputs"]
    tok = T.build("S2L2A", "cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    out = V.TOK_ROOT[a.task] / C.tok_dir_name("S2L2A")
    out.mkdir(parents=True, exist_ok=True)
    tpath, ppath = out / "tokens.npy", out / "present.npy"
    if (out / "metadata.json").exists():                 # one card per artifact: codes can differ across cards
        prev_gpu = json.loads((out / "metadata.json").read_text()).get("gpu")
        if prev_gpu and prev_gpu != gpu:
            raise SystemExit(f"{out} was written on {prev_gpu}; refusing to add rows from {gpu}")
    n = len(inputs)
    if tpath.exists():
        tokens = np.load(tpath, mmap_mode="r+")
        assert tokens.shape == (n, C.TOKENS_PER_SAMPLE) and tokens.dtype == np.uint16
    else:
        tokens = np.lib.format.open_memmap(tpath, mode="w+", dtype=np.uint16, shape=(n, C.TOKENS_PER_SAMPLE))
    present = np.load(ppath) if ppath.exists() else np.zeros(n, bool)
    split_of = views.set_index("view").split
    sel = inputs[inputs.view.map(split_of).isin(a.splits.split(","))]
    if a.task == "pastis":
        want = set()
        for f in a.frames.split(","):
            want |= {"p1"} if f == "p1" else set(V.FRAMES_P7) if f == "p7" else {f}
        sel = sel[sel.frame.isin(want)]
    todo = sel[~present[sel.row.values]]
    todo = todo.assign(_s=todo.view.map(views.set_index("view").sample_idx)).sort_values(["_s", "row"])
    if a.limit:
        todo = todo.iloc[: a.limit]
    print(f"{a.task}: {len(sel)} inputs selected, {len(todo)} to go, imputation {method}, on {gpu}", flush=True)
    t0, nodata = time.time(), 0
    for c0 in range(0, len(todo), CHUNK):
        ch = todo.iloc[c0:c0 + CHUNK]
        x, nd = build_inputs(a.task, ch, views, method)
        nodata += nd
        t = encode_fixed(tok, x)
        assert t.shape == (len(ch), C.TOKENS_PER_SAMPLE) and t.min() >= 0 and t.max() < C.CODEBOOK["S2L2A"]
        tokens[ch.row.values] = t.astype(np.uint16)
        tokens.flush()
        present[ch.row.values] = True
        with open(str(ppath) + ".tmp", "wb") as fh:
            np.save(fh, present)
        os.replace(str(ppath) + ".tmp", ppath)
        done = c0 + len(ch)
        print(f"  {done}/{len(todo)}  {done / (time.time() - t0):.1f} rows/s", flush=True)

    meta_path = out / "metadata.json"
    prev = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tk = np.asarray(tokens[present])
    meta = {"task": a.task, "inputs": str(V.TOK_ROOT[a.task] / "inputs.parquet"), "n_rows": n,
            "n_present": int(present.sum()),
            "requested": sorted(set(prev.get("requested", [])) | {f"{a.splits}|{a.frames if a.task == 'pastis' else '-'}"}),
            "tokenizer": C.TOKENIZER["S2L2A"], "crop": C.CROP, "tokens_per_sample": C.TOKENS_PER_SAMPLE,
            "imputation": method if a.task == "pastis" else None,
            "linear_coef": G.LINEAR if (a.task == "pastis" and method == "linear") else None,
            "linear_source": G.LINEAR_SOURCE,
            "resample": "PASTIS: bilinear 128 -> 224 in one interpolation, align_corners False; CloudSen12: none (224 crop)",
            "stats": "v1_pretraining (contract.V1_TOK_MEAN/STD)",
            "nodata_pixels_filled_this_run": nodata, "batch": BATCH, "tail_padding": "first row repeated",
            "tf32": False, "flatten_order": C.FLATTEN_ORDER,
            "token_min": int(tk.min()) if len(tk) else None, "token_max": int(tk.max()) if len(tk) else None,
            "codes_used": int(len(np.unique(tk))) if len(tk) else 0,
            "gpu": gpu, "fork_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                                       capture_output=True, text=True).stdout.strip(),
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(str(meta_path) + ".tmp", "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(str(meta_path) + ".tmp", meta_path)
    print(json.dumps({k: meta[k] for k in ("n_present", "codes_used", "imputation")}), flush=True)


if __name__ == "__main__":
    main()
