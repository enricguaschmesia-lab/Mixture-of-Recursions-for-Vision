#!/usr/bin/env python
"""Tokenize A2's label maps with a frozen map tokenizer (A2 Step 2.5, `a2seg` env, GPU).

    python eo/scripts/tokenize_maps.py --task pastis --arm aligned --splits train,val,test --expect-gpu "GTX TITAN X"

One row per view of <task root>/views.parquet (eo/geobench/views.py): the view's native label crop,
its dihedral variant, rendered with the universal palette and the task's fixed assignment
(eo.maps.palette.render, nearest, as the harness renders), encoded at the tokenizer's fixed batch shape
with cuDNN deterministic and TF32 off (eo.maps.tokenizers.run_fixed). Output, in the TerraMesh format:

    <task root>/MAP_<arm>/tokens.npy    (n_views, tokens_per_map) uint16, row-major (row = k // grid)
                         /present.npy   (n_views,) bool
                         /metadata.json the arm's frozen record (hashes), palette sha256, gpu, counts

Resumable and atomic per chunk, as tokenize_ben.py. Test views may be tokenized; nothing here scores.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eo.geobench import s2 as G, sources as S, views as V  # noqa: E402
from eo.maps import frozen, palette as P  # noqa: E402
from eo.maps.tokenizers import run_fixed  # noqa: E402

CHUNK = 2048


def view_labels(task: str, vs) -> np.ndarray:
    """(n, size, size) uint8 label crops of views `vs` (a DataFrame slice), dihedral applied."""
    out, cache = [], {}
    for v in vs.itertuples():
        if v.sample_idx not in cache:
            cache.clear()
            cache[v.sample_idx] = (S.pastis_sample(v.sample_idx, ("semantic",))["semantic"] if task == "pastis"
                                   else S.cloudsen12_sample(v.sample_idx)[1])
        lab = cache[v.sample_idx][v.y0:v.y0 + v.size, v.x0:v.x0 + v.size]
        out.append(G.dihedral(lab, v.k))
    return np.stack(out)


def encode_maps(tok, task: str, labels: np.ndarray, device: str = "cuda") -> np.ndarray:
    t = P.TASKS[task]
    P.validate(P.UNIVERSAL, t.assign, t.ignore, t.n_classes)
    x = P.render(torch.as_tensor(labels, device=device), t.assign, tok.input_size, P.UNIVERSAL)
    codes = run_fixed(tok.encode, x, tok.batch)
    return codes.reshape(len(codes), -1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pastis", "cloudsen12"], required=True)
    ap.add_argument("--arm", choices=list(frozen.ALL), required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    V.write(a.task)
    views = V.load(a.task)["views"]
    tok = frozen.load(a.arm)
    T = tok.frozen["tokens"]
    out = V.TOK_ROOT[a.task] / f"MAP_{a.arm}"
    out.mkdir(parents=True, exist_ok=True)
    tpath, ppath = out / "tokens.npy", out / "present.npy"
    if (out / "metadata.json").exists():                 # one card per artifact: codes can differ across cards
        prev_gpu = json.loads((out / "metadata.json").read_text()).get("gpu")
        if prev_gpu and prev_gpu != gpu:
            raise SystemExit(f"{out} was written on {prev_gpu}; refusing to add rows from {gpu}")
    n = len(views)
    if tpath.exists():
        tokens = np.load(tpath, mmap_mode="r+")
        assert tokens.shape == (n, T) and tokens.dtype == np.uint16, tokens.shape
    else:
        tokens = np.lib.format.open_memmap(tpath, mode="w+", dtype=np.uint16, shape=(n, T))
    present = np.load(ppath) if ppath.exists() else np.zeros(n, bool)
    sel = views[views.split.isin(a.splits.split(","))]
    todo = sel[~present[sel.view.values]]
    if a.limit:
        todo = todo.iloc[: a.limit]
    print(f"{a.task} MAP_{a.arm}: {len(sel)} views selected, {len(todo)} to go, {T} tokens/map, on {gpu}", flush=True)
    t0 = time.time()
    for c0 in range(0, len(todo), CHUNK):
        ch = todo.iloc[c0:c0 + CHUNK]
        codes = encode_maps(tok, a.task, view_labels(a.task, ch))
        assert codes.shape == (len(ch), T) and codes.min() >= 0 and codes.max() < tok.n_codes
        tokens[ch.view.values] = codes.astype(np.uint16)
        tokens.flush()
        present[ch.view.values] = True
        with open(str(ppath) + ".tmp", "wb") as fh:
            np.save(fh, present)
        os.replace(str(ppath) + ".tmp", ppath)
        done = c0 + len(ch)
        print(f"  {done}/{len(todo)}  {done / (time.time() - t0):.1f} maps/s", flush=True)

    tk = np.asarray(tokens[present])
    meta = {"task": a.task, "arm": a.arm, "views": str(V.TOK_ROOT[a.task] / "views.parquet"),
            "n_views": n, "n_present": int(present.sum()),
            "present_by_split": {s: int(present[views.view.values[views.split == s]].sum()) for s in ("train", "val", "test")},
            "frozen": tok.frozen, "tokenizer_record": tok.record(),
            "palette": "eo.maps.palette.UNIVERSAL", "palette_sha256": P.palette_record()["sha256"],
            "assignment": list(P.TASKS[a.task].assign), "render": "nearest to the tokenizer's input size",
            "batch": tok.batch, "run_fixed": "cuDNN deterministic, TF32 off, tail padded with zeros",
            "flatten": "row-major", "codes_used": int(len(np.unique(tk))) if len(tk) else 0,
            "gpu": gpu, "fork_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                                       capture_output=True, text=True).stdout.strip(),
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    assert meta["palette_sha256"] == tok.frozen["palette_sha256"], "palette differs from the frozen record"
    with open(str(out / "metadata.json") + ".tmp", "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    os.replace(str(out / "metadata.json") + ".tmp", out / "metadata.json")
    print(json.dumps({k: meta[k] for k in ("n_present", "present_by_split", "codes_used")}), flush=True)


if __name__ == "__main__":
    main()
