#!/usr/bin/env python
"""Tokenize PASTIS-1's S2 at its NATIVE 128 px with TerraMind's S2L2A tokenizer (meeting 2026-11-25, option A; `a2seg`, GPU).

    python eo/scripts/tokenize_geobench_native.py --expect-gpu "GTX TITAN X"            # train + val p1 rows
    python eo/scripts/tokenize_geobench_native.py --verify 64 --expect-gpu "GTX TITAN X" # fresh-process integrity check

eo/geobench/s2.py's contract WITHOUT its step 3 (the resampling): bands to contract order, B01/B09 imputed by
s2.CHOSEN_IMPUTATION at native resolution, the dihedral variant applied to the native raster first, v1 standardization,
encoded at 128 x 128 -> 8 x 8 tokens of 16 px (160 m), row-major. Batch 32 with the tail padded by repeating the first row
(tokens are bit-reproducible only at a fixed batch shape), TF32 off after terratorch's import (CLAUDE.md). The contract at
128 px was checked before this ran (docs/notes/downstream/2026-11-25_meeting_2_patches_positions.md §8.3).

ONLY the p1 frame of the train and val views is tokenized: the test split is refused. Output, row-aligned with
<pastis root>/inputs.parquet (the rows of other frames and of test stay absent):

    <pastis root>/S2L2A_tok128/tokens.npy    (n_inputs, 64) uint16
                              /present.npy   (n_inputs,) bool
                              /metadata.json

Resumable and atomic per chunk; one card per artifact. eo/data/seg_dataset.py reads it with seg.s2_crop: 128.
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

NATIVE, PER = 128, 64
BATCH, CHUNK = 32, 512
SPLITS = ("train", "val")
OUT_NAME = f"S2L2A_tok{NATIVE}"


def native_inputs(rows, views, method: str, device: str = "cuda") -> torch.Tensor:
    """Encoder inputs (n, 12, 128, 128) for input rows `rows` (a sample-ordered DataFrame slice)."""
    vv = views.set_index("view")
    xs, cache = [], {}
    for r in rows.itertuples():
        v = vv.loc[r.view]
        if v.sample_idx not in cache:
            cache.clear()
            cache[v.sample_idx] = S.pastis_sample(v.sample_idx, ("s2",))["s2"]
        xs.append(G.dihedral(cache[v.sample_idx][r.t], v.k))                    # (10, 128, 128), native
    x = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=device)
    assert x.shape[1:] == (10, NATIVE, NATIVE), x.shape
    return G.standardize(G.assemble_pastis(x, method))                        # no resampling


def encode_fixed(tok, x: torch.Tensor) -> np.ndarray:
    out = []
    for s in range(0, len(x), BATCH):
        xb = x[s:s + BATCH]
        n = len(xb)
        if n < BATCH:
            xb = torch.cat([xb, xb[:1].expand(BATCH - n, *xb.shape[1:])])
        t = T.encode(tok, xb)
        assert t.shape[1:] == (NATIVE // 16, NATIVE // 16), t.shape
        out.append(P.flatten_tokens(t).cpu().numpy()[:n])
    return np.concatenate(out)


def selected_rows(views, inputs):
    split_of = views.set_index("view").split
    sel = inputs[(inputs.frame == "p1") & inputs.view.map(split_of).isin(SPLITS)]
    assert not sel.view.map(split_of).eq("test").any()
    return sel.assign(_s=sel.view.map(views.set_index("view").sample_idx)).sort_values(["_s", "row"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", type=int, default=None, help="re-encode N stored rows (half train, half val) and compare")
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    method = G.CHOSEN_IMPUTATION
    idx = V.load("pastis")
    views, inputs = idx["views"], idx["inputs"]
    tok = T.build("S2L2A", "cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    out = V.TOK_ROOT["pastis"] / OUT_NAME
    tpath, ppath, meta_path = out / "tokens.npy", out / "present.npy", out / "metadata.json"
    sel = selected_rows(views, inputs)

    if a.verify:                                       # a fresh process: N stored rows encoded again, bit for bit
        meta = json.loads(meta_path.read_text())
        if meta.get("gpu") != gpu:
            raise SystemExit(f"{out} was written on {meta.get('gpu')}; verify on that card, not {gpu}")
        tokens, present = np.load(tpath, mmap_mode="r"), np.load(ppath)
        split_of = views.set_index("view").split
        rng = np.random.default_rng(20261125)
        pick = []
        for sp in SPLITS:
            rows = sel[sel.view.map(split_of) == sp]
            pick.append(rows.iloc[np.sort(rng.choice(len(rows), a.verify // 2, replace=False))])
        import pandas as pd
        pick = pd.concat(pick).sort_values(["_s", "row"])
        assert present[pick.row.values].all()
        t = encode_fixed(tok, native_inputs(pick, views, method))
        stored = np.asarray(tokens[pick.row.values]).astype(np.int64)
        eq = (t == stored)
        res = {"n_rows": int(len(pick)), "rows_equal": int(eq.all(1).sum()), "tokens_equal": float(eq.mean()),
               "pass": bool(eq.all()), "gpu": gpu}
        # controls that must fail the check: the fresh codes against the stored codes of the NEXT picked row (one row
        # off), and against each row's own stored codes transposed on the 8 x 8 grid
        off = (t == np.roll(stored, 1, axis=0))
        tr_ = stored.reshape(-1, 8, 8).transpose(0, 2, 1).reshape(-1, PER)
        res["control_one_row_off_tokens_equal"] = float(off.mean())
        res["control_transposed_tokens_equal"] = float((t == tr_).mean())
        res["control_fails"] = bool(not off.all() and not (t == tr_).all())
        print(json.dumps(res), flush=True)
        (out / "verify.json").write_text(json.dumps(res, indent=1))
        sys.exit(0 if res["pass"] and res["control_fails"] else 1)

    out.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():                             # one card per artifact: codes can differ across cards
        prev_gpu = json.loads(meta_path.read_text()).get("gpu")
        if prev_gpu and prev_gpu != gpu:
            raise SystemExit(f"{out} was written on {prev_gpu}; refusing to add rows from {gpu}")
    n = len(inputs)
    if tpath.exists():
        tokens = np.load(tpath, mmap_mode="r+")
        assert tokens.shape == (n, PER) and tokens.dtype == np.uint16
    else:
        tokens = np.lib.format.open_memmap(tpath, mode="w+", dtype=np.uint16, shape=(n, PER))
    present = np.load(ppath) if ppath.exists() else np.zeros(n, bool)
    todo = sel[~present[sel.row.values]]
    if a.limit:
        todo = todo.iloc[: a.limit]
    print(f"pastis p1 {SPLITS}: {len(sel)} rows selected, {len(todo)} to go, imputation {method}, {NATIVE} px, on {gpu}",
          flush=True)
    t0 = time.time()
    for c0 in range(0, len(todo), CHUNK):
        ch = todo.iloc[c0:c0 + CHUNK]
        t = encode_fixed(tok, native_inputs(ch, views, method))
        assert t.shape == (len(ch), PER) and t.min() >= 0 and t.max() < C.CODEBOOK["S2L2A"]
        tokens[ch.row.values] = t.astype(np.uint16)
        tokens.flush()
        present[ch.row.values] = True
        with open(str(ppath) + ".tmp", "wb") as fh:
            np.save(fh, present)
        os.replace(str(ppath) + ".tmp", ppath)
        done = c0 + len(ch)
        print(f"  {done}/{len(todo)}  {done / (time.time() - t0):.1f} rows/s", flush=True)

    split_of = views.set_index("view").split
    tk = np.asarray(tokens[present])
    by_split = {sp: int(present[inputs.row[inputs.view.map(split_of) == sp].values].sum()) for sp in ("train", "val", "test")}
    meta = {"task": "pastis", "inputs": str(V.TOK_ROOT["pastis"] / "inputs.parquet"), "n_rows": n,
            "n_present": int(present.sum()), "present_by_split": by_split, "frames": ["p1"], "splits": list(SPLITS),
            "tokenizer": C.TOKENIZER["S2L2A"], "crop": NATIVE, "grid": [NATIVE // 16, NATIVE // 16], "tokens_per_sample": PER,
            "imputation": method, "linear_coef": G.LINEAR if method == "linear" else None, "linear_source": G.LINEAR_SOURCE,
            "resample": "none: native 128 px (10 m); the encoder's 16 x 16 sin-cos position table is resized to 8 x 8",
            "stats": "v1_pretraining (contract.V1_TOK_MEAN/STD)", "batch": BATCH, "tail_padding": "first row repeated",
            "tf32": False, "flatten_order": C.FLATTEN_ORDER,
            "token_min": int(tk.min()) if len(tk) else None, "token_max": int(tk.max()) if len(tk) else None,
            "codes_used": int(len(np.unique(tk))) if len(tk) else 0,
            "contract_check": "/data/enric/reports/a2_seg/meeting_2026-11-25/probes/native128/contract128.json",
            "gpu": gpu, "fork_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                                       capture_output=True, text=True).stdout.strip(),
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    if by_split["test"]:
        raise SystemExit("test rows present: refused")
    with open(str(meta_path) + ".tmp", "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(str(meta_path) + ".tmp", meta_path)
    print(json.dumps({k: meta[k] for k in ("n_present", "present_by_split", "codes_used")}), flush=True)


if __name__ == "__main__":
    main()
