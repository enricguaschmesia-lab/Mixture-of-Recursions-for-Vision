#!/usr/bin/env python
"""A2 Step 7.0e test 3: fold 4 in all 8 dihedral variants, tokenized exactly as training's were, into a SEPARATE root
(docs/plans/a2_segmentation.md, Step 7, "7.0e"; `a2seg` env, GPU). Today's artifacts are read, never written.

    python eo/experiments/segmentation/step70e_tta_tokens.py --part index
    python eo/experiments/segmentation/step70e_tta_tokens.py --part s2   --expect-gpu "GTX TITAN X"   # as S2L2A_tok224
    python eo/experiments/segmentation/step70e_tta_tokens.py --part maps --expect-gpu "TITAN V"       # as MAP_aligned
    python eo/experiments/segmentation/step70e_tta_tokens.py --part maps --arm c4375_224 --expect-gpu "TITAN V"  # 7.0f

The root /data/enric/data/geobench2_tta/pastis/tok/ holds fold 4's 482 patches only, PATCH-MAJOR: view 8 i + k is
patch i (in today's val order) under dihedral variant k (eo/geobench/s2.dihedral), so a patch's 8 views are
consecutive and `--limit 16` is two whole patches. Each view has today's 8 input frames (p1, p7_0..6: the same series
index t and date as its k = 0 view).
  * index   views.parquet and inputs.parquet (today's columns), and the copy map: the k = 0 rows' tokens are copied
            from today's artifacts, by (patch, frame) for S2 and by patch for the map;
  * s2      the k != 0 rows by eo/scripts/tokenize_geobench.py's own build_inputs and encode_fixed (B01/B09 imputed by
            s2.CHOSEN_IMPUTATION, one bilinear resample to 224, v1 stats, the variant applied to the native raster
            first; batch 32, the tail padded with the first row; TF32 off), on the card today's S2 artifact was written on;
  * maps    the k != 0 rows by eo/scripts/tokenize_maps.py's own view_labels and encode_maps (frozen aligned arm, its
            palette, run_fixed at its batch), on the card today's MAP_aligned was written on.
Each artifact's metadata.json is today's with the counts, the root and a `tta` record. Resumable per chunk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/enric/hf")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "scripts"))

SRC = Path("/data/enric/data/geobench2/pastis/tok")
DST = Path("/data/enric/data/geobench2_tta/pastis/tok")
FRAMES = ["p1"] + [f"p7_{j}" for j in range(7)]
CHUNK = 512


def build_index(src: Path = SRC):
    """(views, inputs, s2_copy: {new row: today's row} for k = 0, map_copy: {new view: today's view} for k = 0)."""
    v0 = pd.read_parquet(src / "views.parquet")
    x0 = pd.read_parquet(src / "inputs.parquet")
    val = v0[(v0.split == "val") & (v0.k == 0)].sort_values("view")
    if len(val) != 482:
        raise RuntimeError(f"today's val holds {len(val)} k = 0 views, not 482")
    rows, irows, s2_copy, map_copy = [], [], {}, {}
    xin = x0.set_index(["view", "frame"])
    for i, r in enumerate(val.itertuples()):
        for k in range(8):
            nv = 8 * i + k
            rows.append((nv, "val", int(r.sample_idx), int(r.sample_id), 0, 0, 128, k, bool(r.in_pinned)))
            if k == 0:
                map_copy[nv] = int(r.view)
            for f in FRAMES:
                src_row = xin.loc[(int(r.view), f)]
                nr = len(irows)
                irows.append((nr, nv, f, int(src_row.t), int(src_row.date)))
                if k == 0:
                    s2_copy[nr] = int(src_row.row)
    views = pd.DataFrame(rows, columns=list(v0.columns))
    inputs = pd.DataFrame(irows, columns=list(x0.columns))
    for c in views.columns:
        views[c] = views[c].astype(v0[c].dtype)
    for c in inputs.columns:
        inputs[c] = inputs[c].astype(x0[c].dtype)
    return views, inputs, s2_copy, map_copy


def write_index(dst: Path = DST) -> dict:
    views, inputs, s2_copy, map_copy = build_index()
    dst.mkdir(parents=True, exist_ok=True)
    for name, df in (("views", views), ("inputs", inputs)):
        p = dst / f"{name}.parquet"
        if p.exists() and not pd.read_parquet(p).equals(df):
            raise RuntimeError(f"{p} exists and differs from the index built now; refusing to overwrite")
        if not p.exists():
            df.to_parquet(p)
    copy = {"s2": {str(k): v for k, v in s2_copy.items()}, "map": {str(k): v for k, v in map_copy.items()}}
    (dst / "tta_copy.json").write_text(json.dumps(copy) + "\n")
    return {"views": len(views), "inputs": len(inputs), "s2_copied": len(s2_copy), "maps_copied": len(map_copy)}


def _open(path: Path, shape):
    if path.exists():
        a = np.load(path, mmap_mode="r+")
        if a.shape != shape or a.dtype != np.uint16:
            raise RuntimeError(f"{path}: {a.shape} {a.dtype}, expected {shape} uint16")
        return a
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=shape)


def _save_present(path: Path, present: np.ndarray) -> None:
    with open(str(path) + ".tmp", "wb") as fh:
        np.save(fh, present)
    os.replace(str(path) + ".tmp", path)


def _meta(src_meta: Path, out_meta: Path, **upd) -> None:
    m = json.loads(src_meta.read_text())
    m.update(upd)
    with open(str(out_meta) + ".tmp", "w") as fh:
        json.dump(m, fh, indent=1, default=str)
    os.replace(str(out_meta) + ".tmp", out_meta)


def part_s2(limit=None, dst: Path = DST) -> None:
    import torch
    import tokenize_geobench as TG
    from eo.geobench import s2 as G
    from eo.terramesh_tok import contract as C, tokenizers as T
    gpu = torch.cuda.get_device_name(0)
    src_meta = json.loads((SRC / "S2L2A_tok224" / "metadata.json").read_text())
    if src_meta["gpu"] != gpu:
        raise SystemExit(f"today's S2 artifact was written on {src_meta['gpu']}; this is {gpu}")
    views, inputs = pd.read_parquet(dst / "views.parquet"), pd.read_parquet(dst / "inputs.parquet")
    copy = {int(k): v for k, v in json.loads((dst / "tta_copy.json").read_text())["s2"].items()}
    out = dst / "S2L2A_tok224"
    out.mkdir(parents=True, exist_ok=True)
    n = len(inputs)
    tokens = _open(out / "tokens.npy", (n, C.TOKENS_PER_SAMPLE))
    present = np.load(out / "present.npy") if (out / "present.npy").exists() else np.zeros(n, bool)
    src_tok = np.load(SRC / "S2L2A_tok224" / "tokens.npy", mmap_mode="r")
    rows = np.array(sorted(copy))
    tokens[rows] = src_tok[[copy[r] for r in rows]]
    tokens.flush()
    present[rows] = True
    _save_present(out / "present.npy", present)
    tok = T.build("S2L2A", "cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    method = G.CHOSEN_IMPUTATION
    todo = inputs[~present[inputs.row.values]]
    todo = todo.assign(_s=todo.view.map(views.set_index("view").sample_idx)).sort_values(["_s", "row"])
    if limit:
        todo = todo.iloc[:limit]
    print(f"s2: {len(rows)} k = 0 rows copied, {len(todo)} to tokenize on {gpu}, imputation {method}", flush=True)
    t0 = time.time()
    for c0 in range(0, len(todo), CHUNK):
        ch = todo.iloc[c0:c0 + CHUNK]
        x, _ = TG.build_inputs("pastis", ch, views, method)
        t = TG.encode_fixed(tok, x)
        assert t.shape == (len(ch), C.TOKENS_PER_SAMPLE) and t.min() >= 0 and t.max() < C.CODEBOOK["S2L2A"]
        tokens[ch.row.values] = t.astype(np.uint16)
        tokens.flush()
        present[ch.row.values] = True
        _save_present(out / "present.npy", present)
        print(f"  {c0 + len(ch)}/{len(todo)}  {(c0 + len(ch)) / (time.time() - t0):.1f} rows/s", flush=True)
    _meta(SRC / "S2L2A_tok224" / "metadata.json", out / "metadata.json", inputs=str(dst / "inputs.parquet"),
          n_rows=n, n_present=int(present.sum()), gpu=gpu, written=time.strftime("%Y-%m-%d %H:%M:%S"),
          tta={"what": "A2 Step 7.0e: fold 4 x 8 dihedral variants, patch-major (view 8 i + k)",
               "k0_rows_copied_from": str(SRC / "S2L2A_tok224" / "tokens.npy"), "k0_rows": len(rows),
               "chunk": CHUNK, "order": "sorted by (sample, row), as tokenize_geobench.py"})


def part_maps(limit=None, dst: Path = DST, arm: str = "aligned") -> None:
    import torch
    import tokenize_maps as TM
    from eo.maps import frozen
    gpu = torch.cuda.get_device_name(0)
    src_meta = json.loads((SRC / f"MAP_{arm}" / "metadata.json").read_text())
    if src_meta["gpu"] != gpu:
        raise SystemExit(f"today's MAP_{arm} was written on {src_meta['gpu']}; this is {gpu}")
    views = pd.read_parquet(dst / "views.parquet")
    copy = {int(k): v for k, v in json.loads((dst / "tta_copy.json").read_text())["map"].items()}
    tok = frozen.load(arm)
    T_ = tok.frozen["tokens"]
    out = dst / f"MAP_{arm}"
    out.mkdir(parents=True, exist_ok=True)
    n = len(views)
    tokens = _open(out / "tokens.npy", (n, T_))
    present = np.load(out / "present.npy") if (out / "present.npy").exists() else np.zeros(n, bool)
    src_tok = np.load(SRC / f"MAP_{arm}" / "tokens.npy", mmap_mode="r")
    vs = np.array(sorted(copy))
    tokens[vs] = src_tok[[copy[v] for v in vs]]
    tokens.flush()
    present[vs] = True
    _save_present(out / "present.npy", present)
    todo = views[~present[views.view.values]]
    if limit:
        todo = todo.iloc[:limit]
    print(f"maps: {len(vs)} k = 0 maps copied, {len(todo)} to tokenize on {gpu}", flush=True)
    for c0 in range(0, len(todo), 2048):
        ch = todo.iloc[c0:c0 + 2048]
        codes = TM.encode_maps(tok, "pastis", TM.view_labels("pastis", ch))
        assert codes.shape == (len(ch), T_) and codes.min() >= 0 and codes.max() < tok.n_codes
        tokens[ch.view.values] = codes.astype(np.uint16)
        tokens.flush()
        present[ch.view.values] = True
        _save_present(out / "present.npy", present)
        print(f"  {c0 + len(ch)}/{len(todo)}", flush=True)
    by = {s: int(present[views.view.values[views.split == s]].sum()) for s in ("train", "val", "test")}
    _meta(SRC / f"MAP_{arm}" / "metadata.json", out / "metadata.json", views=str(dst / "views.parquet"),
          n_views=n, n_present=int(present.sum()), present_by_split=by, gpu=gpu,
          written=time.strftime("%Y-%m-%d %H:%M:%S"),
          tta={"what": "A2 Step 7.0e: fold 4 x 8 dihedral variants, patch-major (view 8 i + k)",
               "k0_maps_copied_from": str(SRC / f"MAP_{arm}" / "tokens.npy"), "k0_maps": len(vs)})


def check(part: str, n: int, dst: Path = DST, arm: str = "aligned") -> dict:
    """Gate Z33d: re-tokenize the first n new rows (S2: in the written order and batch alignment, sorted by (sample,
    row); maps: the first n new views) and compare with the stored tokens; control: the same rows read at k = 0."""
    import torch
    views = pd.read_parquet(dst / "views.parquet")
    if part == "s2":
        import tokenize_geobench as TG
        from eo.geobench import s2 as G
        from eo.terramesh_tok import tokenizers as T
        inputs = pd.read_parquet(dst / "inputs.parquet")
        copy = {int(k) for k in json.loads((dst / "tta_copy.json").read_text())["s2"]}
        new = inputs[~inputs.row.isin(copy)]
        new = new.assign(_s=new.view.map(views.set_index("view").sample_idx)).sort_values(["_s", "row"]).iloc[:n]
        tok = T.build("S2L2A", "cuda")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        stored = np.load(dst / "S2L2A_tok224" / "tokens.npy", mmap_mode="r")[new.row.values]
        again = TG.encode_fixed(tok, TG.build_inputs("pastis", new, views, G.CHOSEN_IMPUTATION)[0])
        v0 = views.copy()
        v0["k"] = 0
        ctl = TG.encode_fixed(tok, TG.build_inputs("pastis", new, v0, G.CHOSEN_IMPUTATION)[0])
    else:
        import tokenize_maps as TM
        from eo.maps import frozen
        new = views[views.k != 0].iloc[:n]
        tok = frozen.load(arm)
        stored = np.load(dst / f"MAP_{arm}" / "tokens.npy", mmap_mode="r")[new.view.values]
        again = TM.encode_maps(tok, "pastis", TM.view_labels("pastis", new))
        ctl = TM.encode_maps(tok, "pastis", TM.view_labels("pastis", new.assign(k=0)))
    return {"part": part, "arm": arm if part == "maps" else None, "rows": int(len(new)),
            "equal": bool(np.array_equal(stored, again)),
            "control_k0_equal": bool(np.array_equal(stored, ctl)),
            "control_k0_token_share_equal": float((np.asarray(stored) == ctl).mean()),
            "gpu": torch.cuda.get_device_name(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["index", "s2", "maps", "check_s2", "check_maps"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--arm", default="aligned", help="maps / check_maps: the map arm (A2 Step 7.0f: c4375_224)")
    a = ap.parse_args()
    if a.part == "index":
        print(json.dumps(write_index()), flush=True)
        return
    import torch
    if a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    if a.part.startswith("check_"):
        print(json.dumps(check(a.part[6:], a.limit or (64 if a.part == "check_s2" else 16), arm=a.arm)), flush=True)
        return
    if a.part == "s2":
        part_s2(a.limit)
    else:
        part_maps(a.limit, arm=a.arm)


if __name__ == "__main__":
    main()
