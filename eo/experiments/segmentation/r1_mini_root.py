#!/usr/bin/env python
"""A2 R1's gate fixture (`.venv`, CPU): a two-shard train token root on the workstation, with both corpora, and the
map artifact row-aligned with it, so gate Z15 and preflight can build R1's data without g49.

    root   /data/enric/data/TerraMesh/train_tok_partial_r1/: train_split.consolidate itself, run on a view of the
           workstation's per-shard train tokens (data/TerraMesh/train_tok, encoded on the TITAN V) holding only
           majortom_shard_000004 and ssl4eos12_shard_000048: 20,000 rows, canonical order, rows renumbered, a PARTIAL
           build.json (so a run needs multimodal.allow_partial_train_root, smoke runs only).
    maps   /data/enric/data/TerraMesh/maps/mini_r1/MAP_aligned/: the full train map artifact's rows of the
           ssl4eos12 shard, moved to the mini root's rows by stem, with a metadata.json naming the mini root's
           tok_index.parquet sha256 and the source artifact (its metadata sha256 and the shard's done record).

    python eo/experiments/segmentation/r1_mini_root.py [--source /data/enric/data/TerraMesh/maps/train/MAP_aligned]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from eo.terramesh_tok import contract as C  # noqa: E402
from eo.terramesh_tok import train_split as TS  # noqa: E402

TM = Path("/data/enric/data/TerraMesh")
SHARDS = ("majortom_shard_000004.tar", "ssl4eos12_shard_000048.tar")
ROOT = TM / "train_tok_partial_r1"
MAPS = TM / "maps/mini_r1/MAP_aligned"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build_root(src: Path, dest: Path) -> dict:
    """consolidate() on a view of `src` holding only SHARDS (symlinks; nothing in src is written)."""
    with tempfile.TemporaryDirectory(dir="/data/enric/data/TerraMesh") as tmp:
        view = Path(tmp)
        (view / "done").mkdir()
        for s in SHARDS:
            os.symlink(src / "done" / s.replace(".tar", ".json"), view / "done" / s.replace(".tar", ".json"))
        for f in ("tok_index.parquet", "plan.json"):
            os.symlink(src / f, view / f)
        for m in TS.OUT_MODS:
            (view / C.tok_dir_name(m)).mkdir()
            os.symlink(src / C.tok_dir_name(m) / "shards", view / C.tok_dir_name(m) / "shards")
        if dest.exists():
            shutil.rmtree(dest)
        return TS.consolidate(view, TM / "val", allow_partial=True, dest=dest)


def build_maps(source: Path, root: Path, dest: Path) -> dict:
    full_idx = pd.read_parquet(TM / "train_coords/tok_index.parquet")
    mini_idx = pd.read_parquet(root / "tok_index.parquet")
    smeta = json.loads((source / "metadata.json").read_text()) if (source / "metadata.json").exists() else None
    card = json.loads((source / "card.json").read_text())
    if card["index_sha256"] != sha256_file(TM / "train_coords/tok_index.parquet"):
        raise SystemExit("the source artifact was not written against the train root's index")
    tokens_src = np.load(source / "tokens.npy", mmap_mode="r")
    present_src = np.load(source / "present.npy")
    ssl = [s for s in SHARDS if s.startswith("ssl4eos12")]
    done = {s: json.loads((source / "done" / f"{s}.json").read_text()) for s in ssl}     # the writer keeps .tar
    full_row = dict(zip(full_idx.stem, full_idx.row))
    n = len(mini_idx)
    T = tokens_src.shape[1]
    tokens = np.zeros((n, T), np.uint16)
    present = np.zeros(n, bool)
    sel = mini_idx[mini_idx.source_shard.isin(ssl)]
    fr = np.array([full_row[s] for s in sel.stem], np.int64)
    if not present_src[fr].all():
        raise SystemExit("the source artifact lacks rows of the shard")
    tokens[sel.row.values] = np.asarray(tokens_src[fr])
    present[sel.row.values] = True
    dest.mkdir(parents=True, exist_ok=True)
    TS.atomic_npy(tokens, dest / "tokens.npy")
    TS.atomic_npy(present, dest / "present.npy")
    base = smeta or {}
    meta = {**{k: base.get(k) for k in ("split", "arm", "task", "frozen", "palette", "palette_sha256", "assignment",
                                        "render", "crop", "crop_window", "tokens_per_sample", "tokenizer_id_bound",
                                        "dtype", "flatten", "batch", "run_fixed", "gpu", "revision")},
            "derived": "rows of the full train map artifact moved by stem onto a two-shard mini root (R1's gate)",
            "source": str(source), "source_metadata_sha256": sha256_file(source / "metadata.json") if smeta else None,
            "source_done": done, "index": str(root / "tok_index.parquet"),
            "index_sha256": sha256_file(root / "tok_index.parquet"), "n_rows": n, "n_present": int(present.sum()),
            "complete": True, "codes_used": int(len(np.unique(tokens[present]))), "token_max": int(tokens[present].max()),
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    if smeta is None:                                    # the source still writing: take the frozen record from its card
        raise SystemExit("the source artifact has no metadata.json yet (still writing); rerun when it is complete")
    TS.atomic_json(meta, dest / "metadata.json")
    TS.atomic_json({k: card[k] for k in card}, dest / "card.json")
    return {"rows": n, "present": int(present.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=TM / "maps/train/MAP_aligned")
    ap.add_argument("--maps-only", action="store_true")
    a = ap.parse_args()
    if not a.maps_only:
        b = build_root(TM / "train_tok", ROOT)
        print("root:", json.dumps({k: b[k] for k in ("rows", "shards", "partial", "modalities", "offset_free")}))
    print("maps:", build_maps(a.source, ROOT, MAPS))


if __name__ == "__main__":
    main()
