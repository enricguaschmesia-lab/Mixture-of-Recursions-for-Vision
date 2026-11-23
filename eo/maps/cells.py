"""A2 Step 4.2's Route B: one class token per cell of a 28 x 28 grid at 8 px on the 224 canvas (plan Step 4.2, item 2).

No tokenizer: a cell's code is the task's label (PASTIS 0-18 and VOID 19; CloudSen12's 4 classes), written by the
decoder through the aliases `cells_<task>` of eo/data/map_vocab.py.

THE RULE
  targets   a view's native label map (its crop and dihedral variant) is rendered to the canvas by the harness's nearest
            mapping (palette.render's F.interpolate(mode="nearest"); the identity for CloudSen12's native 224-px crops).
            A cell's label is the majority over its 8 x 8 canvas pixels that are not void, ties to the lower class. A
            cell with no non-void pixel is VOID (the task's ignored class; PASTIS only).
  back      cells (N, 28, 28) -> each cell's one-hot class score repeated over its 8 x 8 canvas pixels -> pooled to the
            native grid by eo/maps/decode.to_native (the render's adjoint: a native pixel's score for a class is the
            share of its rendered pixels lying in a cell of that class) -> (N, C, native, native). Stitching CloudSen12's
            crops, the nearest rule (argmax, void masked: a pixel whose rendered pixels all lie in VOID cells takes class
            0) and the K-sample vote are Route A's (eo/eval/seg_routes.py).

TWO INDEPENDENT PATHS (gate Z19, eo/scripts/verify_seg_routes.py)
  *_torch   the harness's own calls: F.interpolate(nearest) to render, one-hot sums over 8 x 8 blocks, decode.to_native
  *_numpy   the same geometry written out: canvas pixel x copies native pixel floor(x * native / canvas) (F.interpolate's
            legacy nearest, whose float scale rounds to this exactly at these sizes: an exact multiple of 1/7 never comes
            within the float error of an integer), integer counts by np.add.at, no torch.
They must agree cell for cell and pixel for pixel; a transposed grid is the control that must not.

Runs in `.venv` and `a2seg`: numpy and torch only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

CANVAS = 224
GRID = 28
CELL = CANVAS // GRID            # 8 px
RULE = {"rule": "cells", "canvas": CANVAS, "grid": GRID, "cell_px": CELL, "render": "nearest (palette.render)",
        "majority": "non-void pixels, ties to the lower class", "empty": "VOID (the ignored class)",
        "back": "one-hot x 8 x 8, decode.to_native, nearest rule (void masked), K-sample vote"}


# ------------------------------------------------------------------------------------------------ path 1: torch
def render(labels: torch.Tensor, canvas: int = CANVAS) -> torch.Tensor:
    """(N, H, W) int -> (N, canvas, canvas) long: palette.render's nearest mapping applied to the labels themselves."""
    lab = labels.long()
    if lab.shape[-1] == canvas and lab.shape[-2] == canvas:
        return lab
    return F.interpolate(lab[:, None].float(), size=(canvas, canvas), mode="nearest")[:, 0].round().long()


def cells_torch(labels: torch.Tensor, n_classes: int, ignore: Optional[int], canvas: int = CANVAS,
                grid: int = GRID) -> torch.Tensor:
    """(N, H, W) native labels -> (N, grid, grid) long cell labels (the rule above)."""
    lab = render(labels, canvas)
    cell = canvas // grid
    n = len(lab)
    cnt = torch.stack([(lab == k).reshape(n, grid, cell, grid, cell).sum((2, 4)) for k in range(n_classes)], 1)
    if ignore is not None:                                                       # cnt: (N, C, grid, grid) int64
        cnt[:, ignore] = 0
    out = cnt.argmax(1)                                                          # the first maximum: the lower class
    if ignore is not None:
        out[cnt.sum(1) == 0] = ignore
    return out


def cell_scores(cells: torch.Tensor, n_classes: int, native: int, canvas: int = CANVAS) -> torch.Tensor:
    """(N, grid, grid) cell labels -> (N, C, native, native) float scores: one-hot per cell, repeated over its canvas
    pixels, pooled to the native grid by the render's adjoint (eo/maps/decode.to_native)."""
    from eo.maps import decode as D
    g = cells.shape[-1]
    cell = canvas // g
    oh = F.one_hot(cells.long(), n_classes).permute(0, 3, 1, 2).float()          # (N, C, g, g)
    up = oh.repeat_interleave(cell, 2).repeat_interleave(cell, 3)               # (N, C, canvas, canvas)
    return D.to_native(up, native)


def predict_torch(cells: torch.Tensor, n_classes: int, ignore: Optional[int], native: int) -> torch.Tensor:
    """(N, grid, grid) cells -> (N, native, native) classes: the nearest rule on the pooled scores (one crop)."""
    from eo.maps import decode as D
    return D.apply_rule(cell_scores(cells, n_classes, native), ignore, {"rule": "nearest"}, "logits")


# ------------------------------------------------------------------------------------------------ path 2: numpy
def _source(native: int, canvas: int = CANVAS) -> np.ndarray:
    """Canvas pixel x -> the native pixel the nearest render copies to it."""
    return (np.arange(canvas, dtype=np.int64) * native) // canvas


def _weights(native: int, canvas: int = CANVAS, grid: int = GRID) -> np.ndarray:
    """(grid, native) int: W[r, i] = how many canvas rows lie in cell row r and copy native row i (same on both axes)."""
    W = np.zeros((grid, native), np.int64)
    np.add.at(W, (np.arange(canvas) // (canvas // grid), _source(native, canvas)), 1)
    return W


def cells_numpy(labels: np.ndarray, n_classes: int, ignore: Optional[int], canvas: int = CANVAS,
                grid: int = GRID) -> np.ndarray:
    """The same rule as cells_torch, written from the geometry: counts[n, k, r, c] = sum_ij W[r, i] W[c, j] [label = k]."""
    labels = np.asarray(labels)
    W = _weights(labels.shape[-1], canvas, grid)
    out = np.empty((len(labels), grid, grid), np.int64)
    for n in range(len(labels)):
        cnt = np.stack([W @ (labels[n] == k).astype(np.int64) @ W.T for k in range(n_classes)])   # (C, grid, grid)
        if ignore is not None:
            cnt[ignore] = 0
        out[n] = cnt.argmax(0)                                                   # numpy's first maximum: the lower class
        if ignore is not None:
            out[n][cnt.sum(0) == 0] = ignore
    return out


def predict_numpy(cells: np.ndarray, n_classes: int, ignore: Optional[int], native: int, canvas: int = CANVAS) -> np.ndarray:
    """The back-projection from the geometry: a native pixel's count per class over its rendered pixels, argmax with the
    ignored class excluded (ties and empty pixels to the lowest class, as argmax over the pooled scores)."""
    cells = np.asarray(cells)
    g = cells.shape[-1]
    W = _weights(native, canvas, g)                                               # (g, native)
    out = np.empty((len(cells), native, native), np.int64)
    for n in range(len(cells)):
        cnt = np.stack([W.T @ (cells[n] == k).astype(np.int64) @ W for k in range(n_classes)])   # (C, native, native)
        if ignore is not None:
            cnt[ignore] = -1
        out[n] = cnt.argmax(0)
    return out


# ------------------------------------------------------------------------------------------------ views
def view_labels(labels: np.ndarray, rows: np.ndarray, y0: np.ndarray, x0: np.ndarray, size: np.ndarray,
                k: np.ndarray) -> np.ndarray:
    """(n, size, size) uint8: each view's native label crop and dihedral variant, as eo/scripts/tokenize_maps.py cut them
    (eo/geobench/s2.dihedral: k % 4 quarter turns counter-clockwise, then a left-right flip if k >= 4)."""
    from eo.geobench.s2 import dihedral
    out = [dihedral(np.asarray(labels[int(r)])[int(a):int(a) + int(s), int(b):int(b) + int(s)], int(kk))
           for r, a, b, s, kk in zip(rows, y0, x0, size, k)]
    return np.stack(out).astype(np.uint8)


def set_rows(task: str, ids: np.ndarray, sample_ids) -> np.ndarray:
    """The row of each view's sample in an eo/maps/sets array (PASTIS ids are patch numbers, CloudSen12's tile names)."""
    key = (lambda s: str(s)) if task == "cloudsen12" else (lambda s: int(s))
    where = {key(i): n for n, i in enumerate(ids)}
    missing = [s for s in sample_ids if key(s) not in where]
    if missing:
        raise RuntimeError(f"{len(missing)} samples are not in eo/maps' {task} set, e.g. {missing[:3]}")
    return np.asarray([where[key(s)] for s in sample_ids], np.int64)


def build_artifact(task: str, root, chunk: int = 2048, check_numpy: int = 64, log=print) -> dict:
    """Write <root>/<task>/tok/MAP_cells_<task>/ (tokens.npy (views, 784) uint16 in raster order, present.npy, metadata.json)
    for the train and val views; test views stay absent and zero (their labels are never read, A2 Steps 3-5). Each chunk's
    first `check_numpy` views are recomputed by the numpy path and must agree (the full check is gate Z19-Z20)."""
    import hashlib
    import json
    import time
    from pathlib import Path

    import pandas as pd

    from eo.data import map_vocab as MV
    from eo.maps import palette as P, sets

    t = P.TASKS[task]
    arm = MV.CELL_ARMS[task]
    codes, n_tok, side = MV.spec(arm)
    if codes != t.n_classes or n_tok != GRID * GRID or side != GRID:
        raise RuntimeError(f"{arm}: {codes} codes, {n_tok} tokens, side {side}; the task has {t.n_classes} classes")
    tok = Path(root) / task / "tok"
    views = pd.read_parquet(tok / "views.parquet")
    out = tok / f"MAP_{arm}"
    out.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(out / "tokens.npy.part", mode="w+", dtype=np.uint16, shape=(len(views), n_tok))
    present = np.zeros(len(views), bool)
    label_sha, counts, t0, n_checked = {}, {}, time.time(), 0
    for split in ("train", "val"):
        counts[split] = np.zeros(codes, np.int64)
        labels, ids = sets.load(task, split)
        label_sha[split] = sets.manifest()["sets"][f"{task}/{split}"]["labels_sha256"]
        vs = views[views.split == split]
        rows = set_rows(task, ids, vs.sample_id.tolist())
        for s in range(0, len(vs), chunk):
            v = vs.iloc[s:s + chunk]
            lab = view_labels(labels, rows[s:s + chunk], v.y0.values, v.x0.values, v["size"].values, v.k.values)
            c = cells_torch(torch.from_numpy(lab), t.n_classes, t.ignore).numpy()
            if check_numpy:
                c2 = cells_numpy(lab[:check_numpy], t.n_classes, t.ignore)
                if not np.array_equal(c[:check_numpy], c2):
                    raise RuntimeError(f"{task} {split} views {s}..: the two paths disagree")
                n_checked += len(c2)
            tokens[v.view.values] = c.reshape(len(c), -1).astype(np.uint16)
            present[v.view.values] = True
            counts[split] += np.bincount(c.ravel(), minlength=codes)
        log(f"{task} {split}: {len(vs)} views ({time.time() - t0:.0f} s)")
    tokens.flush()
    del tokens
    (out / "tokens.npy.part").replace(out / "tokens.npy")
    np.save(out / "present.npy", present)
    h = hashlib.sha256()
    with open(out / "tokens.npy", "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    src = Path(__file__).read_bytes()
    meta = {"arm": arm, "task": task, "n_views": int(len(views)), "splits": ["train", "val"],
            "absent": "test views: zero and present=False; their labels are never read (A2 Steps 3-5)",
            "frozen": {"kind": "cells", "tokens": n_tok, "n_codes": codes, "grid": GRID, "rule": RULE,
                       "classes": list(t.class_names), "ignore": t.ignore, "native": t.native,
                       "cells_py_sha256": hashlib.sha256(src).hexdigest()},
            "labels_sha256": label_sha, "tokens_sha256": h.hexdigest(),
            "cell_counts": {k: v.tolist() for k, v in counts.items()},
            "checked_by_numpy": n_checked, "seconds": time.time() - t0, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"wrote {out}: {int(present.sum())} of {len(views)} views, sha256 {meta['tokens_sha256'][:12]}")
    return meta
