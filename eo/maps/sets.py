"""The fixed map sets of A2 Step 1: training pools and read sets, frozen with a manifest.

    dataset      train (training pool)                  val (read set)
    pastis       PASTIS-R folds 1-3, 1,455 maps, 128 px  fold 4, 482 maps
    cloudsen12   GEO-Bench-2 train, 4,000 tiles, 512 px  GEO-Bench-2 validation, 535 tiles
    lulc         TerraMesh val LULC, 224 px centre crop, by 5 km group (eo/data/eval_split.py):
                 rows outside eval_rows_v1 and outside every group that holds a held-out row or a
                 temporal dev/test location (all of ssl4eos12); the eligible groups are split into a
                 read set (about 2,000 maps) and a training pool, no group in both

`train_sub` is a fixed subset of `train` (PASTIS: all of folds 1-3; CloudSen12: 1,000 tiles; LULC: 5,000
maps). The decoding rules' parameters are chosen on it and the code statistics take it as their
reference. `monitor` is another fixed subset of `train` (PASTIS 100 patches, CloudSen12 200 tiles, LULC
whole 5 km groups, 1,000+ maps) and `fit` is the rest: a trained tokenizer trains on `fit` only and
picks its checkpoints and stops on `monitor`, so the read sets (`val`) are read once, at the end. The test splits (PASTIS fold 5, CloudSen12 test) are never extracted, and `load` refuses
them: Step 1 never reads a test fold.

Built by `python -m eo.maps.build_sets` (a2seg env); every array's sha256 is in the manifest, and
`load` checks it once per process.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

# A2_MAPS_ROOT overrides it (a copy of the same files elsewhere, e.g. on RCP's lts4 scratch); every array
# is still checked against the manifest's sha256.
ROOT = Path(os.environ.get("A2_MAPS_ROOT", "/data/enric/reports/a2_seg/step1/maps"))
MANIFEST = ROOT / "manifest.json"
DATASETS = ("pastis", "cloudsen12", "lulc")
SPLITS = ("train", "train_sub", "monitor", "fit", "val")
FORBIDDEN = ("test",)


class ForbiddenSplit(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=None)
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


@lru_cache(maxsize=None)
def _checked(path: str, sha: str) -> bool:
    got = _sha256(Path(path))
    if got != sha:
        raise RuntimeError(f"{path}: sha256 {got} does not match the manifest's {sha}")
    return True


def load(dataset: str, split: str, mmap: bool = True, check: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(labels (N, H, W) uint8, ids (N,)) of one fixed set. ids are PASTIS patch ids, CloudSen12
    tortilla ids, or TerraMesh val row indices (tok_index.parquet order)."""
    if split in FORBIDDEN or "test" in split:
        raise ForbiddenSplit(f"{dataset}/{split}: Step 1 never reads a test split")
    if dataset not in DATASETS or split not in SPLITS:
        raise KeyError(f"unknown set {dataset}/{split}")
    ent = manifest()["sets"][f"{dataset}/{'val' if split == 'val' else 'train'}"]
    lab_path = ROOT / ent["labels"]
    if check:
        _checked(str(lab_path), ent["labels_sha256"])
    labels = np.load(lab_path, mmap_mode="r" if mmap else None)
    ids = np.load(ROOT / ent["ids"], allow_pickle=False)
    if split in ("train_sub", "monitor", "fit"):
        sub = subset_indices(dataset, split)
        return labels[sub], ids[sub]
    return labels, ids


def subset_indices(dataset: str, split: str) -> np.ndarray:
    """Indices into `train` of train_sub, monitor or fit (fit = train less monitor). For LULC's 62,936
    maps, a trainer should index the memory-mapped `train` with these rather than load `fit` whole."""
    ent = manifest()["sets"][f"{dataset}/train"]
    if split == "train_sub":
        return np.load(ROOT / ent["subset"])
    mon = np.load(ROOT / ent["monitor"])
    if split == "monitor":
        return mon
    if split == "fit":
        return np.setdiff1d(np.arange(ent["n"]), mon)
    raise KeyError(split)


def training_maps() -> dict:
    """The dedicated tokenizer's training maps: per dataset, (the memory-mapped training pool, the
    indices of its `fit` part). A tokenizer trains on these and nothing else."""
    return {d: (load(d, "train")[0], subset_indices(d, "fit")) for d in DATASETS}
