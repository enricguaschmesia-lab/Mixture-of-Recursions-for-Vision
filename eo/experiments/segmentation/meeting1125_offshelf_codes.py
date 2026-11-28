#!/usr/bin/env python
"""Meeting of 2026-11-25, topic 1: how spread each map tokenizer's codes are, as released or trained (CPU only).

A2 Step 4.0's follow-up 2 found that a Cosmos fine-tuned to the aligned tokenizer's ceiling, geometry and codebook size
generates 2.8 points worse, and described (one seed, untested) that it spreads the same maps over more codes. This
script measures that spread for every 224-px tokenizer that matters to the question, from the tokens the harness
already cached (eo/maps/harness.tokens; never re-encoded here), against each token's exact rendered content:
  - distinct codes, entropy H(code), the token mass of the 1,000 commonest codes (as the harness reports);
  - H(code | content): the entropy of the code given the token's exact block of palette indices (16 x 16 at f16,
    8 x 8 at f8). 0 for a tokenizer whose code is a function of its own block. What is left is code variety that the
    block does not explain (context, synonyms): codes a decoder has to choose among without the map needing them;
  - H(content | code): how many different blocks one code stands for (the lossy side, which the ceiling reads);
  - on pure blocks (one class), the number of distinct codes per class and the share of pure tokens on their class's
    modal code.
Read rule (before any number): description only. It is set beside the generated mIoU of the tokenizers that have one at
the same geometry (aligned 0.1258 against fine-tuned Cosmos 0.0974 at Step 4.0's recipe; c4375_224n 0.1195 nested);
two points calibrate nothing, so no prediction is drawn from it. Control: a block content shuffled across tokens of
the same set (the codes kept) must raise H(code | content) to near H(code) (the measure sees the pairing).

    python eo/experiments/segmentation/meeting1125_offshelf_codes.py --out /data/enric/reports/a2_seg/meeting_2026-11-25/offshelf/codes.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eo.maps import harness as H             # noqa: E402
from eo.maps import palette as P             # noqa: E402
from eo.maps import sets, tiling             # noqa: E402

CACHE = Path("/data/enric/reports/a2_seg/step1/cache/tokens/step1-v1")
#: name -> (cache directory, factor, batch tag, provenance)
TOKS = {
    "cosmos_t1_di16_224 (as released)": ("t1_di16_224__abd6286bd49c", 16, 16, "NVIDIA Cosmos-Tokenize1-DI16x16-360p, released JIT, bf16"),
    "cosmos_ft_d224_cosmos16_lg (fine-tuned)": ("D_cosmos16_224_64000_bd5ad38599__caef6912f8a6", 16, 16, "Cosmos DI16x16 weights fine-tuned 24k steps on renders, logit head"),
    "aligned_b224_64000_long (ours, frozen pick)": ("B_f16_224_64000_17be36cf21__ecc5bab34450", 16, 16, "ours (agent B), from scratch"),
    "b224_15360_long (ours, fails LULC)": ("B_f16_224_15360_53c2486b4a__0e00a00efd8f", 16, 16, "ours (agent B), from scratch"),
    "cosmos_t1_di8_224 (as released)": ("t1_di8_224__db2f7192c2dc", 8, 16, "NVIDIA Cosmos-Tokenize1-DI8x8-360p, released JIT, bf16"),
    "c4375_224 (ours, 784 nested arm)": ("stepC_f8_224_L75555_892ae5aed7__6bb99f9b0d52", 8, 32, "ours (agent C), from scratch"),
    "c1000_224a (ours)": ("stepC_f8_224_L8555_e2518eb44f__dcc52132da06", 8, 32, "ours (agent C), from scratch"),
}


def token_path(cache_dir: str, dataset: str, split: str, batch: int) -> Path:
    hits = sorted((CACHE / cache_dir).glob(f"{dataset}_{split}_*_universal_crop224_b{batch}.npy"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{cache_dir}: {dataset}/{split}: {hits}")
    return hits[0]


def block_contents(dataset: str, split: str, factor: int, size: int = 224) -> np.ndarray:
    """(N * crops, g, g) int64 ids of each token's exact block of palette indices at render resolution, in the harness's
    crop order (harness.render_crops: cut, then nearest upsampling)."""
    labels, _ = sets.load(dataset, split)
    task = P.TASKS[dataset]
    crops = tiling.plan(dataset, task.native, "crop224")
    a = torch.as_tensor(np.asarray(task.assign), dtype=torch.long)
    g = size // factor
    out = []
    for s in range(0, len(labels), 64):
        lab = torch.as_tensor(np.ascontiguousarray(labels[s:s + 64])).long()
        idx = a[tiling.cut(lab, crops)]                                   # (n * crops, c, c) palette indices
        if idx.shape[-1] != size:
            idx = F.interpolate(idx[:, None].float(), size=(size, size), mode="nearest")[:, 0].long()
        b = idx.to(torch.uint8).reshape(-1, g, factor, g, factor).permute(0, 1, 3, 2, 4).reshape(-1, g, g, factor * factor)
        out.append(b.numpy())
    blocks = np.concatenate(out)                                          # (N*crops, g, g, f*f) uint8
    flat = np.ascontiguousarray(blocks.reshape(-1, factor * factor))
    _, inv = np.unique(flat.view(np.dtype((np.void, flat.shape[1]))).ravel(), return_inverse=True)
    pure = (flat == flat[:, :1]).all(1)
    return inv.reshape(blocks.shape[:3]), pure.reshape(blocks.shape[:3]), flat[:, 0].reshape(blocks.shape[:3])


def ent(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def joint_entropy(a: np.ndarray, b: np.ndarray) -> float:
    key = a.astype(np.int64) * (int(b.max()) + 1) + b.astype(np.int64)
    return ent(np.unique(key, return_counts=True)[1])


def stats(codes: np.ndarray, content: np.ndarray, pure: np.ndarray, colour: np.ndarray, rng) -> dict:
    c, k = codes.ravel().astype(np.int64), content.ravel().astype(np.int64)
    pu, col = pure.ravel(), colour.ravel()
    _, cc = np.unique(c, return_counts=True)
    _, kc = np.unique(k, return_counts=True)
    h_c, h_k, h_ck = ent(cc), ent(kc), joint_entropy(c, k)
    shuf = rng.permutation(k)
    h_ck_shuf = joint_entropy(c, shuf)
    out = {"tokens": int(c.size), "distinct_codes": int(len(cc)), "H_code_bits": h_c,
           "top1000_mass": float(np.sort(cc)[::-1][:1000].sum() / c.size),
           "distinct_contents": int(len(kc)), "H_content_bits": h_k,
           "H_code_given_content": h_ck - h_k, "H_content_given_code": h_ck - h_c,
           "control_shuffled_H_code_given_content": h_ck_shuf - h_k,
           "pure_share": float(pu.mean())}
    if pu.any():
        pc, pcol = c[pu], col[pu]
        per, modal, w = [], 0, []
        for v in np.unique(pcol):
            m = pcol == v
            u, n = np.unique(pc[m], return_counts=True)
            per.append(len(u)); w.append(m.sum()); modal += n.max()
        out.update({"pure_H_code_given_class": joint_entropy(pc, pcol) - ent(np.unique(pcol, return_counts=True)[1]),
                    "pure_codes_per_class_weighted": float(np.average(per, weights=w)),
                    "pure_modal_code_share": float(modal / len(pc))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    res = {"started": time.strftime("%F %T"), "cache": str(CACHE), "rows": []}
    contents = {}
    for name, (cdir, factor, batch, prov) in TOKS.items():
        for dataset in sets.DATASETS:
            for split in ("train_sub", "val"):
                key = (dataset, split, factor)
                if key not in contents:
                    contents[key] = block_contents(dataset, split, factor)
                content, pure, colour = contents[key]
                codes = np.load(token_path(cdir, dataset, split, batch))
                if codes.shape != content.shape:
                    raise RuntimeError(f"{name} {dataset}/{split}: codes {codes.shape} vs blocks {content.shape}")
                row = {"tokenizer": name, "provenance": prov, "cache": cdir, "dataset": dataset, "split": split,
                       **stats(codes, content, pure, colour, rng)}
                res["rows"].append(row)
                print(f"{name:45s} {dataset:10s} {split:9s} distinct {row['distinct_codes']:6d} H {row['H_code_bits']:.2f} "
                      f"top1k {row['top1000_mass']:.3f} H(c|blk) {row['H_code_given_content']:.2f} "
                      f"(shuffled {row['control_shuffled_H_code_given_content']:.2f}) H(blk|c) {row['H_content_given_code']:.2f} "
                      f"pure {row['pure_share']:.3f} codes/class {row.get('pure_codes_per_class_weighted', float('nan')):.0f} "
                      f"modal {row.get('pure_modal_code_share', float('nan')):.3f}", flush=True)
                Path(a.out).write_text(json.dumps(res, indent=1))
    res["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
