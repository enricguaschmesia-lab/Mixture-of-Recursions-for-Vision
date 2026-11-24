#!/usr/bin/env python
"""A2 Step 4.3, the chain's LULC: TerraMind's 196 LULC tokens the map base writes for each PASTIS-1 view (`.venv`).

Plan: docs/plans/a2_segmentation.md, Step 4, "Steps 4.3 and 4.4's design and reading" (2026-11-23, written before any of
their numbers), item 4. The map base (b10cltrxnc_ext_map_nc_ckpt33000), the weights Route A's fine-tune starts from,
reads each view's S2 block and writes LULC:

    prompt  [BO_S2L2A s1..s196 EO_S2L2A][BO_LULC]   at positions 0..198
    output  196 codes constrained to LULC's codebook slot (76,800 + 0..4,374), one sample at T 1.0

Views: the split's views in index order (train: the 1,200 pinned patches x 8 dihedral variants, 9,600; val: fold 4's
482). Batches of --batch views in that order; batch b's sampling generator is seeded SEED + its first view's index, so a
view's LULC is fixed by the view list and the batch size, both recorded. fp32, TF32 off. Before generating, A2's cache
check (eo/eval/seg.cache_check, gate Z8's fp32 rule, or Step 4.1's 1,000x rule) on the first 8 prompts, and its control
(every cached step one position off) must fail it.

Output, beside the map artifacts (eo/data/seg_dataset.py, CHAIN): <root>/pastis/tok/LULCGEN_<name>/
    tokens.npy    (n_views, 196) uint16, raw LULC codes (0..4,374), zero where not generated
    present.npy   (n_views,) bool
    slot_mass.npy (n_views,) float32: the mean unconstrained softmax mass on LULC's slot
    metadata.json written last (the done marker): weights and their SHA-256, slot, seed rule, precision, GPU, the cache
                  check, the splits, the code's SHA-256

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python eo/experiments/segmentation/step43_chain_gen.py \
        --expect-gpu "GTX TITAN X"
Never reads a test split (the dataset refuses it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

MAP_BASE = Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000")
CONFIG = "eo_seg/r0_s41_lora16"            # builds the 161,665-id model; LoRA is not applied by build_model
SEED = 20261123
TARGET = "LULC"
RATIO_RULE = 1000.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def weights_file(d: Path) -> Path:
    for n in ("model.safetensors", "pytorch_model.bin"):
        if (d / n).is_file():
            return d / n
    raise FileNotFoundError(f"no weights in {d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/enric/data/geobench2"))
    ap.add_argument("--base", type=Path, default=MAP_BASE)
    ap.add_argument("--name", default="base")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="the first N views of each split (a smoke; not a done artifact)")
    ap.add_argument("--expect-gpu", required=True)
    ap.add_argument("--out", type=Path, default=None, help="default <root>/pastis/tok/LULCGEN_<name>")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU")
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu not in gpu:
        raise SystemExit(f"CUDA sees {gpu!r}, expected {a.expect_gpu!r}")
    if "TITAN V" in gpu:
        raise SystemExit("refusing the TITAN V (stalls under load; CLAUDE.md)")

    from eo.data import seg_dataset as SD
    from eo.eval import seg as SG
    from eo.finetune.init import strict_load, verify_extension
    from eo.generate import conditional as G

    out = a.out or a.root / "pastis" / "tok" / f"LULCGEN_{a.name}"
    if (out / "metadata.json").exists():
        raise SystemExit(f"{out} is done already; refusing to overwrite")
    model, cfg = G.build_model(CONFIG, None, device="cpu")
    V = int(model.config.vocab_size)
    ext = verify_extension(a.base, V)
    load = strict_load(model, a.base)
    model.to("cuda").eval()
    wf = weights_file(a.base)

    splits = [s for s in a.splits.split(",") if s]
    if "test" in splits:
        raise SystemExit("never the test split")
    dsets = {s: SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", s, in_pinned=True if s == "train" else None,
                                         root=a.root)) for s in splits}
    import pandas as pd
    n_views = len(pd.read_parquet(a.root / "pastis" / "tok" / "views.parquet"))
    info = next(iter(dsets.values())).V.get_modality(TARGET)
    slot = (int(info.codebook_offset), int(info.codebook_size))
    if slot != (76800, 4375):
        raise SystemExit(f"LULC's slot is {slot}, not the plan's (76800, 4375)")

    def prompt(ds, view):
        row = int(ds.in_rows[view][0])
        return torch.cat([ds._s2_chunk(row), torch.tensor([info.bo_id], dtype=torch.long)])

    tokens = np.zeros((n_views, 196), np.uint16)
    present = np.zeros(n_views, bool)
    mass = np.zeros(n_views, np.float32)
    rec = {"splits": {}}
    t0 = time.time()
    with torch.no_grad():
        first = next(iter(dsets.values()))
        v0 = first.views.view.values[:8]
        ids = torch.stack([prompt(first, int(v)) for v in v0]).cuda()
        pos = torch.arange(ids.shape[1], device="cuda").expand(len(ids), -1)
        chk = SG.cache_check(model, ids, pos, slot, n_steps=4)
        ctl = SG.cache_check(model, ids, pos, slot, n_steps=4, cache_offset=1)
        ratio = ctl["max_abs_logit_diff"] / max(chk["max_abs_logit_diff"], 1e-30)
        ok = (chk["within"] or ratio >= RATIO_RULE) and chk["codes_in_slot"] and not ctl["within"]
        cache = {"check": chk, "control": ctl, "control_ratio": ratio, "by": "z8" if chk["within"] else "ratio_1000",
                 "pass": bool(ok)}
        print(f"cache check: {chk['max_ulp_diff']:.1f} fp32 steps (tol {chk['tol']:.2e}); control "
              f"{ctl['max_abs_logit_diff']:.3f}; ratio {ratio:.3g}; {'pass' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise SystemExit("cache check failed")
        for s, ds in dsets.items():
            vs = ds.views.view.values.astype(np.int64)
            if a.limit:
                vs = vs[: a.limit]
            ts = time.time()
            for b in range(0, len(vs), a.batch):
                chunk = vs[b:b + a.batch]
                ids = torch.stack([prompt(ds, int(v)) for v in chunk]).cuda()
                if ids.shape[1] != 199:
                    raise SystemExit(f"prompt of {ids.shape[1]} tokens, not 199")
                pos = torch.arange(ids.shape[1], device="cuda").expand(len(chunk), -1)
                codes, m = SG.generate(model, ids, pos, slot, 196, greedy=False, temperature=1.0,
                                       seed=SEED + int(chunk[0]))
                c = codes.cpu().numpy()
                if c.min() < 0 or c.max() >= slot[1]:
                    raise SystemExit("a code outside LULC's slot")
                tokens[chunk] = c.astype(np.uint16)
                present[chunk] = True
                mass[chunk] = m.float().cpu().numpy()
                done = b + len(chunk)
                if (b // a.batch) % 10 == 0 or done == len(vs):
                    el = time.time() - ts
                    print(f"{s}: {done}/{len(vs)} views, {el:.0f} s, slot mass {mass[vs[:done]].mean():.4f}", flush=True)
            rec["splits"][s] = {"views": int(len(vs)), "first_view": int(vs[0]), "last_view": int(vs[-1]),
                                "in_pinned": True if s == "train" else None, "seconds": round(time.time() - ts, 1),
                                "slot_mass_mean": float(mass[vs].mean())}

    out.mkdir(parents=True, exist_ok=True)
    for n, arr in (("tokens", tokens), ("present", present), ("slot_mass", mass)):
        tmp = out / f"{n}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out / f"{n}.npy")
    meta = {"what": "A2 Step 4.3's chain: TerraMind LULC tokens the map base wrote from each view's S2 block",
            "plan": "docs/plans/a2_segmentation.md, Step 4, Steps 4.3 and 4.4's design and reading, item 4",
            "task": "pastis", "frames": "p1", "n_views": int(n_views), "n_present": int(present.sum()),
            "tokens_per_sample": 196, "target": TARGET, "slot": list(slot),
            "base": str(a.base), "weights_file": wf.name, "weights_sha256": sha256(wf),
            "base_extension_weights_sha256": ext.get("weights_sha256"), "load": load, "config": CONFIG,
            "prompt": "[BO_S2L2A s EO_S2L2A][BO_LULC] at positions 0..198", "sampling": "multinomial over the slot, T 1.0",
            "seed_rule": f"batch b's generator seeded {SEED} + its first view's index; batches of {a.batch} views "
                         f"in each split's view order", "batch": a.batch, "limit": a.limit,
            "precision": "fp32", "tf32": False, "gpu": gpu, "cache_check": cache, **rec,
            "seconds": round(time.time() - t0, 1),
            "tokens_sha256": sha256(out / "tokens.npy"), "present_sha256": sha256(out / "present.npy"),
            "code_sha256": sha256(Path(__file__)), "seg_dataset_sha256": sha256(REPO / "eo/data/seg_dataset.py")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(f"wrote {out}: {int(present.sum())} views", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
