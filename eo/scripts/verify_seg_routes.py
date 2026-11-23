#!/usr/bin/env python
"""A2 Step 4.2's gate: Routes B and C (docs/plans/a2_segmentation.md, Step 4.2, item 7; 2026-11-23). Every check ships
with a control that must fail it. A file of its own, so Step 4.2 adds no lines to the gates other A2 work edits.

    .venv:  python eo/scripts/verify_seg_routes.py [--only Z19,Z20,Z21c,Z22,Z23] [--out results.json]     (CPU)
    a2seg:  python eo/scripts/verify_seg_routes.py --only Z20s,Z20d,Z21t [--out ...]   (Z20d decodes on one GPU)

  Z19  The cell map by two independent paths (eo/maps/cells.py: torch render-and-pool, numpy geometry) on every val
       view of both tasks: the cells agree cell for cell, the back-projection pixel for pixel, and the ceilings (path 1
       through eo/eval/seg_routes.score_cells, stitched and scored by the harness; path 2 stitched in numpy) agree
                                     | a transposed grid: its cells disagree and its ceiling falls
  Z20  The cell artifacts (MAP_cells_<task>): every train and val view's stored cells equal the rule recomputed from its
       label crop and dihedral variant, a random 2% (all of val) by the numpy path too; test rows absent and zero
                                     | the next sample's labels; the k = 0 variant for k != 0
  Z20s (a2seg) eo/maps/sets' labels equal what Step 2 tokenized (eo/geobench/sources), 24 samples per task and split
                                     | the next sample's labels
  Z20d (a2seg, GPU) the labels' view geometry is Step 2's: the stored MAP_aligned codes of 40 training views per task,
       decoded by the frozen aligned tokenizer, match the view's own dihedral variant of the label crop best of the 8
                                     | the next variant
  Z21t (a2seg) Route C's head equals terratorch 1.2.11's modules (LearnedInterpolateToPyramidal, UNetDecoder,
       SegmentationHead) with the same weights: parameter count, and the output on random features
                                     | the features fed in reverse depth order
  Z21c Route C reads the second copy: on the map base, changing the first copy's LAST image token moves the state of the
       second copy's FIRST image token (it sees the whole image), and the head's features are exactly hidden_states[i]
       at positions 199..394
                                     | the first copy's first token (a slice 198 off) does not move
  Z22  Overlays: s42_b_* differ from their parents in exactly seg.arm, seg.map_order and max_length; s42_c_* in exactly
       seg.route and the seg_head block. Built from the map base, B trains 51,014,208 parameters (R0's), C the adapters
       and the head with the map rows and every base row frozen; the head's optimizer group runs at seg_head.lr
                                     | a stray key; the head in the adapters' group
  Z23  Route C through pretrain.py, 2 steps on CPU at fp32 (weight decay 0.1): the frozen tensors stay bit-identical to
       the map base, the map rows too, every adapter and head tensor has optimizer state, the optimizer holds the two
       LRs, eval_loss and the views' mIoU are finite; the plateau stop's decay replaces both groups' lambdas
                                     | masked rows under AdamW with decay move
  Z23b Route B through pretrain.py, 2 steps on CPU: the cells train (finite loss, the map rows move, a save)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS: list = []
MAP_BASE = Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000")
TASKS = ("pastis", "cloudsen12")
FRAMES = {"pastis": "p1", "cloudsen12": "single"}


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def _compose(name: str, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def _views_labels(task: str, split: str, arm: str = None, nested: bool = True):
    """(dataset, views, label views) of a task's split, the labels cut as the artifact cut them."""
    from eo.data import map_vocab as MV, seg_dataset as SD
    from eo.maps import cells as CE, sets
    arm = arm or MV.CELL_ARMS[task]
    ds = SD.SegDataset(SD.SegSpec(task, arm, FRAMES[task], split, map_order="nested" if nested else "raster"))
    v = ds.views
    labels, ids = sets.load(task, split)
    rows = CE.set_rows(task, ids, v.sample_id.tolist())
    return ds, v, labels, rows


def ceiling_numpy(task: str, cells: np.ndarray, views, labels_val: np.ndarray, rows: np.ndarray) -> float:
    """Path 2's ceiling: per crop the class shares of each native pixel from the geometry (cells._weights), CloudSen12's
    crops averaged where they overlap by explicit offsets (0, 144, 288 in 512), argmax without the ignored class, the
    macro Jaccard from summed per-sample matrices."""
    from eo.eval.seg_routes import miou_of, per_sample_cm
    from eo.maps import cells as CE, palette as P
    t = P.TASKS[task]
    n = t.n_classes
    if task == "pastis":
        W = CE._weights(t.native).astype(np.float64)                          # (28, 128)
        tot = W.sum(0)                                                        # rendered pixels per native row
        preds = []
        for c in cells:
            s = np.stack([(W.T @ (c == k) @ W) / np.outer(tot, tot) for k in range(n)])
            s[t.ignore] = -1
            preds.append(s.argmax(0))
        gt = np.stack([labels_val[r] for r in rows])
        return miou_of(per_sample_cm(np.stack(preds), gt, n, t.ignore).sum(0), t.ignore)
    W = CE._weights(224).astype(np.float64)                                   # identity geometry: 8 x 8 blocks
    offs = [0, 144, 288]
    samples = list(dict.fromkeys(views.sample_id.tolist()))
    preds, gts = [], []
    for si, sid in enumerate(samples):
        idx = np.nonzero(views.sample_id.values == sid)[0]
        acc = np.zeros((n, 512, 512))
        cnt = np.zeros((512, 512))
        for j, i in enumerate(idx):
            y, x = offs[j // 3], offs[j % 3]
            if (int(views.y0.values[i]), int(views.x0.values[i])) != (y, x):
                raise RuntimeError(f"tile {sid}: crop {j} at {views.y0.values[i]}, {views.x0.values[i]}, not {y}, {x}")
            sh = np.stack([W.T @ (cells[i] == k) @ W for k in range(n)]) / 1.0
            acc[:, y:y + 224, x:x + 224] += sh
            cnt[y:y + 224, x:x + 224] += 1
        preds.append((acc / cnt).argmax(0))
        gts.append(labels_val[rows[idx[0]]])
    return miou_of(per_sample_cm(np.stack(preds), np.stack(gts), n, None).sum(0), None)


# ---------------------------------------------------------------- Z19
def z19() -> None:
    import torch
    from eo.eval import seg_routes as R
    from eo.finetune.seg_head import crop_native
    from eo.maps import cells as CE, palette as P
    for task in TASKS:
        t = P.TASKS[task]
        ds, v, labels, rows = _views_labels(task, "val")
        lab = CE.view_labels(labels, rows, v.y0.values, v.x0.values, v["size"].values, v.k.values)
        t0 = time.time()
        c1 = CE.cells_torch(torch.from_numpy(lab), t.n_classes, t.ignore).numpy()
        c2 = CE.cells_numpy(lab, t.n_classes, t.ignore)
        nat = crop_native(task)
        p1 = CE.predict_torch(torch.from_numpy(c1), t.n_classes, t.ignore, nat).numpy()
        p2 = CE.predict_numpy(c2, t.n_classes, t.ignore, nat)
        rec, _, _ = R.score_cells(task, c1.reshape(len(c1), -1), v)
        ceil1 = rec["result"]["mIoU"]
        ceil2 = ceiling_numpy(task, c2, v, labels, rows)
        ok = np.array_equal(c1, c2) and np.array_equal(p1, p2) and abs(ceil1 - ceil2) < 1e-12
        record(f"Z19 {task} cells by two paths", ok,
               f"{len(v)} val views: cells equal {np.array_equal(c1, c2)}, back-projection equal {np.array_equal(p1, p2)};"
               f" ceiling {ceil1:.6f} (harness path) vs {ceil2:.6f} (numpy) ({time.time() - t0:.0f} s)")
        cT = np.swapaxes(c1, 1, 2)
        recT, _, _ = R.score_cells(task, cT.reshape(len(cT), -1), v)
        diffT = float((cT != c2).mean())
        record(f"Z19 {task} control: a transposed grid", diffT > 0.05 and recT["result"]["mIoU"] < ceil1 - 0.3,
               f"{100 * diffT:.1f}% of cells differ; ceiling {recT['result']['mIoU']:.4f} against {ceil1:.4f}")


# ---------------------------------------------------------------- Z20
def z20() -> None:
    import torch
    from eo.maps import cells as CE, palette as P
    rng = np.random.default_rng(0)
    for task in TASKS:
        t = P.TASKS[task]
        t0 = time.time()
        bad, n_views, n_np, np_bad, ctl_next, ctl_k0, n_ctl = 0, 0, 0, 0, [], [], 0
        for split in ("train", "val"):
            ds, v, labels, rows = _views_labels(task, split)
            stored = np.asarray(ds.maps[v.view.values]).reshape(len(v), CE.GRID, CE.GRID)
            for s in range(0, len(v), 4096):
                sl = slice(s, s + 4096)
                lab = CE.view_labels(labels, rows[sl], v.y0.values[sl], v.x0.values[sl], v["size"].values[sl],
                                     v.k.values[sl])
                c = CE.cells_torch(torch.from_numpy(lab), t.n_classes, t.ignore).numpy()
                bad += int((c != stored[sl]).any((1, 2)).sum())
                n_views += len(c)
                pick = np.nonzero(rng.random(len(c)) < (1.0 if split == "val" and task == "pastis" else 0.02))[0]
                if len(pick):
                    c2 = CE.cells_numpy(lab[pick], t.n_classes, t.ignore)
                    np_bad += int((c2 != stored[sl][pick]).any((1, 2)).sum())
                    n_np += len(pick)
            # controls on 200 training views with k != 0: the next sample's labels, and the k = 0 variant
            if split == "train":
                cand = np.nonzero(v.k.values != 0)[0]
                pick = rng.choice(cand, 200, replace=False)
                for i in pick:
                    nxt = (rows[i] + 1) % len(labels)
                    a = CE.view_labels(labels, [nxt], [v.y0.values[i]], [v.x0.values[i]], [v["size"].values[i]],
                                       [v.k.values[i]])
                    b = CE.view_labels(labels, [rows[i]], [v.y0.values[i]], [v.x0.values[i]], [v["size"].values[i]], [0])
                    ca = CE.cells_numpy(a, t.n_classes, t.ignore)[0]
                    cb = CE.cells_numpy(b, t.n_classes, t.ignore)[0]
                    ctl_next.append(np.array_equal(ca, stored[i]))
                    ctl_k0.append(np.array_equal(cb, stored[i]))
                    n_ctl += 1
        import pandas as pd
        allv = pd.read_parquet(Path(ds.spec.root) / task / "tok" / "views.parquet")
        pres = np.load(Path(ds.spec.root) / task / "tok" / f"MAP_{ds.spec.arm}" / "present.npy")
        tst = allv.view.values[allv.split.values == "test"]
        tok_all = np.load(Path(ds.spec.root) / task / "tok" / f"MAP_{ds.spec.arm}" / "tokens.npy", mmap_mode="r")
        test_ok = (not pres[tst].any()) and pres[allv.view.values[allv.split.values != "test"]].all() and \
            not np.asarray(tok_all[tst]).any()
        record(f"Z20 {task} artifact = the rule", bad == 0 and np_bad == 0 and n_np > 0 and test_ok,
               f"{n_views} train+val views recomputed, {bad} differ; {n_np} by the numpy path, {np_bad} differ; test "
               f"views absent and zero {test_ok} ({time.time() - t0:.0f} s)")
        record(f"Z20 {task} controls: the next sample; the k = 0 variant",
               np.mean(ctl_next) < 0.05 and np.mean(ctl_k0) < 0.5,
               f"of {n_ctl} training views with k != 0, the next sample's cells equal the stored ones for "
               f"{100 * np.mean(ctl_next):.1f}%, the k = 0 variant's for {100 * np.mean(ctl_k0):.1f}% (symmetric maps)")


def z20s() -> None:
    """(a2seg) eo/maps/sets = Step 2's sources."""
    from eo.geobench import sources as S
    from eo.maps import sets
    rng = np.random.default_rng(1)
    for task in TASKS:
        for split in ("train", "val"):
            labels, ids = sets.load(task, split)
            pick = rng.choice(len(ids), 24, replace=False)
            same, nxt = [], []
            for i in pick:
                src = _source_label(S, task, ids[i])
                same.append(np.array_equal(np.asarray(labels[i]), src))
                # the control's sample: the next one whose map differs from sample i's (CloudSen12 has single-class
                # tiles, identical across samples, which no equality check could tell apart: 2026-11-23)
                j = (i + 1) % len(ids)
                while np.array_equal(np.asarray(labels[j]), np.asarray(labels[i])):
                    j = (j + 1) % len(ids)
                nxt.append(np.array_equal(np.asarray(labels[j]), src))
            record(f"Z20s {task} {split} sets = Step 2's labels", all(same) and not any(nxt),
                   f"{sum(same)}/24 equal; control: the next sample with another map equal for {sum(nxt)}/24")


_IDX = {}


def _source_label(S, task, sid):
    """The label Step 2 tokenized (eo/scripts/tokenize_maps.view_labels' reader), by the sample's index in views.parquet."""
    import pandas as pd
    if task not in _IDX:
        v = pd.read_parquet(Path("/data/enric/data/geobench2") / task / "tok" / "views.parquet")
        key = (lambda x: str(x)) if task == "cloudsen12" else (lambda x: int(x))
        _IDX[task] = {key(a): int(b) for a, b in zip(v.sample_id.values, v.sample_idx.values)}
    idx = _IDX[task][str(sid) if task == "cloudsen12" else int(sid)]
    if task == "pastis":
        return np.asarray(S.pastis_sample(idx, ("semantic",))["semantic"])
    return np.asarray(S.cloudsen12_sample(idx)[1])


def z20d() -> None:
    """(a2seg, GPU) the label views' geometry is Step 2's: decoded MAP_aligned codes best match the own variant."""
    import torch
    from eo.data import seg_dataset as SD
    from eo.geobench.s2 import dihedral
    from eo.maps import decode as D, frozen, palette as P, sets
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = frozen.load("aligned", dev)
    rng = np.random.default_rng(2)
    for task in TASKS:
        t = P.TASKS[task]
        ds = SD.SegDataset(SD.SegSpec(task, "aligned", FRAMES[task], "train"))
        v = ds.views
        labels, ids = sets.load(task, "train")
        from eo.maps import cells as CE
        rows = CE.set_rows(task, ids, v.sample_id.tolist())
        # views whose crop the dihedral group can tell apart: every pair of the 8 variants differs on >= 5% of its
        # pixels (single-class or near-symmetric crops tie, and say nothing about the geometry: 14 of 40 random
        # CloudSen12 crops are one class, 2026-11-23)
        cand = rng.permutation(np.nonzero(v.k.values != 0)[0])
        pick = []
        for i in cand:
            nat0 = int(v["size"].values[i])
            crop0 = np.asarray(labels[rows[i]])[v.y0.values[i]:v.y0.values[i] + nat0, v.x0.values[i]:v.x0.values[i] + nat0]
            var = [dihedral(crop0, k) for k in range(8)]
            if min(float((var[a] != var[b]).mean()) for a in range(8) for b in range(a + 1, 8)) >= 0.05:
                pick.append(i)
            if len(pick) == 40:
                break
        pick = np.asarray(pick)
        codes = torch.as_tensor(np.asarray(ds.maps[v.view.values[pick]]).astype(np.int64), device=dev).view(-1, 14, 14)
        # decode each view's codes as ONE crop at the crop's native size (PASTIS 128; CloudSen12 224)
        from eo.maps.tokenizers import run_fixed
        out = run_fixed(tok.decode, codes, tok.batch)
        sc = D.logit_scores(out, t.assign) if tok.output == "logits" else D.rgb_scores(out, np.asarray(P.UNIVERSAL)[list(t.assign)])
        nat = int(v["size"].values[pick[0]])
        pred = D.apply_rule(D.to_native(sc, nat), t.ignore, {"rule": "nearest"}, tok.output).cpu().numpy()
        best_own, ctl = 0, 0
        for j, i in enumerate(pick):
            crop = np.asarray(labels[rows[i]])[v.y0.values[i]:v.y0.values[i] + nat, v.x0.values[i]:v.x0.values[i] + nat]
            acc = []
            for k in range(8):
                g = dihedral(crop, k)
                m = g != t.ignore if t.ignore is not None else np.ones_like(g, bool)
                acc.append(float((pred[j][m] == g[m]).mean()))
            own = int(v.k.values[i])
            best_own += int(np.argmax(acc) == own)
            ctl += int(np.argmax(acc) == (own + 1) % 8)
        record(f"Z20d {task} view geometry = Step 2's", best_own == len(pick) == 40 and ctl == 0,
               f"the own variant is the best of 8 for {best_own}/{len(pick)} decoded training views (crops whose 8 "
               f"variants differ pairwise on >= 5% of pixels); control: the next variant is best for {ctl}/{len(pick)}")


# ---------------------------------------------------------------- Z21
def z21t() -> None:
    """(a2seg) our head = terratorch's modules with the same weights."""
    import torch
    from terratorch.models.decoders.unet_decoder import UNetDecoder as TUNet
    from terratorch.models.heads.segmentation_head import SegmentationHead as THead
    from terratorch.models.necks import LearnedInterpolateToPyramidal as TNeck
    from eo.finetune.seg_head import UNetHead
    torch.manual_seed(0)
    for n_cls in (20, 4):
        ours = UNetHead(576, n_cls).eval()
        neck = TNeck([576] * 4)
        dec = TUNet(neck.embedding_dim, [512, 256, 128, 64])
        head = THead(dec.out_channels, n_cls, dropout=0.1)
        theirs = torch.nn.ModuleDict({"neck": neck, "decoder": dec, "head": head}).eval()
        po = [p for p in ours.state_dict().values()]
        pt = [p for p in theirs.state_dict().values()]
        shapes_ok = len(po) == len(pt) and all(a.shape == b.shape for a, b in zip(po, pt))
        n_o = sum(p.numel() for p in ours.parameters())
        n_t = sum(p.numel() for p in theirs.parameters())
        with torch.no_grad():
            for a, b in zip(ours.state_dict().values(), theirs.state_dict().values()):
                b.copy_(a)
        x = [torch.randn(2, 196, 576) * s for s in (3.1, 4.4, 8.0, 2.5)]
        with torch.no_grad():
            y = ours(x)
            imgs = [ours.reshape(f) for f in x]
            m = theirs["head"](theirs["decoder"](theirs["neck"](imgs)))
            m = torch.nn.functional.interpolate(m, size=(224, 224), mode="bilinear")
            yr = ours(x[::-1])
        d = float((y - m).abs().max())
        dr = float((yr - m).abs().max())
        record(f"Z21t head = terratorch's ({n_cls} classes)", shapes_ok and n_o == n_t and d < 1e-5,
               f"{len(po)} tensors, shapes equal {shapes_ok}; parameters {n_o:,} vs {n_t:,}; max |output difference| "
               f"{d:.2e} on (2, {n_cls}, 224, 224)")
        record(f"Z21t control ({n_cls} classes): features in reverse depth order", dr > 1e-2,
               f"max |output difference| {dr:.2e}")


def z21c() -> None:
    import torch
    from eo.data import seg_dataset as SD
    from eo.data.seg_head_dataset import SegHeadDataset
    from eo.finetune.seg_head import build_head_model
    torch.manual_seed(0)
    m, cfg = build_head_model("eo_seg/s42_c_pastis", None, device="cpu",
                              overrides=[f"finetune.init_from={MAP_BASE}", "seg_head.lr=1e-3"])
    # build_head_model with no checkpoint loads the map base under an untrained head
    ds = SegHeadDataset(SD.SegSpec("pastis", "aligned", "p1", "val"))
    it = ds.head_item(int(ds.views.view.values[0]))
    ids = it["input_ids"][None].clone()
    pos = it["position_ids"][None]
    info = ds.V.get_modality("S2L2A")
    ids2 = ids.clone()
    ids2[0, 196] = info.codebook_offset + (int(ids[0, 196]) - info.codebook_offset + 1) % info.codebook_size
    with torch.no_grad():
        o1 = m.lm.model(input_ids=ids, position_ids=pos, output_hidden_states=True, use_cache=False).hidden_states
        o2 = m.lm.model(input_ids=ids2, position_ids=pos, output_hidden_states=True, use_cache=False).hidden_states
        f = m.features(ids, None, pos)
    exact = all(torch.equal(f[j], o1[i][:, 199:395]) for j, i in enumerate(m.layers))
    sees = [float((o1[i][0, 199] - o2[i][0, 199]).abs().max()) for i in m.layers]
    first = [float((o1[i][0, 1] - o2[i][0, 1]).abs().max()) for i in m.layers]
    record("Z21c the head reads the second copy, which sees the whole image",
           exact and all(s > 1e-4 for s in sees) and int(ids[0, 198]) == int(ids[0, 0]),
           f"features = hidden_states{m.layers} at 199..394: {exact}; changing the first copy's last image token "
           f"(position 196) moves the second copy's first image token by {[f'{s:.2e}' for s in sees]}")
    record("Z21c control: the first copy's first token does not move", all(s == 0.0 for s in first),
           f"position 1 moves by {first} (causal: a head on the first copy would not see the image's end)")


# ---------------------------------------------------------------- Z22
def groups_ok(groups, head_ids: set, lr: float, head_lr: float) -> bool:
    """Exactly the head's parameters at head_lr, every other trainable one at lr."""
    seen_head = {id(p) for g in groups if abs(g["lr"] - head_lr) < 1e-15 for p in g["params"]}
    seen_rest = {id(p) for g in groups if abs(g["lr"] - lr) < 1e-15 for p in g["params"]}
    return seen_head == head_ids and not (seen_rest & head_ids) and abs(lr - head_lr) > 0


def z22() -> None:
    import torch
    from omegaconf import OmegaConf
    from eo.finetune.init import init_from
    from eo.finetune.seg_head import wrap_model
    from eo.generate.conditional import build_model
    pairs = {"s42_b_pastis": ("r0_s41_lora16", {"seg.arm", "seg.map_order", "max_length"}),
             "s42_b_cloudsen12": ("s52_cloudsen12", {"seg.arm", "seg.map_order", "max_length"}),
             "s42_c_pastis": ("r0_s41_lora16", None), "s42_c_cloudsen12": ("s52_cloudsen12", None)}
    head_keys = {"seg.route", "seg_head.enable", "seg_head.lr", "seg_head.layers", "seg_head.copy_start",
                 "seg_head.channels", "seg_head.dropout"}
    for name, (parent, want) in pairs.items():
        want = want or head_keys
        a = _flat(OmegaConf.to_container(_compose(f"eo_seg/{parent}"), resolve=False))
        b = _flat(OmegaConf.to_container(_compose(f"eo_seg/{name}"), resolve=False))
        d = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        record(f"Z22 {name}", d == want, f"differs from {parent} in {sorted(d)}")
    a = _flat(OmegaConf.to_container(_compose("eo_seg/r0_s41_lora16"), resolve=False))
    b = _flat(OmegaConf.to_container(_compose("eo_seg/s42_c_pastis", ["weight_decay=0.1"]), resolve=False))
    d = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    record("Z22 control: a stray key", d != head_keys and "weight_decay" in d,
           f"with weight_decay=0.1 it differs in {sorted(d)}")
    # trainable parameters
    m, cfg = build_model("eo_seg/s42_b_pastis", None, device="cpu", overrides=[f"finetune.init_from={MAP_BASE}"])
    rep = init_from(m, cfg)
    record("Z22 Route B trains R0's parameters", rep["trainable_params"] == 51_014_208,
           f"trainable {rep['trainable_params']:,} (R0: 51,014,208)")
    del m
    for task in TASKS:
        m, cfg = build_model(f"eo_seg/s42_c_{task}", None, device="cpu",
                             overrides=[f"finetune.init_from={MAP_BASE}", "seg_head.lr=1e-3"])
        rep = init_from(m, cfg)
        w, hrep = wrap_model(m, cfg)
        emb = m.model.embed_tokens
        head_ids = {id(p) for p in w.head.parameters()}
        tr = [p for p in w.parameters() if p.requires_grad]
        adapters = rep["lora"]["adapter_params"]
        ok = (not emb.old.requires_grad and not emb.new.requires_grad and hrep["lm_trainable_params"] == adapters
              and hrep["trainable_params"] == adapters + hrep["head_params"] == sum(p.numel() for p in tr))
        record(f"Z22 Route C {task} trains the adapters and the head", ok,
               f"trainable {hrep['trainable_params']:,} = adapters {adapters:,} + head {hrep['head_params']:,}; map rows "
               f"frozen {not emb.new.requires_grad}, base rows frozen {not emb.old.requires_grad}")
        # the optimizer's groups, as SegHeadTrainer builds them (without a Trainer: the same partition)
        lr, hlr = 3e-3, 1e-3
        groups = [{"params": [p for p in tr if id(p) not in head_ids], "lr": lr},
                  {"params": [p for p in tr if id(p) in head_ids], "lr": hlr}]
        ctl = [{"params": tr, "lr": lr}]
        record(f"Z22 Route C {task} head group at seg_head.lr", groups_ok(groups, head_ids, lr, hlr)
               and not groups_ok(ctl, head_ids, lr, hlr), "two groups pass; control: one group at the adapters' LR "
               "is flagged")
        del m, w


# ---------------------------------------------------------------- Z23
def _run_pretrain(config: str, extra: list, tag: str) -> tuple:
    import subprocess
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {save!r} is not under /data")
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    ov = {"name": out.name, "output_dir": rel, "wandb_run_name": out.name, "wandb": "false",
          "finetune.init_from": str(MAP_BASE), "num_train_steps": "2", "stop_steps": "2", "num_warmup_steps": "0",
          "lr_scheduler_kwargs.num_decay_steps": "0", "save_steps": "2", "eval_on_start": "false", "eval_steps": "2",
          "total_batch_size": "2", "per_device_train_batch_size": "2", "per_device_eval_batch_size": "8",
          "logging_steps": "1", "dataloader_num_workers": "0", "precision": "fp32", "plateau_stop.enable": "false",
          "hydra.run.dir": f"{out}/hydra"}
    ov.update(dict(e.split("=", 1) for e in extra))
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name", config] + \
        [f"{k}={v}" for k, v in ov.items()]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pretrain.log").write_text(r.stdout + r.stderr)
    return r, out, time.time() - t0


def z23() -> None:
    import torch
    from eo.finetune.init import _load_state, _weights
    from eo.data import map_vocab as MV
    r, out, dt = _run_pretrain("eo_seg/s42_c_pastis", ["learning_rate=3e-3", "seg_head.lr=1e-3", "weight_decay=0.1"], "z23c")
    ck = out / "checkpoint-2"
    if r.returncode != 0 or not ck.is_dir():
        record("Z23 Route C through pretrain.py", False, f"exit {r.returncode}; log {out / 'pretrain.log'}: "
               + " | ".join((r.stdout + r.stderr).strip().splitlines()[-4:]))
        return
    after = _load_state(_weights(ck))
    before = _load_state(_weights(MAP_BASE))
    nb = MV.LAYOUT.base_vocab
    emb_b = before["model.embed_tokens.weight"]
    moved, checked = [], 0
    for k, v in after.items():
        if not k.startswith("lm.") or "lora_" in k:
            continue
        kk = k[len("lm."):]
        ref = emb_b[:nb] if kk.endswith("embed_tokens.old") else emb_b[nb:] if kk.endswith("embed_tokens.new") \
            else before.get(kk.replace(".base.weight", ".weight"))
        if ref is None:
            moved.append(f"{k}: no base tensor")
            continue
        checked += 1
        if not torch.equal(v, ref):
            moved.append(k)
    opt = torch.load(ck / "optimizer.pt", map_location="cpu", weights_only=False)
    # each group's base LR (LambdaLR's initial_lr; "lr" is the schedule's, 0 at the end of this 2-step run)
    lrs = {g.get("group"): round(float(g["initial_lr"]), 12) for g in opt["param_groups"]}
    n_state = sum(1 for s in opt["state"].values() if float(s["exp_avg"].abs().sum()) > 0)
    n_params = sum(len(g["params"]) for g in opt["param_groups"])
    st = json.loads((out / "trainer_state.json").read_text())
    ev = [h for h in st["log_history"] if "eval_loss" in h]
    mi = [h for h in st["log_history"] if "eval_miou_views" in h]
    rep = json.loads((out / "finetune_init.json").read_text())
    import math
    ok = (not moved and checked > 100 and lrs == {"adapters": 3e-3, "head": 1e-3} and n_state == n_params and ev
          and math.isfinite(ev[-1]["eval_loss"]) and mi and math.isfinite(mi[-1]["eval_miou_views"])
          and rep["seg_head"]["trainable_params"] == rep["seg_head"]["lm_trainable_params"] + rep["seg_head"]["head_params"])
    record("Z23 Route C through pretrain.py", ok,
           f"{checked} frozen tensors (map rows included) bit-identical to the map base (moved: {moved[:3]}); the "
           f"optimizer's base LRs {lrs} (learning_rate 3e-3, seg_head.lr 1e-3); {n_state}/{n_params} trainable tensors with state; eval_loss {ev[-1]['eval_loss'] if ev else None}, "
           f"views mIoU {mi[-1]['eval_miou_views'] if mi else None}; trainable {rep['seg_head']['trainable_params']:,}; "
           f"{dt:.0f} s, CPU fp32, weight decay 0.1")
    # the plateau stop with two groups
    from transformers.optimization import get_wsd_schedule
    from types import SimpleNamespace
    from eo.train.plateau_stop import PlateauStopCallback
    w = [torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))]
    o = torch.optim.AdamW([{"params": [w[0]], "lr": 3e-3}, {"params": [w[1]], "lr": 1e-3}])
    sch = get_wsd_schedule(o, num_warmup_steps=2, num_decay_steps=20, num_training_steps=100)
    cb = PlateauStopCallback(None, patience=1, rel=1.0, verbose=False)
    args = SimpleNamespace(eval_strategy="steps", eval_steps=4, save_steps=4, lr_scheduler_type="warmup_stable_decay",
                           lr_scheduler_kwargs={"num_decay_steps": 20}, output_dir=None)
    state = SimpleNamespace(global_step=0, max_steps=100)
    cb.on_train_begin(args, state, None)
    for step in range(1, 13):
        o.step()
        sch.step()
        state.global_step = step
        if step % 4 == 0 and step <= 8:
            cb.on_evaluate(args, state, None, metrics={"eval_loss": 1.0}, lr_scheduler=sch)
    lr_now = [g["lr"] for g in o.param_groups]
    record("Z23 the plateau stop decays both groups", cb.fired is not None and len(sch.lr_lambdas) == 2
           and abs(lr_now[0] / 3e-3 - lr_now[1] / 1e-3) < 1e-12 and lr_now[0] < 3e-3,
           f"fired at epoch {cb.fired and cb.fired['epoch']}; {len(sch.lr_lambdas)} lambdas; LRs after it {lr_now}")
    # control: rows in a trainable Parameter, gradient masked, AdamW with decay -> they move
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(128, 8))
    p0 = p.detach().clone()
    opt2 = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.1)
    for _ in range(2):
        opt2.zero_grad()
        (p ** 2).sum().backward()
        p.grad[:64] = 0
        opt2.step()
    record("Z23 control: masked rows under AdamW", not torch.equal(p.detach()[:64], p0[:64]),
           f"masked rows moved by up to {float((p.detach()[:64] - p0[:64]).abs().max()):.2e}")


def z23b() -> None:
    import math
    import torch
    from eo.finetune.init import _load_state, _weights
    from eo.data import map_vocab as MV
    r, out, dt = _run_pretrain("eo_seg/s42_b_pastis", ["learning_rate=3e-3", "eval_steps=1000"], "z23b")
    ck = out / "checkpoint-2"
    if r.returncode != 0 or not ck.is_dir():
        record("Z23b Route B through pretrain.py", False, f"exit {r.returncode}; log {out / 'pretrain.log'}: "
               + " | ".join((r.stdout + r.stderr).strip().splitlines()[-4:]))
        return
    after = _load_state(_weights(ck))
    before = _load_state(_weights(MAP_BASE))
    nb = MV.LAYOUT.base_vocab
    off, size = MV.LAYOUT.slot("cells_pastis")
    new_a = after["model.embed_tokens.new"]
    new_b = before["model.embed_tokens.weight"][nb:]
    cell_rows = slice(off - nb, off - nb + size)
    moved_cells = float((new_a[cell_rows] - new_b[cell_rows]).abs().max())
    st = json.loads((out / "trainer_state.json").read_text())
    losses = [h["loss"] for h in st["log_history"] if "loss" in h]
    ok = moved_cells > 0 and losses and all(math.isfinite(x) for x in losses)
    record("Z23b Route B through pretrain.py", ok, f"losses {losses}; the 20 cell rows moved by up to {moved_cells:.2e}; "
           f"{dt:.0f} s, CPU fp32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = {"Z19": z19, "Z20": z20, "Z20s": z20s, "Z20d": z20d, "Z21t": z21t, "Z21c": z21c, "Z22": z22, "Z23": z23,
              "Z23b": z23b}
    default = ("Z19", "Z20", "Z21c", "Z22", "Z23", "Z23b")
    want = a.only.split(",") if a.only else list(default)
    t0 = time.time()
    for k in want:
        try:
            checks[k]()
        except Exception as e:
            traceback.print_exc()
            record(k, False, f"raised {type(e).__name__}: {e}")
    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {n_fail} fail ({time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in RESULTS], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
