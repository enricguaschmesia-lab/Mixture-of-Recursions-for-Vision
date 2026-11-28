#!/usr/bin/env python
"""Why the 784-token map beats the 196-token one on PASTIS-7 (Enric's question, 2026-11-27; A2 plan, Step 7, "Why 784
beats 196"). Descriptive, nothing pre-registered; `.venv`, CPU. Fold 4 (val) and the training patches only.

    python eo/experiments/segmentation/step70_why784.py [--out <dir>]

  geometry  per class, the share of its canvas pixels inside a pure token (a token whose whole block is one label) for
            aligned (14 x 14, 16-px blocks) and c4375_224 (28 x 28, 8-px blocks), on the 1,200 pinned training patches;
  codes     per arm: distinct codes over the 9,600 pinned training views (all 8 dihedral variants), occurrences per
            used code, the share of fold-4 tokens whose code no training view uses; per class: distinct codes among its
            pure training tokens (k = 0), the most common code's share, and the share of its pure fold-4 tokens whose
            code never encodes that class's pure tokens in training;
  depth     B1 K8 and E784 K8 (plain votes) scored on the pixels whose (2r+1)^2 neighbourhood holds one label (deep:
            no boundary within r native pixels, r = 2 and 4) and on the rest: whether the gain is in where the
            boundaries go or in which class is written;
  shares    each run's predicted class shares against fold 4's, and the stack's and B1 TTA-64's (frequency-corrected,
            cross-fitted, as 7.0f read them).
  ladder    (Enric's follow-up, 2026-11-27) the 196 map with the other generation-side changes and without the 784
            map: plain or frequency-corrected vote (alpha cross-fitted, 7.0e) over 8 samples or the 64 rotation votes,
            for B1 and E784-s42; 784 - 196 at each level and each change's gain on 196, by 7.0c's paired bootstrap.
A block already in <out>/why784.json is reused, not recomputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step70e_read as RD                           # noqa: E402
import step70e_votes as VT                          # noqa: E402
from step52_read import IGNORE                      # noqa: E402
from step53_verdict import iou_per_class, miou      # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70_why784")
TOK = Path("/data/enric/data/geobench2/pastis/tok")
ARMS = {"aligned": (14, 16), "c4375_224": (28, 8)}          # tokens per side, canvas px per token
CANVAS = 224
RUNS = {"B1 K8": "b1_k8", "E784 K8": "p7e784_s42_20261127_004451_k8"}
N_CLASSES = 20
VOID = 19


def canvas(labels: np.ndarray) -> torch.Tensor:
    """(N, 128, 128) native labels -> (N, 224, 224), the render's nearest upsampling (eo/maps/palette.render)."""
    x = torch.from_numpy(np.asarray(labels)).float()[:, None]
    return F.interpolate(x, size=(CANVAS, CANVAS), mode="nearest")[:, 0].to(torch.uint8)


def blocks(cv: torch.Tensor, n: int, s: int):
    """(N, 224, 224) -> pure (N, n*n) bool, label (N, n*n) of a pure block (255 if mixed), per-class pixel counts
    (N, n*n, C)."""
    b = cv.reshape(len(cv), n, s, n, s).permute(0, 1, 3, 2, 4).reshape(len(cv), n * n, s * s).long()
    lo, hi = b.min(-1).values, b.max(-1).values
    pure = lo == hi
    lab = torch.where(pure, lo, torch.full_like(lo, 255))
    cnt = F.one_hot(b, N_CLASSES).sum(2)
    return pure, lab, cnt


def geometry_and_codes(lab_tr, ids_tr, lab_va, ids_va, views) -> dict:
    tr0 = views[(views.split == "train") & views.in_pinned & (views.k == 0)]
    trall = views[(views.split == "train") & views.in_pinned]
    va = views[views.split == "val"]
    w_tr = {int(i): n for n, i in enumerate(ids_tr)}
    w_va = {int(i): n for n, i in enumerate(ids_va)}
    cv_tr = canvas(lab_tr[[w_tr[int(s)] for s in tr0.sample_id]])
    cv_va = canvas(lab_va[[w_va[int(s)] for s in va.sample_id]])
    out = {}
    for arm, (n, s) in ARMS.items():
        tok = np.load(TOK / f"MAP_{arm}" / "tokens.npy", mmap_mode="r")
        t_tr0 = torch.from_numpy(np.asarray(tok[tr0.view.values]).astype(np.int64))
        t_all = np.asarray(tok[trall.view.values]).astype(np.int64)
        t_va = torch.from_numpy(np.asarray(tok[va.view.values]).astype(np.int64))
        used = np.bincount(t_all.ravel(), minlength=int(t_all.max()) + 1)
        seen = used > 0
        unseen_va = float((~seen[np.clip(t_va.numpy(), 0, len(seen) - 1)] | (t_va.numpy() >= len(seen))).mean())
        p_tr, l_tr, c_tr = blocks(cv_tr, n, s)
        p_va, l_va, c_va = blocks(cv_va, n, s)
        per = {}
        for c in range(N_CLASSES):
            if c == VOID:
                continue
            pix = int(c_tr[..., c].sum())
            pix_pure = int((c_tr[..., c] * p_tr).sum())
            codes = Counter(t_tr0[l_tr == c].tolist())
            vcodes = t_va[l_va == c].tolist()
            per[c] = {"pure_pixel_share": pix_pure / max(pix, 1),
                      "pure_tokens_train": int((l_tr == c).sum()),
                      "pure_codes_train": len(codes),
                      "top_code_share": (codes.most_common(1)[0][1] / sum(codes.values())) if codes else None,
                      "val_pure_tokens": len(vcodes),
                      "val_pure_code_unseen_as_pure": (float(np.mean([x not in codes for x in vcodes]))
                                                       if vcodes else None),
                      "val_tokens_with_class_unseen": float((~seen[t_va[c_va[..., c] > 0].numpy()]).mean())
                      if int((c_va[..., c] > 0).sum()) else None}
        out[arm] = {"tokens_per_map": n * n, "codes_in_book": int(json.loads((TOK / f"MAP_{arm}" / "metadata.json")
                                                                           .read_text())["frozen"]["n_codes"]),
                    "distinct_codes_train_all_variants": int(seen.sum()),
                    "train_tokens_all_variants": int(t_all.size),
                    "occurrences_per_used_code": float(t_all.size / seen.sum()),
                    "distinct_codes_train_k0": int(len(np.unique(t_tr0.numpy()))),
                    "val_tokens_unseen_in_training": unseen_va,
                    "pure_token_share_train": float(p_tr.float().mean()),
                    "per_class": per}
    return out


def deep_mask(labels: torch.Tensor, r: int) -> torch.Tensor:
    """(N, H, W) -> bool: the (2r+1)^2 neighbourhood (replicated at the border) holds one label."""
    x = F.pad(labels.float()[:, None], (r, r, r, r), mode="replicate")
    hi = F.max_pool2d(x, 2 * r + 1, stride=1)
    lo = -F.max_pool2d(-x, 2 * r + 1, stride=1)
    return (hi == lo)[:, 0]


def cm_of(pred: torch.Tensor, lab: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    keep = mask & (lab != VOID)
    idx = lab[keep].long() * N_CLASSES + pred[keep].long()
    return torch.bincount(idx, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES).numpy()


def depth_read() -> dict:
    res, preds = {}, {}
    labels = None
    for name, vdir in RUNS.items():
        maps, vj, _ = RD.load_maps(vdir)
        sids = [str(x) for x in vj["sample_id"]] if isinstance(vj, dict) else [str(x["sample_id"]) for x in vj]
        lab = torch.from_numpy(RD.labels_for(sids).astype(np.int64))
        if labels is None:
            labels, sids0 = lab, sids
        elif sids != sids0:
            raise RuntimeError("the runs' patches differ")
        st = torch.from_numpy(np.asarray(maps)).permute(1, 0, 2, 3)          # (K, V, H, W)
        preds[name] = VT.plain_vote(st, N_CLASSES)
    masks = {"all": torch.ones_like(labels, dtype=torch.bool)}
    for r in (2, 4):
        d = deep_mask(labels, r)
        masks[f"deep r{r}"], masks[f"near r{r}"] = d, ~d
    for name, p in preds.items():
        res[name] = {}
        for m, mk in masks.items():
            cm = cm_of(p, labels, mk)
            res[name][m] = {"mIoU": miou(cm, IGNORE[RD.TASK]) * 100,
                            "iou": [None if np.isnan(x) else float(x) * 100
                                    for x in iou_per_class(cm, IGNORE[RD.TASK])],
                            "pixel_acc": float(np.trace(cm) / cm.sum() * 100), "pixels": int(cm.sum()),
                            "pred_share": (cm.sum(0) / cm.sum()).tolist(), "true_share": (cm.sum(1) / cm.sum()).tolist()}
    res["pixel_share"] = {m: float((mk & (labels != VOID)).sum() / (labels != VOID).sum()) for m, mk in masks.items()}
    return res


def stack_shares() -> dict:
    """Predicted class shares and IoU of 7.0f's corrected TTA-64 maps (E784's stack and B1's), from step70h's path."""
    import step70h_read as H
    prior = json.loads(RD.PRIOR.read_text())
    out = {}
    for label, name in (("stack (E784 TTA-64 corrected)", "p7fstack_20261127_080904"),
                        ("B1 TTA-64 corrected", "p7etta_20261127_004451")):
        cm = H.hard_of("tta", name, prior)["_cf"].sum(0)
        out[label] = {"mIoU": miou(cm, IGNORE[RD.TASK]) * 100, "pred_share": (cm.sum(0) / cm.sum()).tolist(),
                      "true_share": (cm.sum(1) / cm.sum()).tolist()}
    return out


def ladder() -> dict:
    import step70_read as S70
    import step70h_read as H
    prior = json.loads(RD.PRIOR.read_text())
    sets = {("196", "K8"): ("k8", "b1_k8"), ("196", "TTA-64"): ("tta", "p7etta_20261127_004451"),
            ("784", "K8"): ("k8", "p7e784_s42_20261127_004451_k8"), ("784", "TTA-64"): ("tta", "p7fstack_20261127_080904")}
    cms, sids = {}, {}
    for key, (kind, name) in sets.items():
        h = H.hard_of(kind, name, prior)
        sids[key] = [str(x) for x in h["_sids"]]
        cms[key + ("plain",)], cms[key + ("corrected",)] = h["_cms"][0.0], h["_cf"]
        cms[key + ("alphas",)] = h["chosen"]
    ref = sids[("196", "K8")]
    for key, sv in sids.items():                     # the pairs need one patch order
        if sv != ref:
            o = [sv.index(x) for x in ref]
            for v in ("plain", "corrected"):
                cms[key + (v,)] = cms[key + (v,)][o]
    out = {"mIoU": {}, "alphas": {}, "784_minus_196": {}, "gain_on_196": {}, "gain_on_784": {}}
    for (arm, lvl), _ in sets.items():
        for v in ("plain", "corrected"):
            out["mIoU"][f"{arm} {lvl} {v}"] = miou(cms[(arm, lvl, v)].sum(0), IGNORE[RD.TASK]) * 100
        out["alphas"][f"{arm} {lvl}"] = cms[(arm, lvl, "alphas")]
    for lvl in ("K8", "TTA-64"):
        for v in ("plain", "corrected"):
            out["784_minus_196"][f"{lvl} {v}"] = S70.pair(cms[("784", lvl, v)], cms[("196", lvl, v)])
    out["784 K8 corrected - 196 TTA-64 corrected"] = S70.pair(cms[("784", "K8", "corrected")],
                                                              cms[("196", "TTA-64", "corrected")])
    for arm in ("196", "784"):
        base = cms[(arm, "K8", "plain")]
        for lvl, v in (("K8", "corrected"), ("TTA-64", "plain"), ("TTA-64", "corrected")):
            out[f"gain_on_{arm}"][f"{lvl} {v}"] = S70.pair(cms[(arm, lvl, v)], base)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from eo.maps import palette as P, sets
    names = P.TASKS[RD.TASK].class_names
    prior = json.loads(RD.PRIOR.read_text())["share"]
    views = pd.read_parquet(TOK / "views.parquet")
    lab_tr, ids_tr = sets.load("pastis", "train")
    lab_va, ids_va = sets.load("pastis", "val")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((out / "why784.json").read_text()) if (out / "why784.json").exists() else {}
    if "geometry_codes" in old:                      # json keys are strings; the md below indexes classes by int
        old["geometry_codes"] = {arm: {**x, "per_class": {int(k): v for k, v in x["per_class"].items()}}
                                 for arm, x in old["geometry_codes"].items()}
    res = {"what": __doc__.split("\n")[0],
           "geometry_codes": old.get("geometry_codes") or geometry_and_codes(lab_tr, ids_tr, lab_va, ids_va, views),
           "depth": old.get("depth") or depth_read(), "class_names": names, "train_share": [prior.get(n) for n in names]}
    res["stack_shares"] = old.get("stack_shares") or stack_shares()
    res["ladder"] = old.get("ladder") or ladder()
    g = res["geometry_codes"]
    order = sorted([c for c in range(N_CLASSES) if c != VOID], key=lambda c: -prior[names[c]])
    L = ["# Why 784 beats 196 on PASTIS-7 (descriptive)", ""]
    for arm, x in g.items():
        L.append(f"- {arm}: {x['tokens_per_map']} tokens, {x['codes_in_book']} codes; {x['distinct_codes_train_all_variants']} "
                 f"used over the training views ({x['occurrences_per_used_code']:.0f} occurrences per used code); "
                 f"fold-4 tokens unseen in training {x['val_tokens_unseen_in_training'] * 100:.2f}%; pure tokens "
                 f"{x['pure_token_share_train'] * 100:.1f}%")
    L += ["", "| class | train % | pure px 196 | pure px 784 | pure codes 196 | pure codes 784 | top code 196 | top code 784 "
          "| val pure unseen 196 | val pure unseen 784 | tokens with class unseen 196 | 784 |", "|" + "---|" * 12]
    f = lambda v, s=100, d=1: "-" if v is None else f"{v * s:.{d}f}"          # noqa: E731
    for c in order:
        A, C = g["aligned"]["per_class"][c], g["c4375_224"]["per_class"][c]
        L.append(f"| {names[c]} | {prior[names[c]] * 100:.2f} | {f(A['pure_pixel_share'])} | {f(C['pure_pixel_share'])} | "
                 f"{A['pure_codes_train']} | {C['pure_codes_train']} | {f(A['top_code_share'])} | {f(C['top_code_share'])} | "
                 f"{f(A['val_pure_code_unseen_as_pure'])} | {f(C['val_pure_code_unseen_as_pure'])} | "
                 f"{f(A['val_tokens_with_class_unseen'])} | {f(C['val_tokens_with_class_unseen'])} |")
    d = res["depth"]
    L += ["", "| set | pixel share | B1 K8 mIoU | E784 K8 mIoU | gain | B1 pixel acc | E784 pixel acc |", "|---|---|---|---|---|---|---|"]
    for m in d["B1 K8"]:
        b, e = d["B1 K8"][m], d["E784 K8"][m]
        L.append(f"| {m} | {d['pixel_share'][m] * 100:.1f}% | {b['mIoU']:.2f} | {e['mIoU']:.2f} | {e['mIoU'] - b['mIoU']:+.2f} | "
                 f"{b['pixel_acc']:.2f} | {e['pixel_acc']:.2f} |")
    L += ["", "| class | train % | B1 all | E784 all | B1 deep r4 | E784 deep r4 | B1 near r4 | E784 near r4 |", "|" + "---|" * 8]
    for c in order:
        row = [names[c], f"{prior[names[c]] * 100:.2f}"]
        for m in ("all", "deep r4", "near r4"):
            for n in ("B1 K8", "E784 K8"):
                v = d[n][m]["iou"][c]
                row.append("-" if v is None else f"{v:.1f}")
        L.append("| " + " | ".join(row) + " |")
    groups = {"5 most common": order[:5], "7 middle": order[5:12], "7 rarest": order[12:]}
    L += ["", "| predicted share of pixels | " + " | ".join(groups) + " |", "|---|---|---|---|"]
    sh = {"fold 4 truth": d["B1 K8"]["all"]["true_share"], "B1 K8": d["B1 K8"]["all"]["pred_share"],
          "E784 K8": d["E784 K8"]["all"]["pred_share"], **{k: v["pred_share"] for k, v in res["stack_shares"].items()}}
    for k, v in sh.items():
        L.append(f"| {k} | " + " | ".join(f"{sum(v[c] for c in g) * 100:.1f}%" for g in groups.values()) + " |")
    lad = res["ladder"]
    f2 = lambda p: f"{p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}]"       # noqa: E731
    L += ["", "| level | 196 (B1) | 784 (E784-s42) | 784 - 196 | gain on 196 | gain on 784 |", "|---|---|---|---|---|---|"]
    for lvl in ("K8", "TTA-64"):
        for v in ("plain", "corrected"):
            k = f"{lvl} {v}"
            L.append(f"| {k} | {lad['mIoU']['196 ' + k]:.2f} | {lad['mIoU']['784 ' + k]:.2f} | {f2(lad['784_minus_196'][k])} | "
                     + (" | ".join(f2(lad[g][k]) for g in ("gain_on_196", "gain_on_784")) if k != "K8 plain" else "- | -")
                     + " |")
    L.append(f"- alphas chosen (two halves): {lad['alphas']}")
    L.append(f"- 784 K8 corrected - 196 TTA-64 corrected: {f2(lad['784 K8 corrected - 196 TTA-64 corrected'])}")
    (out / "why784.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "why784.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
