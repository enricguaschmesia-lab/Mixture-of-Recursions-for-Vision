#!/usr/bin/env python
"""A2 Step 4.1, item 12: the region rule's logit form on the recipe's generated fold-4 maps, chosen and read by
cross-fitting (`a2seg`, one card: the Titan X under A1's flock, /data/enric/reports/a2_seg/queues/step41_titanx.sh).

The frozen rule (step4/pick.json) is nearest on palette logits. Step 0's region rule (eo/maps/decode.snap_region) reads
RGB distances, which the aligned tokenizer does not write, so its logit form is read here (the plan, Step 4.1, item 12):
  a pixel is CONFIDENT if its nearest class's palette probability, pooled to the native grid, is at least tau (the
  harness pools log-probabilities, so this is their geometric mean over the pixel's rendered pixels); any other pixel
  may only take a class that a confident pixel holds within r native pixels; with none near, it keeps its nearest
  class. It runs on each sample before the per-pixel vote, as nearest does in eo/eval/seg.score.
Grid: tau in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99} x r in {1, 2, 3, 4, 6, 8}, plus nearest.

For each map set (the recipe's K = 8 maps at seeds 42 and 43), every patch's confusion matrix under every rule, in one
decode. Then: fold 4's 482 patches are split into two fixed halves (seed 0); the rule is chosen on one half (the highest
mIoU of the summed matrices; ties to nearest, then the smaller r, then the larger tau) and read on the other, both ways;
the cross-fitted mIoU is the union's, against nearest on the same maps. The bootstrap: 1,000 resamples of the 482
patches (seed 0), the same for both seeds, the mean over seeds of (cross-fitted - nearest), the choices held fixed.
ADOPTED if the mean cross-fitted gain is >= 0.005 with its 2.5 % bound above 0; then (tau, r) is the rule with the
highest mean mIoU over the two seeds on all of fold 4, and the true-token ceiling is read again under it.

Self-tests before any map: tau 0 equals nearest exactly on real scores; a synthetic map where a doubtful pixel next to a
confident region takes that region's class, and keeps its own with nobody confident nearby (the controls).

    python eo/experiments/segmentation/step41_region.py --maps <seed-42 full_k8 dir> <seed-43 full_k8 dir> \\
        --out /data/enric/reports/a2_seg/step41/region [--expect-gpu "GTX TITAN X"]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eo.data import map_vocab as MV          # noqa: E402
from eo.data import seg_dataset as SD        # noqa: E402
from eo.eval import seg as SEG                # noqa: E402
from eo.maps import decode as D, frozen, harness as H, palette as P   # noqa: E402

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
RADII = (1, 2, 3, 4, 6, 8)
GAIN, N_BOOT, BOOT_SEED, SPLIT_SEED = 0.005, 1000, 0, 0
NEAREST = "nearest"


def key(tau, r) -> str:
    return NEAREST if tau is None else f"tau{tau:g}_r{r}"


RULES = [(None, None)] + list(itertools.product(TAUS, RADII))


def region_logit(sc: torch.Tensor, ignore, tau: float, radius: int) -> torch.Tensor:
    """sc: pooled log-probabilities (B, K, H, W). Step 0's snap_region with confidence = probability >= tau."""
    sc = D.mask_ignored(sc, ignore)
    best, k1 = sc.max(1)
    conf = best >= (math.log(tau) if tau > 0 else -math.inf)
    K = sc.shape[1]
    onehot = F.one_hot(k1, K).permute(0, 3, 1, 2).float() * conf[:, None].float()
    near = F.max_pool2d(onehot, 2 * radius + 1, stride=1, padding=radius) > 0
    allowed = near | conf[:, None]
    has_any = near.any(1, keepdim=True)
    allowed = torch.where(has_any, allowed, torch.ones_like(allowed))
    out = torch.where(allowed, sc, torch.full_like(sc, D.NEG)).argmax(1)
    return torch.where(conf, k1, out)


def apply(sc, ignore, tau, r):
    return D.snap_nearest(sc, ignore) if tau is None else region_logit(sc, ignore, tau, r)


def miou(cm: np.ndarray, ignore) -> float:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(denom > 0, tp / np.where(denom > 0, denom, 1), 0.0)
    w = np.ones(len(cm))
    if ignore is not None:
        w[ignore] = 0
    w[(cm.sum(0) + cm.sum(1)) == 0] = 0
    return float((w * iou).sum() / w.sum())


def per_patch(pred: torch.Tensor, gt: torch.Tensor, n: int, ignore) -> np.ndarray:
    """(B, H, W) predictions and labels -> (B, n, n) confusion matrices (target x prediction), the ignored class out."""
    g, p = gt.long().flatten(1), pred.long().flatten(1)
    v = (g != ignore) if ignore is not None else torch.ones_like(g, dtype=torch.bool)
    idx = torch.where(v, g * n + p, torch.full_like(g, n * n))
    oh = torch.zeros(len(g), n * n + 1, dtype=torch.int64, device=g.device)
    oh.scatter_add_(1, idx, torch.ones_like(idx))
    return oh[:, : n * n].view(-1, n, n).cpu().numpy()


def self_tests(sc_real: torch.Tensor, ignore) -> dict:
    eq0 = bool(torch.equal(region_logit(sc_real, ignore, 0.0, 3), D.snap_nearest(sc_real, ignore)))
    # synthetic: 3 classes, 1 x 9 strip; left 3 pixels class 0 at p 0.95; pixel 3 doubtful, nearest class 2 (0.40 vs
    # 0.35 for class 0); the rest class 1 confident; pixel 8 doubtful with nobody confident within r = 1
    p = torch.full((1, 3, 1, 9), 0.025)
    p[0, 0, 0, :3] = 0.95
    p[0, :, 0, 3] = torch.tensor([0.35, 0.25, 0.40])
    p[0, 1, 0, 4:7] = 0.95
    p[0, :, 0, 7] = torch.tensor([0.30, 0.30, 0.40])
    p[0, :, 0, 8] = torch.tensor([0.30, 0.30, 0.40])
    sc = torch.log(p / p.sum(1, keepdim=True))
    out = region_logit(sc, None, 0.8, 1)[0, 0].tolist()
    near = D.snap_nearest(sc, None)[0, 0].tolist()
    want, want_near = [0, 0, 0, 0, 1, 1, 1, 1, 2], [0, 0, 0, 2, 1, 1, 1, 2, 2]
    return {"tau0_equals_nearest": eq0, "synthetic_region": out, "synthetic_nearest": near,
            "synthetic_expected": want, "pass": eq0 and out == want and near == want_near}


def score_set(d: Path, tok, dev, chunk: int):
    rec = json.loads((d / "record.json").read_text())
    if rec["split"] != "val" or rec["greedy"] or int(rec["k"]) < 1:
        raise SystemExit(f"{d}: not sampled val maps")
    codes = np.load(d / "codes.npy")
    task, arm = rec["task"], rec["arm"]
    t = P.TASKS[task]
    side = MV.spec(arm)[2]
    ds = SD.SegDataset(SD.SegSpec(task, arm, rec["frames"], rec["split"]))
    views = ds.views.set_index("view").loc[rec["views"]].reset_index()
    labels, s_idx = SEG.gt_for_views(task, views)
    if not np.array_equal(s_idx, np.arange(len(labels))):
        raise RuntimeError("PASTIS views are not one per patch in order")
    V, K, G = codes.shape
    cms = {key(*r): np.zeros((V, t.n_classes, t.n_classes), np.int64) for r in RULES}
    tests = None
    with torch.no_grad():
        for s in range(0, V, chunk):
            e = min(s + chunk, V)
            ib = torch.as_tensor(codes[s:e].reshape(-1, G).astype(np.int64), device=dev)
            sc = H.class_scores(tok, ib.view(-1, side, side), task, "universal", "crop224")     # ((e-s)*K, C, H, W)
            sc = sc.view(e - s, K, *sc.shape[1:])
            if tests is None:
                tests = self_tests(sc[:, 0].float(), t.ignore)
                if not tests["pass"]:
                    raise SystemExit(f"self-tests failed: {tests}")
            gt = torch.as_tensor(np.ascontiguousarray(labels[s:e]), device=dev)
            for r in RULES:
                pr = torch.stack([apply(sc[:, j], t.ignore, *r) for j in range(K)])          # (K, B, H, W)
                cms[key(*r)][s:e] = per_patch(SEG.vote(pr, t.n_classes), gt, t.n_classes, t.ignore)
    # the nearest matrices must reproduce step4_score's for the same maps
    # (exactly on the same card; a cross-card read differs at ties: the TITAN V's ceiling against the Titan X's, 1e-7)
    ref = d / "per_patch_cm.npy"
    cmp_ = None
    if ref.exists():
        r0 = np.load(ref)
        cmp_ = {"equal": bool(np.array_equal(r0, cms[NEAREST])), "max_abs_count_diff": int(np.abs(r0 - cms[NEAREST]).max()),
                "mIoU_diff": miou(cms[NEAREST].sum(0), t.ignore) - miou(r0.sum(0), t.ignore)}
    return {"dir": str(d), "K": int(K), "cms": cms, "ignore": t.ignore, "nearest_vs_step4_score": cmp_,
            "self_tests": tests}


def crossfit(cms: dict, ignore, halves):
    out = {}
    m_all = {k: miou(v.sum(0), ignore) for k, v in cms.items()}
    order = sorted(cms, key=lambda k: (k != NEAREST, int(k.split("_r")[1]) if k != NEAREST else 0,
                                       -float(k.split("_")[0][3:]) if k != NEAREST else 0))
    choice = {}
    for h, other in ((0, 1), (1, 0)):
        idx = halves[other]
        sc = {k: miou(cms[k][idx].sum(0), ignore) for k in order}
        best = max(sc.values())
        choice[h] = next(k for k in order if sc[k] == best)          # ties: nearest, smaller r, larger tau
    cf = cms[choice[0]][halves[0]].sum(0) + cms[choice[1]][halves[1]].sum(0)
    out.update({"choice_for_half": {str(h): c for h, c in choice.items()}, "crossfit_mIoU": miou(cf, ignore),
                "nearest_mIoU": m_all[NEAREST], "gain": miou(cf, ignore) - m_all[NEAREST], "all_rules_mIoU": m_all})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True, help="the recipe's K = 8 map dirs, one per seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    t0 = time.time()
    tok = frozen.load("aligned", dev)
    sets_ = [score_set(Path(d), tok, dev, a.chunk) for d in a.maps]
    V = len(next(iter(sets_[0]["cms"].values())))
    perm = np.random.default_rng(SPLIT_SEED).permutation(V)
    halves = (np.sort(perm[: V // 2]), np.sort(perm[V // 2:]))
    ign = sets_[0]["ignore"]
    cf = [crossfit(s["cms"], ign, halves) for s in sets_]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, V, size=(N_BOOT, V))
    gains = []
    for i in idx:
        g = []
        for s, c in zip(sets_, cf):
            m0 = np.isin(i, halves[0])
            cm_cf = (s["cms"][c["choice_for_half"]["0"]][i[m0]].sum(0) + s["cms"][c["choice_for_half"]["1"]][i[~m0]].sum(0))
            g.append(miou(cm_cf, ign) - miou(s["cms"][NEAREST][i].sum(0), ign))
        gains.append(np.mean(g))
    gains = np.array(gains)
    mean_gain = float(np.mean([c["gain"] for c in cf]))
    lo, hi = float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))
    adopt = bool(mean_gain >= GAIN and lo > 0)
    mean_all = {k: float(np.mean([c["all_rules_mIoU"][k] for c in cf])) for k in cf[0]["all_rules_mIoU"]}
    final = max(mean_all, key=lambda k: (mean_all[k], k == NEAREST)) if adopt else NEAREST
    res = {"protocol": "A2 Step 4.1, item 12 (docs/plans/a2_segmentation.md)", "grid": {"tau": TAUS, "r": RADII},
           "halves_seed": SPLIT_SEED, "half_sizes": [len(h) for h in halves],
           "sets": [{k: v for k, v in s.items() if k != "cms"} for s in sets_], "crossfit": cf,
           "mean_crossfit_gain": mean_gain, "bootstrap_gain": {"lo95": lo, "hi95": hi, "mean": float(gains.mean()),
                                                               "resamples": N_BOOT, "seed": BOOT_SEED},
           "adopted": adopt, "rule": "adopted if the mean cross-fitted gain >= 0.005 with its 2.5 % bound above 0",
           "final_rule": final, "mean_mIoU_all_rules": mean_all, "seconds": time.time() - t0,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "region.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for s, c in zip(a.maps, cf):
        print(f"{s}: nearest {c['nearest_mIoU']:.4f}, cross-fitted {c['crossfit_mIoU']:.4f} (gain {c['gain']:+.4f}; "
              f"choices {c['choice_for_half']})")
    print(f"mean cross-fitted gain {mean_gain:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {'ADOPTED: ' + final if adopt else 'nearest stays'}")


if __name__ == "__main__":
    main()
