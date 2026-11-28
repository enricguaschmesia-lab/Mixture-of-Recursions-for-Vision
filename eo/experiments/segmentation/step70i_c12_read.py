#!/usr/bin/env python
"""A2 Step 7.0i's CloudSen12 read (docs/plans/a2_segmentation.md, Step 7, "7.0i", the CloudSen12 read rule, item 3; the
rule pre-registered 2026-11-27 and its prior and halves fixed 2026-11-28, before any corrected CloudSen12 number).
`.venv`, CPU; step70i_c12_votes.py's per-tile vote counts.

    python eo/experiments/segmentation/step70i_c12_read.py [--votes <dir>] [--out <dir>]

  validity  each seed's counts give the scorer's plain-vote matrices, at K = 40 and at K = 8 (votes.json);
  primary   corrected - plain at K = 40 per seed (alpha cross-fitted over the fixed halves on 7.0e's grid, with the fixed
            prior): gains iff seed 42's interval is above 0 and seeds 43 and 44 read above 0; costs on the mirror; else
            level. The candidate decoding for CloudSen12's row: the corrected vote at K = 40 if it gains, else the plain
            vote at K = 40 (the test-fold protocol stays Enric's);
  beside    the three-seed means (plain and corrected, K = 40 and K = 8), the alphas, per class.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step4_score import per_patch_cm                # noqa: E402
from step53_verdict import iou_per_class, miou      # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70i")
VOTES = OUT / "c12_votes"
PRIOR = OUT / "class_prior_cloudsen12.json"
HALVES = OUT / "halves_cloudsen12.json"
SEEDS = {42: "r0c12s52_s42_20261122_190655", 43: "r0c12s52_s43_20261122_190655", 44: "r0c12s52_s44_20261122_190655"}
PRIMARY = 42
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_CLASSES, IGNORE = 4, None
N_BOOT, BOOT_SEED = 1000, 0
TARGET, TERRAMIND_L = 61.9, 63.9


def pair_task(x: np.ndarray, y: np.ndarray, ignore) -> dict:
    """x - y in mIoU points, paired bootstrap over patches or tiles: step70_read.pair's, with the task's ignore."""
    if x.shape != y.shape:
        raise RuntimeError(f"shapes {x.shape} {y.shape}")
    mx, my = miou(x.sum(0), ignore) * 100, miou(y.sum(0), ignore) * 100
    rng = np.random.default_rng(BOOT_SEED)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d = np.array([miou(x[i].sum(0), ignore) - miou(y[i].sum(0), ignore) for i in idx]) * 100
    return {"x": mx, "y": my, "delta": mx - my, "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5)), "share_positive": float((d > 0).mean())}


def weights(prior: dict, alpha: float) -> np.ndarray:
    w = np.ones(N_CLASSES, dtype=np.float64)
    for name, i in prior["class_index"].items():
        w[i] = float(prior["share"][name]) ** (-float(alpha)) if alpha else 1.0
    return w


def weighted_pred(counts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(C, H, W) votes -> (H, W): argmax_c w_c V_c, ties to the lower class; a class no sample wrote never wins."""
    return np.argmax(counts.astype(np.float64) * w[:, None, None], axis=0).astype(np.uint8)


def cms_of(counts: np.ndarray, labels: np.ndarray, w: np.ndarray) -> np.ndarray:
    pred = np.stack([weighted_pred(np.asarray(counts[i]), w) for i in range(len(counts))])
    return per_patch_cm(pred, labels, N_CLASSES, IGNORE)


def best_alpha(cms_by_alpha: dict, mask: np.ndarray, ignore) -> float:
    """step70e_read.best_alpha's rule with the task's ignore: the highest mIoU over the masked tiles, ties to the smaller."""
    best, val = None, -1.0
    for a in sorted(cms_by_alpha):
        m = miou(cms_by_alpha[a][mask].sum(0), ignore)
        if m > val + 1e-15:
            best, val = a, m
    return best


def cross_fit(cms_by_alpha: dict, h: np.ndarray, ignore) -> tuple:
    a0, a1 = best_alpha(cms_by_alpha, h == 0, ignore), best_alpha(cms_by_alpha, h == 1, ignore)
    return np.where((h == 0)[:, None, None], cms_by_alpha[a1], cms_by_alpha[a0]), {0: a0, 1: a1}


def verdict(primary: dict, others: list) -> str:
    if primary["lo95"] > 0 and all(o["delta"] > 0 for o in others):
        return "gains"
    if primary["hi95"] < 0 and all(o["delta"] < 0 for o in others):
        return "costs"
    return "level"


def seed_read(seed: int, run: str, prior: dict, half: dict, votes: Path = VOTES) -> tuple:
    from eo.maps import sets
    d = votes / run
    meta = json.loads((d / "votes.json").read_text())
    why = [f"seed {seed}: {k}'s plain vote differs from the scorer's matrices" for k, v in meta["check"].items()
           if not v["equal_to_scorer"]]
    if meta.get("drop_first"):
        why.append(f"seed {seed}: counts made with --drop-first")
    tiles = meta["tiles"]
    lab, ids = sets.load("cloudsen12", "val")
    where = {str(i): n for n, i in enumerate(ids)}
    labels = np.asarray(lab[[where[str(s)] for s in tiles]])
    h = np.asarray([half[str(s)] for s in tiles], dtype=np.int64)
    r = {"run": run, "tiles": len(tiles)}
    for k in (f"k{meta['K']}", "k8"):
        counts = np.load(d / f"counts_{k}.npy", mmap_mode="r")
        cms = {a: cms_of(counts, labels, weights(prior, a)) for a in ALPHAS}
        cf, chosen = cross_fit(cms, h, IGNORE)
        r[k] = {"plain": miou(cms[0.0].sum(0), IGNORE) * 100, "corrected": miou(cf.sum(0), IGNORE) * 100,
                "alphas": {str(a): v for a, v in chosen.items()},
                "by_alpha": {str(a): miou(c.sum(0), IGNORE) * 100 for a, c in cms.items()},
                "corrected_minus_plain": pair_task(cf, cms[0.0], IGNORE),
                "per_class_plain": (iou_per_class(cms[0.0].sum(0), IGNORE) * 100).tolist(),
                "per_class_corrected": (iou_per_class(cf.sum(0), IGNORE) * 100).tolist()}
    return r, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", default=str(VOTES))
    ap.add_argument("--out", default=str(OUT / "c12_read"))
    a = ap.parse_args()
    prior = json.loads(PRIOR.read_text())
    half = json.loads(HALVES.read_text())["half"]
    res = {"rule": "A2 plan, Step 7, 7.0i, CloudSen12 item 3 (pre-registered 2026-11-27; prior and halves fixed "
                   "2026-11-28)", "prior": prior["share"], "seeds": {}}
    why = []
    for seed, run in SEEDS.items():
        r, w = seed_read(seed, run, prior, half, Path(a.votes))
        res["seeds"][seed] = r
        why += w
    K = [k for k in res["seeds"][PRIMARY] if k.startswith("k") and k != "k8"][0]
    prim = res["seeds"][PRIMARY][K]["corrected_minus_plain"]
    others = [res["seeds"][s][K]["corrected_minus_plain"] for s in SEEDS if s != PRIMARY]
    res["valid"], res["why"] = not why, why
    res["verdict"] = verdict(prim, others) if not why else "not read"
    res["row_decoding"] = (f"corrected vote at K = {K[1:]}" if res["verdict"] == "gains" else f"plain vote at K = {K[1:]}")
    m = lambda k, v: float(np.mean([res["seeds"][s][k][v] for s in SEEDS]))       # noqa: E731
    res["means"] = {f"{k} {v}": m(k, v) for k in (K, "k8") for v in ("plain", "corrected")}
    res["target"], res["terramind_l"] = TARGET, TERRAMIND_L
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda p: f"{p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}]"       # noqa: E731
    L = ["# A2 Step 7.0i: CloudSen12's frequency-corrected vote", "", f"- valid {res['valid']} {why}", "",
         f"| seed | {K} plain | {K} corrected | corrected - plain ({K}) | alphas | k8 plain | k8 corrected | corrected - plain (k8) |",
         "|---|---|---|---|---|---|---|---|"]
    for s, r in res["seeds"].items():
        L.append(f"| {s} | {r[K]['plain']:.2f} | {r[K]['corrected']:.2f} | {f(r[K]['corrected_minus_plain'])} | "
                 f"{r[K]['alphas']} | {r['k8']['plain']:.2f} | {r['k8']['corrected']:.2f} | {f(r['k8']['corrected_minus_plain'])} |")
    mm = res["means"]
    L += ["", f"- verdict: {res['verdict']}; the row's candidate decoding: {res['row_decoding']}",
          f"- three-seed means: {K} plain {mm[K + ' plain']:.2f}, corrected {mm[K + ' corrected']:.2f}; k8 plain "
          f"{mm['k8 plain']:.2f}, corrected {mm['k8 corrected']:.2f}; target {TARGET}, TerraMind-L {TERRAMIND_L}"]
    (out / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
