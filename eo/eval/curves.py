# eo/eval/curves.py
"""Per-checkpoint curves for one arm, and the compute axis D3.11 is read on.

Joins three independently produced artifacts on checkpoint name:

  trainer_state.json                         the logged train/eval curves
  reports/arm_eval/<arm>/<ckpt>/metrics.json teacher-forced accuracy, depth
                                             (eo/scripts/eval_checkpoint.py)
  reports/d311/<arm>/<ckpt>/<T>_*/metrics_<T>.json
                                             generation vs held-out pixels
                                             (eo/scripts/sweep_d311.sh)

⚠ THE FLOPs AXIS IS A CUMULATIVE INTEGRAL, not step x constant (handover
section 6 item 3). Arm A's cost per step moves as the router learns, so it is
measured at every checkpoint (step 0 = the untrained model) and integrated with
the trapezoid rule. Taking one late measurement and applying it backwards
credits arm A with a cheapness it did not have early, in MoR's favour.

TWO DEFINITIONS are computed, and they disagree by enough to matter:

  plan   Step 8's `compute_accounting`: 6 x P_layer x layer applications,
         body tokens only. Excludes lm_head, attention's quadratic term and
         padding. This is the definition PHASE3_PLAN Step 10.2b names.
  full   every position the model actually processes (1048 per row, padding
         included), the router, attention (causal, ideal: 2 n^2 d per layer
         over the n tokens that layer sees) and the 50.4M-parameter lm_head.
         lm_head alone costs ~14 transformer layers per token and is identical
         in both arms, so it DILUTES MoR's relative saving. This is the
         conservative (MoR-unfavourable) reading of "FLOPs consumed".

Training FLOPs = 3 x forward (backward ~ 2 x forward). Implementation
overhead (MoR pads each recursion step to the longest selection in the batch)
is not counted: these are the FLOPs the algorithm needs, not what one kernel
schedule spent. Routing is measured on the HELD-OUT rows as a proxy for the
training rows at the same checkpoint.

Runs in `.venv` (numpy only).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# SmolLM-135M geometry (conf: hidden 576, intermediate 1536, 9 heads, 3 KV heads).
D, FFN, V = 576, 1536, 87556
P_LAYER = (576 * 576 * 2 + 576 * 192 * 2) + 3 * 576 * FFN + 2 * 576      # 3,540,096
P_ROUTER = 576 * 1152 + 1152 * 3                                          # 667,008
assert 29 * P_LAYER == 102_662_784                 # arm B non-embedding, measured 102.66M
assert 11 * P_LAYER + P_ROUTER == 39_608_064       # arm A non-embedding, measured 39.61M
SEQ, BATCH, BODY_TOKENS = 1048, 256, 5 * 196 + 3   # every row: 5 image modalities + Coords
OUTER, BLOCKS_PER_STEP, N_REC = 2, 9, 3
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
TARGETS = ["LULC", "DEM", "S2L2A", "NDVI", "S1RTC", "S1GRD"]
EPOCH_STEPS = 84_672 / BATCH


# ------------------------------------------------------------------ compute

def step_flops(depth: Optional[np.ndarray], input_mod: Optional[np.ndarray]) -> Dict[str, float]:
    """Training FLOPs per optimizer step under both definitions.

    `depth` is (rows, positions) passes-through-the-recursed-stack, or None for a
    model with no router (every token takes all 29 layers).
    """
    if depth is None:
        lin = 2 * P_LAYER * 29 * SEQ
        att = 2 * D * 29 * SEQ ** 2
        layers_body = 29.0
    else:
        scale = SEQ / depth.shape[1]                 # the captured frame drops one position
        n = np.stack([(depth >= i).sum(1) * scale for i in range(1, N_REC + 1)], 1)  # (rows, 3)
        lin = (2 * P_LAYER * (OUTER * SEQ + BLOCKS_PER_STEP * n.sum(1)) + 2 * P_ROUTER * SEQ).mean()
        att = (2 * D * (OUTER * SEQ ** 2 + BLOCKS_PER_STEP * (n ** 2).sum(1))).mean()
        body = input_mod > 0
        layers_body = float(OUTER + BLOCKS_PER_STEP * depth[body].mean())
    head = 2 * D * V * SEQ
    return {"full": 3 * BATCH * float(lin + att + head),
            "plan": 6 * P_LAYER * layers_body * BODY_TOKENS * BATCH,
            "layers_per_body_token": layers_body}


def cumulative(steps: List[int], per_step: List[float]) -> List[float]:
    """Trapezoid integral of per-step cost from step 0 to each measured step."""
    out, acc = [0.0], 0.0
    for i in range(1, len(steps)):
        acc += (steps[i] - steps[i - 1]) * (per_step[i] + per_step[i - 1]) / 2
        out.append(acc)
    return out


# ------------------------------------------------------------------ loaders

def logged_curves(run_dir: Path) -> Dict:
    lh = json.loads((Path(run_dir) / "trainer_state.json").read_text())["log_history"]
    ev = defaultdict(dict)
    for e in lh:                    # ⚠ per-modality eval keys are a SIBLING entry
        if any(k.startswith("eval_loss") for k in e):
            ev[e["step"]].update(e)
    train = [e for e in lh if "loss" in e and "learning_rate" in e]
    return {"eval": [dict(v, step=s) for s, v in sorted(ev.items())], "train": train}


def load_arm(arm: str, run_dir: str, eval_root="/data/enric/reports/arm_eval",
             d311_root="/data/enric/reports/d311") -> Dict:
    """Everything known about one arm, one record per evaluated checkpoint."""
    run_dir = Path(run_dir)
    recs = []
    for mp in sorted(Path(eval_root, arm).glob("*/metrics.json")):
        doc = json.loads(mp.read_text())
        if "router" not in doc.get("modes", {}):
            continue
        ck = doc["checkpoint"]
        r = {"checkpoint": ck, "step": doc["step"], "epoch": doc["step"] / EPOCH_STEPS,
             "tf": doc["modes"]["router"]["per_modality"], "tl": doc.get("target_last", {}),
             "modes": doc["modes"], "logged": doc.get("logged_eval") or {},
             "has_router": doc["has_router"], "dir": str(mp.parent)}
        tok = np.load(mp.parent / "tokens_router.npz")
        depth = tok["depth"] if "depth" in tok.files else None
        r.update(step_flops(depth, tok["input_mod"]))
        gen = {}
        for T in TARGETS:
            hits = list(Path(d311_root, arm, ck).glob(f"{T}_*/metrics_{T}.json"))
            if hits:
                g = json.loads(hits[0].read_text())
                st = Path(g["provenance"]["decode_dir"]) / f"stats_{T}.json"
                gen[T] = {"summary": g["summary"], "metric": g["collapse_metric"],
                          "per_scene": g["per_scene"],
                          "gen_stats": json.loads(st.read_text())["stats"] if st.exists() else {}}
        r["gen"] = gen
        recs.append(r)
    recs.sort(key=lambda r: r["step"])
    if recs and recs[0]["step"] == 0:
        steps = [r["step"] for r in recs]
        for key in ("full", "plan"):
            for r, c in zip(recs, cumulative(steps, [r[key] for r in recs])):
                r[f"cum_flops_{key}"] = c
    return {"arm": arm, "run_dir": str(run_dir), "checkpoints": recs,
            "logged": logged_curves(run_dir)}


def gen_value(rec: Dict, T: str, source: str = "generated"):
    g = rec["gen"].get(T)
    if not g:
        return None
    return g["summary"][source].get(g["metric"])


# ------------------------------------------------------------------ recursion

def _eta2(v: np.ndarray, g: np.ndarray) -> float:
    tot = v.var()
    if tot == 0:
        return 0.0
    m = v.mean()
    return float(sum((g == k).sum() * (v[g == k].mean() - m) ** 2 for k in np.unique(g)) / (v.size * tot))


def routing_summary(tok) -> Dict:
    """Depth by modality, the modality/position decomposition, depth vs difficulty.

    Attribution follows Step 8: a token's depth belongs to the modality of the
    INPUT token. Depth vs difficulty pairs depth[t] with the CE of predicting
    token t+1, over positions whose input and label are the same modality's body.
    """
    from eo.data.eo_vocab import MODALITY_TO_ID
    depth, im, lm, ce = tok["depth"], tok["input_mod"], tok["label_mod"], tok["ce"].astype(np.float32)
    body = im > 0
    pos = np.broadcast_to(np.arange(depth.shape[1]), depth.shape)
    d = depth[body].astype(np.float64)
    out = {"by_modality": {}, "depth_vs_ce": {},
           "eta2_modality": _eta2(d, im[body]),
           "eta2_position": _eta2(d, (pos[body] * 16) // depth.shape[1])}
    for name in MODS:
        mid = MODALITY_TO_ID[name]
        sel = im == mid
        if not sel.any():
            continue
        dm = depth[sel]
        out["by_modality"][name] = {
            "mean_depth": float(dm.mean()),
            "share": [float((dm == k).mean()) for k in (1, 2, 3)]}
        pair = sel & (lm == mid)
        dd, cc = depth[pair], ce[pair]
        by = {k: float(cc[dd == k].mean()) for k in (1, 2, 3) if (dd == k).sum() >= 50}
        rho = float(np.corrcoef(_rank(dd), _rank(cc))[0, 1]) if dd.std() > 0 else None
        out["depth_vs_ce"][name] = {"ce_by_depth": by, "spearman": rho,
                                    "n_by_depth": {k: int((dd == k).sum()) for k in (1, 2, 3)}}
        # within modality: scene vs patch position (image modalities only)
        if name != "Coords":
            rows = np.flatnonzero(sel.sum(1) == 196)
            if len(rows) >= 5:
                vals = np.stack([depth[r][sel[r]] for r in rows]).astype(np.float64)
                out["by_modality"][name]["by_scene"] = _eta2(vals.ravel(), np.repeat(np.arange(len(rows)), 196))
                out["by_modality"][name]["by_patch_position"] = _eta2(vals.ravel(), np.tile(np.arange(196), len(rows)))
                out["by_modality"][name]["spatial_mean"] = vals.mean(0).reshape(14, 14).tolist()
    return out


PATCH_STATS = {"LULC": ["entropy", "n_classes", "majority_share"], "DEM": ["std", "range"],
               "NDVI": ["mean", "std"], "S2L2A": ["texture", "brightness"],
               "S1RTC": ["vv_std", "vv_mean"], "S1GRD": ["vv_std", "vv_mean"]}


def _rank(x):
    """AVERAGE ranks for ties (Spearman's definition).

    ⚠ Not argsort(argsort(x)): that breaks ties by array position, and depth has
    three values while LULC entropy is mostly exactly 0. Both arrays are laid
    out scene-by-scene, so tie-broken-by-position ranks correlate with each
    other through the layout alone -- measured: an UNTRAINED router scored
    rho +0.17 against LULC entropy that way.
    """
    x = np.asarray(x).ravel()
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv].astype(np.float64)


def depth_vs_content(tok, eval_rows: np.ndarray,
                     stats_dir="/data/enric/reports/arm_eval/patch_stats") -> Dict:
    """Does a token's depth track the content of ITS OWN 16x16 patch?

    Two readings per (modality, statistic):
      pooled        Spearman over every token of every scene -- mixes
                    between-scene and within-scene variation.
      within_scene  Pearson of scene-demeaned ranks: patch-to-patch, inside one
                    scene. This is "depth tracks patch complexity" in the
                    narrow sense; the pooled number can be carried entirely by
                    scenes that are uniformly deep or shallow.
    Depth is attributed to the INPUT token, i.e. the token that encodes the
    patch (Step 8 convention).
    """
    from eo.data.eo_vocab import MODALITY_TO_ID
    depth, im = tok["depth"], tok["input_mod"]
    out = {}
    for mod, keys in PATCH_STATS.items():
        f = Path(stats_dir) / f"{mod}.npz"
        if not f.exists():
            continue
        S = np.load(f)
        if not np.array_equal(S["rows"], eval_rows):
            raise RuntimeError(f"{mod}: patch-stat rows are not in the eval loader's order")
        mid = MODALITY_TO_ID[mod]
        sel = im == mid
        rows = np.flatnonzero(sel.sum(1) == 196)
        dep = np.stack([depth[r][sel[r]] for r in rows]).astype(np.float64)       # (n, 196)
        res = {"n_scenes": int(len(rows))}
        for k in keys:
            st = S[k][rows].astype(np.float64)
            ok = np.isfinite(st).all(1)
            d, s = dep[ok], st[ok]
            rd = _rank(d.ravel()).reshape(d.shape); rs = _rank(s.ravel()).reshape(s.shape)
            pooled = float(np.corrcoef(rd.ravel(), rs.ravel())[0, 1]) if d.std() > 0 else None
            wd, ws = rd - rd.mean(1, keepdims=True), rs - rs.mean(1, keepdims=True)
            within = (float((wd * ws).sum() / np.sqrt((wd ** 2).sum() * (ws ** 2).sum()))
                      if (wd ** 2).sum() > 0 and (ws ** 2).sum() > 0 else None)
            q = np.quantile(s, [0, .25, .5, .75, 1])
            qb = np.clip(np.searchsorted(q[1:-1], s, side="right"), 0, 3)
            res[k] = {"pooled_spearman": pooled, "within_scene": within,
                      "mean_depth_by_quartile": [float(d[qb == i].mean()) if (qb == i).any() else None
                                                 for i in range(4)]}
        if mod == "LULC":
            nc = S["n_classes"][rows]
            res["mean_depth_by_n_classes"] = {
                str(k): {"mean_depth": float(dep[(nc == k) if k < 4 else (nc >= 4)].mean()),
                         "n": int(((nc == k) if k < 4 else (nc >= 4)).sum())}
                for k in (1, 2, 3, 4) if ((nc == k) if k < 4 else (nc >= 4)).any()}
            mc = S["majority_class"][rows].astype(np.int64)
            res["mean_depth_by_majority_class"] = {
                int(c): {"mean_depth": float(dep[mc == c].mean()), "n": int((mc == c).sum())}
                for c in np.unique(mc) if (mc == c).sum() >= 200}
        out[mod] = res
    return out
