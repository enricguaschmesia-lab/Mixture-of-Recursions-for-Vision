#!/usr/bin/env python
"""Paper B's convergence criteria as pass or fail, with criterion 4 as a signal, for one arm of runs (B1).

The criteria were fixed on 2026-11-05 (direction note, "What counts as convergence"); the meeting of 11 Nov ratified
how they are read (worklog §5, item 56): time means over B0's four windows, criterion 4 by the regret test on five
slots in both frames, Coords out, S1 as one slot. This script applies them to the per-update depth shares every
learned-router run writes since wave 1 (`depth_shares.jsonl`, util/trainer_pt.py, gate W31), so the factorial's arms
(Paper B plan, B2) are all read by one rule. Windows, in update steps, ends included: early 330-990, plateau 990-2970,
late 1980-3300, anneal 2970-3300. A time mean is the mean over updates of each class's shares (b3_depth_shares.py).

SUCCESS (changed 2026-11-16, Enric; worklog §5): C1, C2 and C3 pass, and C5 passes or is n/a. C4, and every depth
direction predicted before a run, are SIGNALS: read by their rules and reported beside the verdict, never part of it.
An allocation that reproduces and beats its comparisons is a success even if it runs against the gain. Until then the
verdict was all five, C4 included; the JSON keeps that one as passes_all_rule_2026_11_15, so an arm reads both ways.

  C1 REPRODUCIBLE  per pair of seeds, in the plateau and in the late window: Spearman of the six image modalities'
                   mean depths >= 0.9 and every modality within 0.3 passes (B0 Step 4's rule). Passes if every pair
                   passes in both windows AND the arm has at least 3 seeds and 2 router forms (the direction note);
                   with fewer it can at most be a "screen pass". Five slots (S1 pooled) and the mean |difference|
                   against its 720 relabellings are reported beside (b3_seed_tables.py, B0 Step 4).
  C2 STABLE        per run and image modality, the difference between its 2970-3300 and 1980-2640 time means, with
                   a 95% interval, is inside +-0.3 passes (stable), beyond it (unstable), or across it
                   (undetermined) (B0 Step 4's rule, "within +-0.3 passes from 60% of training through the anneal").
                   Consecutive updates are not independent (a tie swings every 3-21 updates), so the interval is by
                   batch means: blocks of 30 updates (22 in 1980-2640, 11 in 2970-3300), each window's standard error
                   from its blocks' spread, Welch's degrees of freedom and Student's t (b0_agree.t_quantile), exactly
                   as b3_criteria.py fixed it (2026-11-15, 12:28), which read arm A first; the two agree on arm A.
                   Passes if every modality of every run is stable.
  C3 WORTH IT      after retraining at equal compute the allocation beats its permuted tables, the other seed's and
                   a histogram-matched random router, each by more than the cell's seed gap. Since 2026-11-16 it
                   must also beat fixed depth (uniform 2) and the matched vanilla (B10c-5L-tr-NC in the cell), each
                   by more than the cell's seed gap too (Enric). Needs B1's transplants or retrains (open
                   decision 2); not readable from depth shares, so "not measured" until they exist.
  C4 MEANINGFUL    A SIGNAL since 2026-11-16, never part of the verdict.
                   per run, the regret of its time-mean allocation (five slots, S1 pooled by held-out token share)
                   at its own load against the transport optimum, with p over its 120 slot relabellings
                   (b0_criterion4.regret_test); follows the gain if p <= 0.05 in the plateau and late windows.
                   The cost table is the cell's random-router CE at depths 1-3 per slot: target only (primary; mean
                   of the five late checkpoints, tforce units) and all forced (beside; the cell has 1980 and 3300).
                   *Proposed 2026-11-15, before any arm's C4 was read:* the primary table is the mean of the cell's
                   two random-router seeds' tables (both scored on the same held-out rows), each seed's verdict
                   reported beside. Reads "follows the gain" if every run follows it in the primary frame, and
                   "does not follow the gain" otherwise.
  C5 DISCOVERED    a method that injects anything per modality must return to the same allocation from a permuted
                   start and reach the same conclusion when its knob moves 2-4x: read with C1's pair rule between
                   those runs and the arm's seeds. An arm that injects nothing per modality (the balancing loss, the
                   compute target, a random-router warm-up) reads "n/a", which does not block.

CONTROLS, which must hold or nothing is written (exit 1): a run against itself passes C1, and the second seed with its
six image labels cyclically shifted fails it; a synthetic constant run reads stable on C2 and one with a modality
moved 0.6 passes between the windows reads it unstable; at every run's load and window the gain-optimal allocation
passes C4 and its worst relabelling fails it. On arm A (the Coords-free 1 + 3 + 1 train cell's learned seeds) the
time means must equal b3_depth_shares' stored readings, and C1's numbers b3_seed_tables' (2026-11-13).

    .venv/bin/python eo/experiments/routing/paper_b_criteria.py --self-test
    .venv/bin/python eo/experiments/routing/paper_b_criteria.py read A
    .venv/bin/python eo/experiments/routing/paper_b_criteria.py read C990     (arm C: from its switch on; C1980 too)

Writes /data/enric/reports/b1/criteria/<arm>.{json,txt}. Runs in `.venv`, CPU, seconds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import b3_depth_shares as DS  # noqa: E402
import b3_seed_tables as BT  # noqa: E402
from b0_agree import check_t, t_quantile  # noqa: E402
from b0_criterion4 import regret_test, transport_min  # noqa: E402
from eo.routing import stats as S  # noqa: E402

OUT = Path("/data/enric/reports/b1/criteria")
RUNS = Path("/data/enric/runs/pretrain/phase3")
LAND = Path("/data/enric/reports/wave1/landscape")
STORED = Path("/data/enric/reports/wave1/depth_shares")
IMG = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
SLOTS = ["S2L2A", "S1", "DEM", "NDVI", "LULC"]
WINDOWS = {"early": (330, 990), "plateau": (990, 2970), "late": (1980, 3300), "anneal": (2970, 3300)}
C1_WINDOWS = ("plateau", "late")
C1_RHO, C1_BAND = 0.9, 0.3
C2_A, C2_B, C2_BLOCK, C2_BAND = (1980, 2640), (2970, 3300), 30, 0.3
C4_WINDOWS, C4_P = ("plateau", "late"), 0.05
LATE5 = (1980, 2310, 2640, 2970, 3300)
AF_CKS = (1980, 3300)

#: The cells' random-router measurements (B0 Step 6's, per wave 1 Step 4): landscape folder and unit prefix per seed.
CELLS = {
    "b3trnc": {"label": "Coords-free 1 + 3 + 1 train cell",
               "seeds": {"a10crb3trnc": LAND / "a10crb3trnc", "a10crb3trs2nc": LAND / "a10crb3trs2nc"}},
}
#: The arms (Paper B plan, B2): run tag -> run directory, and what the arm injects. Factorial arms are added as they land.
ARMS = {
    "A": {"label": "arm A: the learned router with the balancing loss (A10c-b3-tr-NC, seeds 42 and 43)",
          "cell": "b3trnc", "router_forms": 1, "injects_per_modality": False,
          "runs": {"a10cb3trnc": RUNS / "mor10cb3trnc_20261112_101703",
                   "a10cb3trs2nc": RUNS / "mor10cb3trs2nc_20261113_072043"},
          "stored": True},
    # Arm C (intervention 4, 2026-11-16): a random-router checkpoint continued exactly under arm A's learned router
    # (eo/train/warm_start.py). Its router exists only from the switch on, so the windows before it are left out, and
    # the windows read are B0's by their steps, from the switch: at 990 the plateau starts with the router's first
    # update; at 1980 the plateau holds only its first 990 updates and C2's first window (1980-2640) starts at the
    # switch, as flagged for the meeting (worklog §4, item 69).
    "C990": {"label": "arm C at 990: A10cR-b3-tr-NC's checkpoint-990 continued under a learned router (seeds 42, 43)",
             "cell": "b3trnc", "router_forms": 1, "injects_per_modality": False, "switch": 990,
             "windows": ("plateau", "late", "anneal"),
             "runs": {"c990s42": RUNS / "mor10cwb3trnc990_20261116_175333",
                      "c990s43": RUNS / "mor10cwb3trs2nc990_20261116_175333"}},
    "C1980": {"label": "arm C at 1980: A10cR-b3-tr-NC's checkpoint-1980 continued under a learned router (seeds 42, 43)",
              "cell": "b3trnc", "router_forms": 1, "injects_per_modality": False, "switch": 1980,
              "windows": ("plateau", "late", "anneal"),
              "runs": {"c1980s42": RUNS / "mor10cwb3trnc1980_20261116_175333",
                       "c1980s43": RUNS / "mor10cwb3trs2nc1980_20261116_175333"}},
}


# ------------------------------------------------------------------------------------------------ time means

def load_run(run_dir: Path):
    classes, steps, counts = DS.load(Path(run_dir) / "depth_shares.jsonl")
    return classes, steps, counts


def mixes(classes, steps, counts, a, b):
    """{modality: time-mean shares (3,)} over updates a..b, as b3_depth_shares reads them."""
    sel = (steps >= a) & (steps <= b)
    out = {}
    for m in IMG:
        c = counts[sel, classes.index(m), :]
        tot = c.sum(1)
        ok = tot > 0
        if ok.sum():
            out[m] = (c[ok] / tot[ok, None]).mean(0)
    return out


def depth(mix) -> float:
    return float((np.asarray(mix) * np.arange(1, len(mix) + 1)).sum())


def five(mx: dict, w_grd: float) -> np.ndarray:
    """(5, 3) slot mixes, S1 pooled by held-out token share (b0_criterion4.pool_s1)."""
    return np.stack([mx["S2L2A"], w_grd * mx["S1GRD"] + (1 - w_grd) * mx["S1RTC"], mx["DEM"], mx["NDVI"], mx["LULC"]])


def per_update_depth(classes, steps, counts, m, a, b):
    sel = (steps >= a) & (steps <= b)
    c = counts[sel, classes.index(m), :]
    tot = c.sum(1)
    ok = tot > 0
    return (c[ok] / tot[ok, None] * np.arange(1, c.shape[1] + 1)).sum(1)


# ------------------------------------------------------------------------------------------------ the criteria

def c1_pair(da: dict, db: dict, w_grd: float, windows=tuple(WINDOWS)) -> dict:
    """C1 for one pair: per window, rho over the six image modalities with its exact p, max and mean |d|, verdict."""
    out = {}
    for w in windows:
        a = [depth(da[w][m]) for m in IMG]
        b = [depth(db[w][m]) for m in IMG]
        mad, pm = BT.meanabs(a, b)
        a5 = [depth(x) for x in five(da[w], w_grd)]
        b5 = [depth(x) for x in five(db[w], w_grd)]
        rho = S.spearman(a, b)
        mx = float(np.max(np.abs(np.subtract(a, b))))
        out[w] = {"rho": rho, "p": S.exact_p(a, b), "max_abs": mx, "mean_abs": mad, "p_mean_abs": pm,
                  "passes": bool(rho >= C1_RHO - 1e-12 and mx <= C1_BAND + 1e-12),
                  "rho5": S.spearman(a5, b5), "max_abs5": float(np.max(np.abs(np.subtract(a5, b5)))),
                  "depth_a": a, "depth_b": b}
    out["passes"] = all(out[w]["passes"] for w in C1_WINDOWS)
    return out


def c2_run(classes, steps, counts) -> dict:
    out = {}
    for m in IMG:
        xa = per_update_depth(classes, steps, counts, m, *C2_A)
        xb = per_update_depth(classes, steps, counts, m, *C2_B)
        na, nb = len(xa) // C2_BLOCK, len(xb) // C2_BLOCK
        if na < 2 or nb < 2:
            out[m] = {"state": "undetermined", "why": f"too few blocks ({na}, {nb})"}
            continue
        ba = xa[:na * C2_BLOCK].reshape(na, C2_BLOCK).mean(1)
        bb = xb[:nb * C2_BLOCK].reshape(nb, C2_BLOCK).mean(1)
        diff = float(xb.mean() - xa.mean())
        sa, sb = ba.std(ddof=1) / math.sqrt(na), bb.std(ddof=1) / math.sqrt(nb)
        se = math.sqrt(sa * sa + sb * sb)
        if se == 0:
            df, h = float("inf"), 0.0
        else:
            df = se ** 4 / ((sa ** 4) / (na - 1) + (sb ** 4) / (nb - 1)) if sa > 0 or sb > 0 else float("inf")
            h = t_quantile(0.975, df) * se
        lo, hi = diff - h, diff + h
        state = ("stable" if -C2_BAND < lo and hi < C2_BAND else
                 "unstable" if lo > C2_BAND or hi < -C2_BAND else "undetermined")
        out[m] = {"diff": diff, "lo": lo, "hi": hi, "blocks": [na, nb], "df": df, "state": state,
                  "depth_a": float(xa.mean()), "depth_b": float(xb.mean())}
    states = [out[m]["state"] for m in IMG]
    out["verdict"] = ("stable" if all(s == "stable" for s in states) else
                      "unstable" if "unstable" in states else "undetermined")
    return out


def cell_costs(cell: str):
    """{seed: {"target_only": C (5, 3), "all_forced": C (5, 3)}}, the mean table, w_grd, w_slot."""
    out, w_grd, w_slot, rows_ref = {}, None, None, None
    for tag, d in CELLS[cell]["seeds"].items():
        A = dict(np.load(d / "perrow.npz"))
        n_tok = A["cnt"].sum(0)
        wg = float(n_tok[1] / (n_tok[1] + n_tok[2]))
        ws = np.array([n_tok[0], n_tok[1] + n_tok[2], n_tok[3], n_tok[4], n_tok[5]], dtype=np.float64)
        ws = ws / ws.sum()
        if w_grd is None:
            w_grd, w_slot = wg, ws
        elif abs(wg - w_grd) > 1e-12 or not np.allclose(ws, w_slot):
            raise SystemExit(f"{cell}: the seeds' held-out token shares differ ({wg} vs {w_grd})")
        to = []
        for dep in (1, 2, 3):
            per_ck = []
            for ck in LATE5:
                z = np.load(d / f"{tag}_tforce{dep}" / f"checkpoint-{ck}" / "target_last_rows.npz")
                rows = {T: z[f"{T}|rows"] for T in IMG}
                if rows_ref is None:
                    rows_ref = rows
                if not all(np.array_equal(rows[T], rows_ref[T]) for T in IMG):
                    raise SystemExit(f"{tag} tforce{dep}@{ck}: another target-last sample than the reference")
                ce = {T: float(z[f"{T}|ce_sum"].sum() / z[f"{T}|n"].sum()) for T in IMG}
                per_ck.append([ce["S2L2A"], wg * ce["S1GRD"] + (1 - wg) * ce["S1RTC"], ce["DEM"], ce["NDVI"], ce["LULC"]])
            to.append(np.mean(per_ck, 0))
        af = []
        cnt = A["cnt"]
        for dep in (1, 2, 3):
            per_ck = []
            for ck in AF_CKS:
                x = A[f"{tag}|f{dep}|{ck}"]
                per_ck.append([x[:, 0].sum() / cnt[:, 0].sum(), x[:, 1:3].sum() / cnt[:, 1:3].sum(),
                               x[:, 3].sum() / cnt[:, 3].sum(), x[:, 4].sum() / cnt[:, 4].sum(),
                               x[:, 5].sum() / cnt[:, 5].sum()])
            af.append(np.mean(per_ck, 0))
        out[tag] = {"target_only": np.array(to).T, "all_forced": np.array(af).T}
    mean = {f: np.mean([v[f] for v in out.values()], 0) for f in ("target_only", "all_forced")}
    return out, mean, w_grd, w_slot


def c4_run(dm: dict, costs: dict, mean: dict, w_grd: float, w_slot: np.ndarray, windows=tuple(WINDOWS)) -> dict:
    out = {}
    for w in windows:
        mix = five(dm[w], w_grd)
        r = {"depth": [depth(x) for x in mix]}
        for frame in ("target_only", "all_forced"):
            t = regret_test(mix, w_slot, mean[frame])
            r[frame] = {"regret": t["R"], "p": t["p"], "optimum_depth": t["dstar"].tolist(),
                        "per_seed_p": {s: regret_test(mix, w_slot, c[frame])["p"] for s, c in costs.items()}}
        out[w] = r
    out["follows"] = all(out[w]["target_only"]["p"] <= C4_P + 1e-12 for w in C4_WINDOWS)
    out["follows_all_forced"] = all(out[w]["all_forced"]["p"] <= C4_P + 1e-12 for w in C4_WINDOWS)
    return out


def c4_controls(dm: dict, mean: dict, w_grd: float, w_slot: np.ndarray) -> bool:
    """At this run's load in each rule window and frame: the gain-optimal allocation passes, its worst relabelling fails."""
    ok = True
    for w in C4_WINDOWS:
        load = w_slot @ five(dm[w], w_grd)
        for frame in ("target_only", "all_forced"):
            C = mean[frame]
            _, plan = transport_min(w_slot, load, C)
            opt = np.clip(plan / w_slot[:, None], 0, None)
            p_opt = regret_test(opt, w_slot, C)["p"]
            worst = max((opt[list(pi)] for pi in itertools.permutations(range(5))),
                        key=lambda X: float((w_slot[:, None] * X * C).sum()))
            p_worst = regret_test(worst, w_slot, C)["p"]
            ok &= p_opt <= C4_P + 1e-12 and p_worst > C4_P
    return bool(ok)


# ------------------------------------------------------------------------------------------------ read an arm

def read_arm(name: str) -> int:
    check_t()
    arm = ARMS[name]
    costs, mean, w_grd, w_slot = cell_costs(arm["cell"])
    WIN = {w: WINDOWS[w] for w in arm.get("windows", WINDOWS)}     # an arm whose router starts late reads fewer
    if not set(C1_WINDOWS) | set(C4_WINDOWS) <= set(WIN):
        raise SystemExit(f"arm {name}: its windows {list(WIN)} leave out a rule window")
    data, dm = {}, {}
    for tag, d in arm["runs"].items():
        if not (Path(d) / "depth_shares.jsonl").is_file():     # an arm read while its later seeds still train
            print(f"  {tag}: no {d}/depth_shares.jsonl yet; read without it")
            continue
        data[tag] = load_run(d)
        if "switch" in arm and int(data[tag][1][0]) != arm["switch"] + 1:
            raise SystemExit(f"{tag}: its depth shares start at {data[tag][1][0]}, not at switch {arm['switch']} + 1")
        dm[tag] = {w: mixes(*data[tag], *ab) for w, ab in WIN.items()}
    tags = list(dm)
    if not tags:
        raise SystemExit(f"arm {name}: no run has depth shares yet")
    L = [f"Paper B criteria, {arm['label']}; cell: {CELLS[arm['cell']]['label']}",
         f"  runs: " + ", ".join(f"{t} ({arm['runs'][t].name})" for t in tags),
         f"  slots: S1 pooled by held-out token share (S1GRD {w_grd:.4f}); slot weights {np.round(w_slot, 4).tolist()}"]
    if "switch" in arm:
        L.append(f"  router from the switch at {arm['switch']}: windows " + ", ".join(
            f"{w} {max(a, arm['switch'] + 1)}-{b}" for w, (a, b) in WIN.items()))

    # ---- controls ----
    ctl = {}
    selfp = c1_pair(dm[tags[0]], dm[tags[0]], w_grd, WIN)
    ctl["c1_self_passes"] = bool(selfp["passes"] and all(abs(selfp[w]["rho"] - 1) < 1e-12 and selfp[w]["max_abs"] == 0
                                                         for w in C1_WINDOWS))
    if len(tags) > 1:
        shifted = {w: {m: dm[tags[1]][w][IMG[(i + 1) % 6]] for i, m in enumerate(IMG)} for w in WIN}
        ctl["c1_shuffled_fails"] = not c1_pair(dm[tags[0]], shifted, w_grd, WIN)["passes"]
    ctl["c2_synthetic"] = c2_synthetic_ok()
    ctl["c4_optimum_passes_worst_fails"] = all(c4_controls(dm[t], mean, w_grd, w_slot) for t in tags)
    if arm.get("stored"):
        ctl["reproduces_stored_time_means"] = all(reproduces_stored(t, dm[t]) for t in tags)
    L.append("  controls: " + ", ".join(f"{k} {'holds' if v else 'FAILS'}" for k, v in ctl.items()))

    # ---- C1 ----
    pairs = {f"{a}|{b}": c1_pair(dm[a], dm[b], w_grd, WIN) for a, b in itertools.combinations(tags, 2)}
    if arm.get("stored") and len(tags) == 2:
        ctl["reproduces_b3_seed_tables"] = reproduces_seed_tables(tags, pairs[f"{tags[0]}|{tags[1]}"])
        L.append(f"  control: C1's numbers equal b3_seed_tables' stored reading: "
                 f"{'holds' if ctl['reproduces_b3_seed_tables'] else 'FAILS'}")
    n_seeds, forms = len(tags), int(arm["router_forms"])
    all_pairs = bool(pairs) and all(p["passes"] for p in pairs.values())
    c1 = ("not measured (one seed)" if n_seeds < 2 else
          "pass" if all_pairs and n_seeds >= 3 and forms >= 2 else
          f"screen pass ({n_seeds} seeds, {forms} router form{'s' if forms > 1 else ''}; the criterion asks 3 and 2)"
          if all_pairs else "fail")
    L.append(f"\n  C1 reproducible: {c1.upper()}  (rule: rho >= {C1_RHO} and every |d| <= {C1_BAND}, six image "
             f"modalities, in {' and '.join(C1_WINDOWS)})")
    for k, p in pairs.items():
        for w in WIN:
            r = p[w]
            L.append(f"    {k:28s} {w:8s} rho {r['rho']:+.2f} (p {r['p']:.3f})  max|d| {r['max_abs']:.3f}  mean|d| "
                     f"{r['mean_abs']:.3f} (p {r['p_mean_abs']:.3f})  five slots: rho {r['rho5']:+.2f} max|d| "
                     f"{r['max_abs5']:.3f}" + (f"  -> {'passes' if r['passes'] else 'fails'}" if w in C1_WINDOWS else ""))
        L.append("      depths, plateau: " + "  ".join(f"{m} {a:.2f}/{b:.2f}" for m, a, b in
                                                   zip(IMG, p["plateau"]["depth_a"], p["plateau"]["depth_b"])))

    # ---- C2 ----
    c2r = {t: c2_run(*data[t]) for t in tags}
    c2 = ("pass" if all(r["verdict"] == "stable" for r in c2r.values()) else
          "fail" if any(r["verdict"] == "unstable" for r in c2r.values()) else "undetermined")
    L.append(f"\n  C2 stable: {c2.upper()}  (rule: 2970-3300 minus 1980-2640 time-mean depth, 95% interval by batch "
             f"means of {C2_BLOCK} updates, inside +-{C2_BAND})")
    for t, r in c2r.items():
        L.append(f"    {t:14s} {r['verdict']:12s} " + "  ".join(
            f"{m} {r[m]['diff']:+.3f} [{r[m]['lo']:+.3f}, {r[m]['hi']:+.3f}] {r[m]['state'][0].upper()}" for m in IMG))

    # ---- C3 ----
    c3 = "not measured"
    L.append(f"\n  C3 worth it after retraining: NOT MEASURED (needs B1's transplants or retrains in the cell; open "
             f"decision 2)")

    # ---- C4 ----
    c4r = {t: c4_run(dm[t], costs, mean, w_grd, w_slot, WIN) for t in tags}
    c4_follows = all(r["follows"] for r in c4r.values())
    c4 = "follows the gain" if c4_follows else "does not follow the gain"
    L.append(f"\n  C4 meaningful, a SIGNAL (never part of the verdict since 2026-11-16): {c4.upper()}  (rule: regret "
             f"at the run's own load, p <= {C4_P} over 120 slot relabellings in {' and '.join(C4_WINDOWS)}; target "
             f"only primary on the mean of the cell's seeds' tables)")
    for frame in ("target_only", "all_forced"):
        C = mean[frame]
        L.append(f"    cell table, {frame.replace('_', ' ')}: CE(1) - CE(2) " + " ".join(
            f"{s} {C[k, 0] - C[k, 1]:+.4f}" for k, s in enumerate(SLOTS)) + "; CE(3) - CE(2) " + " ".join(
            f"{s} {C[k, 2] - C[k, 1]:+.4f}" for k, s in enumerate(SLOTS)))
    for t, r in c4r.items():
        for w in WIN:
            x = r[w]
            L.append(f"    {t:14s} {w:8s} depths " + " ".join(f"{s} {d:.2f}" for s, d in zip(SLOTS, x["depth"]))
                     + f" | target only: regret {x['target_only']['regret']:.4f} p {x['target_only']['p']:.3f} "
                     f"(seeds {', '.join(f'{v:.3f}' for v in x['target_only']['per_seed_p'].values())}), optimum "
                     + " ".join(f"{d:.2f}" for d in x["target_only"]["optimum_depth"])
                     + f" | all forced: p {x['all_forced']['p']:.3f}")
        L.append(f"    {t:14s} follows the gain: target only {r['follows']}, all forced {r['follows_all_forced']}")

    # ---- C5 ----
    c5 = "n/a" if not arm["injects_per_modality"] else "not measured"
    L.append(f"\n  C5 discovered, not imposed: {c5.upper()}"
             + (" (the arm injects nothing per modality)" if c5 == "n/a" else
                " (needs a permuted-start run and a knob run 2-4x, read with C1's pair rule)"))

    verdicts = {"C1": c1, "C2": c2, "C3": c3, "C5": c5}
    signals = {"C4": c4}
    success = all(verdicts[k] == "pass" for k in ("C1", "C2", "C3")) and c5 in ("pass", "n/a")
    old_rule = success and c4_follows          # 2026-11-15's verdict: all five, with C4 a condition
    L.append(f"\n  VERDICTS (success needs C1, C2 and C3, and C5 where it applies): {verdicts}; success: {success}")
    L.append(f"  SIGNALS (read and reported, never a condition): {signals}")
    L.append(f"  under the rule of 2026-11-15 (C4 a condition): passes all five: {old_rule}")
    text = "\n".join(L)
    print(text)
    if not all(ctl.values()):
        print("A CONTROL FAILED: nothing written")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.txt").write_text(text + "\n")
    (OUT / f"{name}.json").write_text(json.dumps({
        "arm": name, "label": arm["label"], "cell": arm["cell"], "runs": {k: str(v) for k, v in arm["runs"].items()},
        "controls": ctl, "verdicts": verdicts, "signals": signals, "success": success,
        "passes_all_rule_2026_11_15": old_rule,
        "c1": pairs, "c2": c2r, "c4": c4r, "w_grd": w_grd, "w_slot": w_slot.tolist(),
        "cell_tables": {s: {f: v.tolist() for f, v in c.items()} for s, c in costs.items()},
        "time_means": {t: {w: {m: dm[t][w][m].tolist() for m in IMG} for w in WIN} for t in tags},
        "windows": {w: list(ab) for w, ab in WIN.items()}, "switch": arm.get("switch"), "runs_read": tags,
    }, indent=1, default=float))
    return 0


def reproduces_stored(tag: str, dmt: dict) -> bool:
    """The time means equal b3_depth_shares' stored reading of the same file (its shares are rounded to 4 decimals)."""
    f = STORED / f"{tag}.json"
    if not f.is_file():
        return False
    st = json.loads(f.read_text())["windows"]
    return all(abs(depth(dmt[w][m]) - st[w]["mods"][m]["mean_depth"]) < 1e-9
               and np.allclose(np.round(dmt[w][m], 4), st[w]["mods"][m]["shares"], atol=1e-12)
               for w in WINDOWS for m in IMG)


def reproduces_seed_tables(tags, pair: dict) -> bool:
    f = STORED / f"seeds_{tags[0]}_vs_{tags[1]}.json"
    if not f.is_file():
        return False
    st = json.loads(f.read_text())["result"]["windows"]
    return all(abs(pair[w]["rho"] - st[w]["rho"]) < 1e-12 and abs(pair[w]["p"] - st[w]["p"]) < 1e-12
               and abs(pair[w]["mean_abs"] - st[w]["meanabs"]) < 1e-12 for w in WINDOWS)


# ------------------------------------------------------------------------------------------------ self-test

TIED = {"S2L2A": [0.9, 0.0, 0.1], "S1GRD": [0.0, 0.7, 0.3], "S1RTC": [0.0, 0.7, 0.3],
        "DEM": [0.2, 0.4, 0.4], "NDVI": [0.0, 0.7, 0.3], "LULC": [0.8, 0.0, 0.2]}      # arm A's shape: three tied at 2.3
DISTINCT = {"S2L2A": [0.9, 0.0, 0.1], "S1GRD": [0.0, 0.4, 0.6], "S1RTC": [0.0, 0.7, 0.3],
            "DEM": [0.2, 0.6, 0.2], "NDVI": [0.0, 0.2, 0.8], "LULC": [0.6, 0.2, 0.2]}  # 1.2, 2.6, 2.3, 2.0, 2.8, 1.6


def _synthetic_run(rng, drift=None, n=3300, per=980, base=None):
    classes = ["none"] + IMG + ["Coords", "delim", "pad"]
    base = TIED if base is None else base
    counts = np.zeros((n, len(classes), 3))
    for t in range(n):
        for m, p in base.items():
            p = np.array(p)
            if drift and m == drift[0] and t + 1 >= drift[1]:
                p = np.array(drift[2])
            counts[t, classes.index(m)] = rng.multinomial(per, p / p.sum())
        counts[t, classes.index("delim")] = [0, 0, 24]
    return classes, np.arange(1, n + 1), counts


def c2_synthetic_ok() -> bool:
    rng = np.random.default_rng(1)
    flat = c2_run(*_synthetic_run(rng))
    moved = c2_run(*_synthetic_run(rng, drift=("NDVI", 2970, [0.0, 0.1, 0.9])))     # 2.30 -> 2.90
    return bool(flat["verdict"] == "stable" and moved["NDVI"]["state"] == "unstable"
                and all(moved[m]["state"] == "stable" for m in IMG if m != "NDVI"))


def self_test() -> int:
    check_t()
    rng = np.random.default_rng(0)
    ok = True

    def chk(label, c):
        nonlocal ok
        ok &= bool(c)
        print(f"  self-test: {label}: {'PASS' if c else 'FAIL'}")

    a = _synthetic_run(rng, base=DISTINCT)
    b = _synthetic_run(rng, base=DISTINCT)
    da = {w: mixes(*a, *ab) for w, ab in WINDOWS.items()}
    db = {w: mixes(*b, *ab) for w, ab in WINDOWS.items()}
    chk("two draws of one table with distinct depths pass C1", c1_pair(da, db, 0.1)["passes"])
    shifted = {w: {m: db[w][IMG[(i + 1) % 6]] for i, m in enumerate(IMG)} for w in WINDOWS}
    chk("the second with its labels shifted fails C1", not c1_pair(da, shifted, 0.1)["passes"])
    # A property of C1 as fixed, not a fault: three modalities tied at one depth (arm A's shape) make their rank order
    # noise, so two draws of ONE table often fail the rho part (a random order of three passes rho >= 0.9 only when it
    # matches or differs by one adjacent swap) while every |d| stays far inside 0.3.
    fails, small = 0, True
    for _ in range(20):
        ta = {w: mixes(*_synthetic_run(rng), *ab) for w, ab in WINDOWS.items()}
        tb = {w: mixes(*_synthetic_run(rng), *ab) for w, ab in WINDOWS.items()}
        tp = c1_pair(ta, tb, 0.1)
        fails += not tp["passes"]
        small &= all(tp[w]["max_abs"] < 0.05 for w in C1_WINDOWS)
    chk(f"C1 as fixed: two draws of a table with three tied modalities fail in {fails} of 20 pairs, every |d| < 0.05 "
        f"(a known limit; expected about half or more)", fails >= 6 and small)
    chk("C2: a constant run is stable, a modality moved 0.6 passes is unstable", c2_synthetic_ok())
    C = np.array([[3.06, 3.00, 3.002], [5.03, 5.00, 5.001], [1.02, 1.00, 1.001], [2.04, 2.00, 2.003], [4.035, 4.00, 4.002]])
    w = np.full(5, 0.2)
    chk("C4 controls on a synthetic table", c4_controls(da, {"target_only": C, "all_forced": C}, 0.1, w))
    gain = np.stack([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]).astype(float)    # DEM (smallest first) at 1
    # A property of C4 as fixed: DEM at depth 1 and four slots at 2 IS the optimum at its load (regret 0), but 24 of
    # the 120 relabellings give the same allocation, so p cannot go below 24/120 = 0.2 (a 1-in-5 choice).
    t = regret_test(gain, w, C)
    chk("C4 as fixed: the optimal one-slot table reads regret 0 and p = 0.2, the test's floor (a known limit)",
        abs(t["R"]) < 1e-12 and abs(t["p"] - 0.2) < 1e-12)
    chk("C4: S2L2A at depth 1 instead (the largest first pass) reads p = 1", abs(regret_test(gain[[2, 1, 0, 3, 4]], w, C)["p"] - 1) < 1e-12)
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["read"])
    ap.add_argument("arm", nargs="?", choices=sorted(ARMS))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd != "read" or not a.arm:
        ap.error("read ARM, or --self-test")
    return read_arm(a.arm)


if __name__ == "__main__":
    sys.exit(main())
