#!/usr/bin/env python
"""B3 (wave 1): do two seeds of a learned router learn one depth table? Their per-update depth shares side by side.

Reads two runs' stored per-update depth-share readings (`b3_depth_shares.py`:
/data/enric/reports/wave1/depth_shares/<tag>.json) and sets each image modality's time-mean depth and state side by side
in B0's windows (early 330-990, plateau 990-2970, late 1980-3300, anneal 2970-3300). Agreement is B0 Step 4's
(`b0_agree.py`): Spearman's rho of the six mean depths with its exact two-sided p over every relabelling of the
modalities (`eo/routing/stats.exact_p`; at n = 6 the 5% critical rho is 0.886), the mean |difference| against the same
null (the fraction of relabellings that come at least as close), and one rho over the windows with one relabelling
applied at every window (`stats.series_p`).

Read the rho with care: modalities tied between the same two depths sit within ~0.1 of each other, so their order is
noise, and the rank correlation is capped well below 1 for tables that match. The mean |difference| is the test that
does not depend on that order.

CONTROLS, which must hold or the script writes nothing and exits 1: a run against itself reads rho 1 and mean
|difference| 0; a random-router run (`--control`, which has no table: every modality split, near 1/3 per depth) against
the first run must NOT agree (the series p and every window's mean-|difference| p above 0.05). Passing a learned run as
`--control` must fail it (A10c-tr against A10c-b3-tr-NC agrees, series p 0.008: checked 2026-11-13).

    .venv/bin/python eo/experiments/routing/b3_seed_tables.py a10cb3trnc a10cb3trs2nc --control a10crb3trnc

Writes /data/enric/reports/wave1/depth_shares/seeds_<a>_vs_<b>.{txt,json}. Runs in `.venv`, CPU, seconds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402

D = Path("/data/enric/reports/wave1/depth_shares")
IMG = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
WINDOWS = ("early", "plateau", "late", "anneal")
PERMS6 = np.array(list(itertools.permutations(range(6))))


def meanabs(a, b) -> tuple[float, float]:
    """b0_agree.meanabs: the mean |difference| and the fraction of relabellings of b at least as close."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    obs = float(np.abs(a - b).mean())
    null = np.abs(a[None, :] - b[PERMS6]).mean(1)
    return obs, float(np.mean(null <= obs + 1e-12))


def load(tag: str) -> dict:
    return json.loads((D / f"{tag}.json").read_text())


def depths(d: dict, w: str) -> list[float]:
    return [d["windows"][w]["mods"][m]["mean_depth"] for m in IMG]


def compare(a: dict, b: dict) -> dict:
    out = {"windows": {}}
    for w in WINDOWS:
        da, db = depths(a, w), depths(b, w)
        mad, pm = meanabs(da, db)
        out["windows"][w] = {"rho": S.spearman(da, db), "p": S.exact_p(da, db), "meanabs": mad, "p_meanabs": pm,
                             "depth_a": da, "depth_b": db,
                             "state_a": [a["windows"][w]["mods"][m]["state"] for m in IMG],
                             "state_b": [b["windows"][w]["mods"][m]["state"] for m in IMG],
                             "shares_a": [a["windows"][w]["mods"][m]["shares"] for m in IMG],
                             "shares_b": [b["windows"][w]["mods"][m]["shares"] for m in IMG]}
    A = np.array([depths(a, w) for w in WINDOWS])
    B = np.array([depths(b, w) for w in WINDOWS])
    out["series"] = dict(zip(("rho", "p"), S.series_p(A, B)))
    out["series_early_plateau_late"] = dict(zip(("rho", "p"), S.series_p(A[:3], B[:3])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--control", required=True, help="a random-router run's tag: it must not agree with a")
    args = ap.parse_args()
    a, b, c = load(args.a), load(args.b), load(args.control)
    lines = [f"{args.a} vs {args.b}: per-update depth shares (b3_depth_shares), time means in B0's windows; "
             f"agreement as B0 Step 4 (exact p over the 720 relabellings of the six image modalities)"]

    selfc = compare(a, a)
    ok_self = all(abs(r["rho"] - 1) < 1e-12 and r["meanabs"] == 0 for r in selfc["windows"].values())
    ctl = compare(a, c)
    ok_ctl = ctl["series"]["p"] > 0.05 and all(r["p_meanabs"] > 0.05 for r in ctl["windows"].values())
    lines.append(f"  control, {args.a} against itself: rho 1 and mean|d| 0 in every window: {'holds' if ok_self else 'FAILS'}")
    lines.append(f"  control, {args.a} against the random router {args.control} (no table), which must not agree: "
                 f"series rho {ctl['series']['rho']:+.2f} (p {ctl['series']['p']:.3f}); mean|d| "
                 + ", ".join(f"{w} {r['meanabs']:.3f} (p {r['p_meanabs']:.3f})" for w, r in ctl["windows"].items())
                 + f": {'does not agree, as it must' if ok_ctl else 'AGREES: the test cannot tell tables apart'}")

    res = compare(a, b)
    for w, r in res["windows"].items():
        lines.append(f"  {w:8s} rho {r['rho']:+.2f} (p {r['p']:.3f})   mean|d| {r['meanabs']:.3f} (p {r['p_meanabs']:.3f})")
        for i, m in enumerate(IMG):
            lines.append(f"     {m:6s} {r['depth_a'][i]:.2f} {r['state_a'][i]:9s} {r['shares_a'][i]}   "
                         f"{r['depth_b'][i]:.2f} {r['state_b'][i]:9s} {r['shares_b'][i]}")
    lines.append(f"  over the four windows (one relabelling at every window): rho {res['series']['rho']:+.2f} "
                 f"(p {res['series']['p']:.3f}); early, plateau and late: rho {res['series_early_plateau_late']['rho']:+.2f} "
                 f"(p {res['series_early_plateau_late']['p']:.3f})")
    text = "\n".join(lines)
    print(text)
    if not (ok_self and ok_ctl):
        print("a control failed: nothing written")
        return 1
    stem = D / f"seeds_{args.a}_vs_{args.b}"
    stem.with_suffix(".txt").write_text(text + "\n")
    stem.with_suffix(".json").write_text(json.dumps({"a": args.a, "b": args.b, "control": args.control,
                                                     "control_self_holds": ok_self, "control_random_disagrees": ok_ctl,
                                                     "control_random": ctl, "result": res}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
