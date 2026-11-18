#!/usr/bin/env python
"""A0.3, does SmolLM-135M's early lead need stage 1? (2026-11-18). Three Coords-free runs over their first 3,300 steps
of full training, read by the rule fixed in the worklog before B10cL0-tr-NC was launched. No training.

  R  B10c-tr-x10-NC: random init.
  S  B10cL-tr-x10-NC: SmolLM-135M's body, stage 1 (330 steps, body frozen, table trained), then stage 2.
  N  B10cL0-tr-NC: SmolLM-135M's body from step 0, no stage 1 (arm_b10cl0trxnc_vanilla), stopped at 3,300 on the
     33,000-step schedule, so its LR to 3,300 is R's and S's stage 2's.

Two x axes:
  - steps of full training (S's stage-2 steps; stage 1 is not on this axis). Steps, scenes, FLOPs and schedule are
    then identical for R and N, and S has stage 1's 330 table-only steps on top;
  - training FLOPs, with S's stage 1 counted at its own cost (262.0 full steps; nc_init_compare.py's accounting).
Measures: the logged held-out eval (bf16, every eval step to 3,300) and E1 / top-1 at fp32 at the scored checkpoints
(33, 99, 165, 330, 660, 990, 1650, 3300), each pair of runs paired over the 4,416 held-out rows with a row bootstrap.

The rule (worklog, 2026-11-18): primary N - S at 3,300, fp32 E1, bar b = 0.010 nats:
  (a) stage 1 earns its place   interval entirely above +b
  (b) no stage 1 needed         interval entirely inside [-b, +b]
  (c) stage 1 costs             interval entirely below -b
  undetermined                  the interval straddles +b or -b (1,650 and the logged eval then read as description)
N - R at 3,300 beside it: below -b, the body helps without stage 1; inside, nothing measurable; above +b, worse.
Equal compute (description): on the logged eval, N at s + 262 against S at s.

    .venv/bin/python eo/experiments/pretrained_init/nc_stage1_compare.py [--states DIR] [--out DIR] [--self-test]

Runs in `.venv` (numpy, matplotlib). Reads trainer states as nc_init_compare.py does (`--states`, one <run>.json per
run, else the newest local checkpoint).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nc_init_compare as NC          # noqa: E402  (loaders, FLOPs geometry, paired_endpoint)

OUT = NC.OUT
WINDOW = 3300
BAR = 0.010
S1_FULL_EQUIV = 330 * NC.ROW_FLOPS["table"] / NC.ROW_FLOPS["full"]       # 262.0
RUNS3 = {
    "R": {"label": "Random init (B10c-tr-x10-NC)", "color": "#2a78d6",
          "run": "vanilla10ctrxnc_20261114_181026", "tag": "b10ctrxnc_fp32", "s1": False},
    "S": {"label": "SmolLM-135M, stage 1 then stage 2 (B10cL-tr-x10-NC)", "color": "#eb6834",
          "run": "vanilla10cltrxnc_20261114_185037", "tag": "b10cltrxnc_fp32", "s1": True},
    "N": {"label": "SmolLM-135M, no stage 1 (B10cL0-tr-NC)", "color": "#1f9e6e",
          "run": None, "tag": "b10cl0trxnc_fp32", "s1": False},
}
PAIRS = (("N", "S"), ("N", "R"), ("S", "R"))
READ_STEPS = (33, 99, 165, 330, 660, 990, 1650, 3300)


def find_run_n() -> str | None:
    """B10cL0-tr-NC's run id: the newest local run directory of its arm."""
    c = sorted(NC.RUNS.glob("vanilla10cl0trxnc_*"))
    return c[-1].name if c else None


def curves(key: str, states: Path | None) -> dict:
    """(step, flops, value) arrays over steps 1..WINDOW of full training; S's FLOPs include stage 1."""
    spec = RUNS3[key]
    f0 = 330 * NC.STEP_FLOPS["table"] if spec["s1"] else 0.0
    sf = NC.STEP_FLOPS["full"]
    out = {"logged": [], "e1": [], "top1": []}
    state = NC.trainer_state(spec["run"], states) if spec["run"] else None
    if state:
        s, l, _ = NC.logged_eval(state)
        for si, li in zip(s, l):
            if 0 < si <= WINDOW:
                out["logged"].append((si, f0 + si * sf, li))
    s, ce, t1, _, _ = NC.scored(spec["tag"])
    for si, ci, ti in zip(s, ce, t1):
        if 0 < si <= WINDOW:
            out["e1"].append((si, f0 + si * sf, ci))
            out["top1"].append((si, f0 + si * sf, ti))
    return {k: np.array(sorted(v), float).reshape(-1, 3) for k, v in out.items()}


def verdict(lo: float, hi: float, bar: float = BAR) -> str:
    """The rule on N - S's 95% interval."""
    if lo > bar:
        return "(a) stage 1 earns its place"
    if hi < -bar:
        return "(c) stage 1 costs"
    if lo >= -bar and hi <= bar:
        return "(b) no stage 1 needed"
    return "undetermined (the interval straddles the bar)"


def versus_random(lo: float, hi: float, bar: float = BAR) -> str:
    if hi < -bar:
        return "the body helps without stage 1"
    if lo > bar:
        return "worse than random init"
    if lo >= -bar and hi <= bar:
        return "nothing measurable over random init"
    return "undetermined (the interval straddles the bar)"


def tok_dir(key: str, step: int) -> Path:
    return NC.ARM_EVAL / RUNS3[key]["tag"] / f"checkpoint-{step}"


def self_test() -> bool:
    """The verdict functions on known intervals, and paired_endpoint on a run against itself (0) and against another
    run (the stored S - R at 3,300, about -0.020), so the reader can tell a difference from none."""
    ok = (verdict(0.012, 0.014).startswith("(a)") and verdict(-0.004, 0.009).startswith("(b)")
          and verdict(-0.02, -0.011).startswith("(c)") and verdict(0.008, 0.012).startswith("undetermined")
          and versus_random(-0.03, -0.02).startswith("the body") and versus_random(-0.005, 0.005).startswith("nothing")
          and versus_random(0.02, 0.03).startswith("worse"))
    print(f"  verdict rules on six known intervals = {ok}")
    a, b = tok_dir("S", 3300), tok_dir("R", 3300)
    if (a / "tokens_router.npz").is_file() and (b / "tokens_router.npz").is_file():
        same = NC.paired_endpoint(a, a, n_boot=200)
        diff = NC.paired_endpoint(a, b, n_boot=2000)
        z = same["e1"]["delta"] == 0 and same["e1"]["ci95"] == [0.0, 0.0]
        d = diff["e1"]["ci95"][1] < -BAR and abs(diff["e1"]["delta"] + 0.0201) < 0.0005
        print(f"  S against itself at 3300: {same['e1']['delta']:+.4f} {same['e1']['ci95']} = {z}")
        print(f"  S against R at 3300: {diff['e1']['delta']:+.4f} [{diff['e1']['ci95'][0]:+.4f}, "
              f"{diff['e1']['ci95'][1]:+.4f}], the stored -0.0201, below -b = {d}")
        ok &= z and d
    else:
        print("  (paired self-test skipped: S's or R's checkpoint-3300 not scored here)")
        ok = False
    return bool(ok)


def figure(cv: dict, path: Path):
    """Rows: logged held-out CE; N minus S and N minus R on it; fp32 E1; fp32 top-1. Columns: steps of full training,
    training FLOPs (S's stage 1 counted). Log x: the early evals are dense."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ink, muted, grid = "#1a1a19", "#6b6a64", "#e4e3dd"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    rows = ["logged", "gap", "e1", "top1"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 2.8 * len(rows)), squeeze=False, sharex="col")
    for i, m in enumerate(rows):
        for j, (xi, xlab) in enumerate(((0, "Steps of full training (S: stage-2 steps)"),
                                        (1, "Training FLOPs (S: stage 1 counted)"))):
            ax = axes[i, j]
            ax.grid(True, color=grid, lw=0.6)
            ax.set_axisbelow(True)
            if m == "gap":
                ax.axhline(0, color=muted, lw=0.8)
                ax.axhspan(-BAR, BAR, color=grid, alpha=0.6, lw=0)
                late = []
                for ref, ls in (("S", "-"), ("R", "--")):
                    cn, cr = cv["N"]["logged"], cv[ref]["logged"]
                    pts = [(x, v - r) for x, v in zip(cn[:, xi], cn[:, 2])
                           if (r := NC.interp_on(cr[:, xi], cr[:, 2], x)) is not None]
                    g = np.array(pts, float).reshape(-1, 2)
                    if len(g):
                        ax.plot(g[:, 0], g[:, 1], color=RUNS3["N"]["color"], lw=2, ls=ls, label=f"N minus {ref}")
                        late += list(g[g[:, 0] >= 10 * g[0, 0], 1])
                if late:
                    lim = max(abs(np.array(late)).max() * 1.4, 2 * BAR)
                    ax.set_ylim(-lim, lim)
                ax.legend(frameon=False, fontsize=7, loc="lower right")
                ax.text(0.99, 0.95, f"band: ±{BAR} (seed noise); below 0: N lower CE", transform=ax.transAxes,
                        ha="right", va="top", color=muted, fontsize=7)
            else:
                for key, spec in RUNS3.items():
                    c = cv[key][m]
                    if len(c):
                        ax.plot(c[:, xi], c[:, 2], color=spec["color"], lw=2,
                                marker=None if m == "logged" else "o", ms=4, label=spec["label"])
            ax.set_xscale("log")
            if m == "top1":
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
            if i == len(rows) - 1:
                ax.set_xlabel(xlab)
            if j == 0:
                ax.set_ylabel({"logged": "Held-out CE, logged (nats)", "gap": "N minus other,\nlogged CE (nats)",
                               "e1": "Held-out CE, fp32 (nats)", "top1": "Held-out top-1, fp32"}[m])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, ncol=1, fontsize=8)
    fig.suptitle("A0.3: SmolLM-135M's body with and without stage 1, first 3,300 steps (Coords-free)", color=ink,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=str(OUT / "states"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        ok = self_test()
        print(f"self-test {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    NC._check_geometry()
    RUNS3["N"]["run"] = find_run_n()
    states = Path(args.states) if Path(args.states).is_dir() else None
    cv = {k: curves(k, states) for k in RUNS3}
    L = [f"A0.3, stage 1 or not: the first {WINDOW} steps of full training. Bar {BAR} nats (seed noise, B3). "
         f"S's stage 1 = {S1_FULL_EQUIV:.1f} full steps of FLOPs."]
    for k, spec in RUNS3.items():
        L.append(f"  {k} {spec['run']}: {len(cv[k]['logged'])} logged evals, fp32 at "
                 f"{[int(x) for x in cv[k]['e1'][:, 0]] or 'none yet'}")
    res = {"bar": BAR, "window": WINDOW, "runs": {k: v["run"] for k, v in RUNS3.items()},
           "points": {k: {m: cv[k][m].tolist() for m in cv[k]} for k in cv}, "paired": {}, "equal_flops": []}

    L.append(f"\n== fp32, paired over the held-out rows; first minus second [95% row bootstrap]")
    for step in READ_STEPS:
        for a, b in PAIRS:
            da, db = tok_dir(a, step), tok_dir(b, step)
            if not ((da / "tokens_router.npz").is_file() and (db / "tokens_router.npz").is_file()):
                continue
            e = NC.paired_endpoint(da, db)
            res["paired"][f"{a}-{b}@{step}"] = e
            L.append(f"  step {step:>4} {a} - {b}: E1 {e['e1']['llm']:.4f} vs {e['e1']['random']:.4f} = "
                     f"{e['e1']['delta']:+.4f} [{e['e1']['ci95'][0]:+.4f}, {e['e1']['ci95'][1]:+.4f}]; top-1 "
                     f"{100 * e['top1']['delta']:+.2f} pp [{100 * e['top1']['ci95'][0]:+.2f}, "
                     f"{100 * e['top1']['ci95'][1]:+.2f}]")

    L.append(f"\n== The rule at {WINDOW}")
    ns, nr = res["paired"].get(f"N-S@{WINDOW}"), res["paired"].get(f"N-R@{WINDOW}")
    if ns and nr:
        res["verdict"] = verdict(*ns["e1"]["ci95"])
        res["versus_random"] = versus_random(*nr["e1"]["ci95"])
        L.append(f"  N - S {ns['e1']['delta']:+.4f} [{ns['e1']['ci95'][0]:+.4f}, {ns['e1']['ci95'][1]:+.4f}]: "
                 f"{res['verdict']}")
        L.append(f"  N - R {nr['e1']['delta']:+.4f} [{nr['e1']['ci95'][0]:+.4f}, {nr['e1']['ci95'][1]:+.4f}]: "
                 f"{res['versus_random']}")
        for step in (1650, 990, 330):
            p = res["paired"].get(f"N-S@{step}")
            if p:
                L.append(f"  (description) step {step}: N - S {p['e1']['delta']:+.4f} -> "
                         f"{verdict(*p['e1']['ci95'])}")
    else:
        L.append(f"  not yet: N's checkpoint-{WINDOW} is not scored at fp32")

    L.append("\n== Logged eval (description): N minus S and N minus R at equal steps; N at s + 262 minus S at s")
    cn, csl, cr = cv["N"]["logged"], cv["S"]["logged"], cv["R"]["logged"]
    for s, _, v in cn:
        vs = NC.interp_on(csl[:, 0], csl[:, 2], s) if len(csl) > 1 else None
        vr = NC.interp_on(cr[:, 0], cr[:, 2], s) if len(cr) > 1 else None
        vsf = NC.interp_on(csl[:, 0], csl[:, 2], s - S1_FULL_EQUIV) if s > S1_FULL_EQUIV and len(csl) > 1 else None
        res["equal_flops"].append({"step": s, "n": v, "n_minus_s": None if vs is None else v - vs,
                                   "n_minus_r": None if vr is None else v - vr,
                                   "n_minus_s_equal_flops": None if vsf is None else v - vsf})
    pick = [r for r in res["equal_flops"] if int(r["step"]) in (33, 99, 165, 330, 660, 990, 1650, 2475, 3300)]
    for r in pick:
        f = lambda x: "  n/a  " if x is None else f"{x:+.4f}"      # noqa: E731
        L.append(f"  step {int(r['step']):>4}: N {r['n']:.4f}; N - S {f(r['n_minus_s'])}; N - R {f(r['n_minus_r'])}; "
                 f"N@s+262 - S@s {f(r['n_minus_s_equal_flops'])}")
    cross = next((r["step"] for r in res["equal_flops"] if r["n_minus_r"] is not None and r["n_minus_r"] < 0), None)
    close = next((r["step"] for r in res["equal_flops"] if r["n_minus_s"] is not None and abs(r["n_minus_s"]) <= BAR),
                 None)
    res["n_below_r_from"], res["n_within_bar_of_s_from"] = cross, close
    L.append(f"  N first below R at step {cross}; first within {BAR} of S at step {close} (logged, description)")

    out = Path(args.out)
    if all(len(cv[k]["logged"]) for k in RUNS3):
        figure(cv, out / "nc_stage1_compare")
    (out / "nc_stage1_compare.json").write_text(json.dumps(res, indent=1))
    (out / "nc_stage1_compare.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out / 'nc_stage1_compare'}.{{png,pdf,json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
