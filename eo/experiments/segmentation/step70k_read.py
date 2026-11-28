#!/usr/bin/env python
"""A2 Step 7.0c's read (docs/plans/a2_segmentation.md, Step 7, "7.0c, the date count and a date curriculum on B1's
base"; the rule pre-registered 2026-11-26 before any of its numbers). `.venv`, CPU; reads the scored maps
(step52_score.py) and the pulled records. Fold 4 only.

    python eo/experiments/segmentation/step70k_read.py [--manifest <runs.json>] [--out <dir>]
    python eo/experiments/segmentation/step70k_read.py --decide-wave2        # the runs wave 2 adds, as json

  1. validity per run: 7.0's items (training finite, the plateau record agreeing with its curve, Z8 by the precision
     rule, Z9), and finetune_init.json naming B1's base; a curriculum stage C_k (k >= 2) also names C_{k-1}'s kept
     save (its run, step and weights SHA-256). A failed run is reported and read as a hole;
  2. pairs: paired bootstrap over the 482 views (1,000 resamples, seed 0; step70_read.pair). A difference GAINS iff
     Delta >= +1.2 and its lower end > 0, COSTS iff Delta <= -1.2 and its upper end < 0, else it is level;
  3. experiment 1: P_k for each run (P_7 = B1), the steps between neighbouring runs, R_k = P_7 - P_k, and k* = the
     smallest k whose R_k does not gain (7 if every R_k gains). Wave 2 (`--decide-wave2`): P_5 and P_6 iff R_4 gains;
     else P_3 iff R_2 gains; else none (`wave2`);
  4. experiment 2: Delta(C_7 - B1): (a) gains, (b) neither, (c) loses;
  5. beside, deciding nothing: C_k - P_k; full - wrong_early on 7.0, B1 and C_7 and the C_7 - B1 difference of those
     drops; per-class IoU; map CE, plateau epochs and training time.
The manifest (written by the kit's runner, /data/enric/rcp_a2s4/s70k/runs.json) maps roles to run ids:
    {"P": {"1": run, "2": run, ...}, "C": {"2": run, ..., "7": run}}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step70_read as S70                           # noqa: E402
import step70b1_read as B                           # noqa: E402
from step52_read import IGNORE, RUNS                # noqa: E402
from step53_verdict import iou_per_class, miou      # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step70k")
GEN = OUT / "gen"
MANIFEST = Path("/data/enric/rcp_a2s4/s70k/runs.json")
B1_RUN, R70_RUN = "p7b1_s42_20261126_084434", B.P70_RUN
BAR = B.GAIN                                         # 1.2 = 2 x sqrt(2) x 0.42
B1_BASE_SHA, A2_BASE_SHA = B.B1_BASE_SHA, B.A2_BASE_SHA
TASK = S70.TASK


# ---------------------------------------------------------------- the rules (pure; the gate tests them)
def call(p: Optional[dict]) -> str:
    """'gains' / 'costs' / 'level' for one paired difference (read rule 2); 'hole' without one."""
    if p is None:
        return "hole"
    if p["delta"] >= BAR and p["lo95"] > 0:
        return "gains"
    if p["delta"] <= -BAR and p["hi95"] < 0:
        return "costs"
    return "level"


def kstar(R: Dict[int, dict]) -> Optional[int]:
    """The smallest k whose R_k = P_7 - P_k does not gain; 7 if every one gains (read rule 3). Holes are skipped."""
    ks = sorted(k for k, p in R.items() if p is not None and k < 7)
    for k in ks:
        if call(R[k]) != "gains":
            return k
    return 7 if ks else None


def wave2(R: Dict[int, dict]) -> List[int]:
    """The runs that can still move k* after wave 1 (P_1, P_2, P_4): P_5 and P_6 iff R_4 gains; else P_3 iff R_2
    gains; else none (A2 plan, 7.0c, design item 3)."""
    if call(R.get(4)) == "gains":
        return [5, 6]
    if call(R.get(2)) == "gains":
        return [3]
    return []


def band_c7(p: dict) -> str:
    c = call(p)
    if c == "gains":
        return "(a) gains: the direct jump to seven dates costs at this recipe; a second seed of both before adopting"
    if c == "costs":
        return "(c) loses: the curriculum hurts"
    return "(b) neither: the curriculum does not help; B1 is not held back by the jump"


def pair_of_drops(a_full: np.ndarray, a_ctl: np.ndarray, b_full: np.ndarray, b_ctl: np.ndarray) -> dict:
    """(a_full - a_ctl) - (b_full - b_ctl) in mIoU points, one index draw per resample shared by all four."""
    ig = IGNORE[TASK]
    for x in (a_ctl, b_full, b_ctl):
        if x.shape != a_full.shape:
            raise RuntimeError(f"shapes {a_full.shape} {x.shape}")
    f = lambda c: miou(c.sum(0), ig)                                    # noqa: E731
    point = ((f(a_full) - f(a_ctl)) - (f(b_full) - f(b_ctl))) * 100
    rng = np.random.default_rng(S70.BOOT_SEED)
    n = a_full.shape[0]
    idx = rng.integers(0, n, size=(S70.N_BOOT, n))
    d = np.array([(f(a_full[i]) - f(a_ctl[i])) - (f(b_full[i]) - f(b_ctl[i])) for i in idx]) * 100
    return {"delta": point, "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "share_positive": float((d > 0).mean())}


# ---------------------------------------------------------------- the records
def gen_root(run: str) -> Path:
    """Where a run's own maps were pulled: B1's and 7.0's under their steps' roots, every 7.0c run under GEN."""
    return B.GEN if run == B1_RUN else S70.GEN if run == R70_RUN else GEN


def gen_dir(run: str) -> Path:
    return gen_root(run) / run


def validity(R: dict) -> tuple:
    """(precision, reasons it fails): 7.0's rule (step70_read.main)."""
    zb = R["z8_bf16"]
    prec = "bf16" if (zb and zb["pass"]) else "fp32"
    x, why = R[prec], []
    if not R["finite"]:
        why.append("training not finite")
    if not R["plateau_agrees"]:
        why.append("the plateau record disagrees with the curve")
    if prec == "fp32" and not (R["z8_fp32"] and R["z8_fp32"]["pass"]):
        why.append("Z8 fails at bf16 and at fp32 (or its records are missing)")
    if x.get("k8") is None:
        why.append(f"K = 8 maps at {prec} missing")
    if all(x.get(k) for k in ("prior", "wrong", "k1")):
        own, pr, wr = x["k1"]["mIoU"], x["prior"]["mIoU"], x["wrong"]["mIoU"]
        R["z9"] = {"pass": bool(pr < own and wr < own), "own_k1": own, "prior": pr, "wrong": wr,
                   "functional": bool(abs(wr - pr) < abs(wr - own))}
        if not R["z9"]["pass"]:
            why.append("Z9: prior or wrong not below the own maps")
    else:
        why.append("Z9's maps missing")
    return prec, why


def lineage(run: str, parent: Optional[str], want_base: str = B1_BASE_SHA) -> List[str]:
    """Reasons the run's finetune_init.json does not name the expected base (and parent save)."""
    f = RUNS / run / "finetune_init.json"
    if not f.is_file():
        return [f"{f} missing"]
    fi = json.loads(f.read_text())
    why = []
    if fi.get("weights_sha256") != want_base:
        why.append(f"base {str(fi.get('weights_sha256'))[:12]}, not {want_base[:12]}")
    c = fi.get("continued")
    if parent is None:
        if c is not None:
            why.append(f"continued from {c.get('run')}, expected a fresh start")
        return why
    if c is None:
        return why + [f"not continued, expected {parent}'s save"]
    psel = (RUNS / parent / "selected_checkpoint.txt")
    pstep = int(psel.read_text().split()[0].split("-")[1]) if psel.is_file() else None
    psha = (RUNS / parent / "selected_sha256.txt")
    if c.get("run") != parent:
        why.append(f"continued from {c.get('run')}, not {parent}")
    if pstep is None or c.get("step") != pstep:
        why.append(f"continued from step {c.get('step')}, {parent}'s kept save is {pstep}")
    if not psha.is_file() or c.get("weights_sha256") != psha.read_text().split()[0]:
        why.append(f"continued from weights {str(c.get('weights_sha256'))[:12]}, not {parent}'s kept save's "
                   f"({psha.read_text().split()[0][:12] if psha.is_file() else 'no selected_sha256.txt'})")
    return why


def read_one(run: str, parent: Optional[str] = None, base: str = B1_BASE_SHA) -> dict:
    try:
        R = S70.read_run(run, gen_root(run))
    except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError) as e:
        return {"run": run, "valid": False, "why": [f"records: {type(e).__name__}: {e}"]}
    prec, why = validity(R)
    why += lineage(run, parent, base)
    R["precision"], R["valid"], R["why"] = prec, not why, why
    if R["valid"]:
        R["mIoU"] = float(R[prec]["k8"]["mIoU"]) * 100
    return R


def cm(run: str, prec: str, line: str = "full_k8") -> np.ndarray:
    d = gen_dir(run) if line in ("full_k8", "one_date_k8", "shuffled_dates_k8") else GEN / run
    d = d.parent / f"{d.name}_fp32" if prec == "fp32" else d
    return S70.cm_of(d / line)


def common(ra: dict, rb: dict) -> str:
    return "bf16" if ra["precision"] == rb["precision"] == "bf16" else "fp32"


def pair_runs(a: dict, b: dict, line_a: str = "full_k8", line_b: str = "full_k8") -> Optional[dict]:
    """a - b at their common precision; None (pending) when a map set is missing at it."""
    p = common(a, b)
    try:
        xa, xb = cm(a["run"], p, line_a), cm(b["run"], p, line_b)
    except FileNotFoundError as e:
        return {"pending": f"{p} maps missing: {e}"}
    out = S70.pair(xa, xb)
    out["precision"] = p
    return out


# ---------------------------------------------------------------- the read
def load_manifest(path: Path) -> dict:
    m = json.loads(path.read_text())
    return {"P": {int(k): v for k, v in (m.get("P") or {}).items()},
            "C": {int(k): v for k, v in (m.get("C") or {}).items()}}


def experiment1(man: dict, runs: Dict[str, dict]) -> dict:
    P = dict(man["P"])
    P[7] = B1_RUN
    ok = {k: runs[r] for k, r in P.items() if runs[r].get("valid")}
    holes = sorted(k for k in P if k not in ok)
    ks = sorted(ok)
    steps = {}
    for a, b in zip(ks, ks[1:]):
        p = pair_runs(ok[b], ok[a])
        steps[f"{b}-{a}"] = {**p, "call": call(p) if "delta" in (p or {}) else "pending"}
    R = {}
    for k in ks:
        if k < 7 and 7 in ok:
            p = pair_runs(ok[7], ok[k])
            R[k] = p if "delta" in p else None
    return {"runs": {k: P[k] for k in sorted(P)}, "holes": holes, "P": {k: ok[k]["mIoU"] for k in ks},
            "steps": steps, "R": {k: ({**p, "call": call(p)} if p else None) for k, p in R.items()},
            "kstar": kstar(R), "wave2_rule": wave2(R) if {2, 4} <= set(R) else None}


def experiment2(man: dict, runs: Dict[str, dict]) -> dict:
    C = {1: man["P"].get(1), **man["C"]}
    res = {"runs": C, "stages": {}}
    for k in sorted(C):
        r = runs.get(C[k]) if C[k] else None
        res["stages"][k] = {"run": C[k], "valid": bool(r and r.get("valid")), "mIoU": (r or {}).get("mIoU"),
                            "why": (r or {}).get("why")}
        if k >= 2 and r and r.get("valid") and man["P"].get(k) and runs[man["P"][k]].get("valid"):
            res["stages"][k]["C_minus_P"] = pair_runs(r, runs[man["P"][k]])
    c7, b1 = runs.get(C.get(7)) if C.get(7) else None, runs[B1_RUN]
    if c7 and c7.get("valid") and b1.get("valid"):
        p = pair_runs(c7, b1)
        res["C7_minus_B1"] = p
        res["band"] = band_c7(p) if "delta" in p else "pending"
    else:
        res["band"] = "not read: C_7 or B1 invalid or missing"
    return res


def early(runs: Dict[str, dict], c7: Optional[str]) -> dict:
    """Beside: full - wrong_early on 7.0, B1 and C_7 (bf16 maps), and the C_7 - B1 difference of those drops."""
    out = {}
    have = {}
    for name, run in (("7.0", R70_RUN), ("B1", B1_RUN), ("C7", c7)):
        if not run or not runs.get(run, {}).get("valid"):
            continue
        d = GEN / run / "wrong_early_k8"
        if not (d / "per_patch_cm.npy").exists():
            out[name] = {"pending": f"{d} not scored"}
            continue
        rec = json.loads((d / "record.json").read_text())
        sel = runs[run]["selected"]["checkpoint"]
        if not str(rec.get("checkpoint", "")).endswith(sel):
            raise RuntimeError(f"{d}: from {rec.get('checkpoint')}, not {sel}")
        if S70.views_of(d) != S70.views_of(gen_dir(run) / "full_k8"):
            raise RuntimeError(f"{d}: views differ from the full maps'")
        full, ctl = S70.cm_of(gen_dir(run) / "full_k8"), S70.cm_of(d)
        out[name] = S70.pair(full, ctl)
        have[name] = (full, ctl)
    if "C7" in have and "B1" in have:
        out["C7_minus_B1_drop"] = pair_of_drops(*have["C7"], *have["B1"])
    return out


def per_class(runs: Dict[str, dict], roles: Dict[str, str]) -> list:
    from eo.maps import palette as P
    names = P.TASKS[TASK].class_names
    cols = {}
    for role, run in roles.items():
        r = runs.get(run)
        if r and r.get("valid"):
            cols[role] = iou_per_class(cm(run, r["precision"]).sum(0), IGNORE[TASK]) * 100
    return [{"class": names[i], **{k: float(v[i]) for k, v in cols.items()}} for i in range(len(names))
            if cols and not all(np.isnan(v[i]) for v in cols.values())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--decide-wave2", action="store_true")
    a = ap.parse_args()
    man = load_manifest(Path(a.manifest))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {B1_RUN: read_one(B1_RUN), R70_RUN: read_one(R70_RUN, base=A2_BASE_SHA)}
    for k, r in man["P"].items():
        runs[r] = read_one(r)
    for k, r in man["C"].items():
        runs[r] = read_one(r, parent=(man["C"].get(k - 1) if k > 2 else man["P"].get(1)))
    e1 = experiment1(man, runs)
    if a.decide_wave2:
        need = {1, 2, 4}
        bad = [k for k in sorted(need) if k not in e1["P"]]
        pend = [k for k in (2, 4) if k in e1["P"] and e1["R"].get(k) is None]
        res = {"wave2": None if (bad or pend) else e1["wave2_rule"], "holes": e1["holes"],
               "R": {k: {x: v.get(x) for x in ("delta", "lo95", "hi95", "call")} if v else None
                     for k, v in e1["R"].items()},
               "why": [f"P_{k} missing or invalid: {runs[man['P'][k]]['why'] if k in man['P'] else 'not run'}"
                       for k in bad] + [f"R_{k} pending (maps at the common precision missing)" for k in pend]}
        (out / "wave2.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res, default=float))
        sys.exit(0 if res["wave2"] is not None else 3)
    e2 = experiment2(man, runs)
    roles = {f"P{k}": r for k, r in e1["runs"].items()}
    roles.update({f"C{k}": r for k, r in man["C"].items()})
    res = {"rule": "A2 plan, Step 7, 7.0c (pre-registered 2026-11-26)", "bar": BAR,
           "runs": {r: {k: v for k, v in R.items() if k in ("valid", "why", "precision", "mIoU", "selected_ce",
                                                              "selected_step", "plateau_stop", "z9", "step0_ce")}
                    | {"train_runtime_s": (R.get("curve") or {}).get("train_runtime_s")} for r, R in runs.items()},
           "experiment1": e1, "experiment2": e2, "early": early(runs, man["C"].get(7)),
           "per_class": per_class(runs, roles)}
    (out / "read.json").write_text(json.dumps(res, indent=1, default=float))
    L = ["# A2 Step 7.0c read", "", "## Experiment 1: the latest k dates on B1's base"]
    for k, v in sorted(e1["P"].items()):
        L.append(f"- P_{k} = {v:.2f}" + (f"; R_{k} = {e1['R'][k]['delta']:+.2f} [{e1['R'][k]['lo95']:+.2f}, "
                                          f"{e1['R'][k]['hi95']:+.2f}] {e1['R'][k]['call']}" if e1["R"].get(k) else ""))
    for s, p in e1["steps"].items():
        if "delta" in p:
            L.append(f"- step {s}: {p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}] {p['call']}")
    L += [f"- holes: {e1['holes']}; k* = {e1['kstar']}", "", "## Experiment 2: the curriculum"]
    for k, s in e2["stages"].items():
        cp = s.get("C_minus_P")
        L.append(f"- C_{k} {s['run']}: " + (f"{s['mIoU']:.2f}" if s["mIoU"] is not None else f"not read {s['why']}")
                 + (f"; C - P {cp['delta']:+.2f} [{cp['lo95']:+.2f}, {cp['hi95']:+.2f}]" if cp and "delta" in cp else ""))
    p = e2.get("C7_minus_B1")
    if p and "delta" in p:
        L.append(f"- C_7 - B1: {p['delta']:+.2f} [{p['lo95']:+.2f}, {p['hi95']:+.2f}]")
    L += [f"- verdict: {e2['band']}", "", "## Beside: full - wrong_early"]
    for k, v in res["early"].items():
        L.append(f"- {k}: " + (f"{v['delta']:+.2f} [{v['lo95']:+.2f}, {v['hi95']:+.2f}]" if "delta" in v else str(v)))
    (out / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
