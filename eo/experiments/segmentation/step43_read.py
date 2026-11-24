#!/usr/bin/env python
"""A2 Steps 4.3 and 4.4's read (`.venv`, CPU): the ablations against Route A, R0's controls, and forgetting.

Plan: docs/plans/a2_segmentation.md, Step 4, "Steps 4.3 and 4.4's design and reading" (2026-11-23, written before any of
their numbers).
  4.3  each ablation (one seed, 42) against Step 5.2's run at seed 42 on its task, R0's (the palette: R1's, on R1's base):
         - the precision: bf16 if both runs' kept saves pass Z8 at bf16, else fp32 for both (each readable by the gate's
           fp32 rule or the 1,000x rule);
         - Delta = ablation - Route A in K = 8 mIoU points, a paired bootstrap over patches or tiles (1,000, seed 0),
           per-class IoU; the bar: Delta beyond 1.0 point with the interval clear of 0;
         - beside it: grid (the plateau epoch, kept save = last save, generation seconds), chain (full - wrong_lulc at
           K = 8, prior and wrong at K = 1).
  4.4  controls: R0's six runs at bf16: own input K = 1 (two samples), prior, wrong, Z9's functional check
       |m_wrong - m_prior| < |m_wrong - m_full| (its control: the second sample in place of the wrong input must fail
       it), and the prior within 2 points of chance (uniform-random maps by the same scorer: PASTIS 0.016, CloudSen12
       0.126); forgetting: each run's fp32 reading against the map base's (eval_forgetting.py, tag per run), the mean CE
       difference over every scored token paired by token, a row bootstrap (1,000, seed 0), per modality, against A1's
       margin +0.02 nats; and the map rows' softmax mass (step44_map_mass.py).

    python eo/experiments/segmentation/step43_read.py [--grid <run>] [--chain <run>] [--palette <run>]
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

from step41_read import z8_of                                     # noqa: E402
from step52_read import FP32_ROUNDING, IGNORE, RUNS, read_run, scored  # noqa: E402
from step53_verdict import iou_per_class, miou                     # noqa: E402

GEN41 = Path("/data/enric/reports/a2_seg/step41/gen")
GEN52 = Path("/data/enric/reports/a2_seg/step52/gen")
GEN43 = Path("/data/enric/reports/a2_seg/step43/gen")
OUT = Path("/data/enric/reports/a2_seg/step43")
FORGET = Path("/data/enric/reports/a2_seg/step44/forgetting")
MASS = Path("/data/enric/reports/a2_seg/step44/map_mass")
ROUTE_A = {"pastis": ("r0p1s41_l16c_s42_20261122_123818", GEN41), "cloudsen12": ("r0c12s52_s42_20261122_190655", GEN52)}
R0_RTS = {"c12": "20261122_190655", "p44": "20261122_185744"}
ABLATIONS = {"grid": "pastis", "chain": "pastis", "palette": "cloudsen12"}
#: each ablation's reference run: Route A's seed 42, except the palette, which runs on R1's base (Enric, 2026-11-23) and
#: pairs with Step 5.2's R1 CloudSen12 seed 42
REFERENCE = {"grid": ROUTE_A["pastis"], "chain": ROUTE_A["pastis"],
             "palette": ("r1c12s52_s42_20261123_100922", GEN52)}
BAR, N_BOOT, BOOT_SEED = 1.0, 1000, 0
CHANCE = {"pastis": 0.016, "cloudsen12": 0.126}
PRIOR_WITHIN = 0.02
MARGIN = 0.02


def read_named(name: str, gen: Path, task: str) -> dict:
    """step52_read.read_run for a run named outright, plus the chain's wrong_lulc maps."""
    sel = json.loads((RUNS / name / "select41.json").read_text())
    gb, gf = gen / name, gen / f"{name}_fp32"
    r = {"task": task, "run": name, "selected": sel["selected"], "plateau_stop": sel.get("plateau_stop"),
         "health": sel.get("health"), "z8_bf16": z8_of(gb, "bf16"), "z8_fp32": z8_of(gb, "fp32")}
    zf = r["z8_fp32"]
    r["fp32_readable"] = None if zf is None else bool(
        zf["pass"] or zf["max_abs_logit_diff"] * FP32_ROUNDING <= zf["control_abs"])
    for p, g in (("bf16", gb), ("fp32", gf)):
        x = {k: scored(g / sub, task) for k, sub in (("k8", "full_k8"), ("k1", "full_k8/k1"), ("k1b", "full_k8/k1b"),
                                                    ("prior", "prior_k1"), ("wrong", "wrong_k1"),
                                                    ("greedy", "full_greedy"), ("wrong_lulc", "wrong_lulc_k8"))}
        if x["k8"] is not None and x["k8"].get("checkpoint") and \
                not str(x["k8"]["checkpoint"]).endswith(sel["selected"]["checkpoint"]):
            raise RuntimeError(f"{name}: K = 8 maps from {x['k8']['checkpoint']}, not {sel['selected']['checkpoint']}")
        rec = g / "full_k8" / "record.json"
        if rec.exists():
            x["k8_seconds"] = json.loads(rec.read_text()).get("seconds")
        r[p] = x
    return r


def pair(task: str, a: dict, b: dict, ref_gen: Path) -> dict:
    """An ablation b against its reference run a (in ref_gen), by the plan's item 1."""
    za, zb = a["z8_bf16"], b["z8_bf16"]
    if za is None or zb is None:
        return {"complete": False, "why": "a Z8 reading is missing"}
    prec = "bf16" if (za["pass"] and zb["pass"]) else "fp32"
    out = {"complete": True, "precision": prec, "z8_bf16": {"A": za["pass"], "ablation": zb["pass"]}}
    if prec == "fp32" and not (a["fp32_readable"] and b["fp32_readable"]):
        out["stopped"] = "a run is readable at fp32 by neither rule: Enric decides"
        return out
    if a[prec]["k8"] is None or b[prec]["k8"] is None:
        return {"complete": False, "why": f"K = 8 maps at {prec} missing"}
    sub = "" if prec == "bf16" else "_fp32"
    ca = np.load((ref_gen / f"{a['run']}{sub}" / "full_k8") / "per_patch_cm.npy")
    cb = np.load((GEN43 / f"{b['run']}{sub}" / "full_k8") / "per_patch_cm.npy")
    if ca.shape != cb.shape:
        raise RuntimeError(f"the runs' maps differ in shape {ca.shape} {cb.shape}")
    ma, mb = miou(ca.sum(0), IGNORE[task]) * 100, miou(cb.sum(0), IGNORE[task]) * 100
    for m, r in ((ma, a), (mb, b)):
        if abs(m - r[prec]["k8"]["mIoU"] * 100) > 1e-9:
            raise RuntimeError(f"{r['run']}: its per-sample matrices do not sum to its read mIoU")
    rng = np.random.default_rng(BOOT_SEED)
    n = ca.shape[0]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d = np.array([miou(cb[i].sum(0), IGNORE[task]) - miou(ca[i].sum(0), IGNORE[task]) for i in idx]) * 100
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    delta = mb - ma
    from eo.maps import palette as P
    names = P.TASKS[task].class_names
    pa, pb = iou_per_class(ca.sum(0), IGNORE[task]) * 100, iou_per_class(cb.sum(0), IGNORE[task]) * 100
    other = "fp32" if prec == "bf16" else "bf16"
    out.update({"A": ma, "ablation": mb, "delta": delta, "bootstrap": {"n": n, "mean": float(d.mean()), "lo95": lo,
                "hi95": hi, "share_positive": float((d > 0).mean())},
                "beyond_bar_up": bool(delta > BAR and lo > 0), "beyond_bar_down": bool(delta < -BAR and hi < 0),
                "other_precision_delta": (None if a[other]["k8"] is None or b[other]["k8"] is None else
                                          (b[other]["k8"]["mIoU"] - a[other]["k8"]["mIoU"]) * 100),
                "per_class": [{"class": names[k], "A": float(pa[k]), "ablation": float(pb[k]),
                               "delta": float(pb[k] - pa[k])} for k in range(len(names))
                              if not (np.isnan(pa[k]) and np.isnan(pb[k]))]})
    return out


def records(r: dict) -> dict:
    ps = (r.get("plateau_stop") or {}).get("fired") or {}
    return {"run": r["run"], "kept": r["selected"]["checkpoint"], "kept_epoch": r["selected"]["epoch"],
            "kept_ce": r["selected"]["eval_ce"], "plateau_epoch": ps.get("epoch"), "stop_step": ps.get("stop_step"),
            "health": r.get("health"), "z8_bf16_steps": (r["z8_bf16"] or {}).get("max_ulp_diff"),
            "z8_fp32_pass": (r["z8_fp32"] or {}).get("pass"), "fp32_readable": r.get("fp32_readable"),
            "k8_seconds": {p: r[p].get("k8_seconds") for p in ("bf16", "fp32")},
            "maps": {p: {k: (v or {}).get("mIoU") for k, v in r[p].items() if isinstance(v, dict) or v is None}
                     for p in ("bf16", "fp32")}}


def controls() -> dict:
    out = {}
    for task in ("pastis", "cloudsen12"):
        rows = []
        for seed in (42, 43, 44):
            r = read_run(task, seed, R0_RTS)
            x = r["bf16"]
            row = {"run": r["run"], "seed": seed}
            if None in (x["k1"], x["k1b"], x["prior"], x["wrong"]):
                row["missing"] = [k for k in ("k1", "k1b", "prior", "wrong") if x[k] is None]
            else:
                mf, mf2, mp, mw = (x[k]["mIoU"] for k in ("k1", "k1b", "prior", "wrong"))
                row.update({"own_k1": mf, "own_k1b": mf2, "prior": mp, "wrong": mw, "k8": x["k8"]["mIoU"],
                            "z9_pass": bool(abs(mw - mp) < abs(mw - mf)),
                            "z9_control_fails": bool(not (abs(mf2 - mp) < abs(mf2 - mf))),
                            "prior_minus_chance": mp - CHANCE[task],
                            "prior_near_chance": bool(abs(mp - CHANCE[task]) <= PRIOR_WITHIN)})
            rows.append(row)
        ok = all("missing" not in r and r["z9_pass"] and r["z9_control_fails"] and r["prior_near_chance"] for r in rows)
        out[task] = {"precision": "bf16", "chance": CHANCE[task], "runs": rows, "pass": ok}
    return out


def forgetting(run: str, ckpt: str) -> dict:
    before, after = FORGET / "r0arm" / "extended_base", FORGET / run / ckpt
    if not ((before / "forgetting.json").exists() and (after / "forgetting.json").exists()):
        return {"run": run, "missing": True}
    jb, ja = (json.loads((d / "forgetting.json").read_text()) for d in (before, after))
    if (jb["precision"], jb["arm_config"], jb["n_rows"]) != (ja["precision"], ja["arm_config"], ja["n_rows"]):
        raise RuntimeError(f"{run}: the readings differ in precision, arm-config or rows")
    tb, ta = np.load(before / "tokens_ce.npz"), np.load(after / "tokens_ce.npz")
    if not (np.array_equal(tb["rows"], ta["rows"]) and np.array_equal(tb["label_mod"], ta["label_mod"])):
        raise RuntimeError(f"{run}: the two readings did not score the same rows and tokens")
    cb, ca, lm = tb["ce"].astype(np.float64), ta["ce"].astype(np.float64), tb["label_mod"]
    body = lm > 0
    dtok = (ca - cb) * body
    rs, ns = dtok.sum(1), body.sum(1)
    mean = float(rs.sum() / ns.sum())
    rng = np.random.default_rng(BOOT_SEED)
    boot = [rs[i].sum() / ns[i].sum() for i in (rng.integers(0, len(rs), len(rs)) for _ in range(N_BOOT))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    from eo.data import eo_vocab_nc as V
    per = {}
    for mid, name in V.ID_TO_MODALITY.items():
        m = lm == mid
        if m.any():
            per[name] = {"before": float((cb * m).sum() / m.sum()), "after": float((ca * m).sum() / m.sum()),
                         "diff": float(((ca - cb) * m).sum() / m.sum()), "tokens": int(m.sum())}
    mass = MASS / f"{run}.json"
    return {"run": run, "checkpoint": ckpt, "ce_before": jb["ce_all_body_tokens"], "ce_after": ja["ce_all_body_tokens"],
            "diff": mean, "ci95": [float(lo), float(hi)], "within_margin": bool(mean <= MARGIN), "per_modality": per,
            "gpu": [jb["gpu"], ja["gpu"]], "precision": ja["precision"],
            "map_mass": json.loads(mass.read_text()) if mass.exists() else None,
            "check_reproduces_json": [abs(float((cb * body).sum() / body.sum()) - jb["ce_all_body_tokens"]) < 1e-3,
                                      abs(float((ca * body).sum() / body.sum()) - ja["ce_all_body_tokens"]) < 1e-3]}


def main():
    ap = argparse.ArgumentParser()
    for k in ABLATIONS:
        ap.add_argument(f"--{k}", default=None, help=f"the {k} ablation's run id")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    res = {"what": "A2 Steps 4.3 and 4.4 (plan Step 4, 'Steps 4.3 and 4.4's design and reading')", "ablations": {}}
    for k, task in ABLATIONS.items():
        name = getattr(a, k)
        if not name:
            continue
        ref, ref_gen = REFERENCE[k]
        ra = read_named(ref, ref_gen, task)
        rb = read_named(name, GEN43, task)
        x = pair(task, ra, rb, ref_gen)
        x["reference"] = ref
        x["records"] = {"A": records(ra), "ablation": records(rb)}
        if k == "chain" and x.get("complete"):
            p = x["precision"]
            wl = rb[p].get("wrong_lulc")
            x["full_minus_wrong_lulc"] = None if wl is None else (rb[p]["k8"]["mIoU"] - wl["mIoU"]) * 100
        res["ablations"][k] = x
    res["controls"] = controls()
    f = {}
    for task in ("pastis", "cloudsen12"):
        for seed in (42, 43, 44):
            r = read_run(task, seed, R0_RTS)
            f[r["run"]] = forgetting(r["run"], r["selected"]["checkpoint"])
    res["forgetting"] = f
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "read.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    L = ["# A2 Steps 4.3 and 4.4", "", "## 4.3 ablations (one seed, 42, against its reference: Route A seed 42; the palette, R1 seed 42)", "",
         "| ablation | task | precision | reference | ablation | Delta | bootstrap 95% | beyond the bar |", "|---|---|---|---|---|---|---|---|"]
    for k, x in res["ablations"].items():
        if not x.get("complete") or "stopped" in x:
            L.append(f"| {k} | {ABLATIONS[k]} | - | - | - | - | - | {x.get('why') or x.get('stopped')} |")
            continue
        b = x["bootstrap"]
        L.append(f"| {k} | {ABLATIONS[k]} | {x['precision']} | {x['A']:.2f} | {x['ablation']:.2f} | {x['delta']:+.2f} | "
                 f"[{b['lo95']:+.2f}, {b['hi95']:+.2f}] | {'up' if x['beyond_bar_up'] else 'down' if x['beyond_bar_down'] else 'no'} |")
    L += ["", "## 4.4 controls (bf16, K = 1)", ""]
    for t, c in res["controls"].items():
        L.append(f"- {t} (chance {c['chance']}): pass {c['pass']}")
        for r in c["runs"]:
            L.append(f"  - seed {r['seed']}: " + (f"missing {r['missing']}" if "missing" in r else
                     f"own {r['own_k1']:.4f} ({r['own_k1b']:.4f}), prior {r['prior']:.4f}, wrong {r['wrong']:.4f}; "
                     f"Z9 {r['z9_pass']}, its control fails {r['z9_control_fails']}; prior - chance {r['prior_minus_chance']:+.4f}"))
    L += ["", "## 4.4 forgetting (fp32, against the map base; A1's margin +0.02)", ""]
    for run, x in f.items():
        L.append(f"- {run}: " + ("missing" if x.get("missing") else
                 f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] nats, within {x['within_margin']}; "
                 f"map mass {(x['map_mass'] or {}).get('mean_map_mass')}"))
    (a.out / "read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
