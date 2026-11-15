#!/usr/bin/env python
"""Paper A, A0.3 (wave 1, 2026-11-11): how much faster pretraining is from a borrowed body, on the val runs' stored
curves. No training.

For each pretrained-body arm (B10cV: CLEVR's layers; B10cL: SmolLM-135M's; B10cS and B10cLS: their shuffled
controls) against B10c, trained from scratch on the same data, seed and schedule:
  - the speedup curve: at each held-out eval of the init arm, the B10c step that first reaches the same loss,
    divided by the init arm's own cost in B10c steps. Stage 1 (330 steps, body frozen, table trained) is counted at
    its FLOPs: 857.6 of a full step's 1,072.7 GF per row (curves.py's "full" definition; REFERENCE, Compute
    accounting), so stage 1 is worth 263.8 B10c steps;
  - A0.3's measure (Paper A plan): the steps, FLOPs and wall clock the init arm needs to reach B10c's held-out CE at
    10%, 25%, 50% and 100% of B10c's run (steps 330, 825, 1650, 3300), or "never";
  - the end point: E1 at fp32 (Titan X) minus B10c's, paired over the 4,416 rows, with a row bootstrap.
Two curves: the trainer's logged eval_loss (every 165 steps, all five arms; includes BO/EO) and E1 at fp32 (every
330 steps; B10c, B10cV and B10cL, the arms scored per checkpoint).

    .venv/bin/python eo/experiments/pretrained_init/a03_speedup.py [--out /data/enric/reports/wave1/speedup]

Runs in `.venv` (numpy only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS = Path("/data/enric/runs/pretrain/phase3")
G49 = Path("/data/enric/reports/wave1/speedup/g49")       # B10cS and B10cLS: trainer states pulled from g49
ARM = Path("/data/enric/reports/arm_eval")
FULL_GF, TABLE_GF = 1072.7, 857.6                          # training GF per row: a full step, a stage-1 step
STAGE1_STEPS = 330
STAGE1_EQ = STAGE1_STEPS * TABLE_GF / FULL_GF              # 263.8 B10c steps
STEP_FLOPS = 256 * FULL_GF * 1e9                           # one B10c step, 2.746e14
TARGETS = (330, 825, 1650, 3300)                           # 10%, 25%, 50%, 100% of B10c's run

ARMS = {
    "B10c": {"s2": RUNS / "vanilla10c_20261011_172253"},
    "B10cV": {"s1": RUNS / "vanilla10cv_s1_20261027_184927", "s2": RUNS / "vanilla10cv_20261027_194530",
              "e1": "b10cv_fp32", "e1_s1": "b10cv_s1_fp32", "body": "CLEVR"},
    "B10cS": {"s1": G49 / "vanilla10cs_s1_20261028_034200", "s2": G49 / "vanilla10cs_20261028_043808",
              "body": "CLEVR, shuffled (control)"},
    "B10cL": {"s1": RUNS / "vanilla10cl_s1_20261028_123122", "s2": RUNS / "vanilla10cl_20261028_133225",
              "e1": "b10cl_fp32", "e1_s1": "b10cl_s1_fp32", "body": "SmolLM-135M"},
    "B10cLS": {"s1": G49 / "vanilla10cls_s1_20261028_212352", "s2": G49 / "vanilla10cls_20261028_221458",
               "body": "SmolLM-135M, shuffled (control)"},
}


def _state(run: Path) -> dict:
    cks = sorted(run.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    return json.loads(cks[-1].read_text())


def logged(run: Path):
    """(steps, eval_loss) with non-finite evals dropped, and seconds per step from the run's own runtime."""
    st = _state(run)
    ev = [(d["step"], d["eval_loss"]) for d in st["log_history"] if "eval_loss" in d]
    ev = [(s, v) for s, v in ev if np.isfinite(v)]
    tot = [d for d in st["log_history"] if "train_runtime" in d]
    allr = run / "all_results.json"
    if allr.is_file():               # train_runtime includes the evals, as a run's wall clock does
        sps = json.loads(allr.read_text())["train_runtime"] / st["global_step"]
    else:
        sps = tot[-1]["train_runtime"] / st["global_step"] if tot else float("nan")
    return np.array([s for s, _ in ev], float), np.array([v for _, v in ev], float), sps


def e1_curve(tag: str):
    cks = sorted(int(p.name.split("-")[1]) for p in (ARM / tag).glob("checkpoint-*"))
    vals = [json.loads((ARM / tag / f"checkpoint-{c}" / "metrics.json").read_text())["modes"]["router"]["ce_all_body_tokens"]
            for c in cks]
    return np.array(cks, float), np.array(vals, float)


def first_reach(steps, loss, target):
    """The first step at which a curve, linear between its points, reaches <= target; None if it never does."""
    if loss[0] <= target:
        return float(steps[0])
    for i in range(1, len(steps)):
        if loss[i] <= target:
            a, b = loss[i - 1], loss[i]
            return float(steps[i - 1] + (steps[i] - steps[i - 1]) * (a - target) / (a - b))
    return None


def init_curve(arm, kind):
    """An init arm's curve on B10c's step axis: stage 1 at its FLOPs, then stage 2 from STAGE1_EQ."""
    a = ARMS[arm]
    if kind == "logged":
        s1, l1, sps1 = logged(a["s1"])
        s2, l2, sps2 = logged(a["s2"])
    else:
        s1, l1 = e1_curve(a["e1_s1"])
        s2, l2 = e1_curve(a["e1"])
        sps1 = sps2 = float("nan")
    s1 = s1 * TABLE_GF / FULL_GF
    keep = s1 > 0                    # stage 1's step 0 is the untrained table on the borrowed body
    x = np.concatenate([s1[keep], STAGE1_EQ + s2[s2 > 0]])
    y = np.concatenate([l1[keep], l2[s2 > 0]])
    return x, y, sps1, sps2


def paired_endpoint(tag_a, tag_b, ck=3300, n_boot=2000, seed=0):
    """E1(a) - E1(b) over body tokens, per row then pooled by tokens, with a row bootstrap."""
    za = np.load(ARM / tag_a / f"checkpoint-{ck}" / "tokens_router.npz")
    zb = np.load(ARM / tag_b / f"checkpoint-{ck}" / "tokens_router.npz")
    assert np.array_equal(za["label_mod"], zb["label_mod"]) and np.array_equal(za["label"], zb["label"])
    body = za["label_mod"] > 0
    sa = np.where(body, za["ce"].astype(np.float64), 0).sum(1)
    sb = np.where(body, zb["ce"].astype(np.float64), 0).sum(1)
    n = body.sum(1).astype(np.float64)
    d = (sa.sum() - sb.sum()) / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), (n_boot, len(n)))
    boot = ((sa[idx] - sb[idx]).sum(1)) / n[idx].sum(1)
    return d, np.percentile(boot, [2.5, 97.5]).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/enric/reports/wave1/speedup")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"stage1_eq_steps": STAGE1_EQ, "step_flops": STEP_FLOPS, "targets_steps": TARGETS, "curves": {}}
    lines = [f"A0.3 on the val runs: stage 1 counted as {STAGE1_EQ:.1f} B10c steps ({TABLE_GF} of {FULL_GF} GF a row)."]
    for kind in ("logged", "e1"):
        if kind == "logged":
            bs, bl, bsps = logged(ARMS["B10c"]["s2"])
        else:
            bs, bl = e1_curve("b10c_fp32")
            bsps = logged(ARMS["B10c"]["s2"])[2]
        first = float(bl[bs > 0][0])     # B10c's first trained eval: above that loss its curve is not resolved
        lines.append(f"\n== {kind} held-out CE ({'eval_loss, every 165 steps' if kind == 'logged' else 'E1 fp32, every 330'})"
                     f"; B10c {bsps:.2f} s a step")
        for arm in ("B10cV", "B10cS", "B10cL", "B10cLS"):
            if kind == "e1" and "e1" not in ARMS[arm]:
                continue
            if kind == "e1":                 # the init arms' E1 has its own sps from the logged runs
                pass
            x, y, sps1, sps2 = init_curve(arm, kind)
            curve = []
            for xi, yi in zip(x, y):
                s_star = first_reach(bs, bl, yi) if yi <= first else None
                curve.append({"init_eq_step": round(float(xi), 1), "loss": round(float(yi), 4), "resolved": bool(yi <= first),
                              "b10c_step_same_loss": None if s_star is None else round(s_star, 1),
                              "speedup": None if s_star is None else round(s_star / xi, 3),
                              "speedup_stage1_free": None if (s_star is None or xi <= STAGE1_EQ) else round(s_star / (xi - STAGE1_EQ), 3)})
            meas = []
            for t in TARGETS:
                target = float(np.interp(t, bs, bl))
                xr = first_reach(x, y, target)
                wall_b = t * bsps / 3600
                if xr is None:
                    meas.append({"b10c_step": t, "b10c_loss": round(target, 4), "init_eq_step": None, "speedup": None,
                                 "init_flops": None, "b10c_flops": t * STEP_FLOPS, "b10c_hours": round(wall_b, 2), "init_hours": None})
                    continue
                s1_used = min(xr, STAGE1_EQ) / (TABLE_GF / FULL_GF)          # stage-1 steps actually run
                s2_used = max(xr - STAGE1_EQ, 0.0)
                if kind == "e1":
                    sps1, sps2 = logged(ARMS[arm]["s1"])[2], logged(ARMS[arm]["s2"])[2]
                wall_i = (s1_used * sps1 + s2_used * sps2) / 3600
                meas.append({"b10c_step": t, "b10c_loss": round(target, 4), "init_eq_step": round(xr, 1),
                             "speedup": round(t / xr, 3), "init_flops": xr * STEP_FLOPS, "b10c_flops": t * STEP_FLOPS,
                             "b10c_hours": round(wall_b, 2), "init_hours": None if wall_i is None else round(wall_i, 2)})
            # where the init arm's lead ends: its last point still below B10c at equal cost
            lead = [c for c in curve if c["speedup"] is not None and c["speedup"] >= 1]
            res["curves"][f"{kind}|{arm}"] = {"curve": curve, "measure": meas,
                                              "lead_ends_after_eq_step": lead[-1]["init_eq_step"] if lead else None,
                                              "s_per_step": [sps1, sps2]}
            lines.append(f"  {arm} ({ARMS[arm]['body']}), speedup at each eval resolved by B10c's curve "
                         f"(B10c-step equivalents: x, and x with stage 1 free in brackets): " + ", ".join(
                f"{c['init_eq_step']:.0f}: {c['speedup']:.2f} ({c['speedup_stage1_free']:.2f})" for c in curve
                if c["speedup"] is not None and c["speedup_stage1_free"] is not None))
            for m in meas:
                if m["init_eq_step"] is None:
                    lines.append(f"    B10c's CE at {m['b10c_step']} ({m['b10c_loss']:.4f}): never reached")
                else:
                    wall = (f", {m['init_hours']:.2f} h vs {m['b10c_hours']:.2f} h" if m["init_hours"] is not None
                            and np.isfinite(m["init_hours"]) else ", wall clock not recorded")
                    lines.append(f"    B10c's CE at {m['b10c_step']} ({m['b10c_loss']:.4f}): at {m['init_eq_step']:.0f} "
                                 f"B10c-step equivalents, x{m['speedup']:.2f} ({m['init_flops']:.2e} vs {m['b10c_flops']:.2e} FLOPs{wall})")
    lines.append("\n== End point, E1 fp32 at 3300, paired over the 4,416 rows (body tokens)")
    res["endpoint"] = {}
    for arm in ("B10cV", "B10cL"):
        d, ci = paired_endpoint(ARMS[arm]["e1"], "b10c_fp32")
        res["endpoint"][arm] = {"delta": d, "ci95": ci}
        lines.append(f"  {arm} - B10c: {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    (out / "a03_speedup_val.json").write_text(json.dumps(res, indent=1))
    (out / "a03_speedup_val.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
