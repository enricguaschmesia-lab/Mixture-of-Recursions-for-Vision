#!/usr/bin/env python
"""The init comparison (2026-11-14), read: B10c-tr-x10-NC (random init) against B10cL-tr-x10-NC (SmolLM-135M's
body), both Coords-free (nc vocabulary), bf16, 33,000 steps of 256 scenes at a flat 3e-3, decayed over the last 3,300.
No training.

Two x axes, as Enric asked (data, compute):
  - scenes seen: optimizer steps x 256. The LLM run's stage 1 (330 steps, body frozen, table trained) is counted:
    its stage 2 starts at 330 x 256 scenes;
  - training FLOPs on curves.py's "full" definition, at the nc geometry (990 positions a row, 81,188 ids, 29 layers;
    no padding: a full nc row is 5 x 198 = 990 = max_length). Stage 1 is counted at its own cost,
    2 lin + 3 att + 3 head a row (the backward still runs through the frozen body, but no weight gradients are
    taken there; REFERENCE, Compute accounting). The SmolLM body's own pretraining is not counted (sunk).
Three held-out measures on y:
  - the trainer's logged eval_loss (bf16, the 4,416 held-out rows, dense early, every 165 steps after 990);
  - E1 at fp32 on the GTX Titan X (eval_checkpoint.py's main pass: CE over every image token of the 4,416 rows);
  - top-1 accuracy at fp32 from the same pass, pooled over image tokens.
fp32 scoring covers selected checkpoints only (`/data/enric/rcp_w1/nc_compare_score.sh`): the script reads
whatever is scored and says what is missing, so it can run while the runs train.

Measures (text and JSON), on each curve with both arms:
  - at 1%, 3%, 10%, 25%, 50% and 100% of the random-init run (steps 330 ... 33,000): the random-init loss there, the
    point at which the LLM run first reaches it (linear between its points), in scenes and FLOPs, and the ratios
    (random init's cost / LLM init's cost; > 1 means the LLM init got there sooner), or "never";
  - the gap at equal scenes and at equal FLOPs at every LLM point inside the random-init curve's range;
  - the end point when both final checkpoints are scored at fp32: E1 and top-1, LLM minus random, paired over the
    4,416 rows, with a row bootstrap (10,000 draws). The two runs saw the same rows in the same order (one seed,
    one data path), but an equal-scene comparison is shifted by stage 1's 330 steps.
This script states readings. The base decision is Enric's.

    .venv/bin/python eo/experiments/pretrained_init/nc_init_compare.py [--states DIR] [--out DIR]

Trainer states come from `--states` (one `<run>.json` per run, the latest checkpoint's trainer_state.json, fetched
from g49 by nc_compare_score.sh), falling back to the newest local checkpoint. Runs in `.venv` (numpy, matplotlib).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ the runs
RUNS = Path("/data/enric/runs/pretrain/phase3")
ARM_EVAL = Path("/data/enric/reports/arm_eval")
OUT = Path("/data/enric/reports/wave1/nc_compare")
ARMS = {
    "random": {"label": "Random init (B10c-tr-x10-NC)", "color": "#2a78d6",
               "stages": [{"run": "vanilla10ctrxnc_20261114_181026", "tag": "b10ctrxnc_fp32", "kind": "full"}]},
    "llm": {"label": "SmolLM-135M init (B10cL-tr-x10-NC)", "color": "#eb6834",
            "stages": [{"run": "vanilla10cltrxnc_s1_20261114_181205", "tag": "b10cltrxnc_s1_fp32", "kind": "table"},
                       {"run": "vanilla10cltrxnc_20261114_185037", "tag": "b10cltrxnc_fp32", "kind": "full"}]},
}
TOTAL_STEPS, BATCH = 33_000, 256
TARGET_FRACS = (0.01, 0.03, 0.10, 0.25, 0.50, 1.00)

# ------------------------------------------------------------------ compute, nc geometry
D, FFN, N_LAYERS = 576, 1536, 29
V_NC, SEQ_NC = 81_188, 990
P_LAYER = (576 * 576 * 2 + 576 * 192 * 2) + 3 * 576 * FFN + 2 * 576       # 3,540,096 (curves.py)
LIN = 2 * P_LAYER * N_LAYERS * SEQ_NC
ATT = 2 * D * N_LAYERS * SEQ_NC ** 2
HEAD = 2 * D * V_NC * SEQ_NC
ROW_FLOPS = {"full": 3 * (LIN + ATT + HEAD), "table": 2 * LIN + 3 * ATT + 3 * HEAD}
STEP_FLOPS = {k: v * BATCH for k, v in ROW_FLOPS.items()}


def _check_geometry():
    """The v1 geometry must reproduce REFERENCE's 1,072.7 / 857.6 GF a row, or the formula here has drifted."""
    seq, v = 1048, 87_556
    lin, att, head = 2 * P_LAYER * N_LAYERS * seq, 2 * D * N_LAYERS * seq ** 2, 2 * D * v * seq
    assert abs(3 * (lin + att + head) / 1e9 - 1072.7) < 0.1, 3 * (lin + att + head) / 1e9
    assert abs((2 * lin + 3 * att + 3 * head) / 1e9 - 857.6) < 0.1, (2 * lin + 3 * att + 3 * head) / 1e9


# ------------------------------------------------------------------ loaders
def trainer_state(run: str, states: Path | None) -> dict | None:
    if states is not None and (states / f"{run}.json").is_file():
        return json.loads((states / f"{run}.json").read_text())
    cks = sorted((RUNS / run).glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    return json.loads(cks[-1].read_text()) if cks else None


def logged_eval(state: dict):
    """(steps, eval_loss, per-modality dict) for every finite logged eval; the per-modality keys are a sibling entry."""
    ev = defaultdict(dict)
    for e in state["log_history"]:
        if any(k.startswith("eval_loss") for k in e):
            ev[e["step"]].update(e)
    pts = [(s, v) for s, v in sorted(ev.items()) if np.isfinite(v.get("eval_loss", np.nan))]
    return (np.array([s for s, _ in pts], float), np.array([v["eval_loss"] for _, v in pts], float),
            [{k[len("eval_loss_"):]: x for k, x in v.items() if k.startswith("eval_loss_")} for _, v in pts])


def scored(tag: str):
    """(steps, E1, top-1, per-modality) from eval_checkpoint's fp32 main passes of one run."""
    rows = []
    for mp in (ARM_EVAL / tag).glob("checkpoint-*/metrics.json"):
        d = json.loads(mp.read_text())
        if d.get("precision") != "fp32" or "router" not in d.get("modes", {}) or d.get("drop_modalities"):
            continue
        assert d.get("vocab_layout") == "eo_vocab_nc", (mp, d.get("vocab_layout"))
        pm = d["modes"]["router"]["per_modality"]
        n = sum(v["n_tokens"] for v in pm.values())
        top1 = sum(v["top1"] * v["n_tokens"] for v in pm.values()) / n
        ce = sum(v["ce"] * v["n_tokens"] for v in pm.values()) / n
        assert abs(ce - d["modes"]["router"]["ce_all_body_tokens"]) < 1e-9, mp      # per-modality covers every token
        rows.append((d["step"], ce, top1, {m: {"ce": v["ce"], "top1": v["top1"]} for m, v in pm.items()}, str(mp.parent)))
    rows.sort()
    return (np.array([r[0] for r in rows], float), np.array([r[1] for r in rows], float),
            np.array([r[2] for r in rows], float), [r[3] for r in rows], [r[4] for r in rows])


# ------------------------------------------------------------------ one arm on the common axes
def arm_curves(arm: str, states: Path | None):
    """Per measure, the arm's points on both axes. A later stage starts where the previous one ended: at its steps
    and FLOPs. Stage 2's step 0 is stage 1's last checkpoint (the same weights), so it is dropped."""
    out = {"logged": [], "e1": [], "top1": []}
    meta = {"stages": []}
    step0, flops0 = 0.0, 0.0
    for st in ARMS[arm]["stages"]:
        state = trainer_state(st["run"], states)
        sf = STEP_FLOPS[st["kind"]]
        last = int(state["global_step"]) if state else 0
        if state:
            s, l, pm = logged_eval(state)
            keep = s > 0
            for si, li in zip(s[keep], l[keep]):
                out["logged"].append((step0 + si, flops0 + si * sf, li))
        s, ce, t1, _, dirs = scored(st["tag"])
        keep = s > 0
        for si, ci, ti in zip(s[keep], ce[keep], t1[keep]):
            out["e1"].append((step0 + si, flops0 + si * sf, ci))
            out["top1"].append((step0 + si, flops0 + si * sf, ti))
        meta["stages"].append({"run": st["run"], "kind": st["kind"], "global_step": last,
                               "scored_steps": [int(x) for x in s], "step_offset": step0, "flops_offset": flops0})
        n_steps = (TOTAL_STEPS if st["kind"] == "full" else 330)
        step0 += n_steps
        flops0 += n_steps * sf
    curves = {k: np.array(sorted(v), float).reshape(-1, 3) for k, v in out.items()}
    return curves, meta


def first_reach(x, y, target, lower_is_better=True):
    """The first x at which a curve, linear between its points, reaches the target; None if it never does."""
    ok = (y <= target) if lower_is_better else (y >= target)
    if not ok.any():
        return None
    i = int(np.argmax(ok))
    if i == 0:
        return float(x[0])
    a, b = y[i - 1], y[i]
    return float(x[i - 1] + (x[i] - x[i - 1]) * (a - target) / (a - b))


def interp_on(x, y, xq):
    """The curve's value at xq, linear between its points (in log x for the step and FLOPs axes); None outside."""
    if len(x) < 2 or xq < x[0] or xq > x[-1]:
        return None
    return float(np.interp(np.log(xq), np.log(x), y))


def measures(cr, cl, measure):
    """A0.3's measure for this comparison on one curve. cr, cl: (n, 3) arrays of (step, flops, value)."""
    lower = measure != "top1"
    res = []
    if len(cr) < 2 or len(cl) < 2:
        return res
    for frac in TARGET_FRACS:
        t = frac * TOTAL_STEPS
        v = interp_on(cr[:, 0], cr[:, 2], t)
        if v is None:
            res.append({"frac": frac, "random_step": t, "status": "random-init run not there yet"})
            continue
        s_l = first_reach(cl[:, 0], cl[:, 2], v, lower)
        f_l = first_reach(cl[:, 1], cl[:, 2], v, lower)
        rec = {"frac": frac, "random_step": t, "random_value": v, "random_scenes": t * BATCH,
               "random_flops": t * STEP_FLOPS["full"]}
        if s_l is None:
            rec["status"] = f"never, up to step {cl[-1, 0]:.0f}"
        else:
            rec.update(status="reached", llm_step=s_l, llm_scenes=s_l * BATCH, llm_flops=f_l,
                       data_ratio=t / s_l, flops_ratio=t * STEP_FLOPS["full"] / f_l)
        res.append(rec)
    return res


def gaps(cr, cl, measure):
    """LLM minus random at equal scenes and at equal FLOPs, at each LLM point inside the random curve's range."""
    out = []
    for s, f, v in cl:
        r_s = interp_on(cr[:, 0], cr[:, 2], s)
        r_f = interp_on(cr[:, 1], cr[:, 2], f)
        out.append({"llm_step": s, "llm_flops": f, "llm_value": v,
                    "gap_equal_scenes": None if r_s is None else v - r_s,
                    "gap_equal_flops": None if r_f is None else v - r_f})
    return out


def paired_endpoint(dir_l: Path, dir_r: Path, n_boot=10_000, seed=0):
    """LLM minus random, E1 and top-1 over image tokens, per row then pooled by tokens, with a row bootstrap."""
    zl, zr = np.load(dir_l / "tokens_router.npz"), np.load(dir_r / "tokens_router.npz")
    assert np.array_equal(zl["label"], zr["label"]) and np.array_equal(zl["label_mod"], zr["label_mod"]), \
        "the two passes did not score the same tokens"
    body = zl["label_mod"] > 0
    n = body.sum(1).astype(np.float64)
    out = {}
    for key, name in (("ce", "e1"), ("correct", "top1")):
        a = np.where(body, zl[key].astype(np.float64), 0).sum(1)
        b = np.where(body, zr[key].astype(np.float64), 0).sum(1)
        d = (a.sum() - b.sum()) / n.sum()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(n), (n_boot, len(n)))
        boot = (a[idx] - b[idx]).sum(1) / n[idx].sum(1)
        out[name] = {"delta": d, "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                     "llm": a.sum() / n.sum(), "random": b.sum() / n.sum()}
    out["n_rows"], out["n_tokens"] = int(len(n)), int(n.sum())
    return out


# ------------------------------------------------------------------ figure
YLAB = {"logged": "Held-out CE, logged (nats)", "e1": "Held-out CE, fp32 (nats)", "top1": "Held-out top-1 accuracy"}


def gap_curve(cr, cl, xi):
    """LLM minus random on the logged curve at each LLM point inside the random curve's range, on axis xi."""
    pts = [(x, v - r) for x, v in zip(cl[:, xi], cl[:, 2])
           if (r := interp_on(cr[:, xi], cr[:, 2], x)) is not None]
    return np.array(pts, float).reshape(-1, 2)


def figure(curves, path: Path):
    """Rows: the logged held-out CE (full range), its gap LLM minus random at equal x, fp32 CE and fp32 top-1 at the
    scored checkpoints. Columns: scenes seen, training FLOPs. Log x: the early evals are dense."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ink, muted, grid = "#1a1a19", "#6b6a64", "#e4e3dd"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    rows = ["logged", "gap"] + [m for m in ("e1", "top1") if any(len(curves[a][m]) for a in ARMS)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 2.8 * len(rows)), squeeze=False, sharex="col")
    s1_end = {0: 330 * BATCH, 1: 330 * STEP_FLOPS["table"]}
    for i, m in enumerate(rows):
        for j, (xi, xlab) in enumerate(((0, "Scenes seen"), (1, "Training FLOPs"))):
            ax = axes[i, j]
            ax.grid(True, color=grid, lw=0.6)
            ax.set_axisbelow(True)
            scale = BATCH if xi == 0 else 1
            ax.axvline(s1_end[xi], color=muted, lw=0.8, ls=":")
            if m == "gap":
                g = gap_curve(curves["random"]["logged"], curves["llm"]["logged"], xi)
                ax.axhline(0, color=muted, lw=0.8)
                if len(g):
                    ax.plot(g[:, 0] * scale, g[:, 1], color=ARMS["llm"]["color"], lw=2)
                    late = g[g[:, 0] * scale >= 10 * s1_end[xi], 1]
                    if len(late):
                        lim = max(abs(late).max() * 1.6, 0.01)
                        ax.set_ylim(-lim, lim)
                ax.text(0.99, 0.95, "below 0: LLM init lower CE", transform=ax.transAxes, ha="right", va="top",
                        color=muted, fontsize=7)
            else:
                for arm, spec in ARMS.items():
                    c = curves[arm][m]
                    if len(c):
                        ax.plot(c[:, xi] * scale, c[:, 2], color=spec["color"], lw=2,
                                marker=None if m == "logged" else "o", ms=4, label=spec["label"])
            ax.set_xscale("log")
            if m == "top1":
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
            if i == len(rows) - 1:
                ax.set_xlabel(xlab)
            if j == 0:
                ax.set_ylabel({"gap": "LLM minus random,\nlogged CE (nats)"}.get(m, YLAB.get(m, m)))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, ncol=2)
    axes[0, 0].text(s1_end[0], axes[0, 0].get_ylim()[0], " stage 1 ends", color=muted, fontsize=7, va="bottom")
    fig.suptitle("Init comparison, Coords-free: SmolLM-135M body vs random init", color=ink, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=str(OUT / "states"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    _check_geometry()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    states = Path(args.states) if Path(args.states).is_dir() else None
    curves, meta = {}, {}
    for arm in ARMS:
        curves[arm], meta[arm] = arm_curves(arm, states)
    res = {"row_gflops": {k: v / 1e9 for k, v in ROW_FLOPS.items()}, "batch": BATCH, "meta": meta, "measures": {},
           "gaps": {}, "points": {a: {m: curves[a][m].tolist() for m in curves[a]} for a in ARMS}}
    L = [f"Init comparison, Coords-free (nc): FLOPs a row {ROW_FLOPS['full'] / 1e9:.1f} GF (full step), "
         f"{ROW_FLOPS['table'] / 1e9:.1f} GF (stage 1); stage 1 = 330 steps = {330 * STEP_FLOPS['table']:.3e} FLOPs "
         f"= {330 * ROW_FLOPS['table'] / ROW_FLOPS['full']:.1f} full steps."]
    for arm in ARMS:
        for st in meta[arm]["stages"]:
            L.append(f"  {arm:6s} {st['run']}: trained to step {st['global_step']}; fp32-scored at "
                     f"{st['scored_steps'] or 'none yet'}")
    for m in ("logged", "e1", "top1"):
        cr, cl = curves["random"][m], curves["llm"][m]
        res["measures"][m], res["gaps"][m] = measures(cr, cl, m), gaps(cr, cl, m)
        L.append(f"\n== {YLAB[m]}: random init {len(cr)} points, LLM init {len(cl)}")
        fmt = (lambda v: f"{100 * v:.2f}%") if m == "top1" else (lambda v: f"{v:.4f}")
        for r in res["measures"][m]:
            if "random_value" not in r:
                L.append(f"  {100 * r['frac']:>4.0f}% (step {r['random_step']:.0f}): {r['status']}")
            elif r["status"] != "reached":
                L.append(f"  {100 * r['frac']:>4.0f}% (step {r['random_step']:.0f}, {fmt(r['random_value'])}): LLM init {r['status']}")
            else:
                L.append(f"  {100 * r['frac']:>4.0f}% (step {r['random_step']:.0f}, {fmt(r['random_value'])}): LLM init at step "
                         f"{r['llm_step']:.0f}; cost ratio random/LLM {r['data_ratio']:.2f} in scenes, {r['flops_ratio']:.2f} in FLOPs")
        g = [x for x in res["gaps"][m] if x["gap_equal_flops"] is not None]
        if g:
            pick = [g[k] for k in sorted({0, len(g) // 4, len(g) // 2, 3 * len(g) // 4, len(g) - 1})]
            L.append("  LLM minus random at equal scenes / equal FLOPs: " + "; ".join(
                f"step {x['llm_step']:.0f}: {x['gap_equal_scenes']:+.4f} / {x['gap_equal_flops']:+.4f}" for x in pick
                if x["gap_equal_scenes"] is not None))
    # the end point, when both final checkpoints are scored
    end = {a: ARM_EVAL / ARMS[a]["stages"][-1]["tag"] / f"checkpoint-{TOTAL_STEPS}" for a in ARMS}
    if all((p / "tokens_router.npz").is_file() for p in end.values()):
        e = paired_endpoint(end["llm"], end["random"])
        res["endpoint"] = e
        L.append(f"\n== End point (both at stage-2 step {TOTAL_STEPS}), fp32, paired over {e['n_rows']:,} rows, "
                 f"{e['n_tokens']:,} image tokens; LLM minus random [95% row bootstrap]")
        L.append(f"  E1    {e['e1']['llm']:.4f} vs {e['e1']['random']:.4f}: {e['e1']['delta']:+.4f} "
                 f"[{e['e1']['ci95'][0]:+.4f}, {e['e1']['ci95'][1]:+.4f}]")
        L.append(f"  top-1 {100 * e['top1']['llm']:.2f}% vs {100 * e['top1']['random']:.2f}%: "
                 f"{100 * e['top1']['delta']:+.2f} pp [{100 * e['top1']['ci95'][0]:+.2f}, {100 * e['top1']['ci95'][1]:+.2f}]")
        L.append(f"  (equal stage-2 steps; the LLM run also had stage 1's {330 * ROW_FLOPS['table'] / ROW_FLOPS['full']:.0f}"
                 f" full-step equivalents)")
    else:
        L.append(f"\n== End point: not yet (checkpoint-{TOTAL_STEPS} not scored at fp32 for both)")
    figure(curves, out / "nc_init_compare")
    (out / "nc_init_compare.json").write_text(json.dumps(res, indent=1))
    (out / "nc_init_compare.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out / 'nc_init_compare.png'} (.pdf, .json, .txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
