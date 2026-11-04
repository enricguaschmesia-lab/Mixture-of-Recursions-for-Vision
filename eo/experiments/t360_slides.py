"""Figures for the T360 deck (docs/slides/t360.md, 2026-11-02): T360 against B10c. `.venv`, CPU.

T360 is the SmolLM-360M shape trained for 2 epochs on TerraMesh's full train split; B10c is the
29-layer vanilla trained for 10 epochs on the 84,672 val rows. Every number is read from a stored
result, nothing is recomputed from a model:
  reports/t360_eval/e1_compare.json      E1: per-modality held-out CE and top-1, fp32, paired by token
  reports/t360_eval/e2_compare.json      E2: the source x target grid, fp32, paired by row
  runs/pretrain/phase3/vanilla10c_20261011_172253/checkpoint-3300/trainer_state.json   B10c's curve
  reports/t360_probe/final/trainer_state.json                                           T360's curve
  reports/ben/summary_t360/summary.json  BEN: test_a micro-mAP, paired CIs and forgetting (ben_t360_summary.py),
                                         and the 50-epoch runs' and stage-2 reruns' val and forgetting per checkpoint
  reports/tm_compare/summary.json        TerraMind's nine generation tasks, every model on the same scenes
                                         (tm_compare_gen.py, tm_compare_eval.py)
The published numbers drawn beside ours (SOFTCON, TM_PUB) are constants, each with its table.
Training FLOPs use curves.py's "full" definition (every position of the 1,048, attention, the output
layer; training = 3 x forward) with each model's own geometry.

Colours: B10c keeps the vanilla orange of every earlier deck; T360 is a deeper blue (#2d5fa8; the pair
passes the dataviz validator: CVD dE 23.2, normal-vision 34.0). Every coloured mark is also labelled.

    .venv/bin/python eo/experiments/t360_slides.py
Writes /data/enric/reports/slides/t360/figs/*.png and numbers.json beside them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midterm_figures as mf  # noqa: E402  (style: rcParams, surface and ink colours, modality names)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REP = Path("/data/enric/reports")
OUT = Path(os.environ.get("T360_FIGS_OUT", REP / "slides/t360/figs"))   # override: smoke tests only
E1 = REP / "t360_eval/e1_compare.json"
E2 = REP / "t360_eval/e2_compare.json"
B10C_STATE = Path("/data/enric/runs/pretrain/phase3/vanilla10c_20261011_172253/checkpoint-3300/trainer_state.json")
T360_STATE = REP / "t360_probe/final/trainer_state.json"

COL = {"b10c": mf.MODEL["b10c"], "t360": "#2d5fa8"}
SURF, INK, INK2, MUTED, GRIDC = mf.SURF, mf.INK, mf.INK2, mf.MUTED, mf.GRID
NICE = mf.NICE
MODS = mf.MODS
TARGETS = mf.TARGETS

# The two runs (configs arm_b10c_vanilla and arm_t360_vanilla; worklog §1).
ROWLEN = 995                     # body tokens per row: 5 x 196 image tokens + 7 location + markers
RUN = {
    "b10c": {"scenes": 84_672, "steps": 3_300, "batch": 256, "epochs": 10, "geom": (576, 29, 3, 1536)},
    "t360": {"scenes": 8_929_841, "steps": 17_441, "batch": 1_024, "epochs": 2, "geom": (960, 32, 5, 2560)},
}
SEQ, VOCAB, HEAD_DIM = 1048, 87_556, 64


def layer_params(d, kv, ff):
    return 2 * d * d + 2 * d * kv * HEAD_DIM + 3 * d * ff + 2 * d


def train_flops(r):
    d, L, kv, ff = r["geom"]
    fwd = 2 * layer_params(d, kv, ff) * L * SEQ + 2 * d * L * SEQ ** 2 + 2 * d * VOCAB * SEQ
    return 3 * fwd * r["batch"] * r["steps"]


def nonemb(r):
    d, L, kv, ff = r["geom"]
    return layer_params(d, kv, ff) * L


assert nonemb(RUN["b10c"]) == 102_662_784 and nonemb(RUN["t360"]) == 314_634_240   # measured counts


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("->", OUT / name)


def evals(path, batch):
    hist = json.loads(Path(path).read_text())["log_history"]
    ev = sorted((h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h)
    s = np.array([e[0] for e in ev]); v = np.array([e[1] for e in ev])
    assert np.isfinite(v).all()
    return s, v, s * batch * ROWLEN / 1e9          # steps, loss, tokens seen (billions)


# ------------------------------------------------------------------------------------------ figures

def fig_setup(num):
    b, t = RUN["b10c"], RUN["t360"]
    rows = [
        ("training scenes", b["scenes"], t["scenes"], "85 k (val split) → 8.9 M (train split)"),
        ("passes over the data", b["epochs"], t["epochs"], "10 epochs → 2"),
        ("scenes per step", b["batch"], t["batch"], "256 → 1,024"),
        ("parameters (layers)", nonemb(b), nonemb(t), "103 M → 315 M (2.6× with the token table)"),
        ("tokens seen", b["steps"] * b["batch"] * ROWLEN, t["steps"] * t["batch"] * ROWLEN, "0.84 B → 17.8 B"),
        ("training compute", train_flops(b), train_flops(t), "0.9 → 48 × 10¹⁸ FLOPs"),
    ]
    num["setup"] = {r[0]: {"b10c": r[1], "t360": r[2], "ratio": r[2] / r[1]} for r in rows}
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, bv, tv, txt) in zip(ys, rows):
        ratio = tv / bv
        ax.plot([1, ratio], [y, y], color=MUTED, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter([1], [y], s=170, color=COL["b10c"], zorder=3, edgecolor=SURF, lw=2)
        ax.scatter([ratio], [y], s=170, color=COL["t360"], zorder=3, edgecolor=SURF, lw=2)
        rtxt = f"{ratio:.1f}×" if ratio < 10 else f"{ratio:.0f}×"
        ax.text(ratio * (1.2 if ratio > 1 else 1 / 1.2), y, rtxt, ha="left" if ratio > 1 else "right",
                va="center", fontsize=18, fontweight="bold", color=INK)
        ax.text(0.042, y + 0.13, lab, ha="right", va="center", fontsize=17, color=INK)
        ax.text(0.042, y - 0.24, txt, ha="right", va="center", fontsize=13.5, color=INK2)
    ax.axvline(1, color=COL["b10c"], lw=1.2, zorder=0, alpha=0.6)
    ax.set_xscale("log"); ax.set_xlim(0.05, 400); ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(["0.1×", "1×", "10×", "100×"])
    ax.set_xlabel("T360 relative to B10c")
    mf.xgrid(ax)
    ax.text(1, len(rows) - 0.4, "B10c", ha="center", va="bottom", fontsize=16, fontweight="bold", color=COL["b10c"])
    save(fig, "setup.png")


def fig_curve(num):
    sb, vb, xb = evals(B10C_STATE, RUN["b10c"]["batch"])
    st, vt, xt = evals(T360_STATE, RUN["t360"]["batch"])
    final_b = float(vb[-1])
    i = int(np.argmax(vt < final_b))                 # first T360 eval below B10c's final
    x0, x1, y0, y1 = xt[i - 1], xt[i], vt[i - 1], vt[i]
    cross = float(x0 + (final_b - y0) * (x1 - x0) / (y1 - y0))   # linear between the two evals
    num["curve"] = {"b10c_final_eval_loss": final_b, "b10c_final_step": int(sb[-1]),
                    "b10c_tokens_B": float(xb[-1]), "t360_last_eval": [int(st[-1]), float(vt[-1])],
                    "t360_tokens_B": RUN["t360"]["steps"] * RUN["t360"]["batch"] * ROWLEN / 1e9,
                    "t360_crosses_b10c_final_at_tokens_B": cross,
                    "t360_evals_bracketing": [[int(st[i - 1]), float(vt[i - 1])], [int(st[i]), float(vt[i])]]}
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    mb, mt = sb > 0, st > 0
    ax.plot(xb[mb], vb[mb], color=COL["b10c"], lw=2.6, marker="o", ms=5, zorder=3)
    ax.plot(xt[mt], vt[mt], color=COL["t360"], lw=2.6, zorder=3)
    ax.axhline(final_b, color=COL["b10c"], lw=1, ls=(0, (4, 3)), alpha=0.8, zorder=1)
    ax.scatter([cross], [final_b], s=110, color=COL["t360"], edgecolor=SURF, lw=2, zorder=4)
    ax.annotate(f"T360 reaches B10c's final loss\nafter {cross:.1f} B tokens",
                (cross, final_b), xytext=(-12, -62), textcoords="offset points", ha="right", fontsize=14.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.text(0.115, 6.86, "B10c\n10 passes over 85 k scenes", ha="left", va="top",
            fontsize=15, color=COL["b10c"], fontweight="bold")
    ax.text(2.4, 4.98, "T360\n2 passes over 8.9 M scenes", ha="left", va="top",
            fontsize=15, color=COL["t360"], fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(0.04, 25); ax.set_ylim(4.6, 6.9)
    ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(["0.1 B", "1 B", "10 B"])
    ax.set_xlabel("training tokens seen"); ax.set_ylabel("held-out loss (nats)")
    mf.ygrid(ax)
    save(fig, "curve.png")


def fig_modalities(num):
    d = json.loads(E1.read_text())
    b, t = d["main"]["b10c"], d["main"]["checkpoint-17441"]
    rows = ["all"] + MODS
    num["e1"] = {m: {"b10c_ce": b[m]["ce"], "t360_ce": t[m]["ce"], "diff": t[m]["vs_b10c"]["ce"]["diff"],
                     "lo": t[m]["vs_b10c"]["ce"]["lo"], "hi": t[m]["vs_b10c"]["ce"]["hi"],
                     "b10c_top1": b[m]["top1"], "t360_top1": t[m]["top1"]} for m in rows}
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[0] += 0.35                                    # "all tokens" set apart at the top
    for y, m in zip(ys, rows):
        bv, tv = b[m]["ce"], t[m]["ce"]
        assert tv < bv and t[m]["vs_b10c"]["ce"]["hi"] < 0
        ax.annotate("", (tv + 0.07, y), (bv, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2,
                                                                 mutation_scale=16, shrinkA=7, shrinkB=0))
        ax.scatter([bv], [y], s=170, color=COL["b10c"], zorder=3, edgecolor=SURF, lw=2)
        ax.scatter([tv], [y], s=170, color=COL["t360"], zorder=3, edgecolor=SURF, lw=2)
        lab = "all tokens" if m == "all" else NICE[m]
        ax.text(0.55, y, lab, ha="right", va="center", fontsize=16, color=INK,
                fontweight="bold" if m == "all" else "normal")
        ax.text(10.6, y, f"{t[m]['vs_b10c']['ce']['diff']:+.2f}".replace("-", "−"), ha="right", va="center",
                fontsize=16, color=INK, fontweight="bold" if m == "all" else "normal")
    ax.text(10.6, ys[0] + 0.75, "T360 − B10c", ha="right", va="bottom", fontsize=14, color=INK2)
    yb = ys[0]
    ax.text(b["all"]["ce"] + 0.22, yb, "B10c", ha="left", va="center", fontsize=15, fontweight="bold", color=COL["b10c"])
    ax.text(t["all"]["ce"] - 0.22, yb, "T360", ha="right", va="center", fontsize=15, fontweight="bold", color=COL["t360"])
    ax.set_xlim(0.6, 10.7); ax.set_ylim(-0.6, ys[0] + 1.1)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("held-out cross-entropy (nats, lower is better)")
    mf.xgrid(ax)
    save(fig, "modalities.png")


def fig_context(num):
    d = json.loads(E2.read_text())
    cb, ct = d["arms"]["B10c"]["cells"], d["arms"]["T360"]["cells"]
    keys = [k for k in ct if "gain" in ct[k]]
    assert len(keys) == 40 and all(ct[k]["gain"]["hi"] < 0 for k in keys)           # every T360 source helps
    assert all(d["split"][k]["d_gain"]["hi"] < 0 for k in keys)                       # and helps more
    odd = [k for k in keys if not cb[k]["gain"]["hi"] < 0]                            # B10c: hurt or nothing
    gx = {k: -cb[k]["gain"]["mean"] for k in keys}; gy = {k: -ct[k]["gain"]["mean"] for k in keys}
    one = [k for k in keys if not k.startswith("all->")]; allk = [k for k in keys if k.startswith("all->")]
    xs = np.array([gx[k] for k in one]); ysv = np.array([gy[k] for k in one])
    num["e2"] = {"n_cells": len(keys), "b10c_odd": odd, "pooled": d["pooled"]["pooled"],
                 "pearson_one_source": float(np.corrcoef(xs, ysv)[0, 1]),
                 "all_others": {k: {"b10c_drop": gx[k], "t360_drop": gy[k],
                                    "d_total": d["paired"][k]["mean"], "d_none": d["split"][k]["d_none_on_rows"]["mean"],
                                    "d_gain": d["split"][k]["d_gain"]["mean"],
                                    "gain_share": d["split"][k]["d_gain"]["mean"] / d["paired"][k]["mean"]} for k in allk},
                 "odd_cells": {k: {"b10c": cb[k]["gain"], "t360": ct[k]["gain"]} for k in odd}}
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    lim = (-0.25, 1.75)
    ax.fill_between(lim, lim, [lim[1]] * 2, color=mf.tint(COL["t360"], 0.9), lw=0, zorder=0)
    ax.plot(lim, lim, color=MUTED, lw=1.2, zorder=1)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.text(-0.18, 1.32, "T360 gets more\nfrom the context", ha="left", va="top", fontsize=15, color=COL["t360"],
            fontweight="bold")
    ax.text(1.62, 1.55, "same", ha="right", va="top", fontsize=13, color=MUTED, rotation=45, rotation_mode="anchor")
    ax.text(-0.02, 1.70, "◀ hurt B10c", ha="right", va="top", fontsize=13, color=INK2)
    for k in one:
        ax.scatter([gx[k]], [gy[k]], s=70, color=COL["t360"], edgecolor=SURF, lw=1.5, zorder=3)
    for k in allk:
        ax.scatter([gx[k]], [gy[k]], s=95, marker="s", facecolor=SURF, edgecolor=COL["t360"], lw=2.2, zorder=3)
    for k in odd:
        ax.scatter([gx[k]], [gy[k]], s=260, facecolor="none", edgecolor=COL["b10c"], lw=2.2, zorder=4)
    lab = {"S1GRD->LULC": ("S1 GRD → LULC", (-0.21, 0.74), "left"), "S2L2A->NDVI": ("S2 → NDVI", (0.52, 1.30), "left"),
           "NDVI->S2L2A": ("NDVI → S2", (1.10, 1.12), "left"), "all->DEM": ("all others → DEM", (0.34, 0.20), "left"),
           "Coords->S1RTC": ("location → 5 targets", (-0.22, -0.14), "left")}
    for k, (txt, xy, ha) in lab.items():
        ax.annotate(txt, (gx[k], gy[k]), xytext=xy, textcoords="data", ha=ha, va="center", fontsize=14, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9, shrinkA=2, shrinkB=6))
    ax.scatter([], [], s=70, color=COL["t360"], label="one source")
    ax.scatter([], [], s=95, marker="s", facecolor=SURF, edgecolor=COL["t360"], lw=2.2, label="all other modalities")
    ax.scatter([], [], s=200, facecolor="none", edgecolor=COL["b10c"], lw=2.2, label="hurt or did nothing in B10c")
    ax.legend(loc="lower right", fontsize=13.5, handletextpad=0.4, borderaxespad=0.2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("B10c: loss drop from the context (nats)")
    ax.set_ylabel("T360: loss drop from the context (nats)")
    save(fig, "context.png")


# ------------------------------------------------------------------------------------------ BEN (Phase 6 protocol)
# reports/ben/summary_t360/summary.json (eo/experiments/ben_t360_summary.py): test_a micro-mAP of the checkpoint each
# model selected on its own val role, paired bootstrap CIs over the 103,944 test patches, forgetting per base.
BEN_SUM = Path(os.environ.get("BEN_SUMMARY", REP / "ben/summary_t360/summary.json"))   # override: smoke tests only
CONDS = ("S1", "S2", "S1S2")
CNAME = {"S1": "S1 (radar)", "S2": "S2 (optical)", "S1S2": "S1 + S2"}
COL_B10 = "#a8461c"    # B10, BEN's original base (B10c's twin on the original tokens): a darker step of B10c's orange;
                       # all-pairs with B10c and T360: CVD dE 13.8, normal-vision 15.7, contrast >= 3:1 (dataviz validator)
SMC = "#3d3c39"                                  # SMARTIES, context (another protocol): dark ink, diamonds


def ben():
    d = json.loads(BEN_SUM.read_text())
    m = d["models"]
    get = lambda t, c: m[t][f"test_{c}"]["micro_map"]
    return d, m, get


def sgn(v, nd=3):
    return f"{v:+.{nd}f}".replace("-", "−")


def dumbbell(ax, y, a, b, ca, cb, diff=None, span=0.2):
    """a -> b on row y: an arrow between two dots, the values below them (b's in bold) and, with `diff`, the change
    above the arrow. When the two nearly coincide, a's dot is drawn larger, so that a ring of it shows."""
    near = abs(b - a) <= 0.03 * span                  # the dots overlap
    if abs(b - a) > 0.09 * span:                      # room for an arrow between them, its head included
        ax.annotate("", (b, y), (a, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2, mutation_scale=16,
                                                        shrinkA=8, shrinkB=8))
    ax.scatter([a], [y], s=320 if near else 190, color=ca, zorder=3, edgecolor=SURF, lw=2)
    ax.scatter([b], [y], s=120 if near else 190, color=cb, zorder=4, edgecolor=SURF, lw=2)
    g, up = (0.012 * span if near else 0.0), b >= a  # near: pull the two labels apart
    ax.text(a - g if up else a + g, y - 0.3, f"{a:.3f}", ha="right" if up else "left", va="top", fontsize=14,
            color=INK2)
    ax.text(b + g if up else b - g, y - 0.3, f"{b:.3f}", ha="left" if up else "right", va="top", fontsize=14,
            color=INK, fontweight="bold")
    if diff is not None:
        ax.text((a + b) / 2, y + 0.2, sgn(diff), ha="center", va="bottom", fontsize=15, color=INK, fontweight="bold")


def solo(ax, y, v, c):
    """One model alone on row y (no twin to compare with): its dot, and its value below in bold, as dumbbell's b."""
    ax.scatter([v], [y], s=190, color=c, zorder=4, edgecolor=SURF, lw=2)
    ax.text(v, y - 0.3, f"{v:.3f}", ha="center", va="top", fontsize=14, color=INK, fontweight="bold")


def fig_ben_inputs(num):
    d, m, get = ben()
    sm = d["smarties"]["ViT-B, 10%"]
    cmp_ = d["comparisons"]["t360_lora - b10c_lora"]
    num["ben_inputs"] = {c: {"b10c_lora": get("b10c_lora", c), "t360_lora": get("t360_lora", c),
                             "b10_lora": get("b10_lora", c), "smarties_vitb": sm[c], "d": cmp_[c]} for c in CONDS}
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ys = np.arange(len(CONDS))[::-1].astype(float)
    x0, x1 = 0.67, 0.885
    col = ax.get_yaxis_transform()                    # x in axes fractions, y in rows: the column right of the plot
    for y, c in zip(ys, CONDS):
        dumbbell(ax, y, get("b10c_lora", c), get("t360_lora", c), COL["b10c"], COL["t360"], span=x1 - x0)
        ax.scatter([sm[c]], [y], s=170, marker="D", color=SMC, zorder=3, edgecolor=SURF, lw=2)
        ax.text(sm[c], y + 0.24, f"{sm[c]:.3f}", ha="center", va="bottom", fontsize=14, color=INK2)
        ax.text(x0 - 0.005, y, CNAME[c], ha="right", va="center", fontsize=16, color=INK)
        ax.text(1.13, y, sgn(cmp_[c]["diff"]), transform=col, ha="right", va="center", fontsize=17, color=INK,
                fontweight="bold")
    ax.text(1.13, ys[0] + 0.62, "T360 − B10c", transform=col, ha="right", va="bottom", fontsize=14, color=INK2)
    ax.scatter([], [], s=150, color=COL["b10c"], label="B10c, LoRA")
    ax.scatter([], [], s=150, color=COL["t360"], label="T360, LoRA")
    ax.scatter([], [], s=130, marker="D", color=SMC, label="SMARTIES ViT-B (another protocol)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.2), ncol=3, fontsize=14, handletextpad=0.3, columnspacing=1.6)
    ax.set_xlim(x0, x1); ax.set_ylim(-0.75, len(CONDS) - 0.25)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("micro-mAP on the 103,944 test patches")
    mf.xgrid(ax)
    save(fig, "ben_inputs.png")


def fig_ben_regimes(num):
    """S1 + S2 micro-mAP and forgetting per training regime, B10 -> T360. B10c is left out: its LoRA run equals B10's
    (summary: +0.000 on S1 + S2, forgetting 0.267 vs 0.278), and its dot would hide B10's. SMARTIES ViT-B is drawn on
    the head-only row alone: its S1 + S2 number is a linear probe of a frozen encoder, the regime head-only is closest to.
    T360's full fine-tune at 3e-5 (worklog item 48) gets a row of its own, T360 alone: B10 was not run at 3e-5."""
    d, m, get = ben()
    lr = "t360_full_lr3e5" in m and "test_S1S2" in m["t360_full_lr3e5"]
    rows = [(r, b, t) for r, b, t in [("head only", "b10_head", "t360_head"), ("LoRA", "b10_lora", "t360_lora"),
                                      ("full, LR 1e-4" if lr else "full fine-tune", "b10_full", "t360_full"),
                                      ("full, LR 3e-5", None, "t360_full_lr3e5")] if t in m and "test_S1S2" in m[t]]
    fo = lambda t: m[t]["forgetting"]["ce_all"]
    sm = d["smarties"]["ViT-B, 10%"]["S1S2"]
    num["ben_regimes"] = {r: {"t360": get(t, "S1S2"), "forget_t360": fo(t)} | ({} if b is None else {
        "b10": get(b, "S1S2"), "d": d["comparisons"][f"{t} - {b}"]["S1S2"], "forget_b10": fo(b)}) for r, b, t in rows}
    num["ben_regimes"]["head only"]["smarties_vitb_linear_probe"] = sm
    if lr:
        num["ben_regimes"]["full, LR 3e-5"] |= {k: d["comparisons"][f"t360_full_lr3e5 - {o}"]["S1S2"]
                                                for k, o in (("d_vs_lr1e4", "t360_full"), ("d_vs_lora", "t360_lora"))}
    h = 1.2 + (1.15 if len(rows) <= 3 else 0.95) * len(rows)       # a fourth row in the page's height
    fig, (ax, af) = plt.subplots(1, 2, figsize=(13.5, h), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.1})
    ys = np.arange(len(rows))[::-1].astype(float)
    fmax = max(fo(x) for _, b, t in rows for x in (b, t) if x)
    vmin = min(get(x, "S1S2") for _, b, t in rows for x in (b, t) if x)
    (x0, x1), (f0, f1) = (min(0.70, np.floor(100 * vmin - 2) / 100), 0.88), (-0.04 * max(1.0, fmax / 0.3), max(0.34, 1.1 * fmax))
    for y, (r, b, t) in zip(ys, rows):
        if b is None:
            solo(ax, y, get(t, "S1S2"), COL["t360"])
            solo(af, y, fo(t), COL["t360"])
        else:
            dumbbell(ax, y, get(b, "S1S2"), get(t, "S1S2"), COL_B10, COL["t360"],
                     diff=d["comparisons"][f"{t} - {b}"]["S1S2"]["diff"], span=x1 - x0)
            dumbbell(af, y, fo(b), fo(t), COL_B10, COL["t360"], diff=fo(t) - fo(b), span=f1 - f0)
        ax.text(x0 - 0.004, y, r, ha="right", va="center", fontsize=16, color=INK)
        if r == "head only":
            ax.scatter([sm], [y], s=170, marker="D", color=SMC, zorder=3, edgecolor=SURF, lw=2)
            ax.text(sm, y + 0.24, f"{sm:.3f}", ha="center", va="bottom", fontsize=14, color=INK2)
    ax.set_xlim(x0, x1); ax.set_xlabel("micro-mAP, S1 + S2 (higher is better)")
    af.set_xlim(f0, f1); af.set_xlabel("forgetting (nats; lower is better)")
    for a in (ax, af):
        a.set_ylim(-0.75, len(rows) - 0.25); a.set_yticks([]); a.spines["left"].set_visible(False)
        mf.xgrid(a)
    ax.scatter([], [], s=150, color=COL_B10, label="B10 (BEN's small model)")
    ax.scatter([], [], s=150, color=COL["t360"], label="T360")
    ax.scatter([], [], s=130, marker="D", color=SMC, label="SMARTIES ViT-B, linear probe of a frozen encoder")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=14, handletextpad=0.3, columnspacing=1.6,
              borderaxespad=0.1)
    save(fig, "ben_regimes.png")


def fig_ben_labels(num):
    """S1 + S2 micro-mAP at 10% and 1% of the labels, B10 -> T360 (LoRA), in fig_ben_inputs' layout. B10, not B10c:
    only B10 was run at 1%, and at 10% the two score the same (0.811)."""
    d, m, get = ben()
    loc = {"10%": 0.7820, "1%": 0.6843}          # location-only kNN (k = 10), reports/ben/t360_plan/leak_by_budget.txt
    sml = {"10%": d["smarties"]["ViT-L, 10%"]["S1S2"], "1%": d["smarties"]["ViT-L, 1%"]["S1S2"]}
    rows = [("10%", "b10_lora", "t360_lora", "10% of the labels\n31,166 patches"),
            ("1%", "b10_lora_1pct", "t360_lora_1pct", "1% of the labels\n3,116 patches")]
    num["ben_labels"] = {k: {"b10_lora": get(b, "S1S2"), "t360_lora": get(t, "S1S2"), "location_knn10": loc[k],
                             "smarties_vitl": sml[k], "d": d["comparisons"][f"{t} - {b}"]["S1S2"]} for k, b, t, _ in rows}
    fig, ax = plt.subplots(figsize=(12.5, 3.5))
    ys = np.arange(len(rows))[::-1].astype(float)
    x0, x1 = 0.665, 0.885
    col = ax.get_yaxis_transform()
    for y, (k, b, t, lab) in zip(ys, rows):
        dumbbell(ax, y, get(b, "S1S2"), get(t, "S1S2"), COL_B10, COL["t360"], span=x1 - x0)
        ax.scatter([sml[k]], [y], s=170, marker="D", color=SMC, zorder=3, edgecolor=SURF, lw=2)
        ax.text(sml[k], y + 0.24, f"{sml[k]:.3f}", ha="center", va="bottom", fontsize=14, color=INK2)
        ax.scatter([loc[k]], [y], s=420, marker="|", color=MUTED, zorder=2, lw=3)
        ax.text(loc[k], y + 0.24, f"{loc[k]:.3f}", ha="center", va="bottom", fontsize=14, color=MUTED)
        ax.text(x0 - 0.005, y, lab, ha="right", va="center", fontsize=16, color=INK, linespacing=1.3)
        ax.text(1.13, y, sgn(d["comparisons"][f"{t} - {b}"]["S1S2"]["diff"]), transform=col, ha="right", va="center",
                fontsize=17, color=INK, fontweight="bold")
    ax.text(1.13, ys[0] + 0.62, "T360 − B10", transform=col, ha="right", va="bottom", fontsize=14, color=INK2)
    ax.scatter([], [], s=150, color=COL_B10, label="B10, LoRA")
    ax.scatter([], [], s=150, color=COL["t360"], label="T360, LoRA")
    ax.scatter([], [], s=130, marker="D", color=SMC, label="SMARTIES ViT-L, linear probe")
    ax.scatter([], [], s=300, marker="|", color=MUTED, lw=3, label="location only")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.3), ncol=4, fontsize=14, handletextpad=0.3, columnspacing=1.6)
    ax.set_xlim(x0, x1); ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("micro-mAP, S1 + S2, on the 103,944 test patches")
    mf.xgrid(ax)
    save(fig, "ben_labels.png")


#: Published BEN numbers on our exact test split (SeCo's 103,944 val patches, 10% of the labels, micro-mAP), checked
#: against the papers on 2026-11-03 (docs notes/sota_compare_2026-11-03.md). SMARTIES' come from summary.json.
SOFTCON = {("S1", "LP"): 0.825, ("S2", "LP"): 0.868, ("S2", "FT"): 0.887}   # arXiv 2405.20462, Tables II and I (ViT-B)


def fig_ben_published(num):
    """T360 against the best published BEN models on the same 103,944 test patches, per input: SoftCon ViT-B (the best
    on S1 and S2) and SMARTIES ViT-L and ViT-B (the best on S1 + S2). Mode is the marker's fill: a linear probe of a
    frozen encoder hollow, a fine-tune filled; T360's head-only run (frozen body, only the 21 label rows train) is
    hollow too, its LoRA run filled. SMARTIES' S1 and S1 + S2 are linear probes and its S2 a fine-tune (Table 1)."""
    d, m, get = ben()
    sm = {"SMARTIES ViT-L": d["smarties"]["ViT-L, 10%"], "SMARTIES ViT-B": d["smarties"]["ViT-B, 10%"]}
    mode = {"S1": "LP", "S2": "FT", "S1S2": "LP"}
    rows = [("SoftCon ViT-B", {(c, md): v for (c, md), v in SOFTCON.items()}),
            *[(k, {(c, mode[c]): v[c] for c in CONDS}) for k, v in sm.items()],
            ("T360 (ours)", {**{(c, "LP"): get("t360_head", c) for c in CONDS},
                             **{(c, "FT"): get("t360_lora", c) for c in CONDS}})]
    best = {c: max(((v, k, md) for k, dd in rows[:-1] for (cc, md), v in dd.items() if cc == c)) for c in CONDS}
    num["ben_published"] = {"rows": {k: {f"{c} {md}": v for (c, md), v in dd.items()} for k, dd in rows},
                            "best": {c: {"value": v, "model": k, "mode": md, "t360_lora_minus_best":
                                         get("t360_lora", c) - v} for c, (v, k, md) in best.items()}}
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True, gridspec_kw={"wspace": 0.08})
    ys = np.arange(len(rows))[::-1].astype(float)
    for ax, c in zip(axs, CONDS):
        lo = {"S1": 0.64, "S2": 0.70, "S1S2": 0.70}[c]
        for y, (k, dd) in zip(ys, rows):
            ours = k.startswith("T360")
            col = COL["t360"] if ours else SMC
            vals = [dd[(c, md)] for md in ("LP", "FT") if (c, md) in dd]
            close = len(vals) == 2 and abs(vals[1] - vals[0]) < 0.03     # labels beside the dots, not above
            for md in ("LP", "FT"):
                if (c, md) not in dd:
                    continue
                v = dd[(c, md)]
                ax.scatter([v], [y], s=150, marker="o" if ours else "D", zorder=3, lw=2.2,
                           facecolor=col if md == "FT" else SURF, edgecolor=col)
                kw = dict(fontsize=13, color=INK if ours and md == "FT" else INK2,
                          fontweight="bold" if ours and md == "FT" else None)
                if close:
                    hi = v == max(vals)
                    ax.text(v + (0.007 if hi else -0.007), y, f"{v:.3f}", ha="left" if hi else "right", va="center", **kw)
                else:
                    ax.text(v, y + 0.22, f"{v:.3f}", ha="center", va="bottom", **kw)
        v, k, md = best[c]
        ax.set_title(f"{CNAME[c]}\nT360 {sgn(get('t360_lora', c) - v)} to the best", fontsize=15, color=INK, pad=8)
        ax.set_xlim(lo, lo + 0.225)                       # one scale in all three panels
        ax.set_ylim(-0.6, len(rows) - 0.35)
        ax.tick_params(axis="y", length=0)
        mf.xgrid(ax)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels([k for k, _ in rows], fontsize=15)
    for t in axs[0].get_yticklabels():
        if t.get_text().startswith("T360"):
            t.set_color(COL["t360"]); t.set_fontweight("bold")
    axs[1].set_xlabel("micro-mAP on the 103,944 test patches, 10% of the labels")
    for a in axs[1:]:
        a.spines["left"].set_visible(False)
    h = [axs[0].scatter([], [], s=130, marker="D", facecolor=SURF, edgecolor=SMC, lw=2.2, label="linear probe (frozen)"),
         axs[0].scatter([], [], s=130, marker="D", color=SMC, label="fine-tune"),
         axs[0].scatter([], [], s=150, facecolor=SURF, edgecolor=COL["t360"], lw=2.2, label="T360, frozen body"),
         axs[0].scatter([], [], s=150, color=COL["t360"], label="T360, LoRA (3 epochs)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.52, 0.99), ncol=4, fontsize=14, handletextpad=0.3,
               columnspacing=1.6, frameon=False)
    save(fig, "ben_published.png")


COL_FULL = "#2e8b3e"   # T360's full fine-tunes on the budget page (LoRA keeps T360's blue): CVD dE 22.3, normal-vision 23.7
#: the 50-epoch runs (notes/ben_t360_budget_2026-11-03.md): tag, the 3-epoch run it is paired with, label, colour, lower LR
E50 = [("t360_lora_e50", "t360_lora", "LoRA, 1e-3", COL["t360"], False),
       ("t360_lora_e50_lr3e4", "t360_lora", "LoRA, 3e-4", COL["t360"], True),      # no 3-epoch run at 3e-4: LoRA 1e-3's
       ("t360_full_e50", "t360_full", "full, 1e-4", COL_FULL, False),
       ("t360_full_e50_lr3e5", "t360_full_lr3e5", "full, 3e-5", COL_FULL, True)]
SPE = 487                                          # steps per epoch: ceil(31,166 / 64)


def fig_ben_budget(num):
    """50 epochs instead of 3. Left: val (mean micro-mAP over S1 / S2 / S1 + S2 on ft_valsel) at every validated
    checkpoint of the four 50-epoch runs, the checkpoint val selected ringed, and the 3-epoch runs' three points (dotted)
    for scale; the last 20% (epochs 40-50) is the LR decay; a point below the axis sits on its floor as a down-triangle,
    with its value. Right: test_a S1 + S2 micro-mAP, each 50-epoch run against
    the 3-epoch run it is paired with (LoRA at 3e-4 against LoRA at 1e-3), with the paired difference; SMARTIES ViT-L's
    probe (the best published S1 + S2 on these patches) as a line."""
    d, m, get = ben()
    runs = [r for r in E50 if r[0] in m and m[r[0]].get("val_mean_micro_map")]
    ep = lambda ck: int(ck.split("-")[1]) / SPE
    curve = lambda t: sorted((ep(k), v) for k, v in m[t]["val_mean_micro_map"].items())
    num["ben_budget"] = {t: {"val": {f"{e:g}": v for e, v in curve(t)}, "selected_epoch": ep(m[t]["selected"]),
                             "twin": tw, "twin_val": {f"{e:g}": v for e, v in curve(tw)}}
                         for t, tw, *_ in runs}
    fig, (ax, ar) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.42})
    lo = 0.60
    ax.axvspan(40, 50, color=GRIDC, alpha=0.7, lw=0, zorder=0)
    h = []
    for t, tw, lab, col, low in runs:
        xs, ys = zip(*curve(t))
        h += ax.plot(xs, np.maximum(ys, lo), color=col, ls="--" if low else "-", lw=2.2, marker="o", ms=6.5,
                     mfc=SURF if low else col, mec=col, mew=1.8, zorder=3, label=lab, clip_on=False)
        for x, y in zip(xs, ys):                       # off the axis: a down-triangle on its floor, with its value
            if y < lo:
                ax.scatter([x], [lo], marker="v", s=70, color=col, zorder=5, clip_on=False)
                ax.text(x + 0.7, lo + 0.004, f"{y:.2f}", ha="left", va="bottom", fontsize=12, color=INK2)
        sel = m[t]["selected"]
        ax.scatter([ep(sel)], [m[t]["val_mean_micro_map"][sel]], s=260, facecolor="none", edgecolor=INK, lw=1.6, zorder=4,
                   clip_on=False)
        if not low:                                    # the 3-epoch run at this LR, for scale
            tx, ty = zip(*curve(tw))
            ax.plot(tx, ty, color=col, ls=":", lw=1.6, marker="s", ms=5, mfc=col, mec=col, zorder=3)
    vals = [v for t, *_ in runs for _, v in curve(t)]
    hi = max(vals)
    ax.set_xlim(0, 50); ax.set_ylim(lo, hi + 0.03)
    ax.text(45, hi + 0.026, "LR decay", ha="center", va="top", fontsize=13, color=INK2)
    ax.set_xlabel("epoch"); ax.set_ylabel("val micro-mAP (mean of 3 inputs)")
    mf.ygrid(ax)
    h += [ax.plot([], [], color=INK2, ls=":", lw=1.6, marker="s", ms=5, label="3-epoch run")[0],
          ax.scatter([], [], s=200, facecolor="none", edgecolor=INK, lw=1.6, label="selected on val")]
    ax.legend(handles=h, loc="lower right", ncol=2, fontsize=13, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.3)
    # right: test S1 + S2, 3 epochs -> 50 epochs
    sm = d["smarties"]["ViT-L, 10%"]["S1S2"]
    tr = [(t, tw, lab, col) for t, tw, lab, col, _ in runs if "test_S1S2" in m[t]]
    ys2 = np.arange(len(tr))[::-1].astype(float)
    x0 = min(0.74, np.floor(100 * min(min(get(t, "S1S2"), get(tw, "S1S2")) for t, tw, *_ in tr) - 1) / 100) if tr else 0.74
    x1 = 0.885
    for y, (t, tw, lab, col) in zip(ys2, tr):
        c = d["comparisons"].get(f"{t} - {tw}", {}).get("S1S2")
        dumbbell(ar, y, get(tw, "S1S2"), get(t, "S1S2"), MUTED, col, diff=c["diff"] if c else None, span=x1 - x0)
        ar.text(x0 - 0.004, y, lab, ha="right", va="center", fontsize=15, color=col, fontweight="bold")
        num["ben_budget"][t] |= {"test_S1S2": get(t, "S1S2"), "twin_test_S1S2": get(tw, "S1S2"), "d": c}
    ar.axvline(sm, color=SMC, ls=(0, (4, 3)), lw=1.6, zorder=1)
    ar.text(sm - 0.003, len(tr) + 0.1, f"SMARTIES ViT-L probe {sm:.3f}", ha="right", va="top", fontsize=12, color=INK2)
    ar.set_xlim(x0, x1); ar.set_ylim(-0.75, len(tr) + 0.15)
    ar.set_yticks([]); ar.spines["left"].set_visible(False)
    ar.set_xlabel("test micro-mAP, S1 + S2\n(grey: 3 epochs; colour: 50 epochs)")
    mf.xgrid(ar)
    save(fig, "ben_budget.png")


#: stage 2 (notes/ben_t360_budget_2026-11-03.md §5): each LR rerun with its budget ending at its 50-epoch run's val peak.
#: tag, its 50-epoch run, the 3-epoch run at its LR (None: none was run), the 3-epoch run it is tested against, label, colour
S2 = [("t360_lora_e6", "t360_lora_e50", "t360_lora", "t360_lora", "LoRA, 1e-3", COL["t360"]),
      ("t360_lora_e10_lr3e4", "t360_lora_e50_lr3e4", None, "t360_lora", "LoRA, 3e-4", COL["t360"]),
      ("t360_full_e6", "t360_full_e50", "t360_full", "t360_full", "full, 1e-4", COL_FULL),
      ("t360_full_e10_lr3e5", "t360_full_e50_lr3e5", "t360_full_lr3e5", "t360_full_lr3e5", "full, 3e-5", COL_FULL)]


def fig_ben_stage2(num):
    """The decay at the peak. Left, one panel per LR: val (mean micro-mAP over S1 / S2 / S1 + S2 on ft_valsel) of the
    stage-2 run at every epoch (its last 20%, shaded, is the LR decay; the checkpoint val selected ringed), of the
    50-epoch run at its validated epochs up to the same budget (no decay yet), and of the 3-epoch run at that LR. Points
    below the axis sit on its floor as down-triangles. Right: test_a S1 + S2 micro-mAP of the three, with the stage-2
    run's value and its paired difference to the 3-epoch run it is tested against (LoRA at 3e-4: LoRA at 1e-3's), and
    SMARTIES' probes (the best published S1 + S2 on these patches) as lines."""
    d, m, get = ben()
    runs = [r for r in S2 if r[0] in m and m[r[0]].get("val_mean_micro_map")]
    ep = lambda ck: int(ck.split("-")[1]) / SPE
    curve = lambda t: sorted((ep(k), v) for k, v in m[t]["val_mean_micro_map"].items()) if t in m else []
    test = lambda t: get(t, "S1S2") if t and "test_S1S2" in m.get(t, {}) else None
    num["ben_stage2"] = {t: {"val": {f"{e:g}": v for e, v in curve(t)}, "selected_epoch": ep(m[t]["selected"]),
                             "e50": e50, "same_lr_3ep": tw} for t, e50, tw, *_ in runs}
    fig = plt.figure(figsize=(14.5, 6.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.42)
    gl = outer[0].subgridspec(2, 2, wspace=0.28, hspace=0.55)
    lo, hi = 0.66, 0.84
    for i, (t, e50, tw, _, lab, col) in enumerate(S2):
        ax = fig.add_subplot(gl[i // 2, i % 2])
        n = int(round(max((e for e, _ in curve(t)), default=10 if "e10" in t else 6)))
        ax.axvspan(0.8 * n, n, color=GRIDC, alpha=0.7, lw=0, zorder=0)
        if i == 0:
            ax.text(0.9 * n, lo + 0.005, "LR\ndecay", ha="center", va="bottom", fontsize=11, color=INK2)
        off = False
        for pts, c, ls, mk, mfc, lw, ms in [(curve(tw) if tw else [], INK2, ":", "s", INK2, 1.6, 5),
                                            ([(e, v) for e, v in curve(e50) if e <= n + 1e-9], col, "--", "o", SURF, 1.6, 6),
                                            (curve(t), col, "-", "o", col, 2.4, 7)]:
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, np.maximum(ys, lo), color=c, ls=ls, lw=lw, marker=mk, ms=ms, mfc=mfc, mec=c, mew=1.6, zorder=3,
                    clip_on=False)
            below = [x for x, y in pts if y < lo]      # off the axis: a down-triangle on its floor
            if below:
                off = True
                ax.scatter(below, [lo] * len(below), marker="v", s=60, color=c, zorder=5, clip_on=False)
        if off:
            ax.text(n + 0.3, lo + 0.006, f"▼ below {lo:.2f}", ha="right", va="bottom", fontsize=11, color=INK2)
        if t in m and m[t].get("selected"):
            s = m[t]["selected"]
            ax.scatter([ep(s)], [m[t]["val_mean_micro_map"][s]], s=230, facecolor="none", edgecolor=INK, lw=1.5, zorder=4)
        ax.set_xlim(0, n + 0.4); ax.set_ylim(lo, hi)
        ax.set_xticks(range(0, n + 1, 2))
        ax.set_title(lab, fontsize=15, color=col, fontweight="bold", pad=6)
        mf.ygrid(ax)
        if i % 2 == 0:
            ax.set_ylabel("val micro-mAP")
        if i // 2 == 1:
            ax.set_xlabel("epoch")
    # right: test S1 + S2 for the three, per LR
    ar = fig.add_subplot(outer[1])
    rows = [r for r in S2 if test(r[0]) is not None]
    ys = np.arange(len(rows))[::-1].astype(float)
    sm = {k: d["smarties"][k]["S1S2"] for k in ("ViT-B, 10%", "ViT-L, 10%")}
    vals = [v for t, e50, tw, *_ in rows for v in (test(t), test(e50), test(tw)) if v is not None]
    x0, x1 = min(0.74, np.floor(100 * min(vals, default=0.74) - 1) / 100), 0.885
    for y, (t, e50, tw, ref, lab, col) in zip(ys, rows):
        v, c = test(t), d["comparisons"].get(f"{t} - {ref}", {}).get("S1S2")
        if test(e50) is not None:
            ar.scatter([test(e50)], [y], s=150, facecolor=SURF, edgecolor=col, lw=2.2, zorder=3)
        ar.scatter([v], [y], s=190, color=col, zorder=4, edgecolor=SURF, lw=2)
        if test(ref) is not None:                      # the run the bracket compares with; on top, to show when dots coincide
            ar.scatter([test(ref)], [y], marker="s", s=70, color=INK2, zorder=5, edgecolor=SURF, lw=1.2)
        ar.text(x0 - 0.004, y - 0.2, f"{v:.3f}" + (f" ({sgn(c['diff'])})" if c else ""), ha="right", va="center",
                fontsize=14, color=INK, fontweight="bold")          # under the row's label, clear of the dots
        ar.text(x0 - 0.004, y + 0.17, lab, ha="right", va="center", fontsize=15, color=col, fontweight="bold")
        num["ben_stage2"][t] |= {"test_S1S2": v, "e50_test_S1S2": test(e50), "same_lr_3ep_test_S1S2": test(tw),
                                 "vs": ref, "d": c}
    for k, ls in (("ViT-B, 10%", (0, (1, 2))), ("ViT-L, 10%", (0, (4, 3)))):
        ar.axvline(sm[k], color=SMC, ls=ls, lw=1.6, zorder=1)
    ar.set_xlim(x0, x1); ar.set_ylim(-0.7, len(rows) - 0.4)
    ar.set_yticks([]); ar.spines["left"].set_visible(False)
    ar.set_xlabel("test micro-mAP, S1 + S2\n(in brackets: change vs 3 epochs)")
    mf.xgrid(ar)
    h = [plt.Line2D([], [], color=INK2, ls=":", lw=1.6, marker="s", ms=5, label="3 epochs"),
         plt.Line2D([], [], color=INK2, ls="--", lw=1.6, marker="o", ms=6, mfc=SURF, label="50 epochs (no decay yet; right: its best)"),
         plt.Line2D([], [], color=INK2, ls="-", lw=2.4, marker="o", ms=7, label="decay at the peak"),
         plt.Line2D([], [], ls="", marker="o", ms=14, mfc="none", mec=INK, mew=1.5, label="selected on val"),
         plt.Line2D([], [], color=SMC, ls=(0, (1, 2)), lw=1.6, label=f"SMARTIES ViT-B probe {sm['ViT-B, 10%']:.3f}"),
         plt.Line2D([], [], color=SMC, ls=(0, (4, 3)), lw=1.6, label=f"SMARTIES ViT-L probe {sm['ViT-L, 10%']:.3f}")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=13, frameon=False,
               handletextpad=0.5, columnspacing=1.6)
    save(fig, "ben_stage2.png")


TMC_SUM = Path(os.environ.get("TMC_SUMMARY", REP / "tm_compare/summary.json"))   # override: smoke tests only
#: TerraMind's nine generation tasks in its Table 3's order; S2L2A as S2.
TM_TASKS = ["S1GRD-S2L2A", "S1GRD-DEM", "S1GRD-NDVI", "S1RTC-S2L2A", "S1RTC-DEM", "S1RTC-NDVI",
            "S2L2A-S1GRD", "S2L2A-S1RTC", "S2L2A-DEM"]
TM_PUB = {  # TerraMind's published MAE: Table 3 (pixel inputs) and Table 14 (token inputs), arXiv 2504.11171
    "S1GRD-S2L2A": (0.074, 0.0820), "S1GRD-DEM": (163.0, 327.4), "S1GRD-NDVI": (0.180, 0.1949),
    "S1RTC-S2L2A": (0.113, 0.1195), "S1RTC-DEM": (298.8, 457.9), "S1RTC-NDVI": (0.172, 0.1895),
    "S2L2A-S1GRD": (2.942, 3.3180), "S2L2A-S1RTC": (2.636, 3.0544), "S2L2A-DEM": (215.8, 572.5)}
TMC = "#7a7974"      # TerraMind: muted ink, diamonds (the other models' marker, as SMARTIES' on the BEN pages). A neutral
                     # by design, so it fails the validator's chroma floor; against T360: CVD dE 14.2, normal-vision 16.1


def task_name(k):
    s, t = k.split("-")
    nm = {"S2L2A": "S2", "S1GRD": "S1 GRD", "S1RTC": "S1 RTC", "DEM": "DEM", "NDVI": "NDVI"}
    return f"{nm[s]} → {nm[t]}"


def fig_gen_terramind(num):
    """T360 against TerraMind on TerraMind's nine generation tasks, on the same 256 held-out scenes per task, every
    model's tokens decoded by the same TerraMind decoder (10 steps): MAE in physical units as a ratio to TerraMind-B
    from pixels (its native input, Table 3's), on a log axis. TerraMind-B from TOKENS is the like-for-like model (the
    input T360 gets; Table 14's setting). T360 samples as TerraMind does (temperature 1, top-p 0.8). The ceiling is the
    true tokens decoded; the level to beat is another scene's true tokens (a sharp but unrelated image)."""
    d = json.loads(TMC_SUM.read_text())["pairs"]
    tasks = [k for k in TM_TASKS if k in d]
    rat = lambda k, c: d[k]["conditions"][c]["mae"] / d[k]["conditions"]["tm_base"]["mae"]
    num["gen_terramind"] = {k: {c: d[k]["conditions"][c]["mae"] for c in d[k]["conditions"]}
                            | {"paired_mae": {p: v["mae"] for p, v in d[k]["paired"].items()},
                               "published_tm_pixels_tokens": TM_PUB[k], "n": d[k]["n"]} for k in tasks}
    fig, ax = plt.subplots(figsize=(12.5, 4.9))
    grp = {"S1GRD": 2, "S1RTC": 1, "S2L2A": 0}                     # a gap between the three sources
    ys = np.arange(len(tasks))[::-1].astype(float) + np.array([0.5 * grp[k.split("-")[0]] for k in tasks])
    x0, x1 = 0.3, 5.7
    col = ax.get_yaxis_transform()                    # x in axes fractions, y in rows: the column right of the plot
    unit = {"S1GRD": " dB", "S1RTC": " dB", "DEM": " m"}
    for y, k in zip(ys, tasks):
        c = d[k]["conditions"]
        lo, hi = rat(k, "ceiling"), rat(k, "wrong_scene")
        ax.plot([max(lo, x0), hi], [y, y], color=GRIDC, lw=6, solid_capstyle="round", zorder=1)
        if lo >= x0:
            ax.scatter([lo], [y], s=260, marker="|", color=MUTED, lw=3, zorder=2)
        else:                                         # off the axis: an arrowhead at the edge, with its value
            ax.scatter([x0 * 1.02], [y], s=110, marker="<", color=MUTED, zorder=2)
            ax.text(x0 * 1.07, y + 0.2, f"{lo:.2f}×", ha="left", va="bottom", fontsize=12, color=MUTED)
        ax.scatter([hi], [y], s=90, marker="x", color=MUTED, lw=2.5, zorder=2)
        dy = 0.17                                            # TerraMind just below, T360 just above: never hidden
        if "tm_large" in c:
            ax.scatter([rat(k, "tm_large")], [y - dy], s=120, marker="D", color=TMC, zorder=3, edgecolor=SURF, lw=1.5)
        ax.scatter([rat(k, "tm_base_tok")], [y - dy], s=120, marker="D", facecolor=SURF, edgecolor=TMC, lw=2.2,
                   zorder=3)
        ax.scatter([rat(k, "t360_p08")], [y + dy], s=190, color=COL["t360"], zorder=4, edgecolor=SURF, lw=2)
        ax.text(-0.01, y, task_name(k), transform=col, ha="right", va="center", fontsize=15, color=INK)
        tgt = k.split("-")[1]
        fmt = (lambda v: f"{v:.0f}") if tgt == "DEM" else (lambda v: f"{v:.3f}" if v < 1 else f"{v:.2f}")
        ax.text(1.24, y, f"{fmt(c['t360_p08']['mae'])} / {fmt(c['tm_base']['mae'])}{unit.get(tgt, '')}",
                transform=col, ha="right", va="center", fontsize=14, color=INK2)
    ax.axvline(1.0, color=TMC, lw=2, zorder=1)
    ax.text(1.03, ys[0] + 0.55, "TerraMind-B from pixels", ha="left", va="bottom", fontsize=13, color=TMC)
    ax.text(1.24, ys[0] + 0.55, "MAE: T360 / TerraMind-B (pixels)", transform=col, ha="right", va="bottom",
            fontsize=13, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_xticks([0.5, 1, 2, 3]); ax.set_xticklabels(["0.5×", "1×", "2×", "3×"])
    ax.minorticks_off()
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.05)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("MAE relative to TerraMind-B from pixels (log scale; left is better)")
    mf.xgrid(ax)
    h = [ax.scatter([], [], s=150, color=COL["t360"], label="T360, from tokens"),
         ax.scatter([], [], s=110, marker="D", facecolor=SURF, edgecolor=TMC, lw=2.2, label="TerraMind-B, from tokens"),
         ax.scatter([], [], s=110, marker="D", color=TMC, label="TerraMind-L, from pixels"),
         ax.scatter([], [], s=220, marker="|", color=MUTED, lw=3, label="true tokens decoded"),
         ax.scatter([], [], s=80, marker="x", color=MUTED, lw=2.5, label="another scene")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=5, fontsize=13, handletextpad=0.2,
              columnspacing=1.1, frameon=False, borderaxespad=0.1)
    save(fig, "gen_terramind.png")


def main():
    num = {}
    fig_setup(num)
    fig_curve(num)
    fig_modalities(num)
    fig_context(num)
    if BEN_SUM.exists():                       # each BEN figure once its models have test results
        m = json.loads(BEN_SUM.read_text())["models"]
        have = lambda *tags: all("test_S1S2" in m[t] for t in tags)
        if have("b10c_lora", "t360_lora"):
            fig_ben_inputs(num)
        if have("b10_lora", "t360_lora", "b10c_lora"):
            fig_ben_regimes(num)
        if have("b10_lora_1pct", "t360_lora_1pct", "b10_lora", "t360_lora"):
            fig_ben_labels(num)
        if have("t360_head", "t360_lora"):
            fig_ben_published(num)
        if any(t in m and m[t].get("val_mean_micro_map") for t, *_ in E50):
            fig_ben_budget(num)
        if any(t in m and m[t].get("val_mean_micro_map") for t, *_ in S2):
            fig_ben_stage2(num)
    if TMC_SUM.exists():
        fig_gen_terramind(num)
    (OUT / "numbers.json").write_text(json.dumps(num, indent=1, default=float))
    print("->", OUT / "numbers.json")


if __name__ == "__main__":
    main()
