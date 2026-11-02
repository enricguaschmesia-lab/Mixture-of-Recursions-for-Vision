"""Figures for the T360 deck (docs/slides/t360.md, 2026-11-02): T360 against B10c. `.venv`, CPU.

T360 is the SmolLM-360M shape trained for 2 epochs on TerraMesh's full train split; B10c is the
29-layer vanilla trained for 10 epochs on the 84,672 val rows. Every number is read from a stored
result, nothing is recomputed from a model:
  reports/t360_eval/e1_compare.json      E1: per-modality held-out CE and top-1, fp32, paired by token
  reports/t360_eval/e2_compare.json      E2: the source x target grid, fp32, paired by row
  runs/pretrain/phase3/vanilla10c_20261011_172253/checkpoint-3300/trainer_state.json   B10c's curve
  reports/t360_probe/final/trainer_state.json                                           T360's curve
  reports/ben/summary_t360/summary.json  BEN: test_a micro-mAP, paired CIs and forgetting (ben_t360_summary.py)
Training FLOPs use curves.py's "full" definition (every position of the 1,048, attention, the output
layer; training = 3 x forward) with each model's own geometry.

Colours: B10c keeps the vanilla orange of every earlier deck; T360 is a deeper blue (#2d5fa8; the pair
passes the dataviz validator: CVD dE 23.2, normal-vision 34.0). Every coloured mark is also labelled.

    .venv/bin/python eo/experiments/t360_slides.py
Writes /data/enric/reports/slides/t360/figs/*.png and numbers.json beside them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midterm_figures as mf  # noqa: E402  (style: rcParams, surface and ink colours, modality names)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REP = Path("/data/enric/reports")
OUT = REP / "slides/t360/figs"
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
BEN_SUM = REP / "ben/summary_t360/summary.json"
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
    the head-only row alone: its S1 + S2 number is a linear probe of a frozen encoder, the regime head-only is closest to."""
    d, m, get = ben()
    rows = [(r, b, t) for r, b, t in [("head only", "b10_head", "t360_head"), ("LoRA", "b10_lora", "t360_lora"),
                                      ("full fine-tune", "b10_full", "t360_full")] if "test_S1S2" in m[t]]
    fo = lambda t: m[t]["forgetting"]["ce_all"]
    sm = d["smarties"]["ViT-B, 10%"]["S1S2"]
    num["ben_regimes"] = {r: {"b10": get(b, "S1S2"), "t360": get(t, "S1S2"), "d": d["comparisons"][f"{t} - {b}"]["S1S2"],
                              "forget_b10": fo(b), "forget_t360": fo(t)} for r, b, t in rows}
    num["ben_regimes"]["head only"]["smarties_vitb_linear_probe"] = sm
    h = 1.2 + 1.15 * len(rows)
    fig, (ax, af) = plt.subplots(1, 2, figsize=(13.5, h), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.1})
    ys = np.arange(len(rows))[::-1].astype(float)
    fmax = max(max(fo(b), fo(t)) for _, b, t in rows)
    (x0, x1), (f0, f1) = (0.70, 0.88), (-0.04 * max(1.0, fmax / 0.3), max(0.34, 1.1 * fmax))
    for y, (r, b, t) in zip(ys, rows):
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
    (OUT / "numbers.json").write_text(json.dumps(num, indent=1, default=float))
    print("->", OUT / "numbers.json")


if __name__ == "__main__":
    main()
