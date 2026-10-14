"""Phase 6 Part 1 on B10: the slide deck's figures, in one house style. `.venv`, CPU. Reads only result files:
summary_b10/summary.json (ben_summary.py), chain/report.json (ben_chain.py report), analysis_b10/{l1,l2}.json
(ben_test_analysis.py, ben_spatial.py), gen_forgetting/report.json (ben_gen_forgetting.py report).
Writes <out>/deck/*.png. The one-off figures of ben_figures.py stay as they were.

A model keeps its colour in every figure (scratch grey, head light blue, LoRA greens, full orange).

    python eo/experiments/ben_slide_figures.py --out /data/enric/reports/ben/summary_b10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

REP = Path("/data/enric/reports/ben")
PRIOR = 0.311
SMARTIES = {"S1": 0.789, "S2": 0.869, "S1S2": 0.854}        # ViT-B, arXiv:2506.19585 Table 1, other protocol
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e0"
COL = {"b10_scratch": "#8d8d8d", "b10_head": "#8ec1e6", "b10_lora_lr1e4": "#a7d8a0", "b10_lora": "#1f8a4c",
       "b10_full": "#e8672a"}
NAME = {"b10_scratch": "from scratch", "b10_head": "head only", "b10_lora_lr1e4": "LoRA, lr 1e-4",
        "b10_lora": "LoRA, lr 1e-3", "b10_full": "full, lr 1e-4"}
ORDER = ["b10_scratch", "b10_head", "b10_lora_lr1e4", "b10_full", "b10_lora"]
GROUP = {"Artificial": "#7a7a7a", "Agricultural": "#d9a520", "Forest & semi-natural": "#2e8b57",
         "Wetlands": "#7b5cc4", "Water": "#3b8bc4"}
INPUT = {"S1": "#3b6ea5", "S2": "#d98a1c", "S1S2": "#1f8a4c"}

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": False, "legend.frameon": False,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "savefig.dpi": 200,
})


def jl(p):
    return json.loads(Path(p).read_text())


def save(fig, out, name):
    fig.savefig(out / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------------------------- setup

def box(ax, x, w, y, h, fc, ec, text, fs=14, bold=False, tc=INK, r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=tc)


def pipeline(out):
    fig, ax = plt.subplots(figsize=(12.4, 2.9))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 2.9); ax.axis("off")
    y, h, cap = 0.85, 0.95, 0.55
    spans = [(0.1, 3.7, "S2 · 196 tokens", "#fbf1dc", "#d98a1c"),
             (4.1, 3.7, "S1 · 196 tokens", "#e3edf7", "#3b6ea5"),
             (8.2, 4.1, "c₁  c₂  …  c_k", "#e0f2e7", "#1f8a4c")]
    for x, w, t, fc, ec in spans:
        box(ax, x, w, y, h, fc, ec, "", r=0.08)
        for cx, lab in ((x, "BO"), (x + w - cap, "EO")):
            ax.add_patch(Rectangle((cx, y), cap, h, fc=ec, ec=ec, alpha=0.85))
            ax.text(cx + cap / 2, y + h / 2, lab, ha="center", va="center", fontsize=12, color="white", fontweight="bold")
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=15)
    ax.plot([0.1, 7.8], [2.3, 2.3], color=MUTED, lw=1.5)
    ax.plot([0.1, 0.1], [2.3, 2.15], color=MUTED, lw=1.5); ax.plot([7.8, 7.8], [2.3, 2.15], color=MUTED, lw=1.5)
    ax.text(3.95, 2.42, "given: S1, S2 or both", ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.plot([8.2, 12.3], [2.3, 2.3], color="#1f8a4c", lw=1.5)
    ax.plot([8.2, 8.2], [2.3, 2.15], color="#1f8a4c", lw=1.5); ax.plot([12.3, 12.3], [2.3, 2.15], color="#1f8a4c", lw=1.5)
    ax.text(10.25, 2.42, "B10 writes the labels", ha="center", va="bottom", fontsize=15, fontweight="bold", color="#1f8a4c")
    ax.text(3.95, 0.45, "TerraMind tokenizers, unchanged", ha="center", va="center", fontsize=13, color=MUTED)
    ax.text(10.25, 0.45, "21 new token ids: 19 classes + BO/EO", ha="center", va="center", fontsize=13, color=MUTED)
    save(fig, out, "pipeline.png")


def regimes(out):
    rows = [("b10_scratch", "random", "trains", "trains", "153 M", "1e-4"),
            ("b10_head", "B10", "trains", "frozen", "12 k", "1e-3"),
            ("b10_lora", "B10", "trains", "LoRA r16 adapters", "4.7 M  (3%)", "1e-3 / 1e-4"),
            ("b10_full", "B10", "trains", "trains", "153 M", "1e-4")]
    names = {"b10_scratch": "from scratch", "b10_head": "head only", "b10_lora": "LoRA", "b10_full": "full fine-tune"}
    heads = ["", "starts from", "label rows", "backbone", "trainable", "lr"]
    xs = [0.0, 2.7, 4.5, 6.3, 9.0, 11.0]
    fig, ax = plt.subplots(figsize=(12.4, 3.3))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 3.3); ax.axis("off")
    for x, t in zip(xs, heads):
        ax.text(x + 0.1, 3.05, t, fontsize=13, color=MUTED, va="center")
    ax.plot([0, 12.4], [2.82, 2.82], color="#bbbbbb", lw=1)
    for i, (k, *cells) in enumerate(rows):
        yc = 2.3 - i * 0.68
        ax.add_patch(Rectangle((0.0, yc - 0.27), 0.16, 0.54, fc=COL[k], ec="none"))
        ax.text(0.3, yc, names[k], fontsize=15, fontweight="bold", va="center")
        for x, t in zip(xs[1:], cells):
            tc = MUTED if t in ("frozen", "random") else INK
            ax.text(x + 0.1, yc, t, fontsize=14, va="center", color=tc, fontweight="bold" if t == "trains" else "normal")
    save(fig, out, "regimes.png")


# --------------------------------------------------------------------------------------------- results

def results(S, out):
    m = S["models"]
    fig, ax = plt.subplots(figsize=(12.4, 4.9))
    hgrid(ax)
    w, n = 0.15, len(ORDER)
    for g, key in enumerate(("S1", "S2", "S1S2")):
        for i, k in enumerate(ORDER):
            v = m[k][f"test_{key}"]["micro_map"]
            x = g + (i - (n - 1) / 2) * (w + 0.015)
            ax.bar(x, v, w, color=COL[k], zorder=3, label=NAME[k] if g == 0 else None)
            ax.text(x, v - 0.012, f"{v:.3f}", ha="center", va="top", fontsize=11, fontweight="bold",
                    color=INK if k in ("b10_head", "b10_lora_lr1e4") else "white")        # inside the bar: no clash with the references
        s = SMARTIES[key]
        ax.plot([g - 0.43, g + 0.43], [s, s], color=INK, lw=1.6, ls=(0, (4, 3)), zorder=4,
                label="SMARTIES ViT-B (other protocol)" if g == 0 else None)
        ax.text(g - 0.43, s + 0.01, f"SMARTIES {s:.3f}", va="bottom", ha="left", fontsize=11.5, color=INK)
    ax.axhline(PRIOR, color=MUTED, lw=1.3, ls=":", zorder=2, label="class prior 0.311")
    ax.set_xticks(range(3), ["S1  (radar)", "S2  (optical)", "S1 + S2"], fontsize=15)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 1.0); ax.set_ylabel("micro-mAP, test_a (103,944 patches)")
    ax.tick_params(axis="x", length=0)
    h, l = ax.get_legend_handles_labels()
    want = [NAME[k] for k in ORDER] + ["SMARTIES ViT-B (other protocol)", "class prior 0.311"]
    ax.legend([h[l.index(t)] for t in want], want, loc="upper center", ncol=4, fontsize=12, columnspacing=1.4,
              handlelength=1.4, bbox_to_anchor=(0.5, 1.2))
    save(fig, out, "results.png")


def pretraining(S, out):
    m = S["models"]
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
    for ax, key, title in zip(axs, ("S1", "S2", "S1S2"), ("S1", "S2", "S1 + S2")):
        hgrid(ax)
        for g, (sc, lo) in enumerate((("b10_scratch", "b10_lora"), ("b10_scratch_1pct", "b10_lora_1pct"))):
            a, b = m[sc][f"test_{key}"]["micro_map"], m[lo][f"test_{key}"]["micro_map"]
            ax.bar(g - 0.2, a, 0.36, color=COL["b10_scratch"], zorder=3, label="from scratch" if (ax is axs[0] and g == 0) else None)
            ax.bar(g + 0.2, b, 0.36, color=COL["b10_lora"], zorder=3, label="pretrained B10 (LoRA)" if (ax is axs[0] and g == 0) else None)
            ax.text(g - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=12)
            ax.text(g + 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=12, fontweight="bold")
            ax.text(g, 0.93, f"+{b - a:.2f}", ha="center", fontsize=16, fontweight="bold", color=COL["b10_lora"])
        ax.axhline(PRIOR, color=MUTED, lw=1.2, ls=":", zorder=2)
        ax.set_xticks([0, 1], ["10% labels", "1% labels"], fontsize=14)
        ax.set_title(title, fontsize=15, fontweight="bold"); ax.set_ylim(0, 1.0); ax.set_xlim(-0.6, 1.6)
        ax.tick_params(axis="x", length=0)
    axs[0].set_ylabel("micro-mAP, test_a")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=13, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.92, bottom=0.2, wspace=0.08)
    save(fig, out, "pretraining.png")


def tradeoff(S, out):
    m = S["models"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    pt = {k: (m[k]["forgetting_ce_all"], m[k]["test_S1S2"]["micro_map"]) for k in ("b10_head", "b10_lora_lr1e4", "b10_full", "b10_lora")}
    for k, (x, y) in pt.items():
        ax.scatter(x, y, s=170, color=COL[k], zorder=4, edgecolor="white", lw=1.5)
    off = {"b10_head": (-12, 0), "b10_lora_lr1e4": (14, -2), "b10_full": (-14, 0), "b10_lora": (-14, 4)}
    ha = {"b10_head": "right", "b10_lora_lr1e4": "left", "b10_full": "right", "b10_lora": "right"}
    for k, (x, y) in pt.items():
        ax.annotate(f"{NAME[k]}  {y:.3f}", (x, y), xytext=off[k], textcoords="offset points", ha=ha[k], va="center", fontsize=13.5,
                    fontweight="bold" if k == "b10_lora" else "normal")
    sc = m["b10_scratch"]["test_S1S2"]["micro_map"]
    ax.axhline(sc, color=COL["b10_scratch"], lw=2, ls="--", zorder=2)
    ax.text(0.298, sc - 0.002, f"from scratch {sc:.3f}", ha="right", va="top", fontsize=13.5, color="#555555")
    (x1, y1), (x2, y2) = pt["b10_lora_lr1e4"], pt["b10_full"]
    ax.plot([x1, x2], [y1, y2], color="#999999", lw=1.3, ls=":", zorder=3)
    ax.text((x1 + x2) / 2 + 0.004, (y1 + y2) / 2 - 0.003, "same lr 1e-4", fontsize=12.5, color="#555555", ha="left", va="top")
    ax.set_xlim(-0.10, 0.30); ax.set_ylim(0.70, 0.83)
    ax.set_xlabel("forgetting on TerraMesh (nats, + = worse)"); ax.set_ylabel("micro-mAP, S1 + S2, test_a")
    save(fig, out, "tradeoff.png")


# --------------------------------------------------------------------------------------------- examples-adjacent

def perclass(L, out):
    c = L["b10_lora"]
    both = c["S1S2"]["per_class"]
    s2 = {r["class"]: r for r in c["S2"]["per_class"]}
    rows = sorted(both, key=lambda r: -r["train_freq"])
    n = len(rows)
    y = np.arange(n)[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 6.0), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.06))
    for ax in (a, b):
        ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    cols = [GROUP[r["level1"]] for r in rows]
    a.barh(y, [r["ap"] for r in rows], 0.7, color=cols, zorder=3)
    a.scatter([s2[r["class"]]["ap"] for r in rows], y, marker="v", s=46, color="white", edgecolor=INK, lw=1.2, zorder=5, label="S2 alone")
    a.scatter([r["test_prev"] for r in rows], y, marker="|", s=170, color=INK, lw=2, zorder=5, label="prevalence (= AP of a constant guess)")
    for yy, r in zip(y, rows):
        a.text(max(r["ap"], s2[r["class"]]["ap"]) + 0.02, yy, f'{r["ap"]:.2f}', va="center", fontsize=11)
    a.set_yticks(y, [f'{r["class"]}  ·  {100 * r["train_freq"]:.1f}%' for r in rows], fontsize=12)
    a.set_xlim(0, 1.08); a.set_xlabel("AP (S1 + S2)"); a.tick_params(axis="y", length=0)
    b.barh(y, [r["recall_greedy"] for r in rows], 0.7, color=cols, zorder=3)
    for yy, r in zip(y, rows):
        b.text(r["recall_greedy"] + 0.015, yy, f'{r["recall_greedy"]:.2f}', va="center", fontsize=11)
    b.set_xlim(0, 1.0); b.set_xlabel("recall of the emitted label set"); b.tick_params(axis="y", length=0)
    hs = [Rectangle((0, 0), 1, 1, fc=v) for v in GROUP.values()]
    h2, l2 = a.get_legend_handles_labels()
    fig.legend(hs + h2, list(GROUP) + l2, loc="lower center", ncol=4, fontsize=11.5, bbox_to_anchor=(0.5, -0.095), columnspacing=1.3)
    save(fig, out, "perclass.png")
    rare = [r for r in rows if r["train_freq"] < 0.05]
    return {"spearman": c["S1S2"]["spearman_ap_vs_log_train_freq"],
            "rare": [(r["class"], round(r["train_freq"], 4), round(r["ap"], 2), round(r["recall_greedy"], 2)) for r in rare]}


def spatial(L2, out):
    r = L2["models"]["b10_lora/S1S2"]
    bins = [k for k in r if k.startswith("d_nn") and "micro_map" in r[k]]
    lab = ["same place\n< 0.1 km", "0.1–1.25", "1.25–1.75", "1.75–5", "5–20"]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    ax.plot(x, [r[b]["micro_map"] for b in bins], "o-", color=COL["b10_lora"], lw=3, ms=9, label="B10 LoRA, from pixels (S1 + S2)")
    ax.plot(x, [r[b]["knn10_micro_map"] for b in bins], "s--", color="#c44e52", lw=2.5, ms=8,
            label="location only: 10 nearest training patches")
    ax.plot(x, [r[b]["prior_micro_map"] for b in bins], ":", color=MUTED, lw=2, label="class prior")
    ax.set_xticks(x, [f"{lab[i]}\nn = {r[b]['n']:,}" for i, b in enumerate(bins)], fontsize=11.5)
    ax.set_xlabel("distance to the nearest training patch (km)")
    ax.set_ylabel("micro-mAP"); ax.set_ylim(0.25, 0.92); ax.set_xlim(-0.3, len(bins) - 0.7)
    ax.legend(loc="center right", fontsize=12, bbox_to_anchor=(1.0, 0.53))
    save(fig, out, "spatial.png")


def chain(C, out):
    r = C["b10_lora"]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    hgrid(ax)
    kinds = [("floor", "another patch's block", "#c9c9c9"), ("single", "real modality alone", "#3b6ea5"),
             ("chain_s0", "+ generated block (1 sample)", "#e59a9c"), ("chain_mean4", "+ generated block (mean of 4)", "#c44e52"),
             ("ceiling", "+ real block", "#1f8a4c")]
    w = 0.15
    for g, d in enumerate(("S1toS2/base", "S2toS1/base")):
        for i, (k, lab, col) in enumerate(kinds):
            x = g + (i - 2) * (w + 0.015)
            ax.bar(x, r[d][k], w, color=col, zorder=3, label=lab if g == 0 else None)
            ax.text(x, r[d][k] + 0.01, f"{r[d][k]:.2f}", ha="center", fontsize=11.5)
        ax.text(g, 0.97, f"{r[d]['chain_minus_single']:+.2f}".replace("-", "−"), ha="center", fontsize=20, fontweight="bold", color="#c44e52")
        ax.text(g, 0.915, "vs real alone", ha="center", fontsize=11.5, color=MUTED)
    ax.axhline(PRIOR, color=MUTED, lw=1.2, ls=":", zorder=2)
    ax.text(-0.64, PRIOR + 0.008, "prior", fontsize=11.5, color=MUTED, ha="left")
    ax.set_xticks([0, 1], ["given S1 → generate Ŝ2", "given S2 → generate Ŝ1"], fontsize=14)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.03); ax.set_xlim(-0.66, 1.55); ax.set_ylabel("micro-mAP (10,000 test_a patches)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=11.5, columnspacing=1.2, handlelength=1.2)
    save(fig, out, "chain.png")


def genforget(G, out):
    models = ["b10_head", "b10_lora_lr1e4", "b10_full", "b10_lora"]
    cells = [("S2L2A/none", "no condition → S2", "decoded S2 error (MSE)"), ("S2L2A/DEM", "DEM → S2", "decoded S2 error (MSE)"),
             ("LULC/S2L2A", "S2 → LULC", "share of pixels misclassified")]
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.2))
    for ax, (cell, title, yl) in zip(axs, cells):
        ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
        mb = G[cell]["minus_base"]
        top = 0
        for i, k in enumerate(models):
            v = mb[k]["mean"]; lo, hi = mb[k]["ci95"]
            firm = not (lo <= 0 <= hi) or k == "b10_head"
            ax.bar(i, v, 0.7, color=COL[k], alpha=1.0 if firm else 0.38, zorder=3, hatch=None)
            if k != "b10_head":
                ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=4, lw=1.4, zorder=4)
            y = max(hi, v) if k != "b10_head" else v
            ax.text(i, y + 0.02 * (1 if cell != "LULC/S2L2A" else 0.2), ((f"{v:+.2f}" if abs(v) >= 0.005 else "0.00") if cell != "LULC/S2L2A" else f"{v:+.3f}").replace("-", "−"),
                    ha="center", va="bottom", fontsize=12, fontweight="bold" if firm else "normal")
            top = max(top, hi if k != "b10_head" else v)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(len(models)), ["head", "LoRA\n1e-4", "full\n1e-4", "LoRA\n1e-3"], fontsize=12.5)
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, fontsize=15, fontweight="bold"); ax.set_ylabel("Δ " + yl + " vs base" if ax is axs[0] else "")
        ax.set_ylabel("Δ vs the base model")
        ax.set_ylim(min(-0.4 if cell != "LULC/S2L2A" else -0.02, ax.get_ylim()[0]), top * 1.18)
    fig.tight_layout()
    fig.text(0.5, -0.02, "bars with whiskers: Δ and 95% CI over scenes (64);  faded = CI includes 0", ha="center", fontsize=12, color=MUTED)
    save(fig, out, "genforget.png")


def calibration(L, out):
    c = L["b10_lora"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.3, ls=":")
    for k, col, lab in (("S1", INPUT["S1"], "S1"), ("S2", INPUT["S2"], "S2"), ("S1S2", INPUT["S1S2"], "S1 + S2")):
        bins = c[k]["calibration"]["bins"]
        ax.plot([b["mean_p"] for b in bins], [b["obs"] for b in bins], "o-", color=col, ms=5, lw=2,
                label=f"{lab}   ECE {c[k]['calibration']['ece']:.3f}")
    ax.set_xlabel("predicted probability of a class"); ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=12)
    save(fig, out, "calibration.png")
    return {"sharpened_ece": c["S1S2"]["calibration_controls"]["sharpened_ece"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out / "deck"
    out.mkdir(parents=True, exist_ok=True)
    S, L = jl(a.out / "summary.json"), jl(REP / "analysis_b10/l1.json")
    pipeline(out); regimes(out); results(S, out); pretraining(S, out); tradeoff(S, out)
    print("perclass", perclass(L, out))
    spatial(jl(REP / "analysis_b10/l2.json"), out)
    chain(jl(REP / "chain/report.json"), out)
    genforget(jl(REP / "gen_forgetting/report.json"), out)
    print("calibration", calibration(L, out))
    print(sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
