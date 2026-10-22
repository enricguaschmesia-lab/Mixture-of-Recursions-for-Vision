"""Figures for the midterm deck (docs/slides/midterm.md, 2026-10-26). `.venv`, CPU.

Every number is read from a stored result; nothing is recomputed from a model:
  grid/<pair>/grid_report.json, grid/<pair>/curve.csv     (teacher-forced CE grid and curves)
  arm_eval/<model>/checkpoint-3300/{metrics.json, tokens_router.npz}   (interventions, depth)
  decoded_grid_clean/<MOD>/illus_{ceiling,b10c}.npz       (decoded illustrations, cleaned tokens)
  runs/pretrain/phase3/{mor,vanilla}_2026092*/checkpoint-16500/trainer_state.json   (50-epoch losses)
  ben/summary_b10/summary.json                            (BigEarthNet)
  data/PASTIS/PASTIS-R + reports/pastis/step0/patches.parquet   (one PASTIS patch)

Colours: a model keeps one colour on every page (MoR blue, vanilla orange, random router aqua,
11-layer vanilla violet; validated with the dataviz checker). Modalities and BEN regimes have their
own sets, and every coloured mark also carries a text label. In the diagrams a flame marks a part
that trains, a snowflake a frozen one, and a red cross one that is thrown away.

    .venv/bin/python eo/experiments/midterm_figures.py [--only NAME ...]
Writes /data/enric/reports/slides/midterm/figs/*.png.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
REP = Path("/data/enric/reports")
RUNS = Path("/data/enric/runs/pretrain/phase3")
DEC = REP / "decoded_grid_clean"
PASTIS = Path("/data/enric/data/PASTIS/PASTIS-R")
OUT = REP / "slides/midterm/figs"

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
MODEL = {"a10c": "#2a78d6", "a10cs2": "#2a78d6", "a10cr": "#1baf7a", "b10c": "#eb6834", "b10c11": "#4a3aa7"}
MNAME = {"a10c": "MoR", "a10cs2": "MoR, 2nd seed", "a10cr": "MoR, random router",
         "b10c": "vanilla, 29 layers", "b10c11": "vanilla, 11 layers"}
MODC = {"S2L2A": "#d98a1c", "S1": "#3b6ea5", "S1GRD": "#3b6ea5", "S1RTC": "#3b6ea5", "DEM": "#a3622a",
        "NDVI": "#6a9a1f", "LULC": "#b0499a", "Coords": "#6b6b6b", "TASK": "#1f8a4c"}
NICE = {"S2L2A": "S2", "S1GRD": "S1 GRD", "S1RTC": "S1 RTC", "DEM": "DEM", "NDVI": "NDVI", "LULC": "LULC",
        "Coords": "location", "all": "all others", "none": "nothing"}
BEN = {"b10_scratch": "#8d8d8d", "b10_head": "#eda100", "b10_lora": "#1f8a4c", "b10_full": "#3d3c39"}
FULL = BEN["b10_full"]
BENNAME = {"b10_scratch": "from scratch", "b10_head": "head only", "b10_lora": "LoRA", "b10_full": "full fine-tune"}
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
TARGETS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
EPOCH = 84672 / 256                       # optimizer steps per epoch
SMARTIES = {"S1": 0.789, "S2": 0.869, "S1S2": 0.854}   # ViT-B, arXiv:2506.19585 Table 1 (other protocol)
PRIOR = 0.311
LULC_CMAP = ListedColormap(["#d9d9d9", "#419bdf", "#397d49", "#7a87c6", "#e49635", "#c4281b",
                            "#a59b8f", "#a8ebff", "#616161", "#e3e2c3"])      # as s12_decoded_eval.py
SCENE = 23          # the example scene of the schematic (a held-out Major TOM scene with S1RTC)
GEN_SCENE = 80403   # the generation example (a held-out SSL4EO-S12 scene with S1GRD)

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def jl(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("->", OUT / name)


def tint(hexcol, a=0.82):
    c = np.array(matplotlib.colors.to_rgb(hexcol))
    return tuple(c + (1 - c) * a)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------------------------------ images

def truth(mod, row):
    return np.load(DEC / mod / "illus_ceiling.npz")[f"truth|{row}|img"]


def gen(mod, src, row, model="b10c"):
    z = np.load(DEC / mod / f"illus_{model}.npz")
    return z[f"{model}|{src}|{row}|img"]


def show(ax, mod, img, ref=None):
    """s12_decoded_eval.show: the reference's 2-98% stretch, LULC classes, DEM relief, NDVI RdYlGn."""
    ref = img if ref is None else ref
    ax.set_axis_off()
    if mod == "LULC":
        ax.imshow(img, cmap=LULC_CMAP, vmin=0, vmax=9, interpolation="nearest")
        return
    if mod == "S2L2A":
        lo = np.nanpercentile(ref, 2, axis=(0, 1)); hi = np.nanpercentile(ref, 98, axis=(0, 1))
        ax.imshow(np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1))
        return
    if mod == "DEM":
        img, ref = img - np.nanmean(img), ref - np.nanmean(ref)
    lo, hi = np.nanpercentile(ref, 2), np.nanpercentile(ref, 98)
    ax.imshow(img, cmap={"DEM": "terrain", "NDVI": "RdYlGn"}.get(mod, "gray"), vmin=lo, vmax=max(hi, lo + 1e-6))


def frame(ax, col, lw=3):
    ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(col); s.set_linewidth(lw)


def scene_coords(row):
    import pandas as pd
    m = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet", columns=["center_lon", "center_lat"])
    return float(m.center_lon.iloc[row]), float(m.center_lat.iloc[row])


def fmt_ll(lon, lat):
    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"


# ------------------------------------------------------------------------------------- title strip

def fig_strip():
    mods = [("S2L2A", "S2, optical"), ("S1RTC", "S1, radar"), ("DEM", "elevation"),
            ("NDVI", "NDVI"), ("LULC", "land cover")]
    fig, axs = plt.subplots(1, 6, figsize=(13.5, 2.75), gridspec_kw=dict(wspace=0.08))
    for ax, (m, t) in zip(axs, mods):
        show(ax, m, truth(m, SCENE))
        frame(ax, MODC[m], 3)
        ax.set_title(t, fontsize=14, loc="center", fontweight="normal", color=INK)
    ax = axs[-1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    lon, lat = scene_coords(SCENE)
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=tint(MODC["Coords"], 0.9), ec=MODC["Coords"], lw=3))
    ax.plot([0.5], [0.62], marker="v", ms=30, color=MODC["Coords"])
    ax.plot([0.5], [0.67], marker="o", ms=11, color="white")
    ax.text(0.5, 0.3, fmt_ll(lon, lat), ha="center", va="center", fontsize=13.5)
    ax.set_title("location", fontsize=14, loc="center", fontweight="normal")
    ax.set_axis_off()
    save(fig, "strip.png")


# ----------------------------------------------------------------------------- 2. the sequence

def tok_block(ax, x, y, w, h, col, label, cap=0.42, fs=16, body=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=tint(col, 0.84) if body is None else body, ec=col, lw=1.8, zorder=3))
    for cx in (x, x + w - cap):
        ax.add_patch(Rectangle((cx, y), cap, h, fc=col, ec=col, zorder=4))
    ax.text(x + cap / 2, y + h / 2, "BO", ha="center", va="center", fontsize=10.5, color="white", fontweight="bold", zorder=5)
    ax.text(x + w - cap / 2, y + h / 2, "EO", ha="center", va="center", fontsize=10.5, color="white", fontweight="bold", zorder=5)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, zorder=5)


def arrow(ax, p0, p1, col=INK2, lw=1.8, ms=16, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=col, lw=lw, zorder=2))


def sq(ax, x, y, s, col, solid=False, new=False, lw=1.4, z=4, text=None, fs=11.5):
    """One token: a square in its modality's colour (BO/EO markers solid); `new`: a task's own token."""
    kw = dict(boxstyle="round,pad=0,rounding_size=0.05", zorder=z)
    if new:
        ax.add_patch(FancyBboxPatch((x, y), s, s, fc=tint(NEW, 0.86), ec=NEW, lw=lw + 0.3, hatch="////", **kw))
    else:
        ax.add_patch(FancyBboxPatch((x, y), s, s, fc=col if solid else tint(col, 0.72), ec=col, lw=lw, **kw))
    if text:
        ax.text(x + s / 2, y + s / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=z + 1,
                color="white" if solid else INK)


def group(mod, n):
    """A modality's tokens as (modality, is_marker): BO, n - 2 tokens, EO."""
    return [(mod, True)] + [(mod, False)] * (n - 2) + [(mod, True)]


def inset(fig, ax, x, y, w, h=None):
    """An axes over the data rectangle (x, y, w, h) of the full-figure axes `ax`."""
    h = w if h is None else h
    (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(ax.transData.transform([(x, y), (x + w, y + h)]))
    return fig.add_axes([x0, y0, x1 - x0, y1 - y0])


def fig_sequence():
    W, H = 14.2, 6.35
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    # top: one scene, each modality through TerraMind's tokenizer
    mods = [("S2L2A", "S2"), ("S1RTC", "S1"), ("DEM", "DEM"), ("NDVI", "NDVI"), ("LULC", "LULC")]
    s, gap, y0 = 0.95, 0.42, 5.0
    x0 = (W - (6 * s + 5 * gap)) / 2
    xs = [x0 + i * (s + gap) for i in range(6)]
    for x, (m, lab) in zip(xs, mods):
        a = inset(fig, ax, x, y0, s); show(a, m, truth(m, SCENE)); frame(a, MODC[m], 2.5)
        ax.text(x + s / 2, y0 + s + 0.07, lab, ha="center", va="bottom", fontsize=15, fontweight="bold")
    x = xs[5]
    ax.add_patch(FancyBboxPatch((x, y0), s, s, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=tint(MODC["Coords"], 0.9), ec=MODC["Coords"], lw=2.5))
    ax.plot([x + s / 2], [y0 + 0.66], marker="v", ms=19, color=MODC["Coords"])
    ax.plot([x + s / 2], [y0 + 0.7], marker="o", ms=6, color="white")
    lon, lat = scene_coords(SCENE)
    ax.text(x + s / 2, y0 + 0.25, fmt_ll(lon, lat).replace(", ", "\n"), ha="center", va="center", fontsize=11.5,
            linespacing=1.05)
    ax.text(x + s / 2, y0 + s + 0.07, "location", ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.text(x0 - 0.35, y0 + s / 2, "one\nscene", ha="right", va="center", fontsize=17, color=INK2, linespacing=1.2)
    ty = 4.3
    ax.add_patch(FancyBboxPatch((x0, ty), xs[5] + s - x0, 0.4, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#f1f0eb", ec=MUTED, lw=1.2))
    ax.text((x0 + xs[5] + s) / 2, ty + 0.2, "TerraMind's tokenizers: each image becomes 196 tokens",
            ha="center", va="center", fontsize=15.5)
    for x in xs:
        arrow(ax, (x + s / 2, y0 - 0.02), (x + s / 2, ty + 0.42), lw=1.4, ms=11)

    # the decoder is one long rectangle, used the same way to train and to generate
    by, bh, yb, yt, s = 1.45, 0.9, 0.55, 2.83, 0.38
    def up(x, y_from, y_to):
        arrow(ax, (x, y_from + 0.02), (x, y_to - 0.02), lw=1.1, ms=8)

    # ---- train: every token in, every next token out, in one pass
    seq = group("S2L2A", 5) + group("S1RTC", 5) + group("DEM", 4)
    tx0, tx1, p = 0.3, 6.75, 0.47
    ax.text(tx0, 3.72, "Train: predict every next token", ha="left", va="center", fontsize=17, fontweight="bold")
    arch_box(ax, tx0, by, tx1 - tx0, bh, "pre", "decoder", fs=18, bold=True)
    xa = tx0 + (tx1 - tx0 - (13 * p - (p - s))) / 2
    for i in range(13):
        x = xa + i * p
        sq(ax, x, yb, s, MODC[seq[i][0]], seq[i][1])
        sq(ax, x, yt, s, MODC[seq[i + 1][0]], seq[i + 1][1])
        up(x + s / 2, yb + s, by); up(x + s / 2, by + bh, yt)
    for lab, mod, k0, k1 in (("S2", "S2L2A", 0, 4), ("S1", "S1RTC", 5, 9), ("DEM", "DEM", 10, 12)):
        ax.text(xa + (k0 + k1) / 2 * p + s / 2, yb - 0.08, lab, ha="center", va="top", fontsize=14.5,
                fontweight="bold", color=MODC[mod])

    # ---- use: given S1, write S2 one token at a time; each token is read back as the next input
    ux0, ux1, p = 7.35, 12.85, 0.58
    ax.text(ux0, 3.72, "Use: write a token, feed it back, repeat", ha="left", va="center", fontsize=17,
            fontweight="bold")
    arch_box(ax, ux0, by, ux1 - ux0, bh, "pre")
    ctx = group("S1RTC", 5) + [("S2L2A", True)]
    xu = ux0 + 0.2
    px = [xu + i * p for i in range(9)]
    ax.text((px[0] + px[4] + s) / 2, by + bh / 2, "decoder", ha="center", va="center", fontsize=18, fontweight="bold",
            zorder=6)
    S2C = MODC["S2L2A"]
    for i in range(9):
        if i < 6:
            sq(ax, px[i], yb, s, MODC[ctx[i][0]], ctx[i][1])
        else:
            sq(ax, px[i], yb, s, S2C, lw=2.2, text=str(i - 5))
        up(px[i] + s / 2, yb + s, by)
    for k in range(4):                        # outputs at the last four positions: tokens 1-4
        x = px[5 + k]
        sq(ax, x, yt, s, S2C, lw=2.2, text=str(k + 1))
        up(x + s / 2, by + bh, yt)
        if k < 3:                             # read back: down the gap to the next input position
            gx = x + s + (p - s) / 2
            ax.plot([x + s, gx, gx], [yt + s / 2, yt + s / 2, yb + s / 2], color=S2C, lw=1.9, zorder=7,
                    solid_joinstyle="round")
            arrow(ax, (gx - 0.01, yb + s / 2), (px[6 + k] - 0.01, yb + s / 2), col=S2C, lw=1.9, ms=10)
    for y in (yt, yb):
        ax.text(px[8] + s + 0.12, y + s / 2, "…", ha="left", va="center", fontsize=17, color=INK2)
    ax.text((px[0] + px[4] + s) / 2, yb - 0.08, "S1: given", ha="center", va="top", fontsize=14.5,
            fontweight="bold", color=MODC["S1RTC"])
    ax.text((px[5] + px[8] + s) / 2, yb - 0.08, "S2: written", ha="center", va="top", fontsize=14.5,
            fontweight="bold", color=S2C)
    # the 196 written tokens, decoded by the tokenizer
    a = inset(fig, ax, 13.25, 2.62, 0.85)
    show(a, "S2L2A", gen("S2L2A", "S1RTC", SCENE), truth("S2L2A", SCENE)); frame(a, S2C, 1.8)
    ax.text(13.675, 2.56, "decoded", ha="center", va="top", fontsize=13, color=INK2)
    save(fig, "sequence.png")


# ------------------------------------- 2a. encoder-decoder and decoder-only: pretraining, then downstream

PRE_FC, PRE_EC = "#dfe6ee", "#4f6477"      # pretrained
NEW = "#d1495b"                            # added for one downstream task
GONE = "#a8a7a1"                           # used only during pretraining
FIRE, FIRE2, ICE, BAD = "#e4572e", "#f6b93b", "#3d8fd1", "#c44e52"
FLAME = [(0, 0), (0.42, 0), (0.55, 0.38), (0.32, 0.72), (0.22, 0.88), (0.12, 1.0), (0.08, 1.38),
         (-0.12, 1.05), (-0.42, 0.85), (-0.45, 0.45), (-0.47, 0.12), (-0.28, 0), (0, 0)]


def arch_box(ax, x, y, w, h, kind, text="", fs=15, bold=False, tc=None):
    kw = dict(boxstyle="round,pad=0,rounding_size=0.08", zorder=3)
    if kind == "pre":
        p = FancyBboxPatch((x, y), w, h, fc=PRE_FC, ec=PRE_EC, lw=1.8, **kw)
    elif kind == "new":
        p = FancyBboxPatch((x, y), w, h, fc=tint(NEW, 0.86), ec=NEW, lw=2.0, hatch="////", **kw)
    else:                                   # "gone": pretraining-only, thrown away afterwards
        p = FancyBboxPatch((x, y), w, h, fc="none", ec=GONE, lw=1.8, ls=(0, (4, 3)), **kw)
    ax.add_patch(p)
    if text:
        col = tc or (MUTED if kind == "gone" else INK)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=col,
                fontweight="bold" if bold else "normal", zorder=6, linespacing=1.1,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85) if kind == "new" else None)


def flame(ax, x, y, s=0.27, z=9):
    """Trains: a flame centred on (x, y)."""
    codes = [MPath.MOVETO] + [MPath.CURVE4] * 12
    for k, col in ((1.0, FIRE), (0.55, FIRE2)):
        pts = [(x + k * s * a, y - 0.69 * s + k * s * b) for a, b in FLAME]
        ax.add_patch(PathPatch(MPath(pts, codes), fc=col, ec="none", zorder=z))


def snowflake(ax, x, y, r=0.2, z=9):
    """Frozen: a six-armed snowflake centred on (x, y)."""
    kw = dict(color=ICE, solid_capstyle="round", zorder=z)
    for k in range(6):
        a = np.pi / 2 + k * np.pi / 3
        ax.plot([x, x + r * np.cos(a)], [y, y + r * np.sin(a)], lw=2.2, **kw)
        bx, by = x + 0.58 * r * np.cos(a), y + 0.58 * r * np.sin(a)
        for d in (-1, 1):
            b = a + d * np.pi / 4
            ax.plot([bx, bx + 0.38 * r * np.cos(b)], [by, by + 0.38 * r * np.sin(b)], lw=1.7, **kw)


def icon(ax, kind, x, y, size=1.0):
    """A badge on a box corner: 'trained' (flame) or 'frozen' (snowflake)."""
    ax.add_patch(Circle((x, y), 0.25 * size, fc=SURF, ec="none", zorder=8))
    if kind == "trained":
        flame(ax, x, y, 0.27 * size)
    else:
        snowflake(ax, x, y, 0.2 * size)


def crossed(ax, x, y, w, h, lw=3.5):
    """Thrown away: a red cross over the rectangle (x, y, w, h)."""
    for ya, yb in ((y, y + h), (y + h, y)):
        ax.plot([x, x + w], [ya, yb], color=BAD, lw=lw, solid_capstyle="round", zorder=8)


def latent(ax, xc, y, n=8, s=0.22, p=0.28):
    """The encoder's output tokens."""
    x = xc - (n * p - (p - s)) / 2
    for i in range(n):
        ax.add_patch(Rectangle((x + i * p, y), s, s, fc=tint(PRE_EC, 0.45), ec=PRE_EC, lw=1, zorder=4))


def out_chips(ax, x, y, w, names, h=0.25, gap=0.06, fs=12.5, col=None):
    """Scene labels as chips, stacked from the top down; (x, y) is the bottom-left corner."""
    col = col or MODC["TASK"]
    for k, n in enumerate(names):
        yy = y + (len(names) - 1 - k) * (h + gap)
        ax.add_patch(FancyBboxPatch((x, yy), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=tint(col, 0.8), ec=col, lw=1.3, zorder=4))
        ax.text(x + w / 2, yy + h / 2, n, ha="center", va="center", fontsize=fs, zorder=5)


def out_cropmap(fig, ax, x, y, s, pid=40305):
    a = inset(fig, ax, x, y, s)
    a.imshow(pastis_grid(pastis_sem(pid)), cmap=ListedColormap(pastis_cols()), vmin=0, vmax=19, interpolation="nearest")
    a.set_axis_off()


def out_biomass(fig, ax, x, y, s):
    """An illustration of a regression output: a smooth field."""
    yy, xx = np.mgrid[0:1:64j, 0:1:64j]
    f = sum(a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg ** 2))
            for a, cx, cy, sg in [(1.0, 0.3, 0.3, 0.2), (0.8, 0.75, 0.6, 0.18), (0.6, 0.45, 0.9, 0.15)])
    a = inset(fig, ax, x, y, s)
    a.imshow(f, cmap="YlGn", vmin=0, vmax=1.1, interpolation="bilinear")
    a.set_axis_off()


def s2_rgb(row):
    """A TerraMesh S2 scene as RGB, with show()'s 2-98% stretch."""
    img = truth("S2L2A", row)
    lo = np.nanpercentile(img, 2, axis=(0, 1)); hi = np.nanpercentile(img, 98, axis=(0, 1))
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def masked(rgb, n=6, frac=0.5, seed=3):
    """The image with a random half of an n x n grid of patches greyed out."""
    out, c = rgb.copy(), rgb.shape[0] // n
    for k in np.random.default_rng(seed).choice(n * n, int(frac * n * n), replace=False):
        i, j = divmod(int(k), n)
        out[i * c:(i + 1) * c, j * c:(j + 1) * c] = 0.62
    return out


def patch_grid(a, img, n=6):
    """Thin white lines between the patches."""
    c = img.shape[0] // n
    for k in range(1, n):
        a.axhline(k * c - 0.5, color="white", lw=0.7); a.axvline(k * c - 0.5, color="white", lw=0.7)


def half_titles(ax, left, right, xdiv=7.05, top=5.1, gap=None):
    """'Pretraining' and 'Downstream' over the two halves, each with a one-line subtitle."""
    for x, t, sub in ((3.5, "Pretraining", left), (10.75, "Downstream", right)):
        ax.text(x, top, t, ha="center", va="center", fontsize=20, fontweight="bold")
        ax.text(x, top - 0.38, sub, ha="center", va="center", fontsize=15.5, color=INK2)
    for y0, y1 in ([(0.0, top + 0.2)] if gap is None else [(0.0, gap[0]), (gap[1], top + 0.2)]):
        ax.plot([xdiv, xdiv], [y0, y1], color=GRID, lw=1.5, zorder=0)


def legend_row(ax, items, xs, y=-0.3):
    for (kind, lab), x in zip(items, xs):
        if kind == "trained":
            flame(ax, x, y, 0.2)
        elif kind == "frozen":
            snowflake(ax, x, y, 0.15)
        elif kind == "gone":
            crossed(ax, x - 0.14, y - 0.11, 0.28, 0.22, lw=2.5)
        else:
            arch_box(ax, x - 0.21, y - 0.12, 0.42, 0.24, "new")
        ax.text(x + 0.32, y, lab, ha="left", va="center", fontsize=15.5)


def arch_canvas():
    W, H, ylo = 14.4, 6.05, -0.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(ylo, H + ylo); ax.set_axis_off()
    return fig, ax


def fig_arch_encdec(pid=40305):
    fig, ax = arch_canvas()
    half_titles(ax, "no labels: fill in what is hidden", "a new decoder for every task, trained from scratch")
    # ---- pretraining: the encoder sees a masked image, the decoder rebuilds it
    rgb = s2_rgb(SCENE)
    for y, img in ((0.3, masked(rgb)), (2.4, rgb)):
        a = inset(fig, ax, 0.35, y, 1.4); a.imshow(img); a.set_axis_off(); patch_grid(a, img)
    arrow(ax, (1.82, 1.0), (2.38, 1.0))
    arch_box(ax, 2.4, 0.25, 3.9, 1.5, "pre", "encoder", fs=19, bold=True)
    icon(ax, "trained", 6.3, 1.75)
    latent(ax, 4.35, 1.98)
    arrow(ax, (4.35, 1.77), (4.35, 1.97), lw=1.4, ms=10)
    arrow(ax, (4.35, 2.22), (4.35, 2.53), lw=1.4, ms=11)
    arch_box(ax, 2.9, 2.55, 2.9, 0.62, "pre", "decoder", fs=17, bold=True)
    icon(ax, "trained", 5.8, 3.17)
    arrow(ax, (2.88, 2.86), (1.82, 2.86))
    # ---- downstream: the encoder is kept ...
    a = inset(fig, ax, 7.45, 0.3, 1.4); a.imshow(pastis_rgb(pid)); a.set_axis_off()
    arrow(ax, (8.92, 1.0), (9.43, 1.0))
    arch_box(ax, 9.45, 0.25, 3.9, 1.5, "pre", "encoder", fs=19, bold=True)
    icon(ax, "frozen", 13.35, 1.75)
    latent(ax, 11.4, 1.98)
    arrow(ax, (11.4, 1.77), (11.4, 1.97), lw=1.4, ms=10)
    # ... the pretrained decoder is thrown away ...
    arch_box(ax, 7.4, 2.55, 1.45, 0.68, "gone", "pretrained\ndecoder", fs=14, tc=INK2)
    crossed(ax, 7.47, 2.6, 1.31, 0.58)
    ax.text(8.125, 3.62, "thrown away", ha="center", va="center", fontsize=15, color=BAD, fontweight="bold")
    ax.add_patch(FancyArrowPatch((5.84, 2.86), (7.36, 2.86), arrowstyle="-|>", mutation_scale=14, color=GONE,
                                 lw=1.6, ls=(0, (4, 3)), zorder=2))
    # ... and every task trains a decoder of its own
    for name, x, out in (("land-cover\nclassifier", 9.05, "chips"), ("crop-type\nsegmenter", 10.82, "map"),
                         ("biomass\nregressor", 12.59, "field")):
        arrow(ax, (11.4, 2.22), (x + 0.8, 2.53), lw=1.3, ms=10)
        arch_box(ax, x, 2.55, 1.6, 0.68, "new", name, fs=14.5)
        icon(ax, "trained", x + 1.6, 3.23, 0.85)
        arrow(ax, (x + 0.8, 3.25), (x + 0.8, 3.48), lw=1.3, ms=10)
        if out == "chips":
            out_chips(ax, x - 0.02, 3.5, 1.64, ["Arable land", "Pastures", "Inland waters"], h=0.27, gap=0.05, fs=14)
        elif out == "map":
            out_cropmap(fig, ax, x + 0.35, 3.5, 0.9, pid)
        else:
            out_biomass(fig, ax, x + 0.35, 3.5, 0.9)
    legend_row(ax, [("trained", "trains"), ("frozen", "frozen"), ("new", "new, for one task"), ("gone", "thrown away")],
               [2.0, 4.6, 7.2, 10.6])
    save(fig, "arch_encdec.png")


def fig_arch_deconly(pid=40305):
    fig, ax = arch_canvas()
    by, bh, bw, s, p = 1.25, 1.25, 6.2, 0.36, 0.46
    half_titles(ax, "no labels: predict the next token", "the same decoder writes every task",
                gap=(by + 0.15, by + bh + 0.75))
    # ---- pretraining: every modality's tokens in, every next token out
    lx = 0.4
    arch_box(ax, lx, by, bw, bh, "pre", "decoder", fs=19, bold=True)
    icon(ax, "trained", lx + bw, by + bh)
    seq = group("S2L2A", 5) + group("S1", 5) + group("DEM", 4)
    x0 = lx + (bw - (13 * p - (p - s))) / 2
    for i in range(13):
        x = x0 + i * p
        sq(ax, x, 0.35, s, MODC[seq[i][0]], seq[i][1])
        sq(ax, x, 3.0, s, MODC[seq[i + 1][0]], seq[i + 1][1])
        arrow(ax, (x + s / 2, 0.73), (x + s / 2, by - 0.02), lw=1.1, ms=8)
        arrow(ax, (x + s / 2, by + bh + 0.02), (x + s / 2, 2.98), lw=1.1, ms=8)
    for lab, mod, k0, k1 in (("S2", "S2L2A", 0, 4), ("S1", "S1", 5, 9), ("DEM", "DEM", 10, 12)):
        ax.text(x0 + (k0 + k1) / 2 * p + s / 2, 0.27, lab, ha="center", va="top", fontsize=15, fontweight="bold",
                color=MODC[mod])
    # ---- downstream: the same decoder, nothing thrown away
    rx = 7.75
    arch_box(ax, rx, by, bw, bh, "pre", "decoder", fs=19, bold=True)
    icon(ax, "frozen", rx + bw, by + bh)
    arrow(ax, (lx + bw + 0.04, by + bh / 2), (rx - 0.04, by + bh / 2), col=INK, lw=2.4, ms=20)
    ax.text((lx + bw + rx) / 2, by + bh + 0.55, "the same\nmodel", ha="center", va="center", fontsize=14.5,
            fontweight="bold", linespacing=1.1)
    # inputs: an image's tokens, then the task's own new tokens
    inp = group("S2L2A", 5) + [("new", False)] * 4
    x0 = rx + (bw - (len(inp) * p - (p - s))) / 2
    for i, (m, mk) in enumerate(inp):
        x = x0 + i * p
        sq(ax, x, 0.35, s, MODC.get(m, NEW), mk, new=m == "new")
        arrow(ax, (x + s / 2, 0.73), (x + s / 2, by - 0.02), lw=1.1, ms=8)
    ax.text(x0 + 2 * p + s / 2, 0.27, "S2", ha="center", va="top", fontsize=15, fontweight="bold", color=MODC["S2L2A"])
    ax.text(x0 + 6.5 * p + s / 2, 0.27, "the task's tokens", ha="center", va="top", fontsize=15, fontweight="bold",
            color=NEW)
    icon(ax, "trained", x0 + 9 * p + 0.12, 0.53, 0.8)
    # outputs: every task, written by the same decoder
    for k, (lab, out) in enumerate((("land cover", "chips"), ("crop types", "map"), ("biomass", "field"))):
        xc = rx + bw * (2 * k + 1) / 6
        arrow(ax, (xc, by + bh + 0.02), (xc, 2.93), lw=1.4, ms=11)
        ax.text(xc, 4.0, lab, ha="center", va="bottom", fontsize=15.5)
        if out == "chips":
            out_chips(ax, xc - 0.82, 2.95, 1.64, ["Arable land", "Pastures", "Inland waters"], h=0.28, gap=0.05, fs=14)
        elif out == "map":
            out_cropmap(fig, ax, xc - 0.45, 2.95, 0.9, pid)
        else:
            out_biomass(fig, ax, xc - 0.45, 2.95, 0.9)
    legend_row(ax, [("trained", "trains"), ("frozen", "frozen"), ("new", "new, for one task")], [3.4, 6.4, 9.4])
    save(fig, "arch_deconly.png")


# ---------------------------------------------------------------------------------- 3. what we built

def fig_pipeline():
    boxes = [("89,088", "TerraMesh scenes,\n6 image modalities\n+ location"),
             ("196", "tokens per image\n(IBM's tokenizers,\n224 × 224 px)"),
             ("87,556", "token ids in\none shared\nvocabulary"),
             ("~1,000", "tokens per scene,\nmodality order\nshuffled"),
             ("153 M", "parameters,\n29-layer decoder\n(MoR: 90 M)"),
             ("10 epochs", "0.84 B tokens,\n~8 h on\none A100")]
    W, H = 14.0, 2.25
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    bw, gap = 2.0, 0.33
    for i, (n, t) in enumerate(boxes):
        x = 0.05 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, 0.06), bw, 2.12, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="white", ec="#d7d6d0", lw=1.3))
        ax.text(x + bw / 2, 1.72, n, ha="center", va="center", fontsize=23, fontweight="bold")
        ax.text(x + bw / 2, 0.74, t, ha="center", va="center", fontsize=13.5, color=INK2, linespacing=1.25)
        if i < len(boxes) - 1:
            arrow(ax, (x + bw + 0.02, 1.12), (x + bw + gap - 0.02, 1.12), ms=13, lw=1.5)
    save(fig, "pipeline.png")


def fig_split_map():
    import pandas as pd
    m = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet", columns=["center_lon", "center_lat"])
    ev = np.array(jl(REPO / "eo/data/eval_rows_v1.json")["eval_rows"], dtype=int)
    held = np.zeros(len(m), bool); held[ev] = True
    fig, ax = plt.subplots(figsize=(7.4, 3.55))
    ax.scatter(m.center_lon[~held], m.center_lat[~held], s=0.35, color="#b9b8b1", lw=0, rasterized=True)
    ax.scatter(m.center_lon[held], m.center_lat[held], s=2.6, color="#b0499a", lw=0, rasterized=True)
    ax.set_xlim(-180, 180); ax.set_ylim(-58, 80); ax.set_aspect("equal"); ax.set_axis_off()
    ax.scatter([], [], s=40, color="#b9b8b1", label=f"training scenes  {int((~held).sum()):,}")
    ax.scatter([], [], s=40, color="#b0499a", label=f"held-out scenes  {int(held.sum()):,}")
    ax.legend(loc="upper center", fontsize=14.5, handletextpad=0.2, ncol=2, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5)
    ax.set_title("Held out by region: ≥ 5.9 km from any training scene", fontsize=15.5, loc="left")
    save(fig, "split_map.png")


def _losses(run):
    s = jl(next((RUNS).glob(f"{run}_202609*/checkpoint-16500/trainer_state.json")))
    ev = np.array([(h["step"], h["eval_loss"]) for h in s["log_history"] if "eval_loss" in h])
    tr = np.array([(h["step"], h["loss"]) for h in s["log_history"] if "loss" in h])
    k = 25
    sm = np.convolve(tr[:, 1], np.ones(k) / k, mode="valid")
    return ev, np.c_[tr[k // 2: k // 2 + len(sm), 0], sm]


def fig_memorise():
    fig, ax = plt.subplots(figsize=(6.3, 3.55))
    ygrid(ax)
    ax.axvspan(0, 10, color="#efeee8", lw=0, zorder=0)
    ax.text(5, 3.68, "our runs:\n10 epochs", ha="center", va="bottom", fontsize=14, color=INK2, linespacing=1.15)
    best = {}
    for run, col, nm in [("vanilla", MODEL["b10c"], "vanilla"), ("mor", MODEL["a10c"], "MoR")]:
        ev, tr = _losses(run)
        ax.plot(tr[:, 0] / EPOCH, tr[:, 1], color=col, lw=1.3, ls=(0, (3, 2)), alpha=0.9)
        ax.plot(ev[:, 0] / EPOCH, ev[:, 1], color=col, lw=2.4, marker="o", ms=3.5)
        best[nm] = ev[np.argmin(ev[:, 1])]
    b = best["vanilla"]
    ax.annotate(f"held-out best: epoch {b[0] / EPOCH:.1f}", (b[0] / EPOCH, b[1]), xytext=(11.5, 6.5),
                fontsize=14, arrowprops=dict(arrowstyle="-", color=INK2, lw=1), va="center")
    ax.text(50.5, 6.83, "held-out", fontsize=14, va="center", ha="left")
    ax.text(50.5, 3.86, "training", fontsize=14, va="center", ha="left")
    ax.plot([], [], color=MODEL["b10c"], lw=2.4, label="vanilla"); ax.plot([], [], color=MODEL["a10c"], lw=2.4, label="MoR")
    ax.legend(loc="upper right", fontsize=14, bbox_to_anchor=(0.92, 1.02), ncol=2, columnspacing=1.0)
    ax.set_xlim(0, 50); ax.set_ylim(3.6, 7.1)
    ax.set_xlabel("epochs (first 50-epoch runs)"); ax.set_ylabel("loss (nats/token)")
    ax.set_title("Held-out loss is best at ~8 epochs,\nthen the model memorises", fontsize=15.5)
    save(fig, "memorise.png")
    return {k: (round(float(v[0] / EPOCH), 2), round(float(v[1]), 4)) for k, v in best.items()}


# ------------------------------------------------------------------------------- 4. pretraining

def fig_gain():
    g = jl(REP / "grid/a10c_b10c/grid_report.json")["arms"]["b10c"]["cells"]
    rows = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords", "all"]
    M = np.full((len(rows), len(TARGETS)), np.nan)
    for i, s in enumerate(rows):
        for j, t in enumerate(TARGETS):
            c = g.get(f"{s}->{t}")
            if c is not None:
                M[i, j] = -c["gain"]["mean"]          # nats/token saved against no context
    cm = LinearSegmentedColormap.from_list("helps", ["#c44e52", "#f2f1ec", "#13708a"])
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=-1.3, vmax=1.3, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec=GRID, hatch="///", lw=0))
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00"), ha="center", va="center",
                    fontsize=13.5, color="white" if v > 0.75 else INK, fontweight="bold" if v > 0.75 else "normal")
    ax.axhline(6.5, color=INK2, lw=1.4)
    ax.set_xticks(range(len(TARGETS)), [NICE[t] for t in TARGETS], fontsize=14)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_yticks(range(len(rows)), [NICE[r] for r in rows], fontsize=14)
    ax.set_xlabel("target the model writes", fontsize=14, labelpad=8)
    ax.set_ylabel("source it is given", fontsize=14)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("")
    save(fig, "gain.png")
    return {f"{r}->{t}": round(float(M[i, j]), 3) for i, r in enumerate(rows) for j, t in enumerate(TARGETS)
            if not np.isnan(M[i, j])}


def fig_generation():
    row = GEN_SCENE
    t = truth("NDVI", row)
    panels = [("S2L2A", "given: S2", truth("S2L2A", row), None, MODC["S2L2A"]),
              ("NDVI", "true NDVI", t, t, INK)]
    for src, lab in [("S2L2A", "from S2"), ("S1GRD", "from S1"), ("DEM", "from DEM"), ("none", "from nothing")]:
        panels.append(("NDVI", lab, gen("NDVI", src, row), t, MODC.get(src, MUTED)))
    fig, axs = plt.subplots(1, 6, figsize=(13.2, 2.75), gridspec_kw=dict(wspace=0.07))
    for ax, (m, lab, img, ref, col) in zip(axs, panels):
        show(ax, m, img, ref)
        frame(ax, col if lab != "true NDVI" else INK, 2.5 if lab != "true NDVI" else 3)
        ax.set_title(lab, fontsize=14.5, loc="center", fontweight="bold" if lab == "true NDVI" else "normal")
    save(fig, "generation.png")


# ---------------------------------------------------------------------------------- 5. recursion idea

def fig_mor():
    W, H = 6.1, 4.95
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    # vanilla tower: 29 different layers
    vx, lw_, base = 0.35, 1.35, 0.55
    lh = 3.75 / 29
    for k in range(29):
        ax.add_patch(Rectangle((vx, base + k * lh), lw_, lh * 0.82, fc=tint(MODEL["b10c"], 0.15 + 0.6 * (k % 2)), ec="none"))
    ax.text(vx + lw_ / 2, base + 3.85, "vanilla", ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.text(vx + lw_ / 2, base - 0.08, "29 different\nlayers", ha="center", va="top", fontsize=15, color=INK2, linespacing=1.1)
    # MoR tower: first layer, a shared block of 9 layers applied 1-3 times, last layer
    mx = 2.6
    ax.add_patch(Rectangle((mx, base), lw_, 0.28, fc=tint(MODEL["a10c"], 0.35), ec="none"))
    ax.add_patch(Rectangle((mx, base + 3.47), lw_, 0.28, fc=tint(MODEL["a10c"], 0.35), ec="none"))
    by, bh = base + 0.55, 2.6
    ax.add_patch(FancyBboxPatch((mx, by), lw_, bh, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=tint(MODEL["a10c"], 0.78), ec=MODEL["a10c"], lw=2))
    for k in range(9):
        ax.plot([mx + 0.12, mx + lw_ - 0.12], [by + 0.18 + k * (bh - 0.36) / 8] * 2, color=MODEL["a10c"], lw=1, alpha=0.5)
    ax.text(mx + lw_ / 2, by + bh / 2, "shared\nblock\n(9 layers)", ha="center", va="center", fontsize=15,
            fontweight="bold", color=INK, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.15", fc=tint(MODEL["a10c"], 0.78), ec="none"))
    ax.add_patch(FancyArrowPatch((mx + lw_ + 0.05, by + bh - 0.2), (mx + lw_ + 0.05, by + 0.2),
                                 connectionstyle="arc3,rad=-0.42", arrowstyle="-|>", mutation_scale=18,
                                 color=MODEL["a10c"], lw=2.2))
    ax.text(mx + lw_ + 0.78, by + bh / 2, "×1, 2 or 3,\nchosen per\ntoken", ha="left", va="center", fontsize=16.5,
            color=MODEL["a10c"], fontweight="bold", linespacing=1.15)
    ax.text(mx + lw_ / 2, base + 3.85, "MoR", ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.text(mx + lw_ / 2, base - 0.08, "11 different\nlayers", ha="center", va="top", fontsize=15, color=INK2, linespacing=1.1)
    save(fig, "mor.png")


# -------------------------------------------------------------------------------- 6. recursion result

def _curve(pair, arm, step=3300):
    for r in csv.DictReader(open(REP / f"grid/{pair}/curve.csv")):
        if r["arm"] == arm and r["checkpoint"] != "untrained" and int(r["step"]) == step:
            return float(r["pooled_ce"]), float(r["layers_per_body_token"])
    raise KeyError((pair, arm, step))


def _depth(tag, step=3300):
    from eo.data.eo_vocab import MODALITY_TO_ID
    z = np.load(REP / f"arm_eval/{tag}/checkpoint-{step}/tokens_router.npz")
    return [float(z["depth"][z["label_mod"] == MODALITY_TO_ID[m]].mean()) for m in MODS]


def fig_recursion():
    """Per-modality depth in two seeds. The interventions and the end-of-training points are kept in
    numbers.json for the presenter notes; the page shows the seeds only."""
    out = {}
    kinds = ["skip", "force1", "force2", "perm_all", "perm_modality"]
    for tag in ("a10c", "a10cs2"):
        mo = jl(REP / f"arm_eval/{tag}/checkpoint-3300/metrics.json")["modes"]
        r = mo["router"]["ce_all_body_tokens"]
        out.setdefault("interventions", {})[tag] = {k: round(mo[k]["ce_all_body_tokens"] - r, 4) for k in kinds}
    pts = {"b10c": _curve("a10c_b10c", "b10c"), "a10c": _curve("a10c_b10c", "a10c"),
           "a10cs2": _curve("a10c_a10cs2", "a10cs2"), "a10cr": _curve("a10c_a10cr", "a10cr"),
           "b10c11": _curve("a10c_b10c11", "b10c11")}
    out["end"] = {k: (round(v[0], 3), round(v[1], 1)) for k, v in pts.items()}
    # per-modality depth, two seeds: a dumbbell per modality
    d1, d2 = _depth("a10c"), _depth("a10cs2")
    out["depth"] = {"a10c": dict(zip(MODS, [round(v, 2) for v in d1])), "a10cs2": dict(zip(MODS, [round(v, 2) for v in d2]))}
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    x = np.arange(len(MODS))
    ygrid(ax)
    for xi, u, v in zip(x, d1, d2):
        ax.plot([xi, xi], [u, v], color=tint(MODEL["a10c"], 0.45), lw=3, zorder=3, solid_capstyle="round")
    ax.scatter(x, d1, s=130, color=MODEL["a10c"], zorder=5, label="MoR, seed 1")
    ax.scatter(x, d2, s=130, color="white", edgecolor=MODEL["a10c"], lw=2.4, zorder=5, label="MoR, seed 2")
    ax.set_xticks(x, [NICE[m].replace("S1 ", "S1\n").replace("location", "loc.") for m in MODS], fontsize=14)
    ax.set_xlim(-0.5, len(MODS) - 0.5)
    ax.set_ylim(0.8, 3.45); ax.set_yticks([1, 2, 3], ["1", "2", "3"])
    ax.set_ylabel("mean passes per token"); ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", fontsize=14, ncol=2, bbox_to_anchor=(0.5, 1.12), handletextpad=0.3)
    save(fig, "recursion.png")
    return out


# ------------------------------------------------------------------------------------------- 7. BEN

def fig_ben_tokens():
    from eo.ben.labels import NEW_LABELS
    W, H = 13.6, 1.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    y, h = 0.18, 0.72
    tok_block(ax, 0.05, y, 2.6, h, MODC["S2L2A"], "S2 · 196 tokens", fs=14.5)
    tok_block(ax, 2.8, y, 2.6, h, MODC["S1"], "S1 · 196 tokens", fs=14.5)
    ax.text(2.72, y + h + 0.12, "given: S1, S2 or both", ha="center", va="bottom", fontsize=14.5, fontweight="bold")
    arrow(ax, (5.5, y + h / 2), (6.2, y + h / 2))
    want = ["Coniferous forest", "Mixed forest", "Inland waters"]          # fig_ben_task's patch
    labs = [n for n in NEW_LABELS if n in want] or want
    tok_block(ax, 6.3, y, 7.2, h, MODC["TASK"], "", body="white")
    cx = 6.3 + 0.55
    for n in labs:
        w = 0.105 * len(n) + 0.4
        ax.add_patch(FancyBboxPatch((cx, y + 0.13), w, h - 0.26, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=tint(MODC["TASK"], 0.8), ec=MODC["TASK"], lw=1.2, zorder=6))
        ax.text(cx + w / 2, y + h / 2, n, ha="center", va="center", fontsize=13.5, zorder=7)
        cx += w + 0.18
    ax.text(9.9, y + h + 0.12, "the model writes the labels: 19 new class ids", ha="center", va="bottom",
            fontsize=14.5, fontweight="bold", color=MODC["TASK"])
    save(fig, "ben_tokens.png")


def fig_ben_results():
    S = jl(REP / "ben/summary_b10/summary.json"); m = S["models"]
    loc = S["location_only_knn10_micro_map"]
    wrong = m["b10_lora"]["test_S1S2_wrong"]
    order = ["b10_scratch", "b10_head", "b10_full", "b10_lora"]          # LoRA, the headline, last
    fig, ax = plt.subplots(figsize=(10.4, 4.75))
    ygrid(ax)
    w = 0.19
    for g, key in enumerate(("S1", "S2", "S1S2")):
        for i, k in enumerate(order):
            v = m[k][f"test_{key}"]["micro_map"]
            x = g + (i - 1.5) * (w + 0.02)
            ax.bar(x, v, w, color=BEN[k], zorder=3, label=BENNAME[k] if g == 0 else None)
            ax.text(x, v - 0.015, f"{v:.2f}", ha="center", va="top", fontsize=14, zorder=6,      # inside the bar
                    fontweight="bold", color=INK if k == "b10_head" else "white")
        s = SMARTIES[key]
        ax.plot([g - 0.42, g + 0.42], [s, s], color=INK, lw=1.6, ls=(0, (4, 3)), zorder=4)
        ax.text(g, s + 0.012, f"SMARTIES {s:.2f}", ha="center", va="bottom", fontsize=14)
    ax.set_xticks(range(3), ["S1  (radar)", "S2  (optical)", "S1 + S2"], fontsize=15)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 1.0); ax.set_ylabel("micro-mAP (103,944 test patches)")
    ax.legend(loc="upper left", ncol=4, fontsize=14.5, bbox_to_anchor=(0.0, 1.14), columnspacing=1.4, handlelength=1.3)
    save(fig, "ben_results.png")
    return {"loc": round(loc, 3), "wrong": round(wrong, 3),
            "bars": {k: {t: round(m[k][f"test_{t}"]["micro_map"], 3) for t in ("S1", "S2", "S1S2")} for k in order}}


def fig_ben_labels():
    S = jl(REP / "ben/summary_b10/summary.json"); m = S["models"]
    v = {k: m[k]["test_S1S2"]["micro_map"] for k in ("b10_scratch_1pct", "b10_lora_1pct", "b10_scratch", "b10_lora")}
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    ygrid(ax)
    x = [0, 1]
    sc, lo = [v["b10_scratch_1pct"], v["b10_scratch"]], [v["b10_lora_1pct"], v["b10_lora"]]
    ax.plot(x, sc, color=BEN["b10_scratch"], lw=2.6, marker="o", ms=10, zorder=4)
    ax.plot(x, lo, color=BEN["b10_lora"], lw=2.6, marker="o", ms=10, zorder=4)
    for xi in x:
        ax.annotate("", xy=(xi, lo[xi] - 0.012), xytext=(xi, sc[xi] + 0.012),
                    arrowprops=dict(arrowstyle="-|>", color=BEN["b10_lora"], lw=1.6, mutation_scale=14))
        ax.text(xi + (0.06 if xi == 0 else -0.06), (lo[xi] + sc[xi]) / 2, f"+{lo[xi] - sc[xi]:.2f}",
                ha="left" if xi == 0 else "right", va="center", fontsize=17, fontweight="bold", color=BEN["b10_lora"])
        ax.text(xi, lo[xi] + 0.022, f"{lo[xi]:.2f}", ha="center", va="bottom", fontsize=14)
        ax.text(xi, sc[xi] - 0.025, f"{sc[xi]:.2f}", ha="center", va="top", fontsize=14)
    ax.text(1.06, lo[1], "pretrained,\nLoRA", ha="left", va="center", fontsize=13.5, color=INK)
    ax.text(1.06, sc[1] + 0.012, "from\nscratch", ha="left", va="center", fontsize=13.5, color=INK)
    ax.set_xticks(x, ["1% of labels\n3,116 patches", "10% of labels\n31,166 patches"], fontsize=13.5)
    ax.set_xlim(-0.3, 1.38); ax.set_ylim(0.4, 0.9); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("micro-mAP, S1 + S2")
    save(fig, "ben_labels.png")
    return {k: round(x_, 3) for k, x_ in v.items()}


def fig_ben_task(row=304566):
    """The first downstream task, and the four ways it was trained. `row`: a test patch the LoRA model
    labels exactly (checked below), so the chips are what the model wrote."""
    import pandas as pd
    from eo.ben import labels as L, layout as Ly
    z = np.load(REP / "ben/b10_lora/checkpoint-1461/test_a/S1S2.npz")
    i = int(np.flatnonzero(z["rows"] == row)[0])
    assert (z["truth"][i] == z["greedy"][i]).all()
    names = [L.NEW_LABELS[j] for j in np.flatnonzero(z["truth"][i])]
    n_rows = len(pd.read_parquet(Ly.INDEX, columns=["row"]))
    a2, r2p2 = Ly.load_packed("S2", n_rows); a1, r2p1 = Ly.load_packed("S1", n_rows)
    b = a2["S2_10m"][r2p2[row]].astype(np.float32)                   # B02 B03 B04 B08, as ben_examples.py
    s2 = np.clip(np.stack([b[2], b[1], b[0]], -1) / 2800.0, 0, 1) ** (1 / 1.3)
    s1 = np.clip((a1["S1"][r2p1[row]][0] + 25.0) / 25.0, 0, 1)       # VV
    W, H = 14.0, 6.4
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    # ---- the task: S1 and S2 in, the classes out, as tokens
    ty, ts = 3.75, 2.05
    yc = ty + ts / 2
    for x, img, lab, mod in ((0.3, s2, "S2", "S2L2A"), (2.5, s1, "S1", "S1")):
        a = inset(fig, ax, x, ty, ts); a.imshow(img, cmap="gray" if lab == "S1" else None); frame(a, MODC[mod], 2.5)
        ax.text(x + ts / 2, ty + ts + 0.08, lab, ha="center", va="bottom", fontsize=17, fontweight="bold")
    arrow(ax, (4.7, yc), (5.25, yc))
    arch_box(ax, 5.3, yc - 0.7, 3.2, 1.4, "pre", "pretrained\ndecoder", fs=19, bold=True)
    arrow(ax, (8.55, yc), (9.1, yc))
    out_chips(ax, 9.2, yc - 0.78, 2.6, names, h=0.44, gap=0.11, fs=16)
    ax.text(11.98, yc, "each class is a\nnew token that\nthe model writes", ha="left", va="center", fontsize=16.5,
            color=MODC["TASK"], fontweight="bold", linespacing=1.2)
    # ---- what was compared: which parts train
    ax.text(0.3, 3.1, "Four ways to train it", ha="left", va="center", fontsize=19, fontweight="bold")
    flame(ax, 10.2, 3.1, 0.24); ax.text(10.55, 3.1, "trains", ha="left", va="center", fontsize=16)
    snowflake(ax, 12.0, 3.1, 0.18); ax.text(12.35, 3.1, "frozen", ha="left", va="center", fontsize=16)
    regimes = [("from scratch", BEN["b10_scratch"], "random", "trained", False, "all 153 M"),
               ("head only", BEN["b10_head"], "pretrained", "frozen", False, "12 k"),
               ("LoRA", BEN["b10_lora"], "pretrained", "frozen", True, "4.7 M  (3%)"),
               ("full fine-tune", FULL, "pretrained", "trained", False, "all 153 M")]
    cw, cg = 3.3, 0.13
    for k, (name, col, init, state, lora, n) in enumerate(regimes):
        x = 0.2 + k * (cw + cg)
        ax.add_patch(FancyBboxPatch((x, 0.05), cw, 2.65, boxstyle="round,pad=0,rounding_size=0.1", fc="white",
                                    ec="#d7d6d0", lw=1.3, zorder=1))
        ax.add_patch(Rectangle((x + 0.18, 2.17), 0.28, 0.28, fc=col, ec="none", zorder=3))
        ax.text(x + 0.58, 2.31, name, ha="left", va="center", fontsize=18.5, fontweight="bold")
        arch_box(ax, x + 0.15, 0.95, 1.42, 0.72, "pre", init, fs=15.5)
        icon(ax, state, x + 0.86, 1.75, 0.85)
        if lora:
            arch_box(ax, x + 1.68, 0.95, 0.7, 0.72, "new", "LoRA", fs=14)
            icon(ax, "trained", x + 2.03, 1.75, 0.85)
        arch_box(ax, x + 2.43, 0.95, 0.76, 0.72, "new", "class\ntokens", fs=13.5)
        icon(ax, "trained", x + 2.81, 1.75, 0.85)
        ax.text(x + cw / 2, 0.45, f"trains {n}", ha="center", va="center", fontsize=17)
    save(fig, "ben_task.png")
    return {"row": row, "labels": names}


# ---------------------------------------------------------------------------------------- 9. PASTIS

PASTIS_CLASSES = ["background", "meadow", "soft winter wheat", "corn", "winter barley", "winter rapeseed",
                  "spring barley", "sunflower", "grapevine", "beet", "winter triticale", "winter durum wheat",
                  "fruits, vegetables, flowers", "potatoes", "leguminous fodder", "soybeans", "orchard",
                  "mixed cereal", "sorghum", "void"]


def pastis_cols():
    """Background, the 18 crop classes (tab20 without its greys), void."""
    base = [c for i, c in enumerate(plt.get_cmap("tab20").colors) if i not in (14, 15)]
    order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 11, 13, 15, 17]
    return ["#f4f3ee"] + [base[i] for i in order] + ["#b4b3ad"]


def pastis_grid(sem, n=28):
    """The n x n class grid: each native pixel goes to the cell that contains its centre (plan §4.3)."""
    cell = np.floor((np.arange(sem.shape[0]) + 0.5) * n / sem.shape[0]).astype(int)
    grid = np.zeros((n, n), int)
    for a in range(n):
        for b in range(n):
            grid[a, b] = np.bincount(sem[np.ix_(cell == a, cell == b)].ravel(), minlength=20).argmax()
    return grid


def pastis_sem(pid):
    return np.load(PASTIS / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0].astype(int)


def pastis_step0(pid):
    import pandas as pd
    return pd.read_parquet(REP / "pastis/step0/patches.parquet").set_index("id").loc[pid]


@functools.lru_cache(maxsize=None)
def pastis_rgb(pid):
    """The patch's protocol image (its clearest July-August date, Step 0) as RGB."""
    import ast
    p = pastis_step0(pid)
    meta = jl(PASTIS / "metadata.geojson")
    props = next(f["properties"] for f in meta["features"] if int(f["properties"]["ID_PATCH"]) == pid)
    d = props["dates-S2"]; d = d if isinstance(d, dict) else ast.literal_eval(d)
    dates = [int(d[k]) for k in sorted(d, key=int)]
    s2 = np.load(PASTIS / "DATA_S2" / f"S2_{pid}.npy", mmap_mode="r")[dates.index(int(p.summer_date))].astype(np.float32)
    return np.clip(s2[[2, 1, 0]].transpose(1, 2, 0) / 2500.0, 0, 1)        # B04, B03, B02


def fig_pastis(pid=40305):
    p = pastis_step0(pid)
    rgb, sem = pastis_rgb(pid), pastis_sem(pid)
    grid = pastis_grid(sem)
    cols = pastis_cols()
    cmap = ListedColormap(cols)
    fig = plt.figure(figsize=(13.4, 3.9))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 0.3, 1, 0.3, 1, 0.22, 1.2], wspace=0.0)
    a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4])
    a0.imshow(rgb); a0.set_axis_off()
    dd = str(p.summer_date)
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(dd[4:6]) - 1]
    a0.set_title(f"input: one S2 image\n{int(dd[6:])} {mon} {dd[:4]}", fontsize=14, loc="center", fontweight="normal")
    a1.imshow(sem, cmap=cmap, vmin=0, vmax=19, interpolation="nearest"); a1.set_axis_off()
    a1.set_title("true crop map\n128 × 128 px of 10 m", fontsize=14, loc="center", fontweight="normal")
    a2.imshow(grid, cmap=cmap, vmin=0, vmax=19, interpolation="nearest")
    for k in np.arange(-0.5, 28, 1):
        a2.axhline(k, color="white", lw=0.35); a2.axvline(k, color="white", lw=0.35)
    a2.set_axis_off()
    a2.set_title("the model writes\n784 class tokens (28 × 28)", fontsize=14, loc="center", fontweight="bold",
                 color=MODC["TASK"])
    for ax_ in (fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])):
        ax_.set_axis_off(); ax_.set_xlim(0, 1); ax_.set_ylim(0, 1)
        ax_.add_patch(FancyArrowPatch((0.1, 0.5), (0.9, 0.5), arrowstyle="-|>", mutation_scale=22, color=INK2, lw=2))
    al = fig.add_subplot(gs[0, 6]); al.set_axis_off()
    present = np.bincount(sem.ravel(), minlength=20)
    top = [k for k in np.argsort(-present) if present[k] > 0][:7]
    for i, k in enumerate(top):
        al.add_patch(Rectangle((0.0, 0.9 - i * 0.13), 0.1, 0.09, fc=cols[k], ec=GRID, transform=al.transAxes))
        al.text(0.15, 0.945 - i * 0.13, PASTIS_CLASSES[k] + (" (not scored)" if k == 19 else ""), va="center",
                fontsize=14, transform=al.transAxes)
    save(fig, "pastis.png")
    return {"pid": pid, "date": int(p.summer_date), "fold": int(p.fold), "tile": p.tile,
            "classes": [PASTIS_CLASSES[k] for k in top]}


def fig_next(pid=40305):
    """The second half: segmentation as tokens, and two tasks that condition each other."""
    W, H = 14.0, 5.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    TC = MODC["TASK"]
    ax.plot([6.95, 6.95], [0.1, 5.2], color=GRID, lw=1.5, zorder=0)
    # ---- left: one model, both tasks
    ax.text(3.4, 5.02, "One model, both tasks", ha="center", va="center", fontsize=19, fontweight="bold")
    a = inset(fig, ax, 0.1, 1.55, 1.8); a.imshow(pastis_rgb(pid)); frame(a, MODC["S2L2A"], 2.5)
    arrow(ax, (1.97, 2.45), (2.3, 2.45))
    arch_box(ax, 2.32, 0.9, 1.6, 3.1, "pre", "decoder", fs=19, bold=True)
    arrow(ax, (3.95, 3.5), (4.33, 3.62))
    out_chips(ax, 4.38, 3.2, 1.9, ["Arable land", "Pastures"], h=0.4, gap=0.1, fs=15.5)
    ax.text(5.33, 4.2, "scene labels", ha="center", va="bottom", fontsize=16)
    arrow(ax, (3.95, 1.3), (4.58, 1.25))
    out_cropmap(fig, ax, 4.62, 0.55, 1.3, pid)
    ax.text(5.27, 0.47, "crop map", ha="center", va="top", fontsize=16)
    ax.add_patch(FancyArrowPatch((5.33, 1.95), (5.33, 3.12), arrowstyle="<|-|>", mutation_scale=18, color=TC, lw=2.4,
                                 zorder=5))
    ax.text(5.5, 2.53, "each informs\nthe other", ha="left", va="center", fontsize=15, color=TC, fontweight="bold",
            linespacing=1.1)
    # ---- right: in one sequence, in either order
    ax.text(10.45, 5.02, "One sequence, either order", ha="center", va="center", fontsize=19, fontweight="bold")
    widths = {"S2": 1.65, "scene labels": 2.4, "crop map": 2.4}
    for y, order, note in ((2.85, ["S2", "scene labels", "crop map"], "the labels guide the map"),
                           (0.55, ["S2", "crop map", "scene labels"], "the map refines the labels")):
        x, cx = 7.15, []
        for lab in order:
            w = widths[lab]
            tok_block(ax, x, y, w, 0.66, MODC["S2L2A"] if lab == "S2" else TC, lab, cap=0.36, fs=16.5,
                      body=None if lab == "S2" else "white")
            cx.append(x + w / 2); x += w + 0.12
        ax.add_patch(FancyArrowPatch((cx[1], y + 0.72), (cx[2], y + 0.72), connectionstyle="arc3,rad=-0.35",
                                     arrowstyle="-|>", mutation_scale=18, color=TC, lw=2.2, zorder=5))
        ax.text((cx[1] + cx[2]) / 2, y + 1.42, note, ha="center", va="center", fontsize=17, color=TC, fontweight="bold")
    save(fig, "next.png")


FIGS = {"strip": fig_strip, "arch_encdec": fig_arch_encdec, "arch_deconly": fig_arch_deconly, "sequence": fig_sequence,
        "pipeline": fig_pipeline, "split_map": fig_split_map, "memorise": fig_memorise, "gain": fig_gain,
        "generation": fig_generation, "mor": fig_mor, "recursion": fig_recursion, "ben_task": fig_ben_task,
        "ben_tokens": fig_ben_tokens, "ben_results": fig_ben_results, "ben_labels": fig_ben_labels,
        "next": fig_next, "pastis": fig_pastis}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {}
    for name in a.only or FIGS:
        r = FIGS[name]()
        if r is not None:
            numbers[name] = r
    if numbers:
        p = OUT / "numbers.json"
        old = jl(p) if p.exists() else {}
        old.update(numbers)
        p.write_text(json.dumps(old, indent=1, default=str))
        print(json.dumps(numbers, indent=1, default=str))


if __name__ == "__main__":
    main()
