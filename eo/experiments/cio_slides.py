#!/usr/bin/env python
"""Figures for the continuous-I/O deck (docs/slides/continuous_io.md, 2026-10-28). `.venv`, CPU.

Diagrams follow the midterm deck (midterm_figures.py: its helpers, colours and icons: a flame trains,
a snowflake is frozen, red hatching is new, a red cross is removed). Charts read the pilot's stored
results; nothing is recomputed from a model:
  /data/enric/experiments/continuous_io/analysis/{report.json, pca_dct.json}
  /data/enric/experiments/continuous_io/{data/test_*.npy, runs/*/rollout/*.npy}   (example images)
Every number a chart shows is also written to figs/numbers.json. Literature numbers are quoted from
docs/notes/continuous_io_literature_review.md (Part I), with the paper on the slide.

Colours: a pilot variant keeps one colour on every page: FSQ tokens grey (the reference); pixels
yellow, level + relief blue, one embedder magenta, with band dropout aqua; regression orange, flow on
raw pixels violet, flow on DCT coefficients green. Each chart's set passes the dataviz validator.

    .venv/bin/python eo/experiments/cio_slides.py [--only NAME ...]
Writes /data/enric/reports/slides/continuous_io/figs/*.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eo.experiments.midterm_figures import (  # noqa: E402  (importing it sets the house rcParams)
    BAD, GONE, GRID, INK, INK2, LULC_CMAP, MODC, MUTED, NEW, PRE_EC, SCENE, SURF, arch_box, arrow, crossed, flame,
    frame, icon, inset, legend_row, out_biomass, pastis_cols, pastis_grid, pastis_rgb, pastis_sem, show,
    snowflake, sq, tint, truth, xgrid, ygrid)

CIO = Path("/data/enric/experiments/continuous_io")
OUT = Path("/data/enric/reports/slides/continuous_io/figs")
COL = {"tok_in": "#52514e", "lin_in": "#eda100", "linraw_in": "#eda100", "linrel_in": "#2a78d6",
       "band_in": "#e87ba4", "bandrop_in": "#1baf7a", "reg_out": "#eb6834", "flow_out": "#4a3aa7",
       "flowdct_out": "#008300"}
NAME = {"tok_in": "FSQ tokens (today)", "lin_in": "pixels → linear", "linraw_in": "pixels → linear, no LayerNorm",
        "linrel_in": "level + relief → linear", "band_in": "one embedder, any bands",
        "bandrop_in": "one embedder + band dropout", "reg_out": "regression", "flow_out": "flow, raw pixels",
        "flowdct_out": "flow, 16 DCT per band"}
GOOD = "#1f8a4c"
TOKC = "#f1f0eb"                       # a tokenizer or other fixed, non-decoder component
SHORT = {"S2L2A": "S2", "S1RTC": "S1", "DEM": "DEM", "LULC": "LULC"}
PAIRS = ["S2L2A>S1RTC", "S2L2A>DEM", "S2L2A>LULC", "S1RTC>S2L2A", "S1RTC>DEM", "S1RTC>LULC",
         "DEM>S2L2A", "DEM>S1RTC", "DEM>LULC", "LULC>S2L2A", "LULC>S1RTC", "LULC>DEM"]
NUM: dict = {}                         # every number a chart shows -> numbers.json
FS = 1.15                              # diagram text scale: legible at slide width


def jl(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("->", OUT / name)


def canvas(W, H, ylo=0.0):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(ylo, ylo + H); ax.set_axis_off()
    return fig, ax


def T(ax, x, y, s, fs=14, c=INK, b=False, ha="center", va="center", z=7, **kw):
    ax.text(x, y, s, fontsize=fs * FS, color=c, fontweight="bold" if b else "normal", ha=ha, va=va, zorder=z,
            linespacing=1.15, **kw)


def rbox(ax, x, y, w, h, text="", fc=TOKC, ec=MUTED, lw=1.5, fs=14, b=False, tc=INK, ls="-", z=3, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))
    if text:
        T(ax, x + w / 2, y + h / 2, text, fs=fs, c=tc, b=b, z=z + 2)


def abox(ax, x, y, w, h, kind, text="", fs=15, bold=False, tc=None):
    arch_box(ax, x, y, w, h, kind, text, fs=fs * FS, bold=bold, tc=tc)


def tick(ax, x, y, s=0.15, col=GOOD, lw=3.2):
    ax.plot([x - s, x - 0.3 * s, x + s], [y + 0.05 * s, y - 0.65 * s, y + 0.85 * s], color=col, lw=lw,
            solid_capstyle="round", solid_joinstyle="round", zorder=9)


def xmark(ax, x, y, s=0.13, col=BAD, lw=3.2):
    for d in (1, -1):
        ax.plot([x - s, x + s], [y - d * s, y + d * s], color=col, lw=lw, solid_capstyle="round", zorder=9)


def verdict(ax, x, y, text, good, fs=15):
    """A tick or a cross, then a short verdict, centred as a unit on x."""
    w = 0.0075 * fs * FS * len(text)
    (tick if good else xmark)(ax, x - w / 2 - 0.25, y)
    T(ax, x - w / 2, y, text, fs=fs, c=GOOD if good else BAD, ha="left", b=True)


def thumb(fig, ax, x, y, s, mod, img=None, lw=2.2):
    a = inset(fig, ax, x, y, s)
    show(a, mod, truth(mod, SCENE) if img is None else img)
    frame(a, MODC[mod], lw)
    return a


def rgb_inset(fig, ax, x, y, s, rgb, col=None, lw=2.0, w=None):
    a = inset(fig, ax, x, y, s if w is None else w, s)
    a.imshow(rgb, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(col is not None); sp.set_color(col or SURF); sp.set_linewidth(lw)
    return a


def qmark(ax, x, y, r=0.24):
    ax.add_patch(Circle((x, y), r, fc=BAD, ec="none", zorder=9))
    T(ax, x, y - 0.01, "?", fs=19, c="white", b=True, z=10)


def s2_patch(row=SCENE, y0=130, x0=60, n=48):
    """A crop of the example scene's S2 image, as an illustration of 'a patch'."""
    img = truth("S2L2A", row)
    lo = np.nanpercentile(img, 2, axis=(0, 1)); hi = np.nanpercentile(img, 98, axis=(0, 1))
    return np.clip((img[y0:y0 + n, x0:x0 + n] - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


# ------------------------------------------------------------------------------- 0. the problem

def fig_today():
    W, H = 14.4, 5.55
    fig, ax = canvas(W, H)
    mods = ["S2L2A", "S1RTC", "DEM", "LULC"]
    s, g = 0.74, 0.18
    ys = [0.3 + (3 - i) * (s + g) for i in range(4)]
    dx, dy, dw, dh = 5.1, 1.0, 4.2, 2.3
    abox(ax, dx, dy, dw, dh, "pre", "decoder", fs=21, bold=True)
    for i, (m, y) in enumerate(zip(mods, ys)):
        yc = y + s / 2
        thumb(fig, ax, 0.2, y, s, m)
        arrow(ax, (0.97, yc), (1.33, yc), lw=1.4, ms=10)
        rbox(ax, 1.35, y + 0.08, 2.05, s - 0.16, f"{SHORT[m]} tokenizer", fc=tint(MODC[m], 0.86), ec=MODC[m], fs=14)
        yt = dy + dh * (0.8 - 0.2 * i)
        arrow(ax, (3.43, yc), (dx - 0.04, yt), lw=1.3, ms=10)
        arrow(ax, (dx + dw + 0.04, yt), (10.0, yc), lw=1.3, ms=10)
        rbox(ax, 10.02, y + 0.08, 2.35, s - 0.16, f"{SHORT[m]} de-tokenizer", fc=tint(MODC[m], 0.86), ec=MODC[m],
             fs=14)
        arrow(ax, (12.4, yc), (12.76, yc), lw=1.4, ms=10)
        thumb(fig, ax, 12.8, y, s, m)
    T(ax, dx + 0.05, 0.55, "in: ids,\n13.9 bits per patch", fs=13, c=INK2, ha="left")
    T(ax, dx + dw - 0.05, 0.55, "out: softmax\nover 87,556 ids", fs=13, c=INK2, ha="right")
    top = ys[0] + s
    for x0, w, lab in ((1.2, 2.35, "a tokenizer per modality,\ntrained separately"),
                       (9.87, 2.65, "a de-tokenizer per modality\n(diffusion, 50 steps)")):
        rbox(ax, x0, ys[3] - 0.1, w, top - ys[3] + 0.2, fc="none", ec=BAD, lw=2.0, ls=(0, (4, 3)), z=2, r=0.12)
        T(ax, x0 + w / 2, top + 0.78, lab, fs=14.5, b=True)
        qmark(ax, x0 + w, top + 0.1)
    T(ax, 0.59, top + 0.3, "in", fs=15, c=INK2)
    T(ax, 13.17, top + 0.3, "out", fs=15, c=INK2)
    save(fig, "today.png")


# --------------------------------------------------------------------------------- 1. literature

def fig_lit_inputs():
    W, H = 14.4, 5.9
    fig, ax = canvas(W, H, ylo=-0.75)
    cx = [2.4, 7.2, 12.0]
    for x in (4.8, 9.6):
        ax.plot([x, x], [-0.6, 4.95], color=GRID, lw=1.5, zorder=0)
    heads = [("a tokenizer", "Chameleon, 4M, our model"), ("a pretrained encoder", "LLaVA and most VLMs"),
             ("no encoder", "Fuyu, EVE, EarthPT")]
    for c, (t, sub) in zip(cx, heads):
        T(ax, c, 4.78, t, fs=19, b=True)
        T(ax, c, 4.38, sub, fs=14.5, c=INK2)
    dy, dh, dw = 3.2, 0.72, 3.4
    for c in cx:
        abox(ax, c - dw / 2, dy, dw, dh, "pre", "decoder", fs=17, bold=True)
    s, py = 0.7, -0.05
    patch = s2_patch()
    for c in cx:
        rgb_inset(fig, ax, c - s / 2, py, s, patch, col=MODC["S2L2A"], lw=2.2)
    up = lambda c, y0, y1: arrow(ax, (c, y0 + 0.02), (c, y1 - 0.02), lw=1.5, ms=11)
    # A: tokenizer -> id -> table
    c = cx[0]
    up(c, py + s, 0.95)
    rbox(ax, c - 1.15, 0.95, 2.3, 0.5, "tokenizer (trained)", fs=14)
    up(c, 1.45, 1.75)
    sq(ax, c - 0.25, 1.75, 0.5, MODC["S2L2A"], solid=True, text="id", fs=13)
    up(c, 2.25, 2.5)
    rbox(ax, c - 1.15, 2.5, 2.3, 0.45, "embedding table", fc=tint(MODC["S2L2A"], 0.86), ec=MODC["S2L2A"], fs=14)
    up(c, 2.95, dy)
    # B: a big pretrained encoder -> projector
    c = cx[1]
    up(c, py + s, 0.85)
    rbox(ax, c - 1.4, 0.85, 2.8, 1.25, "image encoder\n(a pretrained ViT)", fc="#e7e6e0", ec=INK2, fs=15, b=True)
    up(c, 2.1, 2.45)
    rbox(ax, c - 0.8, 2.45, 1.6, 0.45, "projector", fs=14)
    up(c, 2.9, dy)
    # C: one linear layer
    c = cx[2]
    up(c, py + s, 1.9)
    abox(ax, c - 0.7, 1.9, 1.4, 0.48, "pre", "linear", fs=15, bold=True)
    up(c, 2.38, dy)
    verdict(ax, cx[0], -0.47, "lossy: 13.9 bits per patch", False)
    verdict(ax, cx[1], -0.47, "a second network", False)
    verdict(ax, cx[2], -0.47, "decoder-only, nothing pretrained", True)
    save(fig, "lit_inputs.png")


def _field(shift, n=16, rng=None):
    """A synthetic 16 x 16 patch: a field edge (green | tan) at column `shift`, along a diagonal."""
    g, t = np.array([0.25, 0.48, 0.23]), np.array([0.80, 0.69, 0.47])
    yy, xx = np.mgrid[0:n, 0:n]
    m = (xx + 0.35 * yy) < shift
    img = np.where(m[..., None], g, t)
    if rng is not None:
        img = img + rng.normal(0, 0.035, img.shape)
    return np.clip(img, 0, 1)


def fig_lit_outputs():
    W, H = 14.4, 5.6
    fig, ax = canvas(W, H, ylo=-0.2)
    abox(ax, 0.3, 1.55, 3.1, 1.6, "pre", "decoder", fs=19, bold=True)
    hx, hy = 3.4, 2.35
    ax.add_patch(Circle((hx + 0.22, hy), 0.2, fc="white", ec=PRE_EC, lw=1.8, zorder=6))
    T(ax, hx + 0.22, hy, "h", fs=14, b=True)
    rows = [(4.25, "softmax over ids", "today: sample an id, then decode it"),
            (2.35, "regression (MSE)", ""),
            (0.45, "sampling head", "small diffusion or flow network")]
    for y, t, sub in rows:
        arrow(ax, (hx + 0.44, hy), (4.6, y), lw=1.5, ms=12)
        rbox(ax, 4.62, y - 0.36, 2.9, 0.72, t, fc="white", ec=INK2, fs=15.5, b=True)
    rng = np.random.default_rng(0)
    A, B = _field(7.5, rng=rng), _field(12.5, rng=rng)
    s = 0.95
    # row 1: an id, decoded
    T(ax, 10.9, 4.25, "sample an id → the de-tokenizer draws the image", fs=14.5, c=INK2)
    # row 2: two equally likely patches average into a blur
    y = 2.35 - s / 2
    rgb_inset(fig, ax, 8.05, y, s, A, col=MUTED, lw=1.2); T(ax, 8.05 + s / 2, y - 0.2, "likely", fs=12.5, c=INK2)
    T(ax, 9.3, 2.35, "or", fs=15, c=INK2)
    rgb_inset(fig, ax, 9.62, y, s, B, col=MUTED, lw=1.2); T(ax, 9.62 + s / 2, y - 0.2, "equally likely", fs=12.5, c=INK2)
    arrow(ax, (10.72, 2.35), (11.32, 2.35), lw=1.8, ms=14)
    rgb_inset(fig, ax, 11.4, y, s, (A + B) / 2, col=BAD, lw=2.2)
    T(ax, 12.6, 2.35, "predicts the\naverage: blurry", fs=14.5, c=BAD, b=True, ha="left")
    # row 3: noise -> denoising steps -> one sharp draw
    y = 0.45 - s / 2
    rgb_inset(fig, ax, 8.05, y, s, np.clip(rng.normal(0.5, 0.25, (16, 16, 3)), 0, 1), col=MUTED, lw=1.2)
    T(ax, 8.05 + s / 2, y - 0.2, "noise", fs=12.5, c=INK2)
    T(ax, 9.65, 0.45, "· · ·", fs=20, c=INK2)
    arrow(ax, (10.1, 0.45), (11.32, 0.45), lw=1.8, ms=14)
    T(ax, 10.7, 0.72, "steps", fs=12.5, c=INK2)
    rgb_inset(fig, ax, 11.4, y, s, B, col=GOOD, lw=2.2)
    T(ax, 12.6, 0.45, "draws one\nlikely patch: sharp", fs=14.5, c=GOOD, b=True, ha="left")
    T(ax, 6.07, 0.45 - 0.62, "MAR · GIVT · NextStep-1 · Sundial", fs=12.5, c=INK2)
    T(ax, 6.07, 2.35 - 0.62, "AIM · EarthPT", fs=12.5, c=INK2)
    save(fig, "lit_outputs.png")


def fig_lit_compression():
    W, H = 14.4, 5.6
    fig, ax = canvas(W, H, ylo=-0.3)
    # a 12-band patch: 12 offset squares
    s, x0, y0 = 1.25, 0.35, 2.0
    for k in range(12):
        col = tint(MODC["S2L2A"], 0.25 + 0.05 * k)
        ax.add_patch(FancyBboxPatch((x0 + 0.06 * k, y0 + 0.06 * k), s, s, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=col, ec=MODC["S2L2A"], lw=1.0, zorder=3 + k))
    T(ax, x0 + 0.95, y0 - 0.38, "one 16 × 16 S2 patch:\n3,072 numbers", fs=14)
    # the funnel
    fx = 2.9
    ax.add_patch(Polygon([(fx, 1.55), (fx, 4.05), (fx + 2.1, 3.15), (fx + 2.1, 2.45)], closed=True, fc=tint(INK2, 0.88),
                         ec=INK2, lw=1.4, zorder=2))
    T(ax, fx + 0.95, 2.8, "compress", fs=15, b=True)
    arrow(ax, (x0 + s + 0.8, 2.8), (fx - 0.04, 2.8), lw=1.6, ms=12)
    # the short vector
    vx = fx + 2.55
    for k in range(6):
        ax.add_patch(Rectangle((vx + 0.22 * k, 2.68), 0.2, 0.24, fc=tint(MODC["S2L2A"], 0.4), ec=MODC["S2L2A"], lw=1,
                               zorder=4))
    T(ax, vx + 0.66, 2.3, "16–200\nnumbers", fs=14, b=True, va="top")
    arrow(ax, (vx + 1.4, 2.8), (vx + 1.85, 2.8), lw=1.6, ms=12)
    rbox(ax, vx + 1.88, 2.42, 1.15, 0.76, "head", fc="white", ec=INK2, fs=15, b=True)
    # three ways to compress
    ways = [("a fixed transform\n(DCT)", "FAST · DCTdiff"), ("a learned latent\n(VAE)", "MAR · MODUS · EO-VAE"),
            ("inside the model", "JiT · PRA")]
    for k, (t, ex) in enumerate(ways):
        x = 0.35 + 2.85 * k
        rbox(ax, x, -0.2, 2.7, 1.25, "", fc="white", ec=GRID, lw=1.4)
        T(ax, x + 1.35, 0.6, t, fs=13.5, b=True)
        T(ax, x + 1.35, 0.03, ex, fs=12, c=INK2)
    T(ax, 4.6, 1.33, "three ways to get there", fs=14, c=INK2)
    # right: PRA, Tab. 1: FID by token size, diffusion vs autoregressive
    a = fig.add_axes([0.665, 0.13, 0.32, 0.66])
    fid = {"diffusion": (3.55, 4.56), "autoregressive": (4.06, 7.68)}
    xs = np.arange(2)
    for j, (k, v) in enumerate(fid.items()):
        c = "#8a8984" if k == "diffusion" else INK2
        a.bar(xs + (j - 0.5) * 0.36, v, 0.32, color=c, zorder=3, edgecolor=SURF, lw=2, label=k)
        for xi, vi in zip(xs + (j - 0.5) * 0.36, v):
            a.text(xi, vi + 0.15, f"{vi:.2f}", ha="center", va="bottom", fontsize=13, color=INK)
    a.set_xticks(xs, ["48 numbers\nper token", "768 numbers\nper token"], fontsize=13.5)
    a.set_ylim(0, 9.5); a.set_ylabel("FID (lower is better)", fontsize=13.5)
    ygrid(a)
    a.set_title("PRA (2026): generating images from raw patches", fontsize=14.5, loc="left", pad=10)
    a.legend(loc="upper left", fontsize=13, handlelength=1.2)
    save(fig, "lit_compression.png")


def fig_lit_eo():
    W, H = 14.4, 5.4
    fig, ax = canvas(W, H)
    cx0, cy0, cw, ch = 4.85, 1.55, 4.7, 2.3
    rbox(ax, cx0, cy0, cw, ch, "", fc="white", ec=BAD, lw=2.2, ls=(0, (5, 3)), r=0.15)
    T(ax, cx0 + cw / 2, cy0 + ch / 2 + 0.38, "one decoder-only model,\ncontinuous in and out,\nevery modality", fs=17,
      b=True)
    T(ax, cx0 + cw / 2, cy0 + 0.38, "not built yet", fs=16, c=BAD, b=True)
    cards = [
        (0.1, 3.35, "Any-band embedders", "DOFA · Copernicus-FM · Panopticon", "a patch embedding from any band set,\nconditioned on wavelength"),
        (9.75, 3.35, "One VAE for S2 and S1", "EO-VAE (2026)", "continuous latents for any S2 or S1\nbands, trained on TerraMesh"),
        (0.1, 0.05, "Continuous any-to-any", "COP-GEN (2026)", "latent diffusion over S2, S1, DEM,\nland cover, location and time"),
        (9.75, 0.05, "GPTs on continuous pixels", "EarthPT · SITS-DECO", "pixel time series; SITS-DECO's\ncontinuous outputs failed (465k)"),
    ]
    for x, y, t, who, what in cards:
        rbox(ax, x, y, 4.55, 1.95, "", fc="white", ec=GRID, lw=1.5, r=0.12)
        T(ax, x + 0.22, y + 1.55, t, fs=15, b=True, ha="left")
        T(ax, x + 0.22, y + 1.13, who, fs=12, c=PRE_EC, b=True, ha="left")
        T(ax, x + 0.22, y + 0.5, what, fs=12.5, c=INK2, ha="left")
    for (x0, y0), (x1, y1) in (((4.65, 4.3), (cx0, cy0 + ch - 0.3)), ((9.75, 4.3), (cx0 + cw, cy0 + ch - 0.3)),
                               ((4.65, 1.0), (cx0, cy0 + 0.3)), ((9.75, 1.0), (cx0 + cw, cy0 + 0.3))):
        ax.plot([x0, x1], [y0, y1], color=MUTED, lw=1.4, ls=(0, (2, 2)), zorder=1)
    save(fig, "lit_eo.png")


# ------------------------------------------------------------------------------------ 2. the pilot

def report():
    return jl(CIO / "analysis/report.json")


def pair_label(p):
    s, t = p.split(">")
    return f"{SHORT[s]} → {SHORT[t]}"


def fig_token_info():
    d = jl(CIO / "analysis/pca_dct.json")["S2L2A"]
    pk = sorted((int(k), v) for k, v in d["pca_mse"].items() if int(k) < d["dim"] // 2)
    dk = sorted((int(k), v) for k, v in d["dct_mse"].items() if int(k) < d["dim"])
    tok, k_pca, k_dct = d["fsq_roundtrip_mse"], d["k_pca_matching_fsq"], d["k_dct_matching_fsq"]
    d192 = dict(dk)[192]
    NUM["token_info"] = {"fsq_roundtrip_mse": tok, "k_pca": k_pca, "k_dct": k_dct, "dct192_mse": d192,
                         "ratio_token_over_dct192": tok / d192, "bits": d["fsq_bits_per_patch"]}
    fig, a = plt.subplots(figsize=(11.8, 4.7))
    a.plot(*zip(*dk), "-s", color=COL["flowdct_out"], ms=7.5, lw=2.6, zorder=4, label="DCT: fixed, the same for every band")
    a.plot(*zip(*pk), "-o", color="#a8a7a1", ms=7, lw=2.4, zorder=3, label="PCA: fitted to the data")
    a.axhline(tok, color=COL["tok_in"], lw=2.2, ls=(0, (5, 3)), zorder=2,
              label=f"one FSQ token ({d['fsq_bits_per_patch']:.1f} bits)")
    a.legend(loc="lower left", fontsize=14, handlelength=2.2)
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1,024"])
    a.set_xlim(0.8, 1700); a.set_ylim(4e-5, 0.6)
    a.set_xlabel("numbers kept per 16 × 16 patch of S2 (3,072 values)")
    a.set_ylabel("reconstruction error (z²)")
    ygrid(a)
    for k, lab, yy, va in ((k_pca, f"{k_pca} PCA numbers", tok * 0.42, "top"), (k_dct, f"{k_dct} DCT numbers", tok * 2.3, "bottom")):
        a.plot([k], [tok], "o", ms=11, mfc="none", mec=INK, mew=2, zorder=6)
        a.annotate(lab, (k, tok), (k, yy), ha="center", va=va, fontsize=14, fontweight="bold",
                   arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2))
    a.plot([192], [d192], "*", ms=17, color=COL["flowdct_out"], mec=INK, mew=0.8, zorder=7)
    a.annotate(f"16 per band:\n{tok / d192:.1f}× less error", (192, d192), (300, 0.033), fontsize=14,
               fontweight="bold", ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2))
    save(fig, "token_info.png")


def _mask_sq(ax, x, y, s, fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.05", fc="#ecebe6", ec=MUTED,
                                lw=1.3, hatch="////", zorder=4))
    T(ax, x + s / 2, y + s / 2, "M", fs=fs, b=True, c=INK2, z=6)


def fig_pilot_seq():
    W, H = 14.4, 5.0
    fig, ax = canvas(W, H, ylo=-0.45)
    s, p = 0.44, 0.56
    S1, S2 = MODC["S1RTC"], MODC["S2L2A"]
    seq = [("S1RTC", 1), ("S1RTC", 0), ("S1RTC", 0), ("S1RTC", 0), ("gap", 0), ("S1RTC", 0), ("S1RTC", 1),
           ("S2L2A", 1), ("S2L2A", 0), ("M", 0), ("S2L2A", 0), ("M", 0), ("M", 0), ("S2L2A", 0)]
    x0, yb, yt, by, bh = 0.45, 0.35, 2.95, 1.45, 1.0
    xs = [x0 + i * p for i in range(len(seq))]
    abox(ax, x0 - 0.2, by, xs[-1] + s - x0 + 0.4, bh, "pre", "decoder · 12.6 M parameters", fs=17, bold=True)
    for i, (m, mk) in enumerate(seq):
        x = xs[i]
        if m == "gap":
            T(ax, x + s / 2, yb + s / 2, "…", fs=17, c=INK2)
            continue
        if m == "M":
            _mask_sq(ax, x, yb, s)
        else:
            sq(ax, x, yb, s, MODC[m], solid=bool(mk))
        arrow(ax, (x + s / 2, yb + s + 0.02), (x + s / 2, by - 0.02), lw=1.1, ms=8)
        if i >= 7:                                 # the target's positions predict its next patch
            sq(ax, x, yt, s, S2, solid=False, lw=1.8)
            arrow(ax, (x + s / 2, by + bh + 0.02), (x + s / 2, yt - 0.02), lw=1.1, ms=8)
    T(ax, (xs[0] + xs[6] + s) / 2, yb - 0.22, "source: S1", fs=15, b=True, c=S1)
    T(ax, (xs[7] + xs[-1] + s) / 2, yb - 0.22, "target: S2, some hidden", fs=15, b=True, c=S2)
    T(ax, (xs[7] + xs[-1] + s) / 2, yt + s + 0.28, "predict every target patch", fs=15, b=True)
    # the two evaluation modes
    ex = 9.6
    ax.plot([ex - 0.35, ex - 0.35], [-0.35, 4.45], color=GRID, lw=1.5, zorder=0)
    T(ax, ex, 4.25, "Scored two ways", fs=17, b=True, ha="left")
    for k, (t1, t2, masked) in enumerate((("source only", "every target input hidden:\none pass, like a downstream map", True),
                                          ("autoregressive", "no target input hidden:\ngeneration, patch by patch", False))):
        y = 2.75 - 2.25 * k
        T(ax, ex, y + 0.7, t1, fs=15.5, b=True, ha="left")
        for j in range(6):
            x = ex + j * 0.5
            if j == 0:
                sq(ax, x, y, 0.38, S2, solid=True)
            elif masked:
                _mask_sq(ax, x, y, 0.38, fs=10)
            else:
                sq(ax, x, y, 0.38, S2)
        T(ax, ex, y - 0.42, t2, fs=13.5, c=INK2, ha="left", va="top")
    save(fig, "pilot_seq.png")


def _option(ax, x, y, w, h, text, cols, sub=None, fs=14.5):
    rbox(ax, x, y, w, h, "", fc="white", ec=GRID, lw=1.4)
    n = len(cols)
    for k, c in enumerate(cols):
        ax.add_patch(Rectangle((x + 0.08, y + 0.1 + k * (h - 0.2) / n), 0.16, (h - 0.2) / n, fc=c, ec="none", zorder=5))
    T(ax, x + 0.42, y + h / 2 + (0.11 if sub else 0), text, fs=fs, ha="left")
    if sub:
        T(ax, x + 0.42, y + h / 2 - 0.2, sub, fs=12.5, c=INK2, ha="left")


def fig_pilot_variants():
    W, H = 14.4, 5.0
    fig, ax = canvas(W, H)
    dx, dy, dw, dh = 5.55, 1.45, 3.3, 1.8
    abox(ax, dx, dy, dw, dh, "pre", "", fs=18)
    T(ax, dx + dw / 2, dy + dh / 2 + 0.2, "decoder", fs=19, b=True)
    T(ax, dx + dw / 2, dy + dh / 2 - 0.28, "the same in every run", fs=13.5, c=INK2)
    T(ax, 2.45, 4.65, "input layer: how the source enters", fs=16, b=True)
    T(ax, 11.95, 4.65, "output head: how the target leaves", fs=16, b=True)
    ins = [("FSQ id → embedding table", [COL["tok_in"]], "today"),
           ("pixels → linear", [COL["lin_in"]], "with and without LayerNorm"),
           ("level + relief → linear", [COL["linrel_in"]], "band mean + detail at unit scale"),
           ("any bands → one embedder", [COL["band_in"], COL["bandrop_in"]], "with and without band dropout")]
    outs = [("softmax over FSQ ids", [COL["tok_in"]], "today"),
            ("regression", [COL["reg_out"]], "the mean, MSE"),
            ("flow on raw pixels", [COL["flow_out"]], "a sampler, 3,072 numbers per S2 patch"),
            ("flow on 16 DCT per band", [COL["flowdct_out"]], "a sampler, 192 numbers per S2 patch")]
    h, gap = 0.78, 0.2
    for k, (t, cols, sub) in enumerate(ins):
        y = 3.55 - k * (h + gap)
        _option(ax, 0.2, y, 4.55, h, t, cols, sub)
        arrow(ax, (4.79, y + h / 2), (dx - 0.04, dy + dh * (0.8 - 0.2 * k)), lw=1.3, ms=10)
    for k, (t, cols, sub) in enumerate(outs):
        y = 3.55 - k * (h + gap)
        _option(ax, 9.65, y, 4.55, h, t, cols, sub)
        arrow(ax, (dx + dw + 0.04, dy + dh * (0.8 - 0.2 * k)), (9.61, y + h / 2), lw=1.3, ms=10)
    T(ax, 11.95, -0.05, "land cover: per-pixel classes in these runs", fs=13, c=INK2)
    save(fig, "pilot_variants.png")


def fig_inputs_delta():
    r = report()["contrasts"]
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 5.4), sharey=True, gridspec_kw=dict(wspace=0.06))
    ys = np.arange(len(PAIRS))[::-1].astype(float)
    NUM["inputs_delta"] = {}
    for ax, mode, ttl in ((axs[0], "mask1", "from the source alone"), (axs[1], "tf", "autoregressive")):
        ax.axhspan(ys[8] - 0.5, ys[6] + 0.5, color="#f1f0eb", zorder=0)
        for y in (2.5, 5.5, 8.5):
            ax.axhline(y, color=GRID, lw=1, zorder=1)
        ax.axvline(0, color=COL["tok_in"], lw=2, ls=(0, (5, 3)), zorder=2)
        for j, run in enumerate(("lin_in", "linrel_in")):
            m = np.array([r[f"{run}_s0-tok_in_s0:{mode}:{p}"] for p in PAIRS])
            off = (0.5 - j) * 0.28
            ax.errorbar(m[:, 0], ys + off, xerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt="o" if j == 0 else "D",
                        ms=8.5 if j == 0 else 8, color=COL[run], ecolor=COL[run], elinewidth=2, capsize=0, zorder=4,
                        label=NAME[run], mec=SURF, mew=1.2)
            NUM["inputs_delta"][f"{run}:{mode}"] = {p: [round(v, 4) for v in r[f"{run}_s0-tok_in_s0:{mode}:{p}"]]
                                                    for p in PAIRS}
            if run == "linrel_in":
                n = int((m[:, 0] < 0).sum())
                NUM["inputs_delta"][f"linrel_better:{mode}"] = n
        ax.set_title(f"{ttl}\nlevel + relief better on {n} of 12", fontsize=15.5, loc="center", pad=8)
        ax.set_xlabel("cross-entropy minus tokens' (nats/token)")
        xgrid(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    lo = min(a.get_xlim()[0] for a in axs); hi = max(a.get_xlim()[1] for a in axs)
    for a in axs:
        a.set_xlim(lo, hi)
    axs[0].set_yticks(ys, [pair_label(p) for p in PAIRS], fontsize=15)
    axs[0].set_ylim(-0.7, 11.7)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), fontsize=14.5, handletextpad=0.4,
               columnspacing=2)
    for a in axs:
        a.text(0.0, -0.13, "← better than tokens", transform=a.get_xaxis_transform(), fontsize=12.5, color=INK2,
               ha="right", va="top")
    save(fig, "inputs_delta.png")


def source_gain(ev, run, src):
    """Cross-entropy with another row's source minus with the true source, mean over src's 3 targets."""
    ps = [p for p in PAIRS if p.startswith(src + ">")]
    return float(np.mean([ev[f"{run}_s0:mask1_shufsrc"][p]["ce"] - ev[f"{run}_s0:mask1"][p]["ce"] for p in ps]))


def fig_level_relief():
    import torch
    from eo.experiments.cio_train import patchify
    ev = report()["eval"]
    runs = ["tok_in", "lin_in", "linraw_in", "linrel_in"]
    gains = {r: source_gain(ev, r, "DEM") for r in runs}
    tr = np.load(CIO / "data/train_DEM.npy", mmap_mode="r")[:256].astype(np.float32)
    x = patchify(torch.from_numpy(tr)).reshape(-1, 196, 1, 256)
    scale = float((x - x.mean(-1, keepdim=True)).std())
    te = np.load(CIO / "data/test_DEM.npy", mmap_mode="r")
    pt = patchify(torch.from_numpy(te[:64].astype(np.float32))).reshape(-1, 256).numpy()
    sd, mu = pt.std(1), pt.mean(1)
    ok = np.where((mu > 0.8) & (mu < 1.6) & (sd > 1.5 * scale) & (sd < 3 * scale))[0]
    patch = pt[ok[0]].reshape(16, 16)
    prof = patch[8]
    NUM["level_relief"] = {"dem_source_gain": gains, "relief_scale_z": scale, "patch_level_z": float(patch.mean()),
                           "patch_relief_sd_z": float(patch.std())}
    fig = plt.figure(figsize=(14.2, 5.0))
    a1 = fig.add_axes([0.05, 0.58, 0.27, 0.32]); a2 = fig.add_axes([0.05, 0.1, 0.27, 0.32])
    a3 = fig.add_axes([0.60, 0.12, 0.38, 0.72])
    px = np.arange(16)
    a1.plot(px, prof, color=MODC["DEM"], lw=2.6); a1.set_ylim(0, 1.6)
    a1.set_title("as standardized values", fontsize=15, loc="left")
    a1.text(15.5, prof.mean() - 0.25, f"level {prof.mean():.2f} z;\nthe relief is a flat line", fontsize=14.5, color=INK2,
            ha="right", va="top")
    rel = (prof - prof.mean()) / scale
    a2.axhline(0, color=GRID, lw=1.2)
    a2.plot(px, rel, color=MODC["DEM"], lw=2.6)
    lim = max(2.5, float(np.abs(rel).max()) * 1.15)
    a2.set_ylim(-lim, lim)
    a2.set_title("relief at unit scale (level kept apart)", fontsize=15, loc="left")
    for a in (a1, a2):
        a.set_xticks([]); a.set_xlabel("16 pixels across one patch", fontsize=13); a.set_yticks(a.get_yticks()[::2])
    # the two paths into the linear layer
    ax = fig.add_axes([0.33, 0.0, 0.25, 1.0]); ax.set_xlim(0, 3.5); ax.set_ylim(0, 5); ax.set_axis_off()
    for y, txt, ok_ in ((3.7, "the linear layer\nsees only the level", False), (1.3, "the linear layer\nsees the shape", True)):
        arrow(ax, (0.1, y), (0.75, y), lw=1.8, ms=14)
        T(ax, 0.85, y, txt, fs=14.5, ha="left", c=GOOD if ok_ else BAD, b=True)
    # right: what the true DEM source is worth to each input layer
    lab = {"tok_in": "FSQ tokens\n(today)", "lin_in": "pixels →\nlinear + LN", "linraw_in": "pixels →\nlinear",
           "linrel_in": "level + relief\n→ linear"}
    xs = np.arange(4)
    for i, r in enumerate(runs):
        a3.bar(i, gains[r], 0.62, color=COL[r], hatch="//" if r == "linraw_in" else None, edgecolor=SURF if r != "linraw_in" else "white",
               lw=0 if r != "linraw_in" else 0.0, zorder=3)
        a3.text(i, gains[r] + 0.015, f"{gains[r]:.2f}", ha="center", va="bottom", fontsize=15, fontweight="bold")
    a3.set_xticks(xs, [lab[r] for r in runs], fontsize=14)
    a3.set_ylabel("nats/token")
    a3.set_ylim(0, max(gains.values()) * 1.2)
    a3.set_title("What the DEM source is worth", fontsize=16, loc="left", pad=10)
    ygrid(a3)
    save(fig, "level_relief.png")


SUBSETS = [("_bands1-2-3-4-5-6-7-8-10-11", "PASTIS's 10"), ("_bands1-2-3-7", "RGB + NIR"),
           ("_bands1-2-3", "RGB"), ("_bands7", "B08 alone")]


def fig_any_band():
    ev = report()["eval"]
    share = {}
    for run in ("band_in", "bandrop_in"):
        allb = ev[f"{run}_s0:mask1"]["S2L2A>LULC"]["ce"]
        ctrl = ev[f"{run}_s0:mask1_shufsrc"]["S2L2A>LULC"]["ce"]
        share[run] = [100 * (ctrl - ev[f"{run}_s0:mask1{k}"]["S2L2A>LULC"]["ce"]) / (ctrl - allb) for k, _ in SUBSETS]
    NUM["any_band_share_pct"] = {r: dict(zip([n for _, n in SUBSETS], v)) for r, v in share.items()}
    fig = plt.figure(figsize=(14.4, 5.3))
    ax = fig.add_axes([0, 0, 0.5, 1]); ax.set_xlim(0, 7.2); ax.set_ylim(0, 5.3); ax.set_axis_off()
    img = np.load(CIO / "data/test_S2L2A.npy", mmap_mode="r")[1, :, 40:88, 120:168].astype(np.float32)
    bands = [(1, "B02"), (2, "B03"), (3, "B04"), (7, "B08"), (10, "B11"), (11, "B12")]
    s, x0 = 0.72, 0.35
    for k, (b, n) in enumerate(bands):
        x = x0 + k * 1.0
        a = inset(fig, ax, x, 3.75, s)
        z = img[b]
        a.imshow(z, cmap="gray", vmin=np.percentile(z, 2), vmax=np.percentile(z, 98)); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(MODC["S2L2A"]); sp.set_linewidth(1.6)
        T(ax, x + s / 2, 4.75, n, fs=13.5, b=True)
        arrow(ax, (x + s / 2, 3.72), (x + s / 2, 3.25), lw=1.2, ms=9)
    T(ax, x0 + 6 * 1.0 + 0.05, 4.1, "…", fs=18, c=INK2, ha="left")
    rbox(ax, 0.2, 2.55, 6.6, 0.68, "one shared linear layer + the band's identity", fc=tint(PRE_EC, 0.88),
         ec=PRE_EC, fs=14)
    for k in range(6):
        arrow(ax, (x0 + k * 1.0 + s / 2, 2.53), (3.5, 1.87), lw=1.0, ms=8)
    rbox(ax, 1.7, 1.25, 3.6, 0.6, "attention pooling over bands", fc=tint(PRE_EC, 0.88), ec=PRE_EC, fs=14)
    arrow(ax, (3.5, 1.23), (3.5, 0.82), lw=1.4, ms=10)
    sq(ax, 3.28, 0.36, 0.44, MODC["S2L2A"], solid=True)
    T(ax, 3.88, 0.58, "one embedding\nper patch", fs=13.5, ha="left", c=INK2)
    T(ax, 3.6, 5.12, "12 bands, PASTIS's 10, RGB or one band: the same weights", fs=14.5, b=True)
    a3 = fig.add_axes([0.56, 0.14, 0.42, 0.68])
    xs = np.arange(len(SUBSETS))
    for j, run in enumerate(("band_in", "bandrop_in")):
        xx = xs + (j - 0.5) * 0.38
        a3.bar(xx, share[run], 0.34, color=COL[run], zorder=3, edgecolor=SURF, lw=1.5, label=NAME[run])
    for i in range(len(SUBSETS)):
        a, b = share["band_in"][i], share["bandrop_in"][i]
        pts = [(xs[i], max(a, b), a)] if abs(a - b) < 4 else [(xs[i] - 0.19, a, a), (xs[i] + 0.19, b, b)]
        for xi, top, v in pts:
            a3.text(xi, top + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=15, fontweight="bold")
    a3.set_xticks(xs, [n for _, n in SUBSETS], fontsize=15)
    a3.set_ylim(0, 118); a3.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    a3.set_title("S2 → land cover: information kept, vs all 12 bands", fontsize=15.5, loc="left", pad=34)
    a3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=14.5, handlelength=1.2,
              columnspacing=1.5)
    ygrid(a3)
    save(fig, "any_band.png")


def _roll(run, s, t, mode, par, decoded=False):
    f = CIO / "runs" / f"{run}_s0" / "rollout" / f"{s}_{t}_k0_{mode}{'_par' if par else ''}{'_decoded' if decoded else ''}.npy"
    return np.load(f, mmap_mode="r")


def fig_outputs_examples(rows=(0, 1)):
    from eo.experiments.cio_analysis import _rgb, to_image
    s, t = "LULC", "S2L2A"
    src = np.load(CIO / f"data/test_{s}.npy", mmap_mode="r")
    gt = np.load(CIO / f"data/test_{t}.npy", mmap_mode="r")
    cols = [("land cover\n(given)", None, None), ("truth", None, None),
            ("tokens", "tok_in", _roll("tok_in", s, t, "greedy", True, True)),
            ("regression", "reg_out", _roll("reg_out", s, t, "greedy", True)),
            ("flow on\nraw pixels", "flow_out", _roll("flow_out", s, t, "sample", True)),
            ("flow on\nDCT", "flowdct_out", _roll("flowdct_out", s, t, "sample", True)),
            ("tokens,\npatch by patch", "tok_in", _roll("tok_in", s, t, "sample", False, True))]
    n = len(cols)
    fig, axs = plt.subplots(len(rows), n + 1, figsize=(1.9 * n + 0.7, 1.9 * len(rows) + 0.9),
                            gridspec_kw=dict(wspace=0.06, hspace=0.06, width_ratios=[1] * (n - 1) + [0.2, 1]))
    for i in range(len(rows)):
        axs[i, n - 1].set_axis_off()
    axs = np.concatenate([axs[:, :n - 1], axs[:, n:]], axis=1)
    for i, r in enumerate(rows):
        for j, (name, run, arr) in enumerate(cols):
            a = axs[i, j]; a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if j == 0:
                a.imshow(src[r], cmap=LULC_CMAP, vmin=0, vmax=9, interpolation="nearest")
            elif j == 1:
                a.imshow(_rgb(gt[r].astype(np.float32)))
            else:
                a.imshow(_rgb(to_image(np.asarray(arr[r:r + 1]), t)[0]))
            if i == 0:
                a.set_title(name, fontsize=15, loc="center", fontweight="bold" if j >= 2 else "normal", pad=12)
                if run:
                    a.add_patch(Rectangle((0.0, 1.02), 1.0, 0.035, transform=a.transAxes, color=COL[run],
                                          clip_on=False))
    p0, p1 = axs[0, -2].get_position(), axs[0, -1].get_position()
    xm = (p0.x1 + p1.x0) / 2
    fig.add_artist(plt.Line2D([xm, xm], [0.04, 0.98], color=GRID, lw=1.6))
    fig.text((axs[0, 2].get_position().x0 + p0.x1) / 2, 1.0, "one pass, from the land-cover map", ha="center",
             va="bottom", fontsize=14.5, color=INK2)
    save(fig, "outputs_examples.png")


OUT_RUNS = ["tok_in", "reg_out", "flow_out", "flowdct_out"]
GEN_PAIRS = ["S1RTC>S2L2A", "LULC>S2L2A", "S2L2A>S1RTC", "S2L2A>DEM"]


def fig_outputs_scores():
    ro = report()["rollouts"]
    par_mode = {"tok_in": "par-greedy", "reg_out": "par-greedy", "flow_out": "par-sample", "flowdct_out": "par-sample"}
    fig, axs = plt.subplots(1, 2, figsize=(13.6, 4.9), sharey=True, gridspec_kw=dict(wspace=0.05))
    NUM["outputs_es"] = {}
    for ax, mode, ttl in ((axs[0], "par", "one pass from the source"), (axs[1], "ar", "patch by patch")):
        xs = np.arange(len(GEN_PAIRS))
        vals = np.array([[ro[f"{run}_s0:{p}:{par_mode[run] if mode == 'par' else 'ar-sample'}"]["energy_score"]
                          for run in OUT_RUNS] for p in GEN_PAIRS])
        NUM["outputs_es"][mode] = {p: dict(zip(OUT_RUNS, map(float, v))) for p, v in zip(GEN_PAIRS, vals)}
        for j, run in enumerate(OUT_RUNS):
            xx = xs + (j - 1.5) * 0.2
            ax.bar(xx, vals[:, j], 0.18, color=COL[run], zorder=3, edgecolor=SURF, lw=1.2, label=NAME[run])
        for i in range(len(GEN_PAIRS)):
            j = int(vals[i].argmin())
            ax.text(xs[i] + (j - 1.5) * 0.2, vals[i, j] + 0.02, f"{vals[i, j]:.2f}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold")
        ax.set_xticks(xs, [pair_label(p) for p in GEN_PAIRS], fontsize=14)
        ax.set_title(ttl, fontsize=16, loc="center", pad=8)
        ygrid(ax)
    axs[0].set_ylabel("energy score (lower is better)")
    axs[0].set_ylim(0, 1.2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=14, handlelength=1.2,
               columnspacing=1.8)
    save(fig, "outputs_scores.png")


def fig_lulc(rows=(0, 8)):
    """Rows 0 and 8: in one the per-pixel head looks better, in the other the FSQ codes do; both collapse
    patch by patch. The bars are the 128-row aggregate."""
    from eo.experiments.cio_analysis import _rgb
    ro = report()["rollouts"]
    s, t = "S2L2A", "LULC"
    src = np.load(CIO / f"data/test_{s}.npy", mmap_mode="r")
    gt = np.load(CIO / f"data/test_{t}.npy", mmap_mode="r")
    cols = [("S2 (given)", None, None), ("truth", None, None),
            ("FSQ codes", "tok_in", _roll("tok_in", s, t, "greedy", True, True)),
            ("per-pixel head", "flowdct_out", _roll("flowdct_out", s, t, "greedy", True)),
            ("FSQ codes,\npatch by patch", "tok_in", _roll("tok_in", s, t, "greedy", False, True))]
    fig = plt.figure(figsize=(14.4, 4.9))
    w, gap, hgt = 0.11, 0.006, 0.39
    for i, r in enumerate(rows):
        for j, (name, run, arr) in enumerate(cols):
            x = 0.005 + j * (w + gap) + (0.018 if j == 4 else 0)
            a = fig.add_axes([x, 0.04 + (len(rows) - 1 - i) * (hgt + 0.02), w, hgt])
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if j == 0:
                a.imshow(_rgb(src[r].astype(np.float32)))
            elif j == 1:
                a.imshow(gt[r], cmap=LULC_CMAP, vmin=0, vmax=9, interpolation="nearest")
            else:
                a.imshow(np.asarray(arr[r]), cmap=LULC_CMAP, vmin=0, vmax=9, interpolation="nearest")
            if i == 0:
                a.set_title(name, fontsize=14, loc="center", fontweight="bold" if j >= 2 else "normal", pad=10)
                if run:
                    a.add_patch(Rectangle((0.0, 1.02), 1.0, 0.03, transform=a.transAxes, color=COL[run], clip_on=False))
    xa = 0.005 + 2 * (w + gap)
    fig.text(xa + w + gap / 2, 0.955, "one pass", ha="center", va="bottom", fontsize=14, color=INK2)
    fig.add_artist(plt.Line2D([xa + 0.01, xa + 2 * w + gap - 0.01], [0.948, 0.948], color=MUTED, lw=1.2))
    order = ["tok_in", "lin_in", "flowdct_out", "flow_out", "reg_out"]
    lab = {"tok_in": "FSQ, tokens in", "lin_in": "FSQ, pixels in", "flowdct_out": "per-pixel (DCT flow)",
           "flow_out": "per-pixel (flow)", "reg_out": "per-pixel (regression)"}
    v = [ro[f"{r}_s0:S2L2A>LULC:par"]["miou"] for r in order]
    NUM["lulc_miou_one_pass"] = dict(zip(order, v))
    NUM["lulc_miou_ar"] = {r: ro[f"{r}_s0:S2L2A>LULC:ar"]["miou"] for r in order}
    a = fig.add_axes([0.79, 0.13, 0.195, 0.72])
    ys = np.array([4.3, 3.3, 2.0, 1.0, 0.0])
    a.barh(ys, v, 0.72, color=[COL[r] for r in order], zorder=3, edgecolor=SURF, lw=1.5)
    for yi, vi in zip(ys, v):
        a.text(vi + 0.008, yi, f"{vi:.2f}", ha="left", va="center", fontsize=14, fontweight="bold")
    a.set_yticks(ys, [lab[r] for r in order], fontsize=14)
    a.tick_params(axis="y", length=0)
    a.set_xlim(0, 0.52); a.set_xticks([0, 0.2, 0.4])
    a.set_xlabel("mIoU, one pass (128 scenes)")
    a.set_title("S2 → land cover", fontsize=16, loc="left", pad=10)
    xgrid(a)
    save(fig, "lulc.png")

# --------------------------------------------------------------------------------------- 3. ideas

def fig_ideas_map():
    W, H = 14.4, 4.8
    fig, ax = canvas(W, H, ylo=-0.55)
    steps = ["Pixels in,\ntokens out", "PASTIS: any bands in,\na per-pixel map out", "Quantities:\na regression head",
             "Generation:\na flow head on DCT", "Any sensor: bands\ndescribed by physics"]
    xs = [1.3 + 2.85 * k for k in range(5)]
    ys = [0.75 + 0.5 * k for k in range(5)]
    ax.add_patch(FancyArrowPatch_((xs[0] - 0.9, ys[0] - 0.55), (xs[-1] + 0.9, ys[-1] + 0.1)))
    for k, (x, y, t) in enumerate(zip(xs, ys, steps)):
        ax.add_patch(Circle((x, y), 0.36, fc=PRE_FC, ec=PRE_EC, lw=2, zorder=5))
        T(ax, x, y, str(k + 1), fs=20, b=True, z=6)
        T(ax, x, y + 0.95, t, fs=15, b=True)
    for x0, x1, lab in ((xs[0] - 0.8, xs[1] + 1.1, "1–2: within Phase 6 Part 2 (PASTIS)"),
                        (xs[2] - 1.1, xs[4] + 1.0, "3–5: the next base model")):
        ax.plot([x0, x0, x1, x1], [-0.08, -0.22, -0.22, -0.08], color=MUTED, lw=1.5)
        T(ax, (x0 + x1) / 2, -0.42, lab, fs=14.5, c=INK2)
    save(fig, "ideas_map.png")


def FancyArrowPatch_(p0, p1):
    from matplotlib.patches import FancyArrowPatch
    return FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=22, color=GRID, lw=6, zorder=1)


PRE_FC = "#dfe6ee"


def fig_idea1():
    W, H = 14.4, 4.6
    fig, ax = canvas(W, H, ylo=0.0)
    s = 1.0
    thumb(fig, ax, 0.2, 1.5, s, "S2L2A")
    T(ax, 0.7, 1.3, "S2 patch", fs=13.5, c=INK2, va="top")
    # today's path, removed
    abox(ax, 1.75, 3.15, 3.3, 0.7, "gone")
    crossed(ax, 1.85, 3.2, 3.1, 0.6, lw=2.5)
    T(ax, 3.4, 3.5, "FSQ tokenizer → id → table", fs=14, c=INK2, z=10,
      bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.92))
    T(ax, 3.4, 4.15, "today", fs=14, c=INK2)
    ax.add_patch(FancyArrowPatch_l((1.2, 2.3), (1.73, 3.45), GONE))
    # the new path
    arrow(ax, (1.24, 2.0), (1.73, 2.0), lw=1.6, ms=13)
    rbox(ax, 1.75, 1.62, 1.75, 0.76, "level +\nrelief", fs=14)
    T(ax, 2.62, 1.38, "fixed, per band", fs=12.5, c=INK2, va="top")
    arrow(ax, (3.52, 2.0), (3.83, 2.0), lw=1.6, ms=13)
    abox(ax, 3.85, 1.62, 1.2, 0.76, "new", "linear", fs=15)
    arrow(ax, (5.07, 2.0), (5.58, 2.0), lw=1.8, ms=14)
    abox(ax, 5.6, 1.0, 3.3, 2.0, "pre", "decoder", fs=20, bold=True)
    # outputs: unchanged
    arrow(ax, (8.92, 2.0), (9.33, 2.0), lw=1.6, ms=13)
    rbox(ax, 9.35, 1.62, 1.8, 0.76, "softmax over\nFSQ ids", fc=PRE_FC, ec=PRE_EC, fs=13.5)
    arrow(ax, (11.17, 2.0), (11.48, 2.0), lw=1.6, ms=13)
    rbox(ax, 11.5, 1.62, 1.55, 0.76, "de-\ntokenizer", fs=13.5)
    arrow(ax, (13.07, 2.0), (13.3, 2.0), lw=1.6, ms=13)
    thumb(fig, ax, 13.32, 1.55, 0.9, "S2L2A")
    ax.plot([9.35, 9.35, 14.22, 14.22], [3.05, 3.2, 3.2, 3.05], color=MUTED, lw=1.5)
    T(ax, 11.78, 3.45, "unchanged: the targets stay FSQ ids", fs=14.5, c=INK2)
    legend_row(ax, [("new", "new"), ("gone", "removed")], [4.6, 7.6], y=0.35)
    save(fig, "idea1.png")


def FancyArrowPatch_l(p0, p1, col):
    from matplotlib.patches import FancyArrowPatch
    return FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=col, lw=1.6, ls=(0, (4, 3)), zorder=2)


def _vec(ax, x, y, n=5, col=PRE_EC, s=0.2, p=0.24):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * p, y), s, s, fc=tint(col, 0.45), ec=col, lw=1, zorder=4))


def fig_idea1_retrofit():
    W, H = 14.4, 5.6
    fig, ax = canvas(W, H, ylo=-0.6)
    xdiv = 7.05
    ax.plot([xdiv, xdiv], [-0.5, 4.85], color=GRID, lw=1.5, zorder=0)
    for x, t, sub in ((3.5, "Stage 1 · align", "only the new embedder trains"),
                      (10.75, "Stage 2 · fine-tune", "everything, on the usual next-token loss")):
        T(ax, x, 4.6, t, fs=20, b=True)
        T(ax, x, 4.2, sub, fs=15, c=INK2)
    # stage 1
    thumb(fig, ax, 0.2, 1.4, 0.95, "S2L2A")
    arrow(ax, (1.17, 2.15), (1.6, 2.95), lw=1.5, ms=12)
    rbox(ax, 1.62, 2.75, 1.75, 0.6, "FSQ tokenizer", fs=13.5)
    icon(ax, "frozen", 3.37, 3.35, 0.8)
    arrow(ax, (3.39, 3.05), (3.68, 3.05), lw=1.4, ms=11)
    sq(ax, 3.7, 2.85, 0.4, MODC["S2L2A"], solid=True, text="id", fs=11)
    arrow(ax, (4.12, 3.05), (4.38, 3.05), lw=1.4, ms=11)
    abox(ax, 4.4, 2.75, 1.9, 0.6, "pre", "B10c's table", fs=13.5)
    icon(ax, "frozen", 6.3, 3.35, 0.8)
    arrow(ax, (1.17, 1.7), (1.6, 1.4), lw=1.6, ms=12)
    abox(ax, 1.62, 1.05, 2.6, 0.7, "new", "level + relief → linear", fs=13.5)
    icon(ax, "trained", 4.22, 1.75, 0.8)
    ax.add_patch(Circle((5.35, 1.4), 0.42, fc="white", ec=INK, lw=1.8, zorder=5))
    T(ax, 5.35, 1.4, "match", fs=13, b=True, z=6)
    arrow(ax, (4.25, 1.4), (4.91, 1.4), lw=1.6, ms=12)
    arrow(ax, (5.35, 2.73), (5.35, 1.84), lw=1.6, ms=12)
    T(ax, 3.5, 0.35, "the targets are free:\nevery patch already has its token", fs=14, c=INK2)
    # stage 2
    thumb(fig, ax, 7.45, 1.6, 0.95, "S2L2A")
    arrow(ax, (8.42, 2.07), (8.73, 2.07), lw=1.6, ms=12)
    abox(ax, 8.75, 1.72, 2.05, 0.7, "new", "embedder", fs=14)
    icon(ax, "trained", 10.8, 2.42, 0.8)
    arrow(ax, (10.82, 2.07), (11.13, 2.07), lw=1.6, ms=12)
    abox(ax, 11.15, 1.25, 2.4, 1.65, "pre", "decoder", fs=18, bold=True)
    icon(ax, "trained", 13.55, 2.9, 0.9)
    T(ax, 10.75, 0.35, "then the usual training, from B10c's weights", fs=14, c=INK2)
    legend_row(ax, [("trained", "trains"), ("frozen", "frozen"), ("new", "new")], [3.6, 6.4, 9.2], y=-0.35)
    save(fig, "idea1_retrofit.png")


def fig_idea2(pid=40305):
    W, H = 14.4, 5.4
    fig, ax = canvas(W, H, ylo=-0.55)
    cmap = ListedColormap(pastis_cols())
    rgb_inset(fig, ax, 0.15, 1.35, 1.35, pastis_rgb(pid), col=MODC["S2L2A"], lw=2.2)
    T(ax, 0.83, 1.15, "PASTIS S2:\n10 bands", fs=13.5, c=INK2, va="top")
    arrow(ax, (1.53, 2.02), (1.88, 2.02), lw=1.6, ms=13)
    abox(ax, 1.9, 1.5, 2.45, 1.05, "new", "one embedder,\nany bands", fs=14.5)
    T(ax, 3.12, 1.3, "no B01 / B09 to impute", fs=12.5, c=INK2, va="top")
    arrow(ax, (4.37, 2.02), (4.78, 2.02), lw=1.8, ms=14)
    abox(ax, 4.8, 1.1, 3.0, 1.85, "pre", "decoder", fs=20, bold=True)
    sem = pastis_sem(pid)
    # planned: class tokens on a 28 x 28 grid
    arrow(ax, (7.82, 2.5), (8.58, 3.3), lw=1.5, ms=12)
    rbox(ax, 8.6, 3.0, 2.9, 0.7, "class tokens", fc=tint(MODC["TASK"], 0.82), ec=MODC["TASK"], fs=14.5)
    T(ax, 10.05, 3.9, "planned (option 2)", fs=13.5, c=INK2)
    a = inset(fig, ax, 11.95, 2.75, 1.25); a.imshow(pastis_grid(sem), cmap=cmap, vmin=0, vmax=19, interpolation="nearest")
    a.set_axis_off()
    T(ax, 13.25 + 0.05, 3.38, "28 × 28", fs=13, c=INK2, ha="left")
    arrow(ax, (11.52, 3.35), (11.92, 3.35), lw=1.5, ms=12)
    # new: a per-pixel head, one pass
    arrow(ax, (7.82, 1.55), (8.58, 0.75), lw=1.5, ms=12)
    abox(ax, 8.6, 0.4, 2.9, 0.7, "new", "per-pixel head", fs=14.5)
    T(ax, 10.05, 0.2, "K logits per pixel, one pass", fs=12.5, c=INK2, va="top")
    a = inset(fig, ax, 11.95, 0.12, 1.25); a.imshow(sem, cmap=cmap, vmin=0, vmax=19, interpolation="nearest")
    a.set_axis_off()
    T(ax, 13.25 + 0.05, 0.75, "every\npixel", fs=13, c=INK2, ha="left")
    arrow(ax, (11.52, 0.75), (11.92, 0.75), lw=1.5, ms=12)
    legend_row(ax, [("new", "new")], [5.6], y=-0.32)
    save(fig, "idea2.png")


def fig_idea3():
    W, H = 14.4, 2.3
    fig, ax = canvas(W, H, ylo=0.6)
    abox(ax, 0.3, 0.9, 4.0, 1.7, "pre", "decoder", fs=20, bold=True)
    ax.add_patch(Circle((4.55, 1.75), 0.22, fc="white", ec=PRE_EC, lw=1.8, zorder=6))
    T(ax, 4.55, 1.75, "h", fs=14, b=True)
    arrow(ax, (4.79, 1.75), (5.38, 1.75), lw=1.8, ms=14)
    abox(ax, 5.4, 1.35, 2.6, 0.8, "new", "regression head", fs=15)
    T(ax, 6.7, 1.15, "MSE, or quantiles", fs=13, c=INK2, va="top")
    arrow(ax, (8.02, 1.75), (8.58, 1.75), lw=1.8, ms=14)
    out_biomass(fig, ax, 8.6, 1.0, 1.5)
    T(ax, 10.4, 2.15, "one value per pixel:", fs=15, b=True, ha="left")
    T(ax, 10.4, 1.62, "biomass, canopy height,\nNDVI as a product", fs=14.5, c=INK2, ha="left", va="top")
    save(fig, "idea3.png")


def fig_idea4():
    from eo.experiments.cio_analysis import dct_order
    W, H = 14.4, 5.4
    fig, ax = canvas(W, H, ylo=-0.5)
    # left: every target patch at once (all target inputs masked)
    s, p = 0.36, 0.46
    x0, yb, by, bh, yt = 0.3, 0.3, 1.25, 1.0, 2.75
    seq = ["S1RTC"] * 4 + ["BO"] + ["M"] * 5
    xs = [x0 + i * p for i in range(len(seq))]
    abox(ax, x0 - 0.1, by, xs[-1] + s - x0 + 0.2, bh, "pre", "decoder", fs=18, bold=True)
    for i, m in enumerate(seq):
        x = xs[i]
        if m == "M":
            _mask_sq(ax, x, yb, s, fs=10)
        elif m == "BO":
            sq(ax, x, yb, s, MODC["S2L2A"], solid=True)
        else:
            sq(ax, x, yb, s, MODC["S1RTC"], solid=i in (0, 3))
        arrow(ax, (x + s / 2, yb + s + 0.02), (x + s / 2, by - 0.02), lw=1.0, ms=7)
        if i >= 4:
            ax.add_patch(Circle((x + s / 2, yt + 0.12), 0.13, fc="white", ec=PRE_EC, lw=1.4, zorder=6))
            arrow(ax, (x + s / 2, by + bh + 0.02), (x + s / 2, yt - 0.03), lw=1.0, ms=7)
    T(ax, (xs[0] + xs[3] + s) / 2, yb - 0.17, "source", fs=13.5, b=True, c=MODC["S1RTC"])
    T(ax, (xs[4] + xs[-1] + s) / 2, yb - 0.17, "target: all hidden", fs=13.5, b=True, c=MODC["S2L2A"])
    T(ax, (xs[4] + xs[-1] + s) / 2, yt + 0.62, "every patch in one pass", fs=14, b=True)
    hx, hy = xs[-1] + s / 2, yt + 0.12
    ax.add_patch(Circle((hx, hy), 0.17, fc="none", ec=BAD, lw=2, zorder=7))
    # right: one position, zoomed
    zx = 5.55
    ax.plot([hx + 0.17, zx - 0.1], [hy, 2.6], color=MUTED, lw=1.2, ls=(0, (2, 2)))
    ax.add_patch(Circle((zx + 0.15, 2.6), 0.22, fc="white", ec=PRE_EC, lw=1.8, zorder=6))
    T(ax, zx + 0.15, 2.6, "h", fs=14, b=True)
    arrow(ax, (zx + 0.39, 2.6), (zx + 0.78, 2.6), lw=1.6, ms=12)
    abox(ax, zx + 0.8, 2.15, 2.1, 0.9, "new", "flow head\n(small MLP)", fs=14)
    T(ax, zx + 1.85, 1.5, "noise", fs=13, c=INK2)
    arrow(ax, (zx + 1.85, 1.68), (zx + 1.85, 2.13), lw=1.4, ms=11)
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((zx + 1.2, 3.08), (zx + 2.5, 3.08), connectionstyle="arc3,rad=-0.9",
                                 arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.4, zorder=6))
    T(ax, zx + 1.85, 3.92, "32 steps", fs=13, c=INK2)
    arrow(ax, (zx + 2.92, 2.6), (zx + 3.3, 2.6), lw=1.6, ms=12)
    # the DCT frequency plane, with the 16 kept coefficients
    gx, gy, gs = zx + 3.35, 1.9, 1.4
    c = gs / 16
    keep = set(int(k) for k in np.asarray(dct_order(16))[:16])
    for u in range(16):
        for v in range(16):
            k = u * 16 + v
            ax.add_patch(Rectangle((gx + v * c, gy + (15 - u) * c), c, c, fc=COL["flowdct_out"] if k in keep else "white",
                                   ec=GRID, lw=0.4, zorder=4))
    ax.add_patch(Rectangle((gx, gy), gs, gs, fc="none", ec=INK2, lw=1.2, zorder=5))
    T(ax, gx + gs / 2, gy - 0.22, "16 of 256 frequencies,\nper band (192 for S2)", fs=13, c=INK2, va="top")
    arrow(ax, (gx + gs + 0.05, 2.6), (gx + gs + 0.4, 2.6), lw=1.6, ms=12)
    rbox(ax, gx + gs + 0.42, 2.2, 1.45, 0.8, "inverse\nDCT", fs=14)
    T(ax, gx + gs + 1.15, 1.95, "fixed", fs=12.5, c=INK2, va="top")
    arrow(ax, (gx + gs + 1.89, 2.6), (gx + gs + 2.2, 2.6), lw=1.6, ms=12)
    rgb_inset(fig, ax, gx + gs + 2.22, 2.0, 1.2, s2_patch(n=16), col=MODC["S2L2A"], lw=2)
    T(ax, gx + gs + 2.82, 1.8, "a 16 × 16 × 12\npatch", fs=13, c=INK2, va="top")
    legend_row(ax, [("new", "new")], [7.4], y=-0.3)
    save(fig, "idea4.png")


def fig_idea5():
    W, H = 14.4, 5.2
    fig, ax = canvas(W, H, ylo=-0.5)
    s2 = np.load(CIO / "data/test_S2L2A.npy", mmap_mode="r")[1, 3, 40:88, 120:168].astype(np.float32)
    s1 = np.load(CIO / "data/test_S1RTC.npy", mmap_mode="r")[1, 1, 40:88, 120:168].astype(np.float32)
    rows = [(s2, "S2L2A", "Sentinel-2 · B04\n665 nm · 31 nm wide · L2A"),
            (s1, "S1RTC", "Sentinel-1 · VV\nC-band · RTC"),
            (None, None, "a new sensor's band\nits wavelength, width, level")]
    ys = [3.1, 1.7, 0.3]
    for (img, m, txt), y in zip(rows, ys):
        if img is None:
            rbox(ax, 0.2, y, 0.95, 0.95, "?", fc="white", ec=NEW, lw=2, ls=(0, (4, 3)), fs=22, b=True, tc=NEW)
        else:
            a = inset(fig, ax, 0.2, y, 0.95)
            a.imshow(img, cmap="gray", vmin=np.percentile(img, 2), vmax=np.percentile(img, 98)); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color(MODC[m]); sp.set_linewidth(2)
        rbox(ax, 1.4, y + 0.08, 3.1, 0.8, txt, fc="white", ec=GRID, fs=13)
        arrow(ax, (4.52, y + 0.48), (5.18, 1.95 + (y - 1.7) * 0.35), lw=1.3, ms=10)
    abox(ax, 5.2, 1.0, 2.3, 1.95, "new", "band embedder:\nweights from\nthe description", fs=14)
    arrow(ax, (7.52, 1.97), (7.98, 1.97), lw=1.8, ms=14)
    abox(ax, 8.0, 1.0, 2.6, 1.95, "pre", "decoder", fs=19, bold=True)
    arrow(ax, (10.62, 1.97), (11.08, 1.97), lw=1.8, ms=14)
    abox(ax, 11.1, 1.0, 3.1, 1.95, "new", "output layer\nbuilt from the\nrequested bands", fs=14)
    T(ax, 9.3, 3.7, "a band is described by its physics, not by its name:\nan unseen sensor needs no new weights",
      fs=15, c=INK2)
    T(ax, 9.3, 0.55, "DOFA · Copernicus-FM", fs=13, c=INK2)
    legend_row(ax, [("new", "new")], [6.2], y=-0.25)
    save(fig, "idea5.png")


FIGS = {"today": fig_today, "lit_inputs": fig_lit_inputs, "lit_outputs": fig_lit_outputs,
        "lit_compression": fig_lit_compression, "lit_eo": fig_lit_eo,
        "token_info": fig_token_info, "pilot_seq": fig_pilot_seq, "pilot_variants": fig_pilot_variants,
        "inputs_delta": fig_inputs_delta, "level_relief": fig_level_relief, "any_band": fig_any_band,
        "outputs_examples": fig_outputs_examples, "outputs_scores": fig_outputs_scores, "lulc": fig_lulc,
        "ideas_map": fig_ideas_map, "idea1": fig_idea1, "idea1_retrofit": fig_idea1_retrofit, "idea2": fig_idea2,
        "idea3": fig_idea3, "idea4": fig_idea4, "idea5": fig_idea5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for k, f in FIGS.items():
        if a.only is None or k in a.only:
            f()
    if NUM:
        p = OUT / "numbers.json"
        old = jl(p) if p.exists() else {}
        old.update(NUM)
        p.write_text(json.dumps(old, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
