#!/usr/bin/env python
"""Step 12's figures and checks: the CE grid (D3.13), the accuracy-vs-tokens curve
(D3.16) and the routed arm's depth per cell (D3.14). PHASE3_PLAN.md 12.1, 12.2, 12.4.

    python eo/scripts/grid_report.py \
        --arm a10=/data/enric/runs/pretrain/phase3/mor10_20261001_204536 \
        --arm b10=/data/enric/runs/pretrain/phase3/vanilla10_20261002_133848 \
        --out /data/enric/reports/grid/a10_b10

The first --arm is A, the second B; every difference is A - B. Reads what
`eval_checkpoint.py --grid` wrote beside each checkpoint's metrics.json: grid.json
(per cell) and grid_rows.npz (per row: `<S>-><T>|rows`, `|n`, `|ce_sum`, and for a
routed arm `|d_pred`, `|d_tgt`, `|d_src`). The FLOPs axis comes from
eo/eval/curves.py, i.e. from eval_checkpoint.py's main pass (tokens_router.npz).
Writes grid_report.json, curve.csv, summary.txt and the figures to --out.

THE NUMBER PER CELL is the mean over the cell's rows of each row's mean target-token
CE (nats/token, lower is better), recomputed here from the per-row arrays and
checked against grid.json. Top-1 is stored upstream and never reported (plan 4.18).

EVERY COMPARISON IS PAIRED BY ROW ID (12.3 rule 3).
  A - B        per cell, the two arms on the same rows. The row sets must be
               identical (12.3 rule 1): a mismatch is an error, not a warning.
  gain over ∅  per cell and arm, CE(S->T) - CE(∅->T) on the cell's own rows; the ∅
               row covers every row of its column. This is the readable view of
               "how much a source helps". Raw CE within a column compares different
               scene populations: S1GRD cells are ssl4eos12-only, S1RTC cells
               majortom-only (plan 12.1, the 2026-10-04 deviation).
Per-cell CIs are percentile bootstraps over the cell's rows (10,000 resamples, a
seed fixed per cell). The curve's pooled and per-target CIs resample SCENES jointly:
one multinomial weight vector over the union of row ids per resample, shared by
every cell and checkpoint. A scene appears in several cells of a column and in
several columns, so resampling cells independently would treat correlated cells as
independent and make the band too narrow.

BUILT-IN CHECKS (12.1, paired), every arm, headline checkpoint. They test the
wiring, not a hypothesis: if one fails, debug before reading anything else.
  1. S2L2A -> NDVI has the most negative mean gain among NDVI's one-to-one cells.
  2. No one-to-one cell's gain CI lies entirely above 0.
     ⚠ That is 34 tests at 95%: a source that truly does not help still has a 2.5%
     chance per cell of a CI above 0. A Bonferroni reading (alpha 0.05/34) is
     printed beside the verdict and reported only; the verdict is the plan's rule.
Exit status 1 if a check fails (after every output is written).

Runs in `.venv`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from eo.data.eo_vocab import IMAGE_MODALITIES, MODALITIES  # noqa: E402
from eo.eval import curves as CV  # noqa: E402

TARGETS = list(IMAGE_MODALITIES)                 # grid columns
SOURCES = list(MODALITIES)                       # one-to-one rows, registry order
ROWS = SOURCES + ["all", "none"]
HOLES = {"S1GRD->S1RTC", "S1RTC->S1GRD"}         # no row carries both
ONE_TO_ONE = [f"{s}->{t}" for t in TARGETS for s in SOURCES
              if s != t and f"{s}->{t}" not in HOLES]
assert TARGETS == ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
assert len(ONE_TO_ONE) == 34
ROW_TOKENS, BATCH = 995, 256                     # worklog section 2: every row is 995 tokens
TOKENS_PER_STEP = ROW_TOKENS * BATCH
assert TOKENS_PER_STEP == 254_720
LN_V = math.log(87_556)
MODES = ["force1", "force2", "force3", "perm_modality", "perm_all", "skip"]
LABEL = {"all": "all others", "none": "∅ (target alone)", "S1GRD": "S1GRD ¹", "S1RTC": "S1RTC ²"}

# Palette as arm_report.py (dataviz reference palette): arms keep slots 1 and 2.
ARM_COLORS = ["#2a78d6", "#eb6834"]
INK, INK_2, GRID_INK, SURFACE, REF = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb", "#8a8984"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
PASS_INK = ["#86b6ef", "#3987e5", "#184f95"]    # ordinal: 1, 2, 3 passes


# ------------------------------------------------------------------ loading

def load_grid(d: Path) -> Optional[Dict]:
    """One checkpoint's grid: per cell, row ids (sorted), per-row CE, depth counts."""
    gp, zp = d / "grid.json", d / "grid_rows.npz"
    if not (gp.exists() and zp.exists()):
        return None
    doc, arr = json.loads(gp.read_text()), np.load(zp)
    holes = {k for k, c in doc["cells"].items() if c.get("hole")}
    if holes != HOLES:
        raise RuntimeError(f"{d}: holes are {sorted(holes)}, expected {sorted(HOLES)}")
    cells = {}
    for key, c in doc["cells"].items():
        if c.get("hole"):
            continue
        rows = arr[f"{key}|rows"].astype(np.int64)
        ce = arr[f"{key}|ce_sum"].astype(np.float64) / np.maximum(arr[f"{key}|n"], 1)
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"{d} {key}: duplicate row ids")
        if abs(ce.mean() - c["ce_row_mean"]) > 1e-9:
            raise RuntimeError(f"{d} {key}: per-row CE mean {ce.mean():.12f} != grid.json "
                               f"{c['ce_row_mean']:.12f}")
        o = np.argsort(rows)
        cell = {"rows": rows[o], "ce": ce[o], "json": c}
        for dk in ("d_pred", "d_tgt", "d_src"):
            if f"{key}|{dk}" in arr.files:
                cell[dk] = arr[f"{key}|{dk}"][o]
        cells[key] = cell
    meta = json.loads((d / "metrics.json").read_text())
    return {"cells": cells, "checkpoint": d.name, "step": meta["step"],
            "gpu": doc.get("gpu"), "routed": bool(meta.get("has_router"))}


def load_arm_grids(eval_root: Path, tag: str) -> Dict[str, Dict]:
    out = {}
    for d in sorted((eval_root / tag).iterdir()):
        g = load_grid(d) if d.is_dir() else None
        if g:
            out[g["checkpoint"]] = g
    return out


# ------------------------------------------------------------------ statistics

def boot(x: np.ndarray, key: str, seed: int, n_boot: int, bonf: int = 1) -> Dict:
    """Mean of x with a percentile-bootstrap 95% CI over its elements (rows).
    With bonf > 1, also the Bonferroni-widened interval (alpha 0.05 / bonf)."""
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    out = {"n": int(len(x)), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi),
           "excludes_0": bool(lo > 0 or hi < 0)}
    if bonf > 1:
        a = 100 * 0.05 / bonf / 2
        out["lo_bonf"], out["hi_bonf"] = (float(v) for v in np.percentile(m, [a, 100 - a]))
    return out


def paired_rows(a: Dict, b: Dict, what: str) -> np.ndarray:
    """A - B per row. Rows are sorted by id on load, so equal arrays are aligned."""
    if not np.array_equal(a["rows"], b["rows"]):
        raise RuntimeError(f"{what}: the arms' row sets differ ({len(a['rows'])} vs "
                           f"{len(b['rows'])} rows); 12.3 rule 1 needs identical rows")
    return a["ce"] - b["ce"]


def gain_rows(cells: Dict, key: str) -> np.ndarray:
    """CE(S->T) - CE(∅->T) per row of the S->T cell, joined by row id."""
    T = key.split("->")[1]
    c, z = cells[key], cells[f"none->{T}"]
    pos = np.searchsorted(z["rows"], c["rows"])
    if (pos >= len(z["rows"])).any() or not np.array_equal(z["rows"][pos], c["rows"]):
        raise RuntimeError(f"{key}: rows missing from none->{T}")
    return c["ce"] - z["ce"][pos]


class SceneBootstrap:
    """Joint resampling of scenes: one multinomial weight per (resample, scene),
    shared by every cell and every checkpoint, so the cells' correlation through
    shared scenes is kept and every checkpoint's band uses the same draws."""

    def __init__(self, rows: np.ndarray, seed: int, n_boot: int):
        self.rows = np.unique(rows)
        rng = np.random.default_rng([seed, zlib.crc32(b"scene-bootstrap")])
        k = len(self.rows)
        self.W = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float32)

    def cell_means(self, rows: np.ndarray, v: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.rows, rows)
        if (pos >= len(self.rows)).any() or not np.array_equal(self.rows[pos], rows):
            raise RuntimeError("a cell's rows are outside the bootstrap's scene union")
        w = self.W[:, pos]
        den = w.sum(1)
        if (den == 0).any():
            raise RuntimeError("a resample left a cell empty; cells are too small for this")
        return (w @ v.astype(np.float32)) / den


def depth_stats(c: Dict, dk: str) -> Optional[Dict]:
    if dk not in c:
        return None
    tot = c[dk].sum(0).astype(np.float64)
    if tot.sum() == 0:
        return None
    mix = tot / tot.sum()
    return {"mean": float((mix * np.arange(1, 4)).sum()), "mix": mix.tolist()}


# ------------------------------------------------------------------ figures

def _style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_INK)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID_INK, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")


def _title(fig, title, subtitle):
    """Title on top; the subtitle as a caption underneath (constrained layout reserves
    room for a suptitle only, and bbox_inches="tight" takes in the caption)."""
    fig.suptitle(title, x=0.005, ha="left", fontsize=13, color=INK)
    fig.text(0.005, -0.005, subtitle, transform=fig.transFigure, ha="left", va="top",
             fontsize=8.5, color=INK_2, wrap=True)


def _save(fig, path):
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _heatmap(ax, M, rows, cmap, norm, fmt, title, outline=None, outline_ls="-",
             cbar_label=None, fig=None, hole_text="—"):
    """Rows x TARGETS heatmap. NaN = hole (hatched, never coloured: a hole is not 0)."""
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=SURFACE,
                                       edgecolor=GRID_INK, hatch="///", lw=0))
                ax.text(j, i, hole_text, ha="center", va="center", fontsize=7, color=REF)
                continue
            r, g, b, _ = cmap(norm(v))
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if dark else INK)
            if outline is not None and outline[i, j]:
                ax.add_patch(Rectangle((j - .46, i - .46), .92, .92, fill=False,
                                       edgecolor=INK, lw=1.6, ls=outline_ls))
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([LABEL.get(t, t) for t in TARGETS], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL.get(r, r) for r in rows], fontsize=8)
    ax.tick_params(colors=INK_2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=22)
    if fig is not None:
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=INK_2)
        cb.outline.set_visible(False)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=7.5, color=INK_2)


def _matrix(rows, f):
    M = np.full((len(rows), len(TARGETS)), np.nan)
    for i, s in enumerate(rows):
        for j, t in enumerate(TARGETS):
            v = f(f"{s}->{t}")
            if v is not None:
                M[i, j] = v
    return M


def fig_grid(R, names, out):
    A, B = names
    ce = {n: R["arms"][n]["cells"] for n in names}
    MA = _matrix(ROWS, lambda k: ce[A].get(k, {}).get("ce"))
    MB = _matrix(ROWS, lambda k: ce[B].get(k, {}).get("ce"))
    MD = _matrix(ROWS, lambda k: R["paired"].get(k, {}).get("mean"))
    XD = _matrix(ROWS, lambda k: float(R["paired"][k]["excludes_0"]) if k in R["paired"] else None) == 1
    gr = ROWS[:-1]
    MG = {n: _matrix(gr, lambda k, n=n: ce[n].get(k, {}).get("gain", {}).get("mean")) for n in names}
    XG = {n: _matrix(gr, lambda k, n=n: float(ce[n][k]["gain"]["lo"] > 0)
                     if "gain" in ce[n].get(k, {}) else None) == 1 for n in names}

    fig, axs = plt.subplots(2, 3, figsize=(18, 11), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    seq = Normalize(0, float(np.nanmax([MA, MB])))
    lab = {n: f"{n} ({R['arms'][n]['kind']})" for n in names}
    _heatmap(axs[0, 0], MA, ROWS, SEQ, seq, lambda v: f"{v:.3f}", f"{lab[A]}: CE of the target, nats/token")
    _heatmap(axs[0, 1], MB, ROWS, SEQ, seq, lambda v: f"{v:.3f}", f"{lab[B]}: same colour scale",
             cbar_label="nats/token (lower is better)", fig=fig)
    lim = max(0.02, float(np.nanmax(np.abs(MD))))
    _heatmap(axs[0, 2], MD, ROWS, DIV, TwoSlopeNorm(0, -lim, lim), lambda v: f"{v:+.3f}",
             f"{A} − {B}, paired by row  (outlined: 95% CI excludes 0)", outline=XD,
             cbar_label=f"nats/token  (blue: {A} lower)", fig=fig)
    glim = max(0.02, float(np.nanmax(np.abs([MG[n] for n in names]))))
    for ax, n, cb in ((axs[1, 0], A, False), (axs[1, 1], B, True)):
        _heatmap(ax, MG[n], gr, DIV, TwoSlopeNorm(0, -glim, glim), lambda v: f"{v:+.3f}",
                 f"{lab[n]}: gain over ∅ = CE(S→T) − CE(∅→T), same rows", outline=XG[n],
                 outline_ls="--", cbar_label="nats/token  (blue: the source helps)",
                 fig=fig if cb else None)
    ax = axs[1, 2]
    ax.axis("off")
    ck = R["headline"]
    lines = [f"Checkpoint: {ck} for both arms (12.3 rule 2). Card: "
             + ", ".join(sorted({str(R['arms'][n]['gpu']) for n in names})) + ".",
             "",
             "Cell: mean over the cell's rows of each row's mean CE",
             "on the target's 196 tokens. Rows of column T: 512",
             "corpus-stratified held-out scenes carrying T; a cell",
             f"uses those that also carry S (n = {R['n_rows_range'][0]}–{R['n_rows_range'][1]}).",
             "'all others' = the target-last pass (D3.11's context).",
             "",
             "¹ S1GRD cells: ssl4eos12 rows only.  ² S1RTC: majortom only.",
             "  Within a column, read the gain panels, not raw CE.",
             "Hatched: diagonal, or no row carries both S1GRD and S1RTC.",
             "Dashed outline: gain CI entirely above 0 (check 2 fails).",
             "",
             "⚠ S1GRD is on 9.8% of rows and overfits first; Coords",
             "  is 3 tokens per scene and memorizes.",
             f"⚠ {B} has 2.6× {A}'s non-embedding parameters" if R["arms"][B]["kind"] == "vanilla"
             else "",
             "  (102.66 M vs 39.61 M): an upper bound, not matched." if R["arms"][B]["kind"] == "vanilla"
             else "",
             "",
             "Built-in checks (wiring, not hypotheses):"]
    for n in names:
        c = R["checks"][n]
        lines.append(f"  {n}: 1 {'PASS' if c['check1']['pass'] else 'FAIL'}"
                     f"  (best NDVI source: {c['check1']['best']})")
        lines.append(f"  {n}: 2 {'PASS' if c['check2']['pass'] else 'FAIL'}"
                     f"  ({len(c['check2']['worse'])} of 34 above 0; "
                     f"Bonferroni: {len(c['check2']['worse_bonf'])})")
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8.5, color=INK_2,
            family="DejaVu Sans", transform=ax.transAxes)
    _title(fig, f"D3.13 — Any-to-any: teacher-forced CE of the target given one source ({ck})",
           "Rows: the source in context (then the target). Columns: the target. "
           "Every comparison is paired on identical rows; CIs are 10,000-resample bootstraps over rows.")
    _save(fig, out / f"fig_grid_{ck}.png")


def fig_curve(R, names, out):
    cur = R["curve"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    drawn = set()
    for ai, n in enumerate(names):
        pts = [(p["tokens"], p[n]["pooled"]) for p in cur if p.get(n, {}).get("pooled") is not None]
        if pts:
            x, y = zip(*pts)
            axs[0, 0].plot(np.array(x) / 1e6, y, color=ARM_COLORS[ai], lw=2, marker="o", ms=4,
                           label=f"{n} ({R['arms'][n]['kind']})")
        for ax, key in ((axs[1, 0], "cum_flops_full"), (axs[1, 1], "cum_flops_plan")):
            fp = [(p[n][key], p[n]["pooled"]) for p in cur
                  if p.get(n, {}).get("pooled") is not None and p[n].get(key) is not None]
            if fp:
                x, y = zip(*fp)
                ax.plot(np.array(x) / 1e17, y, color=ARM_COLORS[ai], lw=2, marker="o", ms=4,
                        label=n)
                drawn.add(key)
    for ax, key, xl, t in (
            (axs[0, 0], None, "training tokens seen (millions)", "mean CE over the 34 one-to-one cells"),
            (axs[1, 0], "cum_flops_full", "cumulative training FLOPs, full count (×10¹⁷)",
             "the same, against compute (headline count)"),
            (axs[1, 1], "cum_flops_plan", "cumulative training FLOPs, plan count (×10¹⁷)",
             "the same, plan count")):
        ax.set_xlim(left=0)
        ax.axhline(LN_V, color=REF, lw=1.2, ls=":")
        ax.text(0.98, LN_V + 0.15, "uniform over the vocabulary (ln 87,556)", ha="right", va="bottom",
                fontsize=7.5, color=INK_2, transform=ax.get_yaxis_transform())
        ax.set_ylim(0, LN_V + 0.8)
        _style(ax, t)
        ax.set_xlabel(xl, fontsize=8, color=INK_2)
        ax.set_ylabel("nats/token (lower is better)", fontsize=8, color=INK_2)
        if key and key not in drawn:
            ax.text(0.5, 0.5, "no FLOPs points: needs eval_checkpoint.py's main pass\n"
                    "on the untrained model and every checkpoint", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=INK_2)
        elif key:
            ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper right",
                      bbox_to_anchor=(1, 0.93))
    axs[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper right",
                     bbox_to_anchor=(1, 0.93))
    A, B = names
    _diff_panel(axs[0, 1], cur, lambda p: p.get("diff", {}).get("pooled"),
                f"{A} − {B}, pooled, paired (band: 95% CI, scenes resampled)", A)
    if any(p.get("diff") for p in cur):
        axs[0, 1].set_ylim(*_sym_lim([p["diff"]["pooled"] for p in cur if p.get("diff")]))
    _title(fig, "D3.16 — Accuracy against training tokens, and against compute",
           "y = equal-weight mean over the 34 one-to-one grid cells of teacher-forced CE; "
           "untrained + every checkpoint. x = step × 254,720 tokens (256 rows × 995). "
           f"{B} spends ~29 layer applications per token, {A} its measured depth: "
           "a tie on tokens is a difference on compute.")
    _save(fig, out / "fig_curve.png")


def _sym_lim(cis, floor=0.05):
    m = max([floor] + [max(abs(c["lo"]), abs(c["hi"]), abs(c["mean"])) for c in cis])
    return -1.15 * m, 1.15 * m


def _diff_panel(ax, cur, get, title, A):
    pts = [(p["tokens"], get(p)) for p in cur if get(p) is not None]
    if pts:
        x = np.array([t for t, _ in pts]) / 1e6
        m = np.array([c["mean"] for _, c in pts])
        lo, hi = np.array([c["lo"] for _, c in pts]), np.array([c["hi"] for _, c in pts])
        ax.fill_between(x, lo, hi, color=INK_2, alpha=0.15, lw=0)
        ax.plot(x, m, color=INK, lw=1.8, marker="o", ms=3.5)
    ax.axhline(0, color=REF, lw=1.2)
    _style(ax, title)
    ax.set_xlim(left=0)
    ax.set_xlabel("training tokens seen (millions)", fontsize=8, color=INK_2)
    ax.set_ylabel(f"nats/token  (below 0: {A} lower CE)", fontsize=8, color=INK_2)


def fig_curve_targets(R, names, out):
    cur = R["curve"]
    A, B = names
    fig, axs = plt.subplots(2, 6, figsize=(20, 7), layout="constrained", sharey="row")
    fig.patch.set_facecolor(SURFACE)
    note = {"S1GRD": "  ⚠ ssl4eos12 only, 9.8% of rows", "S1RTC": "  (majortom only)"}
    for j, T in enumerate(TARGETS):
        ax = axs[0, j]
        for ai, n in enumerate(names):
            pts = [(p["tokens"], p[n]["per_target"][T]) for p in cur
                   if p.get(n, {}).get("per_target", {}).get(T) is not None]
            if pts:
                x, y = zip(*pts)
                ax.plot(np.array(x) / 1e6, y, color=ARM_COLORS[ai], lw=2, marker="o", ms=3.5,
                        label=n)
        ax.axhline(LN_V, color=REF, lw=1.2, ls=":")
        ax.set_ylim(0, LN_V + 0.8)
        ax.set_xlim(left=0)
        _style(ax, T + note.get(T, ""))
        ax.set_xlabel("tokens seen (M)", fontsize=8, color=INK_2)
        _diff_panel(axs[1, j], cur, lambda p, T=T: p.get("diff", {}).get("per_target", {}).get(T),
                    f"{A} − {B}, {T}", A)
        axs[1, j].set_xlabel("tokens seen (M)", fontsize=8, color=INK_2)
        if j:
            axs[1, j].set_ylabel("")
    axs[0, 0].set_ylabel("mean CE of the column's one-to-one cells", fontsize=8, color=INK_2)
    axs[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK_2)
    cis = [c for p in cur if p.get("diff") for c in p["diff"]["per_target"].values()]
    if cis:
        axs[1, 0].set_ylim(*_sym_lim(cis))
    _title(fig, "D3.16 per target: the curve by column, and the paired difference",
           "Top: mean over the column's one-to-one cells (6, or 5 for S1GRD/S1RTC); dotted: ln 87,556. "
           "Bottom: A − B paired by row, 95% CI with scenes resampled jointly. One shared y-scale per row.")
    _save(fig, out / "fig_curve_targets.png")


def fig_depth(R, n, out):
    """Depth per cell for a routed arm, with the interventions printed beside it."""
    D = R["depth"][n]
    ck = R["headline"]
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), layout="constrained",
                            gridspec_kw={"width_ratios": [1, 1, 1.15]})
    fig.patch.set_facecolor(SURFACE)
    norm = Normalize(1, 3)
    def mean(k, dk):
        return ((D.get(k) or {}).get(dk) or {}).get("mean")
    MP = _matrix(ROWS, lambda k: mean(k, "d_pred"))
    MS = _matrix(ROWS, lambda k: mean(k, "d_src"))
    _heatmap(axs[0], MP, ROWS, SEQ, norm, lambda v: f"{v:.2f}",
             "mean passes at the positions predicting the target")
    _heatmap(axs[1], MS, ROWS, SEQ, norm, lambda v: f"{v:.2f}",
             "mean passes on the source tokens", cbar_label="recursion passes (1–3)", fig=fig)
    ax = axs[2]
    iv = R["interventions"].get(n)
    if iv:
        ms = [m for m in MODES if m in iv["delta"]]
        mods = list(iv["delta"][ms[0]])
        M = np.array([[iv["delta"][m][x] for m in ms] for x in mods])
        lim = max(0.05, float(np.abs(M[:, [i for i, m in enumerate(ms) if m != "skip"]]).max()))
        nrm = TwoSlopeNorm(0, -lim, lim)
        ax.imshow(M, cmap=DIV, norm=nrm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                r, g, b, _ = DIV(nrm(M[i, j]))
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK)
        lbl = {"force1": "all at 1", "force2": "all at 2", "force3": "all at 3",
               "perm_modality": "shuffle\nin modality", "perm_all": "shuffle\nacross all",
               "skip": "0 passes\n(skip stack)"}
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([lbl[m] for m in ms], fontsize=7.5)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(mods)))
        ax.set_yticklabels(mods, fontsize=8)
        ax.tick_params(colors=INK_2, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"interventions at {iv['checkpoint']}: CE(override) − CE(router), nats\n"
                     f"router mean depth {iv['router_depth']:.2f}; skip clips the colour scale",
                     fontsize=9.5, color=INK, loc="left", pad=30)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"⚠ NO INTERVENTIONS at {ck} for {n}.\nDo not show the depth panels "
                "without them\n(plan 12.4 guardrail).", ha="center", va="center", fontsize=11,
                color="#d03b3b", transform=ax.transAxes)
    _title(fig, f"D3.14 — Where {n}'s router spends depth, per grid cell ({ck})",
           "⚠ A deeper cell means the router ALLOCATED more passes there, not that the model needed "
           "them. In the 50-epoch arm a fixed depth matched the router and shuffling depths within a "
           "modality moved CE by 0.000 (ARM_A_REPORT §7.2). Read the left panels only through the right one.")
    _save(fig, out / f"fig_depth_{n}_{ck}.png")


def fig_compute(R, n, out):
    """Layer applications per predicted target token, stacked by pass count, vs 29."""
    D = R["depth"][n]
    keys = [f"{s}->{t}" for t in TARGETS for s in ROWS if (D.get(f"{s}->{t}") or {}).get("d_pred")]
    fig, ax = plt.subplots(figsize=(9, 0.21 * len(keys) + 1.8), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        mix = D[k]["d_pred"]["mix"]
        segs = [CV.OUTER] + [CV.BLOCKS_PER_STEP * (i + 1) * mix[i] for i in range(3)]
        left = 0.0
        for si, w in enumerate(segs):
            ax.barh(yi, w, left=left, height=0.78, color=([REF] + PASS_INK)[si],
                    edgecolor=SURFACE, lw=1)
            left += w
        inside = left > 26                     # keep clear of the vanilla line at 29
        ax.text(left - 0.3 if inside else left + 0.3, yi, f"{left:.1f}", va="center",
                ha="right" if inside else "left", fontsize=7, color="#ffffff" if inside else INK_2)
    ax.axvline(29, color=INK, lw=1.2, ls="--")
    ax.text(29, y[0] + 0.9, "vanilla: 29", ha="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("none", "∅").replace("all", "all others").replace("->", " → ")
                        for k in keys], fontsize=7)
    ax.set_xlim(0, 31)
    _style(ax, None)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("layer applications per predicted target token (2 unshared + 9 per pass)",
                  fontsize=8, color=INK_2)
    h = [Patch(color=REF)] + [Patch(color=c) for c in PASS_INK]
    ax.legend(h, ["2 unshared layers", "tokens at 1 pass", "2 passes", "3 passes"], fontsize=7.5,
              frameon=False, labelcolor=INK_2, loc="lower right")
    _title(fig, f"D3.14 — Compute per predicted target token, {n} ({R['headline']})",
           "Plan count, body positions predicting the target. Router and attention's quadratic "
           "term excluded. ⚠ Read with the interventions (fig_depth).")
    _save(fig, out / f"fig_compute_{n}_{R['headline']}.png")


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="tag=run_dir; exactly two, A then B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="checkpoint-3300", help="the headline checkpoint (12.3 rule 2)")
    ap.add_argument("--eval-root", default="/data/enric/reports/arm_eval")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if len(args.arm) != 2:
        ap.error("exactly two --arm, A then B")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.eval_root)
    names, run_dirs = zip(*(s.split("=", 1) for s in args.arm))
    A, B = names
    grids = {n: load_arm_grids(root, n) for n in names}
    ck = args.checkpoint
    for n in names:
        if ck not in grids[n]:
            sys.exit(f"{n}: no complete grid at {ck} under {root / n}")
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    # --- the grid at the headline checkpoint
    R = {"headline": ck, "arms": {}, "paired": {}, "checks": {}, "depth": {}, "interventions": {},
         "provenance": {"eval_root": str(root), "n_boot": args.n_boot, "seed": args.seed,
                        "run_dirs": dict(zip(names, run_dirs))}}
    for n in names:
        g = grids[n][ck]
        missing = [f"{s}->{t}" for s in ROWS for t in TARGETS
                   if s != t and f"{s}->{t}" not in HOLES and f"{s}->{t}" not in g["cells"]]
        if missing:
            sys.exit(f"{n} {ck}: cells missing {missing}")
        cells = {}
        for k, c in g["cells"].items():
            cells[k] = {"n": len(c["rows"]), "ce": float(c["ce"].mean())}
            if not k.startswith("none->"):
                cells[k]["gain"] = boot(gain_rows(g["cells"], k), f"gain|{n}|{k}", args.seed,
                                        args.n_boot, bonf=len(ONE_TO_ONE))
        R["arms"][n] = {"kind": "MoR" if g["routed"] else "vanilla", "gpu": g["gpu"], "cells": cells}
        if g["routed"]:
            R["depth"][n] = {k: {dk: depth_stats(c, dk) for dk in ("d_pred", "d_tgt", "d_src")}
                             for k, c in g["cells"].items()}
            meta = json.loads((root / n / ck / "metrics.json").read_text())
            if all(m in meta["modes"] for m in MODES):
                base = meta["modes"]["router"]["per_modality"]
                R["interventions"][n] = {
                    "checkpoint": ck, "router_depth": meta["modes"]["router"]["mean_depth_body"],
                    "delta": {m: {x: meta["modes"][m]["per_modality"][x]["ce"] - base[x]["ce"]
                                  for x in base} for m in MODES}}
    if {grids[n][ck]["gpu"] for n in names} != {grids[A][ck]["gpu"]}:
        say(f"⚠ the arms were evaluated on different cards: {[grids[n][ck]['gpu'] for n in names]} "
            "(12.3 rule 1)")
    for k in ONE_TO_ONE + [f"{s}->{t}" for s in ("all", "none") for t in TARGETS]:
        R["paired"][k] = boot(paired_rows(grids[A][ck]["cells"][k], grids[B][ck]["cells"][k], k),
                              f"paired|{k}", args.seed, args.n_boot)
    ns = [R["arms"][A]["cells"][k]["n"] for k in ONE_TO_ONE]
    R["n_rows_range"] = (min(ns), max(ns))

    # --- built-in checks
    ok = True
    for n in names:
        cells = R["arms"][n]["cells"]
        ndvi = {k: cells[k]["gain"]["mean"] for k in ONE_TO_ONE if k.endswith("->NDVI")}
        best = min(ndvi, key=ndvi.get)
        worse = [k for k in ONE_TO_ONE if cells[k]["gain"]["lo"] > 0]
        worse_b = [k for k in ONE_TO_ONE if cells[k]["gain"]["lo_bonf"] > 0]
        R["checks"][n] = {"check1": {"pass": best == "S2L2A->NDVI", "best": best,
                                     "gains_ndvi": dict(sorted(ndvi.items(), key=lambda kv: kv[1]))},
                          "check2": {"pass": not worse, "worse": worse, "worse_bonf": worse_b}}
        ok &= best == "S2L2A->NDVI" and not worse

    # --- the curve: every checkpoint with a grid in both arms (or one, then no diff)
    union = np.concatenate([grids[A][ck]["cells"][k]["rows"] for k in ONE_TO_ONE])
    SB = SceneBootstrap(union, args.seed, args.n_boot)
    flops = {}
    for n, rd in zip(names, run_dirs):
        arm = CV.load_arm(n, rd, eval_root=str(root))
        flops[n] = {r["checkpoint"]: r for r in arm["checkpoints"]}
    cks = sorted(set(grids[A]) | set(grids[B]), key=lambda c: next(grids[n][c]["step"] for n in names
                                                                   if c in grids[n]))
    curve = []
    for c in cks:
        step = next(grids[n][c]["step"] for n in names if c in grids[n])
        p = {"checkpoint": c, "step": step, "tokens": step * TOKENS_PER_STEP}
        for n in names:
            g = grids[n].get(c)
            if not g or any(k not in g["cells"] for k in ONE_TO_ONE):
                if g:
                    say(f"⚠ {n} {c}: incomplete grid, left out of the curve")
                continue
            m = {k: float(g["cells"][k]["ce"].mean()) for k in ONE_TO_ONE}
            f = flops[n].get(c, {})
            p[n] = {"pooled": float(np.mean(list(m.values()))),
                    "per_target": {T: float(np.mean([v for k, v in m.items() if k.endswith(f"->{T}")]))
                                   for T in TARGETS},
                    "cum_flops_full": f.get("cum_flops_full"), "cum_flops_plan": f.get("cum_flops_plan"),
                    "layers_per_body_token": f.get("layers_per_body_token")}
        if A in p and B in p:
            per = {k: SB.cell_means(grids[A][c]["cells"][k]["rows"],
                                    paired_rows(grids[A][c]["cells"][k], grids[B][c]["cells"][k], f"{c} {k}"))
                   for k in ONE_TO_ONE}
            pt = {k: float(paired_rows(grids[A][c]["cells"][k], grids[B][c]["cells"][k], k).mean())
                  for k in ONE_TO_ONE}

            def ci(ks):
                bm = np.mean([per[k] for k in ks], 0)
                lo, hi = np.percentile(bm, [2.5, 97.5])
                return {"mean": float(np.mean([pt[k] for k in ks])), "lo": float(lo), "hi": float(hi)}
            p["diff"] = {"pooled": ci(ONE_TO_ONE),
                         "per_target": {T: ci([k for k in ONE_TO_ONE if k.endswith(f"->{T}")])
                                        for T in TARGETS}}
        curve.append(p)
    R["curve"] = curve

    # --- text summary
    say(f"Step 12 grid report: {A} vs {B}, headline {ck}, cards "
        f"{sorted({str(grids[n][ck]['gpu']) for n in names})}")
    for n in names:
        cells = R["arms"][n]["cells"]
        say(f"\n{n} ({R['arms'][n]['kind']}): CE per cell, nats/token  [n rows]")
        say("source       " + "".join(f"{t:>16}" for t in TARGETS))
        for s in ROWS:
            row = [(f"{cells[k]['ce']:.3f} [{cells[k]['n']}]" if (k := f"{s}->{t}") in cells else "—")
                   for t in TARGETS]
            say(f"{LABEL.get(s, s)[:12]:<13}" + "".join(f"{v:>16}" for v in row))
    say(f"\n{A} − {B}, paired, mean [95% CI]  (* = CI excludes 0)")
    for s in ROWS:
        row = []
        for t in TARGETS:
            d = R["paired"].get(f"{s}->{t}")
            row.append(f"{d['mean']:+.3f}{'*' if d['excludes_0'] else ' '}[{d['lo']:+.3f},{d['hi']:+.3f}]"
                       if d else "—")
        say(f"{LABEL.get(s, s)[:12]:<13}" + "  ".join(f"{v:>24}" for v in row))
    for n in names:
        c = R["checks"][n]
        say(f"\n{n} built-in checks:")
        say(f"  1. S2L2A->NDVI has the largest gain in the NDVI column: "
            f"{'PASS' if c['check1']['pass'] else 'FAIL'}  "
            + ", ".join(f"{k.split('->')[0]} {v:+.3f}" for k, v in c["check1"]["gains_ndvi"].items()))
        say(f"  2. no one-to-one cell significantly worse than ∅: "
            f"{'PASS' if c['check2']['pass'] else 'FAIL'}  worse: {c['check2']['worse'] or 'none'}"
            f"  (Bonferroni, reported only: {c['check2']['worse_bonf'] or 'none'})")
        for k in c["check2"]["worse"]:
            gn = R["arms"][n]["cells"][k]["gain"]
            say(f"     {k}: gain {gn['mean']:+.4f} [{gn['lo']:+.4f}, {gn['hi']:+.4f}], n {gn['n']}")
    say(f"\ncurve: pooled mean CE over the 34 one-to-one cells; {A} − {B} [95% CI, scenes resampled]")
    for p in curve:
        d = p.get("diff", {}).get("pooled")
        say(f"  {p['checkpoint']:<16} {p['tokens'] / 1e6:7.1f} M tokens  "
            + "  ".join(f"{n} {p[n]['pooled']:.4f}" for n in names if n in p)
            + (f"  diff {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]" if d else ""))
    say("\nok: every built-in check passed" if ok else
        "\n⚠⚠ A BUILT-IN CHECK FAILED: debug the wiring before reading anything else (plan 12.1)")

    # --- outputs
    def dump(path: Path, text: str):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    dump(out / "grid_report.json", json.dumps(R, indent=1, default=float) + "\n")
    dump(out / "summary.txt", "\n".join(log) + "\n")
    with open(out / "curve.csv.tmp", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "step", "tokens", "arm", "pooled_ce", *[f"ce_{T}" for T in TARGETS],
                    "cum_flops_full", "cum_flops_plan", "layers_per_body_token",
                    "diff_pooled", "diff_lo", "diff_hi"])
        for p in curve:
            for n in names:
                if n in p:
                    d = p.get("diff", {}).get("pooled", {})
                    w.writerow([p["checkpoint"], p["step"], p["tokens"], n, p[n]["pooled"],
                                *[p[n]["per_target"][T] for T in TARGETS], p[n]["cum_flops_full"],
                                p[n]["cum_flops_plan"], p[n]["layers_per_body_token"],
                                d.get("mean"), d.get("lo"), d.get("hi")])
    os.replace(out / "curve.csv.tmp", out / "curve.csv")
    fig_grid(R, names, out)
    fig_curve(R, names, out)
    fig_curve_targets(R, names, out)
    for n in names:
        if n in R["depth"]:
            fig_depth(R, n, out)
            fig_compute(R, n, out)
    print(f"-> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
