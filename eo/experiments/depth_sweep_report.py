"""Depth sweep (2026-10-27): does applying the shared block more times help, and when? `.venv`, CPU.

A RANDOM-router MoR model is evaluated with every token forced to d passes of the shared block
(eval_checkpoint.py --force-depth d), at every epoch checkpoint, and compared with the two vanilla
baselines at the same checkpoints:

  a10cr_force{d}    A10cR, trained with depth ~ U{1,2,3}; d = 1..5 (4, 5 extrapolated)
  a10cr5_force{d}   A10cR5, trained with depth ~ U{1..5}; d = 1..7 (6, 7 extrapolated)
  b10c11            vanilla, 11 layers = MoR's unique layers = the compute of d = 1
  b10c              vanilla, 29 layers = the compute of d = 3
  a10cr             A10cR at its trained random depth (the reference "as trained")

Three views of held-out accuracy (teacher-forced next-token top-1 of the target's tokens):
  any-to-any   the trainer's held-out pass (4,416 rows, random modality order): every token
               predicted from whatever precedes it; token-weighted over all modalities
  all-to-one   target-last: every other modality, then the target; pooled = mean of 6 targets
  one-to-one   one source, then the target (the 34 grid cells); pooled = mean of 34 cells
CE (nats/token) is reported beside top-1 throughout: top-1 is near its floor for S1 and NDVI.

Every number is read from stored results; nothing runs a model.
    .venv/bin/python eo/experiments/depth_sweep_report.py [--only NAME ...]
Writes /data/enric/reports/depth_sweep/{summary.json, summary.csv, figs/*.png}.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARM = Path("/data/enric/reports/arm_eval")
RUNS = Path("/data/enric/runs/pretrain/phase3")
OUT = Path("/data/enric/reports/depth_sweep")
FIGS = OUT / "figs"

CKS = [330 * i for i in range(1, 11)]                      # one checkpoint per epoch
EPOCH = {ck: i + 1 for i, ck in enumerate(CKS)}
MODELS = {"a10cr": dict(nr=3, depths=[1, 2, 3, 4, 5], name="MoR, random router, 3 recursions",
                        short="3 recursions", run="mor10cr_20261012_095834"),
          "a10cr5": dict(nr=5, depths=[1, 2, 3, 4, 5, 6, 7], name="MoR, random router, 5 recursions",
                         short="5 recursions", run=None)}
VAN = {"b10c11": dict(name="vanilla, 11 layers", layers=11, run="vanilla10c11_20261013_070539"),
       "b10c": dict(name="vanilla, 29 layers", layers=29, run="vanilla10c_20261011_172253")}
TARGETS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
SOURCES = TARGETS + ["Coords"]
NICE = {"S2L2A": "S2", "S1GRD": "S1 GRD", "S1RTC": "S1 RTC", "DEM": "DEM", "NDVI": "NDVI", "LULC": "LULC",
        "Coords": "location", "all": "all others"}
VIEWS = {"any": "any-to-any", "a2o": "all-to-one", "o2o": "one-to-one"}
LAYERS = lambda d: 2 + 9 * d                               # layer applications per token at depth d

# Colours (the dataviz reference palette; the midterm deck's model colours where they exist).
SURF, INK, INK2, MUTED, GRIDC, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#efeee9"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
        "#104281", "#0d366b"]                              # epochs 1..10: blue, light -> dark (ordinal)
MCOL = {"a10cr": "#1baf7a", "a10cr5": "#2a78d6", "b10c11": "#4a3aa7", "b10c": "#eb6834"}

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRIDC,
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "savefig.dpi": 200, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.markersize": 6,
})


def jl(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ loading

def load_unit(tag: str, ck: int):
    """One evaluated checkpoint: the three views, top-1 / top-5 / CE. None if incomplete."""
    d = ARM / tag / f"checkpoint-{ck}"
    if not (d / "metrics.json").exists() or not (d / "grid.json").exists():
        return None
    m, g = jl(d / "metrics.json"), jl(d / "grid.json")
    if "router" not in m.get("modes", {}) or "target_last" not in m:
        return None
    cells = g["cells"]
    o2o_keys = [f"{s}->{t}" for t in TARGETS for s in SOURCES if s != t]
    if any(k not in cells for k in o2o_keys):
        return None
    pm = m["modes"]["router"]["per_modality"]
    n = sum(v["n_tokens"] for v in pm.values())
    u = {"force_depth": m.get("force_depth"),
         "any": {"top1": sum(v["top1"] * v["n_tokens"] for v in pm.values()) / n,
                 "top5": sum(v["top5"] * v["n_tokens"] for v in pm.values()) / n,
                 "ce": m["modes"]["router"]["ce_all_body_tokens"],
                 "per_mod": {k: {"top1": v["top1"], "top5": v["top5"], "ce": v["ce"]} for k, v in pm.items()}},
         "a2o": {T: {"top1": m["target_last"][T]["top1"], "top5": m["target_last"][T]["top5"],
                     "ce": m["target_last"][T]["ce"]} for T in m["target_last"]},
         "o2o": {}, "none": {}}
    for k in o2o_keys:
        c = cells[k]
        if c.get("hole"):
            continue
        u["o2o"][k] = {"top1": c["top1"], "top5": c.get("top5"), "ce": c["ce_row_mean"]}
    for T in TARGETS:
        c = cells.get(f"none->{T}")
        if c:
            u["none"][T] = {"top1": c["top1"], "ce": c["ce_row_mean"]}
    u["any"]["pooled"] = u["any"]["top1"]
    u["a2o_pooled"] = {k: float(np.mean([u["a2o"][T][k] for T in TARGETS])) for k in ("top1", "top5", "ce")}
    u["o2o_pooled"] = {k: float(np.mean([v[k] for v in u["o2o"].values()])) for k in ("top1", "ce")}
    return u


def load_all():
    U = {}
    for mk, spec in MODELS.items():
        for d in spec["depths"]:
            for ck in CKS:
                u = load_unit(f"{mk}_force{d}", ck)
                if u is not None:
                    if u["force_depth"] != d:
                        raise SystemExit(f"{mk}_force{d}/checkpoint-{ck} records force_depth {u['force_depth']}")
                    U[(mk, d, ck)] = u
    V = {}
    for vk in list(VAN) + ["a10cr"]:
        for ck in CKS:
            u = load_unit(vk, ck)
            if u is not None:
                V[(vk, ck)] = u
    return U, V


def val(u, view, metric="top1"):
    if view == "any":
        return u["any"][metric]
    return (u["a2o_pooled"] if view == "a2o" else u["o2o_pooled"])[metric]


# ------------------------------------------------------------------ paired bootstrap (rows)

def _rows_any(tag, ck):
    """Per-row (hits, tokens, ce_sum) of the any-to-any pass: tokens_router.npz is (rows, L-1)."""
    z = np.load(ARM / tag / f"checkpoint-{ck}" / "tokens_router.npz")
    v = z["label_mod"] > 0
    return (z["correct"] & v).sum(1).astype(float), v.sum(1).astype(float), \
        np.where(v, z["ce"].astype(np.float64), 0).sum(1)


def boot_any(tag_a, tag_b, ck, n_boot=2000, seed=0):
    """Paired bootstrap over held-out rows of top-1(b) - top-1(a), any-to-any, token-weighted."""
    ha, na, _ = _rows_any(tag_a, ck)
    hb, nb, _ = _rows_any(tag_b, ck)
    if not np.array_equal(na, nb):
        raise SystemExit(f"{tag_a} / {tag_b} {ck}: the passes did not score the same tokens")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(na), (n_boot, len(na)))
    diff = (hb[idx].sum(1) - ha[idx].sum(1)) / na[idx].sum(1)
    point = (hb.sum() - ha.sum()) / na.sum()
    return float(point), [float(x) for x in np.percentile(diff, [2.5, 97.5])]


def boot_o2o(tag_a, tag_b, ck, n_boot=2000, seed=0):
    """Paired bootstrap of the pooled one-to-one top-1 difference (b - a): rows resampled within
    each cell, the same rows for both, then the mean over the 34 cells."""
    za = np.load(ARM / tag_a / f"checkpoint-{ck}" / "grid_rows.npz")
    zb = np.load(ARM / tag_b / f"checkpoint-{ck}" / "grid_rows.npz")
    rng = np.random.default_rng(seed)
    keys = [f"{s}->{t}" for t in TARGETS for s in SOURCES if s != t and f"{s}->{t}|rows" in za]
    pts, boots = [], []
    for k in keys:
        if not np.array_equal(za[f"{k}|rows"], zb[f"{k}|rows"]):
            raise SystemExit(f"{k}: different rows in {tag_a} and {tag_b}")
        n, a, b = za[f"{k}|n"].astype(float), za[f"{k}|top1"].astype(float), zb[f"{k}|top1"].astype(float)
        pts.append((b.sum() - a.sum()) / n.sum())
        idx = rng.integers(0, len(n), (n_boot, len(n)))
        boots.append((b[idx].sum(1) - a[idx].sum(1)) / n[idx].sum(1))
    boots = np.mean(boots, 0)
    return float(np.mean(pts)), [float(x) for x in np.percentile(boots, [2.5, 97.5])]


# ------------------------------------------------------------------ figure helpers

def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{100 * y:.1f}%"))


def shade_extrapolation(ax, nr, dmax, label=True):
    ax.axvspan(nr + 0.5, dmax + 0.5, color=BAND, zorder=0, lw=0)
    if label:
        ax.text(nr + 0.55, 1.0, "not trained\nat this depth", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=11, color=MUTED)


def epoch_legend(fig, ncks, loc="upper center", y=1.04):
    hs = [plt.Line2D([], [], color=RAMP[i], marker="o", lw=2) for i in range(ncks)]
    fig.legend(hs, [f"epoch {i + 1}" for i in range(ncks)], loc=loc, ncol=min(ncks, 10), fontsize=12,
               bbox_to_anchor=(0.5, y), handlelength=1.4, columnspacing=1.0)


def sweep_panel(ax, U, V, mk, getter, title, ylabel, vanilla=True, legend_vanilla=True, label_extra=True):
    """One line per epoch over depth; vanilla (epoch 10) as dashed reference lines."""
    spec = MODELS[mk]
    ds = spec["depths"]
    shade_extrapolation(ax, spec["nr"], ds[-1], label=label_extra)
    for i, ck in enumerate(CKS):
        xs = [d for d in ds if (mk, d, ck) in U]
        if not xs:
            continue
        ys = [getter(U[(mk, d, ck)]) for d in xs]
        ax.plot(xs, ys, color=RAMP[i], marker="o", ms=5, lw=1.8, zorder=3)
    if vanilla:
        for vk in VAN:
            if (vk, CKS[-1]) in V:
                y = getter(V[(vk, CKS[-1])])
                ax.axhline(y, color=MCOL[vk], ls=(0, (5, 3)), lw=1.8, zorder=2)
                if legend_vanilla:
                    ax.text(1.0, y, f" {VAN[vk]['name']}\n (epoch 10)", transform=ax.get_yaxis_transform(),
                            color=MCOL[vk], fontsize=11, va="center", ha="left")
    ax.set_xticks(ds)
    ax.set_xlim(ds[0] - 0.4, ds[-1] + 0.4)
    ax.set_xlabel("recursion depth (passes of the shared block)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# ------------------------------------------------------------------ figures

def fig_sweep_views(U, V, mk, metric="top1"):
    """The headline: accuracy vs depth, one line per epoch, for the three views."""
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
    for ax, view in zip(axs, VIEWS):
        sweep_panel(ax, U, V, mk, lambda u, v=view: val(u, v, metric), VIEWS[view],
                    "top-1 accuracy" if metric == "top1" else "cross-entropy (nats/token)",
                    legend_vanilla=view == "o2o", label_extra=view == "any")
        if metric == "top1":
            pct(ax)
        if metric == "ce":
            ax.invert_yaxis()
    epoch_legend(fig, len(CKS))
    fig.tight_layout(w_pad=2.5)
    save(fig, f"sweep_{metric}_{mk}.png")


def fig_sweep_single(U, V, mk, view, metric="top1"):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    sweep_panel(ax, U, V, mk, lambda u: val(u, view, metric), "",
                "top-1 accuracy" if metric == "top1" else "cross-entropy (nats/token)")
    if metric == "top1":
        pct(ax)
    else:
        ax.invert_yaxis()
    epoch_legend(fig, len(CKS), y=1.08)
    fig.tight_layout()
    save(fig, f"sweep_{view}_{metric}_{mk}.png")


def fig_delta_vanilla(U, V, mk, vk="b10c11", metric="top1"):
    """MoR at depth d minus the vanilla at the SAME epoch: above 0, recursion beats it."""
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
    spec = MODELS[mk]
    for ax, view in zip(axs, VIEWS):
        shade_extrapolation(ax, spec["nr"], spec["depths"][-1], label=view == "any")
        ax.axhline(0, color=MCOL[vk], lw=1.8, ls=(0, (5, 3)), zorder=2)
        for i, ck in enumerate(CKS):
            if (vk, ck) not in V:
                continue
            xs = [d for d in spec["depths"] if (mk, d, ck) in U]
            ys = [val(U[(mk, d, ck)], view, metric) - val(V[(vk, ck)], view, metric) for d in xs]
            if metric == "ce":
                ys = [-y for y in ys]
            ax.plot(xs, ys, color=RAMP[i], marker="o", ms=5, lw=1.8, zorder=3)
        ax.set_xticks(spec["depths"])
        ax.set_xlabel("recursion depth")
        ax.set_title(VIEWS[view])
        if metric == "top1":
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{100 * y:+.2f}"))
            ax.set_ylabel(f"top-1 minus {VAN[vk]['name']} (pp)")
        else:
            ax.set_ylabel(f"CE gain over {VAN[vk]['name']} (nats)")
    epoch_legend(fig, len(CKS))
    fig.tight_layout(w_pad=2.5)
    save(fig, f"delta_{vk}_{metric}_{mk}.png")


def fig_gain_over_training(U, V, metric="top1"):
    """When does recursion help: deepest trained depth minus depth 1, per epoch, both models;
    with the vanilla gap (29 - 11 layers) for scale."""
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4))
    for ax, view in zip(axs, VIEWS):
        ax.axhline(0, color=INK2, lw=1)
        for mk, spec in MODELS.items():
            nr = spec["nr"]
            xs, ys = [], []
            for ck in CKS:
                if (mk, nr, ck) in U and (mk, 1, ck) in U:
                    g = val(U[(mk, nr, ck)], view, metric) - val(U[(mk, 1, ck)], view, metric)
                    xs.append(EPOCH[ck]); ys.append(-g if metric == "ce" else g)
            ax.plot(xs, ys, color=MCOL[mk], marker="o", label=f"{spec['short']}: depth {nr} minus depth 1")
        xs, ys = [], []
        for ck in CKS:
            if ("b10c", ck) in V and ("b10c11", ck) in V:
                g = val(V[("b10c", ck)], view, metric) - val(V[("b10c11", ck)], view, metric)
                xs.append(EPOCH[ck]); ys.append(-g if metric == "ce" else g)
        ax.plot(xs, ys, color=MCOL["b10c"], marker="s", ls=(0, (5, 3)), label="vanilla 29 minus 11 layers")
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("epoch")
        ax.set_title(VIEWS[view])
        if metric == "top1":
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{100 * y:+.2f}"))
            ax.set_ylabel("top-1 gain (pp)")
        else:
            ax.set_ylabel("CE gain (nats)")
    axs[0].legend(loc="best", fontsize=11)
    fig.tight_layout(w_pad=2.5)
    save(fig, f"gain_over_training_{metric}.png")


def fig_iso_compute(U, V, ck=CKS[-1], metric="top1"):
    """Final epoch: accuracy against layer applications per token. MoR at depth d costs 2 + 9d."""
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
    for ax, view in zip(axs, VIEWS):
        for mk, spec in MODELS.items():
            ds = [d for d in spec["depths"] if (mk, d, ck) in U]
            xs = [LAYERS(d) for d in ds]
            ys = [val(U[(mk, d, ck)], view, metric) for d in ds]
            tr = [d <= spec["nr"] for d in ds]
            ax.plot(xs, ys, color=MCOL[mk], lw=1.8, zorder=3, label=spec["name"])
            ax.scatter([x for x, t in zip(xs, tr) if t], [y for y, t in zip(ys, tr) if t], color=MCOL[mk],
                       s=40, zorder=4)
            ax.scatter([x for x, t in zip(xs, tr) if not t], [y for y, t in zip(ys, tr) if not t],
                       facecolor=SURF, edgecolor=MCOL[mk], s=40, lw=1.8, zorder=4)
        for vk, spec in VAN.items():
            if (vk, ck) in V:
                ax.scatter([spec["layers"]], [val(V[(vk, ck)], view, metric)], marker="D", s=70,
                           color=MCOL[vk], zorder=5, label=spec["name"])
        ax.set_xticks([LAYERS(d) for d in range(1, 8)])
        ax.set_xlabel("layer applications per token")
        ax.set_title(VIEWS[view])
        if metric == "top1":
            pct(ax); ax.set_ylabel("top-1 accuracy")
        else:
            ax.invert_yaxis(); ax.set_ylabel("cross-entropy (nats/token)")
    axs[0].legend(loc="lower right", fontsize=11)
    fig.tight_layout(w_pad=2.5)
    save(fig, f"iso_compute_{metric}_ep{EPOCH[ck]}.png")


def fig_cells(U, V, mk, T, metric="top1"):
    """Every source -> T (one-to-one) and all others -> T (all-to-one), one line per epoch."""
    srcs = [s for s in SOURCES if s != T and not {s, T} <= {"S1GRD", "S1RTC"}] + ["all"]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.2))
    for ax, s in zip(axs.flat, srcs):
        if s == "all":
            g = lambda u: u["a2o"][T][metric]
            title = f"all others → {NICE[T]}"
        else:
            g = lambda u, k=f"{s}->{T}": u["o2o"][k][metric]
            title = f"{NICE[s]} → {NICE[T]}"
        sweep_panel(ax, U, V, mk, g, title, "top-1 accuracy" if metric == "top1" else "CE",
                    legend_vanilla=False, label_extra=False)
        if metric == "top1":
            pct(ax)
        else:
            ax.invert_yaxis()
        ax.set_xlabel("depth")
    for ax in axs.flat[len(srcs):]:
        ax.axis("off")
    hs = [plt.Line2D([], [], color=RAMP[i], marker="o", lw=2) for i in range(len(CKS))]
    hs += [plt.Line2D([], [], color=MCOL[vk], ls=(0, (5, 3)), lw=2) for vk in VAN]
    labs = [f"epoch {i + 1}" for i in range(len(CKS))] + [f"{VAN[vk]['name']}, epoch 10" for vk in VAN]
    fig.legend(hs, labs, loc="upper center", ncol=6, fontsize=12, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, f"cells_{metric}_{mk}_{T}.png")


def fig_cell_heatmap(U, V, mk, ck=CKS[-1], metric="top1"):
    """Source x target: top-1 at the deepest trained depth minus depth 1 (one-to-one cells and the
    all-to-one column), final epoch."""
    nr = MODELS[mk]["nr"]
    if (mk, nr, ck) not in U or (mk, 1, ck) not in U:
        return
    hi, lo = U[(mk, nr, ck)], U[(mk, 1, ck)]
    rows = SOURCES + ["all"]
    M = np.full((len(rows), len(TARGETS)), np.nan)
    for j, T in enumerate(TARGETS):
        for i, s in enumerate(rows):
            if s == "all":
                M[i, j] = hi["a2o"][T][metric] - lo["a2o"][T][metric]
            elif f"{s}->{T}" in hi["o2o"]:
                M[i, j] = hi["o2o"][f"{s}->{T}"][metric] - lo["o2o"][f"{s}->{T}"][metric]
    if metric == "ce":
        M = -M
    lim = np.nanmax(np.abs(M))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#c23b3b", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(False)
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                txt = f"{100 * M[i, j]:+.2f}" if metric == "top1" else f"{M[i, j]:+.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=11, color=INK)
    ax.set_xticks(range(len(TARGETS)), [NICE[t] for t in TARGETS])
    ax.set_yticks(range(len(rows)), [NICE[s] for s in rows])
    ax.set_xlabel("target"); ax.set_ylabel("source")
    ax.axhline(len(SOURCES) - 0.5, color=INK2, lw=1.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(f"depth {nr} minus depth 1, " + ("top-1 (pp)" if metric == "top1" else "CE gain (nats)"))
    if metric == "top1":
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{100 * y:+.1f}"))
    ax.set_title(f"{MODELS[mk]['name']}, epoch {EPOCH[ck]}")
    fig.tight_layout()
    save(fig, f"cell_gain_{metric}_{mk}.png")


def fig_train_curves():
    """Held-out loss as logged by the trainer (random depth for MoR), per epoch."""
    fig, ax = plt.subplots(figsize=(10.5, 6))
    runs = {"a10cr": MODELS["a10cr"]["run"], "a10cr5": MODELS["a10cr5"]["run"],
            "b10c11": VAN["b10c11"]["run"], "b10c": VAN["b10c"]["run"]}
    names = {"a10cr": MODELS["a10cr"]["name"], "a10cr5": MODELS["a10cr5"]["name"],
             "b10c11": VAN["b10c11"]["name"], "b10c": VAN["b10c"]["name"]}
    for k, run in runs.items():
        if not run:
            continue
        f = RUNS / run / "trainer_state.json"
        if not f.exists():
            cands = sorted((RUNS / run).glob("checkpoint-*/trainer_state.json"),
                           key=lambda p: int(p.parent.name.split("-")[1]))
            if not cands:
                continue
            f = cands[-1]
        pts = sorted({e["step"]: e["eval_loss"] for e in jl(f)["log_history"] if "eval_loss" in e}.items())
        pts = [(s, l) for s, l in pts if s > 0]
        ax.plot([s / 330 for s, _ in pts], [l for _, l in pts], color=MCOL[k], marker="o", ms=4, label=names[k])
    ax.set_xlabel("epoch"); ax.set_ylabel("held-out cross-entropy (nats/token)")
    ax.set_xticks(range(0, 11))
    ax.legend(fontsize=12)
    fig.tight_layout()
    save(fig, "train_curves.png")


def fig_schematic():
    """The test: the shared block applied d times (every token alike), against two vanillas."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.2); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, fc, txt, ec=None, fs=13, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec or fc, lw=1.5))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc)

    # MoR: first layer, the shared block d times, last layer
    ax.text(0.2, 5.75, "MoR, random router", fontsize=16, fontweight="bold", color=MCOL["a10cr5"])
    box(0.2, 4.35, 1.3, 0.9, "#d9d8d2", "layer 1")
    for i in range(3):
        box(1.8 + i * 1.95, 4.35, 1.75, 0.9, "#cde2fb", f"shared block\n(9 layers)", ec=MCOL["a10cr5"], fs=12)
    ax.text(7.75, 4.8, "× d", fontsize=20, color=MCOL["a10cr5"], va="center", fontweight="bold")
    box(8.55, 4.35, 1.3, 0.9, "#d9d8d2", "last layer")
    ax.annotate("", xy=(1.8, 4.1), xytext=(7.65, 4.1),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text(4.7, 3.75, "the SAME weights, applied d times", ha="center", fontsize=12, color=INK2)
    ax.text(10.2, 5.05, "training: every token draws its own d at random\n(d = 1..3; the new model d = 1..5)",
            fontsize=13, color=INK, va="center")
    ax.text(10.2, 4.25, "test: every token gets the same d = 1, 2, 3, …", fontsize=13, color=INK,
            va="center", fontweight="bold")
    ax.text(10.2, 3.6, "cost: 2 + 9d layers per token", fontsize=12, color=INK2, va="center")

    # the two vanillas
    ax.text(0.2, 2.65, "baselines (no recursion, trained the same way)", fontsize=16, fontweight="bold", color=INK)
    box(0.2, 1.35, 2.4, 0.85, "#e3dff7", "vanilla, 11 layers", ec=MCOL["b10c11"])
    ax.text(2.8, 1.78, "same parameters as MoR;\nsame cost as d = 1", fontsize=12, color=INK2, va="center")
    box(6.9, 1.35, 6.3, 0.85, "#fde1d3", "vanilla, 29 layers", ec=MCOL["b10c"])
    ax.text(13.4, 1.78, "2.6× the parameters;\nsame cost as d = 3", fontsize=12, color=INK2, va="center")
    ax.text(0.2, 0.45, "All: 10 epochs on TerraMesh (cleaned tokens), the same data, recipe and seed; "
            "a checkpoint at every epoch.", fontsize=12, color=INK2)
    save(fig, "schematic.png")


# ------------------------------------------------------------------ summary

def summary(U, V):
    rows, doc = [], {"units": {}, "vanilla": {}, "contrasts": {}}
    for (mk, d, ck), u in sorted(U.items()):
        r = {"model": mk, "depth": d, "trained_depth": d <= MODELS[mk]["nr"], "epoch": EPOCH[ck],
             "layers_per_token": LAYERS(d)}
        for view in VIEWS:
            for met in ("top1", "ce"):
                r[f"{view}_{met}"] = val(u, view, met)
        rows.append(r)
        doc["units"][f"{mk}|{d}|{ck}"] = {k: r[k] for k in r if k not in ("model", "depth", "epoch")}
    for (vk, ck), u in sorted(V.items()):
        r = {"model": vk, "depth": "", "trained_depth": "", "epoch": EPOCH[ck],
             "layers_per_token": VAN[vk]["layers"] if vk in VAN else 20}
        for view in VIEWS:
            for met in ("top1", "ce"):
                r[f"{view}_{met}"] = val(u, view, met)
        rows.append(r)
        doc["vanilla"][f"{vk}|{ck}"] = r
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # Paired contrasts at the final epoch (row bootstrap, 95%): depth Nr vs 1, best vs the vanillas.
    ck = CKS[-1]
    for mk, spec in MODELS.items():
        nr = spec["nr"]
        pairs = [(f"{mk}_force1", f"{mk}_force{nr}", f"{mk}: depth {nr} - depth 1")]
        pairs += [("b10c11", f"{mk}_force{d}", f"{mk}: depth {d} - vanilla 11 layers") for d in (1, nr)]
        pairs += [("b10c", f"{mk}_force{d}", f"{mk}: depth {d} - vanilla 29 layers") for d in (1, nr)]
        for a, b, name in pairs:
            ok = all((ARM / t / f"checkpoint-{ck}" / "tokens_router.npz").exists() and
                     (ARM / t / f"checkpoint-{ck}" / "grid_rows.npz").exists() for t in (a, b))
            if not ok:
                continue
            doc["contrasts"][name] = {"epoch": EPOCH[ck], "any_top1": boot_any(a, b, ck),
                                      "o2o_top1": boot_o2o(a, b, ck)}
    (OUT / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--a10cr5-run", default=None, help="run id of A10cR5 (for its training curve)")
    args = ap.parse_args()
    if args.a10cr5_run:
        MODELS["a10cr5"]["run"] = args.a10cr5_run
    U, V = load_all()
    print(f"units: {len(U)} forced, {len(V)} reference")
    for mk in MODELS:
        have = sorted({(d, EPOCH[ck]) for (m, d, ck) in U if m == mk})
        print(f"  {mk}: {len(have)} of {len(MODELS[mk]['depths']) * len(CKS)}")
    want = lambda n: args.only is None or n in args.only
    if want("summary") and U:
        doc = summary(U, V)
        for k, v in doc["contrasts"].items():
            print(f"  {k:40s} any {100 * v['any_top1'][0]:+.3f} pp {[round(100 * x, 3) for x in v['any_top1'][1]]}"
                  f"   o2o {100 * v['o2o_top1'][0]:+.3f} pp {[round(100 * x, 3) for x in v['o2o_top1'][1]]}")
    for mk in MODELS:
        if not any(m == mk for (m, _, _) in U):
            continue
        for met in ("top1", "ce"):
            if want("sweep"):
                fig_sweep_views(U, V, mk, met)
                for view in VIEWS:
                    fig_sweep_single(U, V, mk, view, met)
            if want("delta"):
                fig_delta_vanilla(U, V, mk, "b10c11", met)
                fig_delta_vanilla(U, V, mk, "b10c", met)
            if want("cells"):
                for T in TARGETS:
                    fig_cells(U, V, mk, T, met)
            if want("heatmap"):
                fig_cell_heatmap(U, V, mk, metric=met)
    for met in ("top1", "ce"):
        if want("gain"):
            fig_gain_over_training(U, V, met)
        if want("iso"):
            ck = max((c for (_, _, c) in U), default=None)
            if ck:
                fig_iso_compute(U, V, ck, met)
    if want("train"):
        fig_train_curves()
    if want("schematic"):
        fig_schematic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
