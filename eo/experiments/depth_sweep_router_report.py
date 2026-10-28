"""Depth sweep with the LEARNED router (2026-10-28): does each modality do best at the depth its router
assigns? `.venv`, CPU.

The depth-sweep analysis (depth_sweep_report.py, random router) on the two learned-router seeds, A10c
(seed 42) and A10c-s2 (seed 43), at every epoch checkpoint (eval_checkpoint.py --force-depth d):

  <seed>            as routed (the October units): any-to-any, target-last, grid; at epochs 1, 3, 6,
                    9, 10 any-to-any also forced to d = 1..3 (modes.force<d>, the same DepthOverride
                    path as below; reproduced on the TITAN V by a10c_force3_repro to <= 0.01 pp)
  <seed>_any<d>     any-to-any forced to d = 1..3 at the other epochs (2, 4, 5, 7, 8)
  <seed>_routed     as routed, target-last with per-row arrays (the pairing baseline for all-to-one)
  <seed>_force<d>   every position at d = 1..5 (4, 5 above the trained 3); target-last + grid, and the
                    any-to-any pass for d = 4, 5
  <seed>_tforce<d>  only the positions predicting the TARGET at d, the context as routed: target-last
                    (and the grid at the final epoch). The clean test of "does the target do best at
                    the depth its router assigns it"

The gate on the recursed update stays the router's own top probability under forcing, so only the
number of passes moves. Rows are identical across all passes of a view, so every contrast is paired
over rows (2,000 bootstrap draws).

Every number is read from stored results; nothing runs a model.
    .venv/bin/python eo/experiments/depth_sweep_router_report.py [--only NAME ...]
Writes /data/enric/reports/depth_sweep_router/{summary.json, figs/*.png}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_sweep_report as R  # noqa: E402

plt, matplotlib = R.plt, R.matplotlib
ARM, CKS, EPOCH = R.ARM, R.CKS, R.EPOCH
TARGETS, SOURCES, NICE, VIEWS = R.TARGETS, R.SOURCES, R.NICE, R.VIEWS
OUT = Path("/data/enric/reports/depth_sweep_router")
R.FIGS = OUT / "figs"                                      # R.save writes here
DEPTHS, NR = [1, 2, 3, 4, 5], 3
SEEDS = {"a10c": dict(short="seed 42", color="#008300", run="mor10c_20261012_013533"),
         "a10cs2": dict(short="seed 43", color="#d55181", run="mor10cs2_20261012_181812")}
# Routed depth as a sequential magnitude (1..3): one hue, light -> dark (orange: blue is the epochs').
DEPTH_RAMP = ["#fde6da", "#f5b08f", "#eb6834", "#b8461b", "#7d2c0c"]
O2O = [f"{s}->{t}" for t in TARGETS for s in SOURCES if s != t]


# ------------------------------------------------------------------ loading

def _views(main, tl, cells):
    """The three views from a main-pass mode dict, target_last and grid cells (any may be None)."""
    u = {}
    if main is not None:
        pm = main["per_modality"]
        n = sum(v["n_tokens"] for v in pm.values())
        u["any"] = {"top1": sum(v["top1"] * v["n_tokens"] for v in pm.values()) / n,
                    "ce": main["ce_all_body_tokens"],
                    "per_mod": {k: {"top1": v["top1"], "ce": v["ce"]} for k, v in pm.items()}}
    if tl is not None:
        u["a2o"] = {T: {"top1": tl[T]["top1"], "ce": tl[T]["ce"], "depth": tl[T].get("mean_depth_target")}
                    for T in tl}
        u["a2o_pooled"] = {k: float(np.mean([u["a2o"][T][k] for T in TARGETS])) for k in ("top1", "ce")}
    if cells is not None and all(k in cells for k in O2O):
        u["o2o"] = {k: {"top1": cells[k]["top1"], "ce": cells[k]["ce_row_mean"], "depth": cells[k].get("d_pred_mean")}
                    for k in O2O if not cells[k].get("hole")}
        u["o2o_pooled"] = {k: float(np.mean([v[k] for v in u["o2o"].values()])) for k in ("top1", "ce")}
    return u


def _doc(tag, ck):
    d = ARM / tag / f"checkpoint-{ck}"
    m = R.jl(d / "metrics.json") if (d / "metrics.json").exists() else None
    g = R.jl(d / "grid.json")["cells"] if (d / "grid.json").exists() else None
    return m, g


def load():
    """U[(seed, kind, d, ck)]: kind 'routed' (d None), 'all' or 'target'. Partial units load partially."""
    U = {}
    for s in SEEDS:
        for ck in CKS:
            m, g = _doc(s, ck)
            if m is None:
                continue
            U[(s, "routed", None, ck)] = _views(m["modes"]["router"], m.get("target_last"), g)
            for d in DEPTHS:
                fm, fg = _doc(f"{s}_force{d}", ck)
                if d > NR:
                    main = (fm or {}).get("modes", {}).get("router")
                elif f"force{d}" in m["modes"]:
                    main = m["modes"][f"force{d}"]
                else:
                    am, _ = _doc(f"{s}_any{d}", ck)
                    if am is not None and am.get("force_depth") != d:
                        raise SystemExit(f"{s}_any{d}/checkpoint-{ck} records force_depth {am.get('force_depth')}")
                    main = (am or {}).get("modes", {}).get("router")
                if fm is not None and fm.get("force_depth") != d:
                    raise SystemExit(f"{s}_force{d}/checkpoint-{ck} records force_depth {fm.get('force_depth')}")
                if fm is not None and fm.get("force_scope", "all") != "all":
                    raise SystemExit(f"{s}_force{d}/checkpoint-{ck} records scope {fm.get('force_scope')}")
                u = _views(main, (fm or {}).get("target_last"), fg)
                if u:
                    U[(s, "all", d, ck)] = u
                tm, tg = _doc(f"{s}_tforce{d}", ck)
                if tm is not None and "target_last" in tm:
                    if (tm.get("force_depth"), tm.get("force_scope")) != (d, "target"):
                        raise SystemExit(f"{s}_tforce{d}/checkpoint-{ck}: {tm.get('force_depth')}, {tm.get('force_scope')}")
                    U[(s, "target", d, ck)] = _views(None, tm["target_last"], tg)
    return U


def routed_depth(s, ck, view="a2o"):
    """Mean routed depth per target at its predicting positions: target-last (stored per target) or
    any-to-any (from the per-token arrays, by predicted modality)."""
    if view == "a2o":
        tl = R.jl(ARM / s / f"checkpoint-{ck}" / "metrics.json")["target_last"]
        return {T: tl[T]["mean_depth_target"] for T in TARGETS}
    z = np.load(ARM / s / f"checkpoint-{ck}" / "tokens_router.npz")
    from eo.data.eo_vocab import MODALITY_TO_ID
    return {T: float(z["depth"][z["label_mod"] == MODALITY_TO_ID[T]].mean()) for T in TARGETS}


def routed_mix(s, ck):
    """Share of each target's predicting positions at depth 1, 2, 3 (target-last, <seed>_routed rows)."""
    p = ARM / f"{s}_routed" / f"checkpoint-{ck}" / "target_last_rows.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {T: (lambda c: c / c.sum())(z[f"{T}|d_pred"].sum(0).astype(float)) for T in TARGETS}


def mean_depth_any(s, ck):
    """Mean routed depth over every scored any-to-any token (the x of 'as routed' in the sweeps)."""
    z = np.load(ARM / s / f"checkpoint-{ck}" / "tokens_router.npz")
    return float(z["depth"][z["label_mod"] > 0].mean())


# ------------------------------------------------------------------ paired bootstraps (rows)

def _any_rows(s, kind, d, ck):
    """Per-row (hits, tokens) of an any-to-any pass; forced d <= 3 lives in the routed unit's
    tokens_force<d>.npz, scored at the routed pass's label positions (the same loader and rows)."""
    base = np.load(ARM / s / f"checkpoint-{ck}" / "tokens_router.npz")
    v = base["label_mod"] > 0
    if kind == "routed":
        z = base
    elif (ARM / s / f"checkpoint-{ck}" / f"tokens_force{d}.npz").exists():
        z = np.load(ARM / s / f"checkpoint-{ck}" / f"tokens_force{d}.npz")
    else:
        f = ARM / (f"{s}_force{d}" if d > NR else f"{s}_any{d}") / f"checkpoint-{ck}" / "tokens_router.npz"
        if not f.exists():
            return None
        z = np.load(f)
        if not np.array_equal(z["label_mod"] > 0, v):
            raise SystemExit(f"{s} forced {d} {ck}: scored tokens differ from the routed pass")
    return (z["correct"] & v).sum(1).astype(float), v.sum(1).astype(float)


def boot_any(s, a, b, ck, n_boot=2000, seed=0):
    """top-1(b) - top-1(a), any-to-any; a, b = (kind, d)."""
    ra, rb = _any_rows(s, *a, ck), _any_rows(s, *b, ck)
    if ra is None or rb is None:
        return None
    (ha, n), (hb, n2) = ra, rb
    assert np.array_equal(n, n2)
    idx = np.random.default_rng(seed).integers(0, len(n), (n_boot, len(n)))
    diff = (hb[idx].sum(1) - ha[idx].sum(1)) / n[idx].sum(1)
    return float((hb.sum() - ha.sum()) / n.sum()), [float(x) for x in np.percentile(diff, [2.5, 97.5])]


def _tl_rows(s, kind, d, ck):
    tag = {"routed": f"{s}_routed", "all": f"{s}_force{d}", "target": f"{s}_tforce{d}"}[kind]
    p = ARM / tag / f"checkpoint-{ck}" / "target_last_rows.npz"
    return np.load(p) if p.exists() else None


def boot_a2o(s, a, b, ck, T=None, metric="top1", n_boot=2000, seed=0):
    """Gain of b over a (top-1 b - a; CE a - b), all-to-one: one target, or pooled as the mean of the
    six targets (rows resampled within each target)."""
    za, zb = _tl_rows(s, *a, ck), _tl_rows(s, *b, ck)
    if za is None or zb is None:
        return None
    rng = np.random.default_rng(seed)
    pts, boots = [], []
    for t in ([T] if T else TARGETS):
        p, bt = R._cell_boot(za, zb, t, rng, n_boot, metric)
        pts.append(p); boots.append(bt)
    return float(np.mean(pts)), [float(x) for x in np.percentile(np.mean(boots, 0), [2.5, 97.5])]


def boot_o2o(s, a, b, ck, n_boot=2000, seed=0):
    tag = lambda k, d: s if k == "routed" else (f"{s}_force{d}" if k == "all" else f"{s}_tforce{d}")
    pa = ARM / tag(*a) / f"checkpoint-{ck}" / "grid_rows.npz"
    pb = ARM / tag(*b) / f"checkpoint-{ck}" / "grid_rows.npz"
    if not (pa.exists() and pb.exists()):
        return None
    return R.boot_o2o(tag(*a), tag(*b), ck, n_boot, seed)


def best_target_depth(U, s, T, ck, metric="top1", kind="target"):
    """The forced depth with the best all-to-one score for target T (top-1 highest, CE lowest)."""
    ds = [d for d in DEPTHS if (s, kind, d, ck) in U and "a2o" in U[(s, kind, d, ck)]]
    if not ds:
        return None
    sc = lambda d: U[(s, kind, d, ck)]["a2o"][T][metric] * (1 if metric == "top1" else -1)
    return max(ds, key=sc)


# ------------------------------------------------------------------ figures

def seed_title(ax, s, text=None):
    """The panel's seed: ink text, its colour on a marker beside it (text never wears the series colour)."""
    ax.set_title("   " + (text or SEEDS[s]["short"]))
    ax.plot([0.012], [1.045], marker="o", ms=11, color=SEEDS[s]["color"], transform=ax.transAxes, clip_on=False)


def fig_schematic():
    """What is forced: an all-to-one row (context, then the target) as routed, with every position at d,
    and with only the target's tokens at d. The gate stays the router's own."""
    from matplotlib.patches import FancyBboxPatch
    MODC = {"S2": "#d98a1c", "S1": "#3b6ea5", "DEM": "#a3622a", "NDVI": "#6a9a1f", "LULC": "#b0499a",
            "loc": "#6b6b6b"}
    seq = ["S2", "S1", "DEM", "NDVI", "loc", "LULC"]
    routed = {"S2": "1 or 3", "S1": "3", "DEM": "2", "NDVI": "1", "loc": "1 or 3", "LULC": "2"}   # measured
    rows = [("as routed", lambda m: routed[m], "the router picks 1, 2 or 3 passes per token (seed 42, epoch 10)"),
            ("every position at d", lambda m: "d", "the same test as with the random router"),
            ("target only at d", lambda m: "d" if m == "LULC" else routed[m],
             "the context stays as routed: only the target's depth moves")]
    fig, ax = plt.subplots(figsize=(16, 4.9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4.9); ax.axis("off"); ax.grid(False)
    for i, (name, lab, note) in enumerate(rows):
        y = 3.85 - i * 1.4
        ax.text(0.1, y + 0.36, name, fontsize=16, fontweight="bold", va="center")
        for j, m in enumerate(seq):
            x = 3.3 + j * 1.25
            tgt = j == len(seq) - 1
            forced = lab(m) == "d"
            ax.add_patch(FancyBboxPatch((x, y), 1.1, 0.72, boxstyle="round,pad=0.01,rounding_size=0.08",
                                        fc=MODC[m], ec=R.INK if tgt else MODC[m], lw=2.6 if tgt else 1,
                                        alpha=1.0 if tgt else 0.6))
            ax.text(x + 0.55, y + 0.47, m, ha="center", va="center", fontsize=12, color="white", fontweight="bold")
            ax.text(x + 0.55, y + 0.17, lab(m), ha="center", va="center", fontsize=12, color="white",
                    fontweight="bold" if forced else "normal")
        ax.text(11.0, y + 0.36, note, fontsize=13, color=R.INK2, va="center")
    ax.text(3.3, 0.12, "all-to-one, target LULC (outlined, scored) · numbers: passes of the shared block · "
            "the gate on each pass stays the router's own: only the number of passes changes",
            fontsize=12, color=R.MUTED)
    R.save(fig, "schematic.png")


def fig_routing(U):
    """What each router assigns: the mean routed depth at each target's predicting positions (all-to-one),
    per epoch, one heatmap per seed."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("depth", DEPTH_RAMP[:4])
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.4))
    for ax, s in zip(axs, SEEDS):
        M = np.array([[routed_depth(s, ck)[T] for ck in CKS] for T in TARGETS])
        ax.grid(False)
        im = ax.imshow(M, cmap=cmap, vmin=1, vmax=3, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=13,
                        color="white" if M[i, j] > 2.3 else R.INK)
        ax.set_xticks(range(len(CKS)), [str(EPOCH[c]) for c in CKS])
        ax.set_yticks(range(len(TARGETS)), [NICE[t] for t in TARGETS])
        ax.set_xlabel("epoch")
        seed_title(ax, s)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02, ticks=[1, 2, 3])
    cb.set_label("routed depth (mean passes)")
    R.save(fig, "routing_depth.png")


def _sweep(ax, U, s, view, metric="top1", vanilla=None):
    """Accuracy against forced depth (every position), one line per epoch; as routed: a hollow marker in
    the epoch's colour at the mean routed depth; vanilla (epoch 10) dashed."""
    R.shade_extrapolation(ax, NR, DEPTHS[-1])
    get = lambda u: R.val(u, view, metric) if view != "any" else u["any"][metric]
    for i, ck in enumerate(CKS):
        xs = [d for d in DEPTHS if (s, "all", d, ck) in U and (view == "any" and "any" in U[(s, "all", d, ck)]
                                                               or view != "any" and f"{view}_pooled" in U[(s, "all", d, ck)])]
        if xs:
            ax.plot(xs, [get(U[(s, "all", d, ck)]) for d in xs], color=R.RAMP[i], marker="o", ms=5, lw=1.8, zorder=3)
        r = U.get((s, "routed", None, ck))
        if r is not None and (view == "any" or f"{view}_pooled" in r):
            x = mean_depth_any(s, ck) if view == "any" else float(np.mean(list(routed_depth(s, ck).values())))
            ax.scatter([x], [get(r)], s=70, facecolor=R.SURF, edgecolor=R.RAMP[i], lw=2, zorder=4)
    if vanilla:
        for vk in R.VAN:
            if (vk, CKS[-1]) in vanilla:
                ax.axhline(R.val(vanilla[(vk, CKS[-1])], view, metric), color=R.MCOL[vk], ls=(0, (5, 3)), lw=1.8, zorder=2)
    ax.set_xticks(DEPTHS)
    ax.set_xlim(0.6, 5.4)
    ax.set_xlabel("forced depth (passes of the shared block)")


def fig_sweep(U, V, view, metric="top1"):
    fig, axs = plt.subplots(1, 2, figsize=(17, 6.6), sharey=True)
    for ax, s in zip(axs, SEEDS):
        _sweep(ax, U, s, view, metric, V)
        seed_title(ax, s)
        R.pct(ax) if metric == "top1" else ax.invert_yaxis()
    axs[0].set_ylabel("top-1 accuracy" if metric == "top1" else "cross-entropy (nats/token)")
    hs, labs = R.epoch_handles(len(CKS))
    hs.insert(len(CKS), plt.Line2D([], [], ls="", marker="o", ms=9, mfc=R.SURF, mec=R.INK2, mew=2))
    labs.insert(len(CKS), "as routed (at its mean depth)")
    fig.legend(hs, labs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=7, fontsize=14, handlelength=1.6,
               columnspacing=1.0)
    fig.tight_layout(w_pad=2.0)
    R.save(fig, f"sweep_{view}_{metric}.png")


def fig_delta_routed(U, metric="top1"):
    """Each forced depth (every position) minus as routed, same epoch: above 0, a fixed depth beats the router."""
    fig, axs = plt.subplots(2, 3, figsize=(19, 9.6), sharex=True)
    for r, s in enumerate(SEEDS):
        for c, view in enumerate(VIEWS):
            ax = axs[r, c]
            R.shade_extrapolation(ax, NR, DEPTHS[-1])
            ax.axhline(0, color=SEEDS[s]["color"], lw=1.8, ls=(0, (5, 3)), zorder=2)
            for i, ck in enumerate(CKS):
                ru = U.get((s, "routed", None, ck))
                if ru is None:
                    continue
                key = "any" if view == "any" else f"{view}_pooled"
                xs = [d for d in DEPTHS if key in U.get((s, "all", d, ck), {}) and key in ru]
                g = lambda u: u["any"][metric] if view == "any" else u[key][metric]
                ys = [g(U[(s, "all", d, ck)]) - g(ru) for d in xs]
                ys = [-y for y in ys] if metric == "ce" else [100 * y for y in ys]
                ax.plot(xs, ys, color=R.RAMP[i], marker="o", ms=5, lw=1.8, zorder=3)
            ax.set_xticks(DEPTHS)
            R.signed(ax)
            if r == 0:
                ax.set_title(VIEWS[view])
            if r == 1:
                ax.set_xlabel("forced depth")
            if c == 0:
                ax.set_ylabel(f"{SEEDS[s]['short']}\n" + ("top-1 minus as routed (pp)" if metric == "top1"
                                                          else "CE gain over as routed (nats)"))
    hs, labs = R.epoch_handles(len(CKS), vanilla=False)
    hs.insert(len(CKS), plt.Line2D([], [], color=R.INK2, ls=(0, (5, 3)), lw=1.8))
    labs.insert(len(CKS), "as routed (0)")
    fig.legend(hs, labs, loc="lower center", ncol=7, fontsize=14, bbox_to_anchor=(0.5, 1.0), handlelength=1.6)
    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    R.save(fig, f"delta_routed_{metric}.png")


def fig_targets(U, ck=CKS[-1], metric="top1", kind="target"):
    """THE TEST, per target: all-to-one with only the target's depth forced (context as routed), minus as
    routed, with paired 95% bands; the hollow marker is where the seed's router puts that target."""
    fig, axs = plt.subplots(2, 3, figsize=(19, 9.4), sharex=True)
    for ax, T in zip(axs.ravel(), TARGETS):
        R.shade_extrapolation(ax, NR, DEPTHS[-1])
        ax.axhline(0, color=R.INK2, lw=1.0, zorder=2)
        for s, spec in SEEDS.items():
            xs, ys, lo, hi = [], [], [], []
            for d in DEPTHS:
                b = boot_a2o(s, ("routed", None), (kind, d), ck, T, metric)
                if b is None:
                    continue
                k = 100 if metric == "top1" else 1
                xs.append(d); ys.append(k * b[0]); lo.append(k * b[1][0]); hi.append(k * b[1][1])
            if not xs:
                continue
            ax.fill_between(xs, lo, hi, color=spec["color"], alpha=0.13, lw=0, zorder=2)
            ax.plot(xs, ys, color=spec["color"], marker="o", ms=6, lw=2, zorder=3)
            rd = routed_depth(s, ck)[T]
            ax.scatter([rd], [0], s=110, facecolor=R.SURF, edgecolor=spec["color"], lw=2.4, zorder=5)
            bd = xs[int(np.argmax(ys))]
            ax.scatter([bd], [max(ys)], s=150, marker="*", color=spec["color"], zorder=6, ec=R.SURF, lw=0.8)
        ax.set_title(f"→ {NICE[T]}")
        ax.set_xticks(DEPTHS)
        R.signed(ax)
    for ax in axs[1]:
        ax.set_xlabel("depth of the target's tokens")
    for ax in axs[:, 0]:
        ax.set_ylabel("top-1 minus as routed (pp)" if metric == "top1" else "CE gain over as routed (nats)")
    hs = [plt.Line2D([], [], color=sp["color"], marker="o", lw=2) for sp in SEEDS.values()]
    labs = [sp["short"] for sp in SEEDS.values()]
    hs += [plt.Line2D([], [], ls="", marker="o", ms=10, mfc=R.SURF, mec=R.INK2, mew=2.2),
           plt.Line2D([], [], ls="", marker="*", ms=13, color=R.INK2), matplotlib.patches.Patch(color=R.BAND)]
    labs += ["router's depth for this target (mean)", "best forced depth", "not trained at this depth"]
    fig.legend(hs, labs, loc="lower center", ncol=5, fontsize=14, bbox_to_anchor=(0.5, 1.0), handlelength=1.6)
    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    R.save(fig, f"targets_{kind}_{metric}_ep{EPOCH[ck]}.png")


def fig_match_over_training(U, metric="top1"):
    """At every epoch, per target and seed: the best forced depth (target only) against the router's.
    Cells: best depth; outlined where the router's modal depth is the best one."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list("depth", DEPTH_RAMP)
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.4))
    for ax, s in zip(axs, SEEDS):
        M = np.full((len(TARGETS), len(CKS)), np.nan)
        hit = np.zeros(M.shape, bool)
        for j, ck in enumerate(CKS):
            mix = routed_mix(s, ck)
            for i, T in enumerate(TARGETS):
                bd = best_target_depth(U, s, T, ck, metric)
                if bd is None:
                    continue
                M[i, j] = bd
                modal = (int(np.argmax(mix[T])) + 1) if mix is not None else round(routed_depth(s, ck)[T])
                hit[i, j] = bd == modal
        ax.grid(False)
        ax.imshow(M, cmap=cmap, vmin=1, vmax=5, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=14,
                            color="white" if M[i, j] >= 3 else R.INK, fontweight="bold" if hit[i, j] else "normal")
                    if hit[i, j]:
                        ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec=R.INK, lw=2))
        ax.set_xticks(range(len(CKS)), [str(EPOCH[c]) for c in CKS])
        ax.set_yticks(range(len(TARGETS)), [NICE[t] for t in TARGETS])
        ax.set_xlabel("epoch")
        seed_title(ax, s, f"{SEEDS[s]['short']}: {int(hit.sum())} of {int(np.isfinite(M).sum())} match")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(w_pad=2.0)
    R.save(fig, f"match_over_training_{metric}.png")


def fig_pairs(U, ck=CKS[-1], metric="top1"):
    """One-to-one at the final epoch, target only at d: per source -> target pair, the best depth, outlined
    where it is the router's modal depth for that pair's target tokens (routed grid, d_pred)."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list("depth", DEPTH_RAMP)
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.2))
    for ax, s in zip(axs, SEEDS):
        cells = R.jl(ARM / s / f"checkpoint-{ck}" / "grid.json")["cells"]
        M = np.full((len(SOURCES), len(TARGETS)), np.nan)
        hit = np.zeros(M.shape, bool)
        for j, T in enumerate(TARGETS):
            for i, S in enumerate(SOURCES):
                k = f"{S}->{T}"
                sc = {d: U[(s, "target", d, ck)]["o2o"][k][metric] for d in DEPTHS
                      if "o2o" in U.get((s, "target", d, ck), {}) and k in U[(s, "target", d, ck)]["o2o"]}
                if len(sc) < len(DEPTHS):
                    continue
                M[i, j] = max(sc, key=lambda d: sc[d] * (1 if metric == "top1" else -1))
                hit[i, j] = M[i, j] == int(np.argmax(cells[k]["d_pred_mix"])) + 1
        ax.grid(False)
        ax.imshow(M, cmap=cmap, vmin=1, vmax=5, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=14,
                            color="white" if M[i, j] >= 3 else R.INK, fontweight="bold" if hit[i, j] else "normal")
                    if hit[i, j]:
                        ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec=R.INK, lw=2))
        ax.set_xticks(range(len(TARGETS)), [NICE[t] for t in TARGETS])
        ax.set_yticks(range(len(SOURCES)), [NICE[x] for x in SOURCES])
        ax.set_xlabel("target"); ax.set_ylabel("source")
        seed_title(ax, s, f"{SEEDS[s]['short']}: {int(hit.sum())} of {int(np.isfinite(M).sum())} pairs match")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(w_pad=2.0)
    R.save(fig, f"pairs_best_{metric}_ep{EPOCH[ck]}.png")


# ------------------------------------------------------------------ summary

def summary(U):
    out = {"targets": {}, "pooled": {}}
    ck = CKS[-1]
    for s in SEEDS:
        mix = routed_mix(s, ck)
        rd = routed_depth(s, ck)
        for T in TARGETS:
            row = {"routed_mean_depth": rd[T], "routed_mix": None if mix is None else mix[T].tolist()}
            for metric in ("top1", "ce"):
                row[f"best_{metric}"] = best_target_depth(U, s, T, ck, metric)
                row[f"gain_{metric}"] = {d: boot_a2o(s, ("routed", None), ("target", d), ck, T, metric)
                                         for d in DEPTHS}
            out["targets"][f"{s}|{T}"] = row
        for view in VIEWS:
            f = boot_any if view == "any" else (boot_a2o if view == "a2o" else boot_o2o)
            out["pooled"][f"{s}|{view}"] = {d: f(s, ("routed", None), ("all", d), ck) for d in DEPTHS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    U = load()
    _, V = R.load_all()
    print(f"loaded {len(U)} (seed, kind, depth, checkpoint) entries")
    want = lambda n: args.only is None or n in args.only
    if want("schematic"):
        fig_schematic()
    if want("routing"):
        fig_routing(U)
    for view in VIEWS:
        if want("sweep"):
            fig_sweep(U, V, view)
    if want("delta"):
        fig_delta_routed(U)
        fig_delta_routed(U, "ce")
    if want("targets"):
        for metric in ("top1", "ce"):
            fig_targets(U, metric=metric)
            fig_targets(U, metric=metric, kind="all")
    if want("pairs"):
        for metric in ("top1", "ce"):
            fig_pairs(U, metric=metric)
    if want("match"):
        fig_match_over_training(U)
        fig_match_over_training(U, "ce")
    if want("summary"):
        summary(U)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
