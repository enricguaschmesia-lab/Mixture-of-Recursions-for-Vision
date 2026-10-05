#!/usr/bin/env python
"""Per-checkpoint curves, D3.11 and recursion figures for one or more arms.

    python eo/scripts/arm_report.py \
        --arm arm_a=/data/enric/runs/pretrain/phase3/mor_20260923_140238 \
        [--arm arm_b=/data/enric/runs/pretrain/phase3/<vanilla run>] \
        --out /data/enric/reports/arm_report/<name>

Reads what eval_checkpoint.py and sweep_d311.sh wrote (see eo/eval/curves.py)
and writes, per arm, `curves_<arm>.csv` (one row per checkpoint: step, epoch,
both cumulative-FLOPs definitions, every accuracy and generation metric) and
`summary_<arm>.json`, plus figures shared by all arms given. With two arms the
figures overlay them in fixed colours (arm A slot 1, arm B slot 2) and
`paired_<a>_vs_<b>.json` holds per-scene paired generation differences over
the shared checkpoints -- the same scenes, so the comparison is paired.

⚠ The D3.11 reading, written before the figure (plan 10.2b.2): GO if MoR's
error curve is AT OR BELOW vanilla's on the FLOPs axis. rmse_z_ml7 is an ERROR
(lower is better); LULC mIoU is a SCORE (higher is better) -- its panel reads
inverted. Every panel's y label says which.

Runs in `.venv`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eo.eval import curves as CV  # noqa: E402

# Reference palette (dataviz skill): categorical slots in fixed order, the
# arm keeps its colour whatever else is plotted.
ARM_COLORS = ["#2a78d6", "#eb6834"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, GRID_INK, SURFACE, REF = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb", "#8a8984"
DIV_NEG, DIV_POS = "#2a78d6", "#e34948"


def style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_INK)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID_INK, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def fig_frame(n, ncols, w=3.3, h=2.6):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 0.9), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def finish(fig, path, title, subtitle, handles=None):
    H = fig.get_figheight()                  # place text in inches, not fractions
    fig.suptitle(title, x=0.01, ha="left", va="top", fontsize=13, color=INK, y=1 - 0.12 / H)
    fig.text(0.01, 1 - 0.45 / H, subtitle, fontsize=8.5, color=INK_2, ha="left", va="top")
    legend_h = 0.0
    if handles:
        ncol = min(len(handles[0]), 4)
        fig.legend(*handles, loc="lower center", ncol=ncol, frameon=False,
                   fontsize=8.5, labelcolor=INK_2)
        legend_h = 0.12 + 0.24 * int(np.ceil(len(handles[0]) / ncol))
    fig.tight_layout(rect=(0, legend_h / H, 1, 1 - 0.7 / H))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ figures

def _schedule(arm):
    """(decay start, decay end, eval cadence) in steps, read from the run's own log.

    ⚠ Not hardcoded: the 50-epoch arms decay over 14,850-16,500 and evaluate every
    500 steps, A10/B10 over 2,970-3,300 every 165. The decay starts at the last
    logged step still at peak LR (WSD: flat until the anneal)."""
    tr = [(e["step"], e["learning_rate"]) for e in arm["logged"]["train"]]
    peak = max(lr for _, lr in tr)
    start = max(s for s, lr in tr if lr >= peak * (1 - 1e-6))
    ev = sorted({e["step"] for e in arm["logged"]["eval"] if e["step"] > 0})
    cadence = int(np.median(np.diff(ev))) if len(ev) > 1 else None
    return start, tr[-1][0], cadence


def fig_training(arms, out):
    fig, axs = fig_frame(8, 4, h=2.4)
    names = ["aggregate"] + CV.MODS
    for ai, A in enumerate(arms):
        c = ARM_COLORS[ai]
        tr, ev = A["logged"]["train"], A["logged"]["eval"]
        for ax, n in zip(axs, names):
            tk = "loss" if n == "aggregate" else f"loss_{n}"
            ek = "eval_loss" if n == "aggregate" else f"eval_loss_{n}"
            x = [e["step"] / CV.EPOCH_STEPS for e in tr if tk in e]
            ax.plot(x, [e[tk] for e in tr if tk in e], color=c, lw=1, alpha=0.35)
            ex = [e["step"] / CV.EPOCH_STEPS for e in ev if ek in e and e["step"] > 0]
            ey = [e[ek] for e in ev if ek in e and e["step"] > 0]
            ax.plot(ex, ey, color=c, lw=2, marker="o", ms=3)
            i = int(np.argmin(ey))
            ax.scatter([ex[i]], [ey[i]], s=60, facecolor="none", edgecolor=INK, lw=1.2, zorder=5)
            # step 0 (~11.4 = ln V) would flatten every panel; the axis starts past warmup
            lo = min(min(e[tk] for e in tr if tk in e and e["step"] > 300), min(ey))
            ax.set_ylim(lo - 0.3, max(ey) + 0.3)
    sched = {_schedule(A) for A in arms}
    if len(sched) > 1:
        print(f"⚠ fig_training: the arms' schedules differ {sorted(sched)}; the band shows the first")
    d0, d1, cadence = _schedule(arms[0])
    for ax, n in zip(axs, names):
        style(ax, n)
        ax.axvspan(d0 / CV.EPOCH_STEPS, d1 / CV.EPOCH_STEPS, color=GRID_INK, alpha=0.35, lw=0)
        ax.set_xlabel("epoch", fontsize=8, color=INK_2)
    axs[0].set_ylabel("cross-entropy (nats)", fontsize=8, color=INK_2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK_2, lw=1, alpha=0.5), Line2D([], [], color=INK_2, lw=2, marker="o", ms=3),
         Line2D([], [], marker="o", ls="", mfc="none", mec=INK)]
    lab = ["train (logged every 10 steps)", f"held-out eval (every {cadence} steps)", "eval minimum"]
    for ai, A in enumerate(arms):
        h.append(Line2D([], [], color=ARM_COLORS[ai], lw=2)); lab.append(A["arm"])
    finish(fig, out / "fig1_training_curves.png", "Training and held-out loss per modality",
           f"grey band = LR decay (steps {d0:,}-{d1:,}). Eval rows: 4,416 geographically held-out. "
           "Lower is better. ⚠ S1GRD on 9.8% of rows; Coords is 3 tokens/scene (memorization).",
           (h, lab))


def fig_accuracy(arms, out, key="top1"):
    fig, axs = fig_frame(7, 4)
    for ai, A in enumerate(arms):
        c = ARM_COLORS[ai]
        cks = A["checkpoints"]
        for ax, n in zip(axs, CV.MODS):
            x = [r["epoch"] for r in cks if n in r["tf"]]
            ax.plot(x, [r["tf"][n][key] for r in cks if n in r["tf"]], color=c, lw=2, marker="o", ms=4)
            xs = [r["epoch"] for r in cks if n in r["tl"]]
            ax.plot(xs, [r["tl"][n][key] for r in cks if n in r["tl"]], color=c, lw=1.6, ls="--",
                    marker="s", ms=3.5)
    for ax, n in zip(axs, CV.MODS):
        style(ax, n)
        ax.set_ylim(0, None)
        ax.set_xlabel("epoch", fontsize=8, color=INK_2)
    axs[0].set_ylabel(f"next-token {key} accuracy", fontsize=8, color=INK_2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK_2, lw=2, marker="o", ms=4),
         Line2D([], [], color=INK_2, lw=1.6, ls="--", marker="s", ms=3.5)]
    lab = ["random modality order (as trained)", "target last, all others as context"]
    for ai, A in enumerate(arms):
        h.append(Line2D([], [], color=ARM_COLORS[ai], lw=2)); lab.append(A["arm"])
    finish(fig, out / f"fig2_epoch_vs_{key}_accuracy.png",
           f"Teacher-forced {key} accuracy on held-out scenes, per checkpoint",
           "Higher is better. Epoch 0 = untrained. Random-order: all 4,416 eval rows; target-last: "
           "512 corpus-stratified rows per target (S1GRD: every eval row that carries it).", (h, lab))


CONTROLS = Path("/data/enric/reports/d311/controls")


def _control(kind, T, source="generated"):
    """A D3.11 reference decoded by the controls pass, or None if not run yet."""
    hits = list((CONTROLS / kind).glob(f"{T}_*/metrics_{T}.json"))
    if not hits:
        return None
    g = json.loads(hits[0].read_text())
    return g["summary"][source].get(g["collapse_metric"]), g


def _ci(rec, T):
    """95% half-width over scenes (in-range only) of one generation point."""
    g = rec["gen"].get(T)
    if not g:
        return 0.0
    v = [s[g["metric"]] for s in g["per_scene"]["generated"] if s["in_range"] and s[g["metric"]] is not None]
    return 1.96 * float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _gen_panel(ax, arms, T, xkey):
    for ai, A in enumerate(arms):
        pts = [(r[xkey], CV.gen_value(r, T), _ci(r, T)) for r in A["checkpoints"]
               if CV.gen_value(r, T) is not None and xkey in r]
        if pts:
            x, y, e = zip(*pts)
            ax.fill_between(x, np.array(y) - e, np.array(y) + e, color=ARM_COLORS[ai], alpha=0.12, lw=0)
            ax.plot(x, y, color=ARM_COLORS[ai], lw=2, marker="o", ms=4, zorder=4)
        # the same final checkpoint regenerated with another sampling seed
        s43 = _control("seed43", T)
        last = max((r for r in A["checkpoints"] if xkey in r and T in r["gen"]),
                   key=lambda r: r["step"], default=None)
        if s43 and last and ai == 0:
            ax.scatter([last[xkey]], [s43[0]], s=46, facecolor=SURFACE, edgecolor=ARM_COLORS[ai],
                       lw=1.6, zorder=6)
    refs = [(r, s) for A in arms for r in A["checkpoints"] for s in ("ceiling", "shuffled")
            if CV.gen_value(r, T, s) is not None]
    if refs:
        ceil = np.mean([CV.gen_value(r, T, "ceiling") for r, s in refs if s == "ceiling"])
        shuf = np.mean([CV.gen_value(r, T, "shuffled") for r, s in refs if s == "shuffled"])
        ax.axhline(ceil, color=REF, lw=1.3, ls=":")
        ax.axhline(shuf, color=REF, lw=1.3, ls="--")
    ws = _control("wrong_scene", T)
    if ws:
        ax.axhline(ws[0], color=INK_2, lw=1.3, ls="-.")
    metric = "mIoU (higher is better)" if T == "LULC" else "rmse_z_ml7 (lower is better)"
    n_in = None
    for A in arms:
        for r in A["checkpoints"]:
            if T in r["gen"]:
                n_in = r["gen"][T]["summary"]["generated"]["n_in_range"]
    style(ax, f"{T}  ·  {n_in}/64 in range" if T != "LULC" else f"{T}  ·  64/64")
    ax.set_ylabel(metric, fontsize=8, color=INK_2)
    ax.set_ylim(0, None)


def fig_generation(arms, out, xkey, fname, xlabel, title):
    fig, axs = fig_frame(6, 3, w=3.6, h=2.8)
    for ax, T in zip(axs, CV.TARGETS):
        _gen_panel(ax, arms, T, xkey)
        ax.set_xlabel(xlabel, fontsize=8, color=INK_2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ARM_COLORS[i], lw=2, marker="o", ms=4) for i in range(len(arms))]
    lab = [f"{A['arm']} (band: 95% CI over scenes)" for A in arms]
    h += [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=ARM_COLORS[0], mew=1.6)]
    lab += ["final ckpt, sampling seed 43"]
    h += [Line2D([], [], color=REF, lw=1.3, ls=":"), Line2D([], [], color=REF, lw=1.3, ls="--"),
          Line2D([], [], color=INK_2, lw=1.3, ls="-.")]
    lab += ["ceiling: true tokens", "true tokens, spatially shuffled", "another scene's true tokens"]
    finish(fig, out / fname, title,
           "64 corpus-stratified held-out scenes per target, slot-masked sampling (T=1, seed 42), "
           "decode seed 0 / 50 steps. Continuous panels: scenes with |z|<1 only.", (h, lab))


def fig_routing(arm, out):
    cks = [r for r in arm["checkpoints"] if r.get("routing")]
    if not cks:
        return
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
    fig.patch.set_facecolor(SURFACE)
    x = [r["epoch"] for r in cks]
    for i, n in enumerate(CV.MODS):
        axs[0].plot(x, [r["routing"]["by_modality"][n]["mean_depth"] for r in cks],
                    color=CAT[i], lw=2, marker="o", ms=3.5, label=n)
    axs[0].set_ylim(0.9, 3.1); axs[0].set_yticks([1, 2, 3])
    style(axs[0], "mean recursion passes, per modality"); axs[0].set_ylabel("passes (1-3)", fontsize=8, color=INK_2)
    axs[0].legend(fontsize=7.5, frameon=False, ncol=2, labelcolor=INK_2)
    axs[1].plot(x, [r["routing"]["eta2_modality"] for r in cks], color=CAT[0], lw=2, marker="o", ms=3.5,
                label="modality")
    axs[1].plot(x, [r["routing"]["eta2_position"] for r in cks], color=CAT[1], lw=2, marker="o", ms=3.5,
                label="sequence position")
    axs[1].set_ylim(0, 1); style(axs[1], "share of depth variance explained by")
    axs[1].legend(fontsize=8, frameon=False, labelcolor=INK_2)
    axs[2].plot(x, [r["layers_per_body_token"] for r in cks], color=CAT[0], lw=2, marker="o", ms=3.5)
    axs[2].axhline(29, color=REF, lw=1.3, ls="--")
    axs[2].text(x[-1], 28.2, "vanilla: 29", ha="right", fontsize=8, color=INK_2)
    axs[2].set_ylim(0, 30); style(axs[2], "layer applications per body token")
    for ax in axs:
        ax.set_xlabel("epoch", fontsize=8, color=INK_2)
    finish(fig, out / f"fig5_routing_over_training_{arm['arm']}.png",
           f"How the router's allocation evolved ({arm['arm']})",
           "All 4,416 held-out rows, modality order as in the training-time eval. Epoch 0 = untrained. "
           "balancing loss coeff 0.1, alpha 1.0, 3 steps x 9 shared blocks.")


def fig_interventions(arm, out):
    rows = [r for r in arm["checkpoints"] if len(r["modes"]) > 1]
    if not rows:
        return
    modes = ["force1", "force2", "force3", "perm_modality", "perm_all", "skip"]
    rows = [r for r in rows if all(m in r["modes"] for m in modes[:-1])]
    if not rows:
        return
    fig, axs = plt.subplots(1, len(rows), figsize=(6.2 * len(rows), 3.9), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, "#f0efec", DIV_POS])
    for ax, r in zip(axs[0], rows):
        base = r["modes"]["router"]["per_modality"]
        ms = [m for m in modes if m in r["modes"]]
        M = np.array([[r["modes"][m]["per_modality"][n]["ce"] - base[n]["ce"] for m in ms] for n in CV.MODS])
        # colour scale from the depth-changing columns; skip's +2-7 nats would
        # wash them out, so it clips (its numbers are still printed)
        lim = max(0.05, float(np.abs(M[:, :5]).max()))
        ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=7.5, color=INK)
        lbl = {"force1": "all 1", "force2": "all 2", "force3": "all 3", "perm_modality": "shuffle\nin modality",
               "perm_all": "shuffle\nacross all", "skip": "0 passes\n(skip stack)"}
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([lbl[m] for m in ms], fontsize=7.5)
        ax.set_yticks(range(len(CV.MODS))); ax.set_yticklabels(CV.MODS, fontsize=8)
        md = {"router": r["modes"]["router"].get("mean_depth_body")}
        ax.set_title(f"{r['checkpoint']} (epoch {r['epoch']:.1f})\nrouter mean depth {md['router']:.2f}",
                     fontsize=9.5, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_color(GRID_INK)
    finish(fig, out / f"fig6_interventions_{arm['arm']}.png",
           "What happens to held-out loss when the router's depth choices are overridden",
           "Cell = CE(intervention) - CE(router), nats. Red = worse than the router, blue = better. "
           "Gate values unchanged; only the number of passes moves. 'shuffle' modes keep each row's compute exactly.")


def fig_depth_vs_ce(arm, out):
    r = [r for r in arm["checkpoints"] if r.get("routing")]
    if not r:
        return
    # The 50-epoch arm's figure used these three; any other run gets its first trained,
    # middle and last checkpoint (A10: 330, 1980, 3300).
    picks = [x for x in r if x["checkpoint"] in ("checkpoint-2000", "checkpoint-8000", "checkpoint-16500")]
    if len(picks) < 3:
        trained = [x for x in r if x["step"] > 0]
        picks = [trained[0], trained[len(trained) // 2], trained[-1]] if len(trained) >= 3 else trained
    fig, axs = fig_frame(7, 4, h=2.5)
    for ci, rec in enumerate(picks):
        for ax, n in zip(axs, CV.MODS):
            dv = rec["routing"]["depth_vs_ce"].get(n, {}).get("ce_by_depth", {})
            ks = sorted(int(k) for k in dv)
            if ks:
                ax.plot(ks, [dv[k] if k in dv else dv[str(k)] for k in ks], color=CAT[ci], lw=2,
                        marker="o", ms=4, label=rec["checkpoint"])
    for ax, n in zip(axs, CV.MODS):
        style(ax, n); ax.set_xticks([1, 2, 3]); ax.set_xlabel("passes assigned", fontsize=8, color=INK_2)
    axs[0].set_ylabel("held-out CE of the prediction", fontsize=8, color=INK_2)
    h, lab = axs[0].get_legend_handles_labels()
    finish(fig, out / f"fig7_depth_vs_difficulty_{arm['arm']}.png",
           "Are the tokens the router sends deeper the hard ones?",
           "Mean CE of the next-token prediction, grouped by the depth assigned to the token that made it "
           "(groups under 50 tokens omitted). Rising = deeper tokens are harder.", (h, lab))


def fig_content(arm, out):
    """Depth vs own-patch content, final checkpoint against the untrained model.

    ⚠ The untrained router is the reference, not zero: any function of the
    hidden state already varies with the input (Step 8's by_scene lesson).
    """
    cks = {r["checkpoint"]: r for r in arm["checkpoints"] if r.get("content")}
    last = max((r for r in cks.values()), key=lambda r: r["step"], default=None)
    if not last or "untrained" not in cks:
        return
    items = [(m, k) for m, ks in CV.PATCH_STATS.items() for k in ks if m in last["content"]]
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.34 * len(items) + 1.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(items))
    for ax, key, title in zip(axs, ("pooled_spearman", "within_scene"),
                              ("pooled over all tokens", "within one scene (patch-to-patch)")):
        for ci, (rec, lab) in enumerate(((cks["untrained"], "untrained"), (last, last["checkpoint"]))):
            v = [rec["content"][m][k][key] for m, k in items]
            ax.scatter(v, y, s=36, color=[REF, ARM_COLORS[0]][ci], zorder=3 + ci, label=lab,
                       edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_xlim(-0.6, 0.6)
        style(ax, title); ax.set_xlabel("rank correlation, depth vs statistic", fontsize=8, color=INK_2)
    axs[0].set_yticks(y); axs[0].set_yticklabels([f"{m} · {k}" for m, k in items], fontsize=8)
    axs[0].invert_yaxis()
    h, lab = axs[0].get_legend_handles_labels()
    finish(fig, out / f"fig8_depth_vs_content_{arm['arm']}.png",
           "Does a token's depth track the content of its own 16x16 patch?",
           "Spearman / scene-demeaned rank correlation over every held-out scene carrying the modality. "
           "Read against the untrained model, not against zero.", (h, lab))


def fig_content_over_training(arm, out):
    """Within-scene depth-vs-content correlation for every checkpoint: is it stable?"""
    cks = [r for r in arm["checkpoints"] if r.get("content")]
    if len(cks) < 3:
        return
    items = [(m, k) for m, ks in CV.PATCH_STATS.items() for k in ks if m in cks[-1]["content"]]
    M = np.array([[(r["content"][m][k]["within_scene"] or 0.0) for r in cks] for m, k in items])
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, "#f0efec", DIV_POS])
    fig, ax = plt.subplots(figsize=(1.0 * len(cks) + 3.2, 0.36 * len(items) + 2.2))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -0.3, 0.3), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(cks)))
    ax.set_xticklabels([f"ep {r['epoch']:.0f}" for r in cks], fontsize=8)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([f"{m} · {k}" for m, k in items], fontsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID_INK)
    finish(fig, out / f"fig9_content_over_training_{arm['arm']}.png",
           "Depth vs own-patch content, within scene, at every checkpoint",
           "Scene-demeaned rank correlation. Red = more complex/brighter patch -> deeper; blue = -> shallower. "
           "Colour saturates at |0.3|.")


# ------------------------------------------------------------------ tables

def write_csv(arm, out):
    cols = ["checkpoint", "step", "epoch", "cum_flops_full", "cum_flops_plan", "step_flops_full",
            "step_flops_plan", "layers_per_body_token", "logged_eval_loss", "tf_ce_all"]
    for n in CV.MODS:
        cols += [f"tf_ce_{n}", f"logged_eval_{n}", f"tf_top1_{n}", f"tf_top5_{n}", f"tf_slot_{n}",
                 f"tl_ce_{n}", f"tl_top1_{n}"]
    for T in CV.TARGETS:
        cols += [f"gen_{T}", f"gen_ceiling_{T}", f"gen_shuffled_{T}", f"gen_pixacc_{T}" if T == "LULC"
                 else f"gen_n_in_range_{T}", f"gen_token_acc_{T}", f"gen_entropy_{T}"]
    with open(out / f"curves_{arm['arm']}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in arm["checkpoints"]:
            row = {"checkpoint": r["checkpoint"], "step": r["step"], "epoch": round(r["epoch"], 3),
                   "cum_flops_full": r.get("cum_flops_full"), "cum_flops_plan": r.get("cum_flops_plan"),
                   "step_flops_full": r["full"], "step_flops_plan": r["plan"],
                   "layers_per_body_token": round(r["layers_per_body_token"], 4),
                   "logged_eval_loss": r["logged"].get("eval_loss"),
                   "tf_ce_all": round(r["modes"]["router"]["ce_all_body_tokens"], 4)}
            for n in CV.MODS:
                tf, tl = r["tf"].get(n, {}), r["tl"].get(n, {})
                row.update({f"tf_ce_{n}": tf.get("ce"), f"logged_eval_{n}": r["logged"].get(f"eval_loss_{n}"),
                            f"tf_top1_{n}": tf.get("top1"), f"tf_top5_{n}": tf.get("top5"),
                            f"tf_slot_{n}": tf.get("slot_mass"), f"tl_ce_{n}": tl.get("ce"),
                            f"tl_top1_{n}": tl.get("top1")})
            for T in CV.TARGETS:
                if T not in r["gen"]:
                    continue
                g = r["gen"][T]
                row.update({f"gen_{T}": CV.gen_value(r, T), f"gen_ceiling_{T}": CV.gen_value(r, T, "ceiling"),
                            f"gen_shuffled_{T}": CV.gen_value(r, T, "shuffled"),
                            f"gen_token_acc_{T}": g["gen_stats"].get("token_accuracy"),
                            f"gen_entropy_{T}": g["gen_stats"].get("entropy_nats")})
                if T == "LULC":
                    row["gen_pixacc_LULC"] = g["summary"]["generated"].get("pixel_acc")
                else:
                    row[f"gen_n_in_range_{T}"] = g["summary"]["generated"]["n_in_range"]
            w.writerow({k: (round(v, 6) if isinstance(v, float) and abs(v) < 1e6 else v) for k, v in row.items()})


def paired(a, b, out):
    """Per-scene paired generation differences at shared checkpoints (same scenes)."""
    res = {}
    cb = {r["checkpoint"]: r for r in b["checkpoints"]}
    for ra in a["checkpoints"]:
        rb = cb.get(ra["checkpoint"])
        if not rb:
            continue
        for T in CV.TARGETS:
            if T not in ra["gen"] or T not in rb["gen"]:
                continue
            m = ra["gen"][T]["metric"]
            pa = {s["row"]: s for s in ra["gen"][T]["per_scene"]["generated"] if s["in_range"]}
            pb = {s["row"]: s for s in rb["gen"][T]["per_scene"]["generated"] if s["in_range"]}
            common = sorted(set(pa) & set(pb))
            d = np.array([pa[k][m] - pb[k][m] for k in common if pa[k][m] is not None and pb[k][m] is not None])
            if len(d) > 1:
                res.setdefault(ra["checkpoint"], {})[T] = {
                    "metric": m, "n": len(d), "mean_a_minus_b": float(d.mean()),
                    "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                    "a_better_share": float((d > 0).mean() if m == "miou" else (d < 0).mean())}
    (out / f"paired_{a['arm']}_vs_{b['arm']}.json").write_text(json.dumps(res, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="name=run_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    arms = []
    for spec in args.arm:
        name, run_dir = spec.split("=", 1)
        A = CV.load_arm(name, run_dir)
        for r in A["checkpoints"]:
            if r["has_router"]:
                tok = np.load(Path(r["dir"]) / "tokens_router.npz")
                r["routing"] = CV.routing_summary(tok)
                r["content"] = CV.depth_vs_content(tok, np.load(Path(r["dir"]) / "eval_rows.npy"))
        arms.append(A)
        write_csv(A, out)
        slim = {**A, "checkpoints": [{k: v for k, v in r.items() if k != "gen"} | {
            "gen": {T: {"summary": g["summary"], "metric": g["metric"], "gen_stats": g["gen_stats"]}
                    for T, g in r["gen"].items()}} for r in A["checkpoints"]]}
        slim.pop("logged")
        (out / f"summary_{name}.json").write_text(json.dumps(slim, indent=1, default=float) + "\n")
        print(f"{name}: {len(A['checkpoints'])} checkpoints "
              f"({', '.join(r['checkpoint'] for r in A['checkpoints'])})")

    fig_training(arms, out)
    fig_accuracy(arms, out, "top1")
    fig_accuracy(arms, out, "top5")
    # A10/B10 have no D3.11 decodes (Step 12 dropped decoded metrics, plan 4.18):
    # draw no empty generation panels and pair nothing.
    has_gen = any(r["gen"] for A in arms for r in A["checkpoints"])
    if has_gen:
        fig_generation(arms, out, "epoch", "fig3_epoch_vs_generation.png", "epoch",
                       "Generation quality against held-out ground truth, per checkpoint")
        fig_generation(arms, out, "cum_flops_full", "fig4_d311_flops_vs_generation.png",
                       "cumulative training FLOPs (full count)",
                       "D3.11 — generation error against compute consumed")
    else:
        print("no D3.11 decodes for these arms: generation figures and pairing skipped")
    for A in arms:
        fig_routing(A, out)
        fig_interventions(A, out)
        fig_depth_vs_ce(A, out)
        fig_content(A, out)
        fig_content_over_training(A, out)
    if len(arms) == 2 and has_gen:
        paired(arms[0], arms[1], out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
