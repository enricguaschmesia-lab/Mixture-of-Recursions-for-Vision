#!/usr/bin/env python
"""Recursion-depth telemetry for the MoR arm (Phase 3 Step 8, D3.9).

Loads a checkpoint, runs corpus-stratified held-out rows through it, records the
recursion depth the router assigned to every token, and writes the figures,
the arrays and a metrics file.

    python eo/scripts/routing_telemetry.py \
        --checkpoint /data/enric/runs/pretrain/phase3/<run>/checkpoint-2000 \
        --n-rows 128

⚠ ARM A ONLY. Arm B has no router and no depth; the script refuses it rather
than reporting a constant.

⚠ STRATIFIED BY CORPUS (plan 8.7). The held-out rows are sorted majortom-first
and majortom never carries S1GRD, so an unstratified head would leave S1GRD out
of every depth figure -- which reads as "the router ignores S1GRD" rather than
as "we did not sample it". Of every failure mode in this step that is the one
most likely to be mistaken for a finding. The rows used are recorded alongside
the arrays.

⚠ THE BALANCING LOSS SHAPES THESE NUMBERS (plan 8.5). Depth is pushed towards a
uniform marginal by `mor.token.balancing: loss` at `coeff: 0.1`, with
`bal_warmup_step: 0`. Those values are printed with every run and stored in the
metrics file; a depth histogram without them invites over-reading.

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from eo.data.eval_split import load_eval_rows, load_row_table  # noqa: E402
from eo.data.terramesh_token_dataset import TerraMeshTokenDataset  # noqa: E402
from eo.generate import conditional as G  # noqa: E402
from eo.routing import telemetry as T  # noqa: E402
from eo.terramesh_tok import contract as C  # noqa: E402

ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")

# Documented steps from the data-viz reference palette's blue ramp. Depth is an
# ORDINAL magnitude (more passes = more compute), so it gets one hue light->dark
# rather than categorical hues. Ordinal rule: on a light surface start no
# lighter than step 250.
DEPTH_COLORS = ["#86b6ef", "#3987e5", "#184f95"]          # steps 250 / 400 / 600
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("blue_seq", SEQ_RAMP)
INK, INK_2, GRID_INK = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"


def stratified_rows(n: int):
    """Held-out rows, half from each corpus, so S1GRD cannot go missing."""
    rows = load_eval_rows(root_dir=ROOT)
    corpus = load_row_table(ROOT).corpus.values
    mt = [int(r) for r in rows if corpus[r] == "majortom"]
    ss = [int(r) for r in rows if corpus[r] == "ssl4eos12"]
    half = n // 2
    picked = mt[:half] + ss[: n - half]
    if len(ss) < n - half:
        picked = mt[: n - len(ss)] + ss
    return sorted(picked)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_INK)
    ax.tick_params(colors=INK_2, labelsize=9)


def fig_depth_by_modality(by_mod, geom, out_path, subtitle):
    """Share of tokens at each depth, one row per modality, sorted by mean depth.

    A stacked bar rather than seven histograms: with only three possible depths
    the distribution IS three numbers, and stacking puts every modality on one
    common axis so the comparison is the figure rather than an exercise for the
    reader.
    """
    names = sorted(by_mod, key=lambda m: by_mod[m]["mean_depth"])
    n_rec = geom["n_recursion"]
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(names) + 2.1))
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for d in range(n_rec):
        w = np.array([by_mod[m]["share"][d] for m in names])
        ax.barh(y, w, left=left, height=0.62, color=DEPTH_COLORS[d],
                edgecolor=SURFACE, linewidth=2, label=f"{d + 1} pass{'es' if d else ''}")
        for i, (xi, wi) in enumerate(zip(left, w)):
            if wi > 0.07:                      # selective labels, never all
                ax.text(xi + wi / 2, y[i], f"{100 * wi:.0f}%", ha="center",
                        va="center", fontsize=8.5,
                        color="#ffffff" if d == n_rec - 1 else INK)
        left += w
    for i, m in enumerate(names):
        ax.text(1.012, y[i], f"{by_mod[m]['mean_depth']:.2f}", va="center",
                fontsize=9, color=INK, fontweight="medium")
    ax.text(1.012, len(names) - 0.42, "mean", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of tokens", fontsize=9.5, color=INK_2)
    ax.invert_yaxis(); _style(ax)
    ax.set_title("Recursion depth by modality", fontsize=13, color=INK,
                 loc="left", pad=30)
    ax.text(0, 1.022, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2)
    # Figure-level legend with reserved space: an axes-level one placed by a
    # height-dependent offset collided with the x label at some row counts.
    h, lb = ax.get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=n_rec, frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_spatial(maps, out_path, subtitle):
    """Mean depth per patch position, one 14x14 map per image modality."""
    names = sorted(maps)
    vmin = min(float(m.min()) for m in maps.values())
    vmax = max(float(m.max()) for m in maps.values())
    fig, axes = plt.subplots(1, len(names), figsize=(2.35 * len(names) + 1.2, 3.3),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes[0], names):
        im = ax.imshow(maps[name], cmap=SEQ_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\nmean {maps[name].mean():.2f}", fontsize=9.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID_INK)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.030, pad=0.015)
    cb.set_label("mean recursion passes", fontsize=9, color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8.5)
    cb.outline.set_edgecolor(GRID_INK)
    fig.suptitle(f"Where depth is spent inside a {C.GRID}x{C.GRID} patch grid",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.915, subtitle, fontsize=8.5, color=INK_2, ha="left")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_depth_by_position(by_pos, mean_depth, geom, out_path, subtitle):
    """Mean depth against sequence position - the position-effect control.

    One series, so no legend: the title names it. The flat reference line is
    the overall mean, which is what this curve would be if position carried no
    information at all.

    ⚠ THE Y AXIS SPANS THE FULL DEPTH RANGE ON PURPOSE. Autoscaled, this curve
    fills the panel and reads as a large position effect; the measured spread
    is under 0.2 of a possible 2.0, and position explains ~0.3% of depth
    variance. A truncated axis here would make the figure argue the opposite of
    the finding, so the real range is shown and the actual span is annotated.
    """
    edges = np.asarray(by_pos["bin_edges"])
    centres = (edges[:-1] + edges[1:]) / 2
    vals = [np.nan if v is None else v for v in by_pos["mean_depth"]]
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    fig.patch.set_facecolor(SURFACE)
    ax.axhline(mean_depth, color=GRID_INK, linewidth=2, zorder=1)
    ax.text(edges[-1], mean_depth, "  overall mean", va="center", fontsize=8.5,
            color=INK_2)
    ax.plot(centres, vals, color=DEPTH_COLORS[2], linewidth=2, zorder=3)
    ax.scatter(centres, vals, s=26, color=DEPTH_COLORS[2], zorder=4,
               edgecolor=SURFACE, linewidth=2)
    lo = min(v for v in vals if not np.isnan(v))
    hi = max(v for v in vals if not np.isnan(v))
    ax.set_ylim(0.88, geom["n_recursion"] + 0.12)
    ax.set_yticks(range(1, geom["n_recursion"] + 1))
    ax.annotate(f"observed span {lo:.2f}-{hi:.2f}, i.e. {hi - lo:.2f} of a possible "
                f"{geom['n_recursion'] - 1:.2f}",
                xy=(edges[0], hi), xytext=(edges[0], hi + 0.22), fontsize=8.5,
                color=INK_2)
    ax.set_xlabel("position in the sequence (tokens)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("mean recursion passes", fontsize=9.5, color=INK_2)
    ax.grid(axis="y", color=GRID_INK, linewidth=0.8, alpha=0.6); ax.set_axisbelow(True)
    _style(ax)
    ax.set_title("Recursion depth by sequence position", fontsize=13, color=INK,
                 loc="left", pad=30)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-config", default="eo_terramesh/arm_a_mor")
    ap.add_argument("--checkpoint", default=None,
                    help="omit for the untrained control")
    ap.add_argument("--n-rows", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-root", default="/data/enric/reports/phase3_step8")
    ap.add_argument("--rand-router", action="store_true",
                    help="CONTROL: replace the router with random assignment; "
                         "per-modality separation must collapse")
    ap.add_argument("--modality-order", default="random", choices=["random", "fixed"],
                    help="random (default) matches training AND decorrelates "
                         "modality from position; fixed confounds them")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    overrides = ["mor.rand_router=true"] if args.rand_router else None
    model, cfg = G.build_model(args.arm_config, args.checkpoint, device=args.device,
                               overrides=overrides)
    geom = T.describe_geometry(model)

    tokcfg = cfg.mor.token
    subtitle = (f"balancing={cfg.mor.token.balancing} coeff={tokcfg.coeff} "
                f"bal_warmup_step={tokcfg.bal_warmup_step} alpha={tokcfg.alpha}  ·  "
                f"{geom['n_recursion']} recursion steps x "
                f"{geom['blocks_per_recursion_step']} shared blocks")

    # cfg.max_length is top-level (cfg.dataset is the dataset NAME, a string).
    ds = TerraMeshTokenDataset(root_dir=ROOT, max_length=int(cfg.max_length),
                               modality_order=args.modality_order)
    rows = stratified_rows(args.n_rows)

    print(f"checkpoint     : {args.checkpoint or 'NONE -- untrained control'}")
    print(f"router         : {'RANDOM (control)' if args.rand_router else 'trained'}")
    print(f"geometry       : {geom}")
    print(f"rows           : {len(rows)} corpus-stratified held-out")
    print(f"modality order : {args.modality_order}")
    print(f"balancing      : {subtitle}\n")

    cap = T.capture_depths(model, ds, rows, batch_size=args.batch_size,
                           device=args.device)

    by_mod = T.depth_by_modality(cap)
    by_pos = T.depth_by_position(cap)
    var = T.variance_explained(cap)
    flops = T.compute_accounting(cap)
    maps = T.spatial_maps(cap)
    within = T.within_modality_decomposition(cap)

    print(f"{'modality':9}{'tokens':>9}{'mean depth':>12}   share at depth 1 / 2 / 3")
    for m in sorted(by_mod, key=lambda m: by_mod[m]["mean_depth"]):
        s = by_mod[m]
        print(f"{m:9}{s['n_tokens']:9d}{s['mean_depth']:12.3f}   "
              + " / ".join(f"{x:.3f}" for x in s["share"]))
    print(f"\nvariance in depth explained by MODALITY {var['by_modality']:.4f}"
          f"   by POSITION {var['by_position']:.4f}")
    print(f"\nwithin each modality -- is depth following the scene, or the layout?")
    print(f"{'modality':9}{'depth sd':>10}{'by SCENE':>10}{'by PATCH POS':>14}{'map S/N':>9}")
    for m in sorted(within, key=lambda m: -within[m]["by_scene"]):
        w = within[m]
        print(f"{m:9}{w['depth_sd']:10.4f}{w['by_scene']:10.4f}"
              f"{w['by_patch_position']:14.4f}{w['spatial_signal_to_noise'] or 0:9.2f}")
    print(f"\ncompute: {flops['layers_per_token_mor']} layer applications per token "
          f"vs vanilla's {flops['layers_per_token_vanilla']}"
          f"  ->  {100 * flops['compute_saving']:.1f}% fewer")

    tag = args.tag or ("rand_router" if args.rand_router else
                       (Path(args.checkpoint).name if args.checkpoint else "untrained"))
    out = Path(args.out_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "telemetry.npz", depth=cap.depth,
                        modality_ids=cap.modality_ids, rows=cap.rows)
    fig_depth_by_modality(by_mod, geom, out / "depth_by_modality.png", subtitle)
    fig_depth_by_position(by_pos, flops["mean_depth"], geom,
                          out / "depth_by_position.png", subtitle)
    if maps:
        fig_spatial(maps, out / "depth_spatial.png", subtitle)

    (out / "routing_metrics.json").write_text(json.dumps({
        "tag": tag, "checkpoint": args.checkpoint, "arm_config": args.arm_config,
        "rand_router": args.rand_router, "modality_order": args.modality_order,
        "n_rows": len(rows), "rows": [int(r) for r in cap.rows],
        "geometry": geom,
        "balancing": {"mode": str(cfg.mor.token.balancing), "coeff": float(tokcfg.coeff),
                      "bal_warmup_step": int(tokcfg.bal_warmup_step),
                      "alpha": float(tokcfg.alpha)},
        "depth_by_modality": by_mod, "depth_by_position": by_pos,
        "variance_explained": var, "compute": flops,
        "within_modality": within,
        "spatial_mean_depth": {k: [[round(float(x), 4) for x in r] for r in v]
                               for k, v in maps.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
