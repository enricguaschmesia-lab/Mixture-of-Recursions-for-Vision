"""One slide: B10cV (CLEVR-pretrained layers) against B10c (from scratch). `.venv` or any python with matplotlib.

Left, what B10cV is and how it trains, in the midterm deck's diagram language (blue-grey pretrained, red hatched
new, flame trains, snowflake frozen, red cross thrown away). Right, B10cV - B10c on the trainer's logged
held-out loss and on the training loss, both on the same sequences (seed 42), with the seed-to-seed gap of
the one model trained twice (A10c, seeds 42 and 43) as the noise scale.

Every number is read from a stored result:
  /data/enric/logs/clevr_init/b10cv-s2-*.log      B10cV stage 2, `runai logs` (its eval and loss dicts)
  runs/pretrain/phase3/vanilla10c_20261011_172253/checkpoint-3300/trainer_state.json        B10c
  runs/pretrain/phase3/{mor10c_20261012_013533,mor10cs2_20261012_181812}/checkpoint-3300/trainer_state.json

    python eo/experiments/clevr_init_slide.py
Writes /data/enric/reports/clevr_init/figs/clevr_init_slide.png and numbers.json beside it.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midterm_figures as mf  # noqa: E402  (diagram helpers and colours; sets the base rcParams)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

RUNS = Path("/data/enric/runs/pretrain/phase3")
LOGS = Path("/data/enric/logs/clevr_init")
OUT = Path("/data/enric/reports/clevr_init/figs")
B10C, MOR, MOR2 = "vanilla10c_20261011_172253", "mor10c_20261012_013533", "mor10cs2_20261012_181812"
V_LOG = "b10cv-s2-20261027-194530.log"
SPE = 165 / 0.4988662131519274          # optimizer steps per epoch (B10c's epoch at step 165)
EVAL, LOGSTEP = 165, 10                  # eval and loss-logging intervals, in steps

COL = {"b10c": mf.MODEL["b10c"], "b10cv": "#9c2f6f"}   # vanilla orange as in both decks; B10cV plum
SURF, INK, INK2, MUTED, GRIDC, BAND = mf.SURF, mf.INK, mf.INK2, mf.MUTED, "#e1e0d9", "#ebeae4"


def state(run):
    return json.loads((RUNS / run / "checkpoint-3300/trainer_state.json").read_text())["log_history"]


def from_state(run):
    hist = state(run)
    ev = {h["step"]: h["eval_loss"] for h in hist if "eval_loss" in h}
    tr = {h["step"]: h["loss"] for h in hist if "loss" in h}
    return ev, tr


def from_log(path):
    """The trainer's eval and loss dicts in a job log; steps from their (2-decimal) epoch."""
    txt = path.read_text(errors="replace")
    ev, tr = {}, {}
    for m in re.finditer(r"\{'(?:eval_)?loss'[^{}]*\}", txt):
        d = ast.literal_eval(m.group(0))
        if "eval_loss" in d:
            ev[round(d["epoch"] * SPE / EVAL) * EVAL] = d["eval_loss"]
        else:
            tr[round(d["epoch"] * SPE / LOGSTEP) * LOGSTEP] = d["loss"]
    return ev, tr


def window_mean(tr, s):
    xs = [v for k, v in tr.items() if s - EVAL < k <= s]
    assert len(xs) in (EVAL // LOGSTEP, EVAL // LOGSTEP + 1), (s, len(xs))   # 16 or 17; both runs share steps
    return sum(xs) / len(xs)


def numbers():
    bev, btr = from_state(B10C)
    vev, vtr = from_log(LOGS / V_LOG)
    steps = [s for s in sorted(bev) if s > 0]
    assert steps == [s for s in sorted(vev) if s > 0] == list(range(EVAL, 3301, EVAL)), "eval steps differ"
    assert sorted(btr) == sorted(vtr) == list(range(LOGSTEP, 3301, LOGSTEP)), "loss steps differ"
    m1, m2 = (from_state(r)[0] for r in (MOR, MOR2))
    seed = [abs(m2[s] - m1[s]) for s in steps]
    return dict(
        steps=steps,
        heldout=[vev[s] - bev[s] for s in steps],
        training=[window_mean(vtr, s) - window_mean(btr, s) for s in steps],
        b10c_final=bev[3300], b10cv_final=vev[3300], b10cv_stage2_start=vev[0],
        seed_max=max(seed), seed_mean=sum(seed) / len(seed))


# ------------------------------------------------------------------ left: what B10cV is

def badge(ax, kind, x, y, size=0.9):
    mf.icon(ax, kind, x, y, size)


def schematic(ax):
    ax.set_xlim(0, 6.3); ax.set_ylim(-0.75, 6.35); ax.set_aspect("equal"); ax.set_axis_off()
    LX, LW, TX, TW, H = 2.05, 1.75, 4.05, 2.15, 0.8         # layers and table columns; box height
    ax.text(LX + LW / 2, 6.1, "layers", ha="center", va="center", fontsize=17, color=INK2, fontweight="bold")
    ax.text(TX + TW / 2, 6.1, "token table", ha="center", va="center", fontsize=17, color=INK2, fontweight="bold")

    def row(y, label, sub=None, col=INK):
        ax.text(1.85, y + H / 2 + (0.14 if sub else 0), label, ha="right", va="center", fontsize=17,
                fontweight="bold", color=col)
        if sub:
            ax.text(1.85, y + H / 2 - 0.22, sub, ha="right", va="center", fontsize=14, color=INK2)

    # CLEVR model: its layers carry over; its RGB rows are set aside, every other row is dropped
    y = 4.95
    row(y, "CLEVR model")
    mf.arch_box(ax, LX, y, LW, H, "pre", "pretrained", fs=15)
    rw = 0.8
    mf.arch_box(ax, TX, y, rw, H, "pre", "RGB", fs=15)
    mf.arch_box(ax, TX + rw + 0.08, y, TW - rw - 0.08, H, "gone", "other", fs=15)
    mf.crossed(ax, TX + rw + 0.08, y, TW - rw - 0.08, H, lw=3)
    ax.text(TX + rw / 2, y - 0.2, "kept aside", ha="center", va="center", fontsize=13, color=INK2)
    ax.annotate("", xy=(LX + LW / 2, 4.5), xytext=(LX + LW / 2, y - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.8, mutation_scale=16))
    ax.text(LX + LW / 2 + 0.12, 4.65, "copied", ha="left", va="center", fontsize=13, color=INK2)

    # B10cV: stage 1 (layers frozen) then stage 2 (B10c's recipe)
    for y, name, sub, kinds in ((3.4, "stage 1", "1 epoch", ("frozen", "trained")),
                                (2.2, "stage 2", "10 epochs", ("trained", "trained"))):
        row(y, name, sub)
        mf.arch_box(ax, LX, y, LW, H, "pre", "pretrained", fs=15)
        mf.arch_box(ax, TX, y, TW, H, "new", "TerraMesh", fs=15)
        badge(ax, kinds[0], LX + LW / 2, y + H)
        badge(ax, kinds[1], TX + TW / 2, y + H)
    ax.plot([0.1, 0.1], [2.15, 4.25], color=COL["b10cv"], lw=4, solid_capstyle="round")
    ax.text(0.28, 4.35, "B10cV", ha="left", va="bottom", fontsize=19, fontweight="bold", color=COL["b10cv"])

    # B10c: the reference, the same table and recipe from random layers
    ax.plot([0.1, 6.2], [1.6, 1.6], color=GRIDC, lw=1.5)
    y = 0.4
    row(y, "B10c", "10 epochs", col=COL["b10c"])
    mf.arch_box(ax, LX, y, LW, H, "new", "random", fs=15)
    mf.arch_box(ax, TX, y, TW, H, "new", "TerraMesh", fs=15)
    badge(ax, "trained", LX + LW / 2, y + H)
    badge(ax, "trained", TX + TW / 2, y + H)

    # legend
    yl = -0.42
    items = [("pre", "pretrained", 0.35), ("new", "new", 2.45), ("trained", "trains", 3.6), ("frozen", "frozen", 4.95)]
    for kind, lab, x in items:
        if kind == "pre":
            ax.add_patch(FancyBboxPatch((x - 0.21, yl - 0.13), 0.42, 0.26, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=mf.PRE_FC, ec=mf.PRE_EC, lw=1.6))
        elif kind == "new":
            mf.arch_box(ax, x - 0.21, yl - 0.13, 0.42, 0.26, "new")
        elif kind == "trained":
            mf.flame(ax, x, yl, 0.2)
        else:
            mf.snowflake(ax, x, yl, 0.15)
        ax.text(x + 0.32, yl, lab, ha="left", va="center", fontsize=14.5)


# ------------------------------------------------------------------ right: the comparison

def signed(v, nd):
    return f"{v:+.{nd}f}".replace("-", "\u2212")


def comparison(ax, n):
    ep = [s / SPE for s in n["steps"]]
    lo, hi = -0.45, 0.13
    ax.set_xlim(0, 10.25); ax.set_ylim(lo, hi)
    ax.axhspan(-n["seed_max"], n["seed_max"], color=BAND, lw=0, zorder=0)
    ax.text(0.15, n["seed_max"] + 0.006, "seed-to-seed", ha="left", va="bottom", fontsize=13, color=MUTED)
    ax.axhline(0, color=COL["b10c"], lw=2.4, zorder=2)
    ax.text(10.22, 0.006, "B10c", ha="right", va="bottom", fontsize=16, fontweight="bold", color=COL["b10c"])

    clip = lo + 0.012
    tr = [max(v, clip) for v in n["training"]]
    ax.plot(ep, tr, color=COL["b10cv"], lw=2, ls=(0, (5, 3)), zorder=3)
    ax.plot(ep, n["heldout"], color=COL["b10cv"], lw=2.6, marker="o", ms=7, zorder=4,
            markeredgecolor=SURF, markeredgewidth=1.5)
    for x, v in zip(ep, n["training"]):                      # off-scale points: at the edge, with their value
        if v < clip:
            ax.scatter([x], [clip], marker="v", s=70, color=COL["b10cv"], clip_on=False, zorder=5)
            ax.annotate(signed(v, 2), (x, clip), xytext=(9, 4), textcoords="offset points", fontsize=13,
                        color=INK2, va="bottom")

    # direct labels
    ax.text(1.9, -0.112, "held-out", fontsize=15, color=INK, va="center", ha="right")
    ax.text(9.55, n["training"][-1] - 0.016, "training", fontsize=15, color=INK, va="top", ha="center")
    ax.annotate(signed(n["heldout"][0], 2), (ep[0], n["heldout"][0]), xytext=(10, -4), textcoords="offset points",
                fontsize=14, color=INK2, va="center")
    ax.annotate(signed(n["heldout"][-1], 3), (ep[-1], n["heldout"][-1]), xytext=(-6, 12),
                textcoords="offset points", fontsize=16, fontweight="bold", color=INK, ha="right")
    ax.annotate(f"B10cV {n['b10cv_final']:.3f} · B10c {n['b10c_final']:.3f}", (ep[-1], n["heldout"][-1]), xytext=(-6, 34),
                textcoords="offset points", fontsize=13, color=INK2, ha="right")

    ax.text(0.15, hi - 0.012, "▲ B10cV worse", fontsize=13, color=INK2, va="top", ha="left")
    ax.text(10.15, lo + 0.012, "▼ B10cV better", fontsize=13, color=INK2, va="bottom", ha="right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss, B10cV − B10c")
    ax.set_xticks(range(0, 11))
    ax.grid(axis="y", color=GRIDC, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    n = numbers()
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig = plt.figure(figsize=(14.2, 6.6))
    schematic(fig.add_axes([0.0, 0.02, 0.42, 0.96]))
    comparison(fig.add_axes([0.52, 0.12, 0.465, 0.83]), n)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "clevr_init_slide.png", dpi=200)
    (OUT / "numbers.json").write_text(json.dumps(n, indent=1))
    print(json.dumps({k: v for k, v in n.items() if k != "steps"}, indent=1))


if __name__ == "__main__":
    main()
