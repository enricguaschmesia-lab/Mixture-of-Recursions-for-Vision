"""One slide: B10cL (SmolLM-135M's layers) against B10c (from scratch), with B10cV (CLEVR's) for reference.
`.venv` or any python with matplotlib.

clevr_init_slide.py's figure for the LLM init. Left, what B10cL is, in the same diagram language. Right,
B10cL - B10c on the trainer's logged held-out loss and on the training loss (same sequences, seed 42), with
B10cV - B10c's held-out curve as a thin reference line and the seed-to-seed gap of A10c (seeds 42 and 43)
as the noise scale.

Every number is read from a stored result:
  /data/enric/logs/smollm_init/b10cl-s2-*.log     B10cL stage 2, `runai logs` (its eval and loss dicts)
  /data/enric/logs/clevr_init/b10cv-s2-*.log      B10cV stage 2
  runs/pretrain/phase3/vanilla10c_20261011_172253/checkpoint-3300/trainer_state.json        B10c
  runs/pretrain/phase3/{mor10c_20261012_013533,mor10cs2_20261012_181812}/checkpoint-3300/trainer_state.json

⚠ B10cL's log holds `nan` (its per-modality accumulators stuck after one fp16 overflow, fork 7c266b2), so
the dicts are read with nan as None; the eval_loss and loss values this figure uses are all finite from
stage 2's step 165 on, and that is asserted.

    python eo/experiments/smollm_init_slide.py
Writes /data/enric/reports/smollm_init/figs/smollm_init_slide.png and numbers.json beside it.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clevr_init_slide as cs  # noqa: E402  (data helpers, layout constants; imports midterm_figures)
import midterm_figures as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

LOGS = Path("/data/enric/logs/smollm_init")
OUT = Path("/data/enric/reports/smollm_init/figs")
L_LOG = "b10cl-s2-20261028-133225.log"
SPE, EVAL, LOGSTEP = cs.SPE, cs.EVAL, cs.LOGSTEP

#: B10cL blue: min simulated-CVD OKLab dE 16.1 against B10cV's plum, 26.1 against B10c's orange (2026-10-28)
COL = {**cs.COL, "b10cl": "#3266cc"}
SURF, INK, INK2, MUTED, GRIDC, BAND = cs.SURF, cs.INK, cs.INK2, cs.MUTED, cs.GRIDC, cs.BAND


def from_log(path):
    """cs.from_log with `nan` read as None (B10cL's stuck per-modality values)."""
    txt = path.read_text(errors="replace")
    ev, tr = {}, {}
    for m in re.finditer(r"\{'(?:eval_)?loss'[^{}]*\}", txt):
        d = ast.literal_eval(re.sub(r"\bnan\b", "None", m.group(0)))
        if "eval_loss" in d:
            ev[round(d["epoch"] * SPE / EVAL) * EVAL] = d["eval_loss"]
        else:
            tr[round(d["epoch"] * SPE / LOGSTEP) * LOGSTEP] = d["loss"]
    return ev, tr


def numbers():
    bev, btr = cs.from_state(cs.B10C)
    lev, ltr = from_log(LOGS / L_LOG)
    vev, _ = cs.from_log(cs.LOGS / cs.V_LOG)
    steps = [s for s in sorted(bev) if s > 0]
    assert steps == [s for s in sorted(lev) if s > 0] == list(range(EVAL, 3301, EVAL)), "eval steps differ"
    assert sorted(btr) == sorted(ltr) == list(range(LOGSTEP, 3301, LOGSTEP)), "loss steps differ"
    assert all(lev[s] is not None for s in steps) and all(v is not None for v in ltr.values()), "a nan in use"
    m1, m2 = (cs.from_state(r)[0] for r in (cs.MOR, cs.MOR2))
    seed = [abs(m2[s] - m1[s]) for s in steps]
    return dict(
        steps=steps,
        heldout=[lev[s] - bev[s] for s in steps],
        training=[cs.window_mean(ltr, s) - cs.window_mean(btr, s) for s in steps],
        b10cv_heldout=[vev[s] - bev[s] for s in steps],
        b10c_final=bev[3300], b10cl_final=lev[3300], b10cv_final=vev[3300],
        seed_max=max(seed), seed_mean=sum(seed) / len(seed))


# ------------------------------------------------------------------ left: what B10cL is

def schematic(ax):
    ax.set_xlim(0, 6.3); ax.set_ylim(-0.75, 6.35); ax.set_aspect("equal"); ax.set_axis_off()
    LX, LW, TX, TW, H = 2.05, 1.75, 4.05, 2.15, 0.8
    ax.text(LX + LW / 2, 6.1, "layers", ha="center", va="center", fontsize=17, color=INK2, fontweight="bold")
    ax.text(TX + TW / 2, 6.1, "token table", ha="center", va="center", fontsize=17, color=INK2, fontweight="bold")

    def row(y, label, sub=None, col=INK):
        ax.text(1.85, y + H / 2 + (0.14 if sub else 0), label, ha="right", va="center", fontsize=17,
                fontweight="bold", color=col)
        if sub:
            ax.text(1.85, y + H / 2 - 0.22, sub, ha="right", va="center", fontsize=14, color=INK2)

    # SmolLM-135M: 29 of its 30 layers carry over; its text table is dropped
    y = 4.95
    row(y, "SmolLM", "LLM, 135M")
    mf.arch_box(ax, LX, y, LW, H, "pre", "pretrained", fs=15)
    mf.arch_box(ax, TX, y, TW, H, "gone", fs=15)
    mf.crossed(ax, TX, y, TW, H, lw=3)
    ax.text(TX + TW / 2, y - 0.2, "text tokens", ha="center", va="center", fontsize=13, color=INK2)
    ax.annotate("", xy=(LX + LW / 2, 4.5), xytext=(LX + LW / 2, y - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.8, mutation_scale=16))
    ax.text(LX + LW / 2 + 0.12, 4.68, "copied,\n29 of 30", ha="left", va="center", fontsize=13, color=INK2,
            linespacing=1.1)

    for y, name, sub, kinds in ((3.4, "stage 1", "1 epoch", ("frozen", "trained")),
                                (2.2, "stage 2", "10 epochs", ("trained", "trained"))):
        row(y, name, sub)
        mf.arch_box(ax, LX, y, LW, H, "pre", "pretrained", fs=15)
        mf.arch_box(ax, TX, y, TW, H, "new", "TerraMesh", fs=15)
        cs.badge(ax, kinds[0], LX + LW / 2, y + H)
        cs.badge(ax, kinds[1], TX + TW / 2, y + H)
    ax.plot([0.1, 0.1], [2.15, 4.25], color=COL["b10cl"], lw=4, solid_capstyle="round")
    ax.text(0.28, 4.35, "B10cL", ha="left", va="bottom", fontsize=19, fontweight="bold", color=COL["b10cl"])

    ax.plot([0.1, 6.2], [1.6, 1.6], color=GRIDC, lw=1.5)
    y = 0.4
    row(y, "B10c", "10 epochs", col=COL["b10c"])
    mf.arch_box(ax, LX, y, LW, H, "new", "random", fs=15)
    mf.arch_box(ax, TX, y, TW, H, "new", "TerraMesh", fs=15)
    cs.badge(ax, "trained", LX + LW / 2, y + H)
    cs.badge(ax, "trained", TX + TW / 2, y + H)

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

def comparison(ax, n):
    ep = [s / SPE for s in n["steps"]]
    lo, hi = -0.56, 0.14
    ax.set_xlim(0, 10.25); ax.set_ylim(lo, hi)
    ax.axhspan(-n["seed_max"], n["seed_max"], color=BAND, lw=0, zorder=0)
    ax.text(0.15, n["seed_max"] + 0.006, "seed-to-seed", ha="left", va="bottom", fontsize=13, color=MUTED)
    ax.axhline(0, color=COL["b10c"], lw=2.4, zorder=2)
    ax.text(10.22, 0.006, "B10c", ha="right", va="bottom", fontsize=16, fontweight="bold", color=COL["b10c"])

    # B10cV (CLEVR layers) for reference: thin, no markers
    ax.plot(ep, n["b10cv_heldout"], color=COL["b10cv"], lw=1.6, zorder=3)

    clip = lo + 0.012
    tr = [max(v, clip) for v in n["training"]]
    ax.plot(ep, tr, color=COL["b10cl"], lw=2, ls=(0, (5, 3)), zorder=3)
    ax.plot(ep, n["heldout"], color=COL["b10cl"], lw=2.6, marker="o", ms=7, zorder=4,
            markeredgecolor=SURF, markeredgewidth=1.5)
    for x, v in zip(ep, n["training"]):
        if v < clip:
            ax.scatter([x], [clip], marker="v", s=70, color=COL["b10cl"], clip_on=False, zorder=5)
            ax.annotate(cs.signed(v, 2), (x, clip), xytext=(9, 4), textcoords="offset points", fontsize=13,
                        color=INK2, va="bottom")

    key = [plt.Line2D([], [], color=COL["b10cl"], lw=2.6, marker="o", ms=7, markeredgecolor=SURF,
                      markeredgewidth=1.5, label="B10cL, held-out"),
           plt.Line2D([], [], color=COL["b10cl"], lw=2, ls=(0, (5, 3)), label="B10cL, training"),
           plt.Line2D([], [], color=COL["b10cv"], lw=1.6, label="B10cV (CLEVR layers), held-out")]
    ax.legend(handles=key, loc="lower right", bbox_to_anchor=(1.0, 0.09), frameon=False, fontsize=14,
              handlelength=2.6)
    ax.annotate(cs.signed(n["heldout"][0], 2), (ep[0], n["heldout"][0]), xytext=(10, -4),
                textcoords="offset points", fontsize=14, color=INK2, va="center")
    ax.annotate(cs.signed(n["heldout"][-1], 3), (ep[-1], n["heldout"][-1]), xytext=(-6, 12),
                textcoords="offset points", fontsize=16, fontweight="bold", color=INK, ha="right")
    ax.annotate(f"B10cL {n['b10cl_final']:.3f} · B10cV {n['b10cv_final']:.3f} · B10c {n['b10c_final']:.3f}",
                (ep[-1], n["heldout"][-1]), xytext=(-6, 34), textcoords="offset points", fontsize=13,
                color=INK2, ha="right")

    ax.text(0.15, hi - 0.012, "▲ worse than B10c", fontsize=13, color=INK2, va="top", ha="left")
    ax.text(10.15, lo + 0.012, "▼ better than B10c", fontsize=13, color=INK2, va="bottom", ha="right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss, arm − B10c")
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
    fig.savefig(OUT / "smollm_init_slide.png", dpi=200)
    (OUT / "numbers.json").write_text(json.dumps(n, indent=1))
    print(json.dumps({k: v for k, v in n.items() if k not in ("steps", "b10cv_heldout")}, indent=1))


if __name__ == "__main__":
    main()
