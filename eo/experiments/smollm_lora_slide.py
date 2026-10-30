"""Two slides: B10cL-LoRA (SmolLM-135M's layers frozen, rank-16 adapters) against B10cL (the same layers
fine-tuned) and B10c (from scratch). `.venv` or any python with matplotlib.

smollm_init_slide.py's diagram language. Page 1, left, what B10cL-LoRA is; right, held-out top-1 per epoch
minus B10c's, with the seed-to-seed gap of A10c (seeds 42 and 43) as the noise scale. Page 2, the compute:
trainable parameters, training FLOPs and measured A100 time per arm.

Every number is read from a stored result:
  /data/enric/reports/smollm_lora/numbers.json     smollm_lora_compare.py: fp32 top-1 per epoch, FLOPs, runtime
  /data/enric/reports/arm_eval/{a10c,a10cs2}/checkpoint-*/metrics.json      the seed pair (fp16 scores)

    python eo/experiments/smollm_lora_slide.py
Writes /data/enric/reports/smollm_lora/figs/{smollm_lora_slide,smollm_lora_compute_slide}.png and
slide_numbers.json beside them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clevr_init_slide as cs  # noqa: E402  (layout constants, badges; imports midterm_figures)
import midterm_figures as mf  # noqa: E402
import smollm_init_slide as si  # noqa: E402  (B10cL's colour)
import smollm_lora_compare as C  # noqa: E402  (token-weighted overall top-1)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REP = Path("/data/enric/reports/smollm_lora")
OUT = REP / "figs"

#: B10cL-LoRA teal: all-pairs simulated-CVD OKLab dE >= 10.2 against B10cL's blue and B10c's orange,
#: 3.2:1 on the surface (dataviz validator, 2026-10-30)
COL = {**si.COL, "lora": "#1f9e89"}
SURF, INK, INK2, MUTED, GRIDC, BAND = cs.SURF, cs.INK, cs.INK2, cs.MUTED, cs.GRIDC, cs.BAND
ARMS = ("b10c", "b10cl", "lora")
NAME = {"b10c": "B10c", "b10cl": "B10cL", "lora": "B10cL-LoRA"}


def numbers():
    n = json.loads((REP / "numbers.json").read_text())["arms"]
    top1 = {}
    for a in ARMS:
        assert n[a]["epochs"] == list(range(11)) and all(n[a]["points"]), f"{a}: an epoch is missing"
        top1[a] = [100 * p["overall"]["top1"] for p in n[a]["points"]]
    seed = []
    for s in C.STEPS:
        o = [C.overall(json.loads((C.EVAL / t / f"checkpoint-{s}" / "metrics.json").read_text())
                       ["modes"]["router"]["per_modality"])["top1"] for t in ("a10c", "a10cs2")]
        seed.append(100 * abs(o[0] - o[1]))
    return dict(
        top1=top1,
        diff={a: [v - b for v, b in zip(top1[a], top1["b10c"])] for a in ("b10cl", "lora")},
        flops={a: n[a]["compute"]["total"] for a in ARMS},
        hours={a: sum(n[a]["runtime_s"]) / 3600 for a in ARMS},
        trainable={a: n[a]["trainable"] / 1e6 for a in ARMS},
        seed_max=max(seed), seed_mean=sum(seed) / len(seed))


# ------------------------------------------------------------------ page 1, left: what B10cL-LoRA is

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

    # stage 1: the one both LLM arms share, bit for bit
    y = 4.95
    row(y, "stage 1", "1 epoch")
    mf.arch_box(ax, LX, y, LW, H, "pre", "pretrained", fs=15)
    mf.arch_box(ax, TX, y, TW, H, "new", "TerraMesh", fs=15)
    cs.badge(ax, "frozen", LX + LW / 2, y + H)
    cs.badge(ax, "trained", TX + TW / 2, y + H)
    ax.text(TX + TW / 2, y - 0.2, "shared by both", ha="center", va="center", fontsize=13, color=INK2)

    # stage 2: B10cL fine-tunes the layers; B10cL-LoRA keeps them frozen and trains small adapters
    y = 3.55
    row(y, "B10cL", "stage 2", col=COL["b10cl"])
    mf.arch_box(ax, LX, y, LW, H, "pre", "pretrained", fs=15)
    mf.arch_box(ax, TX, y, TW, H, "new", "TerraMesh", fs=15)
    cs.badge(ax, "trained", LX + LW / 2, y + H)
    cs.badge(ax, "trained", TX + TW / 2, y + H)

    y = 2.25
    row(y, "LoRA", "stage 2", col=COL["lora"])
    AX, AW, AH = LX + LW - 0.98, 0.9, 0.4          # the adapters: a "new" pill across the frozen box's edge
    mf.arch_box(ax, LX, y, LW, H, "pre")
    ax.text(LX + LW / 2, y + H / 2 + 0.11, "pretrained", ha="center", va="center", fontsize=15, color=INK, zorder=6)
    mf.arch_box(ax, AX, y - AH / 2, AW, AH, "new", "LoRA", fs=13)
    mf.arch_box(ax, TX, y, TW, H, "new", "TerraMesh", fs=15)
    cs.badge(ax, "frozen", LX + LW / 2, y + H)
    cs.badge(ax, "trained", AX - 0.02, y, size=0.75)
    cs.badge(ax, "trained", TX + TW / 2, y + H)
    ax.text(AX + AW + 0.1, y - 0.27, "rank-16 adapters, 4.7 M", ha="left", va="center", fontsize=13, color=INK2)

    ax.plot([0.1, 6.2], [1.5, 1.5], color=GRIDC, lw=1.5)
    y = 0.4
    row(y, "B10c", "10 epochs", col=COL["b10c"])
    mf.arch_box(ax, LX, y, LW, H, "new", "random", fs=15)
    mf.arch_box(ax, TX, y, TW, H, "new", "TerraMesh", fs=15)
    cs.badge(ax, "trained", LX + LW / 2, y + H)
    cs.badge(ax, "trained", TX + TW / 2, y + H)

    yl = -0.42
    for kind, lab, x in (("pre", "pretrained", 0.35), ("new", "new", 2.45), ("trained", "trains", 3.6),
                         ("frozen", "frozen", 4.95)):
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


# ------------------------------------------------------------------ page 1, right: top-1 against B10c

def accuracy(ax, n):
    ep = list(range(1, 11))                 # epoch 0: B10c is untrained (0%), both LLM arms at 10.8%
    lo, hi = -1.08, 2.4
    ax.set_xlim(0, 10.25); ax.set_ylim(lo, hi)
    ax.axhspan(-n["seed_max"], n["seed_max"], color=BAND, lw=0, zorder=0)
    ax.text(3.3, n["seed_max"] + 0.02, "seed-to-seed", ha="left", va="bottom", fontsize=13, color=MUTED)
    ax.axhline(0, color=COL["b10c"], lw=2.4, zorder=2)
    ax.text(10.22, 0.035, "B10c", ha="right", va="bottom", fontsize=16, fontweight="bold", color=COL["b10c"])
    for a in ("b10cl", "lora"):
        ax.plot(ep, n["diff"][a][1:], color=COL[a], lw=2.6, marker="o", ms=7, zorder=4,
                markeredgecolor=SURF, markeredgewidth=1.5)
    ax.annotate(cs.signed(n["diff"]["lora"][-1], 2), (10, n["diff"]["lora"][-1]), xytext=(-8, -16),
                textcoords="offset points", fontsize=16, fontweight="bold", color=INK, ha="right", va="center")
    ax.annotate(cs.signed(n["diff"]["b10cl"][-1], 2), (10, n["diff"]["b10cl"][-1]), xytext=(-8, -15),
                textcoords="offset points", fontsize=14, color=INK2, ha="right", va="center")

    key = [plt.Line2D([], [], color=COL[a], lw=2.6, marker="o", ms=7, markeredgecolor=SURF, markeredgewidth=1.5,
                      label=lab) for a, lab in (("b10cl", "B10cL: layers fine-tuned"),
                                                ("lora", "B10cL-LoRA: layers frozen, adapters"))]
    ax.legend(handles=key, loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=False, fontsize=14,
              handlelength=2.6)
    t = n["top1"]
    ax.text(0.99, 0.6, f"final top-1: B10c {t['b10c'][-1]:.2f}%\nB10cL {t['b10cl'][-1]:.2f}% · "
            f"LoRA {t['lora'][-1]:.2f}%", transform=ax.transAxes, fontsize=13, color=INK2, ha="right", va="bottom",
            multialignment="right", linespacing=1.25)

    ax.text(0.15, hi - 0.04, "▲ better than B10c", fontsize=13, color=INK2, va="top", ha="left")
    ax.text(0.15, lo + 0.04, "▼ worse than B10c", fontsize=13, color=INK2, va="bottom", ha="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out top-1, arm − B10c (pp)")
    ax.set_xticks(range(0, 11))
    ax.grid(axis="y", color=GRIDC, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------------------------------------------ page 2: compute

def bars(ax, vals, title, fmt, rel=None, note=None):
    ys = [2, 1, 0]                          # B10c on top, as in the schematic's legend order
    top = max(vals.values())
    for a, y in zip(ARMS, ys):
        ax.barh(y, vals[a], height=0.62, color=COL[a], zorder=3)
        lab = fmt(vals[a])
        if rel and a == "lora":
            lab += f"  ({100 * (vals[a] / vals[rel] - 1):+.0f}% vs B10cL)".replace("-", "−")
        ax.text(vals[a] + 0.02 * top, y, lab, ha="left", va="center", fontsize=15, color=INK,
                fontweight="bold" if a == "lora" else "normal")
    ax.set_xlim(0, top * 1.75); ax.set_ylim(-0.6, 2.6)
    ax.set_yticks(ys, [NAME[a] for a in ARMS])
    for tl, a in zip(ax.get_yticklabels(), ARMS):
        tl.set_color(COL[a]); tl.set_fontweight("bold"); tl.set_fontsize(16)
    ax.set_title(title, loc="left", fontsize=16, color=INK, pad=10)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    if note:
        ax.text(0, -0.55, note, ha="left", va="top", fontsize=13, color=INK2, transform=ax.transData)


def compute_page(fig, n):
    ax = fig.add_axes([0.13, 0.57, 0.37, 0.33])
    bars(ax, n["trainable"], "trainable parameters", lambda v: f"{v:.1f} M",
         note="LoRA's: 50.4 M token table + 4.7 M adapters")
    ax = fig.add_axes([0.62, 0.57, 0.37, 0.33])
    bars(ax, {a: v / 1e17 for a, v in n["flops"].items()}, "training FLOPs (×10¹⁷)", lambda v: f"{v:.2f}",
         rel="b10cl")
    ax = fig.add_axes([0.13, 0.12, 0.37, 0.33])
    bars(ax, n["hours"], "A100 hours, measured", lambda v: f"{v:.1f} h", rel="b10cl")
    ax = fig.add_axes([0.62, 0.12, 0.37, 0.33]); ax.set_axis_off()
    ax.text(0.0, 0.62, "SmolLM's own pretraining", fontsize=16, color=INK, ha="left", va="bottom")
    ax.text(0.0, 0.52, f"{n['smollm_flops'] / 1e20:.1f} × 10²⁰ FLOPs", fontsize=26, fontweight="bold",
            color=INK, ha="left", va="top")
    ax.text(0.0, 0.25, f"= {n['smollm_flops'] / n['flops']['b10c']:.0f}× B10c's whole run", fontsize=16,
            color=INK2, ha="left", va="top")


def main():
    n = numbers()
    n["smollm_flops"] = json.loads((REP / "numbers.json").read_text())["smollm_pretrain_flops"]
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})
    OUT.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.2, 6.6))
    schematic(fig.add_axes([0.0, 0.02, 0.42, 0.96]))
    accuracy(fig.add_axes([0.52, 0.12, 0.465, 0.83]), n)
    fig.savefig(OUT / "smollm_lora_slide.png", dpi=200)

    fig = plt.figure(figsize=(14.2, 6.6))
    compute_page(fig, n)
    fig.savefig(OUT / "smollm_lora_compute_slide.png", dpi=200)

    (OUT / "slide_numbers.json").write_text(json.dumps(n, indent=1))
    print(json.dumps({k: v for k, v in n.items() if k != "top1"}, indent=1))


if __name__ == "__main__":
    main()
