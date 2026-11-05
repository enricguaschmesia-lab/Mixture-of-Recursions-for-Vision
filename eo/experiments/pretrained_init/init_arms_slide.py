"""One slide: held-out top-1 per epoch for the four ways the vanilla started. B10c (from scratch), B10cV
(CLEVR's layers), B10cL (SmolLM-135M's layers, fine-tuned) and B10cL-LoRA (SmolLM's layers frozen, rank-16
adapters). `.venv` or any python with matplotlib.

Left, top-1 per epoch for all four on one axis. Right, epoch 10 alone, where the curves meet, with the
seed-to-seed gap of A10c (seeds 42 and 43) as the noise scale around B10c.

One measure for all four: eo/scripts/eval_checkpoint.py's teacher-forced pass at fp32 on the held-out split
(4,416 rows), token-weighted over every body token (smollm_lora_compare.overall). Every checkpoint is scored
on the same card (asserted). Epoch 0 is each arm's start: B10c untrained, the pretrained arms after their
1-epoch stage 1 (layers frozen, table trained; B10cL and B10cL-LoRA share theirs).

  /data/enric/reports/arm_eval/{b10c_fp32, b10cv_s1_fp32, b10cv_fp32, b10cl_s1_fp32, b10cl_fp32,
                                b10cl_lora_fp32}/<checkpoint>/metrics.json
  /data/enric/reports/arm_eval/{a10c,a10cs2}/checkpoint-*/metrics.json     the seed pair (fp16 scores)

    python eo/experiments/pretrained_init/init_arms_slide.py
Writes /data/enric/reports/init_arms/figs/init_arms_slide.png and numbers.json beside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clevr_init_slide as cs  # noqa: E402  (colours, surface and ink; imports midterm_figures)
import smollm_lora_slide as sl  # noqa: E402  (B10cL's and B10cL-LoRA's colours)
import smollm_lora_compare as C  # noqa: E402  (the fp32 loader, token-weighted overall top-1)
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/data/enric/reports/init_arms/figs")
COL = sl.COL                        # all four pass the dataviz validator, all pairs (worst CVD dE 10.2)
SURF, INK, INK2, MUTED, GRIDC, BAND = cs.SURF, cs.INK, cs.INK2, cs.MUTED, cs.GRIDC, cs.BAND
#: arm: (step-0 tag, step-0 checkpoint, the run's tag)
ARMS = {
    "b10c": ("b10c_fp32", "untrained", "b10c_fp32"),
    "b10cv": ("b10cv_s1_fp32", "checkpoint-330", "b10cv_fp32"),
    "b10cl": ("b10cl_s1_fp32", "checkpoint-330", "b10cl_fp32"),
    "lora": ("b10cl_s1_fp32", "checkpoint-330", "b10cl_lora_fp32"),
}
NAME = {"b10c": "B10c", "b10cv": "B10cV", "b10cl": "B10cL", "lora": "B10cL-LoRA"}
WHAT = {"b10c": "from scratch", "b10cv": "CLEVR's layers", "b10cl": "SmolLM's layers, fine-tuned",
        "lora": "SmolLM's layers frozen + adapters"}


def numbers():
    top1, gpus = {}, set()
    for a, (t0, c0, tag) in ARMS.items():
        pts = [C.load_point(t0, c0)] + [C.load_point(tag, f"checkpoint-{s}") for s in C.STEPS]
        assert all(pts), f"{a}: an epoch is not scored yet"
        top1[a] = [100 * p["overall"]["top1"] for p in pts]
        gpus |= {p["gpu"] for p in pts}
    assert len(gpus) == 1, f"scored on more than one card: {gpus}"
    seed = []
    for s in C.STEPS:
        o = [C.overall(json.loads((C.EVAL / t / f"checkpoint-{s}" / "metrics.json").read_text())
                       ["modes"]["router"]["per_modality"])["top1"] for t in ("a10c", "a10cs2")]
        seed.append(100 * abs(o[0] - o[1]))
    return dict(top1=top1, gpu=gpus.pop(), seed_max=max(seed),
                final_minus_b10c={a: top1[a][-1] - top1["b10c"][-1] for a in ARMS})


def style(ax):
    ax.grid(axis="y", color=GRIDC, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def curves(ax, n):
    t = n["top1"]
    lo = min(t[a][0] for a in ARMS if a != "b10c") - 0.9
    hi = max(v[-1] for v in t.values()) + 0.45
    ax.set_xlim(-0.25, 10.25); ax.set_ylim(lo, hi)
    for a in ARMS:                           # B10c's epoch 0 (0%, untrained) falls below the axis: clipped
        ax.plot(range(11), t[a], color=COL[a], lw=2.6, marker="o", ms=7, zorder=4 if a == "b10c" else 3,
                markeredgecolor=SURF, markeredgewidth=1.5, label=f"{NAME[a]}: {WHAT[a]}")
    ax.text(1.05, lo + 0.08, "B10c: 0% at epoch 0 (untrained)", ha="left", va="bottom", fontsize=13,
            color=INK2)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), frameon=False, fontsize=14, handlelength=2.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out top-1 (%)")
    ax.set_xticks(range(0, 11))
    style(ax)


def final(ax, n):
    t = {a: v[-1] for a, v in n["top1"].items()}
    order = sorted(ARMS, key=t.get)          # highest on top
    b = t["b10c"]
    ax.axvspan(b - n["seed_max"], b + n["seed_max"], color=BAND, lw=0, zorder=0)
    ax.text(b + n["seed_max"] + 0.02, -0.3, "seed-to-seed", ha="left", va="center", fontsize=13, color=MUTED)
    for y, a in enumerate(order):
        ax.plot([t[a]], [y], "o", ms=12, color=COL[a], markeredgecolor=SURF, markeredgewidth=2, zorder=4)
        lab = f"{t[a]:.2f}%" if a == "b10c" else f"{t[a]:.2f}%  ({cs.signed(t[a] - b, 2)})"
        ax.text(t[a] - 0.07, y + 0.28, lab, ha="left", va="center", fontsize=15, color=INK,
                fontweight="bold" if a == "b10c" else "normal")
    ax.set_yticks(range(len(order)), [NAME[a] for a in order])
    for tl, a in zip(ax.get_yticklabels(), order):
        tl.set_color(COL[a]); tl.set_fontweight("bold"); tl.set_fontsize(16)
    lo = min(t.values())
    ax.set_xlim(lo - 0.15, b + 0.55); ax.set_ylim(-0.5, len(order) - 0.2)
    ax.set_title("epoch 10", loc="left", fontsize=16, color=INK, pad=10)
    ax.set_xlabel("held-out top-1 (%)")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color=GRIDC, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def main():
    n = numbers()
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.2, 6.6))
    curves(fig.add_axes([0.065, 0.12, 0.53, 0.83]), n)
    final(fig.add_axes([0.745, 0.12, 0.24, 0.76]), n)
    fig.savefig(OUT / "init_arms_slide.png", dpi=200)
    (OUT / "numbers.json").write_text(json.dumps(n, indent=1))
    print(json.dumps(n, indent=1))


if __name__ == "__main__":
    main()
