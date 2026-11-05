#!/usr/bin/env python
"""LoRA on an LLM as the EO foundation model: B10c, B10cL and B10cL-LoRA compared (2026-10-29).

Test accuracy at every epoch and the compute each arm needed (docs/notes/pretraining/smollm_lora_2026-10-29.md).
`.venv`, CPU, numpy + matplotlib.

    .venv/bin/python eo/experiments/pretrained_init/smollm_lora_compare.py [--out /data/enric/reports/smollm_lora]

Reads eo/scripts/eval_checkpoint.py's fp32 scores on the held-out split (4,416 rows):
  b10c_fp32/{untrained,checkpoint-330..3300}          B10c, vanilla from scratch
  b10cl_s1_fp32/checkpoint-330                        B10cL's stage 1 = step 0 of BOTH LLM arms
  b10cl_fp32/checkpoint-330..3300                     B10cL, SmolLM's body fully fine-tuned
  b10cl_lora_fp32/checkpoint-330..3300                B10cL-LoRA, the body frozen + rank-16 adapters
and the runs' all_results.json for the measured A100 time.

TEST ACCURACY: teacher-forced top-1 on the held-out split, any-to-any (the trainer's own eval
loader: every row's modalities in its training order), per modality and token-weighted over every
body token; and all-to-one (target-last, 512 rows per target). The epoch axis is the recipe's
(step / 330); the LLM arms' stage 1 is one more epoch before it, drawn at epoch 0.

COMPUTE: training FLOPs per row of 1,048 positions under eo/eval/curves.py's "full" definition
(every position, causal attention 2n^2d per layer, the 50.4M lm_head), with each arm's backward:
a trainable linear costs 2x its forward in backward (dX and dW), a frozen one 1x (dX: the table at
the bottom and the adapters in every layer still need gradients), the attention core 2x.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EVAL = Path("/data/enric/reports/arm_eval")
RUNS = Path("/data/enric/runs/pretrain/phase3")
B10C_RUN = RUNS / "vanilla10c_20261011_172253"
B10CL_S1_RUN = RUNS / "vanilla10cl_s1_20261028_123122"
B10CL_RUN = RUNS / "vanilla10cl_20261028_133225"
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
STEPS = list(range(330, 3301, 330))
EPOCH = 330
BATCH, SEQ = 256, 1048
STAGE1_STEPS, STAGE2_STEPS = 330, 3300

# SmolLM-135M geometry at 29 layers (eo/eval/curves.py).
D, FFN, KV, V, L = 576, 1536, 192, 87556, 29
P_LAYER = (D * D * 2 + D * KV * 2) + 3 * D * FFN + 2 * D          # 3,540,096
R = 16
P_LORA_LAYER = R * ((D + D) + 2 * (D + KV) + (D + D) + 3 * (D + FFN))   # 162,816
assert L * P_LAYER == 102_662_784 and P_LORA_LAYER == 162_816
LIN = 2 * P_LAYER * L * SEQ                 # forward FLOPs per row
ATT = 2 * D * L * SEQ ** 2
HEAD = 2 * D * V * SEQ
LORA = 2 * P_LORA_LAYER * L * SEQ
PER_ROW = {
    "full": 3 * (LIN + ATT + HEAD),                      # everything trains
    "table": 2 * LIN + 3 * ATT + 3 * HEAD,               # stage 1: body frozen, table trains
    "lora": 2 * LIN + 3 * ATT + 3 * HEAD + 3 * LORA,     # body frozen, table + adapters train
}
#: SmolLM-135M's own pretraining (SmolLM-Corpus, 600B tokens; eo/train/smollm_init.py), 6ND.
SMOLLM_PRETRAIN = 6 * 134.5e6 * 600e9

ARMS = {
    "b10c": dict(label="B10c (vanilla, from scratch)", color="#eb6834",
                 step0=("b10c_fp32", "untrained"), tag="b10c_fp32",
                 stage1=None, stage2="full", trainable=153_095_616, runs=[B10C_RUN]),
    "b10cl": dict(label="B10cL (SmolLM body, full fine-tune)", color="#3266cc",
                  step0=("b10cl_s1_fp32", "checkpoint-330"), tag="b10cl_fp32",
                  stage1="table", stage2="full", trainable=153_095_616, runs=[B10CL_S1_RUN, B10CL_RUN]),
    "lora": dict(label="B10cL-LoRA (SmolLM body frozen, r 16)", color="#1f9e89",
                 step0=("b10cl_s1_fp32", "checkpoint-330"), tag="b10cl_lora_fp32",
                 stage1="table", stage2="lora", trainable=50_432_256 + L * P_LORA_LAYER, runs=None),
}


def overall(pm):
    """Token-weighted top-1 / top-5 / CE over every body token of the any-to-any pass."""
    n = sum(pm[m]["n_tokens"] for m in MODS)
    return {k: sum(pm[m][k] * pm[m]["n_tokens"] for m in MODS) / n for k in ("top1", "top5", "ce")}


def load_point(tag, ck):
    f = EVAL / tag / ck / "metrics.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    if d.get("precision") != "fp32":
        raise SystemExit(f"{f}: scored at {d.get('precision', 'fp16')}, not fp32")
    if "router" not in d.get("modes", {}) or "target_last" not in d:
        print(f"  (skipped, still being scored: {f.parent})")
        return None
    pm = d["modes"]["router"]["per_modality"]
    return {"overall": overall(pm), "per_modality": {m: {k: pm[m][k] for k in ("top1", "top5", "ce")} for m in MODS},
            "target_last": {m: {k: d["target_last"][m][k] for k in ("top1", "ce")} for m in MODS},
            "gpu": d["gpu"], "logged_eval_loss": (d.get("logged_eval") or {}).get("eval_loss")}


def compute(arm):
    a = ARMS[arm]
    s1 = STAGE1_STEPS * BATCH * PER_ROW[a["stage1"]] if a["stage1"] else 0.0
    per_step = BATCH * PER_ROW[a["stage2"]]
    return {"stage1": s1, "per_step": per_step, "cum": [s1 + s * per_step for s in [0] + STEPS],
            "total": s1 + STAGE2_STEPS * per_step}


def runtime(arm, lora_run):
    runs = ARMS[arm]["runs"] if arm != "lora" else ([B10CL_S1_RUN, Path(lora_run)] if lora_run else None)
    if not runs:
        return None
    out = []
    for r in runs:
        f = Path(r) / "all_results.json"
        if not f.is_file():
            return None
        out.append(json.loads(f.read_text())["train_runtime"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/enric/reports/smollm_lora")
    ap.add_argument("--lora-run", default=None, help="the B10cL-LoRA run dir (for its measured runtime)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    res = {}
    for arm, a in ARMS.items():
        pts = [load_point(*a["step0"])] + [load_point(a["tag"], f"checkpoint-{s}") for s in STEPS]
        res[arm] = {"epochs": [0] + [s // EPOCH for s in STEPS], "points": pts, "compute": compute(arm),
                    "runtime_s": runtime(arm, args.lora_run), "trainable": a["trainable"]}
    doc = {"per_row_flops": PER_ROW, "smollm_pretrain_flops": SMOLLM_PRETRAIN, "arms": res,
           "ratios": {"lora_stage2_vs_full": PER_ROW["lora"] / PER_ROW["full"],
                      "table_vs_full": PER_ROW["table"] / PER_ROW["full"]}}
    (out / "numbers.json").write_text(json.dumps(doc, indent=1))

    # --- the table ------------------------------------------------------------------------
    lines = ["| epoch | " + " | ".join(f"{k} top-1" for k in ARMS) + " | LoRA - B10cL | LoRA - B10c |",
             "|---|" + "---|" * (len(ARMS) + 2)]
    for i, e in enumerate(res["b10c"]["epochs"]):
        v = {k: (res[k]["points"][i] or {}).get("overall", {}).get("top1") for k in ARMS}
        f = lambda x: "–" if x is None else f"{100 * x:.2f}%"
        d = lambda a, b: "–" if a is None or b is None else f"{100 * (a - b):+.2f} pp"
        lines.append(f"| {e} | " + " | ".join(f(v[k]) for k in ARMS) + f" | {d(v['lora'], v['b10cl'])} | {d(v['lora'], v['b10c'])} |")
    lines += ["", "| arm | trainable | stage 1 FLOPs | stage 2 FLOPs | total | A100 time |", "|---|---|---|---|---|---|"]
    for k in ARMS:
        c, rt = res[k]["compute"], res[k]["runtime_s"]
        lines.append(f"| {k} | {res[k]['trainable'] / 1e6:.1f} M | {c['stage1']:.3g} | {STAGE2_STEPS * c['per_step']:.3g} | "
                     f"{c['total']:.3g} | {'–' if rt is None else ' + '.join(f'{x / 3600:.2f} h' for x in rt)} |")
    lines.append(f"\nSmolLM-135M's own pretraining: {SMOLLM_PRETRAIN:.2g} FLOPs = "
                 f"{SMOLLM_PRETRAIN / res['b10c']['compute']['total']:.0f}x B10c's whole run.")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- the figures ------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

    def series(arm, get):
        xs, ys = [], []
        for e, p in zip(res[arm]["epochs"], res[arm]["points"]):
            if p is not None:
                xs.append(e)
                ys.append(get(p))
        return xs, ys

    # Fig 1: overall top-1 vs epoch, and vs cumulative training FLOPs.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for arm, a in ARMS.items():
        xs, ys = series(arm, lambda p: 100 * p["overall"]["top1"])
        axs[0].plot(xs, ys, color=a["color"], lw=2, marker="o", ms=5, label=a["label"])
        cum = res[arm]["compute"]["cum"]
        fx = [cum[res[arm]["epochs"].index(x)] for x in xs]
        axs[1].plot(fx, ys, color=a["color"], lw=2, marker="o", ms=5, label=a["label"])
    axs[0].set_xlabel("epoch of the 3,300-step recipe (LLM arms: after their 1-epoch stage 1)")
    axs[0].set_ylabel("held-out top-1, all body tokens (%)")
    axs[1].set_xlabel("cumulative training FLOPs (stage 1 included)")
    for ax in axs:
        ax.grid(color=GRID, lw=0.8)
    axs[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "figs" / "top1_overall.png", dpi=150)
    plt.close(fig)

    # Fig 2: per-modality top-1 vs epoch, small multiples.
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, m in zip(axs.flat, MODS):
        for arm, a in ARMS.items():
            xs, ys = series(arm, lambda p: 100 * p["per_modality"][m]["top1"])
            ax.plot(xs, ys, color=a["color"], lw=2, marker="o", ms=3.5)
        ax.set_title(m, color=INK)
        ax.grid(color=GRID, lw=0.8)
    ax = axs.flat[-1]
    for arm, a in ARMS.items():
        ax.plot([], [], color=a["color"], lw=2, marker="o", label=a["label"])
    ax.legend(frameon=False, loc="center")
    ax.axis("off")
    fig.supxlabel("epoch")
    fig.supylabel("held-out top-1 (%), any-to-any")
    fig.tight_layout()
    fig.savefig(out / "figs" / "top1_per_modality.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}/numbers.json, table.md, figs/top1_overall.png, figs/top1_per_modality.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
