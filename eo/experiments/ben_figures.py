"""Phase 6 Part 1 on B10: the slide figures (PHASE6_PLAN §2b L9). `.venv`, CPU. Reads only result files:
summary_b10/summary.json (ben_summary.py), chain/report.json (ben_chain.py report), analysis_b10/l2.json
(ben_spatial.py). Writes <out>/{tradeoff,chain,spatial,budget}.png.

    python eo/experiments/ben_figures.py --out /data/enric/reports/ben/summary_b10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REP = Path("/data/enric/reports/ben")
PRIOR = 0.311


def jl(p):
    return json.loads(Path(p).read_text())


def tradeoff(S, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lab = {"b10_head": "head only", "b10_lora_lr1e4": "LoRA, lr 1e-4", "b10_full": "full, lr 1e-4", "b10_lora": "LoRA, lr 1e-3"}
    pts = [(t, S["models"][t]) for t in lab if "test_S1S2" in S["models"][t] and "forgetting_ce_all" in S["models"][t]]
    for t, r in pts:
        ax.scatter(r["forgetting_ce_all"], r["test_S1S2"]["micro_map"], s=60, zorder=3)
        ax.annotate(lab[t], (r["forgetting_ce_all"], r["test_S1S2"]["micro_map"]), xytext=(6, -4), textcoords="offset points")
    sc = S["models"].get("b10_scratch", {})
    if "test_S1S2" in sc:
        ax.axhline(sc["test_S1S2"]["micro_map"], ls="--", c="grey", lw=1)
        ax.text(0.0, sc["test_S1S2"]["micro_map"] + 0.003, "from scratch (no pretraining)", color="grey", fontsize=9)
    ax.set_xlabel("forgetting: TerraMesh held-out CE − extended base (nats)")
    ax.set_ylabel("BEN test_a micro-mAP, S1+S2")
    ax.set_title("B10 on BigEarthNet: accuracy against forgetting")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "tradeoff.png", dpi=150); plt.close(fig)


def chain(C, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    kinds = [("floor", "+ another patch's real block", "#bbbbbb"), ("single", "one real modality alone", "#4c72b0"),
             ("chain_mean4", "+ generated block (mean of 4)", "#dd8452"), ("ceiling", "+ the real block", "#55a868")]
    for ax, cls in zip(axs, ("b10_full", "b10_lora")):
        if cls not in C:
            continue
        for gi, d in enumerate(("S1toS2", "S2toS1")):
            r = C[cls][f"{d}/base"]
            for ki, (k, lab, col) in enumerate(kinds):
                ax.bar(gi * 5 + ki, r[k], color=col, label=lab if gi == 0 else None)
        ax.set_xticks([1.5, 6.5], ["given S1, generate S2", "given S2, generate S1"])
        ax.axhline(PRIOR, ls=":", c="k", lw=1)
        ax.set_title(f"classifier: {cls}"); ax.set_ylim(0, 0.9)
    axs[0].set_ylabel("micro-mAP (10,000 test_a patches)")
    axs[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Chained generation: imputing the missing modality hurts (dotted: class prior)")
    fig.tight_layout(); fig.savefig(out / "chain.png", dpi=150); plt.close(fig)


def spatial(L2, out):
    r = L2["models"]["b10_lora/S1S2"]
    bins = [k for k in r if k.startswith("d_nn") and "micro_map" in r[k]]
    x = range(len(bins))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, [r[b]["micro_map"] for b in bins], "o-", label="B10 LoRA, S1+S2 (pixels)")
    ax.plot(x, [r[b]["knn10_micro_map"] for b in bins], "s--", label="location only: 10 nearest training patches")
    ax.plot(x, [r[b]["prior_micro_map"] for b in bins], ":", c="grey", label="class prior")
    ax.set_xticks(list(x), [b.replace("d_nn ", "").replace(" km", "") + " km\n(n=" + str(r[b]["n"]) + ")" for b in bins], fontsize=8)
    ax.set_xlabel("distance from a test patch to the nearest training patch")
    ax.set_ylabel("micro-mAP"); ax.set_ylim(0.25, 0.9); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("BEN's random split: location predicts labels; the model does not use it")
    fig.tight_layout(); fig.savefig(out / "spatial.png", dpi=150); plt.close(fig)


def budget(S, out):
    m = S["models"]
    pairs = [("10%", "b10_lora", "b10_scratch"), ("1%", "b10_lora_1pct", "b10_scratch_1pct")]
    if not all("test_S1S2" in m.get(t, {}) for _, a, b in pairs for t in (a, b)):
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for k, (lab, a, b) in enumerate(pairs):
        ax.bar(k - 0.18, m[a]["test_S1S2"]["micro_map"], 0.36, color="#4c72b0", label="pretrained B10 (LoRA)" if k == 0 else None)
        ax.bar(k + 0.18, m[b]["test_S1S2"]["micro_map"], 0.36, color="#bbbbbb", label="from scratch" if k == 0 else None)
    ax.axhline(PRIOR, ls=":", c="k", lw=1)
    ax.set_xticks([0, 1], ["10% labels (31,166)", "1% labels (3,116)"])
    ax.set_ylabel("test_a micro-mAP, S1+S2"); ax.set_ylim(0, 0.9); ax.legend(fontsize=8)
    ax.set_title("What pretraining is worth, by label budget")
    fig.tight_layout(); fig.savefig(out / "budget.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = jl(a.out / "summary.json")
    tradeoff(S, a.out)
    chain(jl(REP / "chain/report.json"), a.out)
    spatial(jl(REP / "analysis_b10/l2.json"), a.out)
    budget(S, a.out)
    print(sorted(p.name for p in a.out.glob("*.png")))


if __name__ == "__main__":
    main()
