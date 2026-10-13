"""Phase 6 L1 (PHASE6_PLAN §2b): what B10's BEN test outputs say beyond one mAP. `mor` env, CPU.

Reads eval_ben.py outputs of the selected checkpoints (test_a; ft_valsel for every threshold) and
writes <out>/l1.json, <out>/l1.md and figures:

  1. PER CLASS: AP and F1 against the class's training frequency. The class-prior AP (a constant
     score has AP = test prevalence) is beside each. F1 uses the greedy set, one val-chosen
     marginal threshold, and per-class val-chosen thresholds.
  2. CARDINALITY: predicted against true label count, for the greedy set and the val-threshold set.
  3. CALIBRATION of the Rao-Blackwellized marginals P(c in S) over all (patch, class) pairs:
     reliability bins, ECE (15 equal-width bins) and Brier score. Controls (rule 2):
       * labels drawn FROM the marginals must give an ECE at the sampling-noise level;
       * a sharpened copy (logit x 3) must give a clearly larger ECE.
  4. CONFUSIONS (greedy set): within a patch, a true class that was missed while an absent class
     was predicted. The share of those pairs inside one CORINE level-1 group is compared with
     the share when false positives are paired with another patch's misses (control: the shuffle
     must land on the analytic independence value). Level-1 micro-F1 (5 groups) beside the
     19-class one.
  5. CONDITIONS: per-class AP(S2) - AP(S1) and AP(S1+S2) - AP(S2).

    python eo/experiments/ben_test_analysis.py --out /data/enric/reports/ben/analysis_b10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.ben import labels as L  # noqa: E402
from eo.eval import ben as E  # noqa: E402

ROOT = Path("/data/enric/reports/ben")
MODELS = {"b10_full": "checkpoint-1461", "b10_lora": "checkpoint-1461"}   # selected on ft_valsel
CONDS = ("S1", "S2", "S1S2")
# CORINE land cover level 1 of the 19 classes (1 artificial, 2 agricultural, 3 forest and
# semi-natural, 4 wetlands, 5 water bodies), in NEW_LABELS order.
LEVEL1_NAMES = ["Artificial", "Agricultural", "Forest & semi-natural", "Wetlands", "Water"]
LEVEL1 = np.array([0] * 2 + [1] * 6 + [2] * 7 + [3] * 2 + [4] * 2)
assert len(LEVEL1) == L.N_CLASSES
GRID = np.linspace(0.02, 0.98, 49)


def load(tag, role, cond):
    return dict(np.load(ROOT / tag / MODELS[tag] / role / f"{cond}.npz"))


def train_freq():
    from eo.data.ben_dataset import load_index
    idx = load_index()
    y = np.stack([np.isin(np.arange(L.N_CLASSES), L.from_mask(int(m))) for m in idx.labels19[idx.ft_train]])
    return y.mean(0)


def f1_one(y, p):
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def per_class_thresholds(yv, sv):
    return np.array([max(GRID, key=lambda t: f1_one(yv[:, c], sv[:, c] >= t)) for c in range(yv.shape[1])])


def calibration(p, y, nb=15):
    p = np.clip(p, 0, 1).ravel()
    y = y.ravel().astype(float)
    b = np.minimum((p * nb).astype(int), nb - 1)
    n = np.bincount(b, minlength=nb)
    mp = np.bincount(b, p, nb) / np.maximum(n, 1)
    my = np.bincount(b, y, nb) / np.maximum(n, 1)
    ece = float(np.sum(n / n.sum() * np.abs(mp - my)))
    return {"ece": ece, "brier": float(np.mean((p - y) ** 2)),
            "bins": [{"n": int(n[i]), "mean_p": float(mp[i]), "obs": float(my[i])} for i in range(nb) if n[i]]}


def sharpen(p, k=3.0):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-k * np.log(q / (1 - q))))


def substitutions(y, g, perm=None):
    """S[a, b] = patches where present class a was missed and absent class b predicted. With
    `perm`, each patch's misses are paired with ANOTHER patch's false positives (the control)."""
    miss, fp = y & ~g, g & ~y
    if perm is not None:
        fp = fp[perm]
    return miss.T.astype(np.int64) @ fp.astype(np.int64), miss.sum(0), fp.sum(0)


def within_share(S):
    same = LEVEL1[:, None] == LEVEL1[None, :]
    return float(S[same].sum() / max(S.sum(), 1))


def level1(y):
    return np.stack([y[:, LEVEL1 == g].any(1) for g in range(len(LEVEL1_NAMES))], 1)


def card_table(kt, kp, kmax=8):
    t = np.zeros((kmax + 1, kmax + 1), int)
    np.add.at(t, (np.minimum(kt, kmax), np.minimum(kp, kmax)), 1)
    return t


def analyse(tag, cond, freq, rng):
    v, t = load(tag, "ft_valsel", cond), load(tag, "test_a", cond)
    y, s, g = t["truth"].astype(bool), t["marginal"].astype(np.float64), t["greedy"].astype(bool)
    thr = E.best_threshold(v["truth"], v["marginal"])
    thr_c = per_class_thresholds(v["truth"].astype(bool), v["marginal"])
    pt, pc = s >= thr, s >= thr_c[None, :]
    prev = y.mean(0)
    per = []
    for c in range(L.N_CLASSES):
        yc = y[:, c]
        per.append({"class": L.SHORT[c], "level1": LEVEL1_NAMES[LEVEL1[c]],
                    "train_freq": float(freq[c]), "test_prev": float(prev[c]),
                    "ap": E.average_precision(yc, s[:, c]), "ap_prior": float(prev[c]),
                    "f1_greedy": f1_one(yc, g[:, c]), "f1_thr": f1_one(yc, pt[:, c]),
                    "f1_thr_class": f1_one(yc, pc[:, c]), "thr_class": float(thr_c[c]),
                    "precision_greedy": float((yc & g[:, c]).sum() / max(g[:, c].sum(), 1)),
                    "recall_greedy": float((yc & g[:, c]).sum() / max(yc.sum(), 1))})
    from scipy.stats import spearmanr
    rho = spearmanr([r["ap"] for r in per], np.log(freq)).statistic
    kt, kg, kth = y.sum(1), g.sum(1), pt.sum(1)
    card = {"mean_true": float(kt.mean()), "mean_greedy": float(kg.mean()), "mean_thr": float(kth.mean()),
            "exact_greedy": float((kt == kg).mean()), "exact_thr": float((kt == kth).mean()),
            "table_greedy": card_table(kt, kg).tolist(), "table_thr": card_table(kt, kth).tolist()}
    over = s > 1
    cal = calibration(s, y)
    cal_ctl = {"resampled_ece": calibration(s, rng.random(s.shape) < np.clip(s, 0, 1))["ece"],
               "sharpened_ece": calibration(sharpen(s), y)["ece"],
               "prior_ece": calibration(np.broadcast_to(freq, s.shape), y)["ece"]}
    S, nm, nf = substitutions(y, g)
    S_shuf, _, _ = substitutions(y, g, perm=rng.permutation(len(g)))
    same = LEVEL1[:, None] == LEVEL1[None, :]
    expect = float((np.outer(nm, nf) * same).sum() / max(nm.sum() * nf.sum(), 1))
    rate = S / np.maximum(nm[:, None], 1)
    top = sorted(((int(S[a, b]), a, b) for a in range(L.N_CLASSES) for b in range(L.N_CLASSES) if a != b), reverse=True)[:15]
    y1, g1, p1 = level1(y), level1(g), level1(pt)
    thr_prior = E.best_threshold(v["truth"], np.broadcast_to(freq, v["truth"].shape))
    pr1 = level1(np.broadcast_to(freq >= thr_prior, y.shape))
    return {
        "n": int(len(y)), "threshold_val": float(thr),
        "micro_map": E.average_precision(y.ravel(), s.ravel()),
        "micro_f1_greedy": E.f1_scores(y, g)["micro_f1"], "micro_f1_thr": E.f1_scores(y, pt)["micro_f1"],
        "micro_f1_thr_class": E.f1_scores(y, pc)["micro_f1"], "macro_f1_thr_class": E.f1_scores(y, pc)["macro_f1"],
        "macro_f1_thr": E.f1_scores(y, pt)["macro_f1"],
        "spearman_ap_vs_log_train_freq": float(rho), "per_class": per, "cardinality": card,
        "calibration": cal, "calibration_controls": cal_ctl, "marginal_above_1": float(over.mean()),
        "substitutions": {"n_pairs": int(S.sum()), "within_level1": within_share(S),
                          "within_level1_shuffled": within_share(S_shuf), "within_level1_expected": expect,
                          "top": [{"missed": L.SHORT[a], "predicted": L.SHORT[b], "count": n,
                                   "rate_given_missed": float(rate[a, b]), "same_level1": bool(same[a, b])}
                                  for n, a, b in top],
                          "rate": rate.tolist()},
        "level1": {"micro_f1_greedy": E.f1_scores(y1, g1)["micro_f1"], "micro_f1_thr": E.f1_scores(y1, p1)["micro_f1"],
                   "prior_micro_f1": E.f1_scores(y1, pr1)["micro_f1"],
                   "prior_19_micro_f1": E.f1_scores(y, np.broadcast_to(freq >= thr_prior, y.shape))["micro_f1"],
                   "per_group_f1_greedy": {LEVEL1_NAMES[i]: f1_one(y1[:, i], g1[:, i]) for i in range(len(LEVEL1_NAMES))}},
    }


def figures(res, freq, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 1. AP against training frequency
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tag in zip(axs, MODELS):
        for cond, mk in zip(CONDS, ("v", "s", "o")):
            ap = [r["ap"] for r in res[tag][cond]["per_class"]]
            ax.scatter(freq, ap, marker=mk, label=cond, alpha=0.8)
        for c in range(L.N_CLASSES):
            ax.annotate(L.SHORT[c], (freq[c], res[tag]["S1S2"]["per_class"][c]["ap"]), fontsize=7)
        xs = np.logspace(np.log10(freq.min() * 0.8), 0, 20)
        ax.plot(xs, xs, "k:", lw=1, label="class prior (AP = prevalence)")
        ax.set_xscale("log"); ax.set_xlabel("training frequency"); ax.set_title(f"{tag} ({MODELS[tag]}), test_a")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("AP"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "ap_vs_freq.png", dpi=130); plt.close(fig)
    # 2. reliability
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, cond in zip(axs, CONDS):
        for tag in MODELS:
            b = res[tag][cond]["calibration"]["bins"]
            ax.plot([x["mean_p"] for x in b], [x["obs"] for x in b], "o-", ms=3,
                    label=f"{tag}  ECE {res[tag][cond]['calibration']['ece']:.4f}")
        ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_title(f"{cond} → BEN: reliability of P(c ∈ S)")
        ax.set_xlabel("predicted marginal"); ax.set_ylabel("observed frequency"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=130); plt.close(fig)
    # 3. cardinality and 4. substitutions, LoRA S1+S2
    r = res["b10_lora"]["S1S2"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.6]})
    t = np.array(r["cardinality"]["table_greedy"], float)
    axs[0].imshow(t / np.maximum(t.sum(1, keepdims=True), 1), cmap="Blues", origin="lower")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if t[i, j]:
                axs[0].text(j, i, f"{int(t[i, j])}", ha="center", va="center", fontsize=6)
    axs[0].set_xlabel("predicted label count (greedy; 8 = 8+)"); axs[0].set_ylabel("true label count")
    axs[0].set_title("b10_lora S1+S2: label count (row-normalized colour)")
    rate = np.array(r["substitutions"]["rate"])
    im = axs[1].imshow(rate, cmap="Reds")
    axs[1].set_xticks(range(L.N_CLASSES), L.SHORT, rotation=90, fontsize=7)
    axs[1].set_yticks(range(L.N_CLASSES), L.SHORT, fontsize=7)
    for e in np.flatnonzero(np.diff(LEVEL1)) + 0.5:
        axs[1].axhline(e, color="k", lw=0.6); axs[1].axvline(e, color="k", lw=0.6)
    axs[1].set_xlabel("wrongly predicted (absent) class"); axs[1].set_ylabel("missed (present) class")
    axs[1].set_title("b10_lora S1+S2: P(class predicted wrongly in the same patch | class missed)")
    fig.colorbar(im, ax=axs[1], fraction=0.04)
    fig.tight_layout(); fig.savefig(out / "cardinality_substitutions.png", dpi=130); plt.close(fig)


def markdown(res, freq):
    m = ["# L1: B10 test outputs beyond one mAP (test_a, selected checkpoints)", ""]
    m += ["| model | cond | micro-mAP | micro-F1 greedy | micro-F1 val thr | micro-F1 per-class thr | macro-F1 per-class thr | "
          "macro-F1 val thr | level-1 micro-F1 greedy (prior) | ECE | ECE resampled (floor) | ECE sharpened (control) | Brier | "
          "marginals > 1 | mean k true / greedy / thr | exact k greedy |",
          "|" + "---|" * 16]
    for tag in MODELS:
        for c in CONDS:
            r = res[tag][c]; k = r["cardinality"]; cc = r["calibration_controls"]
            m.append(f"| {tag} | {c} | {r['micro_map']:.4f} | {r['micro_f1_greedy']:.4f} | {r['micro_f1_thr']:.4f} | "
                     f"{r['micro_f1_thr_class']:.4f} | {r['macro_f1_thr_class']:.4f} | {r['macro_f1_thr']:.4f} | "
                     f"{r['level1']['micro_f1_greedy']:.4f} ({r['level1']['prior_micro_f1']:.4f}) | "
                     f"{r['calibration']['ece']:.4f} | {cc['resampled_ece']:.4f} | {cc['sharpened_ece']:.4f} | "
                     f"{r['calibration']['brier']:.4f} | {r['marginal_above_1']:.1e} | "
                     f"{k['mean_true']:.2f} / {k['mean_greedy']:.2f} / {k['mean_thr']:.2f} | {k['exact_greedy']:.3f} |")
    m += ["", "## Per class, S1+S2 (AP prior = test prevalence)", "",
          "| class | level 1 | train freq | AP prior | AP full | AP LoRA | F1 LoRA greedy | P / R LoRA greedy | AP LoRA S1 | S2 | ΔAP S2−S1 | ΔAP S1S2−S2 |",
          "|" + "---|" * 12]
    for c in range(L.N_CLASSES):
        f, lo = res["b10_full"]["S1S2"]["per_class"][c], res["b10_lora"]["S1S2"]["per_class"][c]
        a1, a2 = res["b10_lora"]["S1"]["per_class"][c]["ap"], res["b10_lora"]["S2"]["per_class"][c]["ap"]
        m.append(f"| {L.SHORT[c]} | {lo['level1']} | {freq[c]:.4f} | {lo['ap_prior']:.3f} | {f['ap']:.3f} | {lo['ap']:.3f} | "
                 f"{lo['f1_greedy']:.3f} | {lo['precision_greedy']:.2f} / {lo['recall_greedy']:.2f} | {a1:.3f} | {a2:.3f} | "
                 f"{a2 - a1:+.3f} | {lo['ap'] - a2:+.3f} |")
    for tag in MODELS:
        r = res[tag]["S1S2"]
        m += ["", f"Spearman(AP, log train freq), {tag} S1+S2: {r['spearman_ap_vs_log_train_freq']:.3f}"]
    sub = res["b10_lora"]["S1S2"]["substitutions"]
    m += ["", "## Substitutions, b10_lora S1+S2 (greedy set)", "",
          f"{sub['n_pairs']} (missed, wrongly predicted) pairs within patches. Inside one level-1 group: "
          f"**{sub['within_level1']:.3f}**; shuffled across patches (control) {sub['within_level1_shuffled']:.3f}; "
          f"independence {sub['within_level1_expected']:.3f}.", "",
          "| missed | predicted instead | pairs | rate given missed | same level 1 |", "|---|---|---|---|---|"]
    for t in sub["top"]:
        m.append(f"| {t['missed']} | {t['predicted']} | {t['count']} | {t['rate_given_missed']:.3f} | {t['same_level1']} |")
    return "\n".join(m) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    freq = train_freq()
    res = {tag: {c: analyse(tag, c, freq, rng) for c in CONDS} for tag in MODELS}
    # cross-check with sklearn (X7a did this on synthetic problems; here on the real outputs)
    from sklearn.metrics import average_precision_score
    t = load("b10_lora", "test_a", "S1S2")
    sk = float(average_precision_score(t["truth"].ravel(), t["marginal"].ravel()))
    res["sklearn_check"] = {"ours": res["b10_lora"]["S1S2"]["micro_map"], "sklearn": sk}
    (a.out / "l1.json").write_text(json.dumps(res, indent=1) + "\n")
    (a.out / "l1.md").write_text(markdown(res, freq))
    figures(res, freq, a.out)
    print((a.out / "l1.md").read_text())
    print("sklearn check:", res["sklearn_check"])


if __name__ == "__main__":
    main()
