#!/usr/bin/env python
"""B0 Step 1: the router's state in every (seed, modality, checkpoint).

Reads /data/enric/reports/b0/router/<tag>/<ck>/ (eo/scripts/capture_router.py), writes
/data/enric/reports/b0/states/{cells.csv, checkpoints.csv, states.json}. Modality = the positions
that predict its body tokens (`label_mod`). Everything is computed on what the router output, so
a cell's numbers need no model and no card.

    .venv/bin/python eo/experiments/routing/b0_states.py

Runs in `.venv`, CPU.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402

ROOT = Path("/data/enric/reports/b0/router")
OUT = Path("/data/enric/reports/b0/states")
STORED = Path("/data/enric/reports/arm_eval")
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
CKS = [330 * i for i in range(1, 11)]
PAD_FROM = 994                                   # 995 real tokens, so position 994 on is padding


def softmax(w):
    w = w.astype(np.float64)
    w = w - w.max(-1, keepdims=True)
    p = np.exp(w)
    return p / p.sum(-1, keepdims=True)


def load(tag, ck, name):
    f = ROOT / tag / ck / f"{name}.npz"
    return np.load(f) if f.exists() else None


def cell_row(tag, ck, name, z, lm, stored=None, other=None, units=None, W=None):
    """One row per modality for one pass. `other`: {precision: depth array} on the same rows."""
    lg, d = z["logits"], z["depth"]
    p = softmax(lg)
    srt = np.sort(p, -1)[..., ::-1]
    gap = np.sort(lg.astype(np.float64), -1)[..., ::-1]
    top2 = np.argsort(lg.astype(np.float64), -1)[..., -2:]
    rows = []
    for j, m in enumerate(MODS):
        s = lm == j + 1
        if not s.any():
            continue
        ps, sp = p[s], srt[s]
        m12 = sp[:, 0] - sp[:, 1]
        tie = gap[s][:, 0] == gap[s][:, 1]
        r = dict(tag=tag, ck=ck, pass_=name, mod=m, n=int(s.sum()),
                 mean_depth=float(d[s].mean()),
                 mix=[float((d[s] == k).mean()) for k in (1, 2, 3)],
                 mean_p=[float(x) for x in ps.mean(0)],
                 sorted_p=[float(x) for x in sp.mean(0)],
                 gate=float(sp[:, 0].mean()), gate_sd=float(sp[:, 0].std()),
                 entropy_token=float((-(ps * np.log(ps + 1e-30)).sum(1)).mean()),
                 margin_med=float(np.median(m12)), margin_q05=float(np.quantile(m12, .05)),
                 margin_q95=float(np.quantile(m12, .95)),
                 logit_gap_med=float(np.median(gap[s][:, 0] - gap[s][:, 1])),
                 exact_tie=float(tie.mean()),
                 # topk runs on fp16 PROBABILITIES: inside one float16 step of p1 the two depths tie
                 # there even when their logits differ, and topk then takes the shallower one
                 within_step=float((m12 < 2.0 ** (np.floor(np.log2(sp[:, 0])) - 10)).mean()),
                 alive=[bool(x >= 0.01) for x in ps.mean(0)],
                 third=float(sp[:, 2].mean()))
        if tie.any():
            t2 = np.sort(top2[s][tie], -1) + 1
            r["tie_pairs"] = {f"{a}{b}": int(((t2[:, 0] == a) & (t2[:, 1] == b)).sum())
                              for a, b in ((1, 2), (1, 3), (2, 3))}
            r["tie_to_shallower"] = float((d[s][tie] == t2[:, 0]).mean())
        if units is not None:
            rr = np.broadcast_to(np.arange(d.shape[0])[:, None], d.shape)
            num = np.bincount(rr[s], weights=d[s].astype(np.float64), minlength=d.shape[0])
            den = np.bincount(rr[s], minlength=d.shape[0]).astype(np.float64)
            keep = den > 0
            _, lo, hi = S.ratio_boot(num[keep], den[keep], np.unique(units[keep], return_inverse=True)[1],
                                     n_draws=500)
            r["depth_ci"] = [float(lo), float(hi)]
        for k, v in (other or {}).items():
            r[f"depth_{k}"] = float(v[s].mean())
            r[f"moved_{k}"] = float((v[s] != d[s]).mean())
        if stored is not None:
            r["moved_card"] = float((stored["depth"][s] != d[s]).mean())
            r["stored_depth"] = float(stored["depth"][s].mean())
        rows.append(r)
    return rows


def checkpoint_row(tag, ck, z, im, lm):
    d = z["depth"]
    pad = np.zeros_like(d, bool)
    pad[:, PAD_FROM:] = True
    none = (im == 0) & (lm == 0)
    cls = {"body": (im > 0) | (lm > 0), "body_only": (im > 0) & (lm > 0), "bo": (im == 0) & (lm > 0),
           "eo": (im > 0) & (lm == 0), "delim": none & ~pad, "pad": pad,
           "all": np.ones_like(d, bool)}
    p = softmax(z["logits"])
    g = p.max(-1)
    out = dict(tag=tag, ck=ck)
    for k, s in cls.items():
        h = np.bincount(d[s], minlength=4)[1:] / s.sum()
        out[k] = dict(share=float(s.mean()), load=[float(x) for x in h], gate=float(g[s].mean()),
                      mean_depth=float(d[s].mean()),
                      entropy=float(-(h * np.log(h + 1e-30)).sum()))
    return out


def figure(cells):
    """Each modality's mean router probabilities over training, on the depth simplex."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#8a8984", "#e4e3de"
    RUNS = {"a10c": ("A10c, seed 42", "#008300"), "a10cs2": ("A10c-s2, seed 43", "#d55181"),
            "a10": ("A10, seed 42, original tokens", "#2a78d6")}
    tags = [t for t in RUNS if any(c["tag"] == t and c["pass_"] == "main" for c in cells)]
    mods = MODS[:6]
    h = np.sqrt(3) / 2
    fig, axes = plt.subplots(len(tags), 6, figsize=(13.2, 2.35 * len(tags) + 0.9), facecolor=SURF,
                             squeeze=False)

    def xy(p):
        p = np.asarray(p, dtype=np.float64)
        return p[..., 1] + 0.5 * p[..., 2], h * p[..., 2]

    for i, tag in enumerate(tags):
        name, col = RUNS[tag]
        for j, m in enumerate(mods):
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            ax.plot([0, 1, .5, 0], [0, 0, h, 0], color=MUTED, lw=1, zorder=1)
            refs = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
            rx, ry = xy(refs / refs.sum(1, keepdims=True))
            ax.scatter(rx, ry, s=26, facecolor="none", edgecolor=GRID, lw=1.2, zorder=1)
            row = sorted([c for c in cells if c["tag"] == tag and c["mod"] == m and c["pass_"] == "main"
                          and c["ck"] != "untrained"], key=lambda c: _n(c["ck"]))
            if row:
                x, y = xy([c["mean_p"] for c in row])
                ax.plot(x, y, color=col, lw=1.3, alpha=.55, zorder=2)
                ax.scatter(x[:-1], y[:-1], s=14, color=col, alpha=.75, lw=0, zorder=3)
                ax.scatter(x[0], y[0], s=44, facecolor=SURF, edgecolor=col, lw=1.6, zorder=4)
                ax.scatter(x[-1], y[-1], s=52, color=col, edgecolor=SURF, lw=1.2, zorder=5)
            ax.set_xlim(-.1, 1.1); ax.set_ylim(-.12, h + .1); ax.set_aspect("equal"); ax.axis("off")
            if i == 0:
                ax.text(.5, h + .17, {"S1GRD": "S1 GRD", "S1RTC": "S1 RTC", "S2L2A": "S2"}.get(m, m),
                        ha="center", va="bottom", fontsize=13, color=INK, fontweight="bold")
            for t, (tx, ty) in {"1": (-.05, -.04), "2": (1.05, -.04), "3": (.5, h + .045)}.items():
                ax.text(tx, ty, t, ha="center", va="center", fontsize=9.5, color=MUTED)
        axes[i, 0].text(-.2, h / 2, name.replace(", ", ",\n", 1), rotation=90, ha="center", va="center",
                        fontsize=11, color=INK)
    fig.suptitle("Modalities sit at the centre, an edge midpoint or a corner of the depth simplex",
                 x=.02, y=.995, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.text(.02, .945 if len(tags) > 2 else .925, "Mean router probabilities of depths 1, 2, 3 on held-out tokens. "
             "Open circle: step 330. Filled: step 3300. Grey rings: ties.", fontsize=11, color="#52514e")
    fig.subplots_adjust(left=.035, right=.995, bottom=.01, top=.90, wspace=.02, hspace=.04)
    (OUT / "figs").mkdir(exist_ok=True)
    fig.savefig(OUT / "figs" / "simplex.png", dpi=170, facecolor=SURF)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    units = S.eval_groups(np.load(STORED / "a10c/checkpoint-3300/eval_rows.npy"))
    cells, cks, repro = [], [], []
    for tag in ("a10c", "a10cs2", "a10"):
        for ck in (["untrained"] if tag == "a10c" else []) + [f"checkpoint-{c}" for c in CKS]:
            z = load(tag, ck, "main_fp16")
            if z is None:
                continue
            stored = np.load(STORED / tag / ck / "tokens_router.npz")
            other = {}
            for pr in ("fp32", "bf16"):
                o = load(tag, ck, f"main_{pr}")
                if o is not None:
                    other[pr] = o["depth"]
            d = {k: z[k] for k in z.files}
            im, lm = d["input_mod"], d["label_mod"]
            cells += cell_row(tag, ck, "main", d, lm, stored, other, units)
            cks.append(checkpoint_row(tag, ck, d, im, lm))
            g = softmax(d["logits"]).max(-1).astype(np.float16)
            repro.append(dict(tag=tag, ck=ck, depth_agree=float((d["depth"] == stored["depth"]).mean()),
                              shifted=float((d["depth"][:-1] == stored["depth"][1:]).mean()),
                              gate_maxdiff=float(np.abs(g.astype(np.float32) - stored["gate"].astype(np.float32)).max()),
                              frames=bool(np.array_equal(d["input_mod"], stored["input_mod"])
                                          and np.array_equal(d["label_mod"], stored["label_mod"]))))
            tr = load(tag, ck, "train_fp16")
            if tr is not None:
                t = {k: tr[k] for k in tr.files}
                cells += cell_row(tag, ck, "train", t, t["label_mod"])
            tl = load(tag, ck, "target_last_fp16")
            if tl is not None:
                for j, T in enumerate(MODS):
                    q = {k.split("|")[1]: tl[k] for k in tl.files if k.startswith(f"{T}|")}
                    sel = q["label_mod"] == j + 1
                    cells += [r for r in cell_row(tag, ck, "target_last", q, np.where(sel, j + 1, 0))]
    (OUT / "states.json").write_text(json.dumps({"cells": cells, "checkpoints": cks, "repro": repro}) + "\n")
    keys = sorted({k for c in cells for k in c})
    with open(OUT / "cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(cells)
    print(f"{len(cells)} cell rows, {len(cks)} checkpoints -> {OUT}")
    report(cells, cks, repro)
    figure(cells)
    return 0


def _n(ck):
    return 0 if ck == "untrained" else int(ck.split("-")[1])


def report(cells, cks, repro):
    out = []
    def P(*a):
        out.append(" ".join(str(x) for x in a))
        print(*a)
    P("== reproduction, Titan X fp16 against the stored fp16 arrays")
    for r in repro:
        P(f"{r['tag']:7s} {_n(r['ck']):5d} depth {r['depth_agree']:.4f} shifted {r['shifted']:.4f} "
          f"gate max|d| {r['gate_maxdiff']:.4f} frames {r['frames']}")
    main = [c for c in cells if c["pass_"] == "main" and c["ck"] != "untrained"]
    img = [c for c in main if c["mod"] != "Coords"]
    P("\n== mean depth, fp16 (fp32) [bf16], per tag and checkpoint")
    for tag in ("a10c", "a10cs2", "a10"):
        for ck in sorted({c["ck"] for c in main if c["tag"] == tag}, key=_n):
            row = [c for c in main if c["tag"] == tag and c["ck"] == ck]
            P(f"{tag:7s} {_n(ck):5d} " + " ".join(
                f"{c['mod'][:5]}:{c['mean_depth']:.2f}" + (f"({c['depth_fp32']:.2f})" if "depth_fp32" in c else "")
                + (f"[{c['depth_bf16']:.2f}]" if "depth_bf16" in c else "") for c in row))
    P("\n== gate")
    for tag in ("a10c", "a10cs2", "a10"):
        for ck in sorted({c["ck"] for c in main if c["tag"] == tag}, key=_n):
            row = [c for c in main if c["tag"] == tag and c["ck"] == ck]
            P(f"{tag:7s} {_n(ck):5d} " + " ".join(f"{c['mod'][:5]}:{c['gate']:.3f}" for c in row))
    m = np.array([c["margin_med"] for c in img])
    P("\n== histogram of the median p1-p2 over", len(img), "image cells")
    edges = [0, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 7e-2, .1, .15, .2, .3, .5, .8, 1.01]
    h = np.histogram(m, edges)[0]
    for a, b, k in zip(edges[:-1], edges[1:], h):
        P(f"  {a:6.3f}-{b:6.3f} {k:4d} " + "#" * int(k))
    P("\n== share of tokens within one fp16 step of a tie in probability, by checkpoint (image cells, mean)")
    for tag in ("a10c", "a10cs2", "a10"):
        by = {}
        for c in img:
            if c["tag"] == tag:
                by.setdefault(_n(c["ck"]), []).append(c["within_step"])
        if by:
            P(f"{tag:7s} " + " ".join(f"{k}:{np.mean(v):.3f}" for k, v in sorted(by.items())))
    P("\n== precision, by checkpoint: mean |fp32-fp16| over image cells, largest cell")
    for tag in ("a10c", "a10cs2", "a10"):
        by = {}
        for c in img:
            if c["tag"] == tag and "depth_fp32" in c:
                by.setdefault(_n(c["ck"]), []).append(abs(c["depth_fp32"] - c["mean_depth"]))
        if by:
            P(f"{tag:7s} " + " ".join(f"{k}:{np.mean(v):.3f}/{np.max(v):.3f}" for k, v in sorted(by.items())))
    P("\n== precision")
    d = np.array([c["depth_fp32"] - c["mean_depth"] for c in img if "depth_fp32" in c])
    if len(d):
        P(f"|fp32-fp16| over {len(d)} image cells: median {np.median(np.abs(d)):.3f}, mean signed {d.mean():+.3f}, "
          f"> 0.1: {(np.abs(d) > .1).mean():.3f}, > 0.3: {(np.abs(d) > .3).mean():.3f}, max {np.abs(d).max():.3f}")
    d = np.array([c["depth_bf16"] - c["mean_depth"] for c in img if "depth_bf16" in c])
    if len(d):
        P(f"|bf16-fp16| over {len(d)} image cells: median {np.median(np.abs(d)):.3f}, mean signed {d.mean():+.3f}, "
          f"> 0.1: {(np.abs(d) > .1).mean():.3f}, > 0.3: {(np.abs(d) > .3).mean():.3f}, max {np.abs(d).max():.3f}")
    t = [c for c in img if "tie_to_shallower" in c]
    if t:
        n = sum(sum(c["tie_pairs"].values()) for c in t)
        P(f"exact ties in image cells: {n}, to the shallower depth {sum(c['tie_to_shallower'] * sum(c['tie_pairs'].values()) for c in t) / n:.4f}; "
          f"pairs {{12: {sum(c['tie_pairs']['12'] for c in t)}, 13: {sum(c['tie_pairs']['13'] for c in t)}, 23: {sum(c['tie_pairs']['23'] for c in t)}}}")
    tr = {(c["tag"], c["ck"], c["mod"]): c for c in cells if c["pass_"] == "train"}
    if tr:
        P("\n== held-out against training rows")
        for k, c in tr.items():
            h_ = next((x for x in main if (x["tag"], x["ck"], x["mod"]) == k), None)
            if h_:
                P(f"{k[0]:7s} {_n(k[1]):5d} {k[2]:7s} depth {h_['mean_depth']:.3f} / {c['mean_depth']:.3f}  gate {h_['gate']:.3f} / {c['gate']:.3f}  margin {h_['margin_med']:.4f} / {c['margin_med']:.4f}")
    P("\n== load: share of positions per depth, all positions and body tokens; gate at padding and delimiters")
    for r in cks:
        P(f"{r['tag']:7s} {_n(r['ck']):5d} all {np.round(r['all']['load'], 3)} H {r['all']['entropy']:.3f} | body {np.round(r['body']['load'], 3)} d {r['body']['mean_depth']:.3f} | "
          f"pad {np.round(r['pad']['load'], 3)} g{r['pad']['gate']:.3f} | delim {np.round(r['delim']['load'], 3)} g{r['delim']['gate']:.3f} | bo g{r['bo']['gate']:.3f} eo g{r['eo']['gate']:.3f}")
    (OUT / "summary.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
