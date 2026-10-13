#!/usr/bin/env python
"""Phase 6 slide examples (B10 LoRA lr 1e-3, the best model): `mor` env, GPU only for decoding.

  examples_best.png   6 RANDOM test_a patches (seed 0): S2 RGB, S1 VV, and the S1+S2 label block
                      (✓ correct, ✗ wrong, ○ missed; the per-class marginal beside each).
  examples_chain.png  4 RANDOM rows of the Step 5b chain run (seed 0): S1, the real S2, the real S2's
                      tokens decoded, two generated Ŝ2 (S1 → S2, base generator, T = 1) decoded, and
                      the labels predicted from S1, S1 + Ŝ2 (sample 1) and S1 + S2.

Random, not picked: the rows come from a seeded draw. Decoding uses Phase 3 12.6b's S2L2A setting
(unclamped DiVAE, imported from s12_decoded_eval.py). RGB = B04/B03/B02 at one fixed scale everywhere.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python eo/experiments/ben_examples.py --out /data/enric/reports/ben/summary_b10
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
import s12_decoded_eval as S12  # noqa: E402  (terratorch decoders, Phase 3 settings)
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from eo.ben import labels as L, layout as Ly  # noqa: E402

REP = pathlib.Path("/data/enric/reports/ben")
BEN = pathlib.Path("/data/enric/data/BigEarthNet")
TEST = REP / "b10_lora/checkpoint-1461/test_a/S1S2.npz"
CHAIN = REP / "chain"
RGB_SCALE = 2800.0          # DN mapped to white


def rgb(b4, b3, b2):
    x = np.stack([b4, b3, b2], -1).astype(np.float32) / RGB_SCALE
    return np.clip(x, 0, 1) ** (1 / 1.3)


def s1_vv(vv):
    return np.clip((vv + 25.0) / 25.0, 0, 1)


class Packed:
    def __init__(self, n_rows):
        self.a2, self.r2p2 = Ly.load_packed("S2", n_rows)
        self.a1, self.r2p1 = Ly.load_packed("S1", n_rows)

    def s2_rgb(self, row):
        a = self.a2["S2_10m"][self.r2p2[row]]               # B02 B03 B04 B08, 120 px
        return rgb(a[2], a[1], a[0])

    def s1(self, row):
        return s1_vv(self.a1["S1"][self.r2p1[row]][0])


def decode_rgb(grids):
    tok = S12.TK.build("S2L2A", device=S12.TK.DEVICE)
    sched = S12.CT.scheduler_like(tok, **S12.CT.VARIANTS["noclip"])
    out = S12.decode(tok, "S2L2A", np.asarray(grids, np.int64).reshape(-1, 14, 14), sched)
    ims = []
    for k in range(len(out)):
        ph = S12.P.destandardize(out[k], "S2L2A").numpy()   # 12 bands, TerraMesh order: B04=3, B03=2, B02=1
        ims.append(rgb(ph[3], ph[2], ph[1]))
    return ims


def label_lines(truth, pred, marg=None):
    """[(text, colour)]: ✓ correct, ✗ wrong, ○ missed; sorted by the marginal."""
    order = np.argsort(-marg) if marg is not None else range(L.N_CLASSES)
    out = []
    for c in order:
        t, p = bool(truth[c]), bool(pred[c])
        if not (t or p):
            continue
        sym, col = ("✓", "#1a7f37") if t and p else (("✗", "#c62828") if p else ("○", "#8a8a8a"))
        m = f" {min(float(marg[c]), 1.0):.2f}" if marg is not None else ""      # the RB estimate can exceed 1 slightly
        out.append((f"{sym} {L.SHORT[c]}{m}", col))
    return out


def pred_line(ax, x, y, title, truth, pred, fs=8.5):
    ax.text(x, y, title, fontsize=fs, fontweight="bold", va="top", transform=ax.transAxes)
    xx = x + 0.17
    cs = list(np.flatnonzero(pred))
    for n, c in enumerate(cs):
        col = "#1a7f37" if truth[c] else "#c62828"
        txt = L.SHORT[c] + ("," if n < len(cs) - 1 else "")
        t = ax.text(xx, y, txt, fontsize=fs, color=col, va="top", transform=ax.transAxes)
        ax.figure.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transAxes.inverted())
        if bb.x1 > 0.99 and xx > x + 0.18:                 # does not fit: next line
            t.remove(); xx, y = x + 0.17, y - 0.13
            t = ax.text(xx, y, txt, fontsize=fs, color=col, va="top", transform=ax.transAxes)
            ax.figure.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transAxes.inverted())
        xx = bb.x1 + 0.012
    return y


def fig_best(P, out, n=6, seed=0):
    z = np.load(TEST)
    idx = np.sort(np.random.default_rng(seed).choice(len(z["rows"]), n, replace=False))
    fig = plt.figure(figsize=(15, 3 * 2.45))
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1.45] * 2, wspace=0.06, hspace=0.1)
    for k, i in enumerate(idx):
        r, c0 = divmod(k, 2)
        row = int(z["rows"][i])
        a1, a2, a3 = (fig.add_subplot(gs[r, 3 * c0 + j]) for j in range(3))
        a1.imshow(P.s2_rgb(row)); a2.imshow(P.s1(row), cmap="gray")
        for a, t in ((a1, "S2"), (a2, "S1 VV")):
            a.axis("off")
            if r == 0:
                a.set_title(t, fontsize=10)
        a3.axis("off")
        for j, (txt, col) in enumerate(label_lines(z["truth"][i], z["greedy"][i], z["marginal"][i])[:9]):
            a3.text(0.04, 0.97 - 0.105 * j, txt, color=col, fontsize=10, va="top", transform=a3.transAxes)
    fig.savefig(out / "examples_best.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return [int(z["rows"][i]) for i in idx]


def fig_chain(P, out, n=4, seed=0):
    g = np.load(CHAIN / "gen/base/S1toS2_T1.npz")
    cls = {k: np.load(CHAIN / f"cls/b10_lora/{f}.npz") for k, f in
           (("S1", "S1"), ("S1+Ŝ2", "S1+S2gen_base_T1_s0"), ("S1+S2", "S1S2"))}
    assert all((c["rows"] == g["rows"]).all() for c in cls.values())
    idx = np.sort(np.random.default_rng(seed).choice(len(g["rows"]), n, replace=False))
    rows = [int(g["rows"][i]) for i in idx]
    real_tok = np.load(BEN / "S2L2A_tok224/tokens.npy", mmap_mode="r")
    grids = [real_tok[r] for r in rows] + [g["tokens"][i, s] for i in idx for s in (0, 1)]
    ims = decode_rgb(np.stack(grids))
    rec, gen = ims[:n], ims[n:]
    fig = plt.figure(figsize=(17, 2.55 * n))
    gs = fig.add_gridspec(n, 6, width_ratios=[1, 1, 1, 1, 1, 2.6], wspace=0.04, hspace=0.06)
    heads = ["S1 VV (input)", "S2 (real)", "S2 tokens decoded", "Ŝ2 from S1 #1", "Ŝ2 from S1 #2"]
    for k, i in enumerate(idx):
        panels = [(P.s1(rows[k]), "gray"), (P.s2_rgb(rows[k]), None), (rec[k], None), (gen[2 * k], None), (gen[2 * k + 1], None)]
        for j, (im, cm) in enumerate(panels):
            a = fig.add_subplot(gs[k, j]); a.imshow(im, cmap=cm); a.axis("off")
            if k == 0:
                a.set_title(heads[j], fontsize=10)
        a = fig.add_subplot(gs[k, 5]); a.axis("off")
        truth = cls["S1"]["truth"][i].astype(bool)
        a.text(0.02, 0.97, "true:", fontsize=10.5, fontweight="bold", va="top", transform=a.transAxes)
        a.text(0.19, 0.97, ", ".join(L.SHORT[c] for c in np.flatnonzero(truth)), fontsize=10.5, va="top",
               transform=a.transAxes)
        y = 0.74
        for name, c in cls.items():
            y = pred_line(a, 0.02, y, name + ":", truth, c["greedy"][i].astype(bool), fs=10.5) - 0.19
    fig.savefig(out / "examples_chain.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    import pandas as pd
    n_rows = len(pd.read_parquet(Ly.INDEX, columns=["row"]))
    P = Packed(n_rows)
    print("best rows", fig_best(P, a.out))
    print("chain rows", fig_chain(P, a.out))


if __name__ == "__main__":
    main()
