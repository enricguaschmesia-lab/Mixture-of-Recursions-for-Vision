"""Summarise agent B's training runs from their curves.json (reporting helper).

    python -m eo.experiments.segmentation.step1_B_curves <run_dir> [<run_dir> ...] [--every 5000] [--png out.png]
Per run, a table of monitor reads (EMA, nearest and vote mIoU per dataset, their mean, the 100-step training
loss per dataset nearest that step, codes used in training batches since the previous read, the activation
probe's maximum when recorded) and the flattening check of the brief: the gain of the mean nearest mIoU over the
last quarter of the run. --png draws every run's monitor mIoU per dataset against steps in one figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DS = ("pastis", "cloudsen12", "lulc")
BAR = {"pastis": 0.85, "cloudsen12": 0.95, "lulc": 0.95}


def table(c: dict, every: int) -> None:
    tr = {r["step"]: r for r in c["train"]}
    ev = c["evals"]
    print(f"\nrun {c['run']}: {c['params']:,} params, gpu {c['gpu']}, wall {c.get('wall_s', 0) / 3600:.2f} h, "
          f"{len(ev)} reads, last step {ev[-1]['step'] if ev else 0}")
    print("| step | lr | train NLL p/c/l | PASTIS | CloudSen12 | LULC | mean (nearest) | mean (vote) | codes used | "
          "code entropy | act max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, e in enumerate(ev):
        s = e["step"]
        if every and s % every and i != len(ev) - 1:
            continue
        t = (tr.get(s) or tr[max(k for k in tr if k <= s)]) if tr else None
        loss = "/".join(f"{t['loss'][d]:.3f}" for d in DS) if t else "-"
        m = e["monitor"]
        act = f"{e['act']['max']:.0f}" if "act" in e else "-"
        print(f"| {s} | {e['lr']:.1e} | {loss} | " + " | ".join(f"{m[d]['nearest']['mIoU']:.4f}" for d in DS)
              + f" | {m['mean_nearest']:.4f} | {m['mean_vote']:.4f} | {e['train_codes']['distinct']} | "
              f"{e['train_codes']['entropy_bits']:.2f} | {act} |")
    if len(ev) >= 4:
        s = np.array([e["step"] for e in ev])
        v = np.array([e["monitor"]["mean_nearest"] for e in ev])
        j = int(np.argmin(np.abs(s - 0.75 * s[-1])))
        print(f"flattening: mean nearest {v[j]:.4f} at step {s[j]} (start of the last quarter) -> {v[-1]:.4f} at "
              f"{s[-1]}: gain {100 * (v[-1] - v[j]):+.2f} points; best {v.max():.4f} at step {s[v.argmax()]}")


def pair(c1: dict, c2: dict, every: int) -> None:
    """Two runs' monitor reads at the steps both read (e.g. a seed replicate): each dataset and the mean, and the
    second minus the first in points."""
    e1 = {e["step"]: e["monitor"] for e in c1["evals"]}
    e2 = {e["step"]: e["monitor"] for e in c2["evals"]}
    steps = sorted(set(e1) & set(e2))
    if every:
        steps = [s for s in steps if s % every == 0 or s == steps[-1]]
    print(f"\nmatched monitor reads: {c1['run']} (A) against {c2['run']} (B), nearest mIoU; diff = B - A in points")
    print("| step | PASTIS A / B | CloudSen12 A / B | LULC A / B | mean A / B | diff P/C/L/mean |")
    print("|---|---|---|---|---|---|")
    for s in steps:
        a, b = e1[s], e2[s]
        va = [a[d]["nearest"]["mIoU"] for d in DS] + [a["mean_nearest"]]
        vb = [b[d]["nearest"]["mIoU"] for d in DS] + [b["mean_nearest"]]
        print(f"| {s} | " + " | ".join(f"{x:.4f} / {y:.4f}" for x, y in zip(va, vb)) + " | "
              + "/".join(f"{100 * (y - x):+.2f}" for x, y in zip(va, vb)) + " |")


def figure(curves: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for k, c in enumerate(curves):
        ev = c["evals"]
        st = [e["step"] for e in ev]
        col = f"C{k}"
        for i, d in enumerate(DS):
            ax[i].plot(st, [e["monitor"][d]["nearest"]["mIoU"] for e in ev], color=col, lw=1.2, label=c["run"])
        tr = c["train"]
        ax[3].plot([r["step"] for r in tr], [np.mean([r["loss"][d] for d in DS]) for r in tr], color=col, lw=0.8,
                   label=c["run"])
    for i, d in enumerate(DS):
        ax[i].axhline(BAR[d], color="k", ls=":", lw=0.8)
        ax[i].set_title(f"{d} monitor mIoU (EMA, nearest); bar {BAR[d]}")
        ax[i].set_xlabel("step")
        lo = min(min(e["monitor"][d]["nearest"]["mIoU"] for e in c["evals"][2:] or c["evals"]) for c in curves)
        ax[i].set_ylim(max(0.0, lo - 0.01), 1.0)
        ax[i].grid(alpha=0.3)
    ax[3].set_yscale("log")
    ax[3].set_title("training NLL (native grid), mean of datasets")
    ax[3].set_xlabel("step")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--every", type=int, default=0, help="only reads at multiples of this step (and the last)")
    ap.add_argument("--png", default=None)
    ap.add_argument("--pair", action="store_true", help="two runs: their monitor reads at matched steps")
    a = ap.parse_args()
    curves = [json.loads((Path(r) / "curves.json").read_text()) for r in a.run_dirs]
    if a.pair:
        if len(curves) != 2:
            raise SystemExit("--pair takes exactly two run dirs")
        pair(curves[0], curves[1], a.every)
        if a.png:
            figure(curves, a.png)
        return
    for c in curves:
        table(c, a.every)
    if a.png:
        figure(curves, a.png)


if __name__ == "__main__":
    main()
