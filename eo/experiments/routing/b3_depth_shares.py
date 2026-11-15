#!/usr/bin/env python
"""B3's train-split control, prediction 1: do the ties and their 3-4-update ringing persist on data seen once?

Reads a run's per-update depth shares (`depth_shares.jsonl`, written by util/trainer_pt.py since wave 1, W31): for
every update, each class's positions counted at each depth (label frame: a position counts under the class of the
token it predicts). B0 found on val (replays saved at every update, held-out probes) that a learned router is a
per-modality table whose tied modalities swing WHOLE between two depths with a period of 3-4 updates, held there by
the balancing term's feedback on a hard count (docs/notes/routing/b0_audit_2026-11-10.md). This reads the same
thing from the training batches' own counts, on time means over B0's windows.

Per modality and window (early 330-990, plateau 990-2970, late 1980-3300, anneal 2970-3300; steps inclusive):
  shares       the time-mean share at each depth, and the mean depth
  whole        the fraction of updates at which >= 90% of the modality's positions sit at one depth
  state        single d   time-mean share at d >= 0.9
               tie a/b    otherwise, if whole >= 0.5 and the two largest time-mean shares are each >= 0.1:
                          the modality moves whole between depths a < b
               split      otherwise (its positions divide WITHIN updates: a random router, or token identity)
  ringing      for a tie: the share at its deeper depth b as a series over the window's updates; its
               autocorrelation at lags 1-8, the periodogram's peak period over 2-50 updates with its power over
               the band's mean, a permutation p for that ratio (the series shuffled in time, 999 draws), and the
               mean run length at one whole depth
  load         the share of ALL counted positions (padding and delimiters included: they count in the balancing
               statistic) at the deepest depth, as a series, with the same period test

Controls, which must fail the checks: --self-test plants (a) a tie swinging with period 4, which must read
"tie 1/3" with a period of 4 and p < 0.01; (b) a tie flipping without memory, which must read "tie" with p > 0.05;
(c) a uniform random router, which must read "split" everywhere. On real data the random-router twin of the same
cell (A10cR-tr for A10c-tr) is the control: every modality "split", at 1/3 per depth.

    .venv/bin/python eo/experiments/routing/b3_depth_shares.py --self-test
    .venv/bin/python eo/experiments/routing/b3_depth_shares.py a10ctr /data/enric/runs/pretrain/phase3/mor10ctr_...

Writes /data/enric/reports/wave1/depth_shares/<tag>.{json,txt}. Runs in `.venv`, CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("/data/enric/reports/wave1/depth_shares")
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
WINDOWS = {"early": (330, 990), "plateau": (990, 2970), "late": (1980, 3300), "anneal": (2970, 3300)}
WHOLE, SINGLE, TIE_SHARE, TIE_WHOLE = 0.9, 0.9, 0.1, 0.5
LAGS = range(1, 9)
PERIODS = (2.0, 50.0)
DRAWS = 999


def load(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """classes, steps (T,), counts (T, class, depth). A resume may append a header and repeat steps: keep the last."""
    classes, by_step = None, {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "header" in r:
            c = r["header"]["classes"]
            if classes is not None and c != classes:
                raise SystemExit(f"{path}: a later header changes the classes")
            classes = c
            continue
        by_step[int(r["step"])] = r["counts"]
    if classes is None or not by_step:
        raise SystemExit(f"{path}: no header or no rows")
    steps = np.array(sorted(by_step))
    return classes, steps, np.array([by_step[s] for s in steps], dtype=np.float64)


def acf(x: np.ndarray, k: int) -> float:
    x = x - x.mean()
    v = (x * x).mean()
    return float("nan") if k >= len(x) or v == 0 else float((x[:-k] * x[k:]).mean() / v)


def peak(x: np.ndarray, rng: np.random.Generator) -> dict:
    """Periodogram peak over periods 2-50 updates, its power over the band's mean, and a permutation p."""
    n = len(x)
    f = np.fft.rfftfreq(n)
    band = (f >= 1 / PERIODS[1]) & (f <= 1 / PERIODS[0])
    if n < 20 or band.sum() < 3 or x.std() == 0:
        return dict(period=None, ratio=None, p=None)

    def ratio(y):
        p = np.abs(np.fft.rfft(y - y.mean())) ** 2
        b = p[band]
        return b.max() / b.mean(), b.argmax()

    r, i = ratio(x)
    null = np.array([ratio(rng.permutation(x))[0] for _ in range(DRAWS)])
    return dict(period=float(1 / f[band][i]), ratio=float(r), p=float((1 + (null >= r).sum()) / (DRAWS + 1)))


def runs(modal: np.ndarray) -> float:
    """Mean length of the runs of updates at one whole depth."""
    if len(modal) < 2:
        return float("nan")
    cuts = np.flatnonzero(np.diff(modal) != 0)
    return float(len(modal) / (len(cuts) + 1))


def read(classes, steps, counts, rng, windows=None) -> dict:
    """`windows` {name: (first, last step)}, inclusive; B0's four by default (B1's smoke reader passes its own)."""
    nd = counts.shape[2]
    depths = np.arange(1, nd + 1)
    out = {"num_depths": nd, "steps": [int(steps[0]), int(steps[-1])], "n_updates": int(len(steps)), "windows": {}}
    for w, (a, b) in (WINDOWS if windows is None else windows).items():
        sel = (steps >= a) & (steps <= b)
        if sel.sum() < 20:
            continue
        res = {"n_updates": int(sel.sum()), "mods": {}}
        for m in MODS:
            if m not in classes:
                continue
            c = counts[sel, classes.index(m), :]
            tot = c.sum(1)
            ok = tot > 0
            if not ok.any():   # no positions of this class in the window (Coords in a Coords-free run): left out
                continue
            s = c[ok] / tot[ok, None]
            mean = s.mean(0)
            whole_t = s.max(1) >= WHOLE
            modal = s.argmax(1)
            top = np.argsort(mean)[::-1]
            r = dict(shares=mean.round(4).tolist(), mean_depth=float((mean * depths).sum()),
                     whole=float(whole_t.mean()), positions_per_update=float(tot.mean()))
            if mean[top[0]] >= SINGLE:
                r["state"] = f"single {top[0] + 1}"
            elif whole_t.mean() >= TIE_WHOLE and mean[top[1]] >= TIE_SHARE:
                lo, hi = sorted((int(top[0]), int(top[1])))
                x = s[:, hi]
                r["state"] = f"tie {lo + 1}/{hi + 1}"
                r["ringing"] = dict(acf=[round(acf(x, k), 3) for k in LAGS], run_length=runs(modal[whole_t]),
                                    **peak(x, rng))
            else:
                r["state"] = "split"
            res["mods"][m] = r
        keep = [i for i, c in enumerate(classes) if c != "none"]
        allc = counts[sel][:, keep, :].sum(1)
        ld = allc[:, -1] / allc.sum(1)
        res["load"] = dict(shares=(allc.sum(0) / allc.sum()).round(4).tolist(), deepest=dict(
            mean=float(ld.mean()), min=float(ld.min()), max=float(ld.max()),
            acf=[round(acf(ld, k), 3) for k in LAGS], **peak(ld, rng)))
        out["windows"][w] = res
    return out


def text(tag: str, src: str, r: dict) -> str:
    L = [f"{tag}: per-update depth shares, {src}", f"  updates {r['n_updates']} (steps {r['steps'][0]}-{r['steps'][1]}),"
         f" {r['num_depths']} depths; tie = whole >= {TIE_WHOLE} and two time-mean shares >= {TIE_SHARE}"]
    for w, res in r["windows"].items():
        a, b = WINDOWS[w]
        ld = res["load"]["deepest"]
        per = f"period {ld['period']:.2f} (x{ld['ratio']:.1f}, p {ld['p']:.3f})" if ld["period"] else "no period"
        L.append(f"  {w} {a}-{b} ({res['n_updates']} updates): load {res['load']['shares']}, deepest {ld['min']:.2f}"
                 f"-{ld['max']:.2f}, {per}")
        for m, x in res["mods"].items():
            s = f"    {m:7s} {x['state']:9s} shares {x['shares']} depth {x['mean_depth']:.2f} whole {x['whole']:.2f}"
            g = x.get("ringing")
            if g:
                s += (f" | period {g['period']:.2f} (x{g['ratio']:.1f}, p {g['p']:.3f}), runs {g['run_length']:.1f},"
                      f" acf {' '.join(f'{v:+.2f}' for v in g['acf'][:5])}" if g["period"] else " | no period")
            L.append(s)
    return "\n".join(L)


def synthetic(kind: str, rng: np.random.Generator, n: int = 3300, per: int = 200) -> tuple:
    classes = ["none"] + MODS + ["delim", "pad"]
    counts = np.zeros((n, len(classes), 3))
    for t in range(n):
        for i, m in enumerate(classes[1:], 1):
            if kind == "uniform" or m != "NDVI":
                counts[t, i] = rng.multinomial(per, [1 / 3] * 3)
                continue
            if kind == "ring4":
                up = (t // 2) % 2 == 1 if rng.random() > .05 else rng.random() < .5   # 2 at 1, 2 at 3: period 4
            else:
                up = rng.random() < .5                                                # no memory
            counts[t, i] = [0, 0, per] if up else [per, 0, 0]
    return classes, np.arange(1, n + 1), counts


def self_test() -> int:
    rng = np.random.default_rng(0)
    ok = True
    for kind in ("ring4", "flip", "uniform"):
        r = read(*synthetic(kind, rng), rng)["windows"]["plateau"]["mods"]
        nd = r["NDVI"]
        other = {m: x["state"] for m, x in r.items() if m != "NDVI"}
        g = nd.get("ringing", {})
        if kind == "ring4":
            good = nd["state"] == "tie 1/3" and g.get("period") and abs(g["period"] - 4) < .5 and g["p"] < .01
        elif kind == "flip":
            good = nd["state"] == "tie 1/3" and (g.get("p") is None or g["p"] > .05)
        else:
            good = nd["state"] == "split"
        good = good and set(other.values()) == {"split"}
        ok &= bool(good)
        print(f"  self-test {kind:7s}: NDVI {nd['state']}, period {g.get('period')}, p {g.get('p')}; others "
              f"{sorted(set(other.values()))} -> {'PASS' if good else 'FAIL'}")
    classes, steps, counts = synthetic("uniform", rng)   # a class with no positions (Coords in a Coords-free run)
    counts[:, classes.index("Coords"), :] = 0
    r = read(classes, steps, counts, rng)["windows"]["plateau"]["mods"]
    good = "Coords" not in r and set(x["state"] for x in r.values()) == {"split"} and len(r) == len(MODS) - 1
    ok &= bool(good)
    print(f"  self-test absent : Coords {'left out' if 'Coords' not in r else r['Coords']['state']}, "
          f"{len(r)} classes read -> {'PASS' if good else 'FAIL'}")
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag", nargs="?")
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.tag and a.run_dir):
        ap.error("tag and run_dir, or --self-test")
    src = a.run_dir / "depth_shares.jsonl"
    r = read(*load(src), np.random.default_rng(0))
    r["source"] = str(src)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(r, indent=1))
    t = text(a.tag, str(src), r)
    (OUT / f"{a.tag}.txt").write_text(t + "\n")
    print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
