#!/usr/bin/env python
"""B0 Step 3: the allocation between checkpoints, from replays saved every 10 steps.

Reads /data/enric/reports/b0/router/<replay>/checkpoint-*/probe_fp16.npz, one fixed probe of held-out
rows per saved state (eo/scripts/capture_router.py --sets probe), and the replay's own
trainer_state.json. Writes /data/enric/reports/b0/replay/<replay>.json and a report on stdout.

Per modality over time: mean depth, depth mix, mean probabilities, gate, margin. From those:
  autocorrelation and decorrelation time     what a 330-step checkpoint can stand for
  flips per interval                         against the 330-step count
  H2, load picks the side                    the load gap over the OTHER modalities at t against a
                                             tied modality's share at the deeper depth: its next
                                             change, and the partial correlation with the share at
                                             t+1 given the share at t. The first alone is regression
                                             to the mean whenever modalities move together and the
                                             series has no memory, which is what R2 showed
  gap against learning rate                  H3: does a tie's gap scale with the step size?
  step against margin (replays saved at every step)
                                             how far one update moves a modality's logit gap, against
                                             the spread of that gap over its tokens

Rules that read several replays: --h1p (R4 against R2), --h3 (R1a, R1b and the stored run),
--r4b (balancing at 0.01) and --r4a (R4 on A10c).

    .venv/bin/python eo/experiments/routing/b0_replay.py b0r2 b0r2f b0r1a b0r1b
    .venv/bin/python eo/experiments/routing/b0_replay.py --h1p --h3

Runs in `.venv`, CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402

ROOT = Path("/data/enric/reports/b0/router")
OUT = Path("/data/enric/reports/b0/replay")
RUNS = Path("/data/enric/runs/pretrain/phase3")
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
LAGS = (10, 20, 30, 50, 100, 200, 300)
FINE_LAGS = (1, 2, 3, 5, 10, 20, 30)
TIED_MARGIN, TIED_GATE = .05, .9


def softmax(w):
    w = w.astype(np.float64)
    w = w - w.max(-1, keepdims=True)
    p = np.exp(w)
    return p / p.sum(-1, keepdims=True)


def series(tag):
    files = sorted((ROOT / tag).glob("checkpoint-*/probe_fp16.npz"),
                   key=lambda f: int(f.parent.name.split("-")[1]))
    steps, rows = [], []
    for f in files:
        z = np.load(f)
        lg, d, lm, im = z["logits"], z["depth"], z["label_mod"], z["input_mod"]
        p = softmax(lg)
        srt = np.sort(p, -1)[..., ::-1]
        r = {"load": (np.bincount(d.ravel(), minlength=4)[1:] / d.size).tolist(), "n_all": int(d.size)}
        for j, m in enumerate(MODS):
            s = lm == j + 1
            m12 = srt[s][:, 0] - srt[s][:, 1]
            r[m] = dict(depth=float(d[s].mean()), mix=[float((d[s] == k).mean()) for k in (1, 2, 3)],
                        p=[float(x) for x in p[s].mean(0)], gate=float(srt[s][:, 0].mean()),
                        margin=float(np.median(m12)), n=int(s.sum()))
        steps.append(int(f.parent.name.split("-")[1]))
        rows.append(r)
    return np.array(steps), rows


def acf(x, k):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if k >= len(x) or x.var() == 0:
        return float("nan")
    return float((x[:-k] * x[k:]).mean() / x.var())


def learning_rates(tag, steps):
    f = RUNS / REPLAY_DIR.get(tag, tag) / f"checkpoint-{steps[-1]}" / "trainer_state.json"
    if not f.exists():
        return None
    lh = json.loads(f.read_text())["log_history"]
    lr = {e["step"]: e["learning_rate"] for e in lh if "learning_rate" in e}
    return np.array([lr.get(int(s), np.nan) for s in steps])


REPLAY_DIR: dict = {}


def partial(x, y0, y1):
    """Partial correlation of x with y1 given y0."""
    c0, c1, r = (np.corrcoef(a, b)[0, 1] for a, b in ((x, y0), (x, y1), (y0, y1)))
    return float((c1 - c0 * r) / np.sqrt((1 - c0 ** 2) * (1 - r ** 2)))


def h2(steps, rows, mods):
    """Load gap over the other modalities at t, against the deeper share, one state on."""
    out = {}
    for m in mods:
        p = np.array([r[m]["p"] for r in rows]).mean(0)
        i, k = sorted(np.argsort(p)[::-1][:2].tolist())           # the tied pair, i shallower
        share = np.array([r[m]["mix"][k] for r in rows])
        load = np.array(r_load(rows, m))
        gap = load[:, k] - load[:, i]
        x, y0, y1 = gap[:-1], share[:-1], share[1:]
        if x.std() == 0 or (y1 - y0).std() == 0 or y0.std() == 0:
            continue
        shifts = range(3, len(x) - 3)
        obs = float(np.corrcoef(x, y1 - y0)[0, 1])
        null = [np.corrcoef(np.roll(x, s), y1 - y0)[0, 1] for s in shifts]
        pa = partial(x, y0, y1)
        pnull = [partial(np.roll(x, s), y0, y1) for s in shifts]
        out[m] = dict(pair=[i + 1, k + 1], corr=obs,
                      null_95=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                      p=float(np.mean(np.abs(null) >= abs(obs))),
                      same_state=float(np.corrcoef(x, y0)[0, 1]), next_state=float(np.corrcoef(x, y1)[0, 1]),
                      partial=pa, partial_null_95=[float(np.percentile(pnull, 2.5)),
                                                   float(np.percentile(pnull, 97.5))],
                      partial_p=float(np.mean(np.abs(pnull) >= abs(pa))))
    return out


def r_load(rows, m):
    """Load over all positions with modality m's tokens taken out."""
    out = []
    for r in rows:
        L = np.array(r["load"]) * r["n_all"] - np.array(r[m]["mix"]) * r[m]["n"]
        out.append((L / L.sum()).tolist())
    return out


def report(tag):
    steps, rows = series(tag)
    if not len(steps):
        print(tag, "no captures")
        return None
    print(f"\n== {tag}: {len(steps)} states, steps {steps[0]}..{steps[-1]}, spacing {np.diff(steps).min()}")
    res = {"steps": steps.tolist(), "rows": rows}
    lr = learning_rates(tag, steps)
    if lr is not None:
        print("  lr", np.round(lr[[0, len(lr) // 2, -1]], 6))
    lags = FINE_LAGS if np.diff(steps).min() == 1 else LAGS
    print("  mod      depth: mean  sd  min  max | flips>=.5 | acf at", lags, "| gate mean sd | margin med")
    for m in MODS:
        d = np.array([r[m]["depth"] for r in rows])
        g = np.array([r[m]["gate"] for r in rows])
        mg = np.array([r[m]["margin"] for r in rows])
        step = int(np.diff(steps).min())
        ac = [acf(d, l // step) for l in lags]
        fl = int((np.abs(np.diff(d)) >= .5).sum())
        print(f"  {m:7s} {d.mean():.2f} {d.std():.2f} {d.min():.2f} {d.max():.2f} | {fl:3d} | "
              + " ".join(f"{a:+.2f}" for a in ac) + f" | {g.mean():.3f} {g.std():.3f} | {np.median(mg):.4f}")
    tied = tied_in(rows)
    res["h2"] = h2(steps, rows, tied)
    print("  H2, load gap (others) at t: corr with the next change, with s_t, with s_t+1, partial given s_t:")
    for m, v in res["h2"].items():
        print(f"    {m:7s} pair {v['pair']} change {v['corr']:+.3f} null [{v['null_95'][0]:+.3f}, {v['null_95'][1]:+.3f}] p {v['p']:.3f}"
              f" | s_t {v['same_state']:+.3f} s_t+1 {v['next_state']:+.3f}"
              f" | partial {v['partial']:+.3f} null [{v['partial_null_95'][0]:+.3f}, {v['partial_null_95'][1]:+.3f}] p {v['partial_p']:.3f}")
    n = sum(v["partial"] < v["partial_null_95"][0] and v["partial_p"] < .05 for v in res["h2"].values())
    print(f"  H2 rule, partial: {n} of {len(tied)} tied modalities negative beyond the null")
    if np.diff(steps).min() == 1:
        res["fine"] = fine(tag, tied, rows)
    if lr is not None and np.isfinite(lr).sum() > 5 and lr.max() / max(lr.min(), 1e-12) > 2:
        for m in tied:
            mg = np.array([r[m]["margin"] for r in rows])
            ok = np.isfinite(lr) & (lr > 0) & (mg > 0)
            print(f"  H3 {m:7s} slope of log gap on log lr {np.polyfit(np.log(lr[ok]), np.log(mg[ok]), 1)[0]:+.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(res) + "\n")
    return res


def tied_in(rows):
    return [m for m in MODS[:6] if np.median([r[m]["margin"] for r in rows]) <= TIED_MARGIN
            and np.mean([r[m]["gate"] for r in rows]) < TIED_GATE]


def fine(tag, tied, rows):
    """How far one update moves a tied modality's logit gap, on a replay saved at every step."""
    files = sorted((ROOT / tag).glob("checkpoint-*/probe_fp16.npz"),
                   key=lambda f: int(f.parent.name.split("-")[1]))
    out = {}
    print("  step against margin: per update, |move of the median gap| and the gap's spread over tokens (logits)")
    for m in tied:
        p = np.array([r[m]["p"] for r in rows]).mean(0)
        i, k = sorted(np.argsort(p)[::-1][:2].tolist())
        j = MODS.index(m) + 1
        gaps = []
        for f in files:
            z = np.load(f)
            s = z["label_mod"] == j
            lg = z["logits"][s].astype(np.float64)
            gaps.append(lg[:, k] - lg[:, i])
        g = np.array(gaps)                                     # state x token
        med = np.median(g, 1)
        move = np.abs(np.diff(med))
        spread = np.percentile(g, 75, 1) - np.percentile(g, 25, 1)
        dg = np.diff(g, axis=0)
        common = dg.mean(1, keepdims=True)
        share = 1 - ((dg - common) ** 2).sum() / (dg ** 2).sum()          # of a move, the part common to all tokens
        share_d = np.array([r[m]["mix"][k] for r in rows])
        out[m] = dict(pair=[i + 1, k + 1], move_median=float(np.median(move)), spread_median=float(np.median(spread)),
                      common_share=float(share), acf_gap=[acf(med, l) for l in FINE_LAGS],
                      acf_share=[acf(share_d, l) for l in FINE_LAGS])
        print(f"    {m:7s} pair {out[m]['pair']} move {out[m]['move_median']:.4f} spread {out[m]['spread_median']:.4f}"
              f" ratio {out[m]['move_median'] / out[m]['spread_median']:.2f} common part of a move {share:.3f}"
              f" | acf of the gap at {FINE_LAGS}: " + " ".join(f"{a:+.2f}" for a in out[m]["acf_gap"]))
    return out


def at(steps, rows, s):
    return rows[int(np.where(steps == s)[0][0])]


def rule_h1p(r4="b0r4", r2="b0r2"):
    """Margins of the modalities tied at step 1650, over steps 1810-1850, R4 against R2."""
    s2, a = series(r2)
    s4, b = series(r4)
    first = at(s2, a, 1650)
    tied = [m for m in MODS[:6] if first[m]["margin"] <= TIED_MARGIN and first[m]["gate"] < TIED_GATE]
    last = [1810, 1820, 1830, 1840, 1850]
    print(f"\n== H1': {r4} against {r2}, modalities tied at 1650: {tied}")
    print("  mod      margin R2  R4  ratio | gate R2 R4 | depth R2 R4 (last five states)")
    ratios = {}
    for m in MODS:
        ma = np.mean([at(s2, a, s)[m]["margin"] for s in last])
        mb = np.mean([at(s4, b, s)[m]["margin"] for s in last])
        ga, gb = (np.mean([at(x, y, s)[m]["gate"] for s in last]) for x, y in ((s2, a), (s4, b)))
        da, db = (np.mean([at(x, y, s)[m]["depth"] for s in last]) for x, y in ((s2, a), (s4, b)))
        if m in tied:
            ratios[m] = float(mb / ma)
        print(f"  {m:7s} {ma:.4f} {mb:.4f} {mb / ma:6.2f} | {ga:.3f} {gb:.3f} | {da:.2f} {db:.2f}" + ("  tied" if m in tied else ""))
    med = float(np.median(list(ratios.values())))
    print(f"  median ratio over the tied modalities {med:.2f}; H1' {'passes' if med >= 5 else 'fails'} at x5")
    return dict(tied=tied, ratios=ratios, median=med)


def rule_h3(stored="a10c", r1a="b0r1a", r1b="b0r1b", start=2970, end=3300):
    """At step 3300: stored run, R1a and R1b. Tied and single as at the stored start state."""
    st, sr = series(stored)
    s0 = at(st, sr, start)
    tied = [m for m in MODS[:6] if s0[m]["margin"] <= TIED_MARGIN and s0[m]["gate"] < TIED_GATE]
    single = [m for m in MODS[:6] if s0[m]["gate"] >= TIED_GATE]
    runs = {stored: (st, sr), r1a: series(r1a), r1b: series(r1b)}
    print(f"\n== H3 at {end}: tied at {start}: {tied}; single: {single}")
    worst = {}
    for m in MODS:
        d = [at(s, r, end)[m]["depth"] for s, r in runs.values()]
        worst[m] = float(max(d) - min(d))
        print(f"  {m:7s} " + " ".join(f"{x:.2f}" for x in d) + f"  largest difference {worst[m]:.2f}"
              + ("  tied" if m in tied else "  single" if m in single else ""))
    over = [m for m in tied if worst[m] > .3]
    bad = [m for m in single if worst[m] > .3]
    ok = len(over) >= len(tied) / 2 and not bad and len(tied) > 0
    print(f"  {len(over)} of {len(tied)} tied exceed 0.3, {len(bad)} of {len(single)} single do; H3 {'stands' if ok else 'fails'}")
    return dict(tied=tied, single=single, largest=worst, stands=bool(ok))


def rule_r4b(r4b="b0r4b", r2="b0r2"):
    """Balancing at 0.01: share of positions at depth 1 over steps 1810-1850, and margins against R2."""
    s, r = series(r4b)
    last = [1810, 1820, 1830, 1840, 1850]
    d1 = float(np.mean([at(s, r, k)["load"][0] for k in last]))
    verdict = "does not hold the split" if d1 >= .6 else "holds the split" if d1 < .45 else "partial"
    print(f"\n== R4b: share of positions at depth 1 over 1810-1850 = {d1:.3f}: balancing at 0.01 {verdict}")
    out = rule_h1p(r4b, r2)
    out.update(depth1=d1, verdict=verdict)
    return out


def rule_r4a(r4a="b0r4a", start=1650):
    """R4 on A10c: tied and single as at the start state; margins and depths over 1810-1850.

    The start state is the replay's own first save, a copy of the stored checkpoint (no probe was
    captured on the stored checkpoint-1650 of A10c)."""
    s, r = series(r4a)
    s0 = at(s, r, start)
    stored = f"{r4a}'s copy of the stored checkpoint"
    tied = [m for m in MODS[:6] if s0[m]["margin"] <= TIED_MARGIN and s0[m]["gate"] < TIED_GATE]
    single = [m for m in MODS[:6] if s0[m]["gate"] >= TIED_GATE]
    last = [1810, 1820, 1830, 1840, 1850]
    print(f"\n== R4a from {stored} {start}: tied {tied}, single {single}")
    print("  mod      margin start end  ratio | depth start end | gate start end")
    ratios, deeper, moved = {}, [], {}
    for m in MODS:
        me = float(np.mean([at(s, r, k)[m]["margin"] for k in last]))
        de = float(np.mean([at(s, r, k)[m]["depth"] for k in last]))
        ge = float(np.mean([at(s, r, k)[m]["gate"] for k in last]))
        ratio = me / max(s0[m]["margin"], 1e-9)
        if m in tied:
            ratios[m] = ratio
            if de > s0[m]["depth"] + .05:
                deeper.append(m)
        if m in single:
            moved[m] = abs(de - s0[m]["depth"])
        print(f"  {m:7s} {s0[m]['margin']:.4f} {me:.4f} {ratio:7.2f} | {s0[m]['depth']:.2f} {de:.2f} | {s0[m]['gate']:.3f} {ge:.3f}"
              + ("  tied" if m in tied else "  single" if m in single else ""))
    med = float(np.median(list(ratios.values()))) if ratios else float("nan")
    repeats = bool(ratios) and med >= 5 and not deeper
    trap = "holds" if all(v <= .05 for v in moved.values()) else "fails" if any(v > .3 for v in moved.values()) else "unclear"
    print(f"  median ratio {med:.2f}, ending deeper: {deeper}; R4 {'repeats' if repeats else 'does not repeat'}; the trap {trap} ({moved})")
    return dict(tied=tied, single=single, ratios=ratios, median=med, deeper=deeper, repeats=repeats, trap=trap, moved=moved)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    if "--h1p" in args:
        (OUT / "h1p.json").write_text(json.dumps(rule_h1p()) + "\n")
    if "--h3" in args:
        (OUT / "h3.json").write_text(json.dumps(rule_h3()) + "\n")
    if "--r4b" in args:
        (OUT / "r4b.json").write_text(json.dumps(rule_r4b()) + "\n")
    if "--r4a" in args:
        (OUT / "r4a.json").write_text(json.dumps(rule_r4a()) + "\n")
    for spec in (a for a in args if not a.startswith("--")):
        tag, _, d = spec.partition("=")
        if d:
            REPLAY_DIR[tag] = d
        report(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
