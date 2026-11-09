#!/usr/bin/env python
"""B0 Step 4: cross-seed agreement and stability of the router's allocation, on time means.

A tied modality is redrawn at every checkpoint (B0 Step 3: the balancing loop rings every 3-4
updates), so one checkpoint describes a phase, not an allocation. Everything that decides here is
read on TIME MEANS over the windows of decision 11, and agreement is judged against the exact
permutation of modality labels, which keeps each run's profile and so its load. The rules and how
they are computed are in docs/plans/b0_router_audit.md, Step 4, fixed before this was first run.

Reads (all written by eo/scripts/capture_router.py, and by eval_checkpoint.py for the landscapes):
  /data/enric/reports/b0/router/{a10c,a10cs2,a10}/checkpoint-*/main_{fp16,fp32}.npz   4,416 rows
  /data/enric/reports/b0/router/{a10c,a10cs2}/checkpoint-*/target_last_fp16.npz      target-last
  /data/enric/reports/b0/router/<run or replay>/checkpoint-*/probe_fp16.npz          256 rows
  /data/enric/reports/arm_eval/{a10c,a10cs2}_{tforce1,tforce2,tforce3,routed}/       landscapes
Writes /data/enric/reports/b0/agree/: agree.json, report.txt, timemeans.csv, cache/.

    .venv/bin/python eo/experiments/routing/b0_agree.py            [--no-cache]

Per (run, modality, window): time-mean share at each depth, mean depth, gate and median margin,
with a 95% interval sqrt((t * SD/sqrt(n))^2 + h^2): SD over the window's checkpoints (Student's t,
n - 1 degrees of freedom) and h the half-width of a group bootstrap of the time mean (the 256
spatial groups, one draw applied at every checkpoint). Per pair of runs: Spearman of time means
with its exact p, mean |difference| against the same permutation null, tie sets, gate and states.

Runs in `.venv`, CPU. Numpy only (no scipy in `.venv`: Student's t is computed here, and checked).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402

ROUTER = Path("/data/enric/reports/b0/router")
ARM = Path("/data/enric/reports/arm_eval")
OUT = Path("/data/enric/reports/b0/agree")
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
IMG = MODS[:6]
CKS = [330 * i for i in range(1, 11)]
RUNS = ("a10c", "a10cs2", "a10")
PAIRS = (("a10c", "a10cs2"), ("a10c", "a10"), ("a10cs2", "a10"))
PRIMARY = ("a10c", "a10cs2")
WINDOWS = {"early": [330, 660, 990],
           "plateau": [990, 1320, 1650, 1980, 2310, 2640, 2970],
           "late": [1980, 2310, 2640, 2970, 3300],
           "anneal": [2970, 3300]}
# Criterion 2's pair (the rule), and the descriptive pair: 1980 goes to the second window only.
STABILITY = {"rule_1980-2640_vs_anneal": ([1980, 2310, 2640], [2970, 3300]),
             "990-1980_vs_1980-2970": ([990, 1320, 1650], [1980, 2310, 2640, 2970])}
REPLAYS = {"b0r2": "a10cs2", "b0r2f": "a10cs2", "b0r3": "a10c", "b0r1a": "a10c", "b0r1b": "a10c"}
TIE_SHARE, TIED_MARGIN, SINGLE_GATE, PIN_SHARE, BAND = 0.02, 0.05, 0.9, 0.98, 0.3
ABBR = {"stable": "S", "unstable": "X", "undetermined": "?"}
N_BOOT = 10_000


# ------------------------------------------------------------------ Student's t, numerically

def _t_pdf(x, nu):
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    return np.exp(c - (nu + 1) / 2 * np.log1p(x * x / nu))


def _t_cdf_pos(x, nu, n=20_000):
    """P(T <= x) for x >= 0, by Simpson's rule on [0, x]."""
    if x == 0:
        return 0.5
    g = np.linspace(0.0, x, n + 1)
    f = _t_pdf(g, nu)
    return 0.5 + (x / n / 3) * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())


@lru_cache(maxsize=None)
def t_quantile(p: float, nu: float) -> float:
    """The p quantile of Student's t with nu degrees of freedom (p > 0.5), by bisection."""
    if not math.isfinite(nu) or nu > 1e6:
        lo, hi = 0.0, 10.0
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p else (lo, mid)
        return (lo + hi) / 2
    if nu == 1:
        return math.tan(math.pi * (p - 0.5))
    lo, hi = 0.0, 1.0
    while _t_cdf_pos(hi, nu) < p:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _t_cdf_pos(mid, nu) < p else (lo, mid)
    return (lo + hi) / 2


def check_t():
    """Known two-sided 95% values; a check that would fail on a wrong density or rule."""
    known = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 6: 2.4469, 33: 2.0345, 1e9: 1.95996}
    bad = {k: (round(t_quantile(0.975, k), 4), v) for k, v in known.items()
           if abs(t_quantile(0.975, k) - v) > 2e-4}
    if bad:
        raise RuntimeError(f"Student's t is wrong: {bad}")
    # control: the normal quantile is NOT the df=2 value, so the check can fail
    assert abs(t_quantile(0.975, 1e9) - 4.3027) > 1


# ------------------------------------------------------------------ reading the captures

def softmax(w):
    w = w.astype(np.float64)
    w = w - w.max(-1, keepdims=True)
    p = np.exp(w)
    return p / p.sum(-1, keepdims=True)


def reduce(lg, d, lm, groups, n_groups, mods=MODS):
    """Per modality: per-group token count, depth sum, counts at each depth, gate sum; median margin."""
    p = softmax(lg)
    top = p.max(-1)
    second = np.sort(p, -1)[..., -2]
    out = {}
    for j, m in enumerate(MODS):
        if m not in mods:
            continue
        s = lm == j + 1
        if not s.any():
            continue
        g = groups[np.nonzero(s)[0]]
        dd = d[s].astype(np.int64)
        out[m] = dict(
            N=np.bincount(g, minlength=n_groups).astype(np.float64),
            S=np.bincount(g, weights=dd.astype(np.float64), minlength=n_groups),
            K=np.stack([np.bincount(g, weights=(dd == k).astype(np.float64), minlength=n_groups)
                        for k in (1, 2, 3)], 1),
            G=np.bincount(g, weights=top[s], minlength=n_groups),
            margin=float(np.median(top[s] - second[s])))
    return out


def summary(r):
    """Cell numbers from one reduce() entry."""
    n = r["N"].sum()
    return dict(depth=r["S"].sum() / n, mix=(r["K"].sum(0) / n).tolist(), gate=r["G"].sum() / n,
                margin=r["margin"], n=int(n))


def _pack(red):
    flat = {}
    for m, r in red.items():
        for k, v in r.items():
            flat[f"{m}|{k}"] = np.asarray(v)
    return flat


def _unpack(z):
    red = {}
    for k in z.files:
        if "|" not in k:
            continue
        m, f = k.split("|")
        red.setdefault(m, {})[f] = z[k] if f != "margin" else float(z[k])
    return red


def load_set(tag, ck, name, cache, groups_of):
    """reduce() of one capture, cached. name: main_fp16, main_fp32, probe_fp16."""
    cf = OUT / "cache" / tag / f"{ck}_{name}.npz"
    if cache and cf.exists():
        z = np.load(cf, allow_pickle=False)
        return _unpack(z), z
    f = ROUTER / tag / ck / f"{name}.npz"
    if not f.exists():
        return None, None
    z = np.load(f)
    groups, ng = groups_of(z["rows"])
    red = reduce(z["logits"], z["depth"], z["label_mod"], groups, ng)
    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cf, **_pack(red), rows=z["rows"])
    return red, np.load(cf)


def load_target_last(tag, ck, cache, groups_of):
    """Per target: reduce() at the positions predicting that target, in target-last rows."""
    cf = OUT / "cache" / tag / f"{ck}_target_last_fp16.npz"
    if cache and cf.exists():
        return _unpack(np.load(cf))
    f = ROUTER / tag / ck / "target_last_fp16.npz"
    if not f.exists():
        return None
    z = np.load(f)
    red = {}
    for j, T in enumerate(MODS):
        groups, ng = groups_of(z[f"{T}|rows"])
        r = reduce(z[f"{T}|logits"], z[f"{T}|depth"], z[f"{T}|label_mod"], groups, ng, mods=[T])
        red.update(r)
    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cf, **_pack(red))
    return red


class Groups:
    """Spatial groups of a row set, as 0..G-1, memoised by the row set."""

    def __init__(self):
        self.memo = {}

    def __call__(self, rows):
        key = np.asarray(rows, dtype=np.int64).tobytes()
        if key not in self.memo:
            g = S.eval_groups(rows)
            self.memo[key] = (g, int(g.max()) + 1)
        return self.memo[key]


# ------------------------------------------------------------------ time means and intervals

def _boot_W(n_groups, seed=20261108):
    return S.boot_weights(n_groups, N_BOOT, seed)


def boot_timemean(cells, W, key="S"):
    """(draws,) time mean over the cells of a window, one group draw applied to every cell.
    key "S": mean depth; "G": gate."""
    return np.mean([(W @ r[key]) / (W @ r["N"]) for r in cells], 0)


def interval(series, boot):
    """Time mean, its 95% half-width, and the two parts."""
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    h_t = t_quantile(0.975, n - 1) * sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    h_b = float(hi - lo) / 2
    return dict(mean=float(x.mean()), half=math.sqrt(h_t ** 2 + h_b ** 2), h_t=h_t, h_b=h_b, sd=sd, n=n)


def diff_interval(x1, x2, b1, b2):
    """Difference of two window means: Welch for the temporal part, the bootstrap for the rows."""
    x1, x2 = np.asarray(x1, dtype=np.float64), np.asarray(x2, dtype=np.float64)
    n1, n2 = len(x1), len(x2)
    v1 = x1.var(ddof=1) / n1 if n1 > 1 else 0.0
    v2 = x2.var(ddof=1) / n2 if n2 > 1 else 0.0
    se = math.sqrt(v1 + v2)
    if se > 0:
        df = se ** 4 / ((v1 ** 2 / (n1 - 1) if n1 > 1 and v1 > 0 else 0.0)
                        + (v2 ** 2 / (n2 - 1) if n2 > 1 and v2 > 0 else 0.0))
        h_t = t_quantile(0.975, df) * se
    else:
        df, h_t = float("inf"), 0.0
    lo, hi = np.percentile(np.asarray(b1) - np.asarray(b2), [2.5, 97.5])
    h_b = float(hi - lo) / 2
    est = float(x1.mean() - x2.mean())
    half = math.sqrt(h_t ** 2 + h_b ** 2)
    if est - half >= -BAND and est + half <= BAND:
        verdict = "stable"
    elif est - half > BAND or est + half < -BAND:
        verdict = "unstable"
    else:
        verdict = "undetermined"
    return dict(diff=est, half=half, h_t=h_t, h_b=h_b, df=df, verdict=verdict)


def window_table(cells_by_ck, cks, W):
    """Per modality: time means with intervals over the checkpoints `cks` (present ones only)."""
    have = [c for c in cks if c in cells_by_ck]
    out = {}
    for m in MODS:
        rs = [cells_by_ck[c][m] for c in have if m in cells_by_ck[c]]
        if len(rs) < len(have) or not rs:
            continue
        sm = [summary(r) for r in rs]
        mixes = np.array([s["mix"] for s in sm])
        out[m] = dict(
            cks=have,
            depth=interval([s["depth"] for s in sm], boot_timemean(rs, W)),
            gate=interval([s["gate"] for s in sm], boot_timemean(rs, W, key="G")),
            mix=mixes.mean(0).tolist(),
            margin=float(np.mean([s["margin"] for s in sm])),
            per_ck_depth=[s["depth"] for s in sm],
            per_ck_mix=mixes.tolist(),
            boot_depth=boot_timemean(rs, W))
    return out


def tie_set(mix):
    return tuple(k + 1 for k, v in enumerate(mix) if v >= TIE_SHARE)


def state(t):
    """States on time means, in the plan's order."""
    if t["gate"]["mean"] >= SINGLE_GATE:
        return "single"
    mix = sorted(t["mix"], reverse=True)
    if t["margin"] <= TIED_MARGIN and mix[1] >= TIE_SHARE:
        return "tied"
    per = np.array(t["per_ck_mix"])
    top = per.argmax(1)
    if t["margin"] <= TIED_MARGIN and len(set(top.tolist())) == 1 and (per.max(1) >= PIN_SHARE).all():
        return "pin"
    return "other"


# ------------------------------------------------------------------ agreement

PERMS6 = S._perms(6)


def meanabs(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    obs = float(np.abs(a - b).mean())
    null = np.abs(a[None, :] - b[PERMS6]).mean(1)
    return obs, float(np.mean(null <= obs + 1e-12)), float(np.abs(a - b).max())


def tie_agree(ta, tb):
    obs = sum(x == y for x, y in zip(ta, tb))
    null = np.array([sum(ta[i] == tb[p[i]] for i in range(6)) for p in PERMS6])
    return int(obs), float(np.mean(null >= obs))


def kappa(sa, sb):
    def k(x, y):
        po = np.mean([u == v for u, v in zip(x, y)])
        cats = set(x) | set(y)
        pe = sum((x.count(c) / len(x)) * (y.count(c) / len(y)) for c in cats)
        return float("nan") if pe >= 1 else float((po - pe) / (1 - pe))
    obs = k(list(sa), list(sb))
    if not math.isfinite(obs):
        return obs, float("nan")
    null = np.array([k(list(sa), [sb[p[i]] for i in range(6)]) for p in PERMS6])
    return obs, float(np.mean(null >= obs - 1e-12))


def agree(ta, tb):
    """Agreement of two runs' window tables over the six image modalities."""
    da = [ta[m]["depth"]["mean"] for m in IMG]
    db = [tb[m]["depth"]["mean"] for m in IMG]
    ga = [ta[m]["gate"]["mean"] for m in IMG]
    gb = [tb[m]["gate"]["mean"] for m in IMG]
    mad, p_mad, mx = meanabs(da, db)
    tsa, tsb = [tie_set(ta[m]["mix"]) for m in IMG], [tie_set(tb[m]["mix"]) for m in IMG]
    nt, p_t = tie_agree(tsa, tsb)
    sa, sb = [state(ta[m]) for m in IMG], [state(tb[m]) for m in IMG]
    kp, p_k = kappa(sa, sb)
    gmad, p_gmad, gmx = meanabs(ga, gb)
    rho = S.spearman(da, db)
    return dict(rho=rho, p=S.exact_p(da, db), meanabs=mad, p_meanabs=p_mad, maxabs=mx,
                crit1=bool(rho >= 0.9 and mx <= BAND),
                ties_a=[list(t) for t in tsa], ties_b=[list(t) for t in tsb], ties_match=nt, p_ties=p_t,
                gate_rho=S.spearman(ga, gb), gate_p=S.exact_p(ga, gb), gate_meanabs=gmad,
                gate_p_meanabs=p_gmad, states_a=sa, states_b=sb, kappa=kp, p_kappa=p_k)


# ------------------------------------------------------------------ landscapes

def landscapes(cks=CKS):
    """CE at depths 1-3 per target (target-last, target-only forcing, A100), with the router's depth."""
    out = {}
    for tag in ("a10c", "a10cs2"):
        for ck in cks:
            arrs = {}
            for k in ("tforce1", "tforce2", "tforce3", "routed"):
                f = ARM / f"{tag}_{k}" / f"checkpoint-{ck}" / "target_last_rows.npz"
                if not f.exists():
                    break
                arrs[k] = np.load(f)
            if len(arrs) < 4:
                continue
            row = {}
            for T in IMG:
                rows = arrs["routed"][f"{T}|rows"]
                for k in ("tforce1", "tforce2", "tforce3"):
                    if not np.array_equal(arrs[k][f"{T}|rows"], rows):
                        raise RuntimeError(f"{tag} {ck} {T}: {k} rows differ from routed")
                ce = [float(arrs[k][f"{T}|ce_sum"].sum() / arrs[k][f"{T}|n"].sum())
                      for k in ("tforce1", "tforce2", "tforce3")]
                dp = arrs["routed"][f"{T}|d_pred"].sum(0)
                ce_r = float(arrs["routed"][f"{T}|ce_sum"].sum() / arrs["routed"][f"{T}|n"].sum())
                row[T] = dict(ce=ce, ce_router=ce_r, g13=ce[0] - ce[2], g12=ce[0] - ce[1],
                              best=int(np.argmin(ce)) + 1, router_depth=float((dp * np.arange(1, 4)).sum() / dp.sum()),
                              router_modal=int(np.argmax(dp)) + 1)
            out[(tag, ck)] = row
    return out


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    cache = not args.no_cache
    check_t()
    OUT.mkdir(parents=True, exist_ok=True)
    groups_of = Groups()
    lines = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        lines.append(s)
        print(s, flush=True)

    # ---- per-cell reductions
    cells = {}            # (tag, prec) -> {ck: red}
    rows_ref = {}
    for tag in RUNS:
        for prec in ("fp16", "fp32"):
            for ck in CKS:
                red, z = load_set(tag, f"checkpoint-{ck}", f"main_{prec}", cache, groups_of)
                if red is None:
                    continue
                rr = z["rows"]
                if tag in rows_ref and not np.array_equal(rows_ref[tag], rr):
                    raise RuntimeError(f"{tag} {ck} {prec}: rows differ between captures")
                rows_ref[tag] = rr
                cells.setdefault((tag, prec), {})[ck] = red
    for a, b in PAIRS:
        if not np.array_equal(rows_ref[a], rows_ref[b]):
            raise RuntimeError(f"{a} and {b} read different rows")
    g_main, ng_main = groups_of(rows_ref["a10c"])
    W_main = _boot_W(ng_main)
    P(f"== B0 Step 4: {len(rows_ref['a10c'])} held-out rows in {ng_main} spatial groups, {N_BOOT} draws; "
      f"Student's t checked (df 1: {t_quantile(0.975, 1):.3f}, df 2: {t_quantile(0.975, 2):.3f})")

    tl = {}
    for tag in ("a10c", "a10cs2"):
        for ck in CKS:
            red = load_target_last(tag, f"checkpoint-{ck}", cache, groups_of)
            if red is not None:
                tl.setdefault(tag, {})[ck] = red

    probe = {}
    probe_rows = None
    for tag in list(RUNS) + list(REPLAYS):
        cks = CKS if tag in RUNS else sorted(int(p.parent.name.split("-")[1])
                                             for p in (ROUTER / tag).glob("checkpoint-*/probe_fp16.npz"))
        for ck in cks:
            red, z = load_set(tag, f"checkpoint-{ck}", "probe_fp16", cache, groups_of)
            if red is None:
                continue
            if probe_rows is None:
                probe_rows = z["rows"]
            elif not np.array_equal(probe_rows, z["rows"]):
                raise RuntimeError(f"{tag} {ck}: probe rows differ")
            probe.setdefault(tag, {})[ck] = red
    g_pr, ng_pr = groups_of(probe_rows)
    W_pr = _boot_W(ng_pr)

    # ---- window tables
    res = {"windows": {}, "agree": {}, "series": {}, "stability": {}, "probe": {}, "landscape": {},
           "settings": dict(windows=WINDOWS, stability=STABILITY, n_boot=N_BOOT, tie_share=TIE_SHARE,
                            tied_margin=TIED_MARGIN, single_gate=SINGLE_GATE, pin_share=PIN_SHARE,
                            n_groups_main=ng_main, n_groups_probe=ng_pr, probe_rows=len(probe_rows))}
    tabs = {}
    for (tag, prec), byck in cells.items():
        for w, cks in WINDOWS.items():
            tabs[(tag, prec, "main", w)] = window_table(byck, cks, W_main)
    for tag, byck in tl.items():
        for w, cks in WINDOWS.items():
            # per target, its own rows and groups: one W per target set
            t = {}
            for m in MODS:
                sub = {c: {m: byck[c][m]} for c in cks if c in byck and m in byck[c]}
                if not sub:
                    continue
                Wt = _boot_W(len(next(iter(sub.values()))[m]["N"]))
                t.update(window_table(sub, cks, Wt))
            tabs[(tag, "fp16", "target_last", w)] = t
    for tag, byck in probe.items():
        if tag in RUNS:
            for w, cks in WINDOWS.items():
                tabs[(tag, "fp16", "probe", w)] = window_table(byck, cks, W_pr)

    P("\n== time means, fp16, any-to-any at label_mod positions, 4,416 rows: depth [95%], gate, mix, margin, tie set, state")
    for w in WINDOWS:
        P(f"-- {w} {WINDOWS[w]}")
        for tag in RUNS:
            t = tabs[(tag, "fp16", "main", w)]
            P(f"  {tag:7s} " + " ".join(
                f"{m[:5]}:{t[m]['depth']['mean']:.2f}±{t[m]['depth']['half']:.2f}" for m in MODS if m in t))
            P(f"  {'':7s} gate " + " ".join(f"{m[:5]}:{t[m]['gate']['mean']:.3f}" for m in MODS if m in t))
            P(f"  {'':7s} ties " + " ".join(f"{m[:5]}:{''.join(map(str, tie_set(t[m]['mix']))) or '-'}/{state(t[m])}"
                                         for m in MODS if m in t))
    res["windows"] = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": {m: {kk: vv for kk, vv in v.items() if kk != "boot_depth"}
                                                      for m, v in t.items()}
                      for k, t in tabs.items()}

    # ---- agreement per pair and window
    P("\n== agreement on time means (six image modalities): rho [exact p], mean|d| [perm p], max|d|, criterion 1; "
      "tie sets matched [p]; gate rho [p]; states kappa [p]")
    for frame, prec in (("main", "fp16"), ("main", "fp32"), ("target_last", "fp16"), ("probe", "fp16")):
        P(f"-- frame {frame}, {prec}")
        for a, b in PAIRS:
            for w in WINDOWS:
                ka, kb = (a, prec, frame, w), (b, prec, frame, w)
                if ka not in tabs or kb not in tabs or not all(m in tabs[ka] and m in tabs[kb] for m in IMG):
                    continue
                r = agree(tabs[ka], tabs[kb])
                res["agree"][f"{a}~{b}|{frame}|{prec}|{w}"] = r
                P(f"  {a:6s}~{b:6s} {w:7s} rho {r['rho']:+.2f} [{r['p']:.3f}]  mean|d| {r['meanabs']:.3f} [{r['p_meanabs']:.3f}]"
                  f"  max {r['maxabs']:.2f}  C1 {'pass' if r['crit1'] else 'fail'}  ties {r['ties_match']}/6 [{r['p_ties']:.3f}]"
                  f"  gate {r['gate_rho']:+.2f} [{r['gate_p']:.3f}]  kappa {r['kappa']:+.2f} [{r['p_kappa']:.3f}]")

    # ---- the rules
    P("\n== rules (docs/plans/b0_router_audit.md, Step 4), primary pair A10c ~ A10c-s2, fp16, any-to-any")
    rules = {}
    for w in ("plateau", "late"):
        for a, b in PAIRS:
            r = res["agree"][f"{a}~{b}|main|fp16|{w}"]
            rules[f"crit1|{a}~{b}|{w}"] = r["crit1"]
            P(f"  criterion 1, {w:7s} {a}~{b}: rho {r['rho']:+.2f}, max|d| {r['maxabs']:.2f} -> {'PASS' if r['crit1'] else 'fail'}")
    e = res["agree"]["a10c~a10cs2|main|fp16|early"]
    l_ = res["agree"]["a10c~a10cs2|main|fp16|late"]
    stands = e["p"] < 0.05 and not l_["p"] < 0.05
    rules["agreed_early_then_stopped"] = dict(early_rho=e["rho"], early_p=e["p"], late_rho=l_["rho"], late_p=l_["p"],
                                              stands=bool(stands))
    P(f"  'the seeds agreed early, then stopped': early rho {e['rho']:+.2f} p {e['p']:.3f}; late rho {l_['rho']:+.2f} "
      f"p {l_['p']:.3f} -> {'STANDS' if stands else 'replaced'}")
    pl = res["agree"]["a10c~a10cs2|main|fp16|plateau"]
    rules["shared_tie_sets"] = dict(matched=pl["ties_match"], p=pl["p_ties"], passes=bool(pl["p_ties"] < 0.05))
    P(f"  shared tie sets, plateau: {pl['ties_match']}/6 matched, p {pl['p_ties']:.3f} -> "
      f"{'shared' if pl['p_ties'] < 0.05 else 'not shown'}  (A10c {pl['ties_a']}, A10c-s2 {pl['ties_b']})")

    # ---- per-checkpoint series (description; corrects earlier notes)
    P("\n== per-checkpoint series (description): rho [exact p] mean|d| per checkpoint; one p per series")
    for prec in ("fp16", "fp32"):
        for a, b in PAIRS:
            A = np.array([[summary(cells[(a, prec)][c][m])["depth"] for m in IMG] for c in CKS])
            B = np.array([[summary(cells[(b, prec)][c][m])["depth"] for m in IMG] for c in CKS])
            GA = np.array([[summary(cells[(a, prec)][c][m])["gate"] for m in IMG] for c in CKS])
            GB = np.array([[summary(cells[(b, prec)][c][m])["gate"] for m in IMG] for c in CKS])
            per = [dict(ck=c, rho=S.spearman(A[i], B[i]), p=S.exact_p(A[i], B[i]),
                        meanabs=float(np.abs(A[i] - B[i]).mean()), gate_rho=S.spearman(GA[i], GB[i]),
                        gate_p=S.exact_p(GA[i], GB[i])) for i, c in enumerate(CKS)]
            sr, sp = S.series_p(A, B)
            gr, gp = S.series_p(GA, GB)
            res["series"][f"{a}~{b}|{prec}"] = dict(per=per, depth_series=[sr, sp], gate_series=[gr, gp])
            P(f"  {prec} {a:6s}~{b:6s} depth series rho {sr:+.2f} p {sp:.3f}; gate series {gr:+.2f} p {gp:.3f}")
            P("     " + " ".join(f"{d['ck']}:{d['rho']:+.2f}[{d['p']:.2f}]/{d['meanabs']:.2f}" for d in per))

    # ---- stability (criterion 2's preview)
    P("\n== stability on time means: window 1 - window 2, 95% interval, verdict (band +-0.3)")
    for (tag, prec), byck in cells.items():
        if prec != "fp16":
            continue
        for name, (w1, w2) in STABILITY.items():
            t1, t2 = window_table(byck, w1, W_main), window_table(byck, w2, W_main)
            row = {}
            for m in MODS:
                d = diff_interval(t1[m]["per_ck_depth"], t2[m]["per_ck_depth"], t1[m]["boot_depth"], t2[m]["boot_depth"])
                row[m] = d
            res["stability"][f"{tag}|{prec}|main|{name}"] = row
            P(f"  {tag:7s} {name:26s} " + " ".join(
                f"{m[:5]}:{row[m]['diff']:+.2f}±{row[m]['half']:.2f}{ABBR[row[m]['verdict']]}" for m in MODS))
    # The replay reading on A10c: R3 against R1a and R1b pooled, on the probe.
    r3 = sorted(probe.get("b0r3", {}))
    r1 = [("b0r1a", c) for c in sorted(probe.get("b0r1a", {}))] + [("b0r1b", c) for c in sorted(probe.get("b0r1b", {}))]
    if r3 and r1:
        row = {}
        for m in MODS:
            s3 = [probe["b0r3"][c][m] for c in r3]
            s1 = [probe[t][c][m] for t, c in r1]
            d = diff_interval([summary(r)["depth"] for r in s3], [summary(r)["depth"] for r in s1],
                              boot_timemean(s3, W_pr), boot_timemean(s1, W_pr))
            d.update(mean_r3=float(np.mean([summary(r)["depth"] for r in s3])),
                     mean_r1=float(np.mean([summary(r)["depth"] for r in s1])))
            row[m] = d
        res["stability"]["a10c|fp16|probe|replay_R3_vs_R1ab"] = row
        P(f"  a10c    replay R3 vs R1a+R1b (probe) " + " ".join(
            f"{m[:5]}:{row[m]['diff']:+.2f}±{row[m]['half']:.2f}{ABBR[row[m]['verdict']]}" for m in MODS))
    verdicts = {}
    for tag in RUNS:
        v = res["stability"][f"{tag}|fp16|main|rule_1980-2640_vs_anneal"]
        verdicts[tag] = {m: v[m]["verdict"] for m in IMG}
    if "a10c|fp16|probe|replay_R3_vs_R1ab" in res["stability"]:
        verdicts["a10c_replay"] = {m: res["stability"]["a10c|fp16|probe|replay_R3_vs_R1ab"][m]["verdict"] for m in IMG}
    rules["crit2"] = verdicts
    P("  criterion 2 verdicts (image modalities): " + "; ".join(
        f"{k}: " + ", ".join(f"{m} {ABBR[v]}" for m, v in vv.items()) for k, vv in verdicts.items()))
    P("  (S stable, X unstable, ? undetermined)")

    # ---- probe rows: checkpoint windows against replay windows, and against the main rows
    P("\n== probe rows (256): checkpoint time means against the main rows' and the replay windows'")
    for tag in RUNS:
        for w in ("plateau", "late", "anneal"):
            k = (tag, "fp16", "probe", w)
            if k in tabs and tabs[k]:
                tm, tp = tabs[(tag, "fp16", "main", w)], tabs[k]
                P(f"  {tag:7s} {w:7s} probe " + " ".join(f"{m[:5]}:{tp[m]['depth']['mean']:.2f}" for m in MODS if m in tp)
                  + " | main " + " ".join(f"{tm[m]['depth']['mean']:.2f}" for m in MODS if m in tm))
    rep = {}
    for tag, base in REPLAYS.items():
        if tag not in probe:
            continue
        cks = sorted(probe[tag])
        t = window_table(probe[tag], cks, W_pr)
        rep[tag] = t
        inside = [c for c in CKS if cks[0] <= c <= cks[-1]]
        tb = window_table(probe[base], inside, W_pr) if base in probe else {}
        res["probe"][tag] = {m: dict(depth=t[m]["depth"], gate=t[m]["gate"], mix=t[m]["mix"],
                                     margin=t[m]["margin"], tie_set=list(tie_set(t[m]["mix"])), state=state(t[m]))
                             for m in MODS if m in t}
        P(f"  {tag:6s} ({base}, {cks[0]}-{cks[-1]}, {len(cks)} states) " + " ".join(
            f"{m[:5]}:{t[m]['depth']['mean']:.2f}±{t[m]['depth']['half']:.2f}" for m in MODS if m in t))
        if tb:
            P(f"  {'':6s}  {base}'s checkpoints in it {inside}: " + " ".join(
                f"{m[:5]}:{tb[m]['depth']['mean']:.2f}" for m in MODS if m in tb))
    # Controls for the agreement and stability machinery (CLAUDE.md rule 2). Two replays of one anneal
    # from one checkpoint (R1a, R1b) are the nearest thing to "identical seeds": criterion 1 must pass on
    # their time means and every modality must read stable; with R1b's six labels shifted by one, it
    # must fail.
    if "b0r1a" in rep and "b0r1b" in rep:
        ta_, tb_ = rep["b0r1a"], rep["b0r1b"]
        pos = agree(ta_, tb_)
        shifted = {IMG[i]: tb_[IMG[(i + 1) % 6]] for i in range(6)}
        neg = agree(ta_, shifted)
        stab = {}
        for m in IMG:
            sa = [probe["b0r1a"][c][m] for c in sorted(probe["b0r1a"])]
            sb = [probe["b0r1b"][c][m] for c in sorted(probe["b0r1b"])]
            stab[m] = diff_interval([summary(r)["depth"] for r in sa], [summary(r)["depth"] for r in sb],
                                    boot_timemean(sa, W_pr), boot_timemean(sb, W_pr))["verdict"]
        res["controls"] = dict(r1a_r1b=pos, r1a_r1b_shifted=neg, r1a_r1b_stability=stab,
                               passes=bool(pos["crit1"] and not neg["crit1"] and all(v == "stable" for v in stab.values())))
        P(f"  CONTROL R1a ~ R1b (one anneal, two replays): rho {pos['rho']:+.2f} max|d| {pos['maxabs']:.2f} C1 "
          f"{'pass' if pos['crit1'] else 'FAIL'}; stability {stab}")
        P(f"  CONTROL R1a ~ R1b with labels shifted: rho {neg['rho']:+.2f} max|d| {neg['maxabs']:.2f} C1 "
          f"{'PASS (control broken)' if neg['crit1'] else 'fails, as it must'}")
    if "b0r2" in rep and "b0r3" in rep:
        a = [rep["b0r3"][m]["depth"]["mean"] for m in IMG]
        b = [rep["b0r2"][m]["depth"]["mean"] for m in IMG]
        mad, pm, mx = meanabs(a, b)
        ta_, tb_ = [tie_set(rep["b0r3"][m]["mix"]) for m in IMG], [tie_set(rep["b0r2"][m]["mix"]) for m in IMG]
        nt, pt = tie_agree(ta_, tb_)
        res["probe"]["R3~R2"] = dict(rho=S.spearman(a, b), p=S.exact_p(a, b), meanabs=mad, p_meanabs=pm, maxabs=mx,
                                     ties_match=nt, p_ties=pt)
        P(f"  plateau sub-windows, A10c R3 (2310-2640) ~ A10c-s2 R2 (1650-1980): rho {S.spearman(a, b):+.2f} "
          f"[{S.exact_p(a, b):.3f}] mean|d| {mad:.3f} [{pm:.3f}] max {mx:.2f} ties {nt}/6 [{pt:.3f}]")

    # ---- landscapes against routers
    L = landscapes()
    P("\n== landscapes (A100, target-last, target-only forcing): per target CE(1) CE(2) CE(3), best, router depth")
    for (tag, ck), row in sorted(L.items()):
        P(f"  {tag:7s} {ck:5d} " + " ".join(f"{T[:5]}:g{row[T]['g13']:+.3f}/b{row[T]['best']}/r{row[T]['router_depth']:.2f}"
                                           for T in IMG))
    lres = {}
    for w, cks in WINDOWS.items():
        if not all(("a10c", c) in L and ("a10cs2", c) in L for c in cks):
            continue
        vals = {}
        for tag in ("a10c", "a10cs2"):
            for k in ("g13", "g12", "best", "router_depth"):
                vals[(tag, k)] = [float(np.mean([L[(tag, c)][T][k] for c in cks])) for T in IMG]
        lres[w] = {k: dict(a10c=vals[("a10c", k)], a10cs2=vals[("a10cs2", k)],
                           rho=S.spearman(vals[("a10c", k)], vals[("a10cs2", k)]),
                           p=S.exact_p(vals[("a10c", k)], vals[("a10cs2", k)]))
                   for k in ("g13", "g12", "best", "router_depth")}
        P(f"  {w:7s} cross-seed rho [exact p]: gain d1-d3 {lres[w]['g13']['rho']:+.2f} [{lres[w]['g13']['p']:.3f}]"
          f"  first pass {lres[w]['g12']['rho']:+.2f} [{lres[w]['g12']['p']:.3f}]"
          f"  best depth {lres[w]['best']['rho']:+.2f} [{lres[w]['best']['p']:.3f}]"
          f"  | router depth {lres[w]['router_depth']['rho']:+.2f} [{lres[w]['router_depth']['p']:.3f}]")
    for k in ("g13", "router_depth"):
        A = np.array([[L[("a10c", c)][T][k] for T in IMG] for c in CKS if ("a10c", c) in L and ("a10cs2", c) in L])
        B = np.array([[L[("a10cs2", c)][T][k] for T in IMG] for c in CKS if ("a10c", c) in L and ("a10cs2", c) in L])
        sr, sp = S.series_p(A, B)
        lres[f"series_{k}"] = [sr, sp]
        P(f"  series over {len(A)} checkpoints, {k}: rho {sr:+.2f} p {sp:.3f}")
    bind = {tag: [int(L[(tag, c)][T]["best"] == L[(tag, c)][T]["router_modal"]) for c in CKS if (tag, c) in L for T in IMG]
            for tag in ("a10c", "a10cs2")}
    lres["binding_best_eq_router_modal"] = {t: [int(sum(v)), len(v)] for t, v in bind.items()}
    P("  best depth = the router's modal depth: " + ", ".join(f"{t} {sum(v)}/{len(v)}" for t, v in bind.items()))
    res["landscape"] = lres
    res["rules"] = rules

    # ---- outputs
    (OUT / "agree.json").write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    with open(OUT / "timemeans.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["run", "precision", "frame", "window", "modality", "depth", "depth_half", "depth_h_t", "depth_h_b",
                     "gate", "gate_half", "share1", "share2", "share3", "margin", "tie_set", "state", "n_checkpoints"])
        for (tag, prec, frame, w), t in sorted(tabs.items()):
            for m, v in t.items():
                wr.writerow([tag, prec, frame, w, m, f"{v['depth']['mean']:.4f}", f"{v['depth']['half']:.4f}",
                             f"{v['depth']['h_t']:.4f}", f"{v['depth']['h_b']:.4f}", f"{v['gate']['mean']:.4f}",
                             f"{v['gate']['half']:.4f}"] + [f"{x:.4f}" for x in v["mix"]]
                            + [f"{v['margin']:.5f}", "".join(map(str, tie_set(v["mix"]))), state(v), len(v["cks"])])
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    P(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
