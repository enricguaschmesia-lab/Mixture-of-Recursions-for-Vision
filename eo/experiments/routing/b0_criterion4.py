#!/usr/bin/env python
"""B0 Step 7: criterion 4's router-free measure, its reliability, and the baseline test
(docs/plans/b0_router_audit.md, Step 7). Reads stored arrays only.

The measure: on A10cR (random router), the first extra pass with the context left alone,
g1 = CE(1) - CE(2) per image target, target-last with only the target's depth forced
(`landscape/a10cr_tforce{1,2}`, TITAN V, fp16, gate seed 0), averaged over the last five
checkpoints (1980 ... 3300).

  selftest   known answers: exact permutation p, the transportation solver against random
             feasible plans (Sinkhorn), the compute-only solver against a brute-force grid.
  read       the controls, the reliability rule for the measure and the other candidates, ties
             within noise, the rank test against the learned routers' time means (Step 4,
             agree.json), the companion regret test (the run's own load given to the slots that
             gain most), and description. Writes /data/enric/reports/b0/criterion4/read.{json,txt}.
  extra      description added after the read: the six-label rule without S1GRD, what decides the
             gain-optimal table, the companion test on the all-forced passes and on the router's own
             window, and the all-forced first pass on five slots. Writes extra.{json,txt} beside it.
  power      the proposed test's pass mark: a router at the optimum for each group draw's costs, observed
             with the noise of a time mean over n states, tested against the point estimate; the learned
             runs against each draw; a path from the optimum to the runs and to its relabellings; DEM at 1
             and S1 at 3 with the rest arranged worst, best or at random. Writes power.{json,txt}.
  early      the test at single checkpoints (each run's mix against A10cR at the same checkpoint), and
             R0's last states against A10cR at 330. Writes early.{json,txt}.
  freeze TAG [DIR]
             the freeze rule for a second A10cR seed: its measure from DIR/TAG{1,2,3}/checkpoint-*/
             target_last_rows.npz (DIR defaults to the landscape folder), Spearman against seed 42's
             on six labels and five slots, usable if >= 0.8; the smallest first pass; and, with
             depth 3, the companion test's verdicts on the learned runs under each seed's measure.
             `freeze a10cr_tforce` reads seed 42 against itself, the control that must give 1.

    .venv/bin/python eo/experiments/routing/b0_criterion4.py selftest|read|extra|power|early|freeze TAG [DIR]

Runs in `.venv`, CPU, a few minutes (the vanilla arrays are the slow part).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402

ARM = Path("/data/enric/reports/arm_eval")
B0 = Path("/data/enric/reports/b0")
LAND = B0 / "landscape"
AGREE = B0 / "agree" / "agree.json"
OUT = B0 / "criterion4"
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
IMG = MODS[:6]
SLOTS = ["S2L2A", "S1", "DEM", "NDVI", "LULC"]
SLOT_OF = {"S2L2A": [0], "S1": [1, 2], "DEM": [3], "NDVI": [4], "LULC": [5]}
CKS = [330 * i for i in range(1, 11)]
LAST5 = CKS[5:]
N_BOOT = 10_000
SEED = 20261109
RUNS = ("a10c", "a10cs2", "a10")
WINDOWS = ("plateau", "late", "early", "anneal")
RULE_WINDOWS = ("plateau", "late")


# ------------------------------------------------------------------ helpers

def perrow(ce, lm):
    """(rows, 7): per-row CE sums of each modality's body tokens (label_mod frame)."""
    ce = ce.astype(np.float64)
    return np.stack([np.where(lm == j + 1, ce, 0.0).sum(1) for j in range(7)], 1)


def groups():
    rows = np.load(ARM / "a10cr/checkpoint-3300/eval_rows.npy")
    g = S.eval_groups(rows)
    return rows, g, {int(r): int(x) for r, x in zip(rows, g)}, int(g.max()) + 1


class Est:
    """A point estimate with its bootstrap draws (the same group draws everywhere: paired)."""

    def __init__(self, point, draws):
        self.p, self.d = float(point), np.asarray(draws, dtype=np.float64)

    def __sub__(self, o):
        return Est(self.p - o.p, self.d - o.d)

    def __add__(self, o):
        return Est(self.p + o.p, self.d + o.d)

    def scale(self, a):
        return Est(a * self.p, a * self.d)

    def ci(self):
        lo, hi = np.percentile(self.d, [2.5, 97.5])
        return float(lo), float(hi)

    def fmt(self, k=4):
        lo, hi = self.ci()
        return f"{self.p:+.{k}f} [{lo:+.{k}f}, {hi:+.{k}f}]"


def mean_est(es):
    return Est(np.mean([e.p for e in es]), np.mean([e.d for e in es], 0))


def ratio(num_g, den_g, W):
    return Est(num_g.sum() / den_g.sum(), (W @ num_g) / (W @ den_g))


def pool_s1(v6, w_grd):
    """Six image values (list of Est or floats) -> five slots, S1 pooled by held-out token share."""
    if isinstance(v6[0], Est):
        s1 = v6[1].scale(w_grd) + v6[2].scale(1.0 - w_grd)
    else:
        s1 = w_grd * v6[1] + (1.0 - w_grd) * v6[2]
    return [v6[0], s1, v6[3], v6[4], v6[5]]


def r9(x):
    """Round depths before ranking: an optimum's 1.9999999999999998 and 2.0 are one depth, and
    S.rank would split them."""
    return np.round(np.asarray(x, dtype=np.float64), 9)


def pts(es):
    return [e.p if isinstance(e, Est) else float(e) for e in es]


def adjacent(series, cks):
    """Spearman of each adjacent pair of checkpoints; series: {ck: values}."""
    return [S.spearman(series[a], series[b]) for a, b in zip(cks[:-1], cks[1:])]


# ------------------------------------------------------------------ the transportation problem

EDGES = [(s, d) for s in range(5) for d in range(3)]


def _trees():
    """The 5 x 3 transportation problem's bases: the spanning trees of K(5,3), as edge index tuples."""
    out = []
    for comb in itertools.combinations(range(15), 7):
        parent = list(range(8))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        ok = True
        for e in comb:
            s, d = EDGES[e]
            a, b = find(s), find(5 + d)
            if a == b:
                ok = False
                break
            parent[a] = b
        if ok:
            out.append(comb)
    return out


TREES = np.array(_trees(), dtype=np.int64)                    # (2025, 7)


def _pinv():
    P = []
    for t in TREES:
        A = np.zeros((8, 7))
        for k, e in enumerate(t):
            s, d = EDGES[e]
            A[s, k] = 1.0
            A[5 + d, k] = 1.0
        P.append(np.linalg.pinv(A))
    return np.stack(P)                                           # (2025, 7, 8)


PINV = _pinv()


def feasible_plans(w, L):
    """Every basic feasible plan for supplies w (5) and demands L (3): (n, 15) masses."""
    b = np.concatenate([w, L])
    X = PINV @ b
    ok = (X >= -1e-12).all(1)
    plans = np.zeros((int(ok.sum()), 15))
    rows = np.arange(plans.shape[0])[:, None]
    plans[rows, TREES[ok]] = np.clip(X[ok], 0.0, None)
    return plans


def transport_min(w, L, C, plans=None):
    """min sum x_sd C_sd with row sums w and column sums L; C (5, 3), or (B, 5, 3) for draws.
    Returns (cost, plan (5, 3) as masses) or, for draws, (B,) costs."""
    plans = feasible_plans(w, L) if plans is None else plans
    if C.ndim == 3:
        return (C.reshape(len(C), 15) @ plans.T).min(1)
    cost = plans @ C.ravel()
    i = int(np.argmin(cost))
    return float(cost[i]), plans[i].reshape(5, 3)


def compute_min(w, c, C, depths=(1, 2, 3)):
    """min over mixes with slot shares w at mean depth c: exact, since a vertex has at most one
    slot split between two depths. Returns (cost, mixes (5, 3) as shares within each slot)."""
    best, arg = np.inf, None
    dep = np.asarray(depths, dtype=np.float64)
    for t in itertools.product(range(3), repeat=5):
        t = np.asarray(t)
        ct = float((w * dep[t]).sum())
        Ct = float((w * C[np.arange(5), t]).sum())
        cand = []
        if abs(ct - c) < 1e-12:
            cand.append((Ct, None, 0.0))
        for s in range(5):
            for d2 in range(3):
                if d2 == t[s]:
                    continue
                f = (c - ct) / (w[s] * (dep[d2] - dep[t[s]]))
                if 0.0 <= f <= 1.0:
                    cand.append((Ct + w[s] * f * (C[s, d2] - C[s, t[s]]), (s, d2), f))
        for cost, split, f in cand:
            if cost < best - 1e-15:
                mix = np.zeros((5, 3))
                mix[np.arange(5), t] = 1.0
                if split is not None:
                    s, d2 = split
                    mix[s, t[s]] = 1.0 - f
                    mix[s, d2] = f
                best, arg = cost, mix
    return float(best), arg


def sinkhorn(R, w, L, it=500):
    R = R.copy()
    for _ in range(it):
        R *= (w / R.sum(1))[:, None]
        R *= (L / R.sum(0))[None, :]
    return R


# ------------------------------------------------------------------ selftest

def selftest() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")

    a = [1, 2, 3, 4, 5, 6]
    check("exact p, identical six", abs(S.exact_p(a, a) - 2 / 720) < 1e-12, f"{S.exact_p(a, a):.5f}")
    check("exact p, identical five", abs(S.exact_p(a[:5], a[:5]) - 2 / 120) < 1e-12, f"{S.exact_p(a[:5], a[:5]):.5f}")
    check("bases: 5^2 * 3^4 spanning trees", len(TREES) == 2025, str(len(TREES)))
    rng = np.random.default_rng(0)
    worst_gap, n_plans = np.inf, 0
    for _ in range(200):
        w = rng.dirichlet(np.ones(5))
        L = rng.dirichlet(np.ones(3))
        C = rng.normal(size=(5, 3))
        cstar, plan = transport_min(w, L, C)
        assert np.allclose(plan.sum(1), w, atol=1e-9) and np.allclose(plan.sum(0), L, atol=1e-9)
        for _ in range(50):
            P = sinkhorn(rng.exponential(size=(5, 3)) ** 3, w, L)
            worst_gap = min(worst_gap, float((P * C).sum() - cstar))
            n_plans += 1
    check("transport: no random feasible plan beats the optimum", worst_gap > -1e-9,
          f"(smallest margin {worst_gap:+.2e} over {n_plans} Sinkhorn plans)")
    # compute-only against an independent route: the transport optimum minimised over every load on
    # the line sum L = 1, sum d L_d = c (L3 = L1 + c - 2), on a fine grid of L1
    below, above = np.inf, 0.0
    for _ in range(100):
        w = rng.dirichlet(np.ones(5))
        C = rng.normal(size=(5, 3))
        c = float(rng.uniform(1.05, 2.95))
        cstar, mix = compute_min(w, c, C)
        assert abs((w[:, None] * mix * np.arange(1, 4)).sum() - c) < 1e-9 and np.allclose(mix.sum(1), 1.0)
        lo, hi = max(0.0, 2.0 - c), (3.0 - c) / 2.0
        grid = [transport_min(w, np.array([a, 1.0 - 2.0 * a - (c - 2.0), a + c - 2.0]), C)[0]
                for a in np.linspace(lo, hi, 401)]
        below = min(below, min(grid) - cstar)
        above = max(above, min(grid) - cstar)
    check("compute-only: no load on the compute line beats it, and the grid comes within 1e-3",
          below > -1e-9 and above < 1e-3, f"(grid minus optimum: smallest {below:+.2e}, largest {above:+.2e})")
    # consistency: the compute-only optimum is feasible at its own load, so transport there equals it
    bad = 0
    for _ in range(100):
        w = rng.dirichlet(np.ones(5))
        C = rng.normal(size=(5, 3))
        c = float(rng.uniform(1.05, 2.95))
        cstar, mix = compute_min(w, c, C)
        L = (w[:, None] * mix).sum(0)
        t, _ = transport_min(w, L, C)
        bad += abs(t - cstar) > 1e-9
    check("transport at the compute optimum's own load equals the compute optimum", bad == 0, f"({bad} of 100 differ)")
    # convex slots with every second pass below every first: the optimum at mean 2 is uniform 2
    C = np.array([[3.0, 2.0, 2.1], [5.0, 4.5, 4.6], [1.0, 0.9, 0.95], [2.0, 1.8, 1.9], [4.0, 3.0, 3.2]])
    w = np.full(5, 0.2)
    cstar, mix = compute_min(w, 2.0, C)
    check("compute-only at 2 with every second pass costing: uniform 2", np.allclose(mix[:, 1], 1.0), str(mix[:, 1]))
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------ data

def load_tl(tag, ck, d=LAND):
    f = d / tag / f"checkpoint-{ck}" / "target_last_rows.npz"
    return np.load(f) if f.exists() else None


def tl_groups(z, gmap, G, rows_ref=None):
    """{T: (ce_g, n_g)} group sums of one target-last file; asserts the rows."""
    out = {}
    for T in MODS:
        r = z[f"{T}|rows"]
        if rows_ref is not None:
            assert np.array_equal(r, rows_ref[T]), T
        g = np.array([gmap[int(x)] for x in r])
        out[T] = (np.bincount(g, weights=z[f"{T}|ce_sum"], minlength=G),
                  np.bincount(g, weights=z[f"{T}|n"].astype(np.float64), minlength=G))
    return out


def target_only(prefix, gmap, G, W, d=LAND, rows_ref=None):
    """{(depth, ck, T): Est of CE} for prefix{1,2,3} where present."""
    out, rows0 = {}, rows_ref
    for dep in (1, 2, 3):
        for ck in CKS:
            z = load_tl(f"{prefix}{dep}", ck, d)
            if z is None:
                continue
            if rows0 is None:
                rows0 = {T: z[f"{T}|rows"] for T in MODS}
            for T, (c, n) in tl_groups(z, gmap, G, rows0).items():
                out[(dep, ck, T)] = (c, n)
    return out, rows0


def pass_est(TO, a, b, ck, T, W):
    """CE(a) - CE(b) as an Est, paired over groups."""
    ca, n = TO[(a, ck, T)]
    cb, n2 = TO[(b, ck, T)]
    assert np.array_equal(n, n2)
    return ratio(ca - cb, n, W)


def any_to_any(A, key, j, g_main, G, W):
    x = np.bincount(g_main, weights=A[key][:, j], minlength=G)
    n = np.bincount(g_main, weights=A["cnt"][:, j], minlength=G)
    return x, n


# ------------------------------------------------------------------ read

def read() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines, res = [], {}

    def P(*a):
        t = " ".join(str(x) for x in a)
        lines.append(t)
        print(t, flush=True)

    rows_main, g_main, gmap, G = groups()
    W = S.boot_weights(G, N_BOOT, SEED)
    A = dict(np.load(LAND / "perrow.npz"))
    assert np.array_equal(A["rows"], rows_main)
    cnt = A["cnt"]
    n_tok = cnt.sum(0)
    w_grd = float(n_tok[1] / (n_tok[1] + n_tok[2]))
    w_slot = np.array([n_tok[SLOT_OF[s]].sum() for s in SLOTS])
    w_slot = w_slot / w_slot.sum()
    lm = np.load(ARM / "a10cr/checkpoint-3300/tokens_router.npz")["label_mod"]
    TO, rows_tl = target_only("a10cr_tforce", gmap, G, W)
    agree = json.loads(AGREE.read_text())["windows"]
    res["settings"] = dict(n_boot=N_BOOT, seed=SEED, groups=G, w_s1grd=w_grd, slot_weights=w_slot.tolist(),
                           last5=LAST5, rows_per_target={T: int(len(rows_tl[T])) for T in MODS})
    P(f"settings: {G} groups, {N_BOOT} draws (seed {SEED}); S1GRD's share of S1 tokens {w_grd:.4f}; "
      f"slot weights {np.round(w_slot, 4).tolist()}; rows per target {res['settings']['rows_per_target']}")

    # ---- the measure and the other passes, per checkpoint (target only)
    g1 = {ck: [pass_est(TO, 1, 2, ck, T, W) for T in MODS] for ck in CKS}
    g2 = {ck: [pass_est(TO, 2, 3, ck, T, W) for T in MODS] for ck in CKS}        # CE(2) - CE(3)
    g13 = {ck: [pass_est(TO, 1, 3, ck, T, W) for T in MODS] for ck in CKS}
    ce = {(d, ck): [ratio(*TO[(d, ck, T)], W) for T in MODS] for d in (1, 2, 3) for ck in CKS}
    m1 = [mean_est([g1[ck][j] for ck in LAST5]) for j in range(7)]
    m2 = [mean_est([g2[ck][j] for ck in LAST5]) for j in range(7)]
    m13 = [mean_est([g13[ck][j] for ck in LAST5]) for j in range(7)]

    # ---- controls
    P("\n== Controls (if one fails, nothing is read)")
    ctrl = {}
    land = json.loads((LAND / "read.json").read_text())["bowl_paired"]
    dev = max(abs(g1[ck][j].p - land[f"{ck}|{T}"]["target"]["d1"][0]) for ck in (1980, 3300) for j, T in enumerate(MODS))
    ctrl["1_step6_values"] = dev < 1e-9
    P(f"  1. g1 at 1980 and 3300 equals Step 6's read.json: max |diff| {dev:.2e} -> {'PASS' if ctrl['1_step6_values'] else 'FAIL'}")

    def af(model, a, b, ck):
        return [(A[f"{model}|f{a}|{ck}"][:, j].sum() - A[f"{model}|f{b}|{ck}"][:, j].sum()) / n_tok[j] for j in range(6)]

    prov = np.mean([af("a10cr", 1, 2, ck) for ck in LAST5], 0)
    want = dict(S2L2A=0.0074, S1GRD=0.0030, S1RTC=0.0057, DEM=0.0045, NDVI=0.0074, LULC=0.0128)
    dev2 = max(abs(round(float(prov[j]), 4) - want[T]) for j, T in enumerate(IMG))
    # the candidates' column, over all ten checkpoints (nine adjacent pairs), as planned
    van = {}
    for ck in CKS:
        z11 = np.load(ARM / "b10c11" / f"checkpoint-{ck}" / "tokens_router.npz")
        z29 = np.load(ARM / "b10c" / f"checkpoint-{ck}" / "tokens_router.npz")
        assert np.array_equal(z11["label_mod"], lm) and np.array_equal(z29["label_mod"], lm)
        van[ck] = (perrow(z11["ce"], lm), perrow(z29["ce"], lm))
    tlaf = {ck: [json.loads((ARM / f"a10cr_force{d}" / f"checkpoint-{ck}" / "metrics.json").read_text())["target_last"]
                 for d in (1, 2, 3)] for ck in CKS}
    cand_series = {
        "A10cR d1-d2 all forced": {ck: af("a10cr", 1, 2, ck) for ck in CKS},
        "A10cR d1-d3 all forced": {ck: af("a10cr", 1, 3, ck) for ck in CKS},
        "A10cR d1-d3 target-last all forced": {ck: [tlaf[ck][0][T]["ce"] - tlaf[ck][2][T]["ce"] for T in IMG] for ck in CKS},
        "A10cR5 d1-d3 all forced": {ck: af("a10cr5", 1, 3, ck) for ck in CKS},
        "vanilla 11L-29L": {ck: [(van[ck][0][:, j].sum() - van[ck][1][:, j].sum()) / n_tok[j] for j in range(6)] for ck in CKS},
        "A10cR d1-d2 target only (the measure)": {ck: pts(g1[ck][:6]) for ck in CKS},
    }
    col = {k: float(np.mean(adjacent(v, CKS))) for k, v in cand_series.items()}
    want_col = {"A10cR d1-d2 all forced": 0.66, "A10cR d1-d3 all forced": 0.45,
                "A10cR d1-d3 target-last all forced": 0.53, "A10cR5 d1-d3 all forced": 0.89, "vanilla 11L-29L": 0.84}
    dev3 = max(abs(round(col[k], 2) - v) for k, v in want_col.items())
    mean5 = {k: np.mean([v[ck] for ck in LAST5], 0) for k, v in cand_series.items()}
    f14 = S.spearman(mean5["A10cR d1-d2 all forced"], mean5["A10cR5 d1-d3 all forced"])
    ctrl["2_planning_numbers"] = dev2 < 1e-9 and dev3 < 1e-9 and abs(round(f14, 2) - 0.94) < 1e-9
    P(f"  2. provisional order (all forced, d1-d2, last five): "
      + " ".join(f"{T} {prov[j]:.4f}" for j, T in enumerate(IMG)) + f" (max |diff| to the plan {dev2:.4f});")
    P("     the candidates' column, nine adjacent pairs: "
      + "; ".join(f"{k} {col[k]:.2f} (plan {v})" for k, v in want_col.items()) + f"; finding 14's rho {f14:+.2f} (plan +0.94)"
      + f" -> {'PASS' if ctrl['2_planning_numbers'] else 'FAIL'}")
    mono = 1.0 + 2.0 * (S.rank(pts(m1[:6])) - 1) / 5
    rev = 4.0 - mono
    ctrl["3_test_can_pass_and_fail"] = (S.spearman(pts(m1[:6]), mono) >= 0.9) and (S.spearman(pts(m1[:6]), rev) < 0.9)
    P(f"  3. the rank test on a depth monotone in the measure: rho {S.spearman(pts(m1[:6]), mono):+.2f}; on its reverse "
      f"{S.spearman(pts(m1[:6]), rev):+.2f} -> {'PASS' if ctrl['3_test_can_pass_and_fail'] else 'FAIL'}")
    for run in RUNS:
        for win in WINDOWS:
            t = agree[f"{run}|fp16|main|{win}"]
            for T in IMG:
                mix = np.asarray(t[T]["mix"], dtype=np.float64)
                assert abs((mix / mix.sum() * np.arange(1, 4)).sum() - t[T]["depth"]["mean"]) < 1e-6, (run, win, T)
    P("     (check: every time-mean mix in agree.json gives its time-mean depth to 1e-6)")
    res["controls"] = ctrl
    if not all(ctrl.values()):
        P("A CONTROL FAILED: nothing below is read.")
        (OUT / "read.txt").write_text("\n".join(lines) + "\n")
        (OUT / "read.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        return 1

    # ---- 1. the measure
    P("\n== 1. The measure: A10cR, target only, g1 = CE(1) - CE(2), nats [95% group interval]")
    for ck in CKS:
        P(f"  {ck:5d} " + "  ".join(f"{T[:5]} {g1[ck][j].p:+.4f}" for j, T in enumerate(MODS)))
    P("  mean of the last five checkpoints (1980-3300):")
    order = sorted(range(6), key=lambda j: -m1[j].p)
    for j in order:
        P(f"    {IMG[j]:6s} {m1[j].fmt()}   second pass CE(2)-CE(3) {m2[j].fmt()}   d1-d3 {m13[j].fmt()}")
    P(f"    {'Coords':6s} {m1[6].fmt()}   (reported apart)")
    res["measure"] = {"per_ck": {str(ck): {T: g1[ck][j].p for j, T in enumerate(MODS)} for ck in CKS},
                      "second_per_ck": {str(ck): {T: g2[ck][j].p for j, T in enumerate(MODS)} for ck in CKS},
                      "mean5": {T: [m1[j].p, *m1[j].ci()] for j, T in enumerate(MODS)},
                      "second_mean5": {T: [m2[j].p, *m2[j].ci()] for j, T in enumerate(MODS)},
                      "d13_mean5": {T: [m13[j].p, *m13[j].ci()] for j, T in enumerate(MODS)},
                      "order": [IMG[j] for j in order]}

    # ---- 2. the reliability rule
    P("\n== 2. The reliability rule (a: adjacent rho >= 0.8 from 1980 on, mean of four pairs, or the five-checkpoint"
      " mean against A10cR5's d1-d3 >= 0.8; b: the group interval of top - bottom excludes zero)")
    ser = {ck: pts(g1[ck][:6]) for ck in CKS}
    adj = adjacent(ser, LAST5)
    r5 = mean5["A10cR5 d1-d3 all forced"]
    route2 = S.spearman(pts(m1[:6]), r5)
    top, bot = order[0], order[-1]
    tb = m1[top] - m1[bot]
    a_ok = bool((np.mean(adj) >= 0.8) or (route2 >= 0.8))
    b_ok = bool(tb.ci()[0] > 0)
    P(f"  route 1: adjacent pairs 1980-2310, 2310-2640, 2640-2970, 2970-3300: "
      + ", ".join(f"{x:+.2f}" for x in adj) + f"; mean {np.mean(adj):+.2f}, smallest {min(adj):+.2f}"
      + f" -> {'meets' if np.mean(adj) >= 0.8 else 'misses'} 0.8")
    P(f"  route 2: rho against A10cR5's d1-d3 (all forced, mean of the same five) {route2:+.2f} "
      f"[exact p {S.exact_p(pts(m1[:6]), r5):.3f}] -> {'meets' if route2 >= 0.8 else 'misses'} 0.8")
    P(f"  (b) top {IMG[top]} - bottom {IMG[bot]}: {tb.fmt()} -> {'separated' if b_ok else 'NOT separated'}")
    verdict = "usable" if (a_ok and b_ok) else "not usable"
    P(f"  VERDICT: the measure is {verdict.upper()} (a by route {'1' if np.mean(adj) >= 0.8 else '2' if route2 >= 0.8 else '-'}, "
      f"b {'yes' if b_ok else 'no'})")
    res["rule"] = dict(adjacent=adj, adjacent_mean=float(np.mean(adj)), route2=route2,
                       route2_p=S.exact_p(pts(m1[:6]), r5), top=IMG[top], bottom=IMG[bot],
                       top_minus_bottom=[tb.p, *tb.ci()], a=a_ok, b=b_ok, verdict=verdict)
    # five slots
    m1s = pool_s1(m1[:6], w_grd)
    m2s = pool_s1(m2[:6], w_grd)
    ser5 = {ck: pts(pool_s1(g1[ck][:6], w_grd)) for ck in CKS}
    adj5 = adjacent(ser5, LAST5)
    r5s = pool_s1(list(r5), w_grd)
    route2s = S.spearman(pts(m1s), r5s)
    order5 = sorted(range(5), key=lambda k: -m1s[k].p)
    tb5 = m1s[order5[0]] - m1s[order5[-1]]
    P(f"  five slots (S1 pooled by tokens): adjacent " + ", ".join(f"{x:+.2f}" for x in adj5)
      + f", mean {np.mean(adj5):+.2f}; against A10cR5 {route2s:+.2f} [exact p {S.exact_p(pts(m1s), r5s):.3f}];"
      f" top {SLOTS[order5[0]]} - bottom {SLOTS[order5[-1]]} {tb5.fmt()}")
    P("    slots: " + "  ".join(f"{SLOTS[k]} {m1s[k].fmt()}" for k in order5))
    res["rule_five_slots"] = dict(adjacent=adj5, adjacent_mean=float(np.mean(adj5)), route2=route2s,
                                  route2_p=S.exact_p(pts(m1s), r5s), measure={SLOTS[k]: [m1s[k].p, *m1s[k].ci()] for k in range(5)},
                                  second={SLOTS[k]: [m2s[k].p, *m2s[k].ci()] for k in range(5)},
                                  top_minus_bottom=[tb5.p, *tb5.ci()], order=[SLOTS[k] for k in order5])

    # ---- 3. ties
    P("\n== 3. Ties within noise: pairwise differences of the five-checkpoint mean, 95% group interval")
    ties = {}
    for i, j in itertools.combinations(order, 2):
        dlt = m1[i] - m1[j]
        lo, hi = dlt.ci()
        ties[f"{IMG[i]}-{IMG[j]}"] = [dlt.p, lo, hi, bool(lo <= 0 <= hi)]
    chain = []
    for i, j in zip(order[:-1], order[1:]):
        chain.append(f"{IMG[i]} {'=' if ties[f'{IMG[i]}-{IMG[j]}'][3] else '>'}")
    P("  order: " + " ".join(chain) + f" {IMG[order[-1]]}   ('=' tied, '>' separated, adjacent pairs)")
    for k, v in ties.items():
        P(f"    {k:13s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]{'  tied' if v[3] else ''}")
    res["ties"] = ties

    # ---- 4. the other candidates under the same rule
    P("\n== 4. The rule applied to the table's other candidates (description; decision 8 picks the measure)")
    res["candidates"] = {}
    perrow_keys = {"A10cR d1-d2 all forced": ("a10cr", 1, 2), "A10cR d1-d3 all forced": ("a10cr", 1, 3),
                   "A10cR5 d1-d3 all forced": ("a10cr5", 1, 3)}
    for k, v in cand_series.items():
        adj_k = adjacent(v, LAST5)
        r2 = S.spearman(mean5[k], r5) if not k.startswith("A10cR5") else float("nan")
        b_txt, b_val = "n/a (aggregates only)", None
        if k in perrow_keys or k.startswith("vanilla") or k.endswith("(the measure)"):
            if k.endswith("(the measure)"):
                ests = m1[:6]
            elif k.startswith("vanilla"):
                ests = []
                for j in range(6):
                    xs = [np.bincount(g_main, weights=van[ck][0][:, j] - van[ck][1][:, j], minlength=G) for ck in LAST5]
                    n = np.bincount(g_main, weights=cnt[:, j], minlength=G)
                    ests.append(mean_est([ratio(x, n, W) for x in xs]))
            else:
                model, a, b = perrow_keys[k]
                ests = []
                for j in range(6):
                    xs = [np.bincount(g_main, weights=A[f"{model}|f{a}|{ck}"][:, j] - A[f"{model}|f{b}|{ck}"][:, j], minlength=G)
                          for ck in LAST5]
                    n = np.bincount(g_main, weights=cnt[:, j], minlength=G)
                    ests.append(mean_est([ratio(x, n, W) for x in xs]))
            o = sorted(range(6), key=lambda j: -ests[j].p)
            dd = ests[o[0]] - ests[o[-1]]
            b_val = dd.ci()[0] > 0
            b_txt = f"{IMG[o[0]]} - {IMG[o[-1]]} {dd.fmt()}"
        usable = ((np.mean(adj_k) >= 0.8) or (r2 >= 0.8)) and bool(b_val)
        res["candidates"][k] = dict(adjacent_from_1980=adj_k, adjacent_mean_from_1980=float(np.mean(adj_k)),
                                    adjacent_mean_all=col[k], route2=r2, b=b_val, usable=usable,
                                    mean5=dict(zip(IMG, map(float, mean5[k]))))
        P(f"  {k:38s} adjacent from 1980 {np.mean(adj_k):+.2f} (pairs " + ", ".join(f"{x:+.2f}" for x in adj_k)
          + f"; all ten {col[k]:+.2f}); vs A10cR5 {r2:+.2f}; b: {b_txt} -> {'usable' if usable else 'not usable'}")
    P("  rank correlations between the candidates' five-checkpoint means:")
    names = list(cand_series)
    for i, j in itertools.combinations(range(len(names)), 2):
        P(f"    {names[i]:38s} ~ {names[j]:38s} {S.spearman(mean5[names[i]], mean5[names[j]]):+.2f}")

    # ---- 5. description: split halves, gate redraw
    P("\n== 5. Description: where the measure's instability comes from")
    rng = np.random.default_rng(SEED)
    sh5, sh1980, sh3300 = [], [], []
    num = {(ck, j): TO[(1, ck, MODS[j])][0] - TO[(2, ck, MODS[j])][0] for ck in CKS for j in range(6)}
    den = {(ck, j): TO[(1, ck, MODS[j])][1] for ck in CKS for j in range(6)}
    for _ in range(1000):
        perm = rng.permutation(G)
        halves = (perm[: G // 2], perm[G // 2:])
        if any(den[(ck, j)][h].sum() == 0 for h in halves for ck in LAST5 for j in range(6)):
            continue
        v = [[np.mean([num[(ck, j)][h].sum() / den[(ck, j)][h].sum() for ck in LAST5]) for j in range(6)] for h in halves]
        sh5.append(S.spearman(*v))
        for ck, acc in ((1980, sh1980), (3300, sh3300)):
            u = [[num[(ck, j)][h].sum() / den[(ck, j)][h].sum() for j in range(6)] for h in halves]
            acc.append(S.spearman(*u))
    P(f"  split halves of the groups (1,000 splits), Spearman between halves: five-checkpoint mean {np.mean(sh5):+.2f} "
      f"(5th percentile {np.percentile(sh5, 5):+.2f}); 1980 alone {np.mean(sh1980):+.2f} ({np.percentile(sh1980, 5):+.2f}); "
      f"3300 alone {np.mean(sh3300):+.2f} ({np.percentile(sh3300, 5):+.2f})")
    P(f"  same rows, next checkpoint (route 1's pairs): mean {np.mean(adj):+.2f}. So the order moves "
      f"{'more over training than over the data' if np.mean(adj) < np.mean(sh5) - 0.1 else 'about as much over training as over the data'}")
    res["split_half"] = dict(mean5=[float(np.mean(sh5)), float(np.percentile(sh5, 5))],
                             ck1980=[float(np.mean(sh1980)), float(np.percentile(sh1980, 5))],
                             ck3300=[float(np.mean(sh3300)), float(np.percentile(sh3300, 5))])
    gs = {}
    for dep in (1, 3):
        z = load_tl(f"a10cr_tforce{dep}_gs1", 3300)
        for T, (c, n) in tl_groups(z, gmap, G, rows_tl).items():
            gs[(dep, T)] = (c, n)
    d13_s1 = [ratio(gs[(1, T)][0] - gs[(3, T)][0], gs[(1, T)][1], W) for T in MODS]
    d13_s0 = g13[3300]
    P("  gate redraw at 3300, d1-d3 (seed 1 minus seed 0): " + "  ".join(
        f"{T[:5]} {d13_s1[j].p - d13_s0[j].p:+.5f}" for j, T in enumerate(MODS))
      + f"; Spearman between seeds {S.spearman(pts(d13_s0[:6]), pts(d13_s1[:6])):+.2f}")
    res["gate_redraw_3300_d13"] = {T: [d13_s0[j].p, d13_s1[j].p] for j, T in enumerate(MODS)}
    P("  the first and second passes by checkpoint, five slots (target only; S1 by tokens):")
    for ck in CKS:
        a1 = pts(pool_s1(g1[ck][:6], w_grd))
        a2 = pts(pool_s1(g2[ck][:6], w_grd))
        P(f"    {ck:5d} first " + " ".join(f"{SLOTS[k][:5]} {a1[k]:+.4f}" for k in range(5))
          + "   second " + " ".join(f"{SLOTS[k][:5]} {a2[k]:+.4f}" for k in range(5)))

    # ---- 6. the rank test against the learned routers' time means (the baseline)
    P("\n== 6. The test as written: Spearman >= 0.9 between the measure and a run's time-mean depth (six image "
      "modalities, exact p; five slots beside), plateau and late windows decide")
    res["test"] = {}
    meas6 = pts(m1[:6])
    meas5 = pts(m1s)
    for run in RUNS:
        for win in WINDOWS:
            t = agree[f"{run}|fp16|main|{win}"]
            dep6 = [t[T]["depth"]["mean"] for T in IMG]
            dep5 = pool_s1(dep6, w_grd)
            r6, p6 = S.spearman(meas6, dep6), S.exact_p(meas6, dep6)
            r5_, p5 = S.spearman(meas5, dep5), S.exact_p(meas5, dep5)
            res["test"][f"{run}|{win}"] = dict(rho6=r6, p6=p6, rho5=r5_, p5=p5, depth6=dep6, depth5=dep5,
                                                pass6=bool(r6 >= 0.9), pass5=bool(r5_ >= 0.9))
            P(f"  {run:6s} {win:7s} rho {r6:+.2f} [p {p6:.3f}]  five slots {r5_:+.2f} [p {p5:.3f}]   depth "
              + " ".join(f"{T[:5]} {d:.2f}" for T, d in zip(IMG, dep6)))
    for run in RUNS:
        ok6 = all(res["test"][f"{run}|{w}"]["pass6"] for w in RULE_WINDOWS)
        res["test"][f"{run}|verdict"] = "passes" if ok6 else "fails"
        P(f"  {run}: the test {'PASSES' if ok6 else 'FAILS'} (plateau and late){'  [reported, not counted]' if run == 'a10' else ''}")

    # ---- 7. the companion test: the run's own load, given to the slots that gain most
    P("\n== 7. Companion test (description, proposed): regret of each run's assignment at its own load, against the "
      "gain-optimal allocation from the measure's CE(d); null = the 120 relabellings of its slots")
    Cs = np.array([pts(pool_s1([ce[(d, ck)][j] for j in range(6)], w_grd)) for d in (1, 2, 3) for ck in LAST5])
    Cs = Cs.reshape(3, len(LAST5), 5).mean(1).T                                    # (5 slots, 3 depths)
    Cb = np.stack([np.stack([np.mean([pool_s1([ce[(d, ck)][j] for j in range(6)], w_grd)[k].d for ck in LAST5], 0)
                             for d in (1, 2, 3)], -1) for k in range(5)], 1)       # (B, 5, 3)
    P("  per-slot costs, five-checkpoint mean, CE(d) - CE(2): " + "  ".join(
        f"{SLOTS[k]} ({Cs[k, 0] - Cs[k, 1]:+.4f}, 0, {Cs[k, 2] - Cs[k, 1]:+.4f})" for k in range(5)))
    perms = list(itertools.permutations(range(5)))

    def mixes(run, win):
        t = agree[f"{run}|fp16|main|{win}"]
        m6 = [np.asarray(t[T]["mix"], dtype=np.float64) / np.sum(t[T]["mix"]) for T in IMG]
        return np.stack(pool_s1(m6, w_grd))                                        # (5, 3) shares

    def regret(mix, C=Cs):
        L = (w_slot[:, None] * mix).sum(0)
        cstar, plan = transport_min(w_slot, L, C)
        return float((w_slot[:, None] * mix * C).sum() - cstar), plan, L

    def report(label, mix):
        R, plan, L = regret(mix)
        null = np.array([regret(mix[list(pi)])[0] for pi in perms])
        p = float(np.mean(null <= R + 1e-12))
        star = plan / w_slot[:, None]
        dstar = (star * np.arange(1, 4)).sum(1)
        drun = (mix * np.arange(1, 4)).sum(1)
        plans = feasible_plans(w_slot, L)
        Rb = (w_slot[None, :, None] * mix[None] * Cb).sum((1, 2)) - transport_min(w_slot, L, Cb, plans)
        lo, hi = np.percentile(Rb, [2.5, 97.5])
        rho_star5 = S.spearman(meas5, r9(dstar))
        rho_star6 = S.spearman(meas6, r9([dstar[0], dstar[1], dstar[1], dstar[2], dstar[3], dstar[4]]))
        P(f"  {label:16s} load {np.round(L, 3).tolist()}  regret {R:.4f} [{lo:.4f}, {hi:.4f}]  null "
          f"min {null.min():.4f} median {np.median(null):.4f} max {null.max():.4f}  p {p:.3f}")
        P(f"  {'':16s} run's depth   " + " ".join(f"{SLOTS[k][:5]} {drun[k]:.2f}" for k in range(5)))
        P(f"  {'':16s} gain-optimal  " + " ".join(f"{SLOTS[k][:5]} {dstar[k]:.2f}" for k in range(5))
          + f"   <- the rank test on it: rho {rho_star6:+.2f} (six labels), {rho_star5:+.2f} (five slots)")
        return dict(load=L.tolist(), regret=R, regret_ci=[float(lo), float(hi)], null=[float(null.min()), float(np.median(null)),
                    float(null.max())], p=p, depth_run=drun.tolist(), depth_opt=dstar.tolist(),
                    rank_test_on_optimum=[rho_star6, rho_star5])

    res["companion"] = {}
    for run in RUNS:
        for win in WINDOWS:
            res["companion"][f"{run}|{win}"] = report(f"{run} {win}", mixes(run, win))
    for run in RUNS:
        ok = all(res["companion"][f"{run}|{w}"]["p"] <= 0.05 for w in RULE_WINDOWS)
        res["companion"][f"{run}|verdict"] = "follows the gain" if ok else "does not follow the gain"
        P(f"  {run}: {'FOLLOWS' if ok else 'does NOT follow'} the gain at its own load (p <= 0.05 in plateau and late)")
    # controls of the companion test
    mix_c = mixes("a10c", "plateau")
    _, plan_c, L_c = regret(mix_c)
    opt_mix = plan_c / w_slot[:, None]
    nullo = np.array([regret(opt_mix[list(pi)])[0] for pi in perms])
    worst_pi = perms[int(np.argmax(nullo))]
    R_rev = regret(opt_mix[list(worst_pi)])[0]
    p_opt = float(np.mean(nullo <= regret(opt_mix)[0] + 1e-12))
    p_rev = float(np.mean(nullo <= R_rev + 1e-12))
    rng = np.random.default_rng(1)
    margin = min(float((sinkhorn(rng.exponential(size=(5, 3)) ** 3, w_slot, L_c) * Cs).sum()
                       - transport_min(w_slot, L_c, Cs)[0]) for _ in range(2000))
    P(f"  controls: the optimum as a run reads regret {regret(opt_mix)[0]:.1e} at p {p_opt:.3f}; its worst relabelling "
      f"p {p_rev:.3f}; 2,000 random feasible plans at A10c's plateau load, smallest margin over the optimum {margin:+.2e}")
    res["companion_controls"] = dict(p_optimum=p_opt, p_worst=p_rev, sinkhorn_margin=margin)
    # references: equal load, and compute alone
    P("  references, five slots (what a run that follows the gain would do):")
    refs = {}
    for label, L in (("equal load", np.full(3, 1 / 3)), ("load (0.4, 0.4, 0.2)", np.array([0.4, 0.4, 0.2]))):
        c, plan = transport_min(w_slot, L, Cs)
        dstar = (plan / w_slot[:, None] * np.arange(1, 4)).sum(1)
        refs[label] = dstar.tolist()
        P(f"    at {label:22s} " + " ".join(f"{SLOTS[k][:5]} {dstar[k]:.2f}" for k in range(5))
          + f"   rank test on it: rho {S.spearman(meas5, r9(dstar)):+.2f} (five slots)")
    for cm in (1.8, 2.0, 2.2):
        c, mix = compute_min(w_slot, cm, Cs)
        dstar = (mix * np.arange(1, 4)).sum(1)
        uni = float((w_slot * Cs[:, 1]).sum())
        refs[f"compute {cm}"] = dstar.tolist()
        P(f"    at mean depth {cm} alone   " + " ".join(f"{SLOTS[k][:5]} {dstar[k]:.2f}" for k in range(5))
          + f"   CE minus uniform 2: {c - uni:+.5f}")
    # stability of the gain-optimal table at equal load and at 1.8, over checkpoints and draws
    P("  the gain-optimal table by checkpoint (target only, five slots), at equal load and at mean depth 1.8:")
    for ck in LAST5:
        Ck = np.array([pts(pool_s1([ce[(d, ck)][j] for j in range(6)], w_grd)) for d in (1, 2, 3)]).T
        _, pe = transport_min(w_slot, np.full(3, 1 / 3), Ck)
        _, pc = compute_min(w_slot, 1.8, Ck)
        P(f"    {ck:5d} equal load " + " ".join(f"{SLOTS[k][:5]} {(pe[k] / w_slot[k] * np.arange(1, 4)).sum():.2f}" for k in range(5))
          + "   1.8 " + " ".join(f"{SLOTS[k][:5]} {(pc[k] * np.arange(1, 4)).sum():.2f}" for k in range(5)))
    plans_eq = feasible_plans(w_slot, np.full(3, 1 / 3))
    costs = Cb.reshape(N_BOOT, 15) @ plans_eq.T
    arg = costs.argmin(1)
    vals, cnts = np.unique(arg, return_counts=True)
    top_plan = plans_eq[vals[np.argmax(cnts)]].reshape(5, 3)
    P(f"  equal load, five-checkpoint costs: the modal gain-optimal plan in {cnts.max() / N_BOOT:.1%} of group draws: "
      + " ".join(f"{SLOTS[k][:5]} {(top_plan[k] / w_slot[k] * np.arange(1, 4)).sum():.2f}" for k in range(5)))
    res["references"] = refs

    (OUT / "read.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "read.txt").write_text("\n".join(lines) + "\n")
    P(f"\n-> {OUT / 'read.txt'}, {OUT / 'read.json'}")
    return 0


# ------------------------------------------------------------------ extra (description, after the read)

WIN_CKS = {"early": [330, 660, 990], "plateau": [990, 1320, 1650, 1980, 2310, 2640, 2970],
           "late": LAST5, "anneal": [2970, 3300]}


def run_mix(agree, run, win, w_grd):
    t = agree[f"{run}|fp16|main|{win}"]
    m6 = [np.asarray(t[T]["mix"], dtype=np.float64) / np.sum(t[T]["mix"]) for T in IMG]
    return np.stack(pool_s1(m6, w_grd))


def regret_test(mix, w_slot, C):
    """Regret of a (5, 3) mix at its own load against the transport optimum, and its 120-relabelling null."""
    def reg(m):
        L = (w_slot[:, None] * m).sum(0)
        c, plan = transport_min(w_slot, L, C)
        return float((w_slot[:, None] * m * C).sum() - c), plan
    R, plan = reg(mix)
    null = np.array([reg(mix[list(pi)])[0] for pi in itertools.permutations(range(5))])
    return dict(R=R, p=float(np.mean(null <= R + 1e-12)), null=null,
                dstar=(plan / w_slot[:, None] * np.arange(1, 4)).sum(1))


def extra() -> int:
    lines, res = [], {}

    def P(*a):
        t = " ".join(str(x) for x in a)
        lines.append(t)
        print(t, flush=True)

    rows_main, g_main, gmap, G = groups()
    W = S.boot_weights(G, N_BOOT, SEED)
    A = dict(np.load(LAND / "perrow.npz"))
    cnt = A["cnt"]
    n_tok = cnt.sum(0)
    w_grd = float(n_tok[1] / (n_tok[1] + n_tok[2]))
    w_slot = np.array([n_tok[SLOT_OF[s]].sum() for s in SLOTS])
    w_slot = w_slot / w_slot.sum()
    TO, rows_tl = target_only("a10cr_tforce", gmap, G, W)
    agree = json.loads(AGREE.read_text())["windows"]
    ce = {(d, ck): [ratio(*TO[(d, ck, T)], W) for T in MODS] for d in (1, 2, 3) for ck in CKS}

    def af_slot(model, d, ck):
        """Five-slot CE (Est) at forced depth d, any-to-any; S1 pooled by summing its columns."""
        out = []
        for s in SLOTS:
            idx = SLOT_OF[s]
            x = np.bincount(g_main, weights=A[f"{model}|f{d}|{ck}"][:, idx].sum(1), minlength=G)
            n = np.bincount(g_main, weights=cnt[:, idx].sum(1), minlength=G)
            out.append(ratio(x, n, W))
        return out

    def af6(model, a, b, ck):
        return [(A[f"{model}|f{a}|{ck}"][:, j].sum() - A[f"{model}|f{b}|{ck}"][:, j].sum()) / n_tok[j] for j in range(6)]

    # 1. attribution: S1GRD left out
    P("== 1. Attribution: the reliability rule with S1GRD left out (five labels; S1RTC stands for S1)")
    keep = [0, 2, 3, 4, 5]
    g1 = {ck: [ce[(1, ck)][j].p - ce[(2, ck)][j].p for j in range(6)] for ck in CKS}
    ser = {ck: [g1[ck][j] for j in keep] for ck in CKS}
    m1 = np.mean([ser[ck] for ck in LAST5], 0)
    r5 = np.mean([af6("a10cr5", 1, 3, ck) for ck in LAST5], 0)[keep]
    adj = adjacent(ser, LAST5)
    P(f"  route 1: " + ", ".join(f"{x:+.2f}" for x in adj) + f"; mean {np.mean(adj):+.2f}.  route 2: "
      f"{S.spearman(m1, r5):+.2f} [exact p {S.exact_p(m1, r5):.3f}]")
    P("  S1GRD's first pass at 1980 ... 3300: " + ", ".join(f"{g1[ck][1]:+.4f}" for ck in LAST5)
      + f"; its rank among six: " + ", ".join(str(int(7 - S.rank(g1[ck][:6])[1])) for ck in LAST5))
    res["without_s1grd"] = dict(adjacent=adj, route2=S.spearman(m1, r5))

    # 2. the robust features of the gain-optimal table
    P("\n== 2. What decides the gain-optimal table at equal load, five slots, five-checkpoint mean, 95% group interval")
    f5 = [mean_est([pool_s1([ce[(1, ck)][j] - ce[(2, ck)][j] for j in range(6)], w_grd)[k] for ck in LAST5]) for k in range(5)]
    c5 = [mean_est([pool_s1([ce[(3, ck)][j] - ce[(2, ck)][j] for j in range(6)], w_grd)[k] for ck in LAST5]) for k in range(5)]
    P("  first pass, CE(1) - CE(2): " + "  ".join(f"{SLOTS[k]} {f5[k].fmt()}" for k in range(5)))
    P("  second-pass cost, CE(3) - CE(2): " + "  ".join(f"{SLOTS[k]} {c5[k].fmt()}" for k in range(5)))
    for k in (1, 3):
        P(f"  DEM's first pass minus {SLOTS[k]}'s: {(f5[2] - f5[k]).fmt()}")
    for k in (2, 3, 0):
        P(f"  S1's second-pass cost minus {SLOTS[k]}'s: {(c5[1] - c5[k]).fmt()}")
    P("  per checkpoint: smallest first pass, cheapest second pass (target only, five slots): " + "; ".join(
        f"{ck} {SLOTS[int(np.argmin([pool_s1([ce[(1, ck)][j].p - ce[(2, ck)][j].p for j in range(6)], w_grd)[k] for k in range(5)]))]}"
        f"/{SLOTS[int(np.argmin([pool_s1([ce[(3, ck)][j].p - ce[(2, ck)][j].p for j in range(6)], w_grd)[k] for k in range(5)]))]}"
        for ck in CKS))
    res["robust"] = dict(first={SLOTS[k]: [f5[k].p, *f5[k].ci()] for k in range(5)},
                         second_cost={SLOTS[k]: [c5[k].p, *c5[k].ci()] for k in range(5)})

    # 3. the companion test with the all-forced any-to-any measure
    P("\n== 3. The companion test with the all-forced any-to-any passes as the measure (A10cR, Titan X, five slots)")
    Caf = np.mean([np.array([[x.p for x in af_slot("a10cr", d, ck)] for d in (1, 2, 3)]).T for ck in LAST5], 0)
    Cto = np.mean([np.array([pts(pool_s1(ce[(d, ck)][:6], w_grd)) for d in (1, 2, 3)]).T for ck in LAST5], 0)
    P("  CE(d) - CE(2), all forced: " + "  ".join(f"{SLOTS[k]} ({Caf[k, 0] - Caf[k, 1]:+.4f}, {Caf[k, 2] - Caf[k, 1]:+.4f})" for k in range(5)))
    P("  CE(d) - CE(2), target only: " + "  ".join(f"{SLOTS[k]} ({Cto[k, 0] - Cto[k, 1]:+.4f}, {Cto[k, 2] - Cto[k, 1]:+.4f})" for k in range(5)))
    res["frame"] = {}
    for run in RUNS:
        for win in RULE_WINDOWS:
            a, b = regret_test(run_mix(agree, run, win, w_grd), w_slot, Cto), regret_test(run_mix(agree, run, win, w_grd), w_slot, Caf)
            res["frame"][f"{run}|{win}"] = dict(target_only=[a["R"], a["p"]], all_forced=[b["R"], b["p"]],
                                               opt_all_forced=b["dstar"].tolist())
            P(f"  {run:6s} {win:7s} target only: regret {a['R']:.4f} p {a['p']:.3f}   all forced: regret {b['R']:.4f} "
              f"p {b['p']:.3f}; its optimum " + " ".join(f"{SLOTS[k][:5]} {b['dstar'][k]:.2f}" for k in range(5)))
    _, pl = transport_min(w_slot, np.full(3, 1 / 3), Caf)
    P("  all forced, equal load, gain-optimal: " + " ".join(f"{SLOTS[k][:5]} {(pl[k] / w_slot[k] * np.arange(1, 4)).sum():.2f}" for k in range(5)))

    # 4. window-matched
    P("\n== 4. Window-matched: the measure (target only) averaged over the router's own window; did the routers follow "
      "A10cR's gains early?")
    res["window_matched"] = {}
    for win, cks in WIN_CKS.items():
        Cw = np.mean([np.array([pts(pool_s1(ce[(d, ck)][:6], w_grd)) for d in (1, 2, 3)]).T for ck in cks], 0)
        P(f"  {win:7s} measure: first " + " ".join(f"{SLOTS[k][:5]} {Cw[k, 0] - Cw[k, 1]:+.4f}" for k in range(5))
          + "   second cost " + " ".join(f"{SLOTS[k][:5]} {Cw[k, 2] - Cw[k, 1]:+.4f}" for k in range(5)))
        for run in RUNS:
            r = regret_test(run_mix(agree, run, win, w_grd), w_slot, Cw)
            d_run = (run_mix(agree, run, win, w_grd) * np.arange(1, 4)).sum(1)
            res["window_matched"][f"{run}|{win}"] = dict(R=r["R"], p=r["p"], dstar=r["dstar"].tolist(), drun=d_run.tolist())
            P(f"    {run:6s} regret {r['R']:.4f} (null {r['null'].min():.4f}-{r['null'].max():.4f}) p {r['p']:.3f}   run "
              + " ".join(f"{SLOTS[k][:5]} {d_run[k]:.2f}" for k in range(5)) + "   optimum "
              + " ".join(f"{SLOTS[k][:5]} {r['dstar'][k]:.2f}" for k in range(5)))

    # 5. the all-forced any-to-any first pass on five slots, under the rule
    P("\n== 5. The all-forced any-to-any first pass on five slots, under the reliability rule")
    s5 = {ck: [x.p for x in np.subtract(af_slot("a10cr", 1, ck), af_slot("a10cr", 2, ck))] for ck in CKS}
    r5s = {ck: [x.p for x in np.subtract(af_slot("a10cr5", 1, ck), af_slot("a10cr5", 3, ck))] for ck in CKS}
    m5 = np.mean([s5[ck] for ck in LAST5], 0)
    q5 = np.mean([r5s[ck] for ck in LAST5], 0)
    adj5 = adjacent(s5, LAST5)
    diff = [mean_est([(af_slot("a10cr", 1, ck)[k] - af_slot("a10cr", 2, ck)[k]) for ck in LAST5]) for k in range(5)]
    o = sorted(range(5), key=lambda k: -diff[k].p)
    tb = diff[o[0]] - diff[o[-1]]
    P(f"  route 1: " + ", ".join(f"{x:+.2f}" for x in adj5) + f"; mean {np.mean(adj5):+.2f}.  route 2 (A10cR5, five slots): "
      f"{S.spearman(m5, q5):+.2f} [p {S.exact_p(m5, q5):.3f}].  (b) {SLOTS[o[0]]} - {SLOTS[o[-1]]} {tb.fmt()}")
    P("  values: " + "  ".join(f"{SLOTS[k]} {diff[k].fmt()}" for k in o)
      + f";  rho with the target-only five-slot measure {S.spearman(m5, pts(pool_s1([mean_est([ce[(1, ck)][j] - ce[(2, ck)][j] for ck in LAST5]) for j in range(6)], w_grd))):+.2f}")
    res["all_forced_five_slots"] = dict(adjacent=adj5, route2=S.spearman(m5, q5), b=[tb.p, *tb.ci()],
                                        values={SLOTS[k]: [diff[k].p, *diff[k].ci()] for k in range(5)})

    # 6. five-slot ties, and the gain-optimal tables off depth 2 in both frames
    P("\n== 6. Five slots, target only: pairwise differences of the first pass, and the optimum off depth 2")
    o5 = sorted(range(5), key=lambda k: -f5[k].p)
    res["ties5"] = {}
    for i, j in itertools.combinations(o5, 2):
        dlt = f5[i] - f5[j]
        lo, hi = dlt.ci()
        res["ties5"][f"{SLOTS[i]}-{SLOTS[j]}"] = [dlt.p, lo, hi, bool(lo <= 0 <= hi)]
        P(f"  {SLOTS[i]:5s} - {SLOTS[j]:5s} {dlt.fmt()}{'  tied' if lo <= 0 <= hi else ''}")
    meas5 = [f.p for f in f5]
    res["off_two"] = {}
    for label, C in (("target only", Cto), ("all forced", Caf)):
        for cm in (1.6, 1.8, 2.2, 2.4):
            c, mix = compute_min(w_slot, cm, C)
            dstar = (mix * np.arange(1, 4)).sum(1)
            rho = S.spearman(meas5, r9(dstar))
            res["off_two"][f"{label}|{cm}"] = dict(depth=dstar.tolist(), rho=rho, minus_uniform=c - float((w_slot * C[:, 1]).sum()))
            P(f"  {label:11s} mean depth {cm}: " + " ".join(f"{SLOTS[k][:5]} {dstar[k]:.2f}" for k in range(5))
              + f"   CE minus uniform 2 {c - float((w_slot * C[:, 1]).sum()):+.5f}; the rank test on it: rho {rho:+.2f}")

    (OUT / "extra.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "extra.txt").write_text("\n".join(lines) + "\n")
    P(f"\n-> {OUT / 'extra.txt'}, {OUT / 'extra.json'}")
    return 0


# ------------------------------------------------------------------ power and early (before B0 Step 8)

PERMS = np.array(list(itertools.permutations(range(5))), dtype=np.int64)   # (120, 5); row 0 is the identity
N_STATES = {"plateau": (198, 7), "late": (132, 5)}                           # (probe callback, checkpoints only)


def _setup(n_boot):
    rows_main, g_main, gmap, G = groups()
    W = S.boot_weights(G, n_boot, SEED)
    A = dict(np.load(LAND / "perrow.npz"))
    n_tok = A["cnt"].sum(0)
    w_grd = float(n_tok[1] / (n_tok[1] + n_tok[2]))
    w_slot = np.array([n_tok[SLOT_OF[s]].sum() for s in SLOTS])
    w_slot = w_slot / w_slot.sum()
    assert np.allclose(w_slot, 0.2), "relabellings keep the load only for equal slot weights"
    TO, rows_tl = target_only("a10cr_tforce", gmap, G, W)
    agree = json.loads(AGREE.read_text())["windows"]
    return dict(G=G, W=W, w_grd=w_grd, w=w_slot, TO=TO, agree=agree)


def slot_costs(TO, W, cks, w_grd, draws=False):
    """Five-slot CE (5, 3) at depths 1-3, averaged over cks; with draws, also (B, 5, 3)."""
    est = [[mean_est([pool_s1([ratio(*TO[(d, ck, T)], W) for T in IMG], w_grd)[k] for ck in cks])
            for d in (1, 2, 3)] for k in range(5)]
    C = np.array([[e.p for e in row] for row in est])
    if not draws:
        return C
    return C, np.stack([np.stack([e.d for e in row], -1) for row in est], 1)


def ptest(X, C, w):
    """The proposed test for one allocation X (5, 3 shares) against costs C (5, 3): (p, regret)."""
    L = w @ X
    plans = feasible_plans(w, L)
    if not len(plans):
        raise RuntimeError(f"no feasible plan at load {L}")
    cstar = float((plans @ C.ravel()).min())
    reg = (w[None, :, None] * X[PERMS] * C[None]).sum((1, 2)) - cstar
    return float(np.mean(reg <= reg[0] + 1e-12)), float(reg[0])


def optimum(w, L, C):
    _, plan = transport_min(w, L, C)
    return np.clip(plan / w[:, None], 0.0, None)


def noisy(X, n, rng):
    """Each slot's time mean as the share of n independent states at each depth."""
    if n is None:
        return X
    return np.stack([rng.multinomial(n, X[s] / X[s].sum()) / n for s in range(5)])


def depths(X):
    return (X * np.arange(1, 4)).sum(1)


def power() -> int:
    lines, res = [], {}

    def P(*a):
        t = " ".join(str(x) for x in a)
        lines.append(t)
        print(t, flush=True)

    B = 1000
    ctx = _setup(B)
    w, w_grd, agree = ctx["w"], ctx["w_grd"], ctx["agree"]
    Chat, Cb = slot_costs(ctx["TO"], ctx["W"], LAST5, w_grd, draws=True)
    rng = np.random.default_rng(SEED)
    loads = {f"{run}|{win}": (run_mix(agree, run, win, w_grd), N_STATES[win])
             for run in ("a10c", "a10cs2") for win in RULE_WINDOWS}
    eq = np.full((5, 3), 1 / 3)
    loads["equal load"] = (eq, N_STATES["late"])

    # controls
    P("== Controls")
    want = {"a10c|plateau": 0.767, "a10c|late": 0.725, "a10cs2|plateau": 0.933, "a10cs2|late": 0.950}
    got = {k: ptest(loads[k][0], Chat, w)[0] for k in want}
    ok1 = all(abs(round(got[k], 3) - v) < 1e-9 for k, v in want.items())
    P("  1. the learned runs without noise against the point estimate: "
      + ", ".join(f"{k} p {got[k]:.3f} (Step 7 {want[k]})" for k in want) + f" -> {'PASS' if ok1 else 'FAIL'}")
    ok2 = True
    for k, (X, _) in loads.items():
        O = optimum(w, w @ X, Chat)
        p0, r0 = ptest(O, Chat, w)
        pbig = np.mean([ptest(noisy(O, 10 ** 7, rng), Chat, w)[0] for _ in range(20)])
        ok2 &= (p0 <= 0.05 + 1e-12) and abs(r0) < 1e-12 and abs(pbig - p0) < 1e-9
        P(f"  2. {k:15s} the optimum without noise reads p {p0:.3f}, regret {r0:.1e}; with n = 10^7, mean p {pbig:.3f}")
    P(f"     -> {'PASS' if ok2 else 'FAIL'}")
    res["controls"] = dict(runs_without_noise=got, ok=bool(ok1 and ok2))
    if not (ok1 and ok2):
        P("A CONTROL FAILED: nothing below is read.")
        (OUT / "power.txt").write_text("\n".join(lines) + "\n")
        return 1

    # the rule's numbers
    P(f"\n== 1. A router at the optimum for each draw's costs (the true gain, {B} group draws), observed with noise, "
      "tested against the point estimate: share passing p <= 0.05 (and p <= 0.10, 0.20)")
    res["rule"] = {}
    for k, (X, (n_cb, n_ck)) in loads.items():
        L = w @ X
        row = {}
        for label, n in (("no noise", None), ("probe callback", n_cb), ("checkpoints only", n_ck)):
            ps, ps_size = [], []
            for b in range(B):
                O = optimum(w, L, Cb[b])
                ps.append(ptest(noisy(O, n, rng), Chat, w)[0])
                perm = PERMS[rng.integers(1, 120)]
                ps_size.append(ptest(noisy(O[perm], n, rng), Chat, w)[0])
            ps, ps_size = np.array(ps), np.array(ps_size)
            row[label] = {f"pass_{t}": float(np.mean(ps <= t + 1e-12)) for t in (0.05, 0.10, 0.20)}
            row[label].update({f"size_{t}": float(np.mean(ps_size <= t + 1e-12)) for t in (0.05, 0.10, 0.20)})
            P(f"  {k:15s} {label:16s} n={str(n):4s} passes " + ", ".join(
                f"{row[label][f'pass_{t}']:.0%} at {t}" for t in (0.05, 0.10, 0.20))
              + "   size " + ", ".join(f"{row[label][f'size_{t}']:.1%}" for t in (0.05, 0.10, 0.20)))
        if k != "equal load":
            pr = np.array([ptest(X, Cb[b], w)[0] for b in range(B)])
            row["run_fails"] = {f"{t}": float(np.mean(pr > t + 1e-12)) for t in (0.05, 0.10, 0.20)}
            P(f"  {k:15s} the run itself against each draw's costs: fails " + ", ".join(
                f"{row['run_fails'][f'{t}']:.0%} at {t}" for t in (0.05, 0.10, 0.20))
              + f" (its p: median {np.median(pr):.3f}, 5th percentile {np.percentile(pr, 5):.3f})")
        res["rule"][k] = row
    four = [k for k in loads if k != "equal load"]
    verdict = None
    for t in (0.05, 0.10, 0.20):
        opt = min(res["rule"][k]["probe callback"][f"pass_{t}"] for k in four)
        fails = min(res["rule"][k]["run_fails"][f"{t}"] for k in four)
        size = max(res["rule"][k]["probe callback"][f"size_{t}"] for k in four)
        ok = opt >= 0.90 and fails >= 0.90 and size <= t + 0.01
        P(f"  pass mark p <= {t}: optimum passes at least {opt:.0%} (callback noise), runs fail at least {fails:.0%}, "
          f"size at most {size:.1%} -> {'meets' if ok else 'misses'} the rule")
        if ok and verdict is None:
            verdict = t
    res["verdict_pass_mark"] = verdict
    P(f"  VERDICT: {'the pass mark p <= 0.05 stands' if verdict == 0.05 else f'too strict; the proposal moves to p <= {verdict}' if verdict else 'no pass mark up to 0.20 meets the rule: the test cannot certify gain-following at this precision'}")

    # description: the path
    P("\n== 2. Description: the path from the optimum (point estimate, at the run's load) towards the run's own "
      "allocation and towards the optimum's 119 relabellings; callback noise; share passing p <= 0.05")
    res["path"] = {}
    alphas = np.round(np.arange(0, 1.01, 0.1), 2)
    for k in four:
        X, (n_cb, _) = loads[k]
        L = w @ X
        O = optimum(w, L, Chat)
        out = {}
        for label, refs, reps in (("to the run", [X], 200), ("to relabellings", [O[p] for p in PERMS[1:]], 10)):
            rates, moves, regs = [], [], []
            for a in alphas:
                hits, tot = 0, 0
                for R_ in refs:
                    Xa = (1 - a) * O + a * R_
                    for _ in range(reps):
                        hits += ptest(noisy(Xa, n_cb, rng), Chat, w)[0] <= 0.05 + 1e-12
                        tot += 1
                rates.append(hits / tot)
                moves.append(float(np.mean([(w * np.abs(depths((1 - a) * O + a * R_) - depths(O))).sum() / w.sum() for R_ in refs])))
                regs.append(float(np.mean([ptest((1 - a) * O + a * R_, Chat, w)[1] for R_ in refs])))
            cross = next((i for i, r in enumerate(rates) if r < 0.5), None)
            out[label] = dict(alpha=alphas.tolist(), rate=rates, move=moves, regret=regs, cross=None if cross is None else float(alphas[cross]))
            P(f"  {k:15s} {label:16s} " + " ".join(f"{a:.1f}:{r:.0%}" for a, r in zip(alphas, rates)))
            if cross is not None:
                P(f"  {'':15s} {'':16s} below 50% from alpha {alphas[cross]:.1f}: a mean move of {moves[cross]:.2f} passes per slot, "
                  f"regret {regs[cross]:.4f} nats (null median at the run {np.median([ptest(X[p], Chat, w)[1] for p in PERMS]):.4f})")
        res["path"][k] = out

    # description: only the robust part right
    P("\n== 3. Description: DEM at depth 1 and S1 at depth 3, the other three slots arranged worst, best, or at random; "
      "callback noise; share passing p <= 0.05")
    res["robust"] = {}
    BIG = 1e3
    for k in four + ["equal load"]:
        X, (n_cb, _) = loads[k]
        L = w @ X
        Lr = L - w[2] * np.array([1.0, 0, 0]) - w[1] * np.array([0, 0, 1.0])
        if (Lr < -1e-12).any():
            P(f"  {k:15s} DEM at 1 and S1 at 3 do not fit this load")
            continue
        out = {}
        for label, sign in (("worst", -1.0), ("best", 1.0)):
            Cf = sign * Chat.copy()
            Cf[2] = [-BIG, BIG, BIG]
            Cf[1] = [BIG, BIG, -BIG]
            Xf = optimum(w, L, Cf)
            assert Xf[2, 0] > 1 - 1e-9 and Xf[1, 2] > 1 - 1e-9
            out[label] = float(np.mean([ptest(noisy(Xf, n_cb, rng), Chat, w)[0] <= 0.05 + 1e-12 for _ in range(200)]))
            out[label + "_depths"] = depths(Xf).tolist()
        hits = []
        others = [0, 3, 4]
        for _ in range(200):
            Pm = sinkhorn(rng.exponential(size=(3, 3)) ** 3, w[others], Lr)
            Xr = np.zeros((5, 3))
            Xr[2, 0] = 1.0
            Xr[1, 2] = 1.0
            Xr[others] = Pm / w[others][:, None]
            hits.append(ptest(noisy(Xr, n_cb, rng), Chat, w)[0] <= 0.05 + 1e-12)
        out["random"] = float(np.mean(hits))
        res["robust"][k] = out
        P(f"  {k:15s} worst {out['worst']:.0%} (" + " ".join(f"{SLOTS[i][:5]} {d:.2f}" for i, d in enumerate(out['worst_depths']))
          + f"), best {out['best']:.0%}, random {out['random']:.0%}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "power.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "power.txt").write_text("\n".join(lines) + "\n")
    P(f"\n-> {OUT / 'power.txt'}, {OUT / 'power.json'}")
    return 0


def early() -> int:
    lines, res = [], {}

    def P(*a):
        t = " ".join(str(x) for x in a)
        lines.append(t)
        print(t, flush=True)

    ctx = _setup(2000)
    w, w_grd, agree, TO, W = ctx["w"], ctx["w_grd"], ctx["agree"], ctx["TO"], ctx["W"]
    C = {ck: slot_costs(TO, W, [ck], w_grd) for ck in CKS}

    # the run's mix at each checkpoint, from Step 4's windows
    def ck_mix(run, ck):
        found = []
        for win in WINDOWS:
            t = agree[f"{run}|fp16|main|{win}"]
            cks = t["S2L2A"]["cks"]
            if ck in cks:
                i = cks.index(ck)
                m6, d6 = [], []
                for T in IMG:
                    v = np.asarray(t[T]["per_ck_mix"][i], dtype=np.float64)
                    m6.append(v / v.sum())
                    d6.append(t[T]["per_ck_depth"][i])
                found.append((np.stack(pool_s1(m6, w_grd)), m6, d6))
        return found

    P("== Controls")
    ok = True
    for run in RUNS:
        for ck in CKS:
            f = ck_mix(run, ck)
            assert f, (run, ck)
            for X, m6, d6 in f:
                ok &= all(abs(depths(np.array([m]))[0] - d) < 1e-6 for m, d in zip(m6, d6))
            ok &= all(np.allclose(f[0][0], g[0]) for g in f[1:])
    j = json.loads((B0 / "replay" / "b0r0.json").read_text())
    last = j["rows"][-1]
    ok &= all(abs(depths(np.array([np.asarray(last[T]["mix"]) / np.sum(last[T]["mix"])]))[0] - last[T]["depth"]) < 1e-6 for T in IMG)
    meas = json.loads((OUT / "read.json").read_text())["measure"]["per_ck"]
    for ck in (1980, 3300):
        g1 = pool_s1([meas[str(ck)][T] for T in IMG], w_grd)
        ok &= np.allclose(C[ck][:, 0] - C[ck][:, 1], g1, atol=1e-12)
    P(f"  per-checkpoint mixes give their stored depths and agree across windows; R0's last state gives its depth; "
      f"the measure at 1980 and 3300 equals read's -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        (OUT / "early.txt").write_text("\n".join(lines) + "\n")
        return 1

    P("\n== 1. Per checkpoint (one draw each): each run's mix against A10cR's target-only CE at the same checkpoint")
    res["per_ck"] = {}
    for ck in CKS:
        g1 = C[ck][:, 0] - C[ck][:, 1]
        c2 = C[ck][:, 2] - C[ck][:, 1]
        rank = [SLOTS[i] for i in np.argsort(-g1)]
        P(f"  {ck:5d} A10cR first pass " + " ".join(f"{SLOTS[i][:5]} {g1[i]:+.4f}" for i in range(5))
          + "   second-pass cost " + " ".join(f"{SLOTS[i][:5]} {c2[i]:+.4f}" for i in range(5)))
        for run in RUNS:
            X = ck_mix(run, ck)[0][0]
            p, R = ptest(X, C[ck], w)
            O = optimum(w, w @ X, C[ck])
            res["per_ck"][f"{run}|{ck}"] = dict(p=p, regret=R, depth_run=depths(X).tolist(), depth_opt=depths(O).tolist(),
                                                first_rank=rank)
            P(f"        {run:6s} p {p:.3f} regret {R:.4f}   run " + " ".join(f"{SLOTS[i][:5]} {d:.2f}" for i, d in enumerate(depths(X)))
              + "   optimum " + " ".join(f"{SLOTS[i][:5]} {d:.2f}" for i, d in enumerate(depths(O))))

    P("\n== 2. R0 (A10c's first 330 steps replayed, probe rows): time mean of its last states against A10cR at 330")
    res["r0"] = {}
    steps = j["steps"]
    for lo in (230, 280):
        idx = [i for i, s in enumerate(steps) if s >= lo]
        m6 = [np.mean([np.asarray(j["rows"][i][T]["mix"]) / np.sum(j["rows"][i][T]["mix"]) for i in idx], 0) for T in IMG]
        X = np.stack(pool_s1(m6, w_grd))
        p, R = ptest(X, C[330], w)
        O = optimum(w, w @ X, C[330])
        res["r0"][str(lo)] = dict(p=p, regret=R, states=len(idx), depth_run=depths(X).tolist(), depth_opt=depths(O).tolist())
        P(f"  steps {lo}-330 ({len(idx)} states): p {p:.3f} regret {R:.4f}   R0 " + " ".join(
            f"{SLOTS[i][:5]} {d:.2f}" for i, d in enumerate(depths(X))) + "   optimum " + " ".join(
            f"{SLOTS[i][:5]} {d:.2f}" for i, d in enumerate(depths(O))))

    P("\n== 3. The reading at 330 (rule fixed 23:30)")
    p_r0 = res["r0"]["230"]["p"]
    p_runs = [res["per_ck"][f"{run}|330"]["p"] for run in RUNS]
    if p_r0 <= 0.05 and sum(p <= 0.05 for p in p_runs) >= 2:
        reading = "followed the gain at 330"
    elif p_r0 < 0.5 and sum(p < 0.5 for p in p_runs) >= 2:
        reading = "partly"
    else:
        reading = "no sign"
    res["reading"] = reading
    P(f"  R0 p {p_r0:.3f}; stored runs at 330 p " + ", ".join(f"{r} {p:.3f}" for r, p in zip(RUNS, p_runs))
      + f" -> {reading.upper()}")
    P("  which slots sit where the optimum puts them at 330 (|depth - optimum| <= 0.3):")
    for run in RUNS:
        d = res["per_ck"][f"{run}|330"]
        P(f"    {run:6s} " + " ".join(f"{SLOTS[i][:5]} {'yes' if abs(a - b) <= 0.3 else 'no'}"
                                     for i, (a, b) in enumerate(zip(d['depth_run'], d['depth_opt']))))
    d = res["r0"]["230"]
    P(f"    {'R0':6s} " + " ".join(f"{SLOTS[i][:5]} {'yes' if abs(a - b) <= 0.3 else 'no'}"
                                  for i, (a, b) in enumerate(zip(d['depth_run'], d['depth_opt']))))
    P("  S2L2A from 330 to 990: its depth per run, and its first-pass rank on A10cR (1 = largest of five):")
    for ck in (330, 660, 990):
        g1 = C[ck][:, 0] - C[ck][:, 1]
        r = int(np.where(np.argsort(-g1) == 0)[0][0]) + 1
        P(f"    {ck:4d} rank {r} ({g1[0]:+.4f})   " + "  ".join(f"{run} {res['per_ck'][f'{run}|{ck}']['depth_run'][0]:.2f}" for run in RUNS))

    (OUT / "early.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "early.txt").write_text("\n".join(lines) + "\n")
    P(f"\n-> {OUT / 'early.txt'}, {OUT / 'early.json'}")
    return 0


# ------------------------------------------------------------------ freeze (after A10cR seed 43)

def freeze(tag: str, d: Path) -> int:
    rows_main, g_main, gmap, G = groups()
    W = S.boot_weights(G, 2000, SEED)
    A = dict(np.load(LAND / "perrow.npz"))
    n_tok = A["cnt"].sum(0)
    w_grd = float(n_tok[1] / (n_tok[1] + n_tok[2]))
    ref, rows_tl = target_only("a10cr_tforce", gmap, G, W)
    new, _ = target_only(tag, gmap, G, W, d, rows_tl)
    have = [ck for ck in LAST5 if all((dd, ck, T) in new for dd in (1, 2) for T in IMG)]
    if len(have) < len(LAST5):
        print(f"{tag}: the measure needs depths 1 and 2 at {LAST5}; found {have}")
        return 1
    a = [np.mean([pass_est(ref, 1, 2, ck, T, W).p for ck in LAST5]) for T in IMG]
    b = [np.mean([pass_est(new, 1, 2, ck, T, W).p for ck in LAST5]) for T in IMG]
    r6, p6 = S.spearman(a, b), S.exact_p(a, b)
    a5, b5 = pool_s1(a, w_grd), pool_s1(b, w_grd)
    r5, p5 = S.spearman(a5, b5), S.exact_p(a5, b5)
    print("seed 42 " + " ".join(f"{T} {v:+.4f}" for T, v in zip(IMG, a)))
    print(f"{tag:7s} " + " ".join(f"{T} {v:+.4f}" for T, v in zip(IMG, b)))
    print(f"between seeds: rho {r6:+.2f} [exact p {p6:.3f}] (the rule as written: "
          f"{'frozen' if r6 >= 0.8 else 'not reliable'}), five slots {r5:+.2f} [p {p5:.3f}] "
          f"({'frozen' if r5 >= 0.8 else 'not reliable'})")
    shifted = b[1:] + b[:1]
    print(f"control, the second seed's labels shifted by one: rho {S.spearman(a, shifted):+.2f} "
          f"({'fails, as it must' if S.spearman(a, shifted) < 0.8 else 'PASSES: the control did not fire'})")
    print(f"smallest first pass, five slots: seed 42 {SLOTS[int(np.argmin(a5))]}, {tag} {SLOTS[int(np.argmin(b5))]}")
    if not all((3, ck, T) in new for ck in LAST5 for T in IMG):
        print("depth 3 missing: the companion test needs CE(3), so it is not recomputed")
        return 0
    # the companion test's baseline verdicts, with each seed's measure
    w_slot = np.array([n_tok[SLOT_OF[s]].sum() for s in SLOTS])
    w_slot = w_slot / w_slot.sum()
    agree = json.loads(AGREE.read_text())["windows"]

    def costs(TO):
        return np.mean([np.array([pool_s1([ratio(*TO[(dd, ck, T)], W).p for T in IMG], w_grd) for dd in (1, 2, 3)]).T
                        for ck in LAST5], 0)

    Ca, Cb = costs(ref), costs(new)
    for name, C in (("seed 42", Ca), (tag, Cb)):
        _, plan = transport_min(w_slot, np.full(3, 1 / 3), C)
        print(f"{name:12s} gain-optimal at equal load: "
              + " ".join(f"{SLOTS[k][:5]} {(plan[k] / w_slot[k] * np.arange(1, 4)).sum():.2f}" for k in range(5)))
        for run in RUNS:
            ps = [regret_test(run_mix(agree, run, win, w_grd), w_slot, C)["p"] for win in RULE_WINDOWS]
            print(f"{'':12s} {run:6s} p (plateau, late) {ps[0]:.3f}, {ps[1]:.3f} -> "
                  f"{'follows the gain' if max(ps) <= 0.05 else 'does not follow the gain'}")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a == ["selftest"]:
        return selftest()
    if a == ["read"]:
        return read()
    if a == ["extra"]:
        return extra()
    if a == ["power"]:
        return power()
    if a == ["early"]:
        return early()
    if len(a) in (2, 3) and a[0] == "freeze":
        return freeze(a[1], Path(a[2]) if len(a) == 3 else LAND)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
