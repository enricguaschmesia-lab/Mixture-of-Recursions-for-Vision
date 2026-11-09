#!/usr/bin/env python
"""B0 Step 6: the depth landscape on A10cR, with the context left alone (docs/plans/b0_router_audit.md).

  cache    per-row, per-modality CE sums of every stored any-to-any forced pass (A10cR d 1-3, A10cR5 d 1-5,
           A10c and A10c-s2 d 1-3), and of this step's TITAN V passes when they exist, into
           /data/enric/reports/b0/landscape/perrow.npz.
  tables   the A10cR table specs for 1980 and 3300 (specs/a10cr_<ck>.json), fixed before their units run:
             uniform  every key at depth 2: the control, which must equal force2 bit for bit
             worst    the equal-compute lattice table (5 slots, S1 pooled, depths 1-3, mean 2) with the highest
                      additive CE on the STORED all-forced passes at that checkpoint
             a10c     A10c's plateau time-mean mix per modality (Step 4)
             a10cs2   A10c-s2's plateau time-mean mix (Step 4)
             perm     A10c's plateau mix with its image rows cycled in registry order (Step 5's control)
           Key 0 (positions predicting no body token) and Coords sit at depth 2 in every table, as in the
           lattice, where Coords is pinned at the middle depth.
  read     the bowl (target-only CE(d) per target against all-forced), real against additive for the
           tables, and one table of spans. Writes read.{json,txt}.

    .venv/bin/python eo/experiments/routing/b0_landscape.py cache|tables|read

Runs in `.venv`, CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402

ARM = Path("/data/enric/reports/arm_eval")
OUT = Path("/data/enric/reports/b0/landscape")
AGREE = Path("/data/enric/reports/b0/agree/agree.json")
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
IMG = MODS[:6]
SLOTS = [("S2L2A", [0]), ("S1", [1, 2]), ("DEM", [3]), ("NDVI", [4]), ("LULC", [5])]
CKS = [330 * i for i in range(1, 11)]
TABLE_CKS = (1980, 3300)
N_BOOT = 10_000


def _tok(tag, ck, fname="tokens_router.npz"):
    f = ARM / tag / f"checkpoint-{ck}" / fname
    return np.load(f) if f.exists() else None


def perrow(ce, lm):
    """(rows, 7): per-row CE sums of each modality's body tokens."""
    ce = ce.astype(np.float64)
    return np.stack([np.where(lm == j + 1, ce, 0.0).sum(1) for j in range(7)], 1)


def cache() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lm = _tok("a10cr", 3300)["label_mod"]
    A = {"cnt": np.stack([(lm == j + 1).sum(1) for j in range(7)], 1).astype(np.float64)}
    rows = np.load(ARM / "a10cr/checkpoint-3300/eval_rows.npy")
    A["rows"] = rows
    for ck in CKS:
        for d in (1, 2, 3):
            z = _tok(f"a10cr_force{d}", ck)
            if z is not None:
                assert np.array_equal(z["label_mod"], lm)
                A[f"a10cr|f{d}|{ck}"] = perrow(z["ce"], lm)
        for d in (1, 2, 3, 4, 5):
            z = _tok(f"a10cr5_force{d}", ck)
            if z is not None:
                assert np.array_equal(z["label_mod"], lm)
                A[f"a10cr5|f{d}|{ck}"] = perrow(z["ce"], lm)
        for s in ("a10c", "a10cs2"):
            for d in (1, 2, 3):
                z = _tok(s, ck, f"tokens_force{d}.npz")
                if z is None:
                    z = _tok(f"{s}_any{d}", ck)
                if z is not None:
                    A[f"{s}|f{d}|{ck}"] = perrow(z["ce"], lm)
        print(ck, flush=True)
    np.savez(OUT / "perrow.npz", **A)
    print(len(A), "arrays ->", OUT / "perrow.npz")
    return 0


def slot(X):
    """(rows, 7) -> (rows, 5), S1GRD and S1RTC pooled (each row carries one of them)."""
    return np.stack([X[:, idx].sum(1) for _, idx in SLOTS], 1)


def additive(A, model, ck, depths):
    """Per slot CE at each forced depth (D, 5) and the slots' token weights (5,), Coords left out."""
    N = slot(A["cnt"])
    ce = np.stack([slot(A[f"{model}|f{d}|{ck}"]).sum(0) / N.sum(0) for d in depths], 0)
    return ce, N.sum(0) / N.sum()


def tables() -> int:
    A = dict(np.load(OUT / "perrow.npz"))
    agree = json.loads(AGREE.read_text())["windows"]
    lat = S.lattice([1, 2, 3], 5, 2)
    two = [0.0, 1.0, 0.0]
    for ck in TABLE_CKS:
        ce, w = additive(A, "a10cr", ck, (1, 2, 3))
        vals = (ce[lat - 1, np.arange(5)] * w).sum(1)
        worst = lat[int(np.argmax(vals))]
        uni = int(np.where((lat == 2).all(1))[0][0])
        mix_worst = {}
        for (name, idx), d in zip(SLOTS, worst):
            for i in idx:
                mix_worst[MODS[i]] = [float(k == d) for k in (1, 2, 3)]

        def plateau(tag):
            t = agree[f"{tag}|fp16|main|plateau"]
            out = {}
            for m in IMG:
                v = np.asarray(t[m]["mix"], dtype=np.float64)
                out[m] = (v / v.sum()).tolist()
            return out

        a10c, a10cs2 = plateau("a10c"), plateau("a10cs2")
        perm = {IMG[i]: a10c[IMG[(i + 1) % 6]] for i in range(6)}
        common = {"Coords": two, "none": two}
        spec = {"tables": {
            "uniform": {"mix": {**{m: two for m in IMG}, **common}, "gate": None, "seed": 0},
            "worst": {"mix": {**mix_worst, **common}, "gate": None, "seed": 0},
            "a10c": {"mix": {**a10c, **common}, "gate": None, "seed": 0},
            "a10cs2": {"mix": {**a10cs2, **common}, "gate": None, "seed": 0},
            "perm": {"mix": {**perm, **common}, "gate": None, "seed": 0}},
            "worst_slots": dict(zip([n for n, _ in SLOTS], worst.tolist())),
            "worst_additive_minus_uniform": float(vals.max() - vals[uni]),
            "span_additive": float(vals.max() - vals.min()),
            "source": "stored all-forced passes (a10cr_force1-3, GTX Titan X), image tokens, Coords left out"}
        out = OUT / "specs" / f"a10cr_{ck}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(spec, indent=1) + "\n")
        print(f"a10cr {ck}: worst table {spec['worst_slots']} (+{spec['worst_additive_minus_uniform']:.4f} over uniform; "
              f"span {spec['span_additive']:.4f}) -> {out}")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a == ["cache"]:
        return cache()
    if a == ["tables"]:
        return tables()
    if a == ["read"]:
        return read()
    raise SystemExit(__doc__)


# ------------------------------------------------------------------ read

def _global_groups():
    rows = np.load(ARM / "a10cr/checkpoint-3300/eval_rows.npy")
    g = S.eval_groups(rows)
    return {int(r): int(x) for r, x in zip(rows, g)}, int(g.max()) + 1


def _tl(tag, ck):
    f = OUT / tag / f"checkpoint-{ck}" / "target_last_rows.npz"
    return np.load(f) if f.exists() else None


def _ratio_diff(n, a, b, groups, W):
    """(sum a - sum b) / sum n, with a group-bootstrap 95% interval; a, b, n per row."""
    A_, B_, N_ = (np.bincount(groups, weights=x, minlength=W.shape[1]) for x in (a, b, n))
    est = float((A_.sum() - B_.sum()) / N_.sum())
    d = ((W @ A_) - (W @ B_)) / (W @ N_)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return est, float(lo), float(hi)


def _main_perrow(tag, ck, fname):
    f = OUT / tag / f"checkpoint-{ck}" / fname
    if not f.exists():
        return None
    z = np.load(f)
    lm = np.load(ARM / "a10cr/checkpoint-3300/tokens_router.npz")["label_mod"]
    return perrow(z["ce"], lm), z["ce"]


def _lattice_stats(ce_d, N_slot_rows, X_rows, depths, mean, groups, W, rng_seed=1):
    """Span, paired interval, best - uniform with interval, cross-validated gain, over the exact lattice.
    X_rows: (D, rows, 5) slot CE sums per row at each depth; N_slot_rows: (rows, 5) counts."""
    lat = S.lattice(list(depths), 5, mean)
    idx = lat - depths[0]
    w = N_slot_rows.sum(0) / N_slot_rows.sum()
    v = (ce_d[idx, np.arange(5)] * w).sum(1)
    u = int(np.where((lat == mean).all(1))[0][0])
    G = W.shape[1]
    Xg = np.stack([np.stack([np.bincount(groups, weights=X_rows[d][:, k], minlength=G) for k in range(5)], 1)
                   for d in range(len(depths))], 0)                                  # D x G x 5
    Ng = np.stack([np.bincount(groups, weights=N_slot_rows[:, k], minlength=G) for k in range(5)], 1)
    Xb = np.einsum("bg,dgs->bds", W, Xg)
    Nb = W @ Ng
    ceb = Xb / Nb[:, None, :]
    wb = Nb / Nb.sum(1, keepdims=True)
    vb = (ceb[:, idx, np.arange(5)] * wb[:, None, :]).sum(-1)
    best, worst = int(v.argmin()), int(v.argmax())
    rng = np.random.default_rng(rng_seed)
    cv = []
    for _ in range(200):
        perm = rng.permutation(G)
        a, b = perm[: G // 2], perm[G // 2:]
        ca = Xg[:, a].sum(1) / Ng[a].sum(0)
        cb = Xg[:, b].sum(1) / Ng[b].sum(0)
        va = (ca[idx, np.arange(5)] * w).sum(1)
        vb2 = (cb[idx, np.arange(5)] * w).sum(1)
        cv.append(vb2[int(va.argmin())] - vb2[u])
    cv = np.asarray(cv)
    sp = vb[:, worst] - vb[:, best]
    bu = vb[:, best] - vb[:, u]
    return dict(n=len(lat), span=float(v[worst] - v[best]), span_ci=np.percentile(sp, [2.5, 97.5]).tolist(),
                best=[int(x) for x in lat[best]], worst=[int(x) for x in lat[worst]],
                best_minus_uniform=float(v[best] - v[u]), bmu_ci=np.percentile(bu, [2.5, 97.5]).tolist(),
                cv_gain=float(cv.mean()), cv_sd=float(cv.std()), uniform_is_best=bool(best == u))


def read() -> int:
    lines, res = [], {}

    def P(*a):
        t = " ".join(str(x) for x in a)
        lines.append(t)
        print(t, flush=True)

    gmap, G = _global_groups()
    W = S.boot_weights(G, N_BOOT, 20261109)
    A = dict(np.load(OUT / "perrow.npz"))
    lm_main = np.load(ARM / "a10cr/checkpoint-3300/tokens_router.npz")["label_mod"]
    rows_main = A["rows"]
    g_main = np.array([gmap[int(r)] for r in rows_main])

    # ---- 1. the bowl
    P("== 1. Is the bowl real? Best depth per target, target scope (context left alone) against every position forced")
    best_to, best_af, cells = {}, {}, []
    for ck in CKS:
        tl = [_tl(f"a10cr_tforce{d}", ck) for d in (1, 2, 3)]
        mets = [ARM / f"a10cr_force{d}" / f"checkpoint-{ck}" / "metrics.json" for d in (1, 2, 3)]
        for T in MODS:
            if all(t is not None for t in tl):
                ce = [float(t[f"{T}|ce_sum"].sum() / t[f"{T}|n"].sum()) for t in tl]
                best_to[(ck, T)] = (int(np.argmin(ce)) + 1, ce)
            if all(m.exists() for m in mets):
                ce = [json.loads(m.read_text())["target_last"][T]["ce"] for m in mets]
                best_af[(ck, T)] = (int(np.argmin(ce)) + 1, ce)
    for ck in CKS:
        row = []
        for T in MODS:
            a, b = best_to.get((ck, T)), best_af.get((ck, T))
            row.append(f"{T[:5]}:{a[0] if a else '-'}/{b[0] if b else '-'}")
        P(f"  {ck:5d} best depth, target scope / all forced: " + " ".join(row))
    img_to = [best_to[(ck, T)][0] for ck in CKS for T in IMG if (ck, T) in best_to]
    img_af = [best_af[(ck, T)][0] for ck in CKS for T in IMG if (ck, T) in best_af]
    n2_to, n2_af = sum(x == 2 for x in img_to), sum(x == 2 for x in img_af)
    verdict = ("incomplete" if len(img_to) < 60 else "holds" if n2_to >= 48 else "goes" if n2_to < 30 else "partial")
    P(f"  depth 2 best: target scope {n2_to} of {len(img_to)} image cells (depth 3: {sum(x == 3 for x in img_to)}, "
      f"depth 1: {sum(x == 1 for x in img_to)}); all forced {n2_af} of {len(img_af)}  ->  the bowl {verdict.upper()}")
    res["bowl"] = dict(target_scope_d2=n2_to, target_scope_n=len(img_to), all_forced_d2=n2_af, all_forced_n=len(img_af),
                       verdict=verdict, best_to={f"{k[0]}|{k[1]}": v for k, v in best_to.items()},
                       best_af={f"{k[0]}|{k[1]}": v for k, v in best_af.items()})
    P("  paired, 95% group intervals: CE(1) - CE(2) and CE(3) - CE(2), target scope | all forced (TITAN V)")
    res["bowl_paired"] = {}
    for ck in TABLE_CKS:
        to = [_tl(f"a10cr_tforce{d}", ck) for d in (1, 2, 3)]
        af = [_tl(f"a10cr_tvforce{d}", ck) for d in (1, 2, 3)]
        for T in MODS:
            out = {}
            for frame, t in (("target", to), ("all", af)):
                if not all(x is not None for x in t):
                    continue
                rws = t[0][f"{T}|rows"]
                assert all(np.array_equal(x[f"{T}|rows"], rws) for x in t)
                g = np.array([gmap[int(r)] for r in rws])
                n = t[0][f"{T}|n"].astype(np.float64)
                c = [x[f"{T}|ce_sum"] for x in t]
                out[frame] = dict(d1=_ratio_diff(n, c[0], c[1], g, W), d3=_ratio_diff(n, c[2], c[1], g, W))
            if out:
                res["bowl_paired"][f"{ck}|{T}"] = out
                P(f"  {ck:5d} {T:7s} " + " | ".join(
                    f"{fr}: d1-d2 {v['d1'][0]:+.4f} [{v['d1'][1]:+.4f},{v['d1'][2]:+.4f}]  d3-d2 {v['d3'][0]:+.4f} "
                    f"[{v['d3'][1]:+.4f},{v['d3'][2]:+.4f}]" for fr, v in out.items()))

    # ---- 2. real against additive
    P("\n== 2. Is the additive estimate right? Real table pass against the additive estimate, image tokens, TITAN V")
    res["additive"] = {}
    verdicts = []
    for ck in TABLE_CKS:
        tv = [_main_perrow(f"a10cr_tvforce{d}", ck, "tokens_router.npz") for d in (1, 2, 3)]
        if any(x is None for x in tv):
            P(f"  {ck}: TITAN V all-forced passes missing")
            continue
        X = np.stack([x[0] for x in tv], 0)                         # D x rows x 7
        cnt = A["cnt"]
        ce_m = X.sum(1) / cnt.sum(0)                                # D x 7
        spec = json.loads((OUT / "specs" / f"a10cr_{ck}.json").read_text())
        Ns = slot(cnt)
        ce_slot = np.stack([slot(X[d]).sum(0) / Ns.sum(0) for d in range(3)], 0)
        lat_stats = _lattice_stats(ce_slot, Ns, np.stack([slot(X[d]) for d in range(3)], 0), (1, 2, 3), 2, g_main, W)
        span = lat_stats["span"]
        n_img = cnt[:, :6].sum(0)
        rows_out, real_add = {}, {}
        uni = _main_perrow("a10cr_tab_uniform", ck, "tokens_table.npz")
        ctrl = None
        if uni is not None:
            ctrl = bool(np.array_equal(uni[1], tv[1][1]))
        for name in ("uniform", "worst", "a10c", "a10cs2", "perm"):
            t = _main_perrow(f"a10cr_tab_{name}", ck, "tokens_table.npz")
            if t is None:
                continue
            mix = spec["tables"][name]["mix"]
            p = np.array([mix[m] for m in IMG])                     # 6 x 3
            add_m = (p * ce_m[:, :6].T).sum(1)                      # per image modality
            additive = float((add_m * n_img).sum() / n_img.sum())
            real_m = t[0][:, :6].sum(0) / n_img
            real = float(t[0][:, :6].sum() / n_img.sum())
            real_add[name] = (real, additive)
            rows_out[name] = dict(real=real, additive=additive, diff=real - additive,
                                  per_modality={m: [float(real_m[j]), float(add_m[j])] for j, m in enumerate(IMG)})
            P(f"  {ck:5d} {name:8s} real {real:.4f} additive {additive:.4f} diff {real - additive:+.4f}   "
              + " ".join(f"{m[:5]}:{real_m[j] - add_m[j]:+.4f}" for j, m in enumerate(IMG)))
        non = [n for n in ("worst", "a10c", "a10cs2", "perm") if n in real_add]
        diffs = [abs(real_add[n][0] - real_add[n][1]) for n in non]
        order_ok = (list(np.argsort([real_add[n][0] for n in non])) == list(np.argsort([real_add[n][1] for n in non])))
        if ctrl is False:
            v = "CONTROL FAILED: the uniform table is not force2; nothing here is read"
        elif len(non) < 4:
            v = "incomplete"
        elif max(diffs) < span / 4 and order_ok:
            v = "right"
        elif max(diffs) > span / 2:
            v = "wrong"
        else:
            v = "approximate"
        verdicts.append(v)
        res["additive"][str(ck)] = dict(tables=rows_out, span_titanv=span, control_uniform_eq_force2=ctrl,
                                        order_ok=bool(order_ok), max_abs_diff=max(diffs) if diffs else None, verdict=v)
        P(f"  {ck:5d} control: uniform table == force2 bit for bit = {ctrl}; 51-table span on the TITAN V {span:.4f}; "
          f"max |real - additive| {max(diffs) if diffs else float('nan'):.4f} (quarter span {span / 4:.4f}); "
          f"order kept = {order_ok} -> {v.upper()}")

    # ---- 3. spans
    P("\n== 3. Spans: 51-table span (381 for A10cR5) [95% paired], best - uniform [95%], cross-validated gain +- SD")
    res["spans"] = {}
    for label, model, depths, mean in (("A10cR all forced", "a10cr", (1, 2, 3), 2), ("A10c all forced", "a10c", (1, 2, 3), 2),
                                       ("A10c-s2 all forced", "a10cs2", (1, 2, 3), 2),
                                       ("A10cR5 all forced", "a10cr5", (1, 2, 3, 4, 5), 3)):
        for ck in TABLE_CKS:
            keys = [f"{model}|f{d}|{ck}" for d in depths]
            if not all(k in A for k in keys):
                continue
            Xr = np.stack([slot(A[k]) for k in keys], 0)
            Ns = slot(A["cnt"])
            ce_d = Xr.sum(1) / Ns.sum(0)
            st = _lattice_stats(ce_d, Ns, Xr, depths, mean, g_main, W)
            res["spans"][f"{label}|{ck}"] = st
            P(f"  {label:22s} {ck:5d} n={st['n']:3d} span {st['span']:.4f} [{st['span_ci'][0]:.4f},{st['span_ci'][1]:.4f}]"
              f"  best-uni {st['best_minus_uniform']:+.4f} [{st['bmu_ci'][0]:+.4f},{st['bmu_ci'][1]:+.4f}]"
              f"  cv {st['cv_gain']:+.4f}+-{st['cv_sd']:.4f}  best {st['best']} worst {st['worst']}")
    # A10cR, target only: the lattice over target-scope CE per slot, the main frame's token shares as weights
    for ck in TABLE_CKS:
        tl = [_tl(f"a10cr_tforce{d}", ck) for d in (1, 2, 3)]
        if not all(t is not None for t in tl):
            continue
        Ns_main = slot(A["cnt"]).sum(0)
        Xr, Nr = [], None
        for d, t in enumerate(tl):
            cols, ncols = [], []
            for name, idx in SLOTS:
                xs = np.zeros(G); ns = np.zeros(G)
                for i in idx:
                    T = MODS[i]
                    g = np.array([gmap[int(r)] for r in t[f"{T}|rows"]])
                    xs += np.bincount(g, weights=t[f"{T}|ce_sum"], minlength=G)
                    ns += np.bincount(g, weights=t[f"{T}|n"].astype(np.float64), minlength=G)
                cols.append(xs); ncols.append(ns)
            Xr.append(np.stack(cols, 1))
            Nr = np.stack(ncols, 1)
        Xr = np.stack(Xr, 0)                                      # D x G x 5, already per group
        ce_d = Xr.sum(1) / Nr.sum(0)
        # weights: the main frame's slot shares, so the lattice reads as the all-forced one does
        scale = Ns_main / Ns_main.sum() / (Nr.sum(0) / Nr.sum())
        st = _lattice_stats(ce_d, Nr * scale, Xr * scale, (1, 2, 3), 2, np.arange(G), W)
        res["spans"][f"A10cR target only|{ck}"] = st
        P(f"  {'A10cR target only':22s} {ck:5d} n={st['n']:3d} span {st['span']:.4f} [{st['span_ci'][0]:.4f},{st['span_ci'][1]:.4f}]"
          f"  best-uni {st['best_minus_uniform']:+.4f} [{st['bmu_ci'][0]:+.4f},{st['bmu_ci'][1]:+.4f}]"
          f"  cv {st['cv_gain']:+.4f}+-{st['cv_sd']:.4f}  best {st['best']} worst {st['worst']}")
    for ck in TABLE_CKS:
        r = res["additive"].get(str(ck), {}).get("tables", {})
        if len(r) == 5:
            reals = {k: v["real"] for k, v in r.items()}
            P(f"  {'A10cR, the five real tables':22s} {ck:5d} real span {max(reals.values()) - min(reals.values()):.4f}; "
              + " ".join(f"{k} {v - reals['uniform']:+.4f}" for k, v in reals.items() if k != "uniform"))

    # ---- 4. the paired baseline and the gate's redraw (description)
    P("\n== 4. As routed (own draw) against the forced depths, and the gate's redraw (target scope, 3300)")
    for ck in TABLE_CKS:
        rt = _tl("a10cr_routed", ck)
        to = [_tl(f"a10cr_tforce{d}", ck) for d in (1, 2, 3)]
        if rt is None or any(t is None for t in to):
            continue
        P(f"  {ck:5d} routed CE minus the mean of CE(1..3): " + " ".join(
            f"{T[:5]}:{rt[f'{T}|ce_sum'].sum() / rt[f'{T}|n'].sum() - np.mean([t[f'{T}|ce_sum'].sum() / t[f'{T}|n'].sum() for t in to]):+.4f}"
            for T in MODS))
    for d in (1, 3):
        a, b = _tl(f"a10cr_tforce{d}", 3300), _tl(f"a10cr_tforce{d}_gs1", 3300)
        if a is None or b is None:
            continue
        P(f"  gate redraw, depth {d}: CE(seed 1) - CE(seed 0) " + " ".join(
            f"{T[:5]}:{b[f'{T}|ce_sum'].sum() / b[f'{T}|n'].sum() - a[f'{T}|ce_sum'].sum() / a[f'{T}|n'].sum():+.5f}" for T in MODS))
    (OUT / "read.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "read.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
