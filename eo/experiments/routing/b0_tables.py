#!/usr/bin/env python
"""B0 Step 5: is the router a per-modality table? (old E6, amended; docs/plans/b0_router_audit.md)

Three parts.

  build TAG CK   the table specs for one (seed, checkpoint), from that checkpoint's `router` unit on the
                 TITAN V (the snapshot tables share its card, rows and orders) and from Step 4's time-mean
                 mixes (`agree.json`). Writes /data/enric/reports/b0/tables/specs/<tag>_<ck>.json:
                   snap      the router's own depth mix per modality (label_mod frame), drawn per position
                   snapgate  the same, the gate per (modality, depth) = the router's mean top probability
                   avg       Step 4's window mix: early (330), plateau (990, 1980, 2970), anneal (3300)
                   perm      the control: snap's rows cycled among the six image modalities in registry
                             order (S2L2A takes S1GRD's, ..., LULC takes S2L2A's); Coords keeps its own
                   snap_s1   snap drawn with seed 1: the table's own redraw noise
                 Key "none" (positions predicting no body token) keeps the router's choice in every table.
  read           every finished unit against its `router` unit, paired: pooled CE over body tokens and
                 top-1 per image modality, with group-bootstrap intervals; the per-unit checks (realised
                 mix within 4 SE of the row, compute, applied gate); and the rule. Writes read.json,
                 read.txt.
  token TAG CK   the cross-fitted token-id table, drawn offline into a per-position depth array for
                 eval_checkpoint's --assign (only where `snap` fails with power: the rule). Per input
                 token id (a block's first position: its BO, per modality), the router's depth mix on the
                 OTHER half of the spatial groups (finding 10's split), drawn per position; an id unseen
                 there falls back to that half's modality mix. Positions predicting no body token keep
                 the router's choice. Writes specs/<tag>_<ck>_token.npz.
  fit            finding 10 per modality and checkpoint (CPU, stored A100 arrays): how much of the
                 router's per-token depth a cross-fitted table reproduces on the other half of the spatial
                 groups, both directions, against `untrained` and A10cR. Writes fit.json, fit.txt.

    .venv/bin/python eo/experiments/routing/b0_tables.py build a10c 1980
    .venv/bin/python eo/experiments/routing/b0_tables.py read
    .venv/bin/python eo/experiments/routing/b0_tables.py fit

Runs in `.venv`, CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402

TAB = Path("/data/enric/reports/b0/tables")
AGREE = Path("/data/enric/reports/b0/agree/agree.json")
ARM = Path("/data/enric/reports/arm_eval")
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
IMG = MODS[:6]
TAGS = ("a10c", "a10cs2")
CKS = (330, 990, 1980, 2970, 3300)
WINDOW_OF = {330: "early", 990: "plateau", 1980: "plateau", 2970: "plateau", 3300: "anneal"}
TABLES = ("snap", "snapgate", "avg", "perm", "snap_s1")
CE_BOUND, TOP1_BOUND = 0.005, 0.08          # nats pooled; percentage points per image modality
N_BOOT = 10_000


def unit_dir(tag, name, ck):
    return TAB / f"{tag}_{name}" / f"checkpoint-{ck}"


# ------------------------------------------------------------------ build

def build(tag: str, ck: int) -> Path:
    z = np.load(unit_dir(tag, "router", ck) / "tokens_router.npz")
    d, lm, g = z["depth"], z["label_mod"], z["gate"].astype(np.float64)
    mix, gate, counts = {}, {}, {}
    for j, m in enumerate(MODS):
        s = lm == j + 1
        cnt = np.bincount(d[s].astype(np.int64), minlength=4)[1:].astype(np.float64)
        counts[m] = cnt.astype(int).tolist()
        mix[m] = (cnt / cnt.sum()).tolist()
        row = []
        for k in (1, 2, 3):
            sk = s & (d == k)
            row.append(float(g[sk].mean()) if sk.any() else float(g[s].mean()))
        gate[m] = [min(1.0, max(v, 1 / 3 + 1e-6)) for v in row]     # a gate is a top probability: >= 1/3
    win = WINDOW_OF[ck]
    tm = json.loads(AGREE.read_text())["windows"][f"{tag}|fp16|main|{win}"]
    avg = {}
    for m in MODS:
        v = np.asarray(tm[m]["mix"], dtype=np.float64)
        avg[m] = (v / v.sum()).tolist()
    perm = {IMG[i]: mix[IMG[(i + 1) % 6]] for i in range(6)}
    perm["Coords"] = mix["Coords"]
    none = {"none": None}
    spec = {"tables": {
        "snap": {"mix": {**mix, **none}, "gate": None, "seed": 0},
        "snapgate": {"mix": {**mix, **none}, "gate": gate, "seed": 0},
        "avg": {"mix": {**avg, **none}, "gate": None, "seed": 0},
        "perm": {"mix": {**perm, **none}, "gate": None, "seed": 0},
        "snap_s1": {"mix": {**mix, **none}, "gate": None, "seed": 1}},
        "source": str(unit_dir(tag, "router", ck) / "tokens_router.npz"), "window": win,
        "counts": counts, "perm_rule": "image rows cycled in registry order; Coords its own"}
    out = TAB / "specs" / f"{tag}_{ck}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1) + "\n")
    print(f"{tag} {ck}: snap " + " ".join(f"{m[:5]}:{np.dot(mix[m], [1, 2, 3]):.2f}" for m in MODS)
          + f" | avg ({win}) " + " ".join(f"{np.dot(avg[m], [1, 2, 3]):.2f}" for m in MODS) + f" -> {out}")
    return out


# ------------------------------------------------------------------ token (the cross-fitted token-id table)

def folds_of(rows):
    """Finding 10's split: the 256 spatial groups shuffled with seed 0, the second half is fold 1."""
    groups = S.eval_groups(rows)
    ug = np.unique(groups)
    np.random.default_rng(0).shuffle(ug)
    fold_of_group = np.zeros(int(groups.max()) + 1, np.int64)
    fold_of_group[ug[len(ug) // 2:]] = 1
    return fold_of_group[groups]


def input_ids(lab, lm):
    """The input token at each position: the label one position back; a block's first predicting
    position reads its BO, keyed per modality as -(modality + 1)."""
    tid = np.full(lab.shape, -1, np.int64)
    tid[:, 1:] = lab[:, :-1]
    return np.where(tid >= 0, tid, -(lm.astype(np.int64) + 1))


def token(tag: str, ck: int, seed: int = 20261108) -> Path:
    z = np.load(unit_dir(tag, "router", ck) / "tokens_router.npz")
    rows = np.load(unit_dir(tag, "router", ck) / "eval_rows.npy")
    d, lm, lab = z["depth"].astype(np.int64), z["label_mod"], z["label"]
    fold = folds_of(rows)
    tid = input_ids(lab, lm)
    body = lm > 0
    frow = np.broadcast_to(fold[:, None], lm.shape)
    rng = np.random.default_rng(seed)
    out = np.zeros(lm.shape, np.int8)
    seen_share = {}
    for f in (0, 1):
        fit_sel, app_sel = body & (frow != f), body & (frow == f)
        keys, inv = np.unique(tid[fit_sel], return_inverse=True)
        cnt = np.zeros((len(keys), 3), np.float64)
        np.add.at(cnt, (inv, d[fit_sel] - 1), 1)
        mod_mix = np.zeros((8, 3))
        np.add.at(mod_mix, (lm[fit_sel].astype(np.int64), d[fit_sel] - 1), 1)
        mod_mix /= np.maximum(mod_mix.sum(1, keepdims=True), 1)
        k_app = tid[app_sel]
        pos = np.searchsorted(keys, k_app)
        pos = np.clip(pos, 0, len(keys) - 1)
        seen = keys[pos] == k_app
        prob = np.where(seen[:, None], cnt[pos] / np.maximum(cnt[pos].sum(1, keepdims=True), 1),
                        mod_mix[lm[app_sel].astype(np.int64)])
        u = rng.random(len(k_app))
        draw = (u[:, None] >= np.cumsum(prob, 1)[:, :2]).sum(1) + 1
        out[app_sel] = draw.astype(np.int8)
        seen_share[f] = float(seen.mean())
    agree = float((out[body] == d[body]).mean())
    path = TAB / "specs" / f"{tag}_{ck}_token.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth=out, rows=rows, fold=fold)
    print(f"{tag} {ck}: token-id table drawn; ids seen in the other half {seen_share[0]:.3f}/{seen_share[1]:.3f}; "
          f"drawn depth = the router's on {agree:.3f} of body positions; mean depth {out[body].mean():.3f} "
          f"(router {d[body].mean():.3f}) -> {path}")
    return path


# ------------------------------------------------------------------ read

def _row_sums(z, lm, key, mod=None):
    """Per row: sum of `key` over body tokens (all, or one modality's)."""
    sel = lm > 0 if mod is None else lm == mod
    return np.where(sel, z[key].astype(np.float64), 0.0).sum(1), sel.sum(1).astype(np.float64)


def paired(zr, zt, lm, groups, W, key, mod=None, scale=1.0):
    """table - router of a token mean, paired by row, with a group-bootstrap 95% interval."""
    nr, den = _row_sums(zr, lm, key, mod)
    nt, _ = _row_sums(zt, lm, key, mod)
    # every group keeps its slot, so the sums line up with W's columns even when a modality (S1GRD)
    # is absent from whole groups
    N1, N2, D = (np.bincount(groups, weights=x, minlength=W.shape[1]) for x in (nt, nr, den))
    est = float((N1.sum() - N2.sum()) / D.sum()) * scale
    draws = ((W @ N1) - (W @ N2)) / (W @ D) * scale
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(d=est, lo=float(lo), hi=float(hi), table=float(N1.sum() / D.sum()) * scale,
                router=float(N2.sum() / D.sum()) * scale)


def mix_check(zt, lm, row, m_id):
    """Realised depth shares of one modality against its table row: worst |z| over depths."""
    s = lm == m_id
    n = int(s.sum())
    d = zt["depth"][s]
    worst = 0.0
    for k, p in enumerate(row, start=1):
        share = float((d == k).mean())
        if p <= 0:
            if share > 0:
                return float("inf")
            continue
        worst = max(worst, abs(share - p) / np.sqrt(p * (1 - p) / n))
    return worst


def read() -> int:
    rows_ref, groups, W = None, None, None
    out, lines = {}, []

    def P(*a):
        s = " ".join(str(x) for x in a)
        lines.append(s)
        print(s, flush=True)

    P("== B0 Step 5: tables against the router, paired (table - router); TITAN V, fp16, 4,416 rows")
    P(f"   bounds: |pooled CE| <= {CE_BOUND} nats and |top-1| <= {TOP1_BOUND} pp for each image modality")
    for tag in TAGS:
        for ck in CKS:
            fr = unit_dir(tag, "router", ck) / "tokens_router.npz"
            if not fr.exists():
                continue
            zr = np.load(fr)
            rows = np.load(unit_dir(tag, "router", ck) / "eval_rows.npy")
            if rows_ref is None:
                rows_ref = rows
                groups = S.eval_groups(rows)
                W = S.boot_weights(int(groups.max()) + 1, N_BOOT, 20261108)
            elif not np.array_equal(rows, rows_ref):
                raise RuntimeError(f"{tag} {ck}: rows differ")
            lm = zr["label_mod"]
            spec = json.loads((TAB / "specs" / f"{tag}_{ck}.json").read_text())
            mr = json.loads((unit_dir(tag, "router", ck) / "metrics.json").read_text())
            depth_r = float(zr["depth"][lm > 0].mean())
            for name in TABLES + ("token",):
                ft = unit_dir(tag, name, ck) / ("tokens_assign.npz" if name == "token" else "tokens_table.npz")
                if not ft.exists():
                    continue
                zt = np.load(ft)
                mt = json.loads((unit_dir(tag, name, ck) / "metrics.json").read_text())
                which = (mt.get("assign") or {}).get("path", "") if name == "token" else mt["table"]["name"]
                if (name == "token" and not which.endswith(f"{tag}_{ck}_token.npz")) or \
                        (name != "token" and which != name) or mt["gpu"] != mr["gpu"]:
                    raise RuntimeError(f"{tag} {name} {ck}: {which} on {mt['gpu']}, router on {mr['gpu']}")
                if not np.array_equal(zt["ce"].shape, zr["ce"].shape):
                    raise RuntimeError(f"{tag} {name} {ck}: shapes differ")
                pooled = paired(zr, zt, lm, groups, W, "ce")
                per = {m: paired(zr, zt, lm, groups, W, "correct", j + 1, 100.0) for j, m in enumerate(MODS)}
                per_ce = {m: paired(zr, zt, lm, groups, W, "ce", j + 1) for j, m in enumerate(MODS)}
                ok = abs(pooled["d"]) <= CE_BOUND and all(abs(per[m]["d"]) <= TOP1_BOUND for m in IMG)
                if name == "token":
                    worst = 0.0                     # drawn offline per token id: no per-modality row to check
                else:
                    tbl = spec["tables"][name]["mix"]
                    worst = max(mix_check(zt, lm, tbl[m], j + 1) for j, m in enumerate(MODS))
                depth_t = float(zt["depth"][lm > 0].mean())
                gate_err = None
                if name == "snapgate":
                    G = spec["tables"][name]["gate"]
                    ga = zt["gate_applied"].astype(np.float64)
                    errs = []
                    for j, m in enumerate(MODS):
                        for k in (1, 2, 3):
                            s = (lm == j + 1) & (zt["depth"] == k)
                            if s.any():
                                errs.append(float(np.abs(ga[s] - G[m][k - 1]).max()))
                    gate_err = max(errs)
                r = dict(pooled=pooled, top1=per, ce=per_ce, within=bool(ok), mix_worst_z=worst,
                         depth_table=depth_t, depth_router=depth_r, gate_max_err=gate_err)
                out[f"{tag}|{name}|{ck}"] = r
                P(f"  {tag:6s} {ck:4d} {name:8s} pooled {pooled['d']:+.4f} [{pooled['lo']:+.4f}, {pooled['hi']:+.4f}]"
                  f"  top-1 pp " + " ".join(f"{m[:5]}:{per[m]['d']:+.3f}" for m in MODS)
                  + f"  {'WITHIN' if ok else 'outside'}  depth {depth_t:.3f}/{depth_r:.3f}  mix |z|max {worst:.1f}"
                  + (f"  gate err {gate_err:.4f}" if gate_err is not None else ""))
                P(f"  {'':6s} {'':4s} {'':8s} top-1 95%: " + " ".join(
                    f"{m[:5]}:[{per[m]['lo']:+.3f},{per[m]['hi']:+.3f}]" for m in IMG)
                  + "  CE: " + " ".join(f"{m[:5]}:{per_ce[m]['d']:+.4f}" for m in MODS))
    # the rule
    P("\n== the rule (old E6, amended): snap within the bounds at every unit, perm outside at each")
    verdict = {}
    for tag in TAGS:
        for ck in CKS:
            s, p = out.get(f"{tag}|snap|{ck}"), out.get(f"{tag}|perm|{ck}")
            if s is None or p is None:
                continue
            power = not p["within"]
            verdict[f"{tag}|{ck}"] = dict(snap_within=s["within"], perm_outside=power,
                                          snapgate_within=out.get(f"{tag}|snapgate|{ck}", {}).get("within"),
                                          token_within=out.get(f"{tag}|token|{ck}", {}).get("within"))
            P(f"  {tag:6s} {ck:4d}: snap {'within' if s['within'] else 'OUTSIDE'}, control {'outside' if power else 'INSIDE (no power)'}"
              f", snapgate {verdict[f'{tag}|{ck}']['snapgate_within']}, token-id {verdict[f'{tag}|{ck}']['token_within']}")
    if verdict:
        units = list(verdict.values())
        with_power = [u for u in units if u["perm_outside"]]
        fails = [k for k, u in verdict.items() if u["perm_outside"] and not u["snap_within"]]
        nopow = [k for k, u in verdict.items() if not u["perm_outside"]]
        if len(units) < 10:
            v = f"incomplete: {len(units)} of 10 units"
        elif fails:
            tok = [verdict[k]["token_within"] for k in fails]
            if any(t is None for t in tok):
                v = f"FAILS at {fails}: the cross-fitted token-id table follows"
            elif all(tok):
                v = f"the modality table FAILS at {fails}, the token-id table passes there: token identity"
            else:
                v = (f"the modality table FAILS at {fails}, the token-id table fails at "
                     f"{[k for k, t in zip(fails, tok) if not t]}: content dependence, B4 decides")
        elif nopow:
            v = f"passes where the test has power ({len(with_power)} of 10); no power at {nopow}"
        else:
            v = "PASSES at all 10 units, the control outside at each: at each moment the router is a per-modality table"
        gates = [u["snapgate_within"] for u in units]
        P(f"  verdict: {v}")
        P(f"  gate convention: snapgate within the bounds at {sum(bool(g) for g in gates)} of {len(gates)} units")
        out["verdict"] = v
    (TAB / "read.json").write_text(json.dumps(out, indent=1) + "\n")
    (TAB / "read.txt").write_text("\n".join(lines) + "\n")
    return 0


# ------------------------------------------------------------------ fit (finding 10, per modality)

def _structure(im):
    """Per position (input frame): rank of its block in the row and index inside the block."""
    R, L = im.shape
    rank = np.full(im.shape, -1, np.int64)
    idx = np.full(im.shape, -1, np.int64)
    for r in range(R):
        row = im[r]
        ch = np.flatnonzero(row[1:] != row[:-1]) + 1
        k = 0
        for a, b in zip(np.r_[0, ch], np.r_[ch, L]):
            if row[a] > 0:
                rank[r, a:b] = k
                idx[r, a:b] = np.arange(b - a)
                k += 1
    return rank, idx


def _cross_fit(key, d, fold, fallback):
    """The fold-f table's majority depth per key, scored on the other fold; unseen keys fall back."""
    u, inv = np.unique(key, return_inverse=True)
    pred = np.zeros_like(d)
    for f in (0, 1):
        tr, te = fold == f, fold != f
        cnt = np.zeros((len(u), 4), np.int64)
        np.add.at(cnt, (inv[tr], d[tr]), 1)
        p = cnt.argmax(1)[inv[te]]
        un = cnt.sum(1)[inv[te]] == 0
        p[un] = fallback[te][un]
        pred[te] = p
    return pred


def fit() -> int:
    rows = np.load(ARM / "a10c/checkpoint-3300/eval_rows.npy")
    groups = S.eval_groups(rows)
    rng = np.random.default_rng(0)
    ug = np.unique(groups)
    rng.shuffle(ug)
    fold_of_group = np.zeros(int(groups.max()) + 1, np.int64)
    fold_of_group[ug[len(ug) // 2:]] = 1
    fold_row = fold_of_group[groups]
    struct, res, lines = None, {}, []

    def P(*a):
        s = " ".join(str(x) for x in a)
        lines.append(s)
        print(s, flush=True)

    P("== finding 10 per modality (label frame): share of a modality's body tokens whose depth a table fitted"
      " on the other half of the spatial groups reproduces (both directions pooled; A -> B, B -> A)")
    units = [("a10c", "untrained")] + [(t, f"checkpoint-{c}") for t in TAGS for c in range(330, 3301, 330)] \
        + [("a10cr", f"checkpoint-{c}") for c in (1980, 3300)]
    for tag, ck in units:
        f = ARM / tag / ck / "tokens_router.npz"
        if not f.exists():
            continue
        z = np.load(f)
        if "depth" not in z.files:
            continue
        im, lm, lab = z["input_mod"], z["label_mod"], z["label"]
        d = z["depth"].astype(np.int64)
        if struct is None:
            struct = _structure(im)
        rank, idx = struct
        # label frame: position t predicts token t+1; its block and patch are those of t+1
        rk = np.full_like(rank, -1)
        ix = np.full_like(idx, -1)
        rk[:, :-1], ix[:, :-1] = rank[:, 1:], idx[:, 1:]
        tid = np.full(im.shape, -1, np.int64)
        tid[:, 1:] = lab[:, :-1]                         # the input token at t is the label at t-1
        tid = np.where(tid >= 0, tid, -(lm.astype(np.int64) + 1))   # a block's first position: its BO, per modality
        s = lm > 0
        rws = np.broadcast_to(np.arange(im.shape[0])[:, None], im.shape)[s]
        ftok = fold_row[rws]
        dd, m = d[s], lm[s].astype(np.int64)
        base = _cross_fit(m, dd, ftok, np.full_like(dd, 2))
        preds = {"modality": base,
                 "mod_x_rank": _cross_fit(m * 10 + rk[s], dd, ftok, base),
                 "mod_x_patch": _cross_fit(m * 1000 + ix[s], dd, ftok, base),
                 "token_id": _cross_fit(tid[s], dd, ftok, base)}
        line = {}
        for j, name in enumerate(MODS):
            sel = m == j + 1
            line[name] = {k: dict(both=float((v[sel] == dd[sel]).mean()),
                                  a_to_b=float((v[sel & (ftok == 1)] == dd[sel & (ftok == 1)]).mean()),
                                  b_to_a=float((v[sel & (ftok == 0)] == dd[sel & (ftok == 0)]).mean()))
                          for k, v in preds.items()}
        res[f"{tag}|{ck}"] = line
        P(f"  {tag:7s} {ck:16s} " + " ".join(
            f"{n[:5]}:{line[n]['modality']['both']:.2f}/{line[n]['token_id']['both']:.2f}" for n in MODS)
          + "   (modality / token id)")
    (TAB / "fit.json").write_text(json.dumps(res, indent=1) + "\n")
    (TAB / "fit.txt").write_text("\n".join(lines) + "\n")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["build"] and len(a) == 3:
        build(a[1], int(a[2]))
        return 0
    if a[:1] == ["token"] and len(a) == 3:
        token(a[1], int(a[2]))
        return 0
    if a == ["read"]:
        return read()
    if a == ["fit"]:
        return fit()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
