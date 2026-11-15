#!/usr/bin/env python
"""B0 Step 6's landscape measurement, for any random-router MoR run (docs/plans/b0_router_audit.md, Step 6;
docs/PAPER_B_PLAN.md, B3: "On every new random-router run, measure the allocation as well as depth").

  cache    per-row, per-modality CE sums of the run's all-forced passes at depths 1..Nr (every position at d),
           and of the comparators' stored passes, into <out>/perrow.npz, with <out>/perrow.json saying where each
           run's passes came from. The run's passes are its STORED any-to-any forced passes
           (<arm-eval-root>/<tag>_force<d>) when they exist at every depth at every table checkpoint, else its
           TITAN V units (<units>/<tag>_tvforce<d>); --forced picks one. A new run has only the TITAN V units.
  tables   the run's table specs (<out>/specs/<tag>_<ck>.json), fixed before their units run:
             uniform  every key at the middle depth: the control, which must equal tvforce<mid> bit for bit
             worst    the equal-compute lattice table (5 slots, S1 pooled, depths 1..Nr, mean the middle one) with
                      the highest additive CE on the run's cached all-forced passes at that checkpoint
           and, for Step 6's A10cR (3 depths), also
             a10c     A10c's plateau time-mean mix per modality (Step 4)
             a10cs2   A10c-s2's plateau time-mean mix (Step 4)
             perm     A10c's plateau mix with its image rows cycled in registry order (Step 5's control)
           Key 0 (positions predicting no body token) and Coords sit at the middle depth in every table, as in the
           lattice, where Coords is pinned there. A spec whose tables differ is never overwritten silently.
  read     1. the bowl (target-only CE(d) per target against all-forced); 2. real against additive for the tables;
           3. one table of spans; 4. as routed and the gate's redraw; 5. B3's measurement: its rule (ratified
           2026-11-11) with the verdict for each random-router run, the forced-depth curve, the target-only passes
           per target and per slot, the swap readout, and Step 6's controls. Writes <out>/read.{json,txt}.

    .venv/bin/python eo/experiments/routing/b0_landscape.py cache|tables|read [--tag a10cr] [--depths 3] ...

Units of a run, as eval_checkpoint.py --arm-tag names them under <units> (B0 Step 6, B3):
  <tag>_tforce<d>       target-last, target scope (--force-depth d --force-scope target --modes '')
  <tag>_tvforce<d>      every position at d, main pass and target-last (--force-depth d --modes router)
  <tag>_routed          every position at its own draw, target-last (--own-draw --modes '')
  <tag>_tforce<d>_gs1   target scope at gate seed 1, depths 1 and Nr, last checkpoint
  <tag>_tab_<name>      main pass under a table (--modes table --table <out>/specs/<tag>_<ck>.json:<name>)

With --tag a10cr (the default) every default is Step 6's: ten target-only checkpoints, the comparators A10c, A10c-s2
and A10cR5, five tables, out and units /data/enric/reports/b0/landscape. With another tag the defaults are B3's for
a new run: target-only checkpoints 1980-3300, no comparators, the tables uniform and worst.

Runs in `.venv`, CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402

ARM = Path("/data/enric/reports/arm_eval")
OUT = Path("/data/enric/reports/b0/landscape")
AGREE = Path("/data/enric/reports/b0/agree/agree.json")
ALL_MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
IMG = ALL_MODS[:6]
#: The targets every per-target reading covers: all seven, or the six image modalities for a run scored with Coords
#: dropped (set by _set_mods; read at call time, so rebinding it reaches every loop).
MODS = list(ALL_MODS)
SLOTS = [("S2L2A", [0]), ("S1", [1, 2]), ("DEM", [3]), ("NDVI", [4]), ("LULC", [5])]
CKS = [330 * i for i in range(1, 11)]
LATE = CKS[5:]                      # 1980 ... 3300: B3's target-only checkpoints (B0 Step 7)
TABLE_CKS = (1980, 3300)
N_BOOT = 10_000
BOOT_SEED = 20261109
SEED_GAP = 0.006                    # B3: A10c against A10c-s2, E1 at 3300, until a cell has two random-router seeds
STEP6 = "a10cr"
PLATEAU = ("a10c", "a10cs2", "perm")
# Step 6's other models, read from their stored all-forced passes beside A10cR (spans; B3's rule for a random router).
COMPARATORS = {"a10c": dict(label="A10c", nr=3, router="learned"),
               "a10cs2": dict(label="A10c-s2", nr=3, router="learned"),
               "a10cr5": dict(label="A10cR5", nr=5, router="random")}
NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


# ------------------------------------------------------------------ settings

def config(argv=None) -> SimpleNamespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("cache", "tables", "read"))
    ap.add_argument("--tag", default=STEP6, help="the run's unit-tag prefix (default a10cr, Step 6's A10cR)")
    ap.add_argument("--depths", type=int, default=3, help="Nr, the run's trained recursions: depths 1..Nr. Odd, since the "
                                                            "lattice's mean depth is the middle one (default 3)")
    ap.add_argument("--label", default=None, help="the run's name in the readings (default A10cR for a10cr, A10cR5 for "
                                                   "a10cr5, else the tag)")
    ap.add_argument("--checkpoints", default=None, help="comma list: the target-only checkpoints read (default Step 6's "
                                                         "ten for a10cr, else 1980,2310,2640,2970,3300)")
    ap.add_argument("--table-checkpoints", default=",".join(map(str, TABLE_CKS)),
                    help="the two checkpoints of B3's rule: all-forced passes, tables, spans (default 1980,3300)")
    ap.add_argument("--arm-eval-root", default=str(ARM), help="stored arm_eval units: the run's main pass and any "
                                                              "<tag>_force<d>, and the comparators'")
    ap.add_argument("--out", default=str(OUT), help="perrow.{npz,json}, specs/ and read.{json,txt} (default B0's landscape)")
    ap.add_argument("--units", default=None, help="the run's landscape units (default --out)")
    ap.add_argument("--forced", choices=("auto", "stored", "titanv"), default="auto",
                    help="the run's all-forced passes: stored <tag>_force<d>, its TITAN V units <tag>_tvforce<d>, or "
                         "auto (stored when they exist at every depth and table checkpoint, else the TITAN V units)")
    ap.add_argument("--comparators", default=None, help="comma list of Step 6's other models (a10c, a10cs2, a10cr5) or "
                                                        "'none' (default all three for a10cr, none otherwise)")
    ap.add_argument("--tables", default=None, help="comma list of table names (default uniform,worst,a10c,a10cs2,perm for "
                                                   "a10cr, else uniform,worst)")
    ap.add_argument("--seed-gap", type=float, default=SEED_GAP, help="B3's seed gap, in nats (default 0.006)")
    ap.add_argument("--overwrite-specs", action="store_true", help="tables: replace a spec whose tables differ")
    a = ap.parse_args(argv)
    step6 = a.tag == STEP6
    nr = int(a.depths)
    if nr < 3 or nr % 2 == 0:
        raise SystemExit(f"--depths {nr}: the lattice's mean is the middle depth, so Nr must be odd and at least 3")

    def ints(s):
        return tuple(int(x) for x in s.split(",") if x.strip())

    def names(s):
        return () if s.strip().lower() in ("", "none") else tuple(x.strip() for x in s.split(",") if x.strip())

    c = SimpleNamespace(
        cmd=a.cmd, tag=a.tag, nr=nr, depths=tuple(range(1, nr + 1)), mid=(nr + 1) // 2,
        label=a.label or (COMPARATORS[a.tag]["label"] if a.tag in COMPARATORS else "A10cR" if step6 else a.tag),
        cks=ints(a.checkpoints) if a.checkpoints else tuple(CKS if step6 else LATE),
        table_cks=ints(a.table_checkpoints), arm=Path(a.arm_eval_root), out=Path(a.out),
        units=Path(a.units) if a.units else Path(a.out), forced=a.forced,
        comparators=names(a.comparators) if a.comparators is not None else (
            ("a10c", "a10cs2", "a10cr5") if step6 else ()),
        tables=names(a.tables) if a.tables is not None else (("uniform", "worst") + (PLATEAU if step6 else ())),
        seed_gap=float(a.seed_gap), overwrite_specs=a.overwrite_specs)
    bad = [k for k in c.comparators if k not in COMPARATORS or k == c.tag]
    if bad:
        raise SystemExit(f"--comparators {bad}: known are {sorted(COMPARATORS)}, and not the run itself")
    bad = [t for t in c.tables if t not in ("uniform", "worst") + PLATEAU]
    if bad or "uniform" not in c.tables:
        raise SystemExit(f"--tables {c.tables}: from uniform, worst, {', '.join(PLATEAU)}; uniform (the control) always")
    if nr != 3 and any(t in PLATEAU for t in c.tables):
        raise SystemExit("the plateau tables (a10c, a10cs2, perm) are mixes over 3 depths")
    if len(c.table_cks) < 2:
        raise SystemExit("--table-checkpoints: B3's rule reads two checkpoints")
    return c


# ------------------------------------------------------------------ data

def _tok(c, tag, ck, fname="tokens_router.npz"):
    f = c.arm / tag / f"checkpoint-{ck}" / fname
    return np.load(f) if f.exists() else None


def perrow(ce, lm):
    """(rows, 7): per-row CE sums of each modality's body tokens."""
    ce = ce.astype(np.float64)
    return np.stack([np.where(lm == j + 1, ce, 0.0).sum(1) for j in range(7)], 1)


def slot(X):
    """(rows, 7) -> (rows, 5), S1GRD and S1RTC pooled (each row carries one of them)."""
    return np.stack([X[:, idx].sum(1) for _, idx in SLOTS], 1)


def additive(A, model, ck, depths):
    """Per slot CE at each forced depth (D, 5) and the slots' token weights (5,), Coords left out."""
    N = slot(A["cnt"])
    ce = np.stack([slot(A[f"{model}|f{d}|{ck}"]).sum(0) / N.sum(0) for d in depths], 0)
    return ce, N.sum(0) / N.sum()


def _stored_dir(c, d, ck):
    return c.arm / f"{c.tag}_force{d}" / f"checkpoint-{ck}"


def _tv_dir(c, d, ck):
    return c.units / f"{c.tag}_tvforce{d}" / f"checkpoint-{ck}"


def _af_dir(c, src, d, ck):
    """The run's all-forced pass at depth d: its stored arm_eval unit or its TITAN V landscape unit."""
    return _stored_dir(c, d, ck) if src == "stored" else _tv_dir(c, d, ck)


def forced_source(c) -> str:
    def have(src):
        return all((_af_dir(c, src, d, ck) / "tokens_router.npz").exists() for d in c.depths for ck in c.table_cks)
    if c.forced == "stored" or (c.forced == "auto" and have("stored")):
        if not have("stored"):
            raise SystemExit(f"--forced stored: {c.arm}/{c.tag}_force<d> missing at a depth 1..{c.nr} or a checkpoint "
                             f"{c.table_cks}")
        return "stored"
    return "titanv"


def cached_source(c) -> str:
    """The source the cache used (perrow.json); B0's own cache predates that file, and the same rule decides it."""
    f = c.out / "perrow.json"
    if f.exists():
        meta = json.loads(f.read_text())
        if c.tag not in meta["runs"]:
            raise SystemExit(f"{c.out}/perrow.npz holds no passes of {c.tag}: run cache with this --tag")
        src = meta["runs"][c.tag]["source"]
        if c.forced != "auto" and c.forced != src:
            raise SystemExit(f"the cache holds {c.tag}'s {src} passes, --forced asks {c.forced}: rerun cache")
        return src
    return forced_source(c)


def _card(gpu) -> str:
    g = str(gpu)
    return ("TITAN V" if "TITAN V" in g else "GTX Titan X" if "TITAN X" in g else "A100" if "A100" in g
            else "V100" if "V100" in g else g)


def _cards(dirs):
    """The card of each pass, from its metrics.json (the card of the last process that wrote the directory)."""
    out = []
    for p in dirs:
        m = p / "metrics.json"
        out.append(_card(json.loads(m.read_text()).get("gpu")) if m.exists() else "?")
    return out


def _cards_text(cards, depths) -> str:
    return cards[0] if len(set(cards)) == 1 else ", ".join(f"d{d} {k}" for d, k in zip(depths, cards))


def _run_cards(c, src, ck):
    return _cards([_af_dir(c, src, d, ck) for d in c.depths])


def _source_text(c, src, ck) -> str:
    """The spec's note on where its worst table came from. Step 6's said "GTX Titan X" for every stored pass; their
    metrics.json put a10cr_force1-3 at 1980 on the TITAN V (bit for bit a10cr_tvforce1-3), and at 3300 d1-d2 on the
    Titan X and d3 on the TITAN V, so the cards are now read from the passes."""
    cs = _cards_text(_run_cards(c, src, ck), c.depths)
    if src == "stored":
        return f"stored all-forced passes ({c.tag}_force1-{c.nr}, {cs}), image tokens, Coords left out"
    return f"TITAN V all-forced units ({c.tag}_tvforce1-{c.nr}, {cs}), image tokens, Coords left out"


def _where(c, model, nr, src) -> str:
    """Where a run's all-forced passes came from, with their cards at each table checkpoint."""
    if model == c.tag:
        dirs = f"{c.tag}_force1-{nr}" if src == "stored" else f"TITAN V units {c.tag}_tvforce1-{nr}"
        cards = [_cards_text(_run_cards(c, src, ck), c.depths) for ck in c.table_cks]
    else:
        dirs = f"{model}_force1-{nr}" if COMPARATORS[model]["router"] == "random" else model
        cards = [_cards_text(_cards([c.arm / f"{model}_force{d}" / f"checkpoint-{ck}" for d in range(1, nr + 1)]),
                             range(1, nr + 1)) for ck in c.table_cks]
    return f"{'stored ' if src == 'stored' else ''}{dirs}: " + "; ".join(f"{ck} {k}" for ck, k in zip(c.table_cks, cards))


def reference(c):
    """(label_mod, rows, dir) of the run's held-out sequences: its own main pass at the last table checkpoint (Step 6:
    arm_eval/a10cr/checkpoint-3300), else its stored or TITAN V all-forced pass at the middle depth."""
    cands = [c.arm / c.tag / f"checkpoint-{c.table_cks[-1]}"]
    for ck in reversed(c.table_cks):
        cands += [_stored_dir(c, c.mid, ck), _tv_dir(c, c.mid, ck)]
    for p in cands:
        if (p / "tokens_router.npz").exists() and (p / "eval_rows.npy").exists():
            return np.load(p / "tokens_router.npz")["label_mod"], np.load(p / "eval_rows.npy"), p
    raise SystemExit(f"{c.tag}: no main pass to take the held-out sequences from (tried {', '.join(map(str, cands))})")


def _comparator_pass(c, name, d, ck):
    """A comparator's stored all-forced pass at depth d. A random router's is <name>_force<d>; a learned one's is
    tokens_force<d>.npz in its own directory, else <name>_any<d> (Step 6's cache)."""
    if COMPARATORS[name]["router"] == "random":
        return _tok(c, f"{name}_force{d}", ck)
    z = _tok(c, name, ck, f"tokens_force{d}.npz")
    return z if z is not None else _tok(c, f"{name}_any{d}", ck)


def _check_lm(z, lm, what):
    if "label_mod" in z.files:
        assert np.array_equal(z["label_mod"], lm), f"{what}: another layout of the held-out rows"
    else:
        assert z["ce"].shape == lm.shape, f"{what}: shape {z['ce'].shape}, the held-out rows {lm.shape}"


def cache(c) -> int:
    c.out.mkdir(parents=True, exist_ok=True)
    lm, rows, ref = reference(c)
    A = {"cnt": np.stack([(lm == j + 1).sum(1) for j in range(7)], 1).astype(np.float64)}
    A["rows"] = rows
    src = forced_source(c)
    dirs = f"{c.arm}/{c.tag}_force<d>" if src == "stored" else f"{c.units}/{c.tag}_tvforce<d>"
    meta = {"reference": str(ref), "runs": {c.tag: {"source": src, "label": c.label, "depths": c.nr, "router": "random",
                                                    "dirs": dirs, "cards": {}}}}
    for k in c.comparators:
        meta["runs"][k] = {"source": "stored", "label": COMPARATORS[k]["label"], "depths": COMPARATORS[k]["nr"],
                           "router": COMPARATORS[k]["router"]}
    for ck in sorted(set(c.cks) | set(c.table_cks)):
        for d in c.depths:
            f = _af_dir(c, src, d, ck) / "tokens_router.npz"
            if f.exists():
                z = np.load(f)
                _check_lm(z, lm, str(f))
                A[f"{c.tag}|f{d}|{ck}"] = perrow(z["ce"], lm)
        if all(f"{c.tag}|f{d}|{ck}" in A for d in c.depths):
            meta["runs"][c.tag]["cards"][str(ck)] = _run_cards(c, src, ck)
        for k in c.comparators:
            for d in range(1, COMPARATORS[k]["nr"] + 1):
                z = _comparator_pass(c, k, d, ck)
                if z is not None:
                    _check_lm(z, lm, f"{k} depth {d} checkpoint-{ck}")
                    A[f"{k}|f{d}|{ck}"] = perrow(z["ce"], lm)
        print(ck, flush=True)
    for ck in c.table_cks:
        if not all(f"{c.tag}|f{d}|{ck}" in A for d in c.depths):
            raise SystemExit(f"{c.tag}: the {src} all-forced passes miss a depth 1..{c.nr} at checkpoint-{ck}")
    np.savez(c.out / "perrow.npz", **A)
    (c.out / "perrow.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(len(A), "arrays ->", c.out / "perrow.npz", f"({c.tag}: {src} all-forced passes)")
    return 0


# ------------------------------------------------------------------ tables

def tables(c) -> int:
    A = dict(np.load(c.out / "perrow.npz"))
    src = cached_source(c)
    lat = S.lattice(list(c.depths), 5, c.mid)
    mid1 = [float(k == c.mid) for k in c.depths]
    agree = json.loads(AGREE.read_text())["windows"] if any(t in PLATEAU for t in c.tables) else None

    def plateau(tag):
        t = agree[f"{tag}|fp16|main|plateau"]
        out = {}
        for m in IMG:
            v = np.asarray(t[m]["mix"], dtype=np.float64)
            out[m] = (v / v.sum()).tolist()
        return out

    for ck in c.table_cks:
        ce, w = additive(A, c.tag, ck, c.depths)
        vals = (ce[lat - 1, np.arange(5)] * w).sum(1)
        worst = lat[int(np.argmax(vals))]
        uni = int(np.where((lat == c.mid).all(1))[0][0])
        mix_worst = {}
        for (name, idx), d in zip(SLOTS, worst):
            for i in idx:
                mix_worst[MODS[i]] = [float(k == d) for k in c.depths]
        common = {"Coords": mid1, "none": mid1}
        made = {"uniform": {"mix": {**{m: mid1 for m in IMG}, **common}, "gate": None, "seed": 0},
                "worst": {"mix": {**mix_worst, **common}, "gate": None, "seed": 0}}
        if agree is not None:
            a10c, a10cs2 = plateau("a10c"), plateau("a10cs2")
            perm = {IMG[i]: a10c[IMG[(i + 1) % 6]] for i in range(6)}
            made.update({"a10c": {"mix": {**a10c, **common}, "gate": None, "seed": 0},
                         "a10cs2": {"mix": {**a10cs2, **common}, "gate": None, "seed": 0},
                         "perm": {"mix": {**perm, **common}, "gate": None, "seed": 0}})
        spec = {"tables": {t: made[t] for t in c.tables},
                "worst_slots": dict(zip([n for n, _ in SLOTS], worst.tolist())),
                "worst_additive_minus_uniform": float(vals.max() - vals[uni]),
                "span_additive": float(vals.max() - vals.min()),
                "source": _source_text(c, src, ck)}
        out = c.out / "specs" / f"{c.tag}_{ck}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and json.loads(out.read_text())["tables"] != spec["tables"] and not c.overwrite_specs:
            raise SystemExit(f"{out} holds other tables, and its units may have run with them: --overwrite-specs to "
                             f"replace it")
        out.write_text(json.dumps(spec, indent=1) + "\n")
        print(f"{c.tag} {ck}: worst table {spec['worst_slots']} (+{spec['worst_additive_minus_uniform']:.4f} over uniform; "
              f"span {spec['span_additive']:.4f}) -> {out}")
    return 0


def main() -> int:
    c = config()
    return {"cache": cache, "tables": tables, "read": read}[c.cmd](c)


# ------------------------------------------------------------------ read

def _tl(c, tag, ck):
    f = c.units / tag / f"checkpoint-{ck}" / "target_last_rows.npz"
    return np.load(f) if f.exists() else None


def _ratio_diff(n, a, b, groups, W):
    """(sum a - sum b) / sum n, with a group-bootstrap 95% interval; a, b, n per row."""
    A_, B_, N_ = (np.bincount(groups, weights=x, minlength=W.shape[1]) for x in (a, b, n))
    est = float((A_.sum() - B_.sum()) / N_.sum())
    d = ((W @ A_) - (W @ B_)) / (W @ N_)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return est, float(lo), float(hi)


def _main_perrow(c, lm, tag, ck, fname):
    f = c.units / tag / f"checkpoint-{ck}" / fname
    if not f.exists():
        return None
    z = np.load(f)
    _check_lm(z, lm, str(f))
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


def _table_sha(entry) -> str:
    """eval_checkpoint.load_table's hash of one table entry, which every table unit records."""
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]


def _set_mods(cnt) -> list:
    """The run's targets, from its own held-out token counts (perrow.npz's cnt, one column per ALL_MODS entry). A run
    scored with eval_checkpoint --drop-modalities Coords (the Coords-free runs of the 1 + 3 + 1 train cell, worklog §4
    items 59-60, 2026-11-12) has no Coords token and no Coords target-last: its per-target readings cover the six image
    modalities. Every other count must be positive. Rebinds the module's MODS."""
    global MODS
    n = cnt.sum(0)
    if len(n) != len(ALL_MODS) or not (n[:6] > 0).all():
        raise SystemExit(f"held-out token counts per modality {n.tolist()}: expected {len(ALL_MODS)} columns, the six "
                         f"image modalities all present")
    MODS = list(ALL_MODS) if n[6] > 0 else list(IMG)
    return MODS


def read(c) -> int:
    lines, res = [], {}

    def P(*a):
        t = " ".join(str(x) for x in a)
        lines.append(t)
        print(t, flush=True)

    lm_main, rows_ref, ref = reference(c)
    g_all = S.eval_groups(rows_ref)
    gmap, G = {int(r): int(x) for r, x in zip(rows_ref, g_all)}, int(g_all.max()) + 1
    W = S.boot_weights(G, N_BOOT, BOOT_SEED)
    A = dict(np.load(c.out / "perrow.npz"))
    _set_mods(A["cnt"])
    rows_main = A["rows"]
    if not np.array_equal(rows_main, rows_ref):
        raise SystemExit(f"{c.out}/perrow.npz holds other rows than {ref}")
    g_main = np.array([gmap[int(r)] for r in rows_main])
    src = cached_source(c)
    D, mid, depths, tag, last = c.nr, c.mid, c.depths, c.tag, c.table_cks[-1]
    others = [d for d in depths if d != mid]
    print(f"[{tag}] {c.label}: depths 1..{D} (middle {mid}); target-only checkpoints {list(c.cks)}; table checkpoints "
          f"{list(c.table_cks)}; all-forced passes {src}; units {c.units}; comparators {list(c.comparators) or 'none'}; "
          f"seed gap {c.seed_gap}", flush=True)
    res["settings"] = dict(tag=tag, label=c.label, depths=D, middle=mid, checkpoints=list(c.cks),
                           table_checkpoints=list(c.table_cks), forced_source=src, units=str(c.units), out=str(c.out),
                           arm_eval_root=str(c.arm), reference=str(ref), comparators=list(c.comparators),
                           seed_gap=c.seed_gap, n_boot=N_BOOT, boot_seed=BOOT_SEED, groups=G, targets=list(MODS))

    # ---- 1. the bowl
    P("== 1. Is the bowl real? Best depth per target, target scope (context left alone) against every position forced")
    best_to, best_af = {}, {}
    for ck in c.cks:
        tl = [_tl(c, f"{tag}_tforce{d}", ck) for d in depths]
        mets = [_af_dir(c, src, d, ck) / "metrics.json" for d in depths]
        for T in MODS:
            if all(t is not None for t in tl):
                ce = [float(t[f"{T}|ce_sum"].sum() / t[f"{T}|n"].sum()) for t in tl]
                best_to[(ck, T)] = (int(np.argmin(ce)) + 1, ce)
            if all(m.exists() for m in mets):
                ce = [json.loads(m.read_text())["target_last"][T]["ce"] for m in mets]
                best_af[(ck, T)] = (int(np.argmin(ce)) + 1, ce)
    for ck in c.cks:
        row = []
        for T in MODS:
            a, b = best_to.get((ck, T)), best_af.get((ck, T))
            row.append(f"{T[:5]}:{a[0] if a else '-'}/{b[0] if b else '-'}")
        P(f"  {ck:5d} best depth, target scope / all forced: " + " ".join(row))
    img_to = [best_to[(ck, T)][0] for ck in c.cks for T in IMG if (ck, T) in best_to]
    img_af = [best_af[(ck, T)][0] for ck in c.cks for T in IMG if (ck, T) in best_af]
    n_to, n_af = sum(x == mid for x in img_to), sum(x == mid for x in img_af)
    n_cells = 6 * len(c.cks)            # Step 6: 60 cells, holds at 48 (0.8), goes below 30 (0.5)
    verdict = ("incomplete" if len(img_to) < n_cells else "holds" if n_to >= 0.8 * n_cells
               else "goes" if n_to < 0.5 * n_cells else "partial")
    P(f"  depth {mid} best: target scope {n_to} of {len(img_to)} image cells ("
      + ", ".join(f"depth {d}: {sum(x == d for x in img_to)}" for d in reversed(others))
      + f"); all forced {n_af} of {len(img_af)}  ->  the bowl {verdict.upper()}")
    if n_cells != 60:
        P(f"  (Step 6's rule, 48 and 30 of 60 cells, scaled to {n_cells}: holds at >= {0.8 * n_cells:g}, goes below "
          f"{0.5 * n_cells:g}; description outside Step 6)")
    res["bowl"] = {f"target_scope_d{mid}": n_to, "target_scope_n": len(img_to), f"all_forced_d{mid}": n_af,
                   "all_forced_n": len(img_af), "verdict": verdict,
                   "best_to": {f"{k[0]}|{k[1]}": v for k, v in best_to.items()},
                   "best_af": {f"{k[0]}|{k[1]}": v for k, v in best_af.items()}}
    P(f"  paired, 95% group intervals: {' and '.join(f'CE({d}) - CE({mid})' for d in others)}, "
      f"target scope | all forced (TITAN V)")
    res["bowl_paired"] = {}
    for ck in c.table_cks:
        to = [_tl(c, f"{tag}_tforce{d}", ck) for d in depths]
        af = [_tl(c, f"{tag}_tvforce{d}", ck) for d in depths]
        for T in MODS:
            out = {}
            for frame, t in (("target", to), ("all", af)):
                if not all(x is not None for x in t):
                    continue
                rws = t[0][f"{T}|rows"]
                assert all(np.array_equal(x[f"{T}|rows"], rws) for x in t)
                g = np.array([gmap[int(r)] for r in rws])
                n = t[0][f"{T}|n"].astype(np.float64)
                cs = [x[f"{T}|ce_sum"] for x in t]
                out[frame] = {f"d{d}": _ratio_diff(n, cs[d - 1], cs[mid - 1], g, W) for d in others}
            if out:
                res["bowl_paired"][f"{ck}|{T}"] = out
                P(f"  {ck:5d} {T:7s} " + " | ".join(
                    f"{fr}: " + "  ".join(f"d{d}-d{mid} {v[f'd{d}'][0]:+.4f} [{v[f'd{d}'][1]:+.4f},{v[f'd{d}'][2]:+.4f}]"
                                          for d in others) for fr, v in out.items()))

    # ---- 2. real against additive
    P("\n== 2. Is the additive estimate right? Real table pass against the additive estimate, image tokens, TITAN V")
    res["additive"] = {}
    tvX, tv_lat, ctrls = {}, {}, {}
    for ck in c.table_cks:
        tv = [_main_perrow(c, lm_main, f"{tag}_tvforce{d}", ck, "tokens_router.npz") for d in depths]
        if any(x is None for x in tv):
            P(f"  {ck}: TITAN V all-forced passes missing")
            continue
        X = np.stack([x[0] for x in tv], 0)                         # D x rows x 7
        tvX[ck] = X
        cnt = A["cnt"]
        ce_m = X.sum(1) / cnt.sum(0)                                # D x 7
        spath = c.out / "specs" / f"{tag}_{ck}.json"
        spec = json.loads(spath.read_text()) if spath.exists() else None
        Ns = slot(cnt)
        ce_slot = np.stack([slot(X[d]).sum(0) / Ns.sum(0) for d in range(D)], 0)
        lat_stats = _lattice_stats(ce_slot, Ns, np.stack([slot(X[d]) for d in range(D)], 0), depths, mid, g_main, W)
        tv_lat[ck] = lat_stats
        span = lat_stats["span"]
        n_img = cnt[:, :6].sum(0)
        rows_out, real_add = {}, {}
        uni = _main_perrow(c, lm_main, f"{tag}_tab_uniform", ck, "tokens_table.npz")
        ctrl = None
        if uni is not None:
            ctrl = bool(np.array_equal(uni[1], tv[mid - 1][1]))
        ctrls[ck] = ctrl
        if spec is None:
            P(f"  {ck:5d} no spec ({spath}): the tables are not read")
        for name in (spec["tables"] if spec else {}):
            t = _main_perrow(c, lm_main, f"{tag}_tab_{name}", ck, "tokens_table.npz")
            if t is None:
                continue
            ran = json.loads((c.units / f"{tag}_tab_{name}" / f"checkpoint-{ck}" / "metrics.json").read_text())["table"]
            if ran["sha"] != _table_sha(spec["tables"][name]):
                raise SystemExit(f"{tag}_tab_{name} checkpoint-{ck} ran table {ran['sha']} ({ran['spec']}); "
                                 f"{spath} holds {_table_sha(spec['tables'][name])}")
            mix = spec["tables"][name]["mix"]
            p = np.array([mix[m] for m in IMG])                     # 6 x D
            add_m = (p * ce_m[:, :6].T).sum(1)                      # per image modality
            additive_ = float((add_m * n_img).sum() / n_img.sum())
            real_m = t[0][:, :6].sum(0) / n_img
            real = float(t[0][:, :6].sum() / n_img.sum())
            real_add[name] = (real, additive_)
            rows_out[name] = dict(real=real, additive=additive_, diff=real - additive_,
                                  per_modality={m: [float(real_m[j]), float(add_m[j])] for j, m in enumerate(IMG)})
            P(f"  {ck:5d} {name:8s} real {real:.4f} additive {additive_:.4f} diff {real - additive_:+.4f}   "
              + " ".join(f"{m[:5]}:{real_m[j] - add_m[j]:+.4f}" for j, m in enumerate(IMG)))
        non = [n for n in real_add if n != "uniform"]
        diffs = [abs(real_add[n][0] - real_add[n][1]) for n in non]
        order_ok = (list(np.argsort([real_add[n][0] for n in non])) == list(np.argsort([real_add[n][1] for n in non])))
        if ctrl is False:
            v = f"CONTROL FAILED: the uniform table is not force{mid}; nothing here is read"
        elif len(non) < 4:
            v = "incomplete"
        elif max(diffs) < span / 4 and order_ok:
            v = "right"
        elif max(diffs) > span / 2:
            v = "wrong"
        else:
            v = "approximate"
        res["additive"][str(ck)] = dict(tables=rows_out, span_titanv=span, control_uniform_eq_force2=ctrl,
                                        order_ok=bool(order_ok), max_abs_diff=max(diffs) if diffs else None, verdict=v)
        P(f"  {ck:5d} control: uniform table == force{mid} bit for bit = {ctrl}; {lat_stats['n']}-table span on the "
          f"TITAN V {span:.4f}; max |real - additive| {max(diffs) if diffs else float('nan'):.4f} (quarter span "
          f"{span / 4:.4f}); order kept = {order_ok} -> {v.upper()}")
        if len(non) < 4 and ctrl is not False:
            P(f"  {ck:5d} (Step 6's rule reads four non-uniform tables; {len(non)} here: the order test and the verdict "
              f"need them, so these differences are description)")

    # ---- 3. spans
    n_main = len(S.lattice(list(depths), 5, mid))
    extra = [f"{len(S.lattice(list(range(1, COMPARATORS[k]['nr'] + 1)), 5, (COMPARATORS[k]['nr'] + 1) // 2))} for "
             f"{COMPARATORS[k]['label']}" for k in c.comparators if COMPARATORS[k]["nr"] != D]
    P(f"\n== 3. Spans: {n_main}-table span{' (' + ', '.join(extra) + ')' if extra else ''} [95% paired], "
      f"best - uniform [95%], cross-validated gain +- SD")
    res["spans"] = {}
    runs = [(f"{c.label} all forced", tag, depths, mid)] + [
        (f"{COMPARATORS[k]['label']} all forced", k, tuple(range(1, COMPARATORS[k]["nr"] + 1)),
         (COMPARATORS[k]["nr"] + 1) // 2) for k in c.comparators]
    for label, model, dd, mean in runs:
        for ck in c.table_cks:
            keys = [f"{model}|f{d}|{ck}" for d in dd]
            if not all(k in A for k in keys):
                continue
            Xr = np.stack([slot(A[k]) for k in keys], 0)
            Ns = slot(A["cnt"])
            ce_d = Xr.sum(1) / Ns.sum(0)
            st = _lattice_stats(ce_d, Ns, Xr, dd, mean, g_main, W)
            res["spans"][f"{label}|{ck}"] = st
            P(f"  {label:22s} {ck:5d} n={st['n']:3d} span {st['span']:.4f} [{st['span_ci'][0]:.4f},{st['span_ci'][1]:.4f}]"
              f"  best-uni {st['best_minus_uniform']:+.4f} [{st['bmu_ci'][0]:+.4f},{st['bmu_ci'][1]:+.4f}]"
              f"  cv {st['cv_gain']:+.4f}+-{st['cv_sd']:.4f}  best {st['best']} worst {st['worst']}")
    # target only: the lattice over target-scope CE per slot, the main frame's token shares as weights; S1 pooled by
    # the target-last sample's rows (Step 6; section 5 pools it by held-out token share, as B3 asks)
    for ck in c.table_cks:
        tl = [_tl(c, f"{tag}_tforce{d}", ck) for d in depths]
        if not all(t is not None for t in tl):
            continue
        Ns_main = slot(A["cnt"]).sum(0)
        Xr, Nt = [], None
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
            Nt = np.stack(ncols, 1)
        Xr = np.stack(Xr, 0)                                      # D x G x 5, already per group
        ce_d = Xr.sum(1) / Nt.sum(0)
        # weights: the main frame's slot shares, so the lattice reads as the all-forced one does
        scale = Ns_main / Ns_main.sum() / (Nt.sum(0) / Nt.sum())
        st = _lattice_stats(ce_d, Nt * scale, Xr * scale, depths, mid, np.arange(G), W)
        res["spans"][f"{c.label} target only|{ck}"] = st
        P(f"  {c.label + ' target only':22s} {ck:5d} n={st['n']:3d} span {st['span']:.4f} [{st['span_ci'][0]:.4f},{st['span_ci'][1]:.4f}]"
          f"  best-uni {st['best_minus_uniform']:+.4f} [{st['bmu_ci'][0]:+.4f},{st['bmu_ci'][1]:+.4f}]"
          f"  cv {st['cv_gain']:+.4f}+-{st['cv_sd']:.4f}  best {st['best']} worst {st['worst']}")
    for ck in c.table_cks:
        r = res["additive"].get(str(ck), {}).get("tables", {})
        if len(r) >= 2 and "uniform" in r:
            reals = {k: v["real"] for k, v in r.items()}
            P(f"  {c.label + ', the ' + NUM.get(len(r), str(len(r))) + ' real tables':22s} {ck:5d} real span "
              f"{max(reals.values()) - min(reals.values()):.4f}; "
              + " ".join(f"{k} {v - reals['uniform']:+.4f}" for k, v in reals.items() if k != "uniform"))

    # ---- 4. the paired baseline and the gate's redraw (description)
    P(f"\n== 4. As routed (own draw) against the forced depths, and the gate's redraw (target scope, {last})")
    routed, redraw = {}, {}
    for ck in c.table_cks:
        rt = _tl(c, f"{tag}_routed", ck)
        to = [_tl(c, f"{tag}_tforce{d}", ck) for d in depths]
        if rt is None or any(t is None for t in to):
            continue
        routed[str(ck)] = {T: float(rt[f'{T}|ce_sum'].sum() / rt[f'{T}|n'].sum()
                                    - np.mean([t[f'{T}|ce_sum'].sum() / t[f'{T}|n'].sum() for t in to])) for T in MODS}
        P(f"  {ck:5d} routed CE minus the mean of CE(1..{D}): " + " ".join(
            f"{T[:5]}:{rt[f'{T}|ce_sum'].sum() / rt[f'{T}|n'].sum() - np.mean([t[f'{T}|ce_sum'].sum() / t[f'{T}|n'].sum() for t in to]):+.4f}"
            for T in MODS))
    for d in (1, D):
        a, b = _tl(c, f"{tag}_tforce{d}", last), _tl(c, f"{tag}_tforce{d}_gs1", last)
        if a is None or b is None:
            continue
        redraw[f"d{d}"] = {T: float(b[f'{T}|ce_sum'].sum() / b[f'{T}|n'].sum() - a[f'{T}|ce_sum'].sum() / a[f'{T}|n'].sum())
                           for T in MODS}
        P(f"  gate redraw, depth {d}: CE(seed 1) - CE(seed 0) " + " ".join(
            f"{T[:5]}:{b[f'{T}|ce_sum'].sum() / b[f'{T}|n'].sum() - a[f'{T}|ce_sum'].sum() / a[f'{T}|n'].sum():+.5f}" for T in MODS))

    # ---- 5. B3
    res["b3"] = _b3(c, P, res, A, src, gmap, G, W, g_main, tvX, tv_lat, ctrls, routed, redraw)
    c.out.mkdir(parents=True, exist_ok=True)
    (c.out / "read.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (c.out / "read.txt").write_text("\n".join(lines) + "\n")
    return 0


# ------------------------------------------------------------------ B3's measurement

def b3_verdict(spans, gap) -> str:
    """B3's rule (docs/PAPER_B_PLAN.md, B3; ratified at the meeting of 11 Nov, item 56). spans: {ck: (span, lo, hi) or
    None}, the all-forced additive equal-compute span with its 95% paired group interval, at the rule's checkpoints.
    Measurable: the lower end >= 2x the seed gap at both. Not measurable: the span below 1x at either. Else marginal."""
    if not spans or any(v is None for v in spans.values()):
        return "incomplete"
    if all(lo >= 2 * gap for _, lo, _ in spans.values()):
        return "measurable"
    if any(s < gap for s, _, _ in spans.values()):
        return "not measurable"
    return "marginal"


def _ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def _est(p, draws):
    return [float(p)] + _ci(draws)


def _fmt(e, k=4):
    return f"{e[0]:+.{k}f} [{e[1]:+.{k}f}, {e[2]:+.{k}f}]"


def _tl_groups(t, gmap, G):
    """{T: (ce_g, n_g)}: one target-last unit's CE and token counts summed per spatial group, per target."""
    out = {}
    for T in MODS:
        g = np.array([gmap[int(r)] for r in t[f"{T}|rows"]])
        out[T] = (np.bincount(g, weights=t[f"{T}|ce_sum"], minlength=G),
                  np.bincount(g, weights=t[f"{T}|n"].astype(np.float64), minlength=G))
    return out


def _target_only(tls, gmap, G, W, w_grd):
    """Target scope at every depth. Per target (7): CE (D x 7) and its group draws (B x D x 7). Per slot (5), S1GRD and
    S1RTC pooled at their HELD-OUT token shares (w_grd), not at the target-last sample's rows (43% S1GRD), as B3's swap
    readout asks: CE (D x 5) and draws (B x D x 5), each a ratio of share-weighted group sums
    (b0/questions/q4_own_first_passes.py, which gave B3's A10cR numbers)."""
    per = [_tl_groups(t, gmap, G) for t in tls]
    ce_t = np.array([[p[T][0].sum() / p[T][1].sum() for T in MODS] for p in per])
    ceb_t = np.stack([np.stack([(W @ p[T][0]) / (W @ p[T][1]) for T in MODS], 1) for p in per], 1)
    nref = {T: per[0][T][1].sum() for T in IMG}
    a, b = w_grd / nref["S1GRD"], (1.0 - w_grd) / nref["S1RTC"]

    def slots(p, k):
        return np.stack([p["S2L2A"][k], a * p["S1GRD"][k] + b * p["S1RTC"][k], p["DEM"][k], p["NDVI"][k], p["LULC"][k]], 1)

    X = np.stack([slots(p, 0) for p in per])                      # D x G x 5
    N = np.stack([slots(p, 1) for p in per])
    ce_s = X.sum(1) / N.sum(1)
    ceb_s = np.einsum("bg,dgs->bds", W, X) / np.einsum("bg,dgs->bds", W, N)
    return ce_t, ceb_t, ce_s, ceb_s


def _b3(c, P, res, A, src, gmap, G, W, g_main, tvX, tv_lat, ctrls, routed, redraw) -> dict:
    D, mid, depths, tag = c.nr, c.mid, c.depths, c.tag
    gap = c.seed_gap
    out = {"seed_gap": gap, "rule": {}, "forced_curve": {}, "target_only": {}, "swap": {}, "controls": {}}
    cnt = A["cnt"]
    n_tok = cnt.sum(0)
    w_grd = float(n_tok[1] / (n_tok[1] + n_tok[2]))
    w_slot = slot(cnt).sum(0) / slot(cnt).sum()
    SN = [n for n, _ in SLOTS]
    P(f"\n== 5. B3's measurement (docs/PAPER_B_PLAN.md, B3). The rule, ratified 2026-11-11: measurable if the lower end of "
      f"the all-forced additive equal-compute span's 95% interval is at least 2x the seed gap at both checkpoints; not "
      f"measurable if the span is below 1x at either; else marginal")
    P(f"  seed gap {gap:.4f} nats{' (the default: A10c against A10c-s2, E1 at 3300, until a cell has two random-router seeds)' if gap == SEED_GAP else ''};"
      f" checkpoints {', '.join(map(str, c.table_cks))}; 5 slots (S1 pooled), mean depth the middle one")

    # -- the rule, for every random-router run read here
    rule_runs = [(c.label, tag, D, src)] + [(COMPARATORS[k]["label"], k, COMPARATORS[k]["nr"], "stored")
                                            for k in c.comparators if COMPARATORS[k]["router"] == "random"]
    for label, model, nr, s in rule_runs:
        sp = {}
        for ck in c.table_cks:
            st = res["spans"].get(f"{label} all forced|{ck}")
            sp[str(ck)] = None if st is None else (st["span"], st["span_ci"][0], st["span_ci"][1])
        v = b3_verdict(sp, gap)
        where = _where(c, model, nr, s)
        out["rule"][label] = dict(source=s, where=where, spans={k: (list(x) if x else None) for k, x in sp.items()},
                                  verdict=v, depths=nr,
                                  lattice_n=len(S.lattice(list(range(1, nr + 1)), 5, (nr + 1) // 2)))
        P(f"  {label} ({where}):")
        for ck, x in sp.items():
            if x is None:
                P(f"    {ck:>5s} no all-forced passes")
                continue
            P(f"    {ck:>5s} span {x[0]:.4f} [{x[1]:.4f}, {x[2]:.4f}]: span {x[0] / gap:.2f}x the seed gap, lower end "
              f"{x[1] / gap:.2f}x")
        P(f"    -> {v.upper()}")
        if model == tag and s == "stored" and len(tv_lat) == len(c.table_cks):
            spt = {str(ck): (tv_lat[ck]["span"], tv_lat[ck]["span_ci"][0], tv_lat[ck]["span_ci"][1]) for ck in c.table_cks}
            vt = b3_verdict(spt, gap)
            out["rule"][label]["titanv_crosscheck"] = dict(spans={k: list(x) for k, x in spt.items()}, verdict=vt)
            P(f"    cross-check on the TITAN V units {tag}_tvforce1-{D}: " + "; ".join(
                f"{ck} span {x[0]:.4f} [{x[1]:.4f}, {x[2]:.4f}]" for ck, x in spt.items()) + f" -> {vt.upper()}")
    learned = [COMPARATORS[k]["label"] for k in c.comparators if COMPARATORS[k]["router"] == "learned"]
    if learned:
        P(f"  ({', '.join(learned)}: learned routers, outside the rule)")
    for ck in c.table_cks:
        st = res["spans"].get(f"{c.label} all forced|{ck}")
        if st:
            P(f"  {c.label} all forced, {ck}: best table {st['best']} (uniform best: {st['uniform_is_best']}), best - uniform "
              f"{st['best_minus_uniform']:+.4f} [{st['bmu_ci'][0]:+.4f}, {st['bmu_ci'][1]:+.4f}], cross-validated gain "
              f"{st['cv_gain']:+.4f} +- {st['cv_sd']:.4f}; worst {st['worst']}")

    # -- the forced-depth curve: every position at d, main pass, pooled image-token CE
    P(f"  forced-depth curve: every position at d (main pass), pooled image-token CE; CE(d) - CE({mid}) and "
      f"CE(1) - CE({D}) with 95% paired group intervals")
    n_rows = cnt[:, :6].sum(1)

    def curve(name, Xs):
        """Xs: per depth, (rows, 7) CE sums."""
        x = [X[:, :6].sum(1) for X in Xs]
        ce = [float(xx.sum() / n_rows.sum()) for xx in x]
        diffs = {f"d{d}-d{mid}": _ratio_diff(n_rows, x[d - 1], x[mid - 1], g_main, W) for d in depths if d != mid}
        diffs[f"d1-d{D}"] = _ratio_diff(n_rows, x[0], x[D - 1], g_main, W)
        per_mod = {m: [float(X[:, j].sum() / cnt[:, j].sum()) for X in Xs] for j, m in enumerate(MODS)}
        P(f"    {name}: CE " + " ".join(f"d{d} {v:.4f}" for d, v in zip(depths, ce)) + "  " + "  ".join(
            f"{k} {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]" for k, v in diffs.items()))
        return dict(ce=ce, diffs={k: list(v) for k, v in diffs.items()}, per_modality=per_mod)

    for ck in c.table_cks:
        if all(f"{tag}|f{d}|{ck}" in A for d in depths):
            out["forced_curve"][f"{src}|{ck}"] = curve(f"{c.label} {ck} ({src})", [A[f"{tag}|f{d}|{ck}"] for d in depths])
        if src != "titanv" and ck in tvX:
            out["forced_curve"][f"titanv|{ck}"] = curve(f"{c.label} {ck} (titanv)", list(tvX[ck]))

    # -- the target-only sweep: per target and per slot, the passes either side of the middle depth
    P(f"  target only (context left alone): CE({mid - 1}) - CE({mid}) and CE({mid}) - CE({mid + 1}) per target and per slot "
      f"(S1 pooled by held-out token share, S1GRD {w_grd:.4f}), 95% group intervals; positive: the pass helps")
    have = [ck for ck in c.cks if all(_tl(c, f"{tag}_tforce{d}", ck) is not None for d in depths)]
    keep = {}
    for ck in have:
        ce_t, ceb_t, ce_s, ceb_s = _target_only([_tl(c, f"{tag}_tforce{d}", ck) for d in depths], gmap, G, W, w_grd)
        keep[ck] = (ce_t, ceb_t, ce_s, ceb_s)
        i, j = mid - 2, mid - 1                                    # CE(mid-1) - CE(mid), CE(mid) - CE(mid+1)
        e = {"ce": {m: ce_t[:, k].tolist() for k, m in enumerate(MODS)},
             "ce_slot": {s: ce_s[:, k].tolist() for k, s in enumerate(SN)},
             "first": {m: _est(ce_t[i, k] - ce_t[j, k], ceb_t[:, i, k] - ceb_t[:, j, k]) for k, m in enumerate(MODS)},
             "second": {m: _est(ce_t[j, k] - ce_t[j + 1, k], ceb_t[:, j, k] - ceb_t[:, j + 1, k]) for k, m in enumerate(MODS)},
             "first_slot": {s: _est(ce_s[i, k] - ce_s[j, k], ceb_s[:, i, k] - ceb_s[:, j, k]) for k, s in enumerate(SN)},
             "second_slot": {s: _est(ce_s[j, k] - ce_s[j + 1, k], ceb_s[:, j, k] - ceb_s[:, j + 1, k]) for k, s in enumerate(SN)}}
        out["target_only"][str(ck)] = e
        P(f"    {ck:5d} first  " + "  ".join(f"{m[:5]} {_fmt(e['first'][m])}" for m in IMG) + "  | slots "
          + "  ".join(f"{s} {_fmt(e['first_slot'][s])}" for s in SN))
        P(f"    {ck:5d} second " + "  ".join(f"{m[:5]} {_fmt(e['second'][m])}" for m in IMG) + "  | slots "
          + "  ".join(f"{s} {_fmt(e['second_slot'][s])}" for s in SN))
    late = [ck for ck in have if ck >= c.table_cks[0]]
    if len(late) >= 2:
        i, j = mid - 2, mid - 1

        def mean_est(f):
            pts = [f(*keep[ck])[0] for ck in late]
            drw = [f(*keep[ck])[1] for ck in late]
            return _est(np.mean(pts), np.mean(drw, 0))

        m1 = {m: mean_est(lambda ct, cbt, cs, cbs, k=k: (ct[i, k] - ct[j, k], cbt[:, i, k] - cbt[:, j, k])) for k, m in enumerate(MODS)}
        m2 = {m: mean_est(lambda ct, cbt, cs, cbs, k=k: (ct[j, k] - ct[j + 1, k], cbt[:, j, k] - cbt[:, j + 1, k])) for k, m in enumerate(MODS)}
        s1 = {s: mean_est(lambda ct, cbt, cs, cbs, k=k: (cs[i, k] - cs[j, k], cbs[:, i, k] - cbs[:, j, k])) for k, s in enumerate(SN)}
        s2 = {s: mean_est(lambda ct, cbt, cs, cbs, k=k: (cs[j, k] - cs[j + 1, k], cbs[:, j, k] - cbs[:, j + 1, k])) for k, s in enumerate(SN)}
        out["target_only"]["mean"] = dict(checkpoints=late, first=m1, second=m2, first_slot=s1, second_slot=s2)
        P(f"    mean of {', '.join(map(str, late))} (criterion 4's measure is its first pass; b0_criterion4.py freeze reads "
          f"it against A10cR seed 42):")
        P("      first  " + "  ".join(f"{m[:5]} {_fmt(m1[m])}" for m in IMG) + "  | slots "
          + "  ".join(f"{s} {_fmt(s1[s])}" for s in SN))
        P("      second " + "  ".join(f"{m[:5]} {_fmt(m2[m])}" for m in IMG) + "  | slots "
          + "  ".join(f"{s} {_fmt(s2[s])}" for s in SN))

    # -- the target-only lattice with S1 by held-out token share, and the swap readout
    lat = S.lattice(list(depths), 5, mid)
    idx = lat - 1
    u = int(np.where((lat == mid).all(1))[0][0])
    for ck in c.table_cks:
        if ck not in keep:
            continue
        _, _, ce_s, ceb_s = keep[ck]
        v = (ce_s[idx, np.arange(5)] * w_slot).sum(1)
        vb = (ceb_s[:, idx, np.arange(5)] * w_slot).sum(-1)
        best, worst = int(v.argmin()), int(v.argmax())
        span = _est(v[worst] - v[best], vb[:, worst] - vb[:, best])
        bmu = _est(v[best] - v[u], vb[:, best] - vb[:, u])
        g1, g2 = ce_s[mid - 2] - ce_s[mid - 1], ce_s[mid - 1] - ce_s[mid]
        g1b, g2b = ceb_s[:, mid - 2] - ceb_s[:, mid - 1], ceb_s[:, mid - 1] - ceb_s[:, mid]
        i1 = int(np.argmin(g1))
        j2 = int(np.argmax(np.where(np.arange(5) == i1, -np.inf, g2)))
        margin = _est(g1[i1] - g2[j2], g1b[:, i1] - g2b[:, j2])
        pooled = _est(w_slot[i1] * g1[i1] - w_slot[j2] * g2[j2], w_slot[i1] * g1b[:, i1] - w_slot[j2] * g2b[:, j2])
        freq = float((g1b.argmin(1) == i1).mean())
        out["swap"][str(ck)] = dict(
            smallest_first=SN[i1], first=_est(g1[i1], g1b[:, i1]), largest_second=SN[j2], second=_est(g2[j2], g2b[:, j2]),
            margin_slot=margin, margin_pooled=pooled, smallest_first_argmin_share=freq,
            uniform_beats_every_swap=bool(margin[1] > 0),
            target_only_share={"n": int(len(lat)), "span": span, "best": [int(x) for x in lat[best]],
                               "worst": [int(x) for x in lat[worst]], "best_minus_uniform": bmu,
                               "uniform_is_best": bool(best == u)})
        P(f"  swap readout {ck}: smallest first pass {SN[i1]} {_fmt(_est(g1[i1], g1b[:, i1]))} against the largest second "
          f"(other slot) {SN[j2]} {_fmt(_est(g2[j2], g2b[:, j2]))}: margin {_fmt(margin)} slot CE, {_fmt(pooled)} pooled; "
          f"{SN[i1]} smallest in {freq:.3f} of draws; uniform beats every single swap: "
          f"{'yes' if margin[1] > 0 else 'not shown' if margin[0] > 0 else 'no'}; a target below {mid} drops {SN[i1]} first")
        P(f"    target-only lattice, S1 by held-out share: {len(lat)} tables, span {span[0]:.4f} [{span[1]:.4f}, {span[2]:.4f}], "
          f"best {lat[best].tolist()} (uniform best: {best == u}), best - uniform {_fmt(bmu)}, worst {lat[worst].tolist()}")

    # -- Step 6's controls and the units' provenance
    for ck in c.table_cks:
        out["controls"][f"uniform_eq_tvforce{mid}|{ck}"] = ctrls.get(ck)
    P("  controls: the uniform table equals tvforce%d bit for bit: %s" % (
        mid, ", ".join(f"{ck} {ctrls.get(ck)}" for ck in c.table_cks)))
    ctx_ok, units = {}, {}
    for ck in c.cks:
        docs = []
        for d in depths:
            m = c.units / f"{tag}_tforce{d}" / f"checkpoint-{ck}" / "metrics.json"
            docs.append(json.loads(m.read_text()) if m.exists() else None)
        if any(x is None or "target_last" not in x for x in docs):
            continue
        ok = all(all(T in x["target_last"] and "mean_depth_context" in x["target_last"][T] for x in docs) for T in MODS)
        ok = ok and all(len({x["target_last"][T]["mean_depth_context"] for x in docs}) == 1 for T in MODS)
        ok = ok and all(x["target_last"][T].get("mean_depth_target") == d for x, d in zip(docs, depths) for T in MODS)
        ok = ok and all(x.get("force_depth") == d and x.get("force_scope") == "target" and x.get("gate_seed", 0) == 0
                        for x, d in zip(docs, depths))
        ctx_ok[str(ck)] = bool(ok)
    out["controls"]["context_left_alone"] = ctx_ok
    P("  controls: target-scope units, the context's mean depth identical across d and the target exactly at d: "
      + (", ".join(f"{ck} {v}" for ck, v in ctx_ok.items()) or "no units"))
    for unit in sorted(p.name for p in c.units.glob(f"{tag}_*") if p.is_dir()):
        for m in sorted((c.units / unit).glob("checkpoint-*/metrics.json")):
            x = json.loads(m.read_text())
            units.setdefault(unit, []).append(dict(ck=m.parent.name, gpu=_card(x.get("gpu")), precision=x.get("precision", "fp16"),
                                                   arm_config=x.get("arm_config"), token_set=x.get("token_set")))
    flat = [u for v in units.values() for u in v]
    prov = {k: sorted({str(u[k]) for u in flat}) for k in ("gpu", "precision", "arm_config", "token_set")}
    out["controls"]["provenance"] = dict(units=len(flat), **prov)
    P(f"  provenance: {len(flat)} unit-checkpoints under {c.units}: cards {prov['gpu']}, precision {prov['precision']}, "
      f"arm config {prov['arm_config']}, token set {prov['token_set']}"
      + ("" if all(len(v) <= 1 for v in prov.values()) else "  ⚠ NOT ONE SETTING"))
    out["routed_minus_mean_forced"] = routed
    out["gate_redraw"] = redraw
    return out


if __name__ == "__main__":
    raise SystemExit(main())
