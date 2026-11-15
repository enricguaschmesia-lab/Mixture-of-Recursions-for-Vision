#!/usr/bin/env python
"""B3 (wave 1): criteria 2 and 4 on the learned runs of the train-split cells, read from their per-update depth shares.

The learned runs and the random-router runs whose passes give criterion 4's measure:
  A10c-b3-tr-NC, A10c-b3-tr-s2-NC   the Coords-free 1 + 3 + 1 train cell, where interventions 4 and 1 run (worklog §5,
                                    item 63); measure: A10cR-b3-tr-NC and A10cR-b3-tr-s2-NC, each and their mean
  A10c-tr                           the train-split control (1 + 9 + 1, Coords); measure: A10cR-tr

Criterion 2 (stable: "within ±0.3 passes from 60% of training through the anneal", direction note), as B0 Step 4 read
it and the meeting of 11 Nov ratified it (worklog §5, item 56): a modality is STABLE if the difference between its
1980-2640 and anneal (2970-3300) time means has a 95% interval inside ±0.3 passes, UNSTABLE if the interval lies beyond
±0.3, and UNDETERMINED otherwise; the run passes if every image modality is stable. Draws, fixed before this read
(2026-11-15): B0 took states 10 steps apart as near-independent draws; the per-update series here ring with periods up
to 21 updates (LULC on A10c-b3-tr-NC), so the draws are the means of consecutive 30-update blocks (10-update blocks are
printed beside, as a sensitivity), and the interval is Welch's. There is no group-bootstrap term: the shares are counted
on the training batch, a new batch at every update, so batch sampling is already in the draws' spread.

Criterion 4 (meaningful), as ratified (item 56; B0 Step 7): the regret of the run's time-mean allocation at its own load
against the gain-optimal allocation, tested against its 120 relabellings of five slots (S1 pooled by held-out token
share); the run follows the gain if p <= 0.05 in the plateau (990-2970) and late (1980-3300) windows. Two frames: TARGET
ONLY (primary): CE at depths 1-3 with only the target's depth forced, the mean of the last five checkpoints (1980-3300);
ALL FORCED (beside): every position at depth d, the main pass at 1980 and 3300. B0's `transport_min` and `regret_test`
(`b0_criterion4.py`, gated by its self-test) do the work, imported unchanged.
Beside them, the measure's reliability between the cell's two random seeds (B0's freeze rule: Spearman of the first
extra pass, CE(1) - CE(2), >= 0.8 on six labels and on five slots), and which slot has the smallest first pass.

CONTROLS, which must hold or the script writes nothing and exits 1:
  criterion 2  the random-router run (no table) reads every modality stable; the first learned run with a move of 0.5
               passes planted in its anneal window (each modality's own direction) reads every modality unstable, and
               with +0.1 planted none unstable
  criterion 4  the gain-optimal allocation at each run's own load passes (p <= 0.05); the run's allocation with its slots
               relabelled worst fails; the measure against itself reads Spearman 1, and with its labels shifted by one
               below 0.8 (as B0's freeze)

    .venv/bin/python eo/experiments/routing/b3_criteria.py

Writes /data/enric/reports/wave1/criteria/b3_criteria.{json,txt}. Runs in `.venv`, CPU, about a minute.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402
from eo.experiments.routing import b0_criterion4 as C4  # noqa: E402
from eo.experiments.routing import b3_depth_shares as DS  # noqa: E402
from eo.experiments.routing.b0_agree import check_t, t_quantile  # noqa: E402

RUNS_DIR = Path("/data/enric/runs/pretrain/phase3")
LAND = Path("/data/enric/reports/wave1/landscape")
OUT = Path("/data/enric/reports/wave1/criteria")
IMG = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
SLOTS = C4.SLOTS
SLOT_OF = C4.SLOT_OF
LAST5 = [1980, 2310, 2640, 2970, 3300]
AF_CKS = [1980, 3300]
WIN = {"early": (330, 990), "plateau": (990, 2970), "late": (1980, 3300), "anneal": (2970, 3300)}
C2_A, C2_B = (1980, 2640), (2970, 3300)
BLOCK, BLOCK_ALT = 30, 10
BAND = 0.3
RULE_WINDOWS = ("plateau", "late")

LEARNED = {   # tag: (run directory, the random-router landscape tags giving its measure)
    "a10cb3trnc": ("mor10cb3trnc_20261112_101703", ("a10crb3trnc", "a10crb3trs2nc")),
    "a10cb3trs2nc": ("mor10cb3trs2nc_20261113_072043", ("a10crb3trnc", "a10crb3trs2nc")),
    "a10ctr": ("mor10ctr_20261111_200306", ("a10crtr",)),
}
RANDOM_CONTROL = ("a10crb3trnc", "mor10crb3trnc_20261112_191925")


# ------------------------------------------------------------------ depth shares

def series(run: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """classes, steps (T,), per-update shares (T, class, depth) with NaN where a class had no position."""
    classes, steps, counts = DS.load(RUNS_DIR / run / "depth_shares.jsonl")
    tot = counts.sum(2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(tot > 0, counts / np.where(tot > 0, tot, 1), np.nan)
    return classes, steps, sh


def window_mix(classes, steps, sh, win) -> np.ndarray:
    """(6, 3): each image modality's mean share at each depth over the window's updates (B0's mix, from every update)."""
    a, b = WIN[win] if isinstance(win, str) else win
    sel = (steps >= a) & (steps <= b)
    return np.stack([np.nanmean(sh[sel, classes.index(m), :], 0) for m in IMG])


def mean_depth_series(classes, sh) -> np.ndarray:
    """(T, 6) per-update mean depth of each image modality."""
    d = np.arange(1, sh.shape[2] + 1)
    return np.stack([(sh[:, classes.index(m), :] * d).sum(1) for m in IMG], 1)


def blocks(x: np.ndarray, k: int) -> np.ndarray:
    x = x[~np.isnan(x)]
    n = len(x) // k
    return x[: n * k].reshape(n, k).mean(1)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """mean(a) - mean(b) and its 95% Welch interval."""
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = t_quantile(0.975, df)
    d = a.mean() - b.mean()
    return float(d), float(d - t * se), float(d + t * se)


def verdict(lo: float, hi: float) -> str:
    if -BAND < lo and hi < BAND:
        return "stable"
    if lo > BAND or hi < -BAND:
        return "unstable"
    return "undetermined"


def criterion2(steps, md, k=BLOCK, plant=None) -> dict:
    """plant: six values added to the anneal window's series (a control), or None."""
    sa = (steps >= C2_A[0]) & (steps <= C2_A[1])
    sb = (steps >= C2_B[0]) & (steps <= C2_B[1])
    plant = np.zeros(6) if plant is None else np.asarray(plant, dtype=np.float64)
    out = {}
    for j, m in enumerate(IMG):
        a, b = blocks(md[sa, j], k), blocks(md[sb, j] + plant[j], k)
        d, lo, hi = welch(a, b)
        out[m] = dict(diff=d, lo=lo, hi=hi, verdict=verdict(lo, hi), n_blocks=[len(a), len(b)],
                      mean_a=float(a.mean()), mean_b=float(b.mean()))
    out["passes"] = all(out[m]["verdict"] == "stable" for m in IMG)
    return out


# ------------------------------------------------------------------ criterion 4's measure

def tl_groups_img(z, gmap, G, rows_ref=None):
    """B0's tl_groups over the six image targets only (Coords-free units hold no Coords target)."""
    out = {}
    for T in IMG:
        r = z[f"{T}|rows"]
        if rows_ref is not None:
            assert np.array_equal(r, rows_ref[T]), T
        g = np.array([gmap[int(x)] for x in r])
        out[T] = (np.bincount(g, weights=z[f"{T}|ce_sum"], minlength=G),
                  np.bincount(g, weights=z[f"{T}|n"].astype(np.float64), minlength=G))
    return out


def target_only(tag: str, gmap, G, rows_ref=None):
    """{(depth, ck): [CE of each image target]} point estimates, and the target-last rows."""
    out, rows0 = {}, rows_ref
    for dep in (1, 2, 3):
        for ck in LAST5:
            z = C4.load_tl(f"{tag}_tforce{dep}", ck, LAND / tag)
            if z is None:
                raise SystemExit(f"{tag}: missing {tag}_tforce{dep} at checkpoint-{ck}")
            if rows0 is None:
                rows0 = {T: z[f"{T}|rows"] for T in IMG}
            grp = tl_groups_img(z, gmap, G, rows0)
            out[(dep, ck)] = [grp[T][0].sum() / grp[T][1].sum() for T in IMG]
    return out, rows0


def costs_target(TO, w_grd) -> np.ndarray:
    """(5, 3) slot CE at depths 1-3, the mean of the last five checkpoints."""
    return np.mean([np.array([C4.pool_s1(TO[(d, ck)], w_grd) for d in (1, 2, 3)]).T for ck in LAST5], 0)


def costs_forced(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """(5, 3) slot CE with every position at depth d (S1 pooled by summing its columns), mean of 1980 and 3300; cnt."""
    A = np.load(LAND / tag / "perrow.npz")
    cnt = A["cnt"]
    C = np.zeros((5, 3))
    for k, s in enumerate(SLOTS):
        idx = SLOT_OF[s]
        for d in (1, 2, 3):
            C[k, d - 1] = np.mean([A[f"{tag}|f{d}|{ck}"][:, idx].sum() / cnt[:, idx].sum() for ck in AF_CKS])
    return C, cnt


def first_pass(TO) -> list[float]:
    """CE(1) - CE(2) per image target, mean of the last five checkpoints."""
    return [float(np.mean([TO[(1, ck)][j] - TO[(2, ck)][j] for ck in LAST5])) for j in range(6)]


def depths5(mix5: np.ndarray) -> np.ndarray:
    return (mix5 * np.arange(1, 4)).sum(1)


def main() -> int:
    lines, res, fails = [], {"rules": {"criterion2": f"1980-2640 vs 2970-3300, {BLOCK}-update block means, Welch 95%, "
                                                       f"band ±{BAND}", "criterion4": "regret at own load vs 120 "
                                                       "relabellings, five slots, p <= 0.05 in plateau and late"}}, []

    def P(*a):
        t = " ".join(str(x) for x in a)
        lines.append(t)
        print(t, flush=True)

    check_t()   # B0's Student-t quantile against known values (raises on a wrong density)
    rows_main, g_main, gmap, G = C4.groups()
    # held-out token shares, from the cell's own all-forced arrays (the same 4,416 rows as B0's)
    _, cnt = costs_forced("a10crb3trnc")
    n_tok = cnt.sum(0)
    B0cnt = np.load(C4.LAND / "perrow.npz")["cnt"].sum(0)
    if not np.array_equal(n_tok[:6], B0cnt[:6]):
        fails.append("held-out image token counts differ from B0's")
    w_grd = float(n_tok[1] / (n_tok[1] + n_tok[2]))
    w_slot = np.array([n_tok[SLOT_OF[s]].sum() for s in SLOTS], dtype=np.float64)
    w_slot /= w_slot.sum()
    P(f"held-out image tokens {n_tok[:6].astype(int).tolist()} (B0's: equal); S1GRD share of S1 {w_grd:.4f}; "
      f"slot weights " + " ".join(f"{s} {w:.3f}" for s, w in zip(SLOTS, w_slot)))

    # ---------------------------------------------------------------- the measures
    P("\n== 1. Criterion 4's measure: first extra pass CE(1) - CE(2), target only, mean of 1980-3300")
    measures, rows_ref = {}, None
    for tag in ("a10crb3trnc", "a10crb3trs2nc", "a10crtr"):
        TO, rows_ref_t = target_only(tag, gmap, G, rows_ref if tag != "a10crtr" else None)
        if rows_ref is None:
            rows_ref = rows_ref_t
        Ct = costs_target(TO, w_grd)
        Cf, _ = costs_forced(tag)
        g1 = first_pass(TO)
        measures[tag] = dict(target=Ct, forced=Cf, g1=g1)
        g5 = C4.pool_s1(g1, w_grd)
        P(f"  {tag:14s} six " + " ".join(f"{T} {v:+.4f}" for T, v in zip(IMG, g1))
          + f" | smallest of five slots: {SLOTS[int(np.argmin(g5))]}")
    meanC = {f: (measures["a10crb3trnc"][f] + measures["a10crb3trs2nc"][f]) / 2 for f in ("target", "forced")}
    a, b = measures["a10crb3trnc"]["g1"], measures["a10crb3trs2nc"]["g1"]
    a5, b5 = C4.pool_s1(a, w_grd), C4.pool_s1(b, w_grd)
    r6, p6, r5, p5 = S.spearman(a, b), S.exact_p(a, b), S.spearman(a5, b5), S.exact_p(a5, b5)
    shifted = b[1:] + b[:1]
    r_self, r_shift = S.spearman(a, a), S.spearman(a, shifted)
    P(f"  between the cell's random seeds (B0's freeze rule, >= 0.8): six labels rho {r6:+.2f} [exact p {p6:.3f}] "
      f"({'reliable' if r6 >= 0.8 else 'not reliable'}), five slots {r5:+.2f} [p {p5:.3f}] "
      f"({'reliable' if r5 >= 0.8 else 'not reliable'})")
    P(f"  controls: seed 42 against itself rho {r_self:+.2f}; seed 43's labels shifted by one {r_shift:+.2f} "
      f"({'fails, as it must' if r_shift < 0.8 else 'PASSES: the control did not fire'})")
    if not (abs(r_self - 1) < 1e-12 and r_shift < 0.8):
        fails.append("measure controls")
    res["measure"] = dict(g1={k: v["g1"] for k, v in measures.items()}, rho6=r6, p6=p6, rho5=r5, p5=p5,
                          control_self=r_self, control_shift=r_shift)

    # ---------------------------------------------------------------- criterion 2
    P(f"\n== 2. Criterion 2: 1980-2640 against the anneal (2970-3300), {BLOCK}-update blocks, Welch 95%, band ±{BAND}")
    res["criterion2"] = {}
    for tag, (run, _) in LEARNED.items():
        classes, steps, sh = series(run)
        md = mean_depth_series(classes, sh)
        r = criterion2(steps, md)
        r_alt = criterion2(steps, md, k=BLOCK_ALT)
        res["criterion2"][tag] = dict(primary=r, blocks10=r_alt)
        P(f"  {tag}: {'PASSES' if r['passes'] else 'fails'}")
        for m in IMG:
            x, y = r[m], r_alt[m]
            P(f"    {m:6s} {x['mean_a']:.2f} -> {x['mean_b']:.2f}: {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] "
              f"{x['verdict']:12s} (blocks {x['n_blocks'][0]}/{x['n_blocks'][1]}; 10-update blocks "
              f"[{y['lo']:+.3f}, {y['hi']:+.3f}] {y['verdict']})")
    # controls
    tag0, (run0, _) = next(iter(LEARNED.items()))
    classes, steps, sh = series(run0)
    md0 = mean_depth_series(classes, sh)
    # The planted move goes the way each modality already moved, so that the planted difference is at least 0.5 (a
    # first version planted +0.5 in every anneal window: DEM's own +0.14 brought its difference to -0.36, whose interval
    # crosses the band, and it read undetermined, as it should; the control, not the check, was wrong; 2026-11-15).
    base = criterion2(steps, md0)
    sgn = np.array([1.0 if base[m]["diff"] >= 0 else -1.0 for m in IMG])
    up = criterion2(steps, md0, plant=-0.5 * sgn)
    small = criterion2(steps, md0, plant=np.full(6, 0.1))
    c_up = all(up[m]["verdict"] == "unstable" for m in IMG)
    c_small = not any(small[m]["verdict"] == "unstable" for m in IMG)
    P("  planted 0.5 (each modality's own direction): " + ", ".join(
        f"{m} {up[m]['diff']:+.2f} [{up[m]['lo']:+.2f}, {up[m]['hi']:+.2f}] {up[m]['verdict']}" for m in IMG))
    cr, sr, shr = series(RANDOM_CONTROL[1])
    rnd = criterion2(sr, mean_depth_series(cr, shr))
    c_rnd = rnd["passes"]
    P(f"  controls: +0.5 planted in {tag0}'s anneal reads every modality unstable = {c_up}; +0.1 planted reads none "
      f"unstable = {c_small}; the random router ({RANDOM_CONTROL[0]}) reads every modality stable = {c_rnd} ("
      + ", ".join(f"{m} {rnd[m]['diff']:+.3f}" for m in IMG) + ")")
    if not (c_up and c_small and c_rnd):
        fails.append("criterion 2 controls")
    res["criterion2_controls"] = dict(plant05=c_up, plant01=c_small, random=c_rnd)

    # ---------------------------------------------------------------- criterion 4
    P("\n== 3. Criterion 4: regret at the run's own load against its 120 relabellings (five slots), plateau and late")
    res["criterion4"] = {}
    for tag, (run, mtags) in LEARNED.items():
        classes, steps, sh = series(run)
        res["criterion4"][tag] = {}
        frames = []
        for mt in mtags:
            frames += [(f"{mt} target only", measures[mt]["target"]), (f"{mt} all forced", measures[mt]["forced"])]
        if len(mtags) == 2:
            frames += [("mean of both seeds, target only", meanC["target"]),
                       ("mean of both seeds, all forced", meanC["forced"])]
        for win in ("early", "plateau", "late", "anneal"):
            mix5 = np.stack(C4.pool_s1(list(window_mix(classes, steps, sh, win)), w_grd))
            P(f"  {tag} {win:7s} depths " + " ".join(f"{s} {d:.2f}" for s, d in zip(SLOTS, depths5(mix5))))
            for name, Cm in frames:
                t = C4.regret_test(mix5, w_slot, Cm)
                res["criterion4"][tag][f"{win}|{name}"] = dict(R=t["R"], p=t["p"], dstar=t["dstar"].tolist(),
                                                              depths=depths5(mix5).tolist())
                P(f"      {name:34s} regret {t['R']:.4f} p {t['p']:.3f}   optimum at this load "
                  + " ".join(f"{s} {d:.2f}" for s, d in zip(SLOTS, t["dstar"])))
        for name, _ in frames:
            ps = [res["criterion4"][tag][f"{w}|{name}"]["p"] for w in RULE_WINDOWS]
            v = "FOLLOWS the gain" if max(ps) <= 0.05 else "does not follow the gain"
            res["criterion4"][tag][f"verdict|{name}"] = v
            P(f"    {tag} under {name}: p (plateau, late) {ps[0]:.3f}, {ps[1]:.3f} -> {v}")
    # controls: the optimum at each run's plateau load passes; the worst relabelling of the run fails
    c4_ok = True
    for tag, (run, mtags) in LEARNED.items():
        classes, steps, sh = series(run)
        mix5 = np.stack(C4.pool_s1(list(window_mix(classes, steps, sh, "plateau")), w_grd))
        Cm = measures[mtags[0]]["target"]
        L = (w_slot[:, None] * mix5).sum(0)
        _, plan = C4.transport_min(w_slot, L, Cm)
        opt = np.clip(plan / w_slot[:, None], 0.0, None)
        p_opt = C4.regret_test(opt, w_slot, Cm)["p"]
        regs = [(C4.regret_test(mix5[list(pi)], w_slot, Cm)["R"], pi) for pi in itertools.permutations(range(5))]
        worst = max(regs)[1]
        p_worst = C4.regret_test(mix5[list(worst)], w_slot, Cm)["p"]
        ok = p_opt <= 0.05 and p_worst > 0.05
        c4_ok &= ok
        P(f"  control {tag}: the optimum at its plateau load p {p_opt:.3f} (passes), its worst relabelling p "
          f"{p_worst:.3f} (fails) = {ok}")
    if not c4_ok:
        fails.append("criterion 4 controls")

    if fails:
        print("CONTROLS FAILED: " + "; ".join(fails) + " -- nothing written")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b3_criteria.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "b3_criteria.txt").write_text("\n".join(lines) + "\n")
    P(f"\n-> {OUT / 'b3_criteria.txt'}, {OUT / 'b3_criteria.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
