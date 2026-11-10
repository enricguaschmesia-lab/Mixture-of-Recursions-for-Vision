#!/usr/bin/env python
"""B0 Step 8: the numbers and figures of the B0 results note, from the steps' stored readings and arrays.

  numbers   gathers the readings of Steps 1-7, the per-pass test and the supplementary checks
            (/data/enric/reports/b0/*/), and recomputes what no reading stores:
              - Step 1's states, tied pairs, deaths and revivals, precision and moves (states.json);
              - Step 2's rule and the LM force's preference between two depths in use (force units);
              - the balancing-off replay against its control, held out at step 1810 (eval_r4);
              - fixed depth 2 against A10cR as routed, and A10cR's all-forced depth curve;
              - the held-out minus training loss gap and the learned routers' held-out edge over the
                random one, from the stored training logs;
              - the best whole-modality load under the balancing term;
              - the plateau rank agreement on five slots (S1 pooled by tokens), with its two-sided p.
            Every recomputation that a stored reading also holds is checked against it, and the
            command fails if one differs. Writes /data/enric/reports/b0/numbers.json.
  figs      the note's four figures, from numbers.json and the stored readings, to
            /data/enric/reports/b0/figs/.

    .venv/bin/python eo/experiments/routing/b0_report.py numbers
    .venv/bin/python eo/experiments/routing/b0_report.py figs

Runs in `.venv`, CPU, in about a minute. Reads only; changes nothing it reads.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402

B0 = Path("/data/enric/reports/b0")
ARM = Path("/data/enric/reports/arm_eval")
RUNS = Path("/data/enric/runs/pretrain/phase3")
OUT = B0 / "numbers.json"
FIGS = B0 / "figs"
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
IMG = MODS[:6]
SLOTS = ["S2L2A", "S1", "DEM", "NDVI", "LULC"]
MOD_ID = {m: i + 1 for i, m in enumerate(MODS)}          # eo/data/eo_vocab.MODALITY_TO_ID
CKS = [330 * i for i in range(1, 11)]
RUN_DIRS = {"a10c": "mor10c_20261012_013533", "a10cs2": "mor10cs2_20261012_181812",
            "a10cr": "mor10cr_20261012_095834"}
S1GRD_SHARE = 0.0870                                      # S1GRD's share of held-out S1 tokens (Step 7)

CHECKS: list = []


def check(name, got, want, tol):
    ok = bool(abs(got - want) <= tol)
    CHECKS.append({"check": name, "got": float(got), "want": float(want), "tol": tol, "ok": ok})
    return ok


def _j(p):
    return json.loads(Path(p).read_text())


def _n(ck):
    return 0 if ck == "untrained" else int(str(ck).split("-")[1])


# ---------------------------------------------------------------- Step 1
def step1():
    D = _j(B0 / "states/states.json")
    main = {(c["tag"], _n(c["ck"]), c["mod"]): c for c in D["cells"] if c["pass_"] == "main"}
    tags = ("a10c", "a10cs2", "a10")

    def state(c):                                         # planning/s1_read.py, the rule of Step 1
        if c["gate"] >= .9:
            return "S"
        if c["margin_med"] <= .05:
            if sorted(c["mix"])[::-1][1] < .02:
                return "P"
            return {3: "T3", 2: "T2", 1: "t1"}[sum(c["alive"])]
        return "o"

    img = [(k, c) for k, c in main.items() if k[2] in IMG and k[1] > 0]
    states = Counter(state(c) for _, c in img)
    pairs = Counter()
    for _, c in img:
        if state(c) in ("T2", "T3"):
            o = tuple(sorted((np.argsort(c["mean_p"])[::-1][:2] + 1).tolist()))
            pairs["".join(map(str, o))] += 1
    deaths = revivals = 0
    for t in tags:
        for m in IMG:
            seq = [main[(t, k, m)]["alive"] for k in CKS]
            revivals += sum(1 for d in range(3) for a, b in zip(seq[:-1], seq[1:]) if not a[d] and b[d])
            deaths += sum(1 for d in range(3) for a, b in zip(seq[:-1], seq[1:]) if a[d] and not b[d])
    ratios = [main[(t, 2970, m)]["margin_med"] / main[(t, 3300, m)]["margin_med"]
              for t in tags for m in IMG
              if state(main[(t, 2970, m)]) in ("T2", "T3") and state(main[(t, 3300, m)]) in ("T2", "T3")]
    d32 = [abs(c["depth_fp32"] - c["mean_depth"]) for k, c in img if "depth_fp32" in c]
    d32_pre = [abs(c["depth_fp32"] - c["mean_depth"]) for k, c in img if "depth_fp32" in c and k[1] < 3300]
    b16 = [c["depth_bf16"] - c["mean_depth"] for k, c in img if "depth_bf16" in c]
    moves = {}
    for a, b in zip(CKS[:-1], CKS[1:]):
        moves[f"{a}-{b}"] = sum(abs(main[(t, b, m)]["mean_depth"] - main[(t, a, m)]["mean_depth"]) >= .5
                                for t in tags for m in IMG)
    shallow = {}
    for t in tags:
        hit = 0
        for k in CKS[:-1]:
            d = {m: main[(t, k, m)]["mean_depth"] for m in IMG}
            hit += set(sorted(d, key=d.get)[:2]) == {"S2L2A", "LULC"}
        shallow[t] = hit
    out = {
        "image_cells": len(img), "states": dict(states), "split_ties": states["T2"] + states["T3"],
        "tied_pairs": dict(pairs), "deaths": deaths, "revivals": revivals,
        "gap_ratio_2970_3300": {"median": float(np.median(ratios)), "max": float(np.max(ratios)), "n": len(ratios)},
        "fp32_minus_fp16": {"over_0.1": float(np.mean(np.array(d32) > .1)), "max": float(max(d32)),
                            "max_before_3300": float(max(d32_pre))},
        "bf16_minus_fp16": {"n": len(b16), "over_0.1": float(np.mean(np.abs(b16) > .1)),
                            "over_0.3": float(np.mean(np.abs(b16) > .3)), "max": float(np.max(np.abs(b16)))},
        "moves_ge_0.5_per_interval": moves,
        "s2l2a_lulc_two_shallowest_checkpoints_330_2970": shallow,
        "final_depth": {t: {m: main[(t, 3300, m)]["mean_depth"] for m in IMG} for t in tags},
    }
    check("step1 states T3", states["T3"], 71, 0)
    check("step1 states T2", states["T2"], 37, 0)
    check("step1 states P", states["P"], 34, 0)
    check("step1 states S", states["S"], 15, 0)
    check("step1 deaths", deaths, 6, 0)
    check("step1 revivals", revivals, 7, 0)
    check("step1 gap ratio median", out["gap_ratio_2970_3300"]["median"], 5.8, .05)
    return out


# ---------------------------------------------------------------- Step 2
def step2():
    D = _j(B0 / "states/states.json")
    main = {(c["tag"], _n(c["ck"]), c["mod"]): c for c in D["cells"] if c["pass_"] == "main"}

    def state(c):
        if c["gate"] >= .9:
            return "S"
        if c["margin_med"] <= .05:
            return "P" if sorted(c["mix"])[::-1][1] < .02 else "T"
        return "o"

    F = B0 / "forces_train"
    units = [_j(f) for f in sorted(F.glob("*/*.json")) if f.parent.name in ("a10", "a10c", "a10cs2")]
    rule = [u["mods"][m]["dL_dw_ci"][0] > 0 for u in units for m in IMG
            if m in u["mods"] and _n(u["ck"]) >= 990 and state(main[(u["tag"], _n(u["ck"]), m)]) == "T"]
    by_mod = {m: float(np.median([u["mods"][m]["dL_dw"] for u in units if m in u["mods"] and _n(u["ck"]) >= 990
                                  and state(main[(u["tag"], _n(u["ck"]), m)]) != "S"])) for m in IMG}
    # the LM force's preference between two depths in use (planning/s2_depth.py, stored units only)
    rows = []
    for u in units:
        z = np.load(F / u["tag"] / f"{u['ck']}.npz")
        G = S.eval_groups(z["rows"])
        U = np.unique(G, return_inverse=True)[1]
        W = S.boot_weights(U.max() + 1, 1000, 1)
        for j, m in enumerate(IMG):
            r = u["mods"].get(m)
            if r is None:
                continue
            ch = np.array(r["chosen"])
            idx = [d for d in range(3) if ch[d] >= .05]
            if len(idx) < 2:
                continue
            a, b = idx[0], idx[1]
            na, nb, ga, gb = z["chosen"][:, j, a], z["chosen"][:, j, b], z["gc_d"][:, j, a], z["gc_d"][:, j, b]
            e = ga.sum() / na.sum() - gb.sum() / nb.sum()
            Na, Nb, Ga, Gb = (S.sum_by(x, U) for x in (na, nb, ga, gb))
            diff = (W @ Ga) / (W @ Na) - (W @ Gb) / (W @ Nb)
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append((e, lo, hi, ga.sum() / na.sum()))
    d = np.array([r[0] for r in rows])
    ex = np.array([not (r[1] <= 0 <= r[2]) for r in rows])
    out = {
        "units": len(units), "rule_cells": len(rule), "rule_above_zero": int(sum(rule)),
        "rule_share": float(np.mean(rule)), "dLdw_median_by_modality_from_990": by_mod,
        "two_depth_cells": len(rows), "lm_force_differs": int(ex.sum()),
        "favours_shallower": int(sum(1 for r in rows if r[0] < 0 and not (r[1] <= 0 <= r[2]))),
        "median_abs_difference": float(np.median(np.abs(d))),
        "median_abs_force": float(np.median(np.abs([r[3] for r in rows]))),
    }
    check("step2 rule cells", out["rule_cells"], 50, 0)
    check("step2 rule above zero", out["rule_above_zero"], 4, 0)
    check("step2 two-depth cells", out["two_depth_cells"], 55, 0)
    check("step2 force differs", out["lm_force_differs"], 33, 0)
    check("step2 favours shallower", out["favours_shallower"], 26, 0)
    return out


# ---------------------------------------------------------------- Step 3
def step3():
    rep = {k: _j(B0 / f"replay/{k}.json") for k in ("h1p", "h3", "r4a", "r4b")}
    r2f = _j(B0 / "replay/b0r2f.json")
    h2 = {m: {"partial": v["partial"], "partial_null_95": v["partial_null_95"], "p": v["partial_p"]}
          for m, v in r2f["h2"].items()}
    fine = {m: {"move_median": v["move_median"], "common_share": v["common_share"]} for m, v in r2f["fine"].items()}
    load3 = [r["load"][2] for r in r2f["rows"]]
    # R4 against R2 held out at step 1810 (planning/s3/r4_eval.py: spatial groups, 2,000 draws, seed 0)
    R = B0 / "eval_r4"
    a = np.load(R / "b0r4/checkpoint-1810/tokens_router.npz")
    b = np.load(R / "b0r2/checkpoint-1810/tokens_router.npz")
    rows = np.load(R / "b0r4/checkpoint-1810/eval_rows.npy")
    assert np.array_equal(rows, np.load(R / "b0r2/checkpoint-1810/eval_rows.npy"))
    lm = a["label_mod"]
    assert np.array_equal(lm, b["label_mod"])
    body = (lm >= 1) & (lm <= 7)
    ca, cb = a["ce"].astype(np.float64), b["ce"].astype(np.float64)
    num = np.where(body, ca - cb, 0).sum(1)
    den = body.sum(1).astype(np.float64)
    e, lo, hi = S.ratio_boot(num, den, S.eval_groups(rows), n_draws=2000, seed=0)
    by_mod = {m: {"dce": float(ca[lm == i].mean() - cb[lm == i].mean()),
                  "depth_r2": float(b["depth"][lm == i].mean()), "depth_r4": float(a["depth"][lm == i].mean())}
              for m, i in MOD_ID.items()}
    out = {
        "h1p": rep["h1p"], "h3": rep["h3"], "r4a": rep["r4a"], "r4b": rep["r4b"],
        "r2f_h2": h2, "r2f_fine": fine, "r2f_states": len(r2f["rows"]),
        "r2f_load_depth3": {"min": float(min(load3)), "max": float(max(load3)), "mean": float(np.mean(load3))},
        "r4_vs_r2_1810": {"pooled": float(e), "ci95_groups": [float(lo), float(hi)],
                          "depth_r2": float(b["depth"][body].mean()), "depth_r4": float(a["depth"][body].mean()),
                          "by_modality": by_mod},
    }
    check("step3 R4-R2 pooled", e, -0.0002, 5e-5)
    check("step3 H1' median ratio", rep["h1p"]["median"], 12.87, .01)
    return out


# ---------------------------------------------------------------- Step 4
def step4():
    A = _j(B0 / "agree/agree.json")
    tm = {}
    for key, w in A["windows"].items():
        tag, prec, frame, win = key.split("|")
        if prec != "fp16" or frame != "main":
            continue
        tm.setdefault(tag, {})[win] = {m: {"depth": w[m]["depth"]["mean"], "half": w[m]["depth"]["half"],
                                           "gate": w[m]["gate"]["mean"], "mix": w[m]["mix"]} for m in MODS}
    agree = {k: v for k, v in A["agree"].items() if "|main|fp16|" in k}
    series = {k: {"rho": [x["rho"] for x in v["per"]], "p": [x["p"] for x in v["per"]],
                  **{kk: vv for kk, vv in v.items() if kk != "per"}} for k, v in A["series"].items()}
    # five slots, S1 pooled by tokens, plateau time means, two-sided exact p over 120 permutations
    five = {}
    for t in ("a10c", "a10cs2", "a10"):
        d = tm[t]["plateau"]
        five[t] = [d["S2L2A"]["depth"], S1GRD_SHARE * d["S1GRD"]["depth"] + (1 - S1GRD_SHARE) * d["S1RTC"]["depth"],
                   d["DEM"]["depth"], d["NDVI"]["depth"], d["LULC"]["depth"]]
    rho5 = S.spearman(five["a10c"], five["a10cs2"])
    p5 = S.exact_p(five["a10c"], five["a10cs2"])
    out = {"timemeans": tm, "agree": agree, "series": series, "stability": A["stability"],
           "landscape": A["landscape"], "rules": A["rules"], "controls": A["controls"],
           "plateau_five_slots": {"depth": five, "rho_a10c_a10cs2": float(rho5), "p_two_sided": float(p5)}}
    check("step4 plateau rho", A["agree"]["a10c~a10cs2|main|fp16|plateau"]["rho"], 0.6, .001)
    check("step4 five-slot rho", rho5, 0.7, .001)
    check("step4 five-slot p, two-sided (states/plateau.txt)", p5, 0.233, .001)
    return out


# ---------------------------------------------------------------- Step 5
def step5():
    R = _j(B0 / "tables/read.json")
    return {"read": R}


# ---------------------------------------------------------------- Step 6
def _img_ce(p):
    z = np.load(p)
    lm = z["label_mod"]
    m = (lm >= 1) & (lm <= 6)
    return float(z["ce"].astype(np.float64)[m].mean())


def step6():
    L = _j(B0 / "landscape/read.json")
    curve, fixed2 = {}, {}
    for ck in (1980, 3300):
        v = [_img_ce(B0 / f"landscape/a10cr_tvforce{d}/checkpoint-{ck}/tokens_router.npz") for d in (1, 2, 3)]
        curve[ck] = {"d1_minus_d2": v[0] - v[1], "d3_minus_d2": v[2] - v[1], "d1_minus_d3": v[0] - v[2]}
        routed = _img_ce(ARM / f"a10cr/checkpoint-{ck}/tokens_router.npz")
        fixed2[ck] = {"routed_A100": routed, "depth2_TITAN_V": v[1], "routed_minus_depth2": routed - v[1]}
    check("step6 fixed 2 vs routed 1980", fixed2[1980]["routed_minus_depth2"], 0.0044, 5e-5)
    check("step6 fixed 2 vs routed 3300", fixed2[3300]["routed_minus_depth2"], 0.0025, 5e-5)
    check("step6 span A10cR all forced 1980", L["spans"]["A10cR all forced|1980"]["span"], 0.0061, 5e-5)
    return {"read": L, "forced_curve_image_tokens": curve, "fixed2_vs_routed": fixed2}


# ---------------------------------------------------------------- the per-pass test
def perpass():
    P = B0 / "perpass"
    out = {k: _j(P / f"{k}.json") for k in ("read", "read_kl", "read_value", "read_value3")}
    r = out["read"]
    out["pooled_ev_over_rv"] = {ck: r[ck]["pooled"]["real"]["ev_over_rv"] for ck in ("1980", "3300")}
    check("perpass EV(d)/RV 1980", r["1980"]["pooled"]["real"]["ev_over_rv"]["d"], 0.0012, 1e-4)
    return out


# ---------------------------------------------------------------- Step 7
def step7():
    C = B0 / "criterion4"
    out = {k: _j(C / f"{k}.json") for k in ("read", "extra", "power", "early")}
    comp = out["read"]["companion"]
    check("step7 A10c plateau p", comp["a10c|plateau"]["p"], 0.767, .001)
    check("step7 A10c-s2 late p", comp["a10cs2|late"]["p"], 0.950, .001)
    return out


# ---------------------------------------------------------------- supplementary checks (2026-11-08)
def questions():
    def logs(tag):
        h = _j(RUNS / RUN_DIRS[tag] / "checkpoint-3300/trainer_state.json")["log_history"]
        tr = {e["step"]: e["loss"] for e in h if "loss" in e}
        ev = {e["step"]: e["eval_loss"] for e in h if "eval_loss" in e}
        ent = [e["balancing_entropy"] for e in h if "balancing_entropy" in e]
        return tr, ev, ent
    L = {t: logs(t) for t in RUN_DIRS}
    q1 = {}
    for s in CKS:
        row = {}
        for t in ("a10c", "a10cs2"):
            tr, ev, _ = L[t]
            w = [tr[k] for k in tr if s - 50 < k <= s]
            row[t] = {"heldout_minus_train": ev[s] - float(np.mean(w)), "heldout_minus_a10cr": ev[s] - L["a10cr"][1][s]}
        q1[s] = row
    ent = {t: float(np.median(L[t][2])) for t in ("a10c", "a10cs2")}
    share = {"pad+delim": .061, "S2L2A": .187, "S1": .187, "DEM": .187, "NDVI": .187, "LULC": .187, "Coords": .003}
    s = np.array(list(share.values()))
    s = s / s.sum()
    best = min((3 * (np.bincount(a, weights=s, minlength=3) ** 2).sum(), a)
               for a in itertools.product(range(3), repeat=len(s)) if a[0] == 2)
    load = np.sort(np.bincount(best[1], weights=s, minlength=3))[::-1]
    q3 = _j(B0 / "questions/q3_loss_grads.json")
    q3r = {u: {"router_bal_over_lm": v["groups"]["router"]["bal"]["norm"] / v["groups"]["router"]["lm"]["norm"],
               "router_cos_lm_bal": v["groups"]["router"]["cos_lm_bal"],
               "router_z_over_lm": v["groups"]["router"]["z"]["norm"] / v["groups"]["router"]["lm"]["norm"],
               "layer0_bal_over_lm": v["groups"]["layer0"]["bal"]["norm"] / v["groups"]["layer0"]["lm"]["norm"],
               "layer0_z_over_lm": v["groups"]["layer0"]["z"]["norm"] / v["groups"]["layer0"]["lm"]["norm"]}
           for u, v in q3["units"].items()}
    q4 = _j(B0 / "questions/q4_own_first_passes.json")
    check("q1 A10c-s2 edge at epoch 1", q1[330]["a10cs2"]["heldout_minus_a10cr"], -0.1098, 1e-4)
    check("q2 best balancing term", best[0], 1.0303, 1e-4)
    return {"q1_by_checkpoint": q1, "logged_balancing_entropy_median": ent,
            "q2_best_whole_modality": {"balancing": float(best[0]), "load_sorted": load.tolist()},
            "q3": q3r, "q4": q4}


def numbers() -> int:
    out = {"step1": step1(), "step2": step2(), "step3": step3(), "step4": step4(), "step5": step5(),
           "step6": step6(), "perpass": perpass(), "step7": step7(), "questions": questions()}
    out["checks"] = CHECKS
    OUT.write_text(json.dumps(out, indent=1, default=float))
    bad = [c for c in CHECKS if not c["ok"]]
    for c in CHECKS:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['got']:+.5f} (expected {c['want']:+.5f} +- {c['tol']})")
    print(f"wrote {OUT}; {len(CHECKS) - len(bad)} of {len(CHECKS)} checks pass")
    return 1 if bad else 0


# ---------------------------------------------------------------- figures
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RUN_STYLE = {"a10c": ("A10c (seed 42)", "#2a78d6"), "a10cs2": ("A10c-s2 (seed 43)", "#eb6834"),
             "a10": ("A10 (seed 42, original tokens)", "#1baf7a")}


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)


def figs() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelcolor": INK, "text.color": INK})
    N = _j(OUT)
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. time means per window: the seeds disagree, S2L2A and LULC sit shallow
    tm = N["step4"]["timemeans"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, win, title in zip(axes, ("plateau", "late"), ("Plateau, 990-2970 (7 checkpoints)",
                                                          "Late, 1980-3300 (5 checkpoints)")):
        for i, t in enumerate(("a10c", "a10cs2", "a10")):
            name, col = RUN_STYLE[t]
            x = np.arange(6) + (i - 1) * .22
            y = [tm[t][win][m]["depth"] for m in IMG]
            e = [tm[t][win][m]["half"] for m in IMG]
            ax.errorbar(x, y, yerr=e, fmt="o", ms=7, color=col, ecolor=col, elinewidth=1.5, capsize=0, label=name)
        ax.set_xticks(range(6), IMG)
        ax.set_ylim(.6, 3.4)
        ax.set_yticks([1, 1.5, 2, 2.5, 3])
        ax.set_title(title, loc="left", color=INK)
        _style(ax)
    axes[0].set_ylabel("time-mean depth (passes), 95% interval")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=10, loc="lower center", ncol=3)
    fig.suptitle("On time means the seeds still disagree; only S2L2A and LULC sit shallow in every run",
                 x=.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(FIGS / "b0_timemeans.png", dpi=150)
    plt.close(fig)
    made.append("b0_timemeans.png")

    # 2. R2f: one update at a time, the tied modalities and the load ring together
    r2f = _j(B0 / "replay/b0r2f.json")
    steps = np.array(r2f["steps"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    for m, col in (("S1RTC", "#2a78d6"), ("NDVI", "#eb6834")):
        axes[0].plot(steps, [r[m]["mix"][2] for r in r2f["rows"]], color=col, lw=1.6, label=m)
    axes[0].set_ylabel("share at depth 3")
    axes[0].set_ylim(-.03, 1.03)
    axes[0].legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=10)
    axes[1].plot(steps, [r["load"][2] for r in r2f["rows"]], color=INK2, lw=1.6)
    axes[1].axhline(1 / 3, color=INK2, lw=.8, ls="--")
    axes[1].text(1.005, 1 / 3, "equal\nload", transform=axes[1].get_yaxis_transform(), ha="left", va="center",
                 fontsize=9, color=INK2)
    axes[1].set_ylabel("load at depth 3\n(all positions)")
    axes[1].set_xlabel("training step (A10c-s2 replay R2f, saved at every update; 256 held-out probe rows)")
    for ax in axes:
        _style(ax)
    fig.suptitle("A tie is the balancing loop ringing: tied modalities and the load swing every 3-4 updates",
                 x=.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, .95, 1))
    fig.savefig(FIGS / "b0_ringing.png", dpi=150)
    plt.close(fig)
    made.append("b0_ringing.png")

    # 3. spans of equal-compute tables: random-depth backbones against the learned seeds
    sp = N["step6"]["read"]["spans"]
    rows = [("A10cR, target only", "A10cR target only"), ("A10cR, all forced", "A10cR all forced"),
            ("A10cR5 (5 recursions, 381 tables)", "A10cR5 all forced"), ("A10c-s2 (learned)", "A10c-s2 all forced"),
            ("A10c (learned)", "A10c all forced")]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for j, (ck, col, mk) in enumerate(((1980, "#2a78d6", "o"), (3300, "#eb6834", "s"))):
        for i, (lab, key) in enumerate(rows):
            r = sp[f"{key}|{ck}"]
            y = i + (j - .5) * .25
            ax.plot(r["span_ci"], [y, y], color=col, lw=2)
            ax.plot(r["span"], y, mk, color=col, ms=7, label=f"checkpoint {ck}" if i == 0 else None)
    ax.axvline(0.006, color=INK2, lw=1, ls="--")
    ax.text(0.0063, 3.5, "one seed gap (A10c vs A10c-s2, 0.006)", fontsize=9, color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.003, 0.2)
    ticks = [0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("span of the equal-compute per-modality tables, nats per image token (log scale)")
    ax.set_title("51 tables at mean depth 2 (A10cR5: 381 at 3), additive, inference only; bars are 95% paired group intervals",
                 loc="left", fontsize=9.5, color=INK2)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Which modality gets which depth is worth 0.004-0.008 nats on A10cR; the learned seeds' 0.09-0.14 is binding",
                 x=.01, ha="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIGS / "b0_spans.png", dpi=150)
    plt.close(fig)
    made.append("b0_spans.png")

    # 4. criterion 4: the gain on A10cR against what the learned routers do
    rd = N["step7"]["read"]
    rob = N["step7"]["extra"]["robust"]                   # five slots, S1 pooled by tokens, [value, lo, hi]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for s in SLOTS:
        f, flo, fhi = rob["first"][s]
        c, clo, chi = rob["second_cost"][s]
        ax.errorbar([f], [c], xerr=[[f - flo], [fhi - f]], yerr=[[c - clo], [chi - c]], fmt="o", ms=7,
                    color=INK2, ecolor="#9a9994", elinewidth=1.2)
        ax.annotate(s, (f, c), textcoords="offset points", xytext=(7, 5), fontsize=10, color=INK)
    ax.set_xlabel("first extra pass, CE(1) - CE(2), nats")
    ax.set_ylabel("cost of a second pass, CE(3) - CE(2), nats")
    ax.set_title("A10cR, target only, mean of 1980-3300", loc="left")
    _style(ax)
    ax = axes[1]
    comp = rd["companion"]
    for i, t in enumerate(("a10c", "a10cs2", "a10")):
        name, col = RUN_STYLE[t]
        c = comp[f"{t}|plateau"]
        x = np.arange(5) + (i - 1) * .22
        ax.plot(x, c["depth_run"], "o", ms=8, color=col, label=f"{name}: plateau time mean")
        ax.plot(x, c["depth_opt"], "o", ms=8, mfc="none", mec=col, mew=1.6)
    ax.plot([], [], "o", ms=8, mfc="none", mec=INK2, mew=1.6, label="gain-optimal at that run's own load")
    ax.set_xticks(range(5), SLOTS)
    ax.set_ylim(.8, 3.2)
    ax.set_yticks([1, 2, 3])
    ax.set_ylabel("depth (passes)")
    ax.set_title("Routers (filled) against the gain-optimal plan (open)", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2)
    _style(ax)
    fig.suptitle("The slots that gain most from a first pass lose most from a second, and every learned router keeps them shallow",
                 x=.01, ha="left", fontsize=12.5)
    fig.set_size_inches(11.5, 5.2)
    fig.tight_layout()
    fig.savefig(FIGS / "b0_criterion4.png", dpi=150)
    plt.close(fig)
    made.append("b0_criterion4.png")
    print("wrote", ", ".join(str(FIGS / m) for m in made))
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "numbers":
        return numbers()
    if cmd == "figs":
        return figs()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
