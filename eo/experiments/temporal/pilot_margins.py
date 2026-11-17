#!/usr/bin/env python
"""A1 Step 9: one pilot checkpoint's readings set against Step 9's margin table (`.venv`, CPU).

Gathers what the reading chain wrote:
- eval_temporal's report.json (<out-root>/<tag>/<checkpoint>/);
- forgetting_compare.json (<forgetting-root>/<tag>/<checkpoint>/);
- pilot_compare.json.

It also pairs T360-T's forecast and gap-fill units with T360's frozen baselines for those tasks
(eo/eval/temporal_layouts_v1.json, "baselines"). Those baselines were scored on the extended base without date
blocks, which reads as T360 there (Step 7). The pairing is per location, over Step 0's 5 km groups, as the report
pairs the left-out season with Step 0's bar. Then each margin is read as Step 9 writes it.

Quarter pass (step 333):
- forgetting: at most +0.02 overall and +0.05 on any modality;
- the two-date bars below T360's best on the same date pairs (6.430, 7.513, 7.271);
- the bars with and without date blocks, the date blocks' remaining cost (+0.42 / +0.20 / +0.20 at step 0).

Full pass (step 1,332), the same forgetting, plus:
- temporal gain: the left-out season below T360's bar, the paired interval of the gain above 0;
- far attention: dates 2 and 3 at or below date 1 on the full chronological profile, and dropping the oldest
  group raises the last date's CE (interval above 0), read only once the first half holds;
- lattice: a trained date block costs under 10% of the gain on the cross-date pair;
- more context: adding the other two modalities at the next date lowers CE (interval below 0);
- the date probe: the right date below the moved one (interval above 0) for S2L2A and NDVI in every season,
  beside its null (one bin);
- the tasks against T360's frozen baselines, for the group order ("if forecast and imputation split").

    python eo/experiments/temporal/pilot_margins.py --tag t360t_p1 --checkpoint checkpoint-333 --pass quarter

THE COORDS-FREE STAGE (docs/plans/a1_nc_stage.md, N4 and N5, 2026-11-17). A report of the nc layouts (its
`layouts` is nc1) is read by `margins_nc`: Step 9's margins, each read against the nc base's own reading
instead of T360's Step 0 numbers. Every bar, gain and task margin is the report's paired difference from the
baseline that the layout file's rule chose on the extended base without date blocks (the base without time),
over Step 0's 5 km groups. A baseline whose candidates were not all read is provisional, and its margins are
reported but given no verdict. The forgetting margin is unchanged (+0.02 overall, +0.05 on any modality),
against the nc base's own fp32 reading (forgetting_compare picks it by vocab). The full pass adds the kit
tasks, the subset histories the modality draw trains: each below its baseline, and its cost against the same
task with every modality.

    python eo/experiments/temporal/pilot_margins.py --tag <tag> --checkpoint checkpoint-2960 --pass full \\
        --out-root /data/enric/reports/temporal/nc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = Path("/data/enric/reports/temporal/step9")
FOUT = Path("/data/enric/reports/temporal/forgetting")
BASE_UNITS = Path("/data/enric/reports/temporal/step7/t360t_base/extended_base/units")
VAL = Path("/data/enric/data/TerraMesh/val_clean")
TG = ("S2L2A", "S1GRD", "NDVI")
BARS_T360 = {"S2L2A": 6.430, "S1GRD": 7.513, "NDVI": 7.271}           # Step 9's quarter-pass bars (Step 0)
BARS_STEP0_COST = {"S2L2A": 0.416, "S1GRD": 0.195, "NDVI": 0.198}     # Step 7: dates minus nodate on the base
LEFT_OUT_BAR = {"S2L2A": 6.221, "S1GRD": 7.169, "NDVI": 6.974}        # Step 9's full-pass bar (Step 0)
BASELINE_UNITS = {"forecast": "c.forecast.statics_same_mod", "gap_fill": "c.gap_fill.same_mod_s1"}


def fmt(ci, nd=3):
    if ci is None:
        return "n/a"
    return f"{ci['mean']:+.{nd}f} [{ci['lo']:+.{nd}f}, {ci['hi']:+.{nd}f}]"


def sign(ci):
    """'below' if the interval is below 0, 'above' if above, 'across' otherwise."""
    return "below" if ci["hi"] < 0 else ("above" if ci["lo"] > 0 else "across")


def task_vs_baselines(units: Path):
    from eo.data.temporal_index import load_temporal_index
    from eo.eval import temporal as ET
    _, groups_all, _ = ET.step0_locations(load_temporal_index(VAL))
    out = {}
    for task, cand in BASELINE_UNITS.items():
        for m in TG:
            c, b = units / f"t.{task}__{m}.npz", BASE_UNITS / f"{cand}__{m}__nodate.npz"
            if not (c.exists() and b.exists()):
                continue
            zc, zb = np.load(c), np.load(b)
            n = min(len(zc["ce"]), len(zb["ce"]))
            if not (np.array_equal(zc["locations"][:n], zb["locations"][:n]) and np.array_equal(zc["t"][:n], zb["t"][:n])):
                raise SystemExit(f"{c.name} and {b.name}: other locations or target dates")
            d = zc["ce"][:n] - zb["ce"][:n]
            out[f"{task}/{m}"] = {"n": int(n), "t360t": float(zc["ce"][:n].mean()), "t360_baseline": float(zb["ce"][:n].mean()),
                                  "baseline": cand, "minus_baseline": ET.group_ci(d, groups_all[:n])}
    return out


def _verdict(c, want="below"):
    """A margin cell's verdict: PASS/fail on its paired interval, or provisional when its baseline is incomplete."""
    mb = c.get("minus_baseline")
    if mb is None:
        return None, "no baseline"
    if not c.get("baseline", {}).get("complete", False):
        return None, "provisional (the base's candidates not all read)"
    s = sign(mb)
    return s == want, ("PASS" if s == want else "fail")


def margins_nc(rep, forg, which, lines, V):
    """An nc report's margins (module docstring, THE COORDS-FREE STAGE), into V (verdicts) and lines."""
    if forg:
        g = forg["groups"]
        worst = max((k for k in g if k != "all"), key=lambda k: g[k]["diff"])
        V["forgetting"] = {"all": g["all"]["diff"], "all_ci": g["all"]["ci95"], "worst_modality": worst,
                           "worst": g[worst]["diff"], "per_modality": {k: v["diff"] for k, v in g.items() if k != "all"},
                           "passed": forg["passed"]}
        lines.append(f"forgetting  all {g['all']['diff']:+.4f} [{g['all']['ci95'][0]:+.4f}, {g['all']['ci95'][1]:+.4f}] "
                     f"(margin +0.02); worst modality {worst} {g[worst]['diff']:+.4f} (margin +0.05): "
                     f"{'within' if forg['passed'] else 'OVER'}")
        lines.append("            " + "  ".join(f"{k} {v['diff']:+.4f}" for k, v in g.items() if k != "all"))
    else:
        lines.append("forgetting  not read yet")

    def row(label, c):
        ok, word = _verdict(c)
        b = c.get("baseline", {})
        cost = c.get("dates_minus_nodate")
        line = (f"{label} {c['ce']['mean']:.3f} vs the base's {b.get('layout', '?')} {b.get('ce', float('nan')):.3f}: "
                f"{fmt(c.get('minus_baseline'))} {word}")
        if "minus_step0" in c:
            line += f"; vs itself at step 0 {fmt(c['minus_step0'])}"
        if cost:
            line += f"; date blocks cost {fmt(cost)}"
        return ok, line, {"ce": c["ce"]["mean"], "baseline": b, "minus_baseline": c.get("minus_baseline"),
                          "minus_step0": c.get("minus_step0"), "dates_minus_nodate": cost, "passed": ok}

    bars = rep.get("bars_quarter_pass", {})
    if bars:
        V["bars"] = {}
        for m in TG:
            if m in bars:
                _, line, cell = row(f"bar {m:6s}", bars[m])
                V["bars"][m] = cell
                lines.append(line)
    kits = rep.get("kits", {})
    if kits:
        V["kits"] = {}
        for k, c in kits.items():
            _, line, cell = row(f"kit {k:42s}", c)
            cell["minus_every_modality"] = c.get("minus_every_modality")
            cell["minus_with_statics"] = c.get("minus_with_statics")
            V["kits"][k] = cell
            extra = (f"; vs every modality {fmt(c['minus_every_modality'])}" if c.get("minus_every_modality") else "")
            lines.append(line + extra)
    if which != "full":
        return
    tasks = rep.get("tasks", {})
    V["gain"], V["tasks_vs_base"] = {}, {}
    for k, c in tasks.items():
        ok, line, cell = row(f"task {k:15s}", c)
        (V["gain"] if k.startswith("left_out/") else V["tasks_vs_base"])[k] = cell
        lines.append(("gain " if k.startswith("left_out/") else "") + line)
    far = rep.get("far")
    if far:
        bb = far["by_block"]
        first = {m: {f"date{t}": bb[f"{m}@{t}"]["mean"] for t in range(4)} for m in TG}
        ok1 = {m: bool(first[m]["date2"] <= first[m]["date1"] and first[m]["date3"] <= first[m]["date1"]) for m in TG}
        drop = far.get("last_date_without_oldest_minus_with", {})
        ok2 = {m: sign(drop[m]) == "above" for m in drop}
        V["far"] = {"profile": first, "dates_2_3_at_or_below_1": ok1, "drop_oldest": drop, "drop_raises": ok2,
                    "passed": all(ok1.values()) and all(ok2.values())}
        for m in TG:
            p = first[m]
            lines.append(f"far {m:6s} dates 0-3 {p['date0']:.2f} {p['date1']:.2f} {p['date2']:.2f} {p['date3']:.2f}: "
                         f"2,3 <= 1 {ok1[m]}; without date 0 the last date {fmt(drop.get(m))} "
                         f"{'raised' if ok2.get(m) else 'not raised'}")
    lat = rep.get("lattice")
    if lat and "t.lattice" in lat:
        V["lattice"] = {k: lat[k]["share_of_gain"] for k in ("t.lattice", "t.lattice_each") if k in lat}
        V["lattice"]["passed"] = lat["t.lattice"]["share_of_gain"] < 0.10
        lines.append(f"lattice share of the gain {lat['t.lattice']['share_of_gain']:.1%} (a date before each block "
                     f"{lat.get('t.lattice_each', {}).get('share_of_gain', float('nan')):.1%}), gain {lat['gain']:.3f}: "
                     f"{'PASS' if V['lattice']['passed'] else 'fail'} (under 10%)")
    ctx = rep.get("more_context", {})
    if ctx:
        V["more_context"] = {m: {"b_minus_a": ctx[m]["b_minus_a"], "passed": sign(ctx[m]["b_minus_a"]) == "below"}
                             for m in TG if m in ctx}
        for m in TG:
            if m in ctx:
                lines.append(f"context {m:6s} b - a {fmt(ctx[m]['b_minus_a'])} "
                             f"{'PASS' if V['more_context'][m]['passed'] else 'fail'}")
    pr, nul = rep.get("probe"), rep.get("probe_null")
    if pr:
        V["probe"] = {}
        for m in TG:
            t = pr["targets"].get(m)
            if not t:
                continue
            q = {k: v["diff"] for k, v in t["by_quarter"].items()}
            every = all(sign(v) == "above" for v in q.values())
            nm = nul["targets"][m]["moved_minus_right"] if nul and m in nul["targets"] else None
            V["probe"][m] = {"all": t["moved_minus_right"], "by_quarter": q, "every_season_above": every, "null": nm}
            lines.append(f"probe {m:6s} moved - right {fmt(t['moved_minus_right'], 4)}; seasons "
                         + " ".join(f"{k} {v['mean']:+.4f}{'*' if sign(v) == 'above' else ''}" for k, v in q.items())
                         + f"; null {fmt(nm, 4)}" + (" PASS" if every and m != "S1GRD" else ""))
        V["probe"]["passed"] = all(V["probe"][m]["every_season_above"] for m in ("S2L2A", "NDVI") if m in V["probe"])
    dt = rep.get("date")
    if dt:
        V["date"] = {k: dt[k] for k in ("ce_y", "ce_d", "top1_year", "day_err_mode") if k in dt}
        lines.append(f"date task: year CE {dt['ce_y']['mean']:.3f}, day CE {dt['ce_d']['mean']:.3f}, top-1 year "
                     f"{dt['top1_year']['mean']:.3f} (chance 0.083), day error {dt['day_err_mode']['mean']:.1f} d (chance 91)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", required=True, help="the checkpoint directory's name, e.g. checkpoint-333")
    ap.add_argument("--pass", dest="which", choices=("quarter", "full"), required=True)
    ap.add_argument("--out-root", type=Path, default=OUT)
    ap.add_argument("--forgetting-root", type=Path, default=FOUT)
    ap.add_argument("--out", type=Path, default=None, help="default <out-root>/<tag>/<checkpoint>/margins.json")
    a = ap.parse_args()
    d = a.out_root / a.tag / a.checkpoint
    rep = json.loads((d / "report.json").read_text())
    fc = a.forgetting_root / a.tag / a.checkpoint / "forgetting_compare.json"
    forg = json.loads(fc.read_text()) if fc.exists() else None
    res = {"tag": a.tag, "checkpoint": a.checkpoint, "pass": a.which, "verdicts": {}}
    V = res["verdicts"]
    lines = [f"{a.tag} {a.checkpoint}, {a.which} pass"]
    if rep.get("layouts", "v1") != "v1":
        res["layouts"] = rep["layouts"]
        margins_nc(rep, forg, a.which, lines, V)
        out = a.out or (d / "margins.json")
        out.write_text(json.dumps(res, indent=1) + "\n")
        print("\n".join(lines))
        print(f"wrote {out}")
        return 0

    # ---- forgetting (both passes)
    if forg:
        worst = max((k for k in forg["groups"] if k != "all"), key=lambda k: forg["groups"][k]["diff"])
        g = forg["groups"]
        V["forgetting"] = {"all": g["all"]["diff"], "all_ci": g["all"]["ci95"], "worst_modality": worst,
                           "worst": g[worst]["diff"], "per_modality": {k: v["diff"] for k, v in g.items() if k != "all"},
                           "passed": forg["passed"]}
        lines.append(f"forgetting  all {g['all']['diff']:+.4f} [{g['all']['ci95'][0]:+.4f}, {g['all']['ci95'][1]:+.4f}] "
                     f"(margin +0.02); worst modality {worst} {g[worst]['diff']:+.4f} (margin +0.05): "
                     f"{'within' if forg['passed'] else 'OVER'}")
        lines.append("            " + "  ".join(f"{k} {v['diff']:+.4f}" for k, v in g.items() if k != "all"))
    else:
        lines.append("forgetting  not read yet")

    # ---- the two-date bars
    bars = rep.get("bars_quarter_pass", {})
    if bars:
        V["bars"] = {}
        for m in TG:
            b = bars.get(m)
            if not b:
                continue
            s = sign(b["minus_t360"])
            V["bars"][m] = {"ce": b["ce"]["mean"], "t360": b["t360"], "minus_t360": b["minus_t360"], "verdict": s,
                            "dates_minus_nodate": b.get("dates_minus_nodate")}
            cost = b.get("dates_minus_nodate")
            lines.append(f"bar {m:6s} {b['ce']['mean']:.3f} vs T360 {b['t360']:.3f}: {fmt(b['minus_t360'])} {s}"
                         + (f"; date blocks cost {fmt(cost)} (step 0 +{BARS_STEP0_COST[m]:.2f})" if cost else ""))

    if a.which == "full":
        # ---- temporal gain: the left-out season against T360's bar
        tasks = rep.get("tasks", {})
        V["gain"] = {}
        for m in TG:
            c = tasks.get(f"left_out/{m}")
            if c and "minus_t360_best_step0" in c:
                s = sign(c["minus_t360_best_step0"])
                V["gain"][m] = {"ce": c["ce"]["mean"], "bar": LEFT_OUT_BAR[m], "minus_bar": c["minus_t360_best_step0"],
                                "passed": s == "below"}
                lines.append(f"gain {m:6s} left-out {c['ce']['mean']:.3f} vs bar {LEFT_OUT_BAR[m]:.3f}: "
                             f"{fmt(c['minus_t360_best_step0'])} {'PASS' if s == 'below' else 'fail'}")
        # ---- far attention
        far = rep.get("far")
        if far:
            bb = far["by_block"]
            first = {m: {f"date{t}": bb[f"{m}@{t}"]["mean"] for t in range(4)} for m in TG}
            ok1 = {m: bool(first[m]["date2"] <= first[m]["date1"] and first[m]["date3"] <= first[m]["date1"]) for m in TG}
            drop = far.get("last_date_without_oldest_minus_with", {})
            ok2 = {m: sign(drop[m]) == "above" for m in drop}
            V["far"] = {"profile": first, "dates_2_3_at_or_below_1": ok1, "drop_oldest": drop, "drop_raises": ok2,
                        "passed": all(ok1.values()) and all(ok2.values())}
            for m in TG:
                p = first[m]
                lines.append(f"far {m:6s} dates 0-3 {p['date0']:.2f} {p['date1']:.2f} {p['date2']:.2f} {p['date3']:.2f}: "
                             f"2,3 <= 1 {ok1[m]}; without date 0 the last date {fmt(drop.get(m))} "
                             f"{'raised' if ok2.get(m) else 'not raised'}")
        # ---- lattice
        lat = rep.get("lattice")
        if lat:
            V["lattice"] = {k: lat[k]["share_of_gain"] for k in ("t.lattice", "t.lattice_each") if k in lat}
            V["lattice"]["passed"] = lat["t.lattice"]["share_of_gain"] < 0.10
            lines.append(f"lattice share of the gain {lat['t.lattice']['share_of_gain']:.1%} (a date before each block "
                         f"{lat.get('t.lattice_each', {}).get('share_of_gain', float('nan')):.1%}), gain {lat['gain']:.3f}: "
                         f"{'PASS' if V['lattice']['passed'] else 'fail'} (under 10%)")
        # ---- more context
        ctx = rep.get("more_context", {})
        if ctx:
            V["more_context"] = {m: {"b_minus_a": ctx[m]["b_minus_a"], "passed": sign(ctx[m]["b_minus_a"]) == "below"}
                                 for m in TG if m in ctx}
            for m in TG:
                if m in ctx:
                    lines.append(f"context {m:6s} b - a {fmt(ctx[m]['b_minus_a'])} "
                                 f"{'PASS' if V['more_context'][m]['passed'] else 'fail'}"
                                 + (f"; without dates {fmt(ctx[m]['nodate']['b_minus_a'])}" if "nodate" in ctx[m] else ""))
        # ---- the date probe
        pr, nul = rep.get("probe"), rep.get("probe_null")
        if pr:
            V["probe"] = {}
            for m in TG:
                t = pr["targets"].get(m)
                if not t:
                    continue
                q = {k: v["diff"] for k, v in t["by_quarter"].items()}
                every = all(sign(v) == "above" for v in q.values())
                nm = nul["targets"][m]["moved_minus_right"] if nul and m in nul["targets"] else None
                V["probe"][m] = {"all": t["moved_minus_right"], "by_quarter": q, "every_season_above": every, "null": nm}
                lines.append(f"probe {m:6s} moved - right {fmt(t['moved_minus_right'], 4)}; seasons "
                             + " ".join(f"{k} {v['mean']:+.4f}{'*' if sign(v) == 'above' else ''}" for k, v in q.items())
                             + f"; null {fmt(nm, 4)}" + (" PASS" if every and m != "S1GRD" else ""))
            V["probe"]["passed"] = all(V["probe"][m]["every_season_above"] for m in ("S2L2A", "NDVI") if m in V["probe"])
        # ---- the tasks against T360's frozen baselines (the group order)
        tv = task_vs_baselines(d / "units")
        if tv:
            V["tasks_vs_t360"] = tv
            for k, v in tv.items():
                lines.append(f"task {k:15s} {v['t360t']:.3f} vs T360 {v['t360_baseline']:.3f}: {fmt(v['minus_baseline'])}")
        lo = rep.get("tasks", {})
        for m in TG:
            c = lo.get(f"left_out/{m}")
            if c and "minus_t360_best_step0" in c:
                lines.append(f"task left_out/{m:6s} {c['ce']['mean']:.3f} vs T360 {LEFT_OUT_BAR[m]:.3f}: "
                             f"{fmt(c['minus_t360_best_step0'])}")
        dt = rep.get("date")
        if dt:
            V["date"] = {k: dt[k] for k in ("ce_y", "ce_d", "top1_year", "day_err_mode") if k in dt}
            lines.append(f"date task: year CE {dt['ce_y']['mean']:.3f}, day CE {dt['ce_d']['mean']:.3f}, top-1 year "
                         f"{dt['top1_year']['mean']:.3f} (chance 0.083), day error {dt['day_err_mode']['mean']:.1f} d (chance 91)")
    pc = d / "pilot_compare.json"
    if pc.exists():
        res["pilot_compare"] = str(pc)
    out = a.out or (d / "margins.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
