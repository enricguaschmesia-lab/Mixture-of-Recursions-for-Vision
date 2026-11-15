#!/usr/bin/env python
"""B3's train-split control, prediction 3: does the learned router's held-out edge over the random one survive?

The edge is B0's (`/data/enric/reports/b0/questions/q1_overfit.py`, 2026-11-08): the learned run's logged eval_loss
minus the random-router run's, at each eval step, on the same 4,416 held-out rows (both runs' in-training eval, fp16 on
an A100). On val it was -0.070 (A10c) and -0.110 (A10c-s2) at epoch 1 (step 330), about -0.01 by epoch 6 and about 0
at epochs 9-10. Each modality's logged eval loss (pure cross-entropy) is differenced too, so that a router loss inside
eval_loss, if any, cannot pass for an edge.

The check reproduces q1's two val numbers at step 330 before reading anything, and a run read against itself must
give 0 at every step (the control that the pairing can fail).

    .venv/bin/python eo/experiments/routing/b3_edge.py a10ctr=mor10ctr_... a10crtr=mor10crtr_...

`e1 <learned-tag> <random-tag>` reads the same edge as E1 (CE over every body token of the 4,416 held-out rows,
eval_checkpoint.py's routed main pass) at 1980 and 3300, paired per token with a bootstrap over rows (a0_read's
pairing), pooled and per modality, the logged evals' edge beside it. Both units must be on one card type at one
precision and must have scored the same sequences; CONTROL: B10 on the original tokens must fail that check, and a
run paired with itself must read 0 with a zero-width interval. Val's A10c and A10c-s2 against A10cR (A100 fp16, their
stored units) are read the same way, beside it.

    .venv/bin/python eo/experiments/routing/b3_edge.py e1 a10ctr_e1tv=mor10ctr_... a10crtr_e1tv=mor10crtr_...

(`=run` is needed when a unit's metrics.json holds no logged eval, as for runs pulled without their top-level
trainer_state.json: the logged eval at that step is then read from the run's checkpoints.)

`gap <seed-a-tag> <seed-b-tag>` gives a cell's own seed gap for B3's rule (Paper B plan, B3, ratified 2026-11-11):
"the difference in pooled image-token CE between two seeds of its random-router arm, at the same checkpoint", from
the routed main pass at 1980 and 3300 (measure_w1.sh / measure_fu.sh: `<run>_e1tv`, TITAN V fp16, both scored with
the cell's seed-42 config). The same checks as `e1`; E1's gap (Coords included, as the default 0.006) beside it.

    .venv/bin/python eo/experiments/routing/b3_edge.py gap a10cr_e1tv a10crs2_e1tv

`rule <seed-a-tag> <seed-b-tag> <landscape-dir>:<label> ...` applies B3's rule with that gap to each random-router
run's stored landscape reading (/data/enric/reports/wave1/landscape/<dir>/read.json, section 5), in two ways: (a) each
checkpoint's own gap, as "at the same checkpoint" reads, which is THE RULE'S READING (Enric, 2026-11-12, worklog §5 item
61); (b) the 3300 gap at both, as the default 0.006 is applied, printed beside it.
CONTROLS: the reading's stored gap must reproduce its stored verdict, and planted gaps (10x the larger span; a third of
the smaller lower end) must read not measurable and measurable.

    .venv/bin/python eo/experiments/routing/b3_edge.py rule a10cr_e1tv a10crs2_e1tv a10cr_owngap:A10cR a10crs2:a10crs2

Writes /data/enric/reports/wave1/edge/<learned>_vs_<random>[_e1].{json,txt}. Runs in `.venv`, CPU.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

RUNS = Path("/data/enric/runs/pretrain/phase3")
OUT = Path("/data/enric/reports/wave1/edge")
VAL = {"a10c": "mor10c_20261012_013533", "a10cs2": "mor10cs2_20261012_181812", "a10cr": "mor10cr_20261012_095834"}
Q1_330 = {"a10c": -0.070, "a10cs2": -0.110}
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]


def evals(run: str) -> dict[int, dict]:
    d = RUNS / run
    cks = sorted(d.glob("checkpoint-*/trainer_state.json"), key=lambda f: int(f.parent.name.split("-")[1]))
    f = d / "trainer_state.json" if (d / "trainer_state.json").exists() else cks[-1]
    out: dict[int, dict] = {}
    for e in json.loads(f.read_text())["log_history"]:
        if "eval_loss" in e:
            out.setdefault(e["step"], {})["all"] = e["eval_loss"]
        for m in MODS:
            if f"eval_loss_{m}" in e:
                out.setdefault(e["step"], {})[m] = e[f"eval_loss_{m}"]
    return out


def diff(a: dict, b: dict) -> dict[int, dict]:
    return {s: {k: a[s][k] - b[s][k] for k in a[s] if k in b[s]} for s in sorted(set(a) & set(b))}


def _a0():
    """a0_read's loaders and pairing (eo/experiments/base/a0_read.py; eo/experiments is not a package)."""
    f = Path(__file__).resolve().parents[1] / "base" / "a0_read.py"
    spec = importlib.util.spec_from_file_location("a0_read", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def e1(lt: str, rt: str, cks=("checkpoint-1980", "checkpoint-3300")) -> int:
    a0 = _a0()
    (lt, lrun), (rt, rrun) = ((x.split("=", 1) + [None])[:2] for x in (lt, rt))
    logged_runs = {lt: lrun, rt: rrun}
    names = list(a0.MODALITIES) + ["image", "all"]
    val_pairs = [("a10c", "a10cr"), ("a10cs2", "a10cr")]
    doc = {"learned": lt, "random": rt, "runs": logged_runs, "checkpoints": list(cks), "bootstrap": f"rows, {a0.NBOOT} draws", "by_ck": {}}
    lines = [f"{lt} - {rt}: E1 (every body token, 4,416 held-out rows, routed main pass), paired per token, 95% row "
             f"bootstrap ({a0.NBOOT}); the logged evals' edge beside it; val's A10c and A10c-s2 - A10cR the same way"]

    def logged(m, tag):
        if m.get("logged_eval"):
            return m["logged_eval"]["eval_loss"]
        run = logged_runs.get(tag)
        return evals(run)[m["step"]]["all"] if run else None

    def pair(l_dir, r_dir, ctl=True):
        ml, mr = (json.loads((p / "metrics.json").read_text()) for p in (l_dir, r_dir))
        for m, p in ((ml, l_dir), (mr, r_dir)):
            if m["token_set"] != "clean" or m["n_eval_rows"] != 4416:
                raise SystemExit(f"{p}: token set {m['token_set']}, {m['n_eval_rows']} rows")
        prec, gpus = {m.get("precision", "fp16") for m in (ml, mr)}, {m["gpu"] for m in (ml, mr)}
        if len(prec) != 1 or len(gpus) != 1:
            raise SystemExit(f"{l_dir} and {r_dir}: precisions {prec}, cards {gpus}: not one card at one precision")
        drops = [sorted(m.get("drop_modalities") or []) for m in (ml, mr)]
        if drops[0] != drops[1]:
            raise SystemExit(f"{l_dir} and {r_dir}: dropped modalities {drops[0]} and {drops[1]}: not one sequence set")
        ok, why = a0.same_sequences(l_dir, r_dir)
        if not ok:
            raise SystemExit(f"{l_dir} vs {r_dir}: {why}: not the same sequences")
        out = {"card": gpus.pop(), "precision": prec.pop(), "pairing": why, "drop_modalities": drops[0]}
        if ctl:
            c_ok, c_why = a0.same_sequences(a0.CONTROL, r_dir)
            if c_ok:
                raise SystemExit(f"CONTROL PASSED (B10 on the original tokens vs {r_dir}: {c_why}): the check is blind")
            out["control"] = f"B10 (original tokens) vs {r_dir.parent.name}: fails as it must ({c_why})"
        rl, rr = a0.per_row(a0.tokens(l_dir), names), a0.per_row(a0.tokens(r_dir), names)
        self_ = a0.paired(rr["all"], rr["all"])
        if self_["diff"] != 0 or self_["lo"] != 0 or self_["hi"] != 0:
            raise SystemExit(f"CHECK FAILED: {r_dir} paired with itself reads {self_}")
        e = {"l": a0.mean(rl["all"]), "r": a0.mean(rr["all"])}
        for k, m in (("l", ml), ("r", mr)):
            ref = m["modes"]["router"]["ce_all_body_tokens"]
            if abs(e[k] - ref) > 2e-3:
                raise SystemExit(f"E1 from tokens {e[k]:.6f} vs metrics.json {ref:.6f}")
        out |= {"e1": e, "edge": a0.paired(rl["all"], rr["all"]),
                "logged": {"l": logged(ml, l_dir.parent.name), "r": logged(mr, r_dir.parent.name)},
                "per_modality": {n: a0.paired(rl[n], rr[n]) for n in list(a0.MODALITIES) + ["image"]
                                 if n not in drops[0]}}   # a dropped modality has no tokens to pair
        lg = out["logged"]
        lg["edge"] = lg["l"] - lg["r"] if lg["l"] is not None and lg["r"] is not None else None
        return out

    fmt = lambda p: f"{p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"  # noqa: E731
    for ck in cks:
        D = {"train": pair(a0.AE / lt / ck, a0.AE / rt / ck)}
        for l, r in val_pairs:
            if (a0.AE / l / ck / "tokens_router.npz").exists():
                D[f"val {l}"] = pair(a0.AE / l / ck, a0.AE / r / ck, ctl=False)
        doc["by_ck"][ck] = D
        t = D["train"]
        dr = f"; {'+'.join(t['drop_modalities'])} dropped" if t["drop_modalities"] else ""
        lines += [f"  {ck}: {t['card']} {t['precision']}{dr}; pairing {t['pairing']}; {t['control']}; self-pair 0 [0, 0]"]
        for k, v in D.items():
            le = v["logged"]["edge"]
            lines.append(f"    {k:11s} E1 {v['e1']['l']:.4f} - {v['e1']['r']:.4f} = {fmt(v['edge'])}   logged "
                         f"{'-' if le is None else f'{le:+.4f}'}   ({v['card'].replace('NVIDIA ', '')})")
        lines.append("    per modality, " + lt + " - " + rt + ": " + "; ".join(
            f"{n} {fmt(p)}" for n, p in t["per_modality"].items()))
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{lt}_vs_{rt}_e1"
    (OUT / f"{stem}.json").write_text(json.dumps(doc, indent=1))
    (OUT / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def gap(at: str, bt: str, cks=("checkpoint-1980", "checkpoint-3300")) -> int:
    a0 = _a0()
    doc = {"a": at, "b": bt, "definition": "pooled image-token CE, a - b, routed main pass, at each checkpoint",
           "bootstrap": f"rows, {a0.NBOOT} draws", "by_ck": {}}
    lines = [f"B3's seed gap: {at} - {bt}, pooled image-token CE of the routed main pass at each checkpoint (the "
             f"rule's gap is its size), paired per token, 95% row bootstrap ({a0.NBOOT}); E1 (all body tokens) beside it"]
    for ck in cks:
        A, B = a0.AE / at / ck, a0.AE / bt / ck
        ma, mb = (json.loads((p / "metrics.json").read_text()) for p in (A, B))
        for m, p in ((ma, A), (mb, B)):
            if m["token_set"] != "clean" or m["n_eval_rows"] != 4416:
                raise SystemExit(f"{p}: token set {m['token_set']}, {m['n_eval_rows']} rows")
        prec, gpus = {m.get("precision", "fp16") for m in (ma, mb)}, {m["gpu"] for m in (ma, mb)}
        if len(prec) != 1 or len(gpus) != 1:
            raise SystemExit(f"{A} and {B}: precisions {prec}, cards {gpus}: not one card at one precision")
        if ma["arm_config"] != mb["arm_config"]:
            raise SystemExit(f"scored with {ma['arm_config']} and {mb['arm_config']}: a seed gap needs one config")
        drops = [sorted(m.get("drop_modalities") or []) for m in (ma, mb)]
        if drops[0] != drops[1]:
            raise SystemExit(f"{A} and {B}: dropped modalities {drops[0]} and {drops[1]}: not one sequence set")
        ok, why = a0.same_sequences(A, B)
        c_ok, c_why = a0.same_sequences(a0.CONTROL, B)
        if not ok or c_ok:
            raise SystemExit(f"pairing: {why}; control (B10, original tokens): {c_why}")
        ra, rb = (a0.per_row(a0.tokens(p), ["image", "all"]) for p in (A, B))
        z = a0.paired(rb["image"], rb["image"])
        if z["diff"] != 0 or z["lo"] != 0 or z["hi"] != 0:
            raise SystemExit(f"CHECK FAILED: {B} paired with itself reads {z}")
        img, e1 = a0.paired(ra["image"], rb["image"]), a0.paired(ra["all"], rb["all"])
        d = {"card": gpus.pop(), "precision": prec.pop(), "config": ma["arm_config"], "pairing": why,
             "control": c_why, "image": {"a": a0.mean(ra["image"]), "b": a0.mean(rb["image"]), **img},
             "e1": {"a": a0.mean(ra["all"]), "b": a0.mean(rb["all"]), **e1}, "seed_gap": abs(img["diff"]),
             "drop_modalities": drops[0]}
        doc["by_ck"][ck] = d
        dr = f", {'+'.join(drops[0])} dropped" if drops[0] else ""
        lines += [f"  {ck}: {d['card']} {d['precision']}, {d['config']}{dr}; pairing {why}; control fails as it must "
                  f"({c_why}); self-pair 0",
                  f"    image tokens {d['image']['a']:.4f} - {d['image']['b']:.4f} = {img['diff']:+.4f} "
                  f"[{img['lo']:+.4f}, {img['hi']:+.4f}]  -> seed gap {d['seed_gap']:.4f}",
                  f"    E1           {d['e1']['a']:.4f} - {d['e1']['b']:.4f} = {e1['diff']:+.4f} "
                  f"[{e1['lo']:+.4f}, {e1['hi']:+.4f}]"]
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"gap_{at}_vs_{bt}"
    (OUT / f"{stem}.json").write_text(json.dumps(doc, indent=1))
    (OUT / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def rule(at: str, bt: str, reads: list[str]) -> int:
    """B3's rule with the cell's own seed gap (gap_<a>_vs_<b>.json, from `gap`), on each random-router run's stored
    landscape reading (`b0_landscape.py read`'s read.json, section 5's all-forced spans with their 95% intervals)."""
    spec = importlib.util.spec_from_file_location("b0_landscape", Path(__file__).with_name("b0_landscape.py"))
    bl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bl)
    g = json.loads((OUT / f"gap_{at}_vs_{bt}.json").read_text())
    own = {str(int(ck.split("-")[1])): d["seed_gap"] for ck, d in g["by_ck"].items()}

    def by_ck(spans, gaps):
        """The rule with a gap per checkpoint: measurable if every lower end >= 2x that checkpoint's gap; not
        measurable if any span < 1x its checkpoint's gap; else marginal. With one gap at both it is b3_verdict."""
        if any(v is None for v in spans.values()):
            return "incomplete"
        if all(lo >= 2 * gaps[ck] for ck, (_, lo, _) in spans.items()):
            return "measurable"
        if any(s < gaps[ck] for ck, (s, _, _) in spans.items()):
            return "not measurable"
        return "marginal"

    doc = {"gap": {"a": at, "b": bt, "by_ck": own}, "runs": {}}
    lines = [f"B3's rule with the cell's own seed gap ({at} - {bt}, pooled image-token CE): "
             + ", ".join(f"{ck} {x:.4f}" for ck, x in own.items()),
             "  (a) each checkpoint's own gap, the rule's \"at the same checkpoint\" and its reading (worklog §5, item 61, "
             "2026-11-12); (b) the 3300 gap at both checkpoints, as the default 0.006 is applied, beside it"]
    for r in reads:
        tag, label = r.split(":", 1)
        rd = json.loads((Path("/data/enric/reports/wave1/landscape") / tag / "read.json").read_text())["b3"]
        ru = rd["rule"][label]
        spans = {ck: tuple(x) for ck, x in ru["spans"].items()}
        if set(spans) != set(own):
            raise SystemExit(f"{tag}:{label}: checkpoints {sorted(spans)}, gap at {sorted(own)}")
        # CONTROLS: the reading's own gap at both checkpoints reproduces its stored verdict, through both functions;
        # a gap 10x the larger span must read not measurable, and one a third of the smaller lower end measurable
        stored = {ck: rd["seed_gap"] for ck in spans}
        for v in (by_ck(spans, stored), bl.b3_verdict(spans, rd["seed_gap"])):
            if v != ru["verdict"]:
                raise SystemExit(f"CHECK FAILED: {tag}:{label} at its stored gap reads {v}, stored {ru['verdict']}")
        big = {ck: 10 * max(s for s, _, _ in spans.values()) for ck in spans}
        small = {ck: min(lo for _, lo, _ in spans.values()) / 3 for ck in spans}
        if by_ck(spans, big) != "not measurable" or by_ck(spans, small) != "measurable":
            raise SystemExit(f"CHECK FAILED: {tag}:{label}: the planted gaps do not move the verdict")
        va = by_ck(spans, own)
        vb = by_ck(spans, {ck: own["3300"] for ck in spans})
        doc["runs"][f"{tag}:{label}"] = {"spans": {ck: list(x) for ck, x in spans.items()}, "where": ru.get("where"),
                                         "stored_gap": rd["seed_gap"], "stored_verdict": ru["verdict"],
                                         "own_by_ck": va, "own_3300_at_both": vb}
        lines.append(f"  {label} ({tag}; {ru.get('where')}); stored at gap {rd['seed_gap']:.4f}: "
                     f"{ru['verdict']} (reproduced; planted gaps move it)")
        for ck, (s, lo, hi) in spans.items():
            lines.append(f"    {ck:>5s} span {s:.4f} [{lo:.4f}, {hi:.4f}]: (a) gap {own[ck]:.4f}, span {s / own[ck]:.2f}x, "
                         f"lower end {lo / own[ck]:.2f}x; (b) gap {own['3300']:.4f}, span {s / own['3300']:.2f}x, "
                         f"lower end {lo / own['3300']:.2f}x")
        lines.append(f"    -> {va.upper()} (the rule, (a)); (b) {vb}" + ("" if va == vb else "  (the two differ)"))
    stem = f"rule_{at}_vs_{bt}"
    (OUT / f"{stem}.json").write_text(json.dumps(doc, indent=1))
    (OUT / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    if len(sys.argv) >= 5 and sys.argv[1] == "rule":
        return rule(sys.argv[2], sys.argv[3], sys.argv[4:])
    if len(sys.argv) == 4 and sys.argv[1] == "gap":
        return gap(sys.argv[2], sys.argv[3])
    if len(sys.argv) == 4 and sys.argv[1] == "e1":
        return e1(sys.argv[2], sys.argv[3])
    if len(sys.argv) != 3 or any("=" not in x for x in sys.argv[1:]):
        print(__doc__)
        return 2
    (lt, lr), (rt, rr) = (x.split("=", 1) for x in sys.argv[1:])
    v = {k: evals(r) for k, r in VAL.items()}
    for k, want in Q1_330.items():
        got = v[k][330]["all"] - v["a10cr"][330]["all"]
        if round(got, 3) != want:
            print(f"CHECK FAILED: {k} - a10cr at 330 reads {got:+.4f}, q1 read {want:+.3f}")
            return 1
    L, R = evals(lr), evals(rr)
    if any(abs(x) > 0 for d in diff(L, L).values() for x in d.values()):
        print("CHECK FAILED: a run against itself is not 0")
        return 1
    d = diff(L, R)
    if not d:
        print("no common eval steps")
        return 1
    dv = {k: diff(v[k], v["a10cr"]) for k in Q1_330}
    lines = [f"{lt} - {rt}: logged eval_loss at each common eval step (held-out, 4,416 rows); val's A10c and A10c-s2 "
             f"- A10cR beside it at the same step (check: q1's -0.070 and -0.110 at 330 reproduced; self-pair 0)",
             f"  {'step':>5} {lt + ' - ' + rt:>17} {'val a10c':>9} {'val s2':>8} | per modality ({' '.join(MODS)})"]
    for s, x in d.items():
        vv = " ".join(f"{dv[k][s]['all']:+9.4f}" if s in dv[k] else f"{'':>9}" for k in Q1_330)
        lines.append(f"  {s:5d} {x['all']:+17.4f} {vv} | " + " ".join(f"{x[m]:+.3f}" if m in x else "  -   " for m in MODS))
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{lt}_vs_{rt}"
    (OUT / f"{stem}.json").write_text(json.dumps({"learned": [lt, lr], "random": [rt, rr], "diff": d,
                                                  "val": {k: dv[k] for k in dv}}, indent=1))
    (OUT / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
