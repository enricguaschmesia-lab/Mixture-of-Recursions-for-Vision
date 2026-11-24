#!/usr/bin/env python
"""B1, D990-s42 as description (2026-11-18): a body retrained at uniform depth by accident, read against the cell.

D990-s42 (arm B's soft compute target 1.8 with the confidence term 0.1, from a warm copy of the random-router run's
checkpoint-990, seed 42) failed as arm D. Its fresh router put every body position at depth 2 within four updates, then
swung whole between depth 2 and depth 1 (worklog 2026-11-18). So its checkpoints are uniform tables: 1650 at depth 2,
1980 at depth 1 and 3300 at depth 2. Uniform depth 1 in this cell is the 5-layer vanilla's architecture (1 + 3 x 1 + 1
layers); uniform depth 2 runs the shared block twice, at the random router's mean depth. This pairs each D unit, per
token, with the cell's units on the same 4,416 held-out rows, all eval_checkpoint.py's main pass on the TITAN V at fp16
with Coords dropped (measure_bd.sh d990s42; measure_d42_1650.sh):
  - the random router, routed, at seeds 42 (D's source) and 43; arm A at seed 42; arm C at the same switch and seed
    (C990-s42: the same warm copy under arm A's router and balancing); the 5-layer vanilla B10c-5L-tr-NC;
  - at 1980 and 3300, the seed-42 random backbone forced to uniform depth 2 at evaluation (B0 Step 6's tvforce2): fixed
    depth without retraining.
Measure: pooled image-token CE, D - other, paired per token, 95% row bootstrap (a0_read.paired, 2,000 draws); negative
means D is better. Per modality beside it. The yardstick is the seed gap at the same checkpoint (B3's rule), recomputed
here at all three checkpoints from the two random-router seeds.

D's depth history (the run's depth_shares.jsonl, every update from 991 on) is summarised beside it: the updates spent
with every body position at depth 1, at depth 2, and in between, and its time-mean depth, which sets its training
compute against the random router's 2.0.

This is description, not a criterion: D's depth history is not a design, its compute over 991-3300 differs from every
comparator, and it is one seed. It sizes the effect that a deliberate fixed-depth retrain (Step 9) would measure.

Checks before anything is read: every unit is the TITAN V fp16 protocol with Coords dropped on 4,416 clean rows
(b1_c_read.unit_ok); each pair holds the same rows, labels and modalities (a0_read.same_sequences); E1 recomputed from
the tokens equals the unit's metrics.json; D's mean body depth in the unit equals the uniform depth its run's record
holds at that update, where every body position sits at that one depth; the seed gap recomputed at 1980 and 3300
equals B3's stored one (b3_edge.py gap). CONTROLS, which must fail: B10 on the original tokens against each D unit,
and a D unit paired with itself must read 0 with a zero-width interval.

    .venv/bin/python eo/experiments/routing/b1_d_read.py

Writes /data/enric/reports/b1/d_read/read.{txt,json}. Runs in `.venv`, CPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

RUNS = Path("/data/enric/runs/pretrain/phase3")
OUT = Path("/data/enric/reports/b1/d_read")
D_TAG, D_RUN = "d990s42_e1tv", "mor10cwb3trncct18c990_20261118_010057"
CKS = ("checkpoint-1650", "checkpoint-1980", "checkpoint-3300")
OTHERS = {"random s42": "a10crb3trnc_e1tv", "random s43": "a10crb3trs2nc_e1tv",
          "learned s42 (A)": "a10cb3trnc_e1tv", "C990-s42": "c990s42_e1tv", "vanilla 5L": "b10c5trnc_e1tv"}
STORED_GAP = Path("/data/enric/reports/wave1/edge/gap_a10crb3trnc_e1tv_vs_a10crb3trs2nc_e1tv.json")
BODY = (1, 2, 3, 4, 5, 6)   # depth_shares classes S2L2A, S1GRD, S1RTC, DEM, NDVI, LULC (Coords, 7, is absent on nc)


def _load(name: str, rel: str):
    f = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def uniform_depth(row: dict) -> int | None:
    """The one depth every body position of this update sits at, or None."""
    at = {d for i in BODY for d in range(3) if row["counts"][i][d] > 0}
    return at.pop() + 1 if len(at) == 1 else None


def history(run: Path) -> tuple[dict, dict[int, dict]]:
    rows = [r for r in (json.loads(x) for x in (run / "depth_shares.jsonl").open()) if "step" in r]
    steps = [r["step"] for r in rows]
    if steps != list(range(steps[0], steps[0] + len(steps))):
        raise SystemExit(f"{run}: depth_shares.jsonl is not one row per update")
    n = {1: 0, 2: 0, None: 0}
    depth_sum = 0.0
    for r in rows:
        u = uniform_depth(r)
        n[u if u in (1, 2) else None] += 1
        c = r["counts"]
        tot = sum(sum(c[i]) for i in BODY)
        hard = sum(c[i][0] + 2 * c[i][1] + 3 * c[i][2] for i in BODY) / tot
        if abs(hard - r["ct"]["hard_mean"]) > 1e-9:
            raise SystemExit(f"step {r['step']}: body recount {hard} vs the trainer's hard_mean {r['ct']['hard_mean']}")
        depth_sum += hard
    h = {"updates": len(rows), "first": steps[0], "last": steps[-1], "all_depth_1": n[1], "all_depth_2": n[2],
         "mixed": n[None], "time_mean_depth": depth_sum / len(rows)}
    return h, {r["step"]: r for r in rows}


def main() -> int:
    c = _load("b1_c_read", "routing/b1_c_read.py")
    a0 = c._a0()
    img = [m for m in a0.MODALITIES if m != "Coords"]
    names = img + ["image", "all"]
    fmt = lambda p: f"{p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"  # noqa: E731
    AE = a0.AE
    missing = [f"{t}/{ck}" for ck in CKS for t in [D_TAG, *OTHERS.values()] if not (AE / t / ck / "metrics.json").exists()]
    if missing:
        raise SystemExit(f"not scored yet: {', '.join(missing)}")
    hist, by_step = history(RUNS / D_RUN)
    stored = json.loads(STORED_GAP.read_text())["by_ck"]
    doc = {"d_unit": D_TAG, "d_run": D_RUN, "others": OTHERS, "forced2": {k: str(v) for k, v in c.FORCED2.items()},
           "history": hist, "measure": f"pooled image-token CE, D - other, paired per token, 95% row bootstrap "
                                        f"({a0.NBOOT} draws)", "by_ck": {}}
    lines = ["B1 D990-s42 (description) against the Coords-free 1 + 3 + 1 train cell: pooled image-token CE, D - other, "
             f"paired per token, 95% row bootstrap ({a0.NBOOT}); TITAN V fp16, Coords dropped. Negative: D is better.",
             f"  D's depth over updates {hist['first']}-{hist['last']} ({hist['updates']}): every body position at depth "
             f"2 in {hist['all_depth_2']}, at depth 1 in {hist['all_depth_1']}, mixed in {hist['mixed']}; time-mean "
             f"depth {hist['time_mean_depth']:.3f} (the random router's 2.0)"]
    for ck in CKS:
        step = int(ck.split("-")[1])
        u = uniform_depth(by_step[step])
        d = AE / D_TAG / ck
        md = c.unit_ok(d)
        if u is None or abs(md["modes"]["router"]["mean_depth_body"] - u) > 1e-3:
            raise SystemExit(f"{ck}: the run's record holds depth {u} at update {step}, the unit's mean body depth is "
                             f"{md['modes']['router']['mean_depth_body']}")
        c_ok, c_why = a0.same_sequences(a0.CONTROL, d)
        if c_ok:
            raise SystemExit(f"CONTROL PASSED (B10, original tokens, vs {d}: {c_why}): the check is blind")
        R = a0.per_row(a0.tokens(d), names)
        z = a0.paired(R["image"], R["image"])
        if z["diff"] != 0 or z["lo"] != 0 or z["hi"] != 0:
            raise SystemExit(f"CHECK FAILED: D paired with itself reads {z}")
        e1 = a0.mean(R["all"])
        if abs(e1 - md["modes"]["router"]["ce_all_body_tokens"]) > 2e-3:
            raise SystemExit(f"{d}: E1 from tokens {e1:.6f} vs metrics.json {md['modes']['router']['ce_all_body_tokens']:.6f}")
        units = {k: AE / t / ck for k, t in OTHERS.items()}
        if ck != "checkpoint-1650":
            units |= {k: p / ck for k, p in c.FORCED2.items()}
        O, why0 = {}, None
        for k, p in units.items():
            c.unit_ok(p)
            ok, why = a0.same_sequences(d, p)
            if not ok:
                raise SystemExit(f"{d} vs {p}: {why}: not the same sequences")
            why0 = why
            O[k] = a0.per_row(a0.tokens(p), names)
        gap = abs(a0.paired(O["random s42"]["image"], O["random s43"]["image"])["diff"])
        if ck in stored and abs(gap - stored[ck]["seed_gap"]) > 1e-9:
            raise SystemExit(f"{ck}: seed gap {gap:.6f} vs B3's stored {stored[ck]['seed_gap']:.6f}")
        D = {"uniform_depth": u, "d_image_ce": a0.mean(R["image"]), "d_e1": e1, "seed_gap": gap, "pairing": why0,
             "gap_check": "equals B3's stored gap" if ck in stored else "not stored (computed here only)", "pairs": {}}
        lines.append(f"  {ck}: D at uniform depth {u}; image tokens {D['d_image_ce']:.4f} (E1 {e1:.4f}); pairing {why0}; "
                     f"control (B10, original tokens) fails as it must; self-pair 0 [0, 0]; seed gap {gap:.4f} "
                     f"({D['gap_check']})")
        for k in units:
            p = a0.paired(R["image"], O[k]["image"])
            pm = {n: a0.paired(R[n], O[k][n]) for n in img}
            beyond, clear = p["diff"] < -gap, p["hi"] < -gap
            worse, wclear = p["diff"] > gap, p["lo"] > gap
            D["pairs"][k] = {"other_image_ce": a0.mean(O[k]["image"]), "image": p, "per_modality": pm,
                             "x_seed_gap": -p["diff"] / gap, "lower_by_more_than_gap": bool(beyond),
                             "interval_clear_of_gap": bool(clear), "higher_by_more_than_gap": bool(worse),
                             "interval_clear_above_gap": bool(wclear)}
            tag = ("; lower beyond it" + (", interval clear" if clear else "") if beyond else
                   "; higher beyond it" + (", interval clear" if wclear else "") if worse else "")
            lines.append(f"      - {k:22s} {fmt(p)}  (other {a0.mean(O[k]['image']):.4f}; {-p['diff'] / gap:+.2f}x "
                         f"the gap{tag})")
        for k in ("random s42", "C990-s42", "vanilla 5L"):
            lines.append(f"        per modality, D - {k}: " + "; ".join(
                f"{n} {fmt(v)}" for n, v in D["pairs"][k]["per_modality"].items()))
        doc["by_ck"][ck] = D
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "read.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT / "read.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
