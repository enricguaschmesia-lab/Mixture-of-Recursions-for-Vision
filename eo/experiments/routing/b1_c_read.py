#!/usr/bin/env python
"""B1, arm C (intervention 4, the warm start): each C run's E1 read against the Coords-free 1 + 3 + 1 train cell.

Arm C continues a random-router checkpoint (990 or 1980) exactly under arm A's learned router with balancing 0.1
(eo/train/warm_start.py). This pairs every C unit, per token, with the cell's units on the same 4,416 held-out rows,
all eval_checkpoint.py's main pass on the TITAN V at fp16 with Coords dropped (measure_c.sh; the cell's measure_nc.sh):
  - the random router, routed: a10crb3trnc_e1tv (seed 42, C's source at seed 42), a10crb3trs2nc_e1tv (seed 43), and
    the mean of the two seeds' per-row sums;
  - the learned router from step 0 (arm A): a10cb3trnc_e1tv, a10cb3trs2nc_e1tv;
  - the 5-layer vanilla B10c-5L-tr-NC (b10c5trnc_e1tv), C3's matched vanilla;
  - the random backbones forced to uniform depth 2 at evaluation (B0 Step 6's tvforce2 units), each seed and their
    mean: fixed depth as B3 measured it, not a model retrained at depth 2;
  - the other C units, which give the arm's own seed gap.
Measure: pooled image-token CE, C - other, paired per token, 95% row bootstrap (a0_read.paired, 2,000 draws); negative
means C is better. Per modality beside it. The yardstick is the cell's seed gap (0.0103 at 1980, 0.0069 at 3300; B3's
rule); each pair also says whether C is lower by more than it, on the point and on the interval's upper end.

C3 (decision 13: beat the permuted tables, the other seed's table, a histogram-matched random router, fixed depth and
the matched vanilla, each by more than the seed gap, after retraining at equal compute) is NOT read here in full: the
tables need retrains (open decision 2), the random router is not histogram-matched (C's body sits at mean depth about
1.9, the random router's at 2.0: C is the cheaper), and fixed depth is forced at evaluation only. This is description.

Checks before anything is read: every unit is the TITAN V fp16 protocol with Coords dropped on 4,416 clean rows; each
pair holds the same rows, labels and modalities (a0_read.same_sequences); E1 recomputed from the tokens equals the
unit's metrics.json. CONTROLS, which must fail: B10 on the original tokens against every C unit (its S2L2A/NDVI labels
differ), and a C unit paired with itself must read 0 with a zero-width interval.

    .venv/bin/python eo/experiments/routing/b1_c_read.py            (every C unit scored at both checkpoints)
    .venv/bin/python eo/experiments/routing/b1_c_read.py B          (arm B, 2026-11-18: its two seeds, D990-s42 at uniform 2
                                                                     beside them at 3300 as description; written to b1/b_read)

Writes /data/enric/reports/b1/c_read/read.{txt,json}. Runs in `.venv`, CPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

AE = Path("/data/enric/reports/arm_eval")
LS = Path("/data/enric/reports/wave1/landscape")
OUT = Path("/data/enric/reports/b1/c_read")
CKS = ("checkpoint-1980", "checkpoint-3300")
C_UNITS = {"C990-s42": "c990s42_e1tv", "C990-s43": "c990s43_e1tv",
           "C1980-s42": "c1980s42_e1tv", "C1980-s43": "c1980s43_e1tv"}
# B1 Step 7b (2026-11-18): arm B read the same way (`b1_c_read.py B`, written to b1/b_read). Beside its units at 3300,
# D990-s42, a body retrained at uniform depth 2 by accident (b1_d_read.py), as description: B runs mean depth 1.8
ARM_UNITS = {"C": C_UNITS, "B": {"B-s42": "b42_e1tv", "B-s43": "b43_e1tv"}}
ARM_OUT = {"C": Path("/data/enric/reports/b1/c_read"), "B": Path("/data/enric/reports/b1/b_read")}
EXTRA_3300 = {"B": {"D990-s42 at uniform 2 (description)": "d990s42_e1tv"}}
ROUTED = {"random s42": "a10crb3trnc_e1tv", "random s43": "a10crb3trs2nc_e1tv",
          "learned s42 (A)": "a10cb3trnc_e1tv", "learned s43 (A)": "a10cb3trs2nc_e1tv",
          "vanilla 5L": "b10c5trnc_e1tv"}
FORCED2 = {"random s42 forced d2": LS / "a10crb3trnc" / "a10crb3trnc_tvforce2",
           "random s43 forced d2": LS / "a10crb3trs2nc" / "a10crb3trs2nc_tvforce2"}
SEED_GAP = {"checkpoint-1980": 0.0103, "checkpoint-3300": 0.0069}   # the Coords-free cell's own (B3 note, section 1)


def _a0():
    """a0_read's loaders and pairing (eo/experiments/base/a0_read.py; eo/experiments is not a package)."""
    f = Path(__file__).resolve().parents[1] / "base" / "a0_read.py"
    spec = importlib.util.spec_from_file_location("a0_read", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unit_ok(d: Path) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    if (m["gpu"] != "NVIDIA TITAN V" or m.get("precision", "fp16") != "fp16"
            or list(m.get("drop_modalities") or []) != ["Coords"]
            or m["token_set"] != "clean" or m["n_eval_rows"] != 4416):
        raise SystemExit(f"{d}: {m['gpu']}, {m.get('precision')}, drop {m.get('drop_modalities')}, "
                         f"{m['token_set']}, {m['n_eval_rows']} rows: not the TITAN V fp16 protocol with Coords dropped")
    return m


def mean_rows(a: dict, b: dict, names) -> dict:
    return {n: {"n": a[n]["n"], "ce": (a[n]["ce"] + b[n]["ce"]) / 2, "top1": (a[n]["top1"] + b[n]["top1"]) / 2}
            for n in names}


def main(arm: str = "C") -> int:
    if arm not in ARM_UNITS:
        raise SystemExit(f"arm {arm!r}: one of {sorted(ARM_UNITS)}")
    C_UNITS, OUT, extra = ARM_UNITS[arm], ARM_OUT[arm], EXTRA_3300.get(arm, {})
    a0 = _a0()
    img = [m for m in a0.MODALITIES if m != "Coords"]
    names = img + ["image", "all"]
    fmt = lambda p: f"{p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"  # noqa: E731
    have = {k: t for k, t in C_UNITS.items() if all((AE / t / ck / "metrics.json").exists() for ck in CKS)}
    if not have:
        raise SystemExit(f"no {arm} unit is scored at both checkpoints yet")
    doc = {f"{arm.lower()}_units": have, "routed": ROUTED, "forced2": {k: str(v) for k, v in FORCED2.items()}, "seed_gap": SEED_GAP,
           "measure": f"pooled image-token CE, {arm} - other, paired per token, 95% row bootstrap ({a0.NBOOT} draws)",
           "by_ck": {}}
    lines = [f"B1 arm {arm} against the Coords-free 1 + 3 + 1 train cell: pooled image-token CE, {arm} - other, paired per token, "
             f"95% row bootstrap ({a0.NBOOT}); TITAN V fp16, Coords dropped. Negative: {arm} is better. "
             f"{arm} units: {', '.join(have)}"]
    for ck in CKS:
        gap = SEED_GAP[ck]
        R, M = {}, {}
        for k, t in have.items():
            d = AE / t / ck
            M[k] = unit_ok(d)
            c_ok, c_why = a0.same_sequences(a0.CONTROL, d)
            if c_ok:
                raise SystemExit(f"CONTROL PASSED (B10, original tokens, vs {d}: {c_why}): the check is blind")
            R[k] = a0.per_row(a0.tokens(d), names)
            z = a0.paired(R[k]["image"], R[k]["image"])
            if z["diff"] != 0 or z["lo"] != 0 or z["hi"] != 0:
                raise SystemExit(f"CHECK FAILED: {k} paired with itself reads {z}")
            e1 = a0.mean(R[k]["all"])
            if abs(e1 - M[k]["modes"]["router"]["ce_all_body_tokens"]) > 2e-3:
                raise SystemExit(f"{d}: E1 from tokens {e1:.6f} vs metrics.json "
                                 f"{M[k]['modes']['router']['ce_all_body_tokens']:.6f}")
        ref = AE / have[next(iter(have))] / ck
        others, why0 = {}, None
        units = {k: AE / t / ck for k, t in ROUTED.items()} | {k: p / ck for k, p in FORCED2.items()}
        if ck == "checkpoint-3300":
            units |= {k: AE / t / ck for k, t in extra.items() if (AE / t / ck / "metrics.json").exists()}
        for k, p in units.items():
            unit_ok(p)
            ok, why = a0.same_sequences(ref, p)
            if not ok:
                raise SystemExit(f"{ref} vs {p}: {why}: not the same sequences")
            why0 = why
            others[k] = a0.per_row(a0.tokens(p), names)
        for k in have:
            ok, why = a0.same_sequences(ref, AE / have[k] / ck)
            if not ok:
                raise SystemExit(f"{ref} vs {have[k]}: {why}: not the same sequences")
        others["random, both seeds"] = mean_rows(others["random s42"], others["random s43"], names)
        others["learned, both seeds (A)"] = mean_rows(others["learned s42 (A)"], others["learned s43 (A)"], names)
        others["forced d2, both seeds"] = mean_rows(others["random s42 forced d2"], others["random s43 forced d2"], names)
        order = ["random s42", "random s43", "random, both seeds", "learned s42 (A)", "learned s43 (A)",
                 "learned, both seeds (A)", "vanilla 5L", "random s42 forced d2", "random s43 forced d2",
                 "forced d2, both seeds"] + [k for k in extra if k in others]
        D = {"pairing": why0, "units": {}}
        lines.append(f"  {ck}: pairing {why0} with every unit; control (B10, original tokens) fails as it must against "
                     f"every {arm} unit; self-pairs 0 [0, 0]; the cell's seed gap {gap:.4f}")
        for k in have:
            U = {"image_ce": a0.mean(R[k]["image"]), "e1": a0.mean(R[k]["all"]), "pairs": {}}
            lines.append(f"    {k}: image tokens {U['image_ce']:.4f} (E1 {U['e1']:.4f})")
            for o in order + [c for c in have if c != k]:
                Ro = others[o] if o in others else R[o]
                p = a0.paired(R[k]["image"], Ro["image"])
                pm = {n: a0.paired(R[k][n], Ro[n]) for n in img}
                beyond, clear = p["diff"] < -gap, p["hi"] < -gap
                U["pairs"][o] = {"other_image_ce": a0.mean(Ro["image"]), "image": p, "per_modality": pm,
                                 "x_seed_gap": -p["diff"] / gap, "lower_by_more_than_gap": bool(beyond),
                                 "interval_clear_of_gap": bool(clear)}
                lines.append(f"      - {o:24s} {fmt(p)}  (other {a0.mean(Ro['image']):.4f}; {-p['diff'] / gap:+.2f}x "
                             f"the gap{'; beyond it' if beyond else ''}{', interval clear' if clear else ''})")
            for o in ("random, both seeds", "learned, both seeds (A)", "vanilla 5L", "forced d2, both seeds"):
                lines.append(f"        per modality, {k} - {o}: " + "; ".join(
                    f"{n} {fmt(v)}" for n, v in U["pairs"][o]["per_modality"].items()))
            D["units"][k] = U
        doc["by_ck"][ck] = D
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "read.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT / "read.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(*sys.argv[1:2]))
