#!/usr/bin/env python
"""B1 (2026-11-15): intervention 1's smoke replays, read by rules written before any of them ran.

Two questions (Paper B plan, B2): does the ringing stop, and does the price hold the mean? Each replay resumes a
learned run's checkpoint-1650 for 200 updates (to 1850) under the compute target (eo/train/compute_target.py) or,
as the control, under the balancing loss it was trained with (/data/enric/reports/b1/smoke/ct_replay.sh). Reads the
replays' depth_shares.jsonl (util/trainer_pt.py): per update, each class's positions at each depth, the per-class
sums of the router's soft expected depth, and under the compute target the controller (the price the update trained
under, its body soft mean, the price after it).

WINDOWS (steps inclusive): transient 1651-1750, read 1751-1850.

RULES, fixed 2026-11-15 before any smoke ran:
  R-hold   The price holds the mean if, over 1751-1850, the time mean of the body soft mean is within 0.02 passes
           of the target, and no update over 1801-1850 is more than 0.10 passes away.
  R-ring   The ringing stops if no image modality reads a tie (b3_depth_shares' state: at least half its updates
           with >= 90% of its positions at one depth, and two time-mean shares >= 0.1) whose deeper share has a
           periodogram peak over 2-50 updates at permutation p < 0.01, over 1751-1850. Read only against the
           control replay (same checkpoint, balancing 0.1, same window): if it shows no such modality, the control
           has not fired and the answer is "cannot tell".
  Description, read by no rule: ties left without a period, each modality's hard and soft time-mean depth, the body
  hard mean against the soft mean (the target holds the soft one; compute follows the hard one), the price's path.
  The prediction written 2026-11-09 (Paper B plan, B2): from A10c-s2's checkpoint-1650, the 1.8 replay sends DEM to
  depth 1 and the 2.0 replay does not; from A10c's, no slot changes depth at 1.8 (DEM and NDVI lose only their
  depth-3 tails), and at 2.0 DEM and NDVI tie for depth 3. Read here as DEM's time-mean depth-1 share over
  1751-1850: "DEM at depth 1" when it is at least 0.5.

    .venv/bin/python eo/experiments/routing/b1_ct_smoke.py --self-test
    .venv/bin/python eo/experiments/routing/b1_ct_smoke.py TAG RUN_DIR [--control CONTROL_TAG CONTROL_RUN_DIR]
    .venv/bin/python eo/experiments/routing/b1_ct_smoke.py TAG RUN_DIR --scratch      # arm B from step 0 (below)

ARM B FROM STEP 0 (--scratch; rule fixed 2026-11-15 before that smoke ran). A replay starts from a trained router whose
Adam state is mature; arm B starts from a uniform router, a fresh Adam and a 165-update warmup, where W34 saw one update
move the soft mean 0.32 passes. So its smoke reads the same R-hold with the windows moved to the end of 330 updates:
transient 1-230, read 231-330, late 281-330 (the plateau's first 165 updates are 166-330).

Writes /data/enric/reports/b1/smoke/<tag>.{json,txt}. Runs in `.venv`, CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b3_depth_shares as DS  # noqa: E402

OUT = Path("/data/enric/reports/b1/smoke")
IMG = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
WIN = {"transient": (1651, 1750), "read": (1751, 1850)}
HOLD_MEAN, HOLD_MAX, HOLD_LATE = 0.02, 0.10, (1801, 1850)
RING_P = 0.01
DEM1 = 0.5


def load(path: Path):
    """classes, steps, counts (T, class, depth), soft (T, class) or None, ct rows {step: dict}, header."""
    classes, rows, hdr = None, {}, {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if "header" in r:
            if classes is not None and r["header"]["classes"] != classes:
                raise SystemExit(f"{path}: a later header changes the classes")
            classes, hdr = r["header"]["classes"], r["header"]
            continue
        rows[int(r["step"])] = r          # a resume repeats steps: keep the last
    steps = np.array(sorted(rows))
    counts = np.array([rows[s]["counts"] for s in steps], dtype=np.float64)
    soft = (np.array([rows[s]["soft"] for s in steps], dtype=np.float64)
            if all("soft" in rows[s] for s in steps) else None)
    ct = {int(s): rows[s]["ct"] for s in steps if "ct" in rows[s]}
    return classes, steps, counts, soft, ct, hdr


def body_means(classes, steps, counts, soft):
    """Per update: the body positions' hard mean depth (counts) and soft mean (per-class sums), (T,), (T,) or None."""
    body = [i for i, c in enumerate(classes) if c not in ("none", "delim", "pad")]
    d = np.arange(1, counts.shape[2] + 1)
    c = counts[:, body, :].sum(1)
    hard = (c * d).sum(1) / c.sum(1)
    sm = None if soft is None else soft[:, body].sum(1) / c.sum(1)
    return hard, sm


def read(classes, steps, counts, soft, ct, hdr, rng) -> dict:
    r = DS.read(classes, steps, counts, rng, windows=WIN)
    hard, sm = body_means(classes, steps, counts, soft)
    out = {"n_updates": int(len(steps)), "steps": [int(steps[0]), int(steps[-1])], "windows": {}}
    for w, (a, b) in WIN.items():
        sel = (steps >= a) & (steps <= b)
        if sel.sum() < 20:
            continue
        res = r["windows"][w]
        mods = {}
        for m, x in res["mods"].items():
            i = classes.index(m)
            tot = counts[sel, i, :].sum(1)
            ok = tot > 0
            x = dict(x)
            if soft is not None and ok.any():
                x["soft_depth"] = float((soft[sel, i][ok] / tot[ok]).mean())
            mods[m] = x
        e = {"mods": mods, "load": res["load"], "body_hard": float(hard[sel].mean())}
        if sm is not None:
            e["body_soft"] = float(sm[sel].mean())
        out["windows"][w] = e
    rd = out["windows"]["read"]
    rings = [m for m in IMG if m in rd["mods"] and (rd["mods"][m].get("ringing") or {}).get("p") is not None
             and rd["mods"][m]["ringing"]["p"] < RING_P]
    ties = [m for m in IMG if m in rd["mods"] and rd["mods"][m]["state"].startswith("tie")]
    out["ringing"] = rings
    out["ties"] = ties
    dem = rd["mods"].get("DEM")
    out["dem_depth1_share"] = None if dem is None else float(dem["shares"][0])
    if ct:
        block = hdr.get("compute_target") or {}
        tgt = float(block["target"])
        s_ct = np.array(sorted(ct))
        m_ct = np.array([ct[s]["soft_mean"] for s in s_ct])
        p_ct = np.array([ct[s]["price"] for s in s_ct])
        a, b = WIN["read"]
        sel = (s_ct >= a) & (s_ct <= b)
        late = (s_ct >= HOLD_LATE[0]) & (s_ct <= HOLD_LATE[1])
        mean_gap = float(m_ct[sel].mean() - tgt) if sel.any() else None
        max_late = float(np.abs(m_ct[late] - tgt).max()) if late.any() else None
        hsel = (steps >= a) & (steps <= b)
        out["hold"] = {"target": tgt, "mean_gap_read": mean_gap, "max_gap_late": max_late,
                       # description, added 2026-11-15 after the first rows (no rule): the hard mean is the compute
                       "hard_mean_gap_read": float(hard[hsel].mean() - tgt) if hsel.any() else None,
                       "holds": bool(sel.sum() >= 50 and late.sum() >= 25 and abs(mean_gap) <= HOLD_MEAN
                                     and max_late <= HOLD_MAX),
                       "price_first": float(p_ct[0]), "price_last": float(p_ct[-1]),
                       "price_read_mean": float(p_ct[sel].mean()) if sel.any() else None,
                       "price_read_range": [float(p_ct[sel].min()), float(p_ct[sel].max())] if sel.any() else None,
                       "soft_path": {int(s): round(float(v), 4) for s, v in zip(s_ct[::10], m_ct[::10])},
                       "price_path": {int(s): round(float(v), 5) for s, v in zip(s_ct[::10], p_ct[::10])}}
    return out


def verdict(res: dict, ctl: dict | None) -> dict:
    v = {}
    if "hold" in res:
        v["R-hold"] = "holds" if res["hold"]["holds"] else "does not hold"
    if ctl is None:
        v["R-ring"] = "no control given"
    elif not ctl["ringing"]:
        v["R-ring"] = "cannot tell: the control replay shows no ringing tie in the read window"
    else:
        v["R-ring"] = ("the ringing stops" if not res["ringing"] else
                       f"the ringing goes on ({', '.join(res['ringing'])})")
    d = res.get("dem_depth1_share")
    v["DEM at depth 1"] = None if d is None else bool(d >= DEM1)
    return v


def text(tag, src, res, v, ctl_tag=None, ctl=None) -> str:
    L = [f"{tag}: {src}", f"  updates {res['n_updates']} (steps {res['steps'][0]}-{res['steps'][1]})"]
    for name, r in ((tag, res), (ctl_tag, ctl)):
        if r is None:
            continue
        for w, e in r["windows"].items():
            a, b = WIN[w]
            L.append(f"  {name} {w} {a}-{b}: body hard {e['body_hard']:.3f}"
                     + (f", soft {e['body_soft']:.3f}" if "body_soft" in e else "")
                     + f"; load {e['load']['shares']}")
            for m, x in e["mods"].items():
                g = x.get("ringing") or {}
                ring = (f" | period {g['period']:.1f} p {g['p']:.3f}" if g.get("period") else
                        (" | no period" if "ringing" in x else ""))
                L.append(f"    {m:7s} {x['state']:9s} {x['shares']} depth {x['mean_depth']:.2f}"
                         + (f" soft {x['soft_depth']:.2f}" if "soft_depth" in x else "") + ring)
    if "hold" in res:
        h = res["hold"]
        L.append(f"  controller: target {h['target']}, soft mean gap over the read window {h['mean_gap_read']:+.4f}, largest "
                 f"late gap {h['max_gap_late']:.4f}; HARD mean gap (the compute; description) {h['hard_mean_gap_read']:+.4f}; "
                 f"price {h['price_first']:+.5f} -> {h['price_last']:+.5f} (read window mean {h['price_read_mean']:+.5f}, "
                 f"range {h['price_read_range']})")
        L.append(f"    soft mean every 10 updates: {h['soft_path']}")
        L.append(f"    price every 10 updates: {h['price_path']}")
    L.append(f"  ringing ties (p < {RING_P}) in the read window: {res['ringing'] or 'none'}; ties: {res['ties'] or 'none'}"
             + (f"; control {ctl_tag}: ringing {ctl['ringing'] or 'none'}, ties {ctl['ties'] or 'none'}" if ctl else ""))
    L.append(f"  VERDICTS: {v}")
    return "\n".join(L)


def synthetic(kind: str, rng, n=200, start=1651):
    classes, _, counts = DS.synthetic("ring4" if kind == "ring" else "uniform", rng, n=n)
    steps = np.arange(start, start + n)
    soft = (counts * np.arange(1, 4)).sum(2)          # a router whose soft depth equals its hard depth
    t = np.arange(n)
    if kind == "osc":
        m = 1.8 + 0.15 * np.sin(2 * np.pi * t / 40)
    else:
        m = 1.8 + rng.normal(0, 0.005, n)
    ct = {int(s): {"price": 0.01, "soft_mean": float(v), "next": 0.01, "skipped": False} for s, v in zip(steps, m)}
    hdr = {"classes": classes, "compute_target": {"target": 1.8, "gain": 0.01, "proportional": 0.5, "price_init": 0.0}}
    return classes, steps, counts, soft, ct, hdr


def self_test() -> int:
    rng = np.random.default_rng(0)
    ok = True
    held = read(*synthetic("flat", rng), rng)
    osc = read(*synthetic("osc", rng), rng)
    ring = read(*synthetic("ring", rng), rng)
    flat_ctl = read(*synthetic("flat", rng), rng)
    checks = [
        ("a price holding 1.8 +- 0.005 holds", held["hold"]["holds"]),
        ("a soft mean swinging +-0.15 does not hold", not osc["hold"]["holds"]),
        ("a planted period-4 tie (NDVI) is a ringing tie", ring["ringing"] == ["NDVI"]),
        ("a uniform router has none", held["ringing"] == []),
        ("against a ringing control, a quiet replay reads 'stops'", verdict(held, ring)["R-ring"] == "the ringing stops"),
        ("against a quiet control, it reads 'cannot tell'", verdict(held, flat_ctl)["R-ring"].startswith("cannot tell")),
        ("a ringing replay against a ringing control 'goes on'", verdict(ring, ring)["R-ring"].startswith("the ringing goes on")),
    ]
    for label, c in checks:
        ok &= bool(c)
        print(f"  self-test: {label}: {'PASS' if c else 'FAIL'}")
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag", nargs="?")
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--control", nargs=2, metavar=("TAG", "RUN_DIR"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--scratch", action="store_true", help="arm B from step 0: windows 1-230 and 231-330, late 281-330")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.scratch:
        global HOLD_LATE
        WIN.clear()
        WIN.update({"transient": (1, 230), "read": (231, 330)})
        HOLD_LATE = (281, 330)
    if not (a.tag and a.run_dir):
        ap.error("tag and run_dir, or --self-test")
    rng = np.random.default_rng(0)
    src = a.run_dir / "depth_shares.jsonl"
    res = read(*load(src), rng)
    ctl = None
    if a.control:
        ctl = read(*load(Path(a.control[1]) / "depth_shares.jsonl"), np.random.default_rng(0))
    v = verdict(res, ctl)
    t = text(a.tag, str(src), res, v, a.control[0] if a.control else None, ctl)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.txt").write_text(t + "\n")
    (OUT / f"{a.tag}.json").write_text(json.dumps({"source": str(src), "result": res, "control": ctl, "verdicts": v},
                                                  indent=1, default=float))
    print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
