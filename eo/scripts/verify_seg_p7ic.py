#!/usr/bin/env python
"""Gate for A2 Step 7.0i's CloudSen12 corrected vote (docs/plans/a2_segmentation.md, Step 7, "7.0i"): the vote-count
decode, the weighted vote, the fixed prior and halves, the cross-fit, the bootstrap and the verdict rule. Every check has
a control that fails it. `.venv`; Z38a decodes 4 tiles on the GTX Titan X (a2seg, through score_locked.sh).

  Z38a  step70i_c12_votes.py on seed 42's K = 40 run, first 4 tiles: the plain vote of its counts equals the scorer's
        per-tile matrices at K = 40 and K = 8. Control: --drop-first (39 votes) does not equal them at K = 40.
  Z38b  weighted_pred: w = 1 is the plain vote (ties to the lower class); a prior favouring class 1 moves a crafted
        pixel; a class with no vote never wins, whatever its weight. Control: the crafted pixel unweighted stays 0.
  Z38c  the halves file equals its recomputation (sorted tile id, alternating; 268 / 267). Control: the same rule on the
        unsorted order gives another split.
  Z38d  pair_task(x, y, 19) equals step70_read.pair on PASTIS's matrices (B1 K8, E784 K8). Control: another bootstrap
        seed gives another interval. (Run 1's control, ignore None, could not fail: the matrices hold no void pixel.)
  Z38e  cross_fit scores each half at the alpha the OTHER half chose (crafted matrices where each half prefers another
        alpha), and equals step70e_read.cross_fit on B1's K8. Control: a same-half assignment differs.
  Z38f  the verdict rule on known pairs. Control: one replicate at or below 0 is not "gains".
  Z38g  the prior file equals its recomputation from the 4,000 training tiles. Control: the val tiles' shares differ.
    python eo/scripts/verify_seg_p7ic.py [--only Z38a,...] [--out gate.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "experiments" / "segmentation"))

RESULTS: list = []
OUT = Path("/data/enric/reports/a2_seg/step70i")
TMP = OUT / "gate_tmp"
K40_42 = Path("/data/enric/reports/a2_seg/step45/gen/r0c12s52_s42_20261122_190655/full_k40")


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def _decode(tag: str, extra: list) -> dict:
    out = TMP / f"c12_counts_{tag}"
    if out.exists():
        shutil.rmtree(out)
    cmd = ["bash", "/data/enric/rcp_a2s4/s70e/score_locked.sh", f"gate_c12_{tag}", "1800", "a2seg",
           str(REPO / "eo/experiments/segmentation/step70i_c12_votes.py"), "--gen", str(K40_42), "--out", str(out),
           "--tiles", "4", "--expect-gpu", "GTX TITAN X"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    f = out / "votes.json"
    return json.loads(f.read_text()) if f.exists() else {"error": r.stdout[-400:] + r.stderr[-400:]}


def z38a() -> None:
    m = _decode("ok", [])
    ok = m.get("check", {}).get("k40", {}).get("equal_to_scorer") and m["check"]["k8"]["equal_to_scorer"]
    record("Z38a the counts' plain vote = the scorer's matrices (4 tiles)", bool(ok), f"{m.get('check', m)}")
    c = _decode("drop_first", ["--drop-first"])
    fired = c.get("check", {}).get("k40", {}).get("equal_to_scorer") is False
    record("Z38a control: 39 votes", fired, f"K = 40 equal {c.get('check', {}).get('k40', {}).get('equal_to_scorer')}")


def z38b() -> None:
    import step70i_c12_read as C
    import step70i_c12_votes as V
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 6, size=(3, 4, 16, 16)).astype(np.uint8)
    same = all(np.array_equal(C.weighted_pred(counts[i], np.ones(4)), V.plain_from_counts(counts)[i]) for i in range(3))
    px = np.zeros((4, 1, 1), np.uint8); px[0], px[1] = 3, 2
    w = np.ones(4); w[1] = 2.0
    moved = int(C.weighted_pred(px, w)[0, 0]) == 1
    nv = np.zeros((4, 1, 1), np.uint8); nv[0] = 1
    big = np.ones(4); big[3] = 1e6
    never = int(C.weighted_pred(nv, big)[0, 0]) == 0
    record("Z38b the weighted vote", same and moved and never, f"w = 1 is plain {same}; moved {moved}; no-vote class never wins {never}")
    stays = int(C.weighted_pred(px, np.ones(4))[0, 0]) == 0
    record("Z38b control: unweighted", stays, f"stays 0 {stays}")


def z38c() -> None:
    from eo.maps import sets
    half = json.loads((OUT / "halves_cloudsen12.json").read_text())["half"]
    _, ids = sets.load("cloudsen12", "val")
    order = sorted(range(len(ids)), key=lambda i: str(ids[i]))
    again = {str(ids[i]): r % 2 for r, i in enumerate(order)}
    n1 = sum(half.values())
    record("Z38c the halves", again == half and (len(half) - n1, n1) == (268, 267), f"equal {again == half}; sizes "
           f"{len(half) - n1} / {n1}")
    unsorted = {str(ids[i]): i % 2 for i in range(len(ids))}
    record("Z38c control: the unsorted order", unsorted != half, f"differs {unsorted != half}")


def z38d() -> None:
    import step70_read as S70
    import step70k_read as K
    import step70e_read as RD
    import step70i_c12_read as C
    b1 = np.load(K.gen_dir(K.B1_RUN) / "full_k8" / "per_patch_cm.npy")
    e = np.load(RD.GEN / "p7e784_s42_20261127_004451" / "full_k8" / "per_patch_cm.npy")
    p, q = S70.pair(e, b1), C.pair_task(e, b1, 19)
    same = all(p[k] == q[k] for k in ("x", "y", "delta", "lo95", "hi95"))
    record("Z38d pair_task = step70_read.pair (PASTIS)", same, f"{q['delta']:+.4f} [{q['lo95']:+.4f}, {q['hi95']:+.4f}]")
    seed = C.BOOT_SEED
    C.BOOT_SEED = seed + 1                     # another bootstrap draw must give another interval
    try:
        r = C.pair_task(e, b1, 19)
    finally:
        C.BOOT_SEED = seed
    record("Z38d control: another bootstrap seed", (r["lo95"], r["hi95"]) != (q["lo95"], q["hi95"]),
           f"[{r['lo95']:+.4f}, {r['hi95']:+.4f}]")


def z38e() -> None:
    import step70e_read as RD
    import step70i_c12_read as C
    A, B = np.zeros((4, 4, 4)), np.zeros((4, 4, 4))
    h = np.array([0, 1, 0, 1])
    for i in range(4):
        A[i] = np.diag([10, 10, 10, 10]) if h[i] == 0 else np.diag([10, 10, 10, 5]) + np.eye(4, k=1) * 0
        B[i] = np.diag([10, 10, 10, 5]) if h[i] == 0 else np.diag([10, 10, 10, 10])
        A[i][3, 0] += 0 if h[i] == 0 else 5
        B[i][3, 0] += 5 if h[i] == 0 else 0
    cf, chosen = C.cross_fit({0.0: A, 0.5: B}, h, None)
    want = np.where((h == 0)[:, None, None], B, A)          # half 0 prefers 0.0 (A), half 1 prefers 0.5 (B)
    ok1 = chosen == {0: 0.0, 1: 0.5} and np.array_equal(cf, want)
    prior = json.loads(RD.PRIOR.read_text())
    t2 = RD.test2_of("b1_k8", prior)
    cf2, ch2 = C.cross_fit(t2["_cms"], RD.halves(t2["_sids"]), 19)
    ok2 = np.array_equal(cf2, t2["_cf"]) and {str(k): v for k, v in ch2.items()} == t2["chosen"]
    record("Z38e cross-fitting", ok1 and ok2, f"crafted {chosen}, other half's alpha {ok1}; = step70e_read on B1 K8 {ok2}")
    same_half = np.where((h == 0)[:, None, None], A, B)
    record("Z38e control: a same-half assignment", not np.array_equal(cf, same_half), "differs")


def z38f() -> None:
    import step70i_c12_read as C
    G = lambda d, lo, hi: {"delta": d, "lo95": lo, "hi95": hi}       # noqa: E731
    v = [C.verdict(G(0.6, 0.2, 1.0), [G(0.3, -0.1, 0.7), G(0.4, 0.0, 0.8)]),
         C.verdict(G(-0.6, -1.0, -0.2), [G(-0.3, -0.7, 0.1), G(-0.4, -0.8, 0.0)]),
         C.verdict(G(0.6, -0.1, 1.3), [G(0.3, -0.1, 0.7), G(0.4, 0.0, 0.8)])]
    record("Z38f the verdict rule", v == ["gains", "costs", "level"], f"{v}")
    c = C.verdict(G(0.6, 0.2, 1.0), [G(0.3, -0.1, 0.7), G(0.0, -0.4, 0.4)])
    record("Z38f control: one replicate at 0", c != "gains", c)


def z38g() -> None:
    from eo.maps import sets
    pr = json.loads((OUT / "class_prior_cloudsen12.json").read_text())
    def shares(split):
        lab, _ = sets.load("cloudsen12", split)
        cnt = np.zeros(4, np.int64)
        for s in range(0, len(lab), 250):
            cnt += np.bincount(np.asarray(lab[s:s + 250]).ravel(), minlength=4)[:4]
        return cnt
    tr = shares("train")
    ok = [int(tr[i]) for i in range(4)] == [pr["pixel_counts"][n] for n in pr["class_index"]] and pr["n_tiles"] == 4000
    record("Z38g the prior = its recomputation (4,000 training tiles)", ok, f"counts equal {ok}")
    va = shares("val")
    record("Z38g control: the val tiles' shares", not np.allclose(va / va.sum(), tr / tr.sum(), atol=1e-4), "differ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    checks = {"Z38a": z38a, "Z38b": z38b, "Z38c": z38c, "Z38d": z38d, "Z38e": z38e, "Z38f": z38f, "Z38g": z38g}
    t0 = time.time()
    for k in (a.only.split(",") if a.only else list(checks)):
        try:
            checks[k]()
        except Exception as e:
            traceback.print_exc()
            record(k, False, f"raised {type(e).__name__}: {e}")
    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {n_fail} fail ({time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in RESULTS], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
