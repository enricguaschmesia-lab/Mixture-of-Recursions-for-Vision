#!/usr/bin/env python
"""Gate for A2 Step 7.0i (docs/plans/a2_segmentation.md, Step 7, "7.0i, E784's seeds 44-46, then CloudSen12 at 40 votes
for R0's seeds 43 and 44"): the overnight runner, the pushed checkpoint, the read's rules and its pooled vote. Every check
has a control that fails it. `.venv`, CPU (Z37b reads RCP's lts4 scratch through the jump host).

  Z37a  runner_p7i.sh in its dry mode (stub submissions and job states, no connection): every job submitted in queue
        order, each after a refused first try; the CloudSen12 jobs only after every cell is submitted; CELL45's failure
        stops nothing; the read named with every run at the end. Control: a malformed queue (CELL45 naming seed 44's
        run) is refused before any submission.
  Z37b  seed 43's CloudSen12 checkpoint on lts4 equals the workstation's, file for file (SHA-256). Control: a hash with
        one digit changed does not.
  Z37c  the read's rules: a seed replicates iff it gains by B1's 1.2 bar; "all five" iff five seeds each replicate.
        Control: one seed level is not "all five", nor are four seeds that replicate.
  Z37d  the pooled vote: one model pooled alone equals its own K8 vote (the scorer's per-patch matrices, seed 42).
        Control: a second model whose patches come in another order is refused.
  Z37e  (added 2026-11-28, after C12_43's vote curve stopped on its checkpoint path) step45_kcurve.same_run_diffs
        accepts seed 43's two records (the K = 8 run naming course's copy, the K = 40 run lts4's) given the push's two
        identical SHA-256 listings. Controls: without the listings it refuses; with listings that differ in one digit it
        refuses; another run's checkpoint is refused even with the listings.
  Z37r  7.0g's read rules (verify_seg_p7g.py --only Z35c): 2/2.
    python eo/scripts/verify_seg_p7i.py [--only Z37a,...] [--out gate.json]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "experiments" / "segmentation"))

RESULTS: list = []
KIT = Path("/data/enric/rcp_a2s4/s70e")
OUT = Path("/data/enric/reports/a2_seg/step70i")
TMP = OUT / "gate_tmp"
LTS4_RUN = "/mnt/lts4-students/scratch/students/guasch/mor-eo-rcp/runs/pretrain/a2_seg/r0c12s52_s43_20261122_190655"


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def _dry(case: str, queue: str | None = None) -> tuple:
    st, log = TMP / f"dry_{case}", TMP / f"dry_{case}.log"
    if st.exists():
        shutil.rmtree(st)
    st.mkdir(parents=True)
    log.unlink(missing_ok=True)
    if queue is not None:
        (st / "queue.txt").write_text(queue)
    env = dict(os.environ, P7I_DRY="1", P7I_STATE=str(st), P7I_LOG=str(log))
    r = subprocess.run(["bash", str(KIT / "runner_p7i.sh"), "20990101_000000"], env=env, capture_output=True, text=True,
                       timeout=600)
    return r.returncode, st, (log.read_text() if log.exists() else "")


def z37a() -> None:
    rc, st, log = _dry("ok")
    lines = log.splitlines()
    sub = [l for l in lines if ": submitted dry-" in l]
    order = [l.split()[2].rstrip(":") for l in sub]
    first_c12 = next((i for i, l in enumerate(lines) if ": submitted dry-C12_" in l), None)
    last_cell = max((i for i, l in enumerate(lines) if ": submitted dry-CELL" in l), default=None)
    retried = all(f"dry-{n}-2" in log and f"dry-{n}-1" not in log for n in ("CELL44", "CELL45", "CELL46", "C12_44", "C12_43"))
    states = {n: sorted(p.suffix for p in st.glob(f"{n}.*")) for n in ("CELL44", "CELL45", "CELL46", "C12_44", "C12_43")}
    read = next((l for l in lines if "(dry) read:" in l), "")
    want_runs = ["p7e784_s44_20990101_000000", "p7e784_s45_20990101_000000", "p7e784_s46_20990101_000000",
                 "r0c12s52_s44_20261122_190655", "r0c12s52_s43_20261122_190655"]
    ok = (rc == 0 and order == ["CELL44", "CELL45", "CELL46", "C12_44", "C12_43"] and retried
          and first_c12 is not None and last_cell is not None and first_c12 > last_cell
          and ".failed" in states["CELL45"] and all(".done" in states[n] for n in ("CELL44", "CELL46", "C12_44", "C12_43"))
          and (st / "runner.done").exists() and all(r in read for r in want_runs))
    record("Z37a the runner's dry run", ok, f"exit {rc}; order {order}; retried after a refusal {retried}; CloudSen12 after "
           f"the cells {first_c12 is not None and last_cell is not None and first_c12 > last_cell}; CELL45 "
           f"{states['CELL45']}; runner.done {(st / 'runner.done').exists()}; the read names every run "
           f"{all(r in read for r in want_runs)}")
    bad = ("CELL44 p7e784_s44_20990101_000000 -\nCELL45 p7e784_s44_20990101_000000 -\nCELL46 p7e784_s46_20990101_000000 -\n"
           "C12_44 r0c12s52_s44_20261122_190655 checkpoint-3500\nC12_43 r0c12s52_s43_20261122_190655 checkpoint-4625\n")
    rc2, st2, log2 = _dry("bad_queue", bad)
    refused = rc2 == 2 and "malformed" in log2 and not list(st2.glob("*.submitted"))
    record("Z37a control: a malformed queue", refused, f"exit {rc2}; refused before any submission {refused}")


def z37b() -> None:
    want = (KIT / "push_c12_s43.sha256").read_text().split("\n")
    want = sorted(l for l in want if l.strip())
    r = subprocess.run(["ssh", "-n", "-o", "ControlPath=none", "-o", "ConnectTimeout=20", "rcp-jump",
                        f"cd {LTS4_RUN} && sha256sum checkpoint-4625/*"], capture_output=True, text=True, timeout=300)
    got = sorted(l for l in r.stdout.splitlines() if len(l) > 66 and l[64:66] == "  ")
    record("Z37b seed 43's checkpoint on lts4 equals the workstation's", got == want and len(want) == 4,
           f"{len(got)} files listed; equal {got == want}")
    bad = list(want)
    h = bad[2]
    bad[2] = ("0" if h[0] != "0" else "1") + h[1:]
    record("Z37b control: one digit changed", got != sorted(bad), f"differs {got != sorted(bad)}")


def z37c() -> None:
    import step70i_read as I
    G = lambda d, lo, hi: {"delta": d, "lo95": lo, "hi95": hi}       # noqa: E731
    r = [I.replicates(G(6.4, 5.0, 7.7)), I.replicates(G(0.9, -0.2, 2.0)), I.replicates(G(1.6, -0.05, 3.1))]
    five = I.all_five(["replicates"] * 5)
    ok = r == ["replicates", "does not replicate", "does not replicate"] and five == "replicates on all five seeds"
    record("Z37c the read's rules", ok, f"calls {r}; five {five!r}")
    c1 = I.all_five(["replicates"] * 4 + ["does not replicate"])
    c2 = I.all_five(["replicates"] * 4)
    record("Z37c control: one seed level, or four seeds", c1 != "replicates on all five seeds" and
           c2 != "replicates on all five seeds", f"{c1!r}; {c2!r}")


def z37d() -> None:
    import numpy as np
    import step70e_read as RD
    import step70i_read as I
    prior = json.loads(RD.PRIOR.read_text())
    run = "p7e784_s42_20261127_004451"
    P = I.pooled([run], prior)
    stored = np.load(RD.GEN / run / "full_k8" / "per_patch_cm.npy")
    same = np.array_equal(P["_cms"][0.0], stored)
    record("Z37d one model pooled alone = its own K8 vote", same and P["votes"] == 8,
           f"equal to the scorer's matrices {same}; plain {P['plain']:.2f}")
    orig = RD.load_maps
    def shuffled(name):
        m, vj, meta = orig(name)
        if name.startswith("p7e784_s43"):
            vj = dict(vj, sample_id=list(reversed(vj["sample_id"])))
        return m, vj, meta
    RD.load_maps = shuffled
    try:
        I.pooled([run, "p7e784_s43_20261127_133608"], prior)
        refused = False
    except RuntimeError:
        refused = True
    finally:
        RD.load_maps = orig
    record("Z37d control: another patch order", refused, f"refused {refused}")


def z37e() -> None:
    import step45_kcurve as KC
    run = "r0c12s52_s43_20261122_190655"
    rec = json.loads((Path("/data/enric/reports/a2_seg/step45/gen") / run / "full_k40" / "record.json").read_text())
    rr = json.loads((Path("/data/enric/reports/a2_seg/step52/gen") / run / "full_k8" / "record.json").read_text())
    sw = [str(KIT / "push_c12_s43.sha256"), str(KIT / "push_c12_s43.remote.sha256")]
    ok = KC.same_run_diffs(rec, rr, sw) == {} and "checkpoint" in KC.same_run_diffs(rec, rr, None)
    record("Z37e seed 43's two records are one checkpoint's, given the push's listings", ok,
           f"with listings {KC.same_run_diffs(rec, rr, sw)}; without {sorted(KC.same_run_diffs(rec, rr, None))}")
    bad = TMP / "z37e_bad.sha256"
    lines = Path(sw[1]).read_text().splitlines()
    lines[0] = ("0" if lines[0][0] != "0" else "1") + lines[0][1:]
    bad.write_text("\n".join(lines) + "\n")
    other = dict(rec, checkpoint=rec["checkpoint"].replace("r0c12s52_s43_", "r0c12s52_s44_"))
    c1 = "checkpoint" in KC.same_run_diffs(rec, rr, [sw[0], str(bad)])
    c2 = "checkpoint" in KC.same_run_diffs(other, rr, sw)
    record("Z37e controls: listings one digit apart; another run's checkpoint", c1 and c2,
           f"one digit refused {c1}; another run refused {c2}")


def z37r() -> None:
    out = TMP / "z37r_gate_p7g_z35c.json"
    r = subprocess.run([sys.executable, "eo/scripts/verify_seg_p7g.py", "--only", "Z35c", "--out", str(out)],
                       cwd=str(REPO), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True,
                       timeout=600)
    res = json.loads(out.read_text()) if out.exists() else []
    n = sum(x["status"] == "PASS" for x in res)
    record("Z37r 7.0g's read rules", r.returncode == 0 and len(res) == 2 and n == 2, f"exit {r.returncode}; {n}/{len(res)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    checks = {"Z37a": z37a, "Z37b": z37b, "Z37c": z37c, "Z37d": z37d, "Z37e": z37e, "Z37r": z37r}
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
