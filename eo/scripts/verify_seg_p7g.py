#!/usr/bin/env python
"""Gate for A2 Step 7.0g (docs/plans/a2_segmentation.md, Step 7, "7.0g, E784's second seed with the full stack"):
the run-agnostic stack job, its submission and the read. Every check has a control that fails it. `.venv`, CPU.

  Z35r  7.0f's gate (verify_seg_p7f.py, 7.0e's and below inside it) on this tree: 11/11.
  Z35a  jobs/p7g_stack.sh finds a run's kept save from its records, against a stub RCP root whose data and base are the
        workstation's own (symlinks) and whose rcp_job.sh only echoes: a consistent save passes every check and reaches
        the generation. Controls (each must stop with exit 2 before generating): a moved hash in selected_sha256.txt,
        a selected_sha256.txt naming another checkpoint, no selected_checkpoint.txt, a run id that is not E784's.
  Z35b  submit_p7g.sh refuses a stack run without E784_RUN, before any connection. Control: a 7.0f run id is refused too.
  Z35c  the read's rules: replicates iff the pair gains; holds unless it costs. Control: an interval reaching 0 does not
        replicate.
    python eo/scripts/verify_seg_p7g.py [--only Z35a] [--out gate.json]
"""
from __future__ import annotations

import argparse
import hashlib
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
OUT = Path("/data/enric/reports/a2_seg/step70g")
TMP = OUT / "gate_tmp"


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def z35r() -> None:
    out = TMP / "z35r_gate_p7f.json"
    t0 = time.time()
    r = subprocess.run([sys.executable, "eo/scripts/verify_seg_p7f.py", "--out", str(out)], cwd=str(REPO),
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=7200)
    (TMP / "z35r_gate_p7f.log").write_text(r.stdout + r.stderr)
    res = json.loads(out.read_text()) if out.exists() else []
    n_pass = sum(x["status"] == "PASS" for x in res)
    record("Z35r 7.0f's gate on this tree", r.returncode == 0 and len(res) == 11 and n_pass == 11,
           f"exit {r.returncode}; {n_pass}/{len(res)} pass; {time.time() - t0:.0f} s; {out}")


def _stub(case: str) -> Path:
    R = TMP / f"stub_{case}"
    if R.exists():
        shutil.rmtree(R)
    (R / "a2_seg").mkdir(parents=True)
    (R / "a2_seg" / "geobench2_tta").symlink_to("/data/enric/data/geobench2_tta")
    base = R / "runs/pretrain/a2_seg/base"
    base.mkdir(parents=True)
    (base / "b10cltrxnct_ext_map_nc_ckpt2960").symlink_to("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnct_ext_map_nc_ckpt2960")
    run = R / "runs/pretrain/a2_seg/p7e784_s43_20990101_000000"
    (run / "checkpoint-12").mkdir(parents=True)
    w = os.urandom(4096)
    (run / "checkpoint-12/model.safetensors").write_bytes(w)
    sha = hashlib.sha256(w).hexdigest()
    if case != "no_selected":
        (run / "selected_checkpoint.txt").write_text("checkpoint-12\n")
    name = "checkpoint-12" if case != "other_ckpt" else "checkpoint-9"
    h = sha if case != "moved_hash" else sha[:-1] + ("0" if sha[-1] != "0" else "1")
    (run / "selected_sha256.txt").write_text(f"{h}  {name}/model.safetensors\n")
    (R / "scripts").mkdir()
    (R / "scripts/rcp_job.sh").write_text('#!/usr/bin/env bash\necho "stub rcp_job: $*"\nexit 0\n')
    return R


def _run_job(R: Path, run_id: str) -> tuple:
    env = dict(os.environ, MOR_EO_RCP=str(R), MOR_EO_CODE=str(R))
    r = subprocess.run(["bash", str(KIT / "jobs/p7g_stack.sh"), "gatestack", run_id], env=env, capture_output=True,
                       text=True, timeout=600)
    log = (R / "runs/joblogs/gen-gatestack.log")
    return r.returncode, (log.read_text() if log.exists() else "") + r.stdout + r.stderr


def z35a() -> None:
    R = _stub("ok")
    rc, log = _run_job(R, "p7e784_s43_20990101_000000")
    ok = ("data: the token files" in log and "equal the recorded SHA-256" in log and "python eo/scripts/eval_seg.py generate"
          in log and "--config eo_seg/s70e_grid784" in log and "checkpoint-12" in log and rc == 1)
    record("Z35a the stack job finds a run's kept save by its records", ok,
           f"exit {rc} (1 = the stub generation wrote no record, after every check passed); checks and generation line "
           f"reached: {ok}")
    stops = {}
    for case, rid in (("moved_hash", "p7e784_s43_20990101_000000"), ("other_ckpt", "p7e784_s43_20990101_000000"),
                      ("no_selected", "p7e784_s43_20990101_000000"), ("ok", "p7b1_s42_20261126_084434")):
        Rc = _stub(case if case != "ok" else "notE784")
        rc, log = _run_job(Rc, rid)
        stops[case if case != "ok" else "not_E784"] = (rc == 2 and "eval_seg.py generate" not in log)
    record("Z35a controls: each bad record stops the job before generating", all(stops.values()), f"{stops}")


def z35b() -> None:
    env = {k: v for k, v in os.environ.items() if k != "E784_RUN"}
    r = subprocess.run(["bash", str(KIT / "submit_p7g.sh"), "p7gstack_20990101_000000"], env=dict(env, SITE="lts4"),
                       capture_output=True, text=True, timeout=60)
    record("Z35b submit_p7g.sh refuses a stack run without E784_RUN", r.returncode == 2 and "E784_RUN" in r.stderr,
           f"exit {r.returncode}: {r.stderr.strip()[:80]}")
    c = subprocess.run(["bash", str(KIT / "submit_p7g.sh"), "p7fstack_20990101_000000"],
                       env=dict(env, SITE="lts4", E784_RUN="p7e784_s42_20261127_004451"), capture_output=True, text=True,
                       timeout=60)
    record("Z35b control: a 7.0f run id", c.returncode == 2 and "not a 7.0g" in c.stderr, f"exit {c.returncode}")


def z35c() -> None:
    import step70g_read as GR
    G = lambda d, lo, hi: {"delta": d, "lo95": lo, "hi95": hi}       # noqa: E731
    r = [GR.replicates(G(5.0, 3.5, 6.4)), GR.replicates(G(0.8, -0.3, 1.9))]
    h = [GR.holds(G(-0.9, -2.0, 0.2))[:5], GR.holds(G(-2.1, -3.4, -0.8))[:5], GR.holds(G(1.5, 0.3, 2.6))[:5]]
    ok = r == ["replicates", "does not replicate"] and h == ["holds", "does ", "holds"]
    record("Z35c the read's rules", ok, f"replicates {r}; holds {h}")
    c = GR.replicates(G(1.6, -0.05, 3.1))
    record("Z35c control: an interval reaching 0 does not replicate", c != "replicates", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    checks = {"Z35r": z35r, "Z35a": z35a, "Z35b": z35b, "Z35c": z35c}
    want = a.only.split(",") if a.only else list(checks)
    t0 = time.time()
    for k in want:
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
