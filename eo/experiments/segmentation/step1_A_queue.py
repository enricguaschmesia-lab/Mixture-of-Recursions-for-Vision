#!/usr/bin/env python
"""A2 Step 1, agent A: the driver. Runs step1_A_run.py's units one at a time on the TITAN V, each through
/data/enric/reports/a2_seg/queues/step1_titanv_unit.sh (TITAN V by name, power watchdog, 3 attempts,
log /data/enric/logs/a2_seg/step1/A_<stage>_<cand>.log), and skips a unit whose JSON exists.

A second watchdog, here: the queue's power watchdog cannot see a stall while agent B's training keeps the
card busy, so every unit touches a heartbeat file on each encode/decode call (step1_A_run.Progress); a
heartbeat older than --stall seconds is a stall, and the unit's own python process (its pid is in the
heartbeat file, and its command line is checked) gets SIGABRT, which the unit script logs and retries.

    python -u eo/experiments/segmentation/step1_A_queue.py --plan main --wait-session a2s1_anchor2
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "eo/experiments/segmentation/step1_A_run.py"
UNIT = Path("/data/enric/reports/a2_seg/queues/step1_titanv_unit.sh")
OUT = Path("/data/enric/reports/a2_seg/step1/A")
HB = OUT / "heartbeat"

TIMEOUT = {"select": 9000, "det": 1200, "read": 4200, "valsweep": 3600, "native256": 3600,
           "transposed": 3600, "prune": 4800}
NEEDS = {"select": [], "det": [], "read": ["select"], "valsweep": ["select"], "native256": ["select"],
         "transposed": ["select", "read"], "prune": ["select"]}

PLANS = {
    "main": [
        ("t1_di16_224", ["select", "det", "read", "valsweep", "transposed"]),
        ("t1_di16_256", ["select", "det", "read", "valsweep", "native256"]),
        ("t1_di8_256", ["select", "det", "read", "valsweep", "native256", "transposed"]),
        ("t1_di8_224", ["select", "det", "read", "valsweep"]),
        ("c01_di16_256", ["select", "det", "read", "valsweep", "native256"]),
        ("c01_di8_256", ["select", "det", "read", "valsweep", "native256"]),
    ],
}


def log(*a):
    print(f"[{time.strftime('%F %T')}]", *a, flush=True)


def out_path(stage: str, cand: str, K: str | None = None) -> Path:
    return OUT / (f"{stage}_{cand}_{'seen' if K == 'seen' else 'K' + str(K)}.json" if K else f"{stage}_{cand}.json")


def tmux_alive(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"={name}"], capture_output=True).returncode == 0


def pid_is_unit(pid: int, stage: str, cand: str) -> bool:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    cmd = [c.decode(errors="replace") for c in cmd]
    return any(c.endswith("step1_A_run.py") for c in cmd) and stage in cmd and cand in cmd


def run_unit(stage: str, cand: str, K: str | None, stall: int) -> int:
    out = out_path(stage, cand, K)
    name = out.stem
    hb = HB / f"{name}.hb"
    if hb.exists():
        hb.unlink()
    cmd = ["bash", str(UNIT), f"A_{name}", str(TIMEOUT[stage]), "python", "-X", "faulthandler", "-u", str(RUN),
           stage, "--cand", cand, "--out", str(out), "--heartbeat", str(hb), "--expect-gpu", "TITAN V"]
    if K:
        cmd += ["--K", str(K)]
    log("launch", name, " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(cmd)
    killed: set[int] = set()
    while p.poll() is None:
        time.sleep(30)
        if not hb.exists():
            continue
        try:
            pid = int(hb.read_text().split()[0])
        except (ValueError, IndexError, OSError):
            continue
        age = time.time() - hb.stat().st_mtime
        if age > stall and pid not in killed and pid_is_unit(pid, stage, cand):
            log(f"STALL: {name} heartbeat {age:.0f} s old; SIGABRT to pid {pid}")
            os.kill(pid, signal.SIGABRT)
            killed.add(pid)
    rc = p.returncode
    log(f"{name} rc={rc} after {time.time() - t0:.0f} s; json {'written' if out.exists() else 'MISSING'}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="main")
    ap.add_argument("--cands", default=None, help="comma list: restrict the plan to these candidates")
    ap.add_argument("--stages", default=None, help="comma list: restrict the plan to these stages")
    ap.add_argument("--prune", default=None, help="comma list of candidates to prune (after the plan)")
    ap.add_argument("--Ks", default="seen,16000,4000,1000")
    ap.add_argument("--wait-session", default=None, help="wait for this tmux session to end first")
    ap.add_argument("--stall", type=int, default=600)
    ap.add_argument("--done", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    HB.mkdir(parents=True, exist_ok=True)
    if a.wait_session:
        while tmux_alive(a.wait_session):
            log("waiting for tmux session", a.wait_session)
            time.sleep(60)
    units = []
    for cand, stages in (PLANS[a.plan] if a.plan != "none" else []):
        if a.cands and cand not in a.cands.split(","):
            continue
        for s in stages:
            if a.stages and s not in a.stages.split(","):
                continue
            units.append((s, cand, None))
    for cand in (a.prune.split(",") if a.prune else []):
        for K in a.Ks.split(","):
            units.append(("prune", cand, K))
    summary = []
    for stage, cand, K in units:
        out = out_path(stage, cand, K)
        if out.exists():
            log("skip (json exists)", out.name)
            summary.append((out.name, "exists"))
            continue
        missing = [n for n in NEEDS[stage] if not out_path(n, cand).exists()]
        if missing:
            log("skip", out.name, "missing", missing)
            summary.append((out.name, f"missing {missing}"))
            continue
        rc = run_unit(stage, cand, K, a.stall)
        summary.append((out.name, "ok" if out.exists() else f"FAILED rc={rc}"))
    done = Path(a.done) if a.done else OUT / f"queue_{a.plan}.done"
    done.write_text(json.dumps({"finished": time.strftime("%F %T"), "units": summary}, indent=1))
    log("queue done", summary)


if __name__ == "__main__":
    main()
