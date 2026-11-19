"""One RCP job of agent B (A2 Step 1): train a run to the end of its schedule, then read its monitor-chosen
checkpoint (best.pt) on the three read sets with the controls (Step 1.2).

    python -X faulthandler -u -m eo.experiments.segmentation.step1_B_rcp_run --run b256_4375 \
        --out-root $L/a2seg/out/b [--read-args "--native256"] -- <step1_B_train args ...>

Safe to restart from the top (a preempted pod reruns the command): the training resumes from its
ckpt_last.pt and exits at once once done.json exists; the read is skipped when read_best.json already holds
its controls, and otherwise rerun whole (its tokens are cached under A2_CACHE_ROOT).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--read-args", default="", help="extra step1_B_read args, e.g. --native256")
    ap.add_argument("--no-read", action="store_true")
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    targs = a.train_args[1:] if a.train_args[:1] == ["--"] else a.train_args
    out = Path(a.out_root) / a.run
    py = [sys.executable, "-X", "faulthandler", "-u", "-m"]
    cmd = py + ["eo.experiments.segmentation.step1_B_train", "--run", a.run, "--out-root", a.out_root] + targs
    print("$", shlex.join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    if not (out / "done.json").exists():
        print(f"{out}: no done.json after training; not reading", flush=True)
        sys.exit(1)
    if a.no_read:
        return
    rj = out / "read_best.json"
    if rj.exists() and "controls" in json.loads(rj.read_text()):
        print(f"{rj} holds its controls: nothing to read", flush=True)
        return
    cmd = py + ["eo.experiments.segmentation.step1_B_read", "--ckpt", str(out / "best.pt"), "--out", str(rj),
                "--select", "--read", "--controls", "--expect-gpu", "A100"] + shlex.split(a.read_args)
    print("$", shlex.join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
