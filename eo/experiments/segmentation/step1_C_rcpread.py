"""Read agent C's finished runs on an RCP A100 (A2 Step 1): the same reads as on the workstation, beside them.

    python -X faulthandler -u -m eo.experiments.segmentation.step1_C_rcpread --root <out root> --runs c1000_256,c4375_224

For each run, from <root>/<run>/tok_best.pt (the best by the monitor mean), into <root>/<run>/read_a100/, each part
through step1_C_read (so through eo.maps.harness only), skipping parts already written (a recreated pod resumes):
  1. --select --read: the rule on train_sub (harness.select_rule) and the three read sets (harness.read_candidate);
  2. --controls with that rule: determinism, transposed maps, an untrained tokenizer, the assignment paths;
  3. --native256 (256-px tokenizers): CloudSen12 val in crop_mode native256, beside the main read.
Writes <root>/<run>/read_a100/DONE when all three parts are there. Encode and decode run in fp32 with TF32 off
(eo.maps.tokenizers.deterministic), so the A100's reads are fp32 reads, as the workstation's are.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--expect-gpu", default="A100")
    a = ap.parse_args()
    py = [sys.executable, "-X", "faulthandler", "-u", "-m", "eo.experiments.segmentation.step1_C_read"]
    bad = 0
    for name in a.runs.split(","):
        rd = Path(a.root) / name
        out = rd / "read_a100"
        tok = rd / "tok_best.pt"
        if (out / "DONE").exists():
            print(f"{name}: read_a100 done already", flush=True)
            continue
        if not (rd / "done.json").exists() or not tok.exists():
            print(f"{name}: not done or no tok_best.pt; skipped", flush=True)
            bad += 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        base = ["--tok", str(tok), "--out", str(out), "--expect-gpu", a.expect_gpu]
        if not all((out / f"read_{d}.json").exists() for d in ("pastis", "cloudsen12", "lulc")):
            if run(py + base + ["--select", "--read"]) != 0:
                bad += 1
                continue
        rule = json.dumps(json.loads((out / "selection.json").read_text())["chosen"])
        if not (out / "summary_controls.json").exists():
            if run(py + base + ["--controls", "--rule", rule]) != 0:
                bad += 1
                continue
        size = json.loads((rd / "config.json").read_text())["config"]["size"]
        if size == 256 and not (out / "read_cloudsen12_native256.json").exists():
            if run(py + base + ["--native256", "--rule", rule]) != 0:
                bad += 1
                continue
        (out / "DONE").write_text("ok\n")
        print(f"{name}: read_a100 complete", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
