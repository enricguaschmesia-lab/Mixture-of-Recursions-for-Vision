#!/usr/bin/env python
"""Package a training checkpoint of a `continue` run as a fine-tune base (`.venv`; eo/finetune/init.write_continued).

A2 Step 5.2 (2026-11-23): R1's base is a training checkpoint (R1's rule 3), with no extension.json, so the fine-tune
loader (verify_extension, preflight's check_init_from) refused it. This copies the checkpoint's weights and config.json
into a new directory beside a `continued.json` that holds the extended base's own extension.json (verified here, its
weights hashed), the run's pretrain_init.json, the run, step and vocabulary size, and the weights' SHA-256; then reads
it back through verify_extension. Refuses an existing output, a base that fails its check, a run that continued from
other weights, or a checkpoint whose trainer state is at another step.

    python eo/scripts/package_continued.py \\
        --checkpoint /data/enric/runs/pretrain/a2_seg/r1aligned_n4000_20261122_193801/checkpoint-4000 \\
        --base /data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000 \\
        --out /data/enric/runs/pretrain/a2_seg/base/r1aligned_n4000_ckpt4000 \\
        [--admission /data/enric/reports/a2_seg/r1/forgetting_r1_checkpoint-4000.json] [--note "..."]

Copies of the output to other sites are compared by SHA-256 (the directory's three files), never rebuilt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path, help="the extended base the run continued from, on this machine")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--admission", type=Path, default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    from eo.finetune.init import CONTINUED, write_continued
    doc = write_continued(a.checkpoint, a.base, a.out, admission=a.admission, note=a.note)
    print(f"wrote {a.out}: {doc['run']} step {doc['step']}, vocab {doc['vocab_size']}, weights {doc['weights_sha256']}, "
          f"from the base {doc['base']['weights_sha256'][:12]}; {CONTINUED} verified back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
