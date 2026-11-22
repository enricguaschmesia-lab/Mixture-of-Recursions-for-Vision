#!/usr/bin/env python
"""A2 Step 4.0's follow-up 2 (Enric, 2026-11-22): freeze the fine-tuned Cosmos (Step 1 candidate (c), agent D) exactly
as Step 1 froze its shortlist (`a2seg`; no GPU). Its read is step1_D_read.py's JSON (the harness, the three read sets,
B's layout): /data/enric/reports/a2_seg/step1/D/rcp/<run>/read_best.json, with an optional read on a second card
beside it (read_best_<card>.json). Step 1's own _row(), attach_rescores() and freeze() copy the weights, the loader
module (step1_D_tok.py) and eo/maps/*.py read-only to /data/enric/reports/a2_seg/step1/frozen/aligned_D_<run>/ with
SHA256SUMS and record.json; a frozen record is never overwritten. It must have 196 tokens (the aligned geometry). The
bar is reported, not required: the plan reads a tokenizer under it as description only.

    python -m eo.experiments.segmentation.step4c_freeze d224_cosmos16_lg   # the candidate (2026-11-22)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from eo.experiments.segmentation import step1_O_shortlist as O


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "d224_cosmos16_lg"
    f = O.ROOT / "D" / "rcp" / run / "read_best.json"
    j = json.loads(f.read_text())
    other = None
    for g in sorted(f.parent.glob("read_best_*.json")):
        t = json.loads(g.read_text())
        if "read" in t and all(d in t["read"] for d in O.DATASETS):
            other = {"gpu": t.get("gpu"), **{d: t["read"][d]["ceiling"]["mIoU"] for d in O.DATASETS}}
    r = O._row(f"D:{run}", "D", j["read"], {"read": str(f)}, other, extra={"gpu": j.get("gpu")})
    O.attach_rescores([r])
    if r["tokens"] != 196 or r["n_codes"] != 64000:
        raise SystemExit(f"{r['name']}: {r['tokens']} tokens, {r['n_codes']} codes: not the aligned geometry")
    rec = O.freeze(r, "aligned")
    print(json.dumps({"frozen": r["name"], "passes_bar": r["passes"], "tokens": rec["tokens"], "n_codes": rec["n_codes"],
                      "grid": rec["grid"], "input_size": rec["input_size"], "rule": rec["rule"], "ceilings": rec["ceilings"],
                      "mean": rec["mean"], "loader_frozen": rec["loader_frozen"], "files": len(rec["frozen_files_sha256"]),
                      "other_card": rec["other_card"]}))


if __name__ == "__main__":
    main()
