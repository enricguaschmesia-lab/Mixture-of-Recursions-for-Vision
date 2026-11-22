#!/usr/bin/env python
"""A2 Step 4.0's follow-ups (Enric, 2026-11-20, evening): freeze the two f8 map tokenizers at 224 px that the nested
arms use, exactly as Step 1 froze its shortlist (`a2seg`; no GPU). It calls step1_O_shortlist's own collect(),
attach_rescores() and freeze(), so each candidate's weights, its loader module and eo/maps/*.py are copied read-only to
/data/enric/reports/a2_seg/step1/frozen/long_<name>/ with SHA256SUMS and record.json; a frozen record is never
overwritten. The candidates must pass rule 1.3's bar and be in the long band (784 tokens).

    python -m eo.experiments.segmentation.step4f_freeze C:c4375_224 C:c1000_224a
"""
from __future__ import annotations

import json
import sys

from eo.experiments.segmentation import step1_O_shortlist as O


def main():
    names = sys.argv[1:] or ["C:c4375_224", "C:c1000_224a"]
    rows = O.collect()
    O.attach_rescores(rows)
    by = {r["name"]: r for r in rows}
    for n in names:
        r = by.get(n)
        if r is None:
            raise SystemExit(f"{n}: not among Step 1's read candidates")
        if not r["passes"] or r["band"] != "long" or r["tokens"] != 784:
            raise SystemExit(f"{n}: passes {r['passes']}, band {r['band']}, {r['tokens']} tokens: not a 784-token passer")
        rec = O.freeze(r, "long")
        print(json.dumps({"frozen": n, "tokens": rec["tokens"], "n_codes": rec["n_codes"], "grid": rec["grid"],
                          "input_size": rec["input_size"], "rule": rec["rule"], "ceilings": rec["ceilings"],
                          "loader_frozen": rec["loader_frozen"], "files": len(rec["frozen_files_sha256"])}))


if __name__ == "__main__":
    main()
