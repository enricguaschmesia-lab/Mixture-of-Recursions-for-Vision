#!/usr/bin/env python
"""A1 Step 9: every reading of a pilot checkpoint paired with its step-0 value, per location (`.venv`).

eval_temporal.py's `report` pairs a checkpoint's readings with Step 0's stored T360 units (the margins). This
pairs them with the same readings of T360-T's extended base, the pilot's step 0 (Step 7), unit by unit: for
every unit key both directories hold, the per-location difference checkpoint - base, its mean and Step 0's
bootstrap over 5 km groups (eval_temporal's group_ci, 10,000 draws). The units must be on the same locations
and target dates, in Step 0's order, and scored under the same layout entry (its hash), or the unit is refused.

    python eo/experiments/temporal/pilot_compare.py --units <out-root>/<tag>/<checkpoint>/units [--base-units DIR]
           [--out OUT.json] [--only SUBSTR,...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

BASE = Path("/data/enric/reports/temporal/step7/t360t_base/extended_base/units")
VAL = Path("/data/enric/data/TerraMesh/val_clean")


def load(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        rec = {k: z[k] for k in z.files}
        rec["spec"] = json.loads(str(z["spec"]))
        out[f.stem] = rec
    return out


def main():
    from eo.data.temporal_index import load_temporal_index
    from eo.eval import temporal as ET

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--base-units", type=Path, default=BASE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", default="", help="comma-separated substrings of unit keys")
    a = ap.parse_args()
    ck, base = load(a.units), load(a.base_units)
    only = [s for s in a.only.split(",") if s]
    keys = sorted(k for k in set(ck) & set(base) if not only or any(s in k for s in only))
    if not keys:
        sys.exit(f"no unit key in both {a.units} and {a.base_units}")
    locs_all, groups_all, _ = ET.step0_locations(load_temporal_index(VAL))
    rep, refused = {}, []
    for k in keys:
        c, b = ck[k], base[k]
        if c["spec"]["entry_sha256"] != b["spec"]["entry_sha256"]:
            refused.append(f"{k}: another layout entry")
            continue
        if "ce" not in c or "ce" not in b:
            continue
        n = min(len(c["ce"]), len(b["ce"]))
        if not (np.array_equal(c["locations"][:n], b["locations"][:n]) and np.array_equal(c["t"][:n], b["t"][:n])
                and np.array_equal(c["locations"][:n], locs_all[:n])):
            refused.append(f"{k}: other locations or target dates")
            continue
        d = c["ce"][:n] - b["ce"][:n]
        ci = ET.group_ci(d, groups_all[:n])
        rep[k] = {"n": int(n), "checkpoint": round(float(c["ce"][:n].mean()), 5), "base": round(float(b["ce"][:n].mean()), 5),
                  "minus_base": {kk: round(float(v), 5) for kk, v in ci.items()} if isinstance(ci, dict) else ci}
    doc = {"units": str(a.units), "base_units": str(a.base_units), "n_keys": len(rep), "refused": refused, "pairs": rep}
    text = json.dumps(doc, indent=1)
    if a.out:
        a.out.write_text(text)
    for k, v in rep.items():
        m = v["minus_base"]
        print(f"{k:60s} {v['base']:9.4f} -> {v['checkpoint']:9.4f}  "
              f"{m.get('mean', float('nan')):+8.4f} [{m.get('lo', float('nan')):+8.4f}, {m.get('hi', float('nan')):+8.4f}]")
    if refused:
        print("refused:", refused)


if __name__ == "__main__":
    main()
