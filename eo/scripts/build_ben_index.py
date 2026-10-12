#!/usr/bin/env python
"""Build BigEarthNet's patch index (PHASE6_PLAN Step 0.5). Any env (pandas only).

    # 1. rows + splits, before packing (copies the SMARTIES lists to /data and hashes them)
    python eo/scripts/build_ben_index.py rows --smarties <clone>/eval_splits
    # 2. after pack_ben.py: 19-class labels from the S2 metadata, cross-checked against S1
    python eo/scripts/build_ben_index.py labels

Writes /data/enric/data/BigEarthNet/index_v1.parquet and index_v1.json (provenance). Step 2
rewrites the parquet with label columns and refuses if any row moved.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.ben import labels as L, layout, splits  # noqa: E402


def _commit() -> str:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--smarties", type=Path, required=True,
                   help=f"SMARTIES eval_splits/ at commit {L.SMARTIES_SHA}")
    sub.add_parser("labels")
    a = ap.parse_args()

    prov_path = layout.INDEX.with_suffix(".json")
    if a.cmd == "rows":
        if layout.INDEX.exists():
            sys.exit(f"{layout.INDEX} exists; the row order is a one-way door. Remove it by hand.")
        dst = layout.SPLITS / f"smarties_{L.SMARTIES_SHA[:8]}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(a.smarties.glob("BigEarthNet-*.txt")):
            shutil.copy2(f, dst / f.name)
        df, prov = splits.build_index(dst)
        prov["smarties_copy"] = str(dst)
        prov["fork_commit"] = _commit()
        df.to_parquet(layout.INDEX)
        prov_path.write_text(json.dumps(prov, indent=2) + "\n")
        print(json.dumps(prov, indent=2))
    else:
        df = pd.read_parquet(layout.INDEX)
        m2 = pd.read_parquet(layout.PACKED / "meta_S2.parquet")
        m1p = layout.PACKED / "meta_S1.parquet"
        m1 = pd.read_parquet(m1p) if m1p.exists() else None
        out, rep = splits.attach_labels(df[[c for c in df.columns if c not in
                                            ("labels43", "labels19", "n_labels19", "date")]],
                                        m2, m1)
        assert (out.row.values == df.row.values).all() and (out.s2.values == df.s2.values).all()
        tmp = layout.INDEX.with_suffix(".parquet.tmp")
        out.to_parquet(tmp)
        tmp.replace(layout.INDEX)
        prov = json.loads(prov_path.read_text())
        prov["labels"] = rep
        prov["labels_fork_commit"] = _commit()
        prov_path.write_text(json.dumps(prov, indent=2) + "\n")
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
