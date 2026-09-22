#!/usr/bin/env python
"""Materialize the Phase 3 held-out evaluation split (D3.1).

Thin CLI. The design, the measurements behind the group key and the one-way
door warning all live in eo/data/eval_split.py -- read that first.

    python eo/scripts/build_eval_split.py            # write eo/data/eval_rows_v1.json
    python eo/scripts/build_eval_split.py --dry-run  # print the summary, write nothing

⚠ The output is COMMITTED and is a one-way door once an arm is trained against
it. Rebuilding it with different constants silently invalidates every
comparison made on the old one, so this refuses to overwrite an existing
artifact without --force.

Runs in the repo `.venv` (numpy/pandas/pyarrow), not the `mor` conda env.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eo.data.eval_split import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    EVAL_GROUP_FRACTION,
    SPLIT_RADIUS_KM,
    SPLIT_SEED,
    SPLIT_TAG,
    build_split,
    default_split_path,
)

DEFAULT_ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="tokenized split root (holds tok_index.parquet)")
    ap.add_argument("--metadata", default=DEFAULT_METADATA_PATH, help="val_metadata.parquet")
    ap.add_argument("--radius-km", type=float, default=SPLIT_RADIUS_KM)
    ap.add_argument("--fraction", type=float, default=EVAL_GROUP_FRACTION)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--tag", default=SPLIT_TAG)
    ap.add_argument("--out", default=None, help="default: eo/data/eval_rows_<tag>.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing artifact")
    args = ap.parse_args()

    out = Path(args.out) if args.out else default_split_path(args.tag)
    if out.exists() and not args.force and not args.dry_run:
        print(
            f"refusing to overwrite {out}\n"
            f"  An existing split may already have checkpoints trained against it; "
            f"replacing it invalidates every comparison made on them.\n"
            f"  Pass --force if that is genuinely what you want.",
            file=sys.stderr,
        )
        return 2

    doc = build_split(
        root_dir=args.root,
        metadata_path=args.metadata,
        radius_km=args.radius_km,
        eval_fraction=args.fraction,
        seed=args.seed,
        tag=args.tag,
    )

    print(f"group key    : single-linkage spatial, {doc['group_key']['radius_km']} km, corpus-agnostic")
    print(f"groups       : {doc['groups_held_out']} held out of {doc['groups_total']} "
          f"({100 * doc['groups_held_out'] / doc['groups_total']:.2f}%)")
    for name, c in doc["strata"].items():
        print(f"  stratum {name:<18} {c['groups_held_out']:>5} / {c['groups_total']:>5}")
    print(f"rows         : {doc['n_eval_rows']} held out of {doc['n_total_rows']} "
          f"({100 * doc['n_eval_rows'] / doc['n_total_rows']:.2f}%)")
    for corpus, c in doc["counts_per_corpus"].items():
        print(f"  {corpus:<12} {c['eval_rows']:>6} / {c['total_rows']:>6} rows")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # Write through a temp file and os.replace: a write that raises partway
    # must not destroy an artifact that may have checkpoints trained against it
    # (worklog 2026-09-21 -- this emptied worklog.md once already).
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    payload = json.dumps(doc, indent=2) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, out)
    print(f"\nwrote {out} ({len(payload)} bytes)")
    print("⚠ commit this file: the split is a one-way door once an arm trains against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
