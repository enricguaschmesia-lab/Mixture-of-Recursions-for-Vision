#!/usr/bin/env python
"""Draw the dev/test split of val's four-date locations (A1, Step 2). Thin CLI over
eo/data/temporal_split.py, which holds the design.

    python eo/scripts/build_temporal_split.py            # write eo/data/temporal_rows_v1.json
    python eo/scripts/build_temporal_split.py --dry-run  # print the summary, write nothing

⚠ COMMITTED, and a one-way door once anything is selected on dev: redrawing it would put
selected-on locations into test. So it refuses to overwrite without --force.

Runs in the repo `.venv`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eo.data.temporal_split import SPLIT_TAG, build_split, load_temporal_split, split_path  # noqa: E402

DEFAULT_ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="tokenized split root (holds tok_index.parquet)")
    ap.add_argument("--tag", default=SPLIT_TAG)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = ap.parse_args()

    doc = build_split(args.root, tag=args.tag)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("dev", "test")}, indent=2))
    if args.dry_run:
        return 0
    path = split_path(args.tag)
    if path.exists() and not args.force:
        print(f"REFUSED: {path} exists, and redrawing it moves dev locations into test.")
        return 1
    path.write_text(json.dumps(doc, indent=1) + "\n")
    load_temporal_split(args.root, args.tag)
    print(f"wrote {path}, reloaded and checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
