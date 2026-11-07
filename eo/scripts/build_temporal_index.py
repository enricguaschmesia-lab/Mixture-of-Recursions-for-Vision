#!/usr/bin/env python
"""Build and check the location and date index of a token root (A1, Step 2). Thin CLI over
eo/data/temporal_index.py, which holds the design.

    python eo/scripts/build_temporal_index.py build --root /data/enric/data/TerraMesh/val_clean
    python eo/scripts/build_temporal_index.py check --root /data/enric/data/TerraMesh/val_clean

    python eo/scripts/build_temporal_index.py build --root /data/enric/data/TerraMesh/train_coords \\
        --metadata /data/enric/data/TerraMesh/train_meta/train_metadata.parquet

`build` needs the metadata and `check` needs the tokens. Train's tokens are on g49 only, so its check
runs in a pod, on the arrays built here, with --metadata naming g49's copy so that both hashes are
compared. `check --only Coords` covers what a partial root holds; that report is not complete, and
load_temporal_index refuses it. Exit status 1 means a check failed or a control did not fire.

⚠ A location's id is its line in locations.npy, so `build` refuses to overwrite without --force.

Runs in the repo `.venv`, and in `mor-eo` on RCP.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eo.data.temporal_index import build_index, check_index, index_dir, write_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build", "check"))
    ap.add_argument("--root", required=True, help="token root (holds tok_index.parquet)")
    ap.add_argument("--metadata", default=None, help="metadata parquet; default TERRAMESH_METADATA_PATH or val's")
    ap.add_argument("--only", nargs="+", default=None, help="check: limit the token checks to these modalities")
    ap.add_argument("--force", action="store_true", help="build: overwrite an existing index")
    args = ap.parse_args()

    if args.command == "build":
        locations, dates, doc = build_index(args.root, args.metadata)
        out = write_index(args.root, locations, dates, doc, force=args.force)
        print(json.dumps({k: doc[k] for k in ("counts", "structure", "structure_controls", "split")}, indent=1))
        print(f"wrote {out}: locations {locations.shape}, dates {dates.shape}")
        return 0

    report = check_index(args.root, args.only, args.metadata)
    print(json.dumps(report, indent=1))
    print(f"wrote {index_dir(args.root)}: passed {report['passed']}, complete {report['complete']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
