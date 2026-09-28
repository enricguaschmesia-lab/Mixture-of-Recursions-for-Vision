#!/usr/bin/env python
"""D3.11's "unconditioned" reference: another scene's ground truth, scored against this one.

    python eo/scripts/wrong_scene_control.py --target LULC        # .venv
    python eo/scripts/prepare_decode.py --gen-dir <out>/slot_masked --target LULC
    python eo/scripts/decode_eo.py --decode-dir <out>/slot_masked --target LULC ...   # mor

Takes the identity control's ground-truth bodies for the same 64 corpus-
stratified scenes the sweep uses, and hands scene i the tokens of scene
pi(i) for a fixed random DERANGEMENT pi (seed 0), keeping scene i's rows --
so the decode path scores a real, well-formed, in-distribution scene that
simply is not the right one. Its `generated` metric is what a model that
ignores its context but produces realistic output would score.

⚠ Why this and not only the shuffled control. The shuffled control keeps the
RIGHT scene's token multiset and scrambles the layout, so it already knows the
scene's class proportions (LULC) and value distribution (DEM level). A
generation below it can still be using its context; a generation no better
than THIS control is not.

Arm-independent: it depends on the scenes, not on a model. Runs in `.venv`.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402


def derangement(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--identity-dir", default=None,
                    help="default /data/enric/generations/d311/controls/identity/<T>")
    ap.add_argument("--out", default=None, help="default /data/enric/generations/d311/controls/wrong_scene/<T>")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    T = args.target
    src = Path(args.identity_dir or f"/data/enric/generations/d311/controls/identity/{T}") / "slot_masked"
    out = Path(args.out or f"/data/enric/generations/d311/controls/wrong_scene/{T}") / "slot_masked"
    out.mkdir(parents=True, exist_ok=True)

    gen = np.load(src / f"generated_{T}.npy")
    rows = np.load(src / f"rows_{T}.npy")
    p = derangement(len(rows), args.seed)
    with open(out / f"generated_{T}.npy", "wb") as fh:
        np.save(fh, gen[p])                              # scene i gets scene p[i]'s truth
    np.save(out / f"rows_{T}.npy", rows)                 # ...scored against scene i
    stats = json.loads((src / f"stats_{T}.json").read_text())
    stats["provenance"] = {"control": "wrong_scene", "derangement_seed": args.seed,
                           "permutation": p.tolist(), "identity_dir": str(src)}
    (out / f"stats_{T}.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"{T}: {len(rows)} scenes, each scored against another scene's ground truth -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
