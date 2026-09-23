#!/usr/bin/env python
"""Generated vocabulary ids -> decoder-ready token grids (Phase 3 Step 7, side A).

⚠ THIS IS HALF A STEP. Step 7 is inherently cross-environment
(PHASE2_REPORT.md section 7 item 7, plan Step 7 preamble): the TerraMind DiVAE
decoders need terratorch and live in the `mor` conda env, the model and its
vocabulary live in `.venv`, and the repo pins transformers==4.52.4 against a
floor terratorch will not accept. They cannot share a process.

    .venv  ->  this script   ->  *.npy on disk  ->  decode_eo.py  ->  `mor`

So this script does every part of the job that needs `eo.data.eo_vocab`, and
decode_eo.py does every part that needs a decoder. The handoff carries LOCAL
codebook ids, never global vocabulary ids, which is what lets the `mor` side
import nothing from `eo/data/` at all -- its only shared import is
`eo/terramesh_tok/contract.py`, which is pure constants.

WHAT IT REPAIRS, AND WHY THE REPAIR IS RECORDED RATHER THAN HIDDEN.
A generated sequence is not automatically a decodable grid. Three things go
wrong, all of them normal model behaviour rather than bugs:

  * OFF-SLOT IDS. Only the target's slot is valid (4,375 ids for LULC, 15,360
    for an image modality, out of 87,556). An UNTRAINED model sits at chance,
    so ~95% of an unconstrained LULC body is un-decodable. That is the point
    of the control -- it must not be silently dropped.
  * SHORT BODIES. A model that emits <EO> early gives fewer than 196 tokens.
  * LONG BODIES. A model that never emits <EO> runs to max_new_tokens = 197.

Each becomes local code `FILL_CODE` and is marked False in the valid mask. The
mask ships beside the grid so every downstream number can be reported next to
the fraction of the grid that was invented. A repair that cannot be seen in
the metrics is a repair that flatters the model.

⚠ FILL_CODE IS NOT A NEUTRAL CHOICE and there is no neutral choice available:
any code decodes to *something*. Code 0 is used because it is deterministic,
carries no ground-truth information (picking the modal training code would
leak the answer into the control), and makes a heavily-repaired scene visually
obvious -- a near-uniform decode is the honest rendering of a model that did
not produce decodable tokens.

Runs in `.venv`. It never imports terratorch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from eo.data.eo_vocab import MODALITIES, PAD_ID, get_modality  # noqa: E402
from eo.terramesh_tok import contract as C  # noqa: E402

#: Local codebook id substituted wherever a generated token cannot be decoded.
FILL_CODE = 0


def _bodies(arr: np.ndarray, target: str) -> list[list[int]]:
    """(N, W) PAD-filled generation array -> one body of ids per scene.

    Undoes `eo.generate.conditional.save`: strip the PAD tail, then strip the
    trailing <EO_target> if the model emitted one. Anything left is the body,
    at whatever length the model actually produced.
    """
    info = get_modality(target)
    out = []
    for r in np.asarray(arr):
        ids = [int(t) for t in r]
        while ids and ids[-1] == PAD_ID:
            ids.pop()
        if ids and ids[-1] == info.eo_id:
            ids.pop()
        out.append(ids)
    return out


def to_grids(arr: np.ndarray, target: str):
    """Global vocabulary ids -> (N, GRID, GRID) local ids + (N, T) valid mask.

    The reshape is row-major, per `contract.FLATTEN_ORDER`: token k is patch
    (k // GRID, k % GRID). Phase 5's patch<->token mapping depends on it, and
    a transposed grid would decode to a plausible-looking wrong image.
    """
    info = get_modality(target)
    lo, size = info.codebook_offset, info.codebook_size
    n_tok = info.tokens_per_sample
    if n_tok != C.TOKENS_PER_SAMPLE:
        raise ValueError(
            f"{target} carries {n_tok} tokens/sample but the contract's image "
            f"grid is {C.TOKENS_PER_SAMPLE}. Only the six image modalities "
            f"decode to pixels; Coords has no decoder."
        )

    bodies = _bodies(arr, target)
    n = len(bodies)
    grids = np.full((n, n_tok), FILL_CODE, dtype=np.int64)
    valid = np.zeros((n, n_tok), dtype=bool)
    stats = {"off_slot": 0, "short": 0, "long": 0, "tokens": n * n_tok}

    for i, body in enumerate(bodies):
        if len(body) < n_tok:
            stats["short"] += 1
        elif len(body) > n_tok:
            stats["long"] += 1
        for k, t in enumerate(body[:n_tok]):
            if lo <= t < lo + size:
                grids[i, k] = t - lo
                valid[i, k] = True
            else:
                stats["off_slot"] += 1

    stats["repaired_tokens"] = int((~valid).sum())
    stats["repaired_fraction"] = round(float((~valid).mean()), 6)
    return grids.reshape(n, C.GRID, C.GRID), valid, stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", required=True,
                    help="a mode directory written by generate_eo.py, e.g. "
                         ".../untrained_arm_a_LULC/slot_masked")
    ap.add_argument("--target", default=None,
                    help="default: inferred from the generated_<T>.npy present")
    ap.add_argument("--out", default=None, help="default: --gen-dir")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    target = args.target
    if target is None:
        hits = sorted(gen_dir.glob("generated_*.npy"))
        if len(hits) != 1:
            return _die(f"expected exactly one generated_*.npy in {gen_dir}, "
                        f"found {[h.name for h in hits]}; pass --target")
        target = hits[0].stem[len("generated_"):]
    if target not in MODALITIES:
        return _die(f"{target!r} is not an EO modality: {sorted(MODALITIES)}")

    gen_path = gen_dir / f"generated_{target}.npy"
    rows_path = gen_dir / f"rows_{target}.npy"
    stats_path = gen_dir / f"stats_{target}.json"
    for p in (gen_path, rows_path):
        if not p.exists():
            return _die(f"missing {p}")

    arr = np.load(gen_path)
    rows = np.load(rows_path)
    if len(arr) != len(rows):
        return _die(f"{len(arr)} generated scenes against {len(rows)} rows")

    grids, valid, stats = to_grids(arr, target)

    out = Path(args.out or gen_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"decode_grids_{target}.npy", grids)
    np.save(out / f"decode_valid_{target}.npy", valid)
    np.save(out / f"decode_rows_{target}.npy", rows.astype(np.int64))

    info = get_modality(target)
    manifest = {
        "target": target,
        "n_scenes": int(len(grids)),
        "grid": [C.GRID, C.GRID],
        "crop": C.CROP,
        "tok_dir": C.tok_dir_name(target),
        "codebook_size": info.codebook_size,
        "codebook_offset": info.codebook_offset,
        "flatten_order": C.FLATTEN_ORDER,
        "fill_code": FILL_CODE,
        "repair": stats,
        "source_dir": str(gen_dir),
        "generation": json.loads(stats_path.read_text()) if stats_path.exists() else None,
    }
    (out / f"decode_manifest_{target}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"target            : {target}")
    print(f"scenes            : {len(grids)}")
    print(f"grid              : {C.GRID}x{C.GRID} local ids, codebook {info.codebook_size}")
    print(f"off-slot tokens   : {stats['off_slot']} / {stats['tokens']}")
    print(f"short / long body : {stats['short']} / {stats['long']} scenes")
    print(f"repaired          : {stats['repaired_tokens']} tokens "
          f"({100 * stats['repaired_fraction']:.2f}%) -> local code {FILL_CODE}")
    print(f"-> {out}")
    return 0


def _die(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
