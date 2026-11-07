#!/usr/bin/env python
"""Why DEM tokens differ between dates of one location on train (A1, Step 2). `.venv` / `mor-eo`, CPU.

The check in eo/data/temporal_index.py asked for identical DEM tokens within a location. Val passes,
and so do the 62,829 locations with two rows in the 24 ssl4eos12 shards kept on the workstation. All
of train, on g49, does not: 1,598 of 239,544 locations differ from their date 0, by 2,675 of 140.9M
tokens. A row from another location would change most of a block, so this asks what the differences are.

Train was tokenized on two kinds of card, and tokens are not bit-identical across them (`CLAUDE.md`,
"An ordinary cross-card difference is ~1e-4"). Each claim below can fail on these data:

  A. BOTH CARDS.   Pairs of rows encoded on one card never differ, and only pairs across cards do.
  B. BIN FLIPS.    A differing token moves one FSQ digit by one level, as a value at a bin boundary
                   does. DEM's codebook is 8-8-8-6-5.
  C. SEPARATION.   Rows of one location differ by a handful of tokens, and rows of two random
                   locations by far more, so a limit between them still sees a misplaced row.

CONTROL for B: digits of a differing pair against those of two tokens drawn from the same block, which
are not bin flips. CONTROL for C: the same random pairs at different places are the distribution a
misplaced row would come from, and the swapped locations of the gate's control must exceed the limit.

Per-pair counts go to --out, so that nothing here has to be recomputed to be read.

    python eo/experiments/temporal/dem_dates.py --root <token root> --done <dir of done/*.json> --out <dir>

Runs on g49: the DEM tokens of train are there, and so are the shards' records. Nothing is written
to the root.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eo.data.temporal_index import CONTROL_SEED, load_temporal_index, swap_rows  # noqa: E402
from eo.terramesh_tok.contract import tok_dir_name  # noqa: E402

LEVELS = (8, 8, 8, 6, 5)                     # FSQ of the image tokenizers, 15,360 codes
BASIS = np.cumprod((1,) + LEVELS[:-1])
PAIRS = tuple(itertools.combinations(range(4), 2))
N_RANDOM = 200_000


def digits(ids: np.ndarray) -> np.ndarray:
    """(..., 5) FSQ levels of each token id."""
    return (ids[..., None].astype(np.int64) // BASIS) % np.asarray(LEVELS)


def shard_cards(done: Path, table: pd.DataFrame) -> dict:
    """shard -> the card that encoded it, from the record each shard wrote when it finished."""
    cards = {}
    for s in table.source_shard.unique():
        rec = json.loads((done / str(s).replace(".tar", ".json")).read_text())
        cards[s] = rec["gpu"]
    return cards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--done", required=True, help="the tokenization's done/ directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = load_temporal_index(root, require_checked=False)
    loc = idx.locations
    table = pd.read_parquet(root / "tok_index.parquet", columns=["row", "source_shard", "corpus"])
    if not (table.row.to_numpy() == np.arange(len(table))).all():
        raise RuntimeError("tok_index.parquet is not in row order")

    cards = shard_cards(Path(args.done), table[table.corpus == "ssl4eos12"])
    names = sorted(set(cards.values()))
    code = {s: names.index(c) for s, c in cards.items()}
    card_of_row = table.source_shard.map(code).to_numpy(dtype=np.float64)     # NaN on Major TOM shards
    print("cards", names, {n: sum(c == n for c in cards.values()) for n in names}, flush=True)

    lo = int(loc.min())
    dem = np.asarray(np.load(root / tok_dir_name("DEM") / "tokens.npy", mmap_mode="r")[lo: int(loc.max()) + 1])
    x = dem[loc - lo]                                                          # (L, 4, 196)
    card = card_of_row[loc].astype(np.int64)                                   # (L, 4)

    # All six date pairs of each location.
    count = np.stack([(x[:, a] != x[:, b]).sum(axis=1) for a, b in PAIRS], axis=1)     # (L, 6)
    cross = np.stack([card[:, a] != card[:, b] for a, b in PAIRS], axis=1)
    np.savez_compressed(out / "dem_dates_pairs.npz", count=count.astype(np.uint8), cross=cross, card=card.astype(np.uint8),
                        pairs=np.asarray(PAIRS))

    # A. By card.
    a = {"pairs_one_card": int((~cross).sum()), "pairs_across_cards": int(cross.sum()),
         "differing_pairs_one_card": int((count[~cross] > 0).sum()),
         "differing_pairs_across_cards": int((count[cross] > 0).sum()),
         "tokens_differing_one_card": int(count[~cross].sum()),
         "tokens_differing_across_cards": int(count[cross].sum()),
         "tokens_compared_one_card": int((~cross).sum() * 196),
         "tokens_compared_across_cards": int(cross.sum() * 196),
         "locations_all_on_one_card": int((card == card[:, :1]).all(axis=1).sum())}
    a["rate_across_cards"] = a["tokens_differing_across_cards"] / max(1, a["tokens_compared_across_cards"])
    a["rate_one_card"] = a["tokens_differing_one_card"] / max(1, a["tokens_compared_one_card"])
    # A location's dates split between cards the same way: same-card pairs of a card-crossing location.
    a["locations_differing_all_on_one_card"] = int(((count > 0).any(axis=1) & (card == card[:, :1]).all(axis=1)).sum())

    # B. Every differing token, digit by digit, against two tokens of the same block.
    rows, flips, shuffled = [], [], []
    rng = np.random.default_rng(CONTROL_SEED)
    for j, (p, q) in enumerate(PAIRS):
        hit_l, hit_t = np.nonzero(x[:, p] != x[:, q])
        u, v = x[hit_l, p, hit_t], x[hit_l, q, hit_t]
        d = np.abs(digits(u) - digits(v))
        flips.append(d)
        rows.append(np.stack([hit_l, np.full_like(hit_l, j), hit_t, u, v], axis=1))
        other = rng.integers(0, 196, size=len(hit_l))
        shuffled.append(np.abs(digits(u) - digits(x[hit_l, p, other])))
    flips, shuffled = np.concatenate(flips), np.concatenate(shuffled)
    np.save(out / "dem_dates_tokens.npy", np.concatenate(rows))
    l1, moved = flips.sum(axis=1), (flips > 0).sum(axis=1)
    b = {"tokens": int(len(flips)),
         "one_digit_one_level": int(((moved == 1) & (l1 == 1)).sum()),
         "share_one_digit_one_level": float(((moved == 1) & (l1 == 1)).mean()),
         "digits_moved": {int(k): int(v) for k, v in zip(*np.unique(moved, return_counts=True))},
         "l1_distance": {int(k): int(v) for k, v in zip(*np.unique(l1, return_counts=True))},
         "control_same_block_share_one_digit_one_level": float(((shuffled > 0).sum(axis=1) == 1).__and__(shuffled.sum(axis=1) == 1).mean())}

    # C. How far true rows sit from misplaced ones.
    r = np.random.default_rng(CONTROL_SEED)
    i, j = r.integers(0, len(loc), N_RANDOM), r.integers(0, len(loc), N_RANDOM)
    keep = i != j
    between = (x[i[keep], 0] != x[j[keep], 0]).sum(axis=1)
    worst = count.max(axis=1)
    quant = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5)
    c = {"within_location": {"pairs": int(count.size), "pairs_differing": int((count > 0).sum()),
                             "max_tokens_in_a_pair": int(count.max()),
                             "pairs_by_tokens": {int(k): int(v) for k, v in zip(*np.unique(count, return_counts=True))},
                             "locations_by_worst_pair": {int(k): int(v) for k, v in zip(*np.unique(worst, return_counts=True))}},
         "between_random_locations": {"pairs": int(keep.sum()), "identical": int((between == 0).sum()),
                                      "quantiles": {str(q): float(np.quantile(between, q)) for q in quant},
                                      "share_at_or_below": {str(k): float((between <= k).mean())
                                                            for k in (0, 1, 2, 4, 8, 16, 32, 64)}}}
    # The gate's control, read per location: each swapped location's worst pair, one modality.
    swapped = swap_rows(loc, 1197)
    moved_lines = np.nonzero((swapped != loc).any(axis=1))[0]
    y = dem[swapped[moved_lines] - lo]
    sw = np.stack([(y[:, p] != y[:, q]).sum(axis=1) for p, q in PAIRS], axis=1).max(axis=1)
    c["swapped_locations"] = {"locations": int(len(moved_lines)),
                              "worst_pair_quantiles": {str(q): float(np.quantile(sw, q)) for q in quant},
                              "share_at_or_below": {str(k): float((sw <= k).mean()) for k in (0, 1, 2, 4, 8, 16, 32, 64)}}

    report = {"root": str(root), "cards": {n: sum(v == n for v in cards.values()) for n in names},
              "both_cards": a, "bin_flips": b, "separation": c}
    (out / "dem_dates.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
