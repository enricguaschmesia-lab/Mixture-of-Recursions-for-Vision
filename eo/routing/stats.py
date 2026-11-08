# eo/routing/stats.py
"""Statistics for routing audits (Paper B, B0; docs/plans/b0_router_audit.md).

WHY THIS EXISTS. Two defaults fail on routing data, and both failed in practice:

  ROWS ARE NOT THE UNIT. The 4,416 held-out rows sit in 256 spatial groups (5 km single-linkage,
  eo/data/eval_split.py), 16 rows to a group at the median. Measured 2026-11-07: an interval on a
  contrast between depths is up to 1.94x wider over groups than over rows. Resample groups.

  A RESAMPLE CANNOT SEE SEED VARIATION. A per-modality mean over 800,000 tokens is so precise that
  resampling rows never swaps two modalities, so a bootstrap of the cross-seed rank correlation
  has zero width ([+0.83, +0.83]). Agreement between seeds is tested by exact permutation of the
  modality labels, which is all two seeds and six modalities can support: the two-sided 5%
  critical value at n = 6 is 0.886.

Numpy only. Runs in `.venv`, and in `mor`.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional, Sequence, Tuple

import numpy as np

ROOT = "/data/enric/data/TerraMesh/val"


def rank(x) -> np.ndarray:
    """AVERAGE ranks for ties, Spearman's definition (see curves._rank for why it matters)."""
    x = np.asarray(x, dtype=np.float64)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def spearman(a, b) -> float:
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _perms(n: int) -> np.ndarray:
    if n > 9:
        raise ValueError(f"{n}! permutations: exact enumeration stops at 9")
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


def exact_p(a, b) -> float:
    """Two-sided p of Spearman's rho under every permutation of b's labels."""
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    za, zb = (ra - ra.mean()) / ra.std(), (rb - rb.mean()) / rb.std()
    null = (zb[_perms(len(za))] * za).mean(1)
    obs = float((za * zb).mean())
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def critical_rho(n: int, alpha: float = 0.05) -> float:
    """The smallest |rho| on untied ranks with two-sided P(|rho| >= it) <= alpha."""
    z = (np.arange(n) - (n - 1) / 2.0)
    z = z / z.std()
    vals = np.sort(np.round(np.abs((z[_perms(n)] * z).mean(1)), 10))
    uniq = np.unique(vals)
    for r in uniq:
        if np.mean(vals >= r - 1e-9) <= alpha:
            return float(r)
    return float("nan")


def series_p(A, B) -> Tuple[float, float]:
    """(mean rho, p) over a series of checkpoints. A and B are (checkpoints, modalities).

    One permutation of B's modality labels is applied at EVERY checkpoint. Checkpoints of one run
    are correlated, so ten per-checkpoint p values are not ten tests; this gives one p for "the
    two seeds' orders agree over training". A checkpoint with a constant column is skipped.
    """
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    keep = [i for i in range(len(A)) if rank(A[i]).std() > 0 and rank(B[i]).std() > 0]
    Ra = np.stack([rank(A[i]) for i in keep])
    Rb = np.stack([rank(B[i]) for i in keep])
    Za = (Ra - Ra.mean(1, keepdims=True)) / Ra.std(1, keepdims=True)
    Zb = (Rb - Rb.mean(1, keepdims=True)) / Rb.std(1, keepdims=True)
    perms = _perms(A.shape[1])
    null = np.einsum("ti,pti->p", Za, Zb[:, perms].transpose(1, 0, 2)) / (Za.size)
    obs = float((Za * Zb).mean())
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def eval_groups(rows: Sequence[int], root: str = ROOT) -> np.ndarray:
    """The spatial group of each row, as 0..G-1, from the split's own clustering."""
    from eo.data.eval_split import load_row_table, spatial_groups
    t = load_row_table(root)
    g = np.asarray(spatial_groups(t["center_lat"].values, t["center_lon"].values))
    return np.unique(g[np.asarray(rows, dtype=np.int64)], return_inverse=True)[1]


def boot_weights(n_units: int, n_draws: int = 2000, seed: int = 0) -> np.ndarray:
    """(draws, units) resampling counts."""
    return np.random.default_rng(seed).multinomial(
        n_units, [1.0 / n_units] * n_units, size=n_draws).astype(np.float64)


def sum_by(x: np.ndarray, units: np.ndarray) -> np.ndarray:
    """(rows, ...) -> (units, ...), summed."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((int(units.max()) + 1,) + x.shape[1:])
    np.add.at(out, units, x)
    return out


def ratio_boot(num, den, units: Optional[np.ndarray] = None, n_draws: int = 2000,
               seed: int = 0, q: Sequence[float] = (2.5, 97.5)):
    """A ratio of sums (a mean over tokens from per-row sums), with a bootstrap over `units`.

    num, den: (rows,) or (rows, K). Returns (estimate, lower, upper), one per K. `units` None
    resamples rows.
    """
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    units = np.arange(len(num)) if units is None else np.asarray(units)
    N, D = sum_by(num, units), sum_by(den, units)
    W = boot_weights(len(N), n_draws, seed)
    draws = (W @ N) / (W @ D)
    lo, hi = np.percentile(draws, q, axis=0)
    return N.sum(0) / D.sum(0), lo, hi


def lattice(depths: Sequence[int], n_slots: int, mean: float) -> np.ndarray:
    """Every table of one depth per slot with exactly this mean depth: (tables, slots), depths.

    Exact on purpose. A tolerance on compute lets a table in that beats uniform by spending
    more; this is where "beats uniform by at most 0.0003" came from (2026-11-05). Slots must hold
    equal token shares for equal mean to mean equal compute: pool S1GRD and S1RTC, leave Coords out.
    """
    target = mean * n_slots
    if not math.isclose(target, round(target)):
        raise ValueError(f"mean {mean} over {n_slots} slots is not reachable by integers")
    return np.array([t for t in itertools.product(depths, repeat=n_slots)
                     if sum(t) == round(target)], dtype=np.int64)
