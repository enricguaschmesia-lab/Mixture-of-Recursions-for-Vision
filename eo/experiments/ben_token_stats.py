"""PHASE6_PLAN Step 1.3, last bullet: the distribution shift, measured instead of guessed (`.venv`, CPU).

BEN's 120 px patches are upsampled to 224 (10 m -> ~5.4 m GSD), which no tokenizer saw. Compare
the TOKEN distributions on an equal number of patches:

  codes used, entropy (nats), and the Jensen-Shannon divergence between code histograms,
  BEN (ft_train) vs TerraMesh val, for S2L2A and S1GRD (TerraMesh S1GRD = its 8,704 ssl4eos12 rows).

The floor that makes a JSD readable: two DISJOINT random halves of the same set (TerraMesh vs
TerraMesh, BEN vs BEN). A shift is the excess over that floor.

Geography confounds it: BEN is ten European countries, 2017-18; TerraMesh is global. So BEN is
also compared with TerraMesh rows inside Europe's bounding box (lat 36..71, lon -11..32), which
removes most of the geographic part and leaves resolution, season and sensor processing.

    python eo/experiments/ben_token_stats.py --n 4000 --out /data/enric/reports/ben/step1_token_stats.json
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BEN = Path("/data/enric/data/BigEarthNet")
TM = Path("/data/enric/data/TerraMesh/val")
CODEBOOK = 15360


def hist(tok: np.ndarray) -> np.ndarray:
    return np.bincount(tok.ravel().astype(np.int64), minlength=CODEBOOK).astype(np.float64)


def entropy(h):
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum())


def jsd(h1, h2):
    p, q = h1 / h1.sum(), h2 / h2.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="patches per side of each comparison")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    idx = pd.read_parquet(BEN / "index_v1.parquet")
    res = {"n": a.n}
    for mod in ("S2L2A", "S1GRD"):
        bt = np.load(BEN / f"{mod}_tok224/tokens.npy", mmap_mode="r")
        bp = np.load(BEN / f"{mod}_tok224/present.npy")
        tt = np.load(TM / f"{mod}_tok224/tokens.npy", mmap_mode="r")
        tp = np.load(TM / f"{mod}_tok224/present.npy")
        b_rows = rng.permutation(np.flatnonzero(bp & idx.ft_train.to_numpy(bool)))[: 2 * a.n]
        t_rows = rng.permutation(np.flatnonzero(tp))[: 2 * a.n]
        assert len(b_rows) == 2 * a.n and len(t_rows) == 2 * a.n, (mod, len(b_rows), len(t_rows))
        b1, b2 = hist(bt[np.sort(b_rows[: a.n])]), hist(bt[np.sort(b_rows[a.n:])])
        t1, t2 = hist(tt[np.sort(t_rows[: a.n])]), hist(tt[np.sort(t_rows[a.n:])])
        r = {
            "codes_used_ben": int((b1 > 0).sum()), "codes_used_terramesh": int((t1 > 0).sum()),
            "entropy_ben": entropy(b1), "entropy_terramesh": entropy(t1),
            "jsd_ben_vs_terramesh": jsd(b1, t1),
            "jsd_floor_terramesh_halves": jsd(t1, t2), "jsd_floor_ben_halves": jsd(b1, b2),
            "top100_overlap": len(set(np.argsort(-b1)[:100]) & set(np.argsort(-t1)[:100])),
            "share_of_ben_tokens_on_codes_terramesh_never_used": float(b1[(t1 + t2) == 0].sum() / b1.sum()),
        }
        # Europe-only TerraMesh rows
        from eo.data.eval_split import load_row_table
        rt = load_row_table(TM).set_index("row")
        eu = (rt.center_lat.between(36, 71) & rt.center_lon.between(-11, 32)).reindex(np.arange(len(tp))).fillna(False).to_numpy(bool)
        e_rows = rng.permutation(np.flatnonzero(tp & eu))
        ne = min(a.n, len(e_rows) // 2)
        e1, e2 = hist(tt[np.sort(e_rows[:ne])]), hist(tt[np.sort(e_rows[ne:2 * ne])])
        bE = hist(bt[np.sort(b_rows[:ne])])
        r.update({"n_europe": int(ne), "entropy_terramesh_europe": entropy(e1),
                  "jsd_ben_vs_terramesh_europe": jsd(bE, e1), "jsd_floor_europe_halves": jsd(e1, e2),
                  "jsd_europe_vs_terramesh_all": jsd(e1, hist(tt[np.sort(t_rows[:ne])]))})
        res[mod] = r
        print(mod, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
