#!/usr/bin/env python
"""Does S1 at date t help predict S2 at date t more than S1 at another date? (A1, Step 1.) `.venv`, GPU.

The dates in s1_dates.py come from product ids. This asks the images: if a row's S1 really belongs to
its date, teacher-forced CE(S2 at t | S1 at t) is below CE(S2 at t | S1 at another date, same place).

⚠ B10c, NOT T360. T360 trained on exactly these same-row pairs, so it would favour them whatever the
dates. B10c trained on val alone: it has seen no train row, and not the 96 val locations of
eval_rows_v1, which are the positive control.

VAL: every (S2 at t, S1 at o) on the 96 held-out locations, where the dates are known to be right.
This is the positive control, and the one place S1 can be moved knowingly. CONTROL, which must fire:
if each row's S1 sat one date along, the same test would compare S1(t + 1) with S1(t + 3), and the
effect must go, so the paired difference from the true contrast must stay above 0. The first
control swapped two rows of train, which only negated its own number and could not fail.

TRAIN: pairs of rows of one location that both sit in the 24 ssl4eos12 token shards kept on the
workstation (train_tok/). The effect grows with the date gap, and the two sides mix gaps differently
(val's other date is always half a year), so both are reported by date step.

Other place: S1 of the next location, which scales the date effect against what S1 is worth at all.
On val it keeps the date index. On train it does not (under half of the pairs match), so that
column mixes place and season. Bootstrap over 5 km spatial groups; per-pair CE stays in --out's npz.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(.venv/bin/python -m eo.train.gpu titanx) \\
      .venv/bin/python -X faulthandler -u eo/experiments/temporal/s1_functional.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from eo.data.eo_vocab import get_modality  # noqa: E402
from eo.data.eval_split import SPLIT_RADIUS_KM, load_eval_rows, spatial_groups  # noqa: E402
from eo.data.temporal_split import load_temporal_split, location_table  # noqa: E402
from eo.terramesh_tok.contract import TOKENS_PER_SAMPLE, tok_dir_name  # noqa: E402

CKPT = "/data/enric/runs/pretrain/phase3/vanilla10c_20261011_172253/checkpoint-3300"
ARM = "eo_terramesh/arm_b10c_vanilla"
TRAIN = Path("/data/enric/data/TerraMesh/train_tok")
META = Path("/data/enric/data/TerraMesh/train_meta/train_metadata.parquet")
VAL = "/data/enric/data/TerraMesh/val"
OUT = Path("/data/enric/reports/temporal/s1_functional.json")
PAIRS = OUT.with_name("s1_functional_pairs.npz")
N_TRAIN, SEED, DRAWS, BATCH = 500, 20261105, 10_000, 16


def chunk(body: np.ndarray, modality: str) -> torch.Tensor:
    """[BO, body, EO] in the unified vocabulary, as TerraMeshTokenDataset._load_chunk."""
    info = get_modality(modality)
    b = torch.from_numpy(np.asarray(body, dtype=np.int64)) + info.codebook_offset
    return torch.cat([torch.tensor([info.bo_id]), b, torch.tensor([info.eo_id])])


class TrainTokens:
    """Tokens of the shards kept locally, by row. Each shard file is in tar order."""

    def __init__(self):
        idx = pd.read_parquet(TRAIN / "tok_index.parquet", columns=["row", "stem", "source_shard", "corpus"])
        self.idx = idx[idx.corpus == "ssl4eos12"].sort_values("row").copy()
        self.idx["pos"] = self.idx.groupby("source_shard").cumcount()
        self.shards = sorted(p.name[:-4] for p in (TRAIN / tok_dir_name("S1GRD") / "shards").glob("*.npy")
                             if not p.name.endswith(".nan.npy"))
        self.where = {r: (s, int(p)) for r, s, p in zip(self.idx.row, self.idx.source_shard, self.idx.pos)
                      if s[:-4] in self.shards}
        self.arrays = {}

    def body(self, modality: str, row: int) -> np.ndarray:
        shard, pos = self.where[int(row)]
        key = (modality, shard)
        if key not in self.arrays:
            self.arrays[key] = np.load(TRAIN / tok_dir_name(modality) / "shards" / f"{shard[:-4]}.npy", mmap_mode="r")
        return self.arrays[key][pos]


class ValTokens:
    def __init__(self, ds):
        self.ds = ds

    def body(self, modality: str, row: int) -> np.ndarray:
        return self.ds._token_arrays()[modality][int(row)]


@torch.no_grad()
def target_ce(model, seqs: list) -> np.ndarray:
    ce = []
    for k in range(0, len(seqs), BATCH):
        x = torch.stack(seqs[k:k + BATCH]).cuda()
        lg = model(input_ids=x, use_cache=False, logits_to_keep=TOKENS_PER_SAMPLE + 2).logits[:, :TOKENS_PER_SAMPLE].float()
        ce.append(F.cross_entropy(lg.transpose(1, 2), x[:, -(TOKENS_PER_SAMPLE + 1):-1], reduction="none").mean(1))
    return torch.cat(ce).double().cpu().numpy()


def boot(v: np.ndarray, unit: np.ndarray, seed: int = 0) -> dict:
    u, inv = np.unique(unit, return_inverse=True)
    s, n = np.bincount(inv, v), np.bincount(inv).astype(float)
    pick = np.random.default_rng(seed).integers(0, len(u), (DRAWS, len(u)))
    b = s[pick].sum(1) / n[pick].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def ce_of(model, tok, s1_rows, s2_rows) -> np.ndarray:
    """CE of S2L2A given one S1GRD block, per pair, in the order given."""
    tgt = [chunk(tok.body("S2L2A", int(r)), "S2L2A") for r in s2_rows]
    src = [chunk(tok.body("S1GRD", int(r)), "S1GRD") for r in s1_rows]
    return target_ce(model, [torch.cat([a, b]) for a, b in zip(src, tgt)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_TRAIN, help="train locations (smoke tests only)")
    ap.add_argument("--n-val", type=int, default=96, help="val locations (smoke tests only)")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from eo.eval import teacher_forced as TF
    from eo.generate import conditional as G
    if torch.cuda.device_count() != 1 or "TITAN X" not in torch.cuda.get_device_name(0):
        raise SystemExit("run on the GTX Titan X, alone")
    model, cfg = G.build_model(ARM, args.checkpoint)
    ds, _ = TF.eval_loader(cfg, num_workers=0)
    report = {"model": args.checkpoint, "arm": ARM, "precision": "fp32", "layout": "[S1GRD][S2L2A], sequential",
              "gpu": torch.cuda.get_device_name(0),
              "bootstrap": {"draws": DRAWS, "unit": f"{SPLIT_RADIUS_KM} km spatial group"}}

    # Val: the 96 complete held-out locations, which B10c never saw. All 16 (t, o) per location.
    held = set(int(r) for r in load_eval_rows(root_dir=VAL))
    loc = location_table(VAL)
    cols = [f"row_{t}" for t in range(4)]
    mine = loc[loc[cols].isin(held).all(axis=1)].reset_index(drop=True)
    assert len(mine) == 96, len(mine)
    mine = mine.iloc[: args.n_val]
    doc = load_temporal_split(VAL)
    group_of = dict(zip(doc["dev"]["locations"] + doc["test"]["locations"], doc["dev"]["groups"] + doc["test"]["groups"]))
    grp = np.array([group_of[x] for x in mine.location])
    R = mine[cols].values.astype(np.int64)
    n = len(R)
    tok = ValTokens(ds)
    t_idx, o_idx = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    C = ce_of(model, tok, np.concatenate([R[i, o_idx.ravel()] for i in range(n)]),
              np.concatenate([R[i, t_idx.ravel()] for i in range(n)])).reshape(n, 4, 4)        # C[i, t, o]
    place = ce_of(model, tok, np.roll(R, -1, axis=0).ravel(), R.ravel()).reshape(n, 4)         # the next location's S1 at t

    same = np.stack([C[:, t, t] for t in range(4)], 1).mean(1)
    other = np.stack([C[:, t, (t + 2) % 4] for t in range(4)], 1).mean(1)
    moved = np.stack([C[:, t, (t + 3) % 4] - C[:, t, (t + 1) % 4] for t in range(4)], 1).mean(1)
    steps = {}
    for s in (1, 2, 3):
        pairs = [(t, o) for t in range(4) for o in range(4) if abs(t - o) == s]
        steps[str(s)] = {"pairs_per_location": len(pairs),
                         "other_minus_same": boot(np.stack([C[:, t, o] - C[:, t, t] for t, o in pairs], 1).mean(1), grp)}
    report["val_held_out"] = {
        "pairs": 4 * n, "locations": n, "groups": int(len(np.unique(grp))),
        "ce": {"same_date": boot(same, grp), "other_date": boot(other, grp), "other_place": boot(place.mean(1), grp)},
        "other_date_minus_same_date": boot(other - same, grp),
        "other_place_minus_same_date": boot(place.mean(1) - same, grp),
        "by_date_step": steps,
        "control_s1_moved_one_date": boot(moved, grp),
        "true_minus_control": boot(other - same - moved, grp)}

    # Train: pairs of a location's rows that are both local. t and the other date come from the pair.
    tok = TrainTokens()
    meta = pd.read_parquet(META, columns=["zarr", "sample_id", "tar", "center_lat", "center_lon"])
    meta = meta[meta.tar.str.startswith("ssl4eos12")]
    stem = meta.zarr.str[: -len(".zarr.zip")]
    m = pd.DataFrame({"stem": stem.values, "location": meta.sample_id.str.rsplit("_", n=1).str[0].values,
                      "t": meta.sample_id.str.rsplit("_", n=1).str[1].astype(int).values,
                      "lat": meta.center_lat.values, "lon": meta.center_lon.values})
    m = m.merge(tok.idx[["stem", "row"]], on="stem", validate="1:1")
    m = m[m.row.isin(set(tok.where))]
    n_rows = m.groupby("location").size()
    pairs = m[m.location.isin(n_rows.index[n_rows >= 2])].sort_values(["location", "t"])
    # Rows are kept in date order inside a location; the two farthest apart give the widest date gap.
    first, last = pairs.groupby("location").first(), pairs.groupby("location").last()
    both = first.join(last, lsuffix="_a", rsuffix="_b").reset_index()
    report["train_locations_with_two_local_rows"] = int(len(both))
    report["train_shards_local"] = len(tok.shards)
    pick = np.sort(np.random.default_rng(SEED).permutation(len(both))[: N_TRAIN])[: args.n]
    both = both.iloc[pick].reset_index(drop=True)
    g = spatial_groups(both.lat_a.values, both.lon_a.values, SPLIT_RADIUS_KM)
    a, b = both.row_a.values, both.row_b.values
    step = (both.t_b - both.t_a).values
    same_t = np.stack([ce_of(model, tok, a, a), ce_of(model, tok, b, b)], 1)
    other_t = np.stack([ce_of(model, tok, b, a), ce_of(model, tok, a, b)], 1)
    place_t = np.stack([ce_of(model, tok, np.roll(a, -1), a), ce_of(model, tok, np.roll(b, -1), b)], 1)
    d = (other_t - same_t).mean(1)
    report["train"] = {
        "pairs": 2 * len(both), "locations": len(both), "groups": int(len(np.unique(g))),
        "ce": {"same_date": boot(same_t.mean(1), g), "other_date": boot(other_t.mean(1), g),
               "other_place": boot(place_t.mean(1), g)},
        "other_date_minus_same_date": boot(d, g),
        "other_place_minus_same_date": boot((place_t - same_t).mean(1), g),
        "by_date_step": {str(s): {"locations": int((step == s).sum()), "other_minus_same": boot(d[step == s], g[step == s])}
                         for s in (1, 2, 3)}}

    v, tr = report["val_held_out"], report["train"]
    ok = v["other_date_minus_same_date"]["lo"] > 0 and tr["other_date_minus_same_date"]["lo"] > 0
    fired = v["true_minus_control"]["lo"] > 0
    report["passed"], report["control_fired"] = bool(ok), bool(fired)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    np.savez(out.with_name(PAIRS.name), val_ce=C, val_other_place=place, val_groups=grp,
             train_same=same_t, train_other=other_t, train_other_place=place_t, train_step=step, train_groups=g)
    print(json.dumps(report, indent=1), flush=True)
    print(f"wrote {out}: passed {ok}, control fired {fired}", flush=True)
    return 0 if ok and fired else 1


if __name__ == "__main__":
    sys.exit(main())
