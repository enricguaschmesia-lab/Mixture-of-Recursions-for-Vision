#!/usr/bin/env python
"""Does S1 at date t help predict S2 at date t more than S1 at another date? (A1, Step 1.) `.venv`, GPU.

The dates in s1_dates.py come from product ids. This asks the images: if a row's S1 really belongs to
its date, teacher-forced CE(S2 at t | S1 at t) is below CE(S2 at t | S1 at another date, same place).

⚠ B10c, NOT T360. T360 trained on exactly these same-row pairs, so it would favour them whatever the
dates. B10c trained on val alone: it has seen no train row, and not the 96 val locations of
eval_rows_v1, which are the positive control.

Train side: pairs of rows of one location that both sit in the 24 ssl4eos12 token shards kept on the
workstation (train_tok/). Control column: S1 from another location at the same date index, which
scales the date effect against what S1 is worth at all. Bootstrap over locations.

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
from eo.data.eval_split import load_eval_rows  # noqa: E402
from eo.data.temporal_split import location_table  # noqa: E402
from eo.terramesh_tok.contract import TOKENS_PER_SAMPLE, tok_dir_name  # noqa: E402

CKPT = "/data/enric/runs/pretrain/phase3/vanilla10c_20261011_172253/checkpoint-3300"
ARM = "eo_terramesh/arm_b10c_vanilla"
TRAIN = Path("/data/enric/data/TerraMesh/train_tok")
META = Path("/data/enric/data/TerraMesh/train_meta/train_metadata.parquet")
VAL = "/data/enric/data/TerraMesh/val"
OUT = Path("/data/enric/reports/temporal/s1_functional.json")
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


def score(model, tok, triples, name: str) -> dict:
    """triples: (location id, S2 row, S1 row at t, S1 row at another date, S1 row at another place)."""
    tgt = [chunk(tok.body("S2L2A", r[1]), "S2L2A") for r in triples]
    src = {k: [chunk(tok.body("S1GRD", r[i]), "S1GRD") for r in triples]
           for k, i in (("same_date", 2), ("other_date", 3), ("other_place", 4))}
    ce = {k: target_ce(model, [torch.cat([a, b]) for a, b in zip(v, tgt)]) for k, v in src.items()}
    unit = pd.factorize(np.asarray([str(r[0]) for r in triples]))[0]
    d = ce["other_date"] - ce["same_date"]
    res = {"pairs": len(triples), "locations": int(len(np.unique(unit))),
           "ce": {k: boot(v, unit) for k, v in ce.items()},
           "other_date_minus_same_date": boot(d, unit),
           "other_place_minus_same_date": boot(ce["other_place"] - ce["same_date"], unit),
           "share_of_pairs_same_date_better": float((d > 0).mean())}
    print(name, json.dumps(res, indent=1), flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_TRAIN)
    ap.add_argument("--checkpoint", default=CKPT)
    args = ap.parse_args()

    from eo.eval import teacher_forced as TF
    from eo.generate import conditional as G
    if torch.cuda.device_count() != 1 or "TITAN X" not in torch.cuda.get_device_name(0):
        raise SystemExit("run on the GTX Titan X, alone")
    model, cfg = G.build_model(ARM, args.checkpoint)
    ds, _ = TF.eval_loader(cfg, num_workers=0)
    report = {"model": args.checkpoint, "arm": ARM, "precision": "fp32", "layout": "[S1GRD][S2L2A], sequential",
              "gpu": torch.cuda.get_device_name(0)}

    # Positive control: val's 96 complete held-out locations, which B10c never saw.
    held = set(int(r) for r in load_eval_rows(root_dir=VAL))
    loc = location_table(VAL)
    rows = [f"row_{t}" for t in range(4)]
    mine = loc[loc[rows].isin(held).all(axis=1)].reset_index(drop=True)
    assert len(mine) == 96, len(mine)
    tv = []
    for i, r in mine.iterrows():
        q = mine.iloc[(i + 1) % len(mine)]
        for t in range(4):
            tv.append((r.location, int(r[f"row_{t}"]), int(r[f"row_{t}"]), int(r[f"row_{(t + 2) % 4}"]), int(q[f"row_{t}"])))
    report["val_held_out"] = score(model, ValTokens(ds), tv, "val")

    # Train: pairs of a location's rows that are both local. t and the other date come from the pair.
    tok = TrainTokens()
    meta = pd.read_parquet(META, columns=["zarr", "sample_id", "tar"])
    meta = meta[meta.tar.str.startswith("ssl4eos12")]
    stem = meta.zarr.str[: -len(".zarr.zip")]
    m = pd.DataFrame({"stem": stem.values, "location": meta.sample_id.str.rsplit("_", n=1).str[0].values,
                      "t": meta.sample_id.str.rsplit("_", n=1).str[1].astype(int).values})
    m = m.merge(tok.idx[["stem", "row"]], on="stem", validate="1:1")
    m = m[m.row.isin(set(tok.where))]
    n_rows = m.groupby("location").size()
    pairs = m[m.location.isin(n_rows.index[n_rows >= 2])].sort_values(["location", "t"])
    # Rows are kept in date order inside a location; the two farthest apart give the widest date gap.
    first, last = pairs.groupby("location").first(), pairs.groupby("location").last()
    both = first.join(last, lsuffix="_a", rsuffix="_b").reset_index()
    report["train_locations_with_two_local_rows"] = int(len(both))
    report["train_shards_local"] = len(tok.shards)
    pick = np.sort(np.random.default_rng(SEED).permutation(len(both))[: args.n])
    both = both.iloc[pick].reset_index(drop=True)
    report["train_date_steps"] = {int(k): int(v) for k, v in (both.t_b - both.t_a).value_counts().sort_index().items()}
    tt = []
    for i, r in both.iterrows():
        q = both.iloc[(i + 1) % len(both)]
        tt.append((r.location, int(r.row_a), int(r.row_a), int(r.row_b), int(q.row_a)))
        tt.append((r.location, int(r.row_b), int(r.row_b), int(r.row_a), int(q.row_b)))
    report["train"] = score(model, tok, tt, "train")

    # The check can fail: label each pair's S1 rows the wrong way round, as a row swap would.
    swapped = [(r[0], r[1], r[3], r[2], r[4]) for r in tt]
    report["control_rows_swapped"] = score(model, tok, swapped, "swapped")

    ok = report["val_held_out"]["other_date_minus_same_date"]["lo"] > 0 and report["train"]["other_date_minus_same_date"]["lo"] > 0
    fired = report["control_rows_swapped"]["other_date_minus_same_date"]["hi"] < 0
    report["passed"], report["control_fired"] = bool(ok), bool(fired)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUT}: passed {ok}, control fired {fired}", flush=True)
    return 0 if ok and fired else 1


if __name__ == "__main__":
    sys.exit(main())
