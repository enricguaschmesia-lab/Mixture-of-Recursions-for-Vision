#!/usr/bin/env python
"""Does the lattice work with real inserted tokens, before any training? (A1, Step 0.) `.venv`, GPU, fp32.

t360_context.py moves position_ids with nothing in the gap. A date block is tokens in the gap, and
T360 has shown it reads such tokens: real Coords between two blocks keep the alignment that the same
5-position shift without them destroys. So this tests the layout itself.

T360's extended base carries BEN's 21 rows at mean + noise, unused by training: the state new date rows
start in. Four of them stand in for [BO_Date Y D EO_Date]. On the sweep's two pairs, per location:

  adjacent         [src][tgt]                   the reference
  date_sequential  [src][date][tgt]             sequential: offset 202
  date_lattice     [src][date][tgt]             the target restarts where the date block began: 198
  dates_lattice    [date][src][date][tgt]       a date before each block, as in a sequence
  coords_between   [src][Coords][tgt]           sequential: 203, T360's own habit
  coords_rewound   [src][Coords][tgt]           the same positions rewound to 198: must Coords stay?
  dates_header     [date][date][src][tgt]       sequential: dates only at the start, none in the gap

PROFILE: CE block by block along one whole sequence, the loss a run at full length would start from:
[Coords][DEM] then four dates chronologically, S2L2A, S1GRD, NDVI, with LULC at date 0. No date tokens.

Read: the rise over `adjacent` as a share of the gain the source gives over no context, against the
same 10% as Step 0's sweep.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(.venv/bin/python -m eo.train.gpu titanx) \\
      .venv/bin/python -X faulthandler -u eo/experiments/temporal/t360_lattice.py
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import t360_context as C  # noqa: E402  (this folder is on sys.path when run as a script)
from eo.data import ben_vocab as BV  # noqa: E402
from eo.data.temporal_split import load_temporal_split  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402

BASE = "/data/enric/runs/pretrain/phase6/base/t360_ext_ckpt17441"
OUT = Path("/data/enric/reports/temporal/t360_lattice.json")
DATE = torch.tensor([BV.BO_BEN, BV.CLASS_IDS[3], BV.CLASS_IDS[11], BV.EO_BEN], dtype=torch.long)
PAIRS = ("S2L2A>NDVI", "S2L2A>S2L2A_other_date")


@torch.no_grad()
def ce_of(model, seqs, pos):
    out = []
    for k in range(0, len(seqs), 16):
        x = torch.stack(seqs[k:k + 16]).cuda()
        p = torch.as_tensor(pos, device="cuda").unsqueeze(0).expand(len(x), -1)
        lg = model(input_ids=x, position_ids=p, use_cache=False, logits_to_keep=C.BLOCK).logits[:, :C.TOKENS_PER_SAMPLE].float()
        out.append(F.cross_entropy(lg.transpose(1, 2), x[:, -(C.TOKENS_PER_SAMPLE + 1):-1], reduction="none").mean(1))
    return torch.cat(out).double().cpu().numpy()


def positions(parts, rewind):
    """Sequential positions; a part named in `rewind` does not advance the counter."""
    pos, p = [], 0
    for name, length in parts:
        pos += list(range(p, p + length))
        if name not in rewind:
            p += length
    return pos


@torch.no_grad()
def profile(model, chunk, rows, groups, batch: int = 1):
    """Mean CE per block along a full chronological sequence, with T360's own trained length marked."""
    names, seqs = None, []
    for r in rows:
        b = [("Coords", r[0]), ("DEM", r[0]), ("LULC", r[0])] + [(m, r[t]) for t in range(4) for m in C.TARGETS]
        names = [f"{m}@{t}" for t in range(4) for m in C.TARGETS]
        seqs.append(torch.cat([chunk(m, x) for m, x in b]))
    names = ["Coords", "DEM", "LULC"] + names
    spans, at = [], 0
    for n in names:
        w = 5 if n == "Coords" else C.BLOCK
        spans.append((n, at + 1, at + w - 1))              # body tokens only, as every CE here
        at += w
    per = []
    for k in range(0, len(seqs), batch):
        x = torch.stack(seqs[k:k + batch]).cuda()
        lg = model(input_ids=x, use_cache=False).logits[:, :-1].float()
        per.append(F.cross_entropy(lg.transpose(1, 2), x[:, 1:], reduction="none").double().cpu().numpy())
    per = np.concatenate(per)
    tab = []
    for n, a, b in spans:
        v = per[:, a - 1:b - 1].mean(1)
        tab.append({"block": n, "first_token": a, "ce": C.group_ci(v, groups)})
        print(f"  {n:10s} from {a:5d}: CE {v.mean():.3f}", flush=True)
    body = np.concatenate([per[:, a - 1:b - 1] for _, a, b in spans], 1)
    return {"n": len(seqs), "tokens": int(at), "mean_ce_all_body": float(body.mean()), "by_block": tab}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=C.N_LOC)
    args = ap.parse_args()
    from eo.generate import conditional as G
    if torch.cuda.device_count() != 1 or "TITAN X" not in torch.cuda.get_device_name(0):
        raise SystemExit("run on the GTX Titan X, alone")
    model, cfg = G.build_model(C.ARM, BASE, overrides=[f"model_config.vocab_size={BV.TOTAL_VOCAB_SIZE}"])
    if model.get_input_embeddings().weight.shape[0] != BV.TOTAL_VOCAB_SIZE:
        raise SystemExit("not the extended table")
    ds, _ = TF.eval_loader(cfg, num_workers=0)
    rows, groups, _ = C.pick_locations(load_temporal_split(C.ROOT), args.n)
    N = len(rows)
    chunk = lambda m, r: ds._load_chunk(m, int(r))
    report = {"base": BASE, "date_stand_in_ids": DATE.tolist(), "n_locations": N, "gpu": torch.cuda.get_device_name(0),
              "share_max": C.SHARE_MAX, "pairs": {}}
    for pair in PAIRS:
        items = []
        for i in range(N):
            t = C.date_of(i, 0)
            if pair == "S2L2A>NDVI":
                items.append((chunk("S2L2A", rows[i, t]), chunk("NDVI", rows[i, t]), chunk("Coords", rows[i, t])))
            else:
                items.append((chunk("S2L2A", rows[i, C.near(t)]), chunk("S2L2A", rows[i, t]), chunk("Coords", rows[i, t])))
        B, D, K = C.BLOCK, len(DATE), 5
        lay = {
            "adjacent": ([lambda s, g, c: [s, g]], [("s", B), ("t", B)], ()),
            "date_sequential": ([lambda s, g, c: [s, DATE, g]], [("s", B), ("d", D), ("t", B)], ()),
            "date_lattice": ([lambda s, g, c: [s, DATE, g]], [("s", B), ("d", D), ("t", B)], ("d",)),
            "dates_lattice": ([lambda s, g, c: [DATE, s, DATE, g]], [("d", D), ("s", B), ("d", D), ("t", B)], ("d",)),
            "coords_between": ([lambda s, g, c: [s, c, g]], [("s", B), ("c", K), ("t", B)], ()),
            "coords_rewound": ([lambda s, g, c: [s, c, g]], [("s", B), ("c", K), ("t", B)], ("c",)),
            "dates_header": ([lambda s, g, c: [DATE, DATE, s, g]], [("d", D), ("d2", D), ("s", B), ("t", B)], ()),
        }
        ce = {n: ce_of(model, [torch.cat(f[0](*it)) for it in items], positions(parts, rw))
              for n, (f, parts, rw) in lay.items()}
        none = ce_of(model, [it[1] for it in items], list(range(C.BLOCK)))
        gain = float((none - ce["adjacent"]).mean())
        tab = {"none": C.group_ci(none, groups), "gain_over_none": C.group_ci(none - ce["adjacent"], groups)}
        for n, v in ce.items():
            rise = C.group_ci(v - ce["adjacent"], groups)
            tab[n] = {"ce": C.group_ci(v, groups), "rise": rise, "share_of_gain": rise["mean"] / gain}
        report["pairs"][pair] = tab
        print(pair, json.dumps({k: (round(v["ce"]["mean"], 4), round(v["share_of_gain"], 3)) if "ce" in v else round(v["mean"], 4)
                                for k, v in tab.items()}), flush=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    report["profile"] = profile(model, chunk, rows[: min(args.n, 64)], groups[: min(args.n, 64)])
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
