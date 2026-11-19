"""A2 Step 1, Phase 2: the orchestrator's independent re-score of an agent's candidate.

Rebuilds the tokenizer from the loader string its record names (or a Cosmos name), re-encodes the read sets
with the token cache OFF, reads them through eo.maps.harness with the agent's rule, and compares every number
with the agent's JSON. A mismatch beyond the tolerance reopens the agent's work.

    python -m eo.experiments.segmentation.step1_O_rescore --loader "<module>.load('<ckpt>', 'cuda')" \
        --rule '{"rule": "nearest"}' --agent-json <the agent's read JSON> --out <json>
    python -m eo.experiments.segmentation.step1_O_rescore --cosmos t1_di8_256 --rule '{...}' ...

The agent's JSON is matched leniently: any mapping dataset -> {..."ceiling": {"mIoU": x}...} or dataset -> mIoU
found in it is compared; the rest is printed side by side for a human check.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import torch

from eo.maps import harness as H
from eo.maps import sets


def build(loader: str):
    """'pkg.mod.load(args...)' -> the tokenizer (the module is imported, then the call evaluated in it)."""
    mod_name, call = loader.split(".load(", 1)
    mod = importlib.import_module(mod_name)
    return eval("load(" + call, {"load": mod.load})


def find_mious(obj, path=()) -> dict:
    """Every (dataset, rule-ish key) -> mIoU number in a nested JSON, keyed by its path."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "mIoU" and isinstance(v, (int, float)):
                out["/".join(map(str, path))] = float(v)
            else:
                out.update(find_mious(v, path + (k,)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(find_mious(v, path + (i,)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", default=None)
    ap.add_argument("--cosmos", default=None)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--datasets", default=",".join(sets.DATASETS))
    ap.add_argument("--agent-json", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="cap this process's share of the card's memory (to share a card with a training run)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction, 0)
    name = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in name:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {name}")
    if a.cosmos:
        from eo.maps.tokenizers import cosmos
        tok = cosmos(a.cosmos)
    else:
        tok = build(a.loader)
    rule = json.loads(a.rule)
    t0 = time.time()
    res = {"tokenizer": tok.record(), "rule": rule, "gpu": name, "harness": H.HARNESS_VERSION, "read": {}}
    for d in a.datasets.split(","):
        r = H.score(H.Candidate(tok, rule), d, "val", cache=False, also=[{"rule": "nearest"}, {"rule": "vote"}])
        res["read"][d] = r
        print(f"{d}: mIoU {r['ceiling']['mIoU']:.6f} boundary {r['ceiling']['boundary_acc']:.4f} "
              f"codes {r['codes']['read_distinct']} H {r['codes']['read_entropy_bits']:.3f} "
              f"unseen {r['codes']['unseen_token_share']:.5f}", flush=True)
    res["seconds"] = time.time() - t0
    if a.agent_json:
        theirs = find_mious(json.loads(Path(a.agent_json).read_text()))
        mine = {d: res["read"][d]["ceiling"]["mIoU"] for d in res["read"]}
        res["agent_mious"] = theirs
        print("agent's JSON, every mIoU found:", flush=True)
        for k, v in sorted(theirs.items()):
            print(f"  {k}: {v:.6f}", flush=True)
        print("mine:", {d: round(v, 6) for d, v in mine.items()}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
