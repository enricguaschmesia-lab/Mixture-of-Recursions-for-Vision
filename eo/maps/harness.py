"""The one entry point of A2 Step 1: `score(candidate, dataset, split)`.

A candidate is a map tokenizer with a decoding rule. `score` renders the fixed set's maps with the task's
fixed palette assignment, cuts them as the test reads them (CloudSen12: 3 x 3 crops, eo/maps/tiling.py),
encodes at a fixed batch shape, decodes, turns the output into class scores on the native grid, applies
the rule, and returns:
  - the ceiling: GEO-Bench-2's mIoU (eo/maps/scoring.py), pixel, boundary and interior accuracy, IoU per class;
  - the code statistics: distinct codes and entropy of the split's tokens, and the share of them whose code
    never occurs in the training tokens (PASTIS fold 4 against folds 1-3, CloudSen12 val against 1,000
    train tiles, LULC read set against 5,000 pool maps: each dataset's `train_sub`).
`sweep` reads many rules in one pass; `select_rule` chooses the region rule's tau and radius on the
training maps (each dataset's `train_sub`): one rule per tokenizer, the best mean mIoU over the three
datasets, with each dataset's own optimum reported beside it.

Tokens are cached per (tokenizer and weights hash, set and its sha256, palette, crop mode) under
/data/enric/reports/a2_seg/step1/cache/. Agents never score with their own code.

    python -m eo.maps.harness --cosmos t1_di8_256 --select --read --expect-gpu "TITAN V" --out x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from eo.maps import decode as D
from eo.maps import palette as P
from eo.maps import scoring as S
from eo.maps import sets, tiling
from eo.maps.tokenizers import MapTokenizer, run_fixed

CODES_VERSION = "step1-v1"     # in the token cache's path: change it when the way codes are made changes
HARNESS_VERSION = "step1-v2"   # recorded in every result. v2 (2026-11-18): scores pooled to the native grid
                               # by the adjoint of the render (decode.to_native), not Step 0's area pooling
CACHE = Path(os.environ.get("A2_CACHE_ROOT", "/data/enric/reports/a2_seg/step1/cache/tokens")) / CODES_VERSION
REF_SPLIT = "train_sub"


@dataclass
class Candidate:
    tokenizer: MapTokenizer
    rule: dict = field(default_factory=lambda: {"rule": "nearest"})
    palette: str = "universal"          # "step0": Step 0's per-dataset palettes (regression anchor only)
    crop_mode: str = "crop224"          # CloudSen12 tiling (eo/maps/tiling.py)

    @property
    def name(self) -> str:
        return f"{self.tokenizer.name}__{D.rule_key(self.rule)}__{self.palette}"


def task_palette(dataset: str, palette: str) -> tuple[np.ndarray, tuple, int | None]:
    """(colours, class -> colour index, void colour index) of a task under a palette."""
    task = P.TASKS[dataset]
    if palette == "universal":
        return P.UNIVERSAL, task.assign, P.VOID_COLOUR
    if palette == "step0":
        colours, assign = P.step0_distinct(task)
        return colours, assign, None
    raise ValueError(palette)


def _weights_tag(tok: MapTokenizer) -> str:
    rec = tok.record()
    blob = json.dumps(rec.get("sha256", rec), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _set_sha(dataset: str, split: str) -> str:
    ent = sets.manifest()["sets"][f"{dataset}/{'val' if split == 'val' else 'train'}"]
    sub = {"train_sub": "subset_sha256", "monitor": "monitor_sha256", "fit": "monitor_sha256"}.get(split)
    return ent["labels_sha256"][:12] + (ent[sub][:12] if sub else "")


def render_crops(labels: np.ndarray, dataset: str, palette: str, size: int, crop_mode: str,
                 device: str) -> torch.Tensor:
    """Maps -> the tokenizer inputs, (N * crops, 3, size, size) in [0, 1], map-major."""
    task = P.TASKS[dataset]
    colours, assign, void = task_palette(dataset, palette)
    P.validate(colours, assign, task.ignore, task.n_classes, void_index=void)
    lab = torch.as_tensor(np.ascontiguousarray(labels), device=device).long()
    crops = tiling.plan(dataset, task.native, crop_mode)
    return P.render(tiling.cut(lab, crops), assign, size, colours)


def tokens(tok: MapTokenizer, dataset: str, split: str, palette: str = "universal",
           crop_mode: str = "crop224", cache: bool = True, device: str = "cuda",
           maps_per_chunk: int = 64) -> np.ndarray:
    """Codes of every crop of a fixed set, (N * crops, h, w) int32, cached on disk."""
    path = CACHE / f"{tok.name}__{_weights_tag(tok)}" / \
        f"{dataset}_{split}_{_set_sha(dataset, split)}_{palette}_{crop_mode}_b{tok.batch}.npy"
    if cache and path.exists():
        return np.load(path)
    labels, _ = sets.load(dataset, split)
    out = []
    for s in range(0, len(labels), maps_per_chunk):
        x = render_crops(labels[s:s + maps_per_chunk], dataset, palette, tok.input_size, crop_mode, device)
        out.append(run_fixed(tok.encode, x, tok.batch).int().cpu().numpy())
    idx = np.concatenate(out)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, idx)
    return idx


def class_scores(tok: MapTokenizer, idx: torch.Tensor, dataset: str, palette: str, crop_mode: str,
                 with_votes: bool = False):
    """Codes of a whole number of maps -> class scores on the native grid, (N, K, native, native):
    the scores at the tokenizer's resolution, area-resampled to each crop's native size and stitched.
    with_votes: also the area share of each class among the pixels snapped at the tokenizer's
    resolution (the vote rule), returned as a second tensor."""
    task = P.TASKS[dataset]
    colours, assign, _ = task_palette(dataset, palette)
    crops = tiling.plan(dataset, task.native, crop_mode)
    out = run_fixed(tok.decode, idx, tok.batch)
    if tok.output == "rgb":
        sc_s = D.rgb_scores(out, np.asarray(colours)[list(assign)])
    elif tok.output == "logits":
        if palette != "universal":
            raise ValueError("palette logits are over the universal palette")
        sc_s = D.logit_scores(out, assign)
    else:
        raise ValueError(tok.output)
    sc = tiling.stitch(D.to_native(sc_s, crops[0][2]), crops, task.native)
    if not with_votes:
        return sc
    v = tiling.stitch(D.to_native(D.votes_at_s(sc_s, task.ignore), crops[0][2]), crops, task.native)
    return sc, v


def sweep(tok: MapTokenizer, dataset: str, split: str, rules: list[dict], palette: str = "universal",
          crop_mode: str = "crop224", cache: bool = True, device: str = "cuda",
          maps_per_chunk: int | None = None, return_pred: dict | None = None) -> dict:
    """Read every rule on one fixed set in one pass. Returns {rule_key: Confusion.result()}.
    return_pred, if a dict, receives {rule_key: predictions (N, H, W) uint8}."""
    if split == "test" or "test" in split:
        raise sets.ForbiddenSplit(split)
    task = P.TASKS[dataset]
    labels, _ = sets.load(dataset, split)
    idx = tokens(tok, dataset, split, palette, crop_mode, cache, device)
    n_crops = len(tiling.plan(dataset, task.native, crop_mode))
    assert len(idx) == len(labels) * n_crops
    if maps_per_chunk is None:
        maps_per_chunk = {"pastis": 64, "lulc": 32, "cloudsen12": 4}[dataset]
    conf = {D.rule_key(r): S.Confusion(task.n_classes, task.ignore, device) for r in rules}
    preds = {k: [] for k in conf} if return_pred is not None else None
    with torch.no_grad():
        for s in range(0, len(labels), maps_per_chunk):
            gt = torch.as_tensor(np.ascontiguousarray(labels[s:s + maps_per_chunk]), device=device).long()
            ib = torch.as_tensor(idx[s * n_crops:(s + len(gt)) * n_crops], device=device).long()
            need_votes = any(r["rule"] == "vote" for r in rules)
            out = class_scores(tok, ib, dataset, palette, crop_mode, with_votes=need_votes)
            sc, votes = out if need_votes else (out, None)
            for r in rules:
                k = D.rule_key(r)
                pr = D.apply_rule(sc, task.ignore, r, tok.output, votes)
                conf[k].update(gt, pr)
                if preds is not None:
                    preds[k].append(pr.to(torch.uint8).cpu().numpy())
    if preds is not None:
        return_pred.update({k: np.concatenate(v) for k, v in preds.items()})
    return {k: c.result() for k, c in conf.items()}


def score(candidate: Candidate, dataset: str, split: str = "val", cache: bool = True, device: str = "cuda",
          code_stats: bool = True, return_pred: dict | None = None, out: str | None = None,
          also: list[dict] | None = None) -> dict:
    """The ceiling and code statistics of one candidate on one fixed set. `also`: other rules read in
    the same pass (reported under "also", never as the candidate's ceiling)."""
    tok = candidate.tokenizer
    t0 = time.time()
    extra = [r for r in (also or []) if D.rule_key(r) != D.rule_key(candidate.rule)]
    res = sweep(tok, dataset, split, [candidate.rule] + extra, candidate.palette, candidate.crop_mode, cache,
                device, return_pred=return_pred)
    r = {"candidate": candidate.name, "dataset": dataset, "split": split, "rule": candidate.rule,
         "palette": candidate.palette, "crop_mode": candidate.crop_mode,
         "palette_sha256": P.palette_record()["sha256"] if candidate.palette == "universal" else None,
         "tokenizer": tok.record(), "tokens_per_map": None, "ceiling": res[D.rule_key(candidate.rule)],
         "also": {D.rule_key(e): res[D.rule_key(e)] for e in extra}, "harness": HARNESS_VERSION}
    if code_stats:
        read = tokens(tok, dataset, split, candidate.palette, candidate.crop_mode, cache, device)
        ref = tokens(tok, dataset, REF_SPLIT, candidate.palette, candidate.crop_mode, cache, device) \
            if split != REF_SPLIT else None
        r["codes"] = S.code_stats(read, ref, tok.n_codes)
        r["codes"]["reference"] = f"{dataset}/{REF_SPLIT}"
        r["tokens_per_map"] = int(np.prod(read.shape[1:]))
    r["seconds"] = time.time() - t0
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(r, indent=1))
    return r


def select_rule(tok: MapTokenizer, palette: str = "universal", rules: list[dict] | None = None,
                datasets=sets.DATASETS, crop_mode: str = "crop224", cache: bool = True,
                device: str = "cuda") -> dict:
    """Choose the decoding rule on the training maps: nearest, vote and (RGB only) the region grid; the
    chosen rule is the best mean mIoU over the datasets (ties: smaller radius, then smaller tau, then
    the order of the list)."""
    if rules is None:
        rules = [{"rule": "nearest"}, {"rule": "vote"}] + (D.region_grid() if tok.output == "rgb" else [])
    by = {d: sweep(tok, d, REF_SPLIT, rules, palette, crop_mode, cache, device) for d in datasets}
    keys = [D.rule_key(r) for r in rules]
    mean = {k: float(np.mean([by[d][k]["mIoU"] for d in datasets])) for k in keys}

    def order(r):
        k = D.rule_key(r)
        return (-round(mean[k], 6), r.get("radius", 0), r.get("tau", 0))
    chosen = sorted(rules, key=order)[0]
    per = {d: max(keys, key=lambda k: by[d][k]["mIoU"]) for d in datasets}
    return {"chosen": chosen, "chosen_key": D.rule_key(chosen), "mean": mean,
            "by_dataset": {d: {k: by[d][k]["mIoU"] for k in keys} for d in datasets},
            "per_dataset_optimum": {d: {"rule": per[d], "mIoU": by[d][per[d]]["mIoU"]} for d in datasets},
            "selected_on": {d: f"{d}/{REF_SPLIT}" for d in datasets}}


def read_candidate(tok: MapTokenizer, rule: dict, palette: str = "universal", datasets=sets.DATASETS,
                   crop_mode: str = "crop224", cache: bool = True, device: str = "cuda") -> dict:
    """Step 1.2's read of one candidate on every read set, plus nearest beside the rule."""
    return {d: score(Candidate(tok, rule, palette, crop_mode), d, "val", cache, device,
                     also=[{"rule": "nearest"}]) for d in datasets}


def bar(read: dict) -> dict:
    """Step 1.3's bar, fixed: PASTIS fold 4 >= 0.85, CloudSen12 val >= 0.95, LULC >= 0.95."""
    need = {"pastis": 0.85, "cloudsen12": 0.95, "lulc": 0.95}
    got = {d: read[d]["ceiling"]["mIoU"] for d in need if d in read}
    return {"need": need, "got": got, "passes": all(got.get(d, -1) >= v for d, v in need.items())}


def main():
    from eo.maps.tokenizers import cosmos
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos", required=True, help="e.g. t1_di8_256")
    ap.add_argument("--palette", default="universal")
    ap.add_argument("--crop-mode", default="crop224")
    ap.add_argument("--select", action="store_true", help="choose tau and radius on the training maps")
    ap.add_argument("--rule", default=None, help='JSON rule, e.g. {"rule":"region","tau":0.15,"radius":4}')
    ap.add_argument("--read", action="store_true", help="read the validation sets")
    ap.add_argument("--datasets", default=",".join(sets.DATASETS))
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in name:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {name}")
    tok = cosmos(a.cosmos)
    ds = tuple(a.datasets.split(","))
    res = {"tokenizer": tok.record(), "gpu": name, "palette": a.palette, "crop_mode": a.crop_mode}
    rule = json.loads(a.rule) if a.rule else {"rule": "nearest"}
    if a.select:
        res["selection"] = select_rule(tok, a.palette, datasets=ds, crop_mode=a.crop_mode)
        rule = res["selection"]["chosen"]
        print("chosen", rule, flush=True)
    if a.read:
        res["read"] = read_candidate(tok, rule, a.palette, ds, a.crop_mode)
        res["bar"] = bar(res["read"])
        for d, v in res["read"].items():
            print(d, round(v["ceiling"]["mIoU"], 4), "boundary", round(v["ceiling"]["boundary_acc"], 4),
                  "codes", v["codes"]["read_distinct"], flush=True)
    res["rule"] = rule
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
