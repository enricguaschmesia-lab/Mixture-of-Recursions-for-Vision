"""Read one of agent B's map tokenizers through the harness (A2 Step 1.2), with its controls.

    python -X faulthandler -u -m eo.experiments.segmentation.step1_B_read --ckpt <best.pt> --out <json> \
        [--select] [--read] [--controls] [--native256] [--expect-gpu "TITAN V"]

  --select    harness.select_rule(tok): nearest against vote, chosen on each dataset's train_sub
  --read      harness.read_candidate(tok, rule) on the three read sets (nearest beside) and harness.bar
  --native256 CloudSen12 also with crop_mode="native256" (256-px candidates; beside, never instead)
  --controls  (1) determinism: the same maps encoded twice at the fixed batch shape give identical codes,
                  and a batch with one map changed gives different codes (the check can fail);
              (2) transposed maps: the read sets encoded transposed and scored against the maps as they
                  are, beside the identity tokenizer read the same way (what transposition alone costs);
              (3) an untrained tokenizer (same architecture, random init) on the read sets;
              (4) the assignment: training draws palette.shuffled_assignment per map, the read uses
                  TASKS[d].assign; counted on 2,000 drawn PASTIS maps, how many training assignments
                  equal the fixed one.
Every ceiling comes from eo.maps.harness; this script only wires candidates and controls to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from eo.experiments.segmentation import step1_B_tok as T
from eo.maps import harness as H
from eo.maps import palette as P
from eo.maps import sets
from eo.maps.tokenizers import Identity, MapTokenizer, run_fixed


class Transposed(MapTokenizer):
    """A control: the tokenizer fed each input transposed; decoded as is, so the read scores the
    transposed map against the map."""

    def __init__(self, tok: MapTokenizer):
        self.t = tok
        self.name, self.input_size, self.n_codes = tok.name + "_transposed_input", tok.input_size, tok.n_codes
        self.output, self.batch = tok.output, tok.batch

    def encode(self, x):
        return self.t.encode(x.transpose(-1, -2).contiguous())

    def decode(self, idx):
        return self.t.decode(idx)

    def record(self):
        r = dict(self.t.record())
        r["control"] = "input transposed before encode"
        r["sha256"] = dict(r.get("sha256", {}), control="transposed_input")
        return r


def short(res: dict) -> dict:
    c = res["ceiling"]
    out = {k: c[k] for k in ("mIoU", "boundary_acc", "interior_acc", "pixel_acc")}
    if "codes" in res:
        out["codes"] = {k: res["codes"][k] for k in ("read_distinct", "read_entropy_bits", "unseen_token_share")
                        if k in res["codes"]}
    return out


def determinism(tok: MapTokenizer, split: str = "val", device: str = "cuda") -> dict:
    out = {}
    for d in sets.DATASETS:
        labels, _ = sets.load(d, split)
        n = {"pastis": 48, "cloudsen12": 4, "lulc": 48}[d]
        x = H.render_crops(np.asarray(labels[:n]), d, "universal", tok.input_size, "crop224", device)
        c1 = run_fixed(tok.encode, x, tok.batch)
        c2 = run_fixed(tok.encode, x, tok.batch)
        x3 = x.clone()
        j = int((x != x.transpose(-1, -2)).flatten(1).sum(1).argmax())   # the crop transposition changes most
        x3[j] = x3[j].transpose(-1, -2).clone()              # one crop changed: the check must see it
        c3 = run_fixed(tok.encode, x3, tok.batch)
        out[d] = {"crops": len(x), "identical_twice": bool(torch.equal(c1, c2)),
                  "control_crop": j, "control_pixels_changed": int((x3[j] != x[j]).any(0).sum()),
                  "control_one_map_changed_identical": bool(torch.equal(c1, c3)),
                  "control_tokens_differing": int((c1 != c3).sum()),
                  "codes_sha256": hashlib.sha256(c1.cpu().numpy().astype(np.int64).tobytes()).hexdigest()}
    return out


def assignment_check(n_maps: int = 2000) -> dict:
    g = torch.Generator()
    g.manual_seed(1)
    t = P.TASKS["pastis"]
    a = P.shuffled_assignment(n_maps, t.n_classes, t.ignore, generator=g, device="cpu")
    fixed = torch.as_tensor(t.assign)
    return {"maps": n_maps, "equal_to_fixed": int((a == fixed).all(1).sum()),
            "mean_classes_on_fixed_colour": float((a == fixed).float()[:, t.scored].sum(1).mean()),
            "train_code_path": "step1_B_train.main: P.shuffled_assignment(...) -> P.render(maps, assign, size)",
            "read_code_path": "eo.maps.harness.render_crops: task_palette -> P.TASKS[d].assign -> P.render"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--rule", default=None, help='JSON rule instead of --select, e.g. {"rule":"nearest"}')
    ap.add_argument("--native256", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="val", help="the read split; anything but val is a code-path test only")
    a = ap.parse_args()
    dv, sp = a.device, a.split
    gpu = torch.cuda.get_device_name(0) if dv == "cuda" else "cpu"
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    t0 = time.time()
    tok = T.load(a.ckpt, dv)
    res = {"tokenizer": tok.record(), "gpu": gpu, "split": sp, "palette_sha256": P.palette_record()["sha256"],
           "harness": H.HARNESS_VERSION}
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        res["seconds"] = time.time() - t0
        outp.write_text(json.dumps(res, indent=1))

    rule = json.loads(a.rule) if a.rule else {"rule": "nearest"}
    if a.select:
        res["selection"] = H.select_rule(tok, device=dv)
        rule = res["selection"]["chosen"]
        print("selection", json.dumps(res["selection"]["mean"]), "chosen", rule, flush=True)
        dump()
    res["rule"] = rule
    if a.read:
        res["read"] = H.read_candidate(tok, rule, device=dv) if sp == "val" else \
            {d: H.score(H.Candidate(tok, rule), d, sp, device=dv, cache=False, also=[{"rule": "nearest"}]) for d in sets.DATASETS}
        res["bar"] = H.bar(res["read"])
        for d, v in res["read"].items():
            print(d, "ceiling", round(v["ceiling"]["mIoU"], 4), "nearest", round(v["also"].get("nearest", v["ceiling"])["mIoU"], 4),
                  "boundary", round(v["ceiling"]["boundary_acc"], 4), "codes", v["codes"]["read_distinct"],
                  "entropy", round(v["codes"]["read_entropy_bits"], 2), "unseen", v["codes"]["unseen_token_share"], flush=True)
        print("bar", res["bar"], flush=True)
        dump()
    if a.native256:
        res["cloudsen12_native256"] = H.score(H.Candidate(tok, rule, crop_mode="native256"), "cloudsen12", sp,
                                              also=[{"rule": "nearest"}], device=dv, cache=sp == "val")
        print("cloudsen12 native256", round(res["cloudsen12_native256"]["ceiling"]["mIoU"], 4), flush=True)
        dump()
    if a.controls:
        ctl = {"determinism": determinism(tok, sp, dv)}
        print("determinism", json.dumps(ctl["determinism"]), flush=True)
        ctl["transposed"], ctl["transposed_identity"] = {}, {}
        ident = Identity(tok.input_size, dv)
        for d in sets.DATASETS:
            ctl["transposed"][d] = short(H.score(H.Candidate(Transposed(tok), rule), d, sp, cache=False,
                                                 code_stats=False, device=dv))
            ctl["transposed_identity"][d] = short(H.score(H.Candidate(Transposed(ident), {"rule": "nearest"}), d, sp,
                                                          cache=False, code_stats=False, device=dv))
            print("transposed", d, ctl["transposed"][d]["mIoU"], "identity transposed", ctl["transposed_identity"][d]["mIoU"], flush=True)
        dump()
        torch.manual_seed(1234)
        untrained = T.BTokenizer(T.MapTok(tok.model.cfg), name=f"B_untrained_{tok.input_size}_{tok.n_codes}_seed1234",
                                 weights=None, sha="untrained_seed1234", batch=tok.batch, device=dv)
        ctl["untrained"] = {d: short(H.score(H.Candidate(untrained, rule), d, sp, cache=False, code_stats=False,
                                             device=dv)) for d in sets.DATASETS}
        print("untrained", {d: v["mIoU"] for d, v in ctl["untrained"].items()}, flush=True)
        ctl["assignment"] = assignment_check()
        print("assignment", ctl["assignment"], flush=True)
        res["controls"] = ctl
        dump()
    dump()


if __name__ == "__main__":
    main()
