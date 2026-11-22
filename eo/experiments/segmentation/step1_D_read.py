"""Read the fine-tuned Cosmos (A2 Step 1 candidate (c), step1_D_tok) through the harness, with B's controls.

The same read as agent B's (step1_B_read: --select, --read, --controls; its helpers imported as they are), with
step1_D_tok's loader; the untrained control is Cosmos's architecture at a random init (seed 1234), and a second
reference reads Cosmos as released with the nearest-colour head (the fine-tune's step 0). Writes B's JSON layout,
so step1_O_shortlist's row builder reads it.

    python -X faulthandler -u -m eo.experiments.segmentation.step1_D_read --ckpt <best.pt> --out <read_best.json> \
        --select --read --controls --expect-gpu "TITAN X"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eo.experiments.segmentation import step1_D_tok as T
from eo.experiments.segmentation.step1_B_read import Transposed, assignment_check, determinism, short
from eo.maps import harness as H
from eo.maps import palette as P
from eo.maps import sets
from eo.maps.tokenizers import Identity

INIT = "/data/enric/reports/a2_seg/step1/D/cosmos/cosmos_tokenize1_di16x16_network.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--init", default=INIT, help="Cosmos as released, for the step-0 reference control")
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

    rule = {"rule": "nearest"}
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
        untrained = T.DTokenizer(T.MapTokD(tok.model.cfg), name=f"D_untrained_{tok.input_size}_seed1234",
                                 weights=None, sha="untrained_seed1234", batch=tok.batch, device=dv)
        ctl["untrained"] = {d: short(H.score(H.Candidate(untrained, rule), d, sp, cache=False, code_stats=False,
                                             device=dv)) for d in sets.DATASETS}
        print("untrained", {d: v["mIoU"] for d, v in ctl["untrained"].items()}, flush=True)
        del untrained
        init_model, _ = T.from_cosmos(a.init, tok.model.cfg)
        released = T.DTokenizer(init_model, name="D_cosmos_released_head", weights=None, sha="cosmos_released",
                                batch=tok.batch, device=dv)
        ctl["released_with_head"] = {d: short(H.score(H.Candidate(released, rule), d, sp, cache=False,
                                                      code_stats=False, device=dv)) for d in sets.DATASETS}
        print("released with head", {d: v["mIoU"] for d, v in ctl["released_with_head"].items()}, flush=True)
        ctl["assignment"] = assignment_check()
        print("assignment", ctl["assignment"], flush=True)
        res["controls"] = ctl
        dump()
    dump()


if __name__ == "__main__":
    main()
