"""A reference point for agent B (A2 Step 1), not a candidate: TerraMind's own LULC tokenizer read through the
harness on LULC only.

TerraMind-1.0-Tokenizer-LULC is an f16 FSQ tokenizer with 4,375 codes (7-5-5-5-5), 196 tokens at 224 px, trained on
LULC alone with a one-hot input over the 10 TerraMesh classes (eo/terramesh_tok). It is what a LULC specialist at
the short slot's token count and codebook reaches, so it calibrates how far a universal palette tokenizer at f16
(step1_B_tok) can be expected to go on the LULC bar. It cannot encode PASTIS or CloudSen12 maps (other classes and
colours), so only LULC is read.

The wrapper behind eo.maps.tokenizers.MapTokenizer: the harness's render (fixed LULC assignment: class c -> colour
c, no data on white = colour 0) is mapped back to classes exactly (step1_B_tok.rgb_to_index; any colour outside
0..9 is refused), one-hot over 10 classes as eo/terramesh_tok/preprocess.py does, encoded and decoded by the
TerraMind tokenizer (fp32; terratorch's import sets matmul precision "medium", reset to "highest" here), and its 10
output channels become palette logits 0..9 (colours 10..31 at -1e4), output = "logits".

    python -X faulthandler -u -m eo.experiments.segmentation.step1_B_ref_terramind --out <json> --expect-gpu "TITAN V"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from eo.experiments.segmentation.step1_B_tok import colour_lut, rgb_to_index
from eo.maps import harness as H
from eo.maps import palette as P
from eo.maps.tokenizers import MapTokenizer, sha256


class TerraMindLULC(MapTokenizer):
    output = "logits"

    def __init__(self, device: str = "cuda", batch: int = 16):
        from eo.terramesh_tok import tokenizers as TT
        from eo.terramesh_tok import contract as C
        self.TT = TT
        self.tok = TT.build("LULC", device)
        torch.set_float32_matmul_precision("highest")
        self.name, self.input_size, self.n_codes, self.batch, self.device = "terramind_lulc_ref_224", 224, \
            C.CODEBOOK["LULC"], batch, device
        self.n_cls = C.LULC_N_CLASSES
        self.lut = colour_lut().to(device)

    @torch.no_grad()
    def encode(self, x):
        idx = rgb_to_index(x.to(self.device), self.lut)
        if bool((idx >= self.n_cls).any()):
            raise ValueError("not a LULC render under the fixed assignment (a colour outside 0..9)")
        oh = F.one_hot(idx, self.n_cls).permute(0, 3, 1, 2).float()
        return self.TT.encode(self.tok, oh).long()

    @torch.no_grad()
    def decode(self, codes):
        rec = self.TT.decode(self.tok, codes.to(self.device)).float()          # (B, 10, S, S)
        out = torch.full((rec.shape[0], P.N_COLOURS, *rec.shape[-2:]), -1e4, device=rec.device)
        out[:, :self.n_cls] = rec
        return out

    def record(self):
        from eo.terramesh_tok import contract as C
        return {"kind": "terramind_lulc_reference", "registry": C.TOKENIZER["LULC"],
                "sha256": {"note": "HF snapshot of ibm-esa-geospatial/TerraMind-1.0-Tokenizer-LULC"},
                "input_size": 224, "tokens": 196, "n_codes": self.n_codes, "output": "logits",
                "code_sha256": sha256(str(Path(__file__).resolve()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="monitor,val")
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    t0 = time.time()
    tok = TerraMindLULC()
    res = {"tokenizer": tok.record(), "gpu": gpu, "harness": H.HARNESS_VERSION, "dataset": "lulc", "reads": {}}
    for sp in a.splits.split(","):
        r = H.score(H.Candidate(tok, {"rule": "nearest"}), "lulc", sp, cache=False, code_stats=sp == "val",
                    also=[{"rule": "vote"}])
        res["reads"][sp] = r
        print(sp, "mIoU", round(r["ceiling"]["mIoU"], 4), "vote", round(r["also"]["vote"]["mIoU"], 4),
              "boundary", round(r["ceiling"]["boundary_acc"], 4), "pixel", round(r["ceiling"]["pixel_acc"], 4),
              "iou", [None if v is None else round(v, 3) for v in r["ceiling"]["iou_per_class"]], flush=True)
    res["seconds"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
