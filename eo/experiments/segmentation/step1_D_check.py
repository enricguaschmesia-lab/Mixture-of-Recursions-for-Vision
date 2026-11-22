"""A2 Step 1 (c), before the fine-tune: the rebuilt Cosmos DI16x16 (step1_D_tok) reproduces NVIDIA's released JITs,
and its palette head starts as the nearest-colour rule. Each check has a control that must fail. TITAN V (the JITs
run in bf16 only), through the watchdog unit.

  1. Encoder, at bf16 on the same renders (fixed task assignments, 224 px): the rebuilt encoder against the JIT's
     `encoder` submodule (max |diff| / max |h|), and codes: the JIT's full encoder against (a) the rebuilt bf16
     path, (b) Cosmos's FSQ applied in fp32 to the JIT's own latent (the share FSQ's precision alone moves), and
     (c) the rebuilt fp32 path through run_fixed (what the fine-tune starts from). Control: the patcher with its
     LH and HL bands swapped.
  2. Decoder: from the JIT's codes, the JIT decoder's RGB against the rebuilt one (bf16 and fp32).
  3. Head: argmax of the initial head = the universal colour nearest the decoded RGB, at every pixel. Control:
     the head's rows permuted.
  4. End to end on the monitor maps (never val): harness.sweep of the rebuilt model (fp32, head, rule nearest)
     beside the JIT (RGB, rule nearest), per dataset.

    bash /data/enric/reports/a2_seg/queues/step4_titanv_unit.sh d_check 3600 \
        python -X faulthandler -u -m eo.experiments.segmentation.step1_D_check --out .../D/check_init.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from eo.experiments.segmentation import step1_D_tok as T
from eo.maps import harness as H
from eo.maps import palette as P
from eo.maps import sets
from eo.maps.tokenizers import CosmosDI, deterministic, run_fixed

INIT = "/data/enric/reports/a2_seg/step1/D/cosmos/cosmos_tokenize1_di16x16_network.pt"


def renders(n_per: dict, size: int, device: str) -> torch.Tensor:
    xs = []
    for d, n in n_per.items():
        lab = torch.as_tensor(np.ascontiguousarray(sets.load(d, "monitor")[0][:n]), device=device).long()
        if d == "cloudsen12":
            lab = lab[:, :224, :224]
        xs.append(P.render(lab, P.TASKS[d].assign, size))
    return torch.cat(xs)


class SwappedBands(T.Patcher):
    """Control: Cosmos's patcher with the LH and HL bands swapped."""

    def _dwt(self, x):
        out = super()._dwt(x)
        a, b, c, d = torch.chunk(out, 4, dim=1)
        return torch.cat([a, c, b, d], dim=1)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=INIT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-gpu", default="TITAN V")
    a = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    t0 = time.time()
    dv = "cuda"
    res = {"init": a.init, "gpu": gpu, "harness": H.HARNESS_VERSION}
    model, info = T.from_cosmos(a.init)
    res["init_info"] = info
    model = model.to(dv).eval()
    cos = CosmosDI("t1_di16", 224, device=dv, batch=16)
    res["jit"] = {k: cos.record()[k] for k in ("hf_repo", "snapshot", "sha256")}
    x = renders({"pastis": 32, "cloudsen12": 16, "lulc": 16}, 224, dv)
    xin = (x * 2 - 1).to(torch.bfloat16)
    mb = copy.deepcopy(model).to(torch.bfloat16)
    ctl = copy.deepcopy(mb)
    ctl.encoder.patcher = SwappedBands(model.cfg["patch_size"]).to(dv).to(torch.bfloat16)
    out = {}
    with deterministic(), torch.no_grad():
        h_jit, h_mine, h_ctl, idx_jit, idx_b, idx_zj, idx_ctl = [], [], [], [], [], [], []
        for s in range(0, len(x), 16):
            xb = xin[s:s + 16]
            hj = cos.enc.encoder(xb)
            h_jit.append(hj.float().cpu())
            hm = mb.encoder(xb)
            h_mine.append(hm.float().cpu())
            hc = ctl.encoder(xb)
            h_ctl.append(hc.float().cpu())
            o = cos.enc(xb)
            idx_jit.append((o[0] if isinstance(o, (tuple, list)) else o).long().cpu())
            idx_b.append(mb.fsq(mb.quant_conv(hm))[1].cpu())
            idx_zj.append(model.fsq(cos.enc.quant_conv(hj))[1].cpu())
            idx_ctl.append(ctl.fsq(ctl.quant_conv(hc))[1].cpu())
        h_jit, h_mine, h_ctl = torch.cat(h_jit), torch.cat(h_mine), torch.cat(h_ctl)
        idx_jit, idx_b, idx_zj, idx_ctl = torch.cat(idx_jit), torch.cat(idx_b), torch.cat(idx_zj), torch.cat(idx_ctl)
    tok = T.DTokenizer(model, "D_init_check", weights=None, sha="cosmos_init", batch=16, device=dv)
    idx32 = run_fixed(tok.encode, x, 16).cpu()
    idx32b = run_fixed(tok.encode, x, 16).cpu()
    out["encoder"] = {
        "renders": len(x), "tokens": int(idx_jit.numel()),
        "h_rel_maxdiff_bf16": rel(h_mine, h_jit), "control_swapped_bands_h_rel_maxdiff": rel(h_ctl, h_jit),
        "codes_equal_bf16_rebuilt": float((idx_b == idx_jit).float().mean()),
        "codes_equal_fp32_fsq_on_jit_latent": float((idx_zj == idx_jit).float().mean()),
        "codes_equal_fp32_rebuilt": float((idx32 == idx_jit).float().mean()),
        "control_swapped_bands_codes_equal": float((idx_ctl == idx_jit).float().mean()),
        "fp32_deterministic_twice": bool(torch.equal(idx32, idx32b)),
    }
    print("encoder", json.dumps(out["encoder"]), flush=True)
    with deterministic(), torch.no_grad():
        rj, rb, r32 = [], [], []
        for s in range(0, len(x), 16):
            ib = idx_jit[s:s + 16].to(dv)
            rj.append(cos.dec(ib).float().cpu())
            rb.append(mb.rgb(mb.fsq.codes_to_q(ib).to(torch.bfloat16)).float().cpu())
            r32.append(model.rgb(model.fsq.codes_to_q(ib)).float().cpu())
        rj, rb, r32 = torch.cat(rj), torch.cat(rb), torch.cat(r32)
    out["decoder"] = {"rgb_maxabs_bf16": float((rb - rj).abs().max()), "rgb_meanabs_bf16": float((rb - rj).abs().mean()),
                      "rgb_maxabs_fp32": float((r32 - rj).abs().max()), "rgb_meanabs_fp32": float((r32 - rj).abs().mean()),
                      "rgb_range_jit": [float(rj.min()), float(rj.max())]}
    print("decoder", json.dumps(out["decoder"]), flush=True)
    with torch.no_grad():
        c = torch.as_tensor(np.asarray(P.UNIVERSAL), dtype=torch.float32) * 2 - 1
        lg = model.head.cpu()(r32)
        nearest = ((r32[:, None] - c[None, :, :, None, None]) ** 2).sum(2).argmin(1)
        perm = torch.randperm(32, generator=torch.Generator().manual_seed(0))
        lg_ctl = lg[:, perm]
        model.head.to(dv)
    out["head"] = {"argmax_equals_nearest": float((lg.argmax(1) == nearest).float().mean()),
                   "control_rows_permuted": float((lg_ctl.argmax(1) == nearest).float().mean()),
                   "scale": model.cfg["head_scale"]}
    print("head", json.dumps(out["head"]), flush=True)
    res["checks"] = out
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    e2e = {}
    for d in sets.DATASETS:
        r1 = H.sweep(tok, d, "monitor", [{"rule": "nearest"}], cache=False)
        r2 = H.sweep(cos, d, "monitor", [{"rule": "nearest"}], cache=False)
        e2e[d] = {"rebuilt_head_nearest": r1["nearest"]["mIoU"], "jit_rgb_nearest": r2["nearest"]["mIoU"]}
        print("monitor", d, json.dumps(e2e[d]), flush=True)
    out["monitor_end_to_end"] = e2e
    res["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1))
    ok = (out["encoder"]["h_rel_maxdiff_bf16"] < 0.05 and out["encoder"]["control_swapped_bands_h_rel_maxdiff"] > 0.2
          and out["encoder"]["codes_equal_fp32_rebuilt"] > max(0.5, 3 * out["encoder"]["control_swapped_bands_codes_equal"])
          and out["head"]["argmax_equals_nearest"] > 0.9999 and out["head"]["control_rows_permuted"] < 0.5
          and out["encoder"]["fp32_deterministic_twice"])
    res["pass"] = bool(ok)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
