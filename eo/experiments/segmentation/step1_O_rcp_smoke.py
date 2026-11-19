"""A2 Step 1: the RCP smoke test of the shared harness (orchestrator). Run on one lts4 A100 by
/data/enric/reports/a2_seg/queues/rcp/submit.sh, from a frozen code snapshot.

Checks, each printed PASS/FAIL:
  - the card is an A100 and the env's torch sees it;
  - every fixed map set loads from A2_MAPS_ROOT and matches the manifest's sha256 (sets.load checks it);
  - the harness's own path with a perfect (identity) tokenizer reads exactly 1.0 on PASTIS fold 4 (224 and
    256 px) and CloudSen12 val (224 px, the 3 x 3 tiling), as gate Z2 does on the workstation;
  - a bf16 conv stack runs, and its speed (for planning).
"""
from __future__ import annotations

import os
import time

import torch

from eo.maps import harness as H
from eo.maps import sets
from eo.maps.tokenizers import Identity


def main():
    ok = True
    name = torch.cuda.get_device_name(0)
    print(f"torch {torch.__version__}, card {name}, A2_MAPS_ROOT {os.environ.get('A2_MAPS_ROOT')}", flush=True)
    ok &= "A100" in name
    for d in sets.DATASETS:
        for split in ("train", "monitor", "val"):
            lab, ids = sets.load(d, split)
            print(f"  {d}/{split}: {tuple(lab.shape)}", flush=True)
    print("PASS map sets load with their manifest sha256", flush=True)
    for d, size in (("pastis", 224), ("pastis", 256), ("cloudsen12", 224)):
        r = H.sweep(Identity(size), d, "val", [{"rule": "nearest"}, {"rule": "vote"}], cache=False)
        good = all(v["mIoU"] == 1.0 for v in r.values())
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} identity {d}@{size}: " +
              ", ".join(f"{k} {v['mIoU']:.6f}" for k, v in r.items()), flush=True)
    net = torch.nn.Sequential(*[torch.nn.Conv2d(128, 128, 3, padding=1) for _ in range(8)]).cuda()
    x = torch.randn(32, 128, 112, 112, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            net(x).sum().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            net(x).sum().backward()
        torch.cuda.synchronize()
    print(f"bf16 conv stack: {(time.time() - t0) / 10 * 1000:.1f} ms per forward+backward", flush=True)
    print("SMOKE", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
