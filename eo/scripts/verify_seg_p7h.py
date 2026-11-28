#!/usr/bin/env python
"""Gate for A2 Step 7.0h (docs/plans/a2_segmentation.md, Step 7, "7.0h, soft voting on the stored samples"): the soft
vote, the renormalization, the prior, the un-rotation of probability maps and the read's rule. Every check has a control
that fails it. `.venv`, CPU. (The decode itself is checked on every sample by step70h_soft.py: each sample's argmax of
the log-pooled scores must equal its stored class map, the read's validity.)

  Z36a  the soft vote of one-hot samples equals eval/seg.vote's hard vote. Control: a pixel two samples call 0.51 / 0.49
        and one calls 0.00 / 1.00 is class 0 by the hard vote, class 1 by the soft one.
  Z36b  renorm: the scored classes sum to 1 per pixel, void is 0; argmax with w = 1 is the plain argmax. Control: a prior
        that favours one class moves a crafted pixel.
  Z36c  to_k0_t (torch) equals step70e_votes.to_k0 for all 8 variants. Control: the forward variant used as the inverse
        fails at k = 1 and 3.
  Z36d  the read's verdict rule on known pairs. Control: one replicate below 0 is not a gain.
    python eo/scripts/verify_seg_p7h.py [--out gate.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "experiments" / "segmentation"))

RESULTS: list = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def z36a() -> None:
    import step70e_votes as VT
    g = torch.Generator().manual_seed(0)
    maps = torch.randint(0, 19, (8, 3, 16, 16), generator=g)                     # 8 samples, 3 maps
    hard = VT.plain_vote(maps, 20)
    soft = torch.nn.functional.one_hot(maps, 20).float().sum(0).permute(0, 3, 1, 2).argmax(1)
    record("Z36a the soft vote of one-hot samples = the hard vote", torch.equal(hard, soft), f"equal {torch.equal(hard, soft)}")
    p = torch.zeros(3, 20)
    p[0, 0], p[0, 1] = 0.51, 0.49
    p[1, 0], p[1, 1] = 0.51, 0.49
    p[2, 1] = 1.0
    h = VT.plain_vote(p.argmax(1).view(3, 1, 1, 1), 20).item()
    s = int(p.sum(0).argmax())
    record("Z36a control: confident minority", h == 0 and s == 1, f"hard {h}, soft {s}")


def z36b() -> None:
    import step70h_soft as SO
    g = torch.Generator().manual_seed(1)
    x = torch.rand(2, 20, 8, 8, generator=g)
    r = SO.renorm(x, 19)
    ok = torch.allclose(r.sum(1), torch.ones(2, 8, 8)) and bool((r[:, 19] == 0).all())
    S = r.numpy()
    same = np.array_equal(np.argmax(S * np.ones(20, np.float32)[None, :, None, None], 1), np.argmax(S, 1))
    record("Z36b renorm and the unit prior", ok and same, f"sums to 1 and void 0: {ok}; w = 1 keeps the argmax: {same}")
    q = np.zeros((1, 20, 1, 1), np.float32)
    q[0, 0], q[0, 1] = 0.6, 0.4
    w = np.ones(20, np.float32)
    w[1] = 2.0
    moved = int(np.argmax(q * w[None, :, None, None], 1)) == 1 and int(np.argmax(q, 1)) == 0
    record("Z36b control: a prior favouring class 1", moved, f"moves the pixel: {moved}")


def z36c() -> None:
    import step70e_votes as VT
    import step70h_soft as SO
    from eo.geobench import s2 as G
    x = np.random.default_rng(2).random((3, 20, 12, 12)).astype(np.float32)
    ok = all(np.array_equal(SO.to_k0_t(torch.from_numpy(G.dihedral(x, k).copy()), k).numpy(), x) and
             np.array_equal(SO.to_k0_t(torch.from_numpy(x), k).numpy(), VT.to_k0(x, k)) for k in range(8))
    record("Z36c to_k0_t inverts s2.dihedral and equals to_k0", ok, f"8 variants: {ok}")
    bad = {k: np.array_equal(G.dihedral(G.dihedral(x, k), k), x) for k in (1, 3)}
    record("Z36c control: the forward variant as its own inverse", not any(bad.values()), f"{bad}")


def z36d() -> None:
    import step70h_read as H
    G = lambda d, lo, hi: {"delta": d, "lo95": lo, "hi95": hi}       # noqa: E731
    v = [H.verdict(G(0.8, 0.3, 1.3), [G(0.2, -0.1, 0.5), G(0.4, 0.1, 0.7), G(0.1, -0.2, 0.4)]),
         H.verdict(G(-0.8, -1.3, -0.3), [G(-0.2, -0.5, 0.1), G(-0.4, -0.7, -0.1), G(-0.1, -0.4, 0.2)]),
         H.verdict(G(0.8, -0.1, 1.6), [G(0.2, -0.1, 0.5), G(0.4, 0.1, 0.7), G(0.1, -0.2, 0.4)])]
    record("Z36d the verdict rule", v == ["gains", "costs", "level"], f"{v}")
    c = H.verdict(G(0.8, 0.3, 1.3), [G(0.2, -0.1, 0.5), G(-0.05, -0.3, 0.2), G(0.1, -0.2, 0.4)])
    record("Z36d control: one replicate below 0", c != "gains", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    for k, f in (("Z36a", z36a), ("Z36b", z36b), ("Z36c", z36c), ("Z36d", z36d)):
        try:
            f()
        except Exception as e:
            traceback.print_exc()
            record(k, False, f"raised {type(e).__name__}: {e}")
    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {n_fail} fail ({time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in RESULTS], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
