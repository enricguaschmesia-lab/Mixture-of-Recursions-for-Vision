#!/usr/bin/env python
"""A0.1's A10c-NC read for its depth allocation, as B0 reads one seed (Paper A plan, A0.1: "descriptively, since one
seed's allocation is arbitrary"; 2026-11-15).

A10c-NC is A10c's recipe without Coords (seed 42, so A10c's initial weights, but every row's modality order differs).
Its router's state at every checkpoint comes from `capture_router.py` (main set, fp16, its own Coords-free held-out
sequences, GTX Titan X): /data/enric/reports/wave1/router/a10cnc/checkpoint-*/main_fp16.npz. Each image modality's
mean depth at its label positions is averaged over B0's windows (stored checkpoints, ends included: early 330-990,
plateau 990-2970, late 1980-3300, anneal 2970-3300; each checkpoint weighted equally), with the time part of B0's
interval (t x SD / sqrt(n); the rows' bootstrap part, under 0.001 passes in B0's readings, is left out). It is set
beside A10c and A10c-s2 (B0 Step 4's agree.json): Spearman over the six modalities with its exact two-sided p, the
largest |difference|, and whether S2L2A and LULC are the two shallowest (B0's prediction P1 for a new seed, written for
A10c seed 44 with Coords). Read by no rule.

CONTROL, which must hold or nothing is written: the same reader on A10c's stored captures (B0 Step 1,
/data/enric/reports/b0/router/a10c) reproduces agree.json's A10c plateau means to 1e-9; and A10c-NC's captures hold no
Coords label.

    .venv/bin/python eo/experiments/routing/b3_a10cnc_allocation.py

Writes /data/enric/reports/wave1/router/a10cnc_allocation.{txt,json}. Runs in `.venv`, CPU, seconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.routing import stats as S  # noqa: E402
from eo.experiments.routing.b0_agree import check_t, t_quantile  # noqa: E402

NC = Path("/data/enric/reports/wave1/router/a10cnc")
B0R = Path("/data/enric/reports/b0/router")
AGREE = Path("/data/enric/reports/b0/agree/agree.json")
OUT = Path("/data/enric/reports/wave1/router")
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
IMG = MODS[:6]
CKS = [330 * i for i in range(1, 11)]
WIN = {"early": [330, 660, 990], "plateau": [990, 1320, 1650, 1980, 2310, 2640, 2970],
       "late": [1980, 2310, 2640, 2970, 3300], "anneal": [2970, 3300]}


def per_ck(root: Path, ck: int) -> dict:
    z = np.load(root / f"checkpoint-{ck}" / "main_fp16.npz")
    d, lm = z["depth"].astype(np.int64), z["label_mod"]
    out = {}
    for j, m in enumerate(MODS):
        s = lm == j + 1
        if s.any():
            dd = d[s]
            out[m] = dict(depth=float(dd.mean()), mix=[float((dd == k).mean()) for k in (1, 2, 3)], n=int(s.sum()))
    return out


def window(cells: dict, win: str, m: str) -> dict:
    x = np.array([cells[ck][m]["depth"] for ck in WIN[win]])
    n = len(x)
    half = t_quantile(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    mix = np.mean([cells[ck][m]["mix"] for ck in WIN[win]], 0)
    return dict(mean=float(x.mean()), half=float(half), mix=mix.round(4).tolist())


def main() -> int:
    check_t()
    agree = json.loads(AGREE.read_text())["windows"]
    # CONTROL: this reader on A10c's B0 captures reproduces agree.json's plateau means
    a10c = {ck: per_ck(B0R / "a10c", ck) for ck in WIN["plateau"]}
    ref = {m: agree[f"a10c|fp16|main|plateau"][m]["depth"]["mean"] for m in IMG}
    mine = {m: float(np.mean([a10c[ck][m]["depth"] for ck in WIN["plateau"]])) for m in IMG}
    c_rep = all(abs(mine[m] - ref[m]) < 1e-9 for m in IMG)
    cells = {ck: per_ck(NC, ck) for ck in CKS}
    c_nc = all("Coords" not in cells[ck] for ck in CKS)
    lines, res = [], {"control_reproduces_a10c": c_rep, "control_no_coords": c_nc}

    def P(*a):
        t = " ".join(str(x) for x in a)
        lines.append(t)
        print(t, flush=True)

    P(f"controls: the reader reproduces agree.json's A10c plateau means to 1e-9 = {c_rep} (largest difference "
      f"{max(abs(mine[m] - ref[m]) for m in IMG):.1e}); A10c-NC's captures hold no Coords label = {c_nc}")
    if not (c_rep and c_nc):
        print("CONTROLS FAILED: nothing written")
        return 1
    P("\nA10c-NC per checkpoint, mean depth at each image modality's label positions (fp16, its own sequences):")
    for ck in CKS:
        P(f"  {ck:4d} " + " ".join(f"{m} {cells[ck][m]['depth']:.2f}" for m in IMG))
    res["per_ck"] = {ck: cells[ck] for ck in CKS}
    res["windows"] = {}
    P("\nTime means [95% time part] and the comparison with B0's two seeds (agree.json, A10c's sequences with Coords):")
    for win in WIN:
        w = {m: window(cells, win, m) for m in IMG}
        res["windows"][win] = w
        dn = [w[m]["mean"] for m in IMG]
        shallow = sorted(IMG, key=lambda m: w[m]["mean"])[:2]
        P(f"  {win:7s} A10c-NC " + " ".join(f"{m} {w[m]['mean']:.2f}±{w[m]['half']:.2f}" for m in IMG)
          + f" | two shallowest: {', '.join(shallow)}")
        for run in ("a10c", "a10cs2"):
            dr = [agree[f"{run}|fp16|main|{win}"][m]["depth"]["mean"] for m in IMG]
            rho, p = S.spearman(dn, dr), S.exact_p(dn, dr)
            mx = max(abs(a - b) for a, b in zip(dn, dr))
            res["windows"][win][f"vs_{run}"] = dict(rho=rho, p=p, max_abs=mx, depths=dr)
            P(f"          vs {run:6s} rho {rho:+.2f} [exact p {p:.3f}], largest |difference| {mx:.2f}  ({run}: "
              + " ".join(f"{m} {d:.2f}" for m, d in zip(IMG, dr)) + ")")
        res["windows"][win]["two_shallowest"] = shallow
    p1 = all(set(res["windows"][w]["two_shallowest"]) == {"S2L2A", "LULC"} for w in ("plateau", "late"))
    res["P1_like"] = p1
    P(f"\nS2L2A and LULC the two shallowest on the plateau and late time means (B0's P1, read here by no rule): {p1}")
    (OUT / "a10cnc_allocation.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "a10cnc_allocation.txt").write_text("\n".join(lines) + "\n")
    P(f"-> {OUT / 'a10cnc_allocation.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
