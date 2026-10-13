#!/usr/bin/env python
"""Phase 6 L7, decode stage: ben_gen_forgetting.py's generations -> decoded errors. `mor` env, GPU.

⚠ RUNS IN THE `mor` CONDA ENV. Reuses Phase 3 12.6b's decoder settings, scoring and raster reader
by IMPORTING eo/experiments/s12_decoded_eval.py (never edited here: another workstream runs it).
Adds what L7 needs and 12.6b did not record:
  S2L2A  p1min: per scene, the minimum over bands of each band's 1st percentile, in DN, of the
         decoded image (and of the decoded TRUE tokens = the ceiling). clean.py's cross-check rule
         calls a scene offset-like when it exceeds 900 DN (6.2a).
  LULC   the 10x10 confusion over all scenes of a cell, and its mIoU (classes present only).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
      python eo/experiments/ben_gen_forgetting_decode.py --tags base b10_full b10_lora ...

Outputs: <out-root>/<T>/errors_<tag>.npz, errors_ceiling.npz (the 12.6b layout, plus the keys above).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import s12_decoded_eval as S12  # noqa: E402  (sets up terratorch, decode_eo, clamp_test)
import numpy as np  # noqa: E402

CELLS = {"S2L2A": ["DEM", "S1GRD", "none"], "LULC": ["S2L2A"]}


def p1min(target, out):
    """(N,C,H,W) decoded z -> (N,) min over bands of the 1st percentile, physical units."""
    v = []
    for k in range(len(out)):
        ph = S12.P.destandardize(out[k], target).numpy()
        v.append(float(np.nanpercentile(ph.reshape(ph.shape[0], -1), 1, axis=1).min()))
    return np.array(v)


def lulc_conf(out, refs):
    c = np.zeros((10, 10), np.int64)
    for k in range(len(out)):
        pred = out[k].argmax(0).numpy().ravel()
        ref = refs[k][0].astype(np.int64).ravel()
        c += np.bincount(10 * ref + pred, minlength=100).reshape(10, 10)
    return c


def miou(c):
    inter = np.diag(c).astype(float)
    union = c.sum(0) + c.sum(1) - np.diag(c)
    ok = union > 0
    return float((inter[ok] / union[ok]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--gen-root", default="/data/enric/generations/p6_gen_forgetting")
    ap.add_argument("--out-root", default="/data/enric/reports/ben/gen_forgetting")
    ap.add_argument("--allow-titanv", action="store_true")
    a = ap.parse_args()
    gen, outr = pathlib.Path(a.gen_root), pathlib.Path(a.out_root)
    gpu = S12.D.assert_gpu(a.allow_titanv)
    print(f"gpu {gpu}", flush=True)
    for T, cells in CELLS.items():
        odir = outr / T
        odir.mkdir(parents=True, exist_ok=True)
        R = np.load(gen / a.tags[0] / T / "cells.npz")["column_rows"]
        for t in a.tags:
            assert np.array_equal(np.load(gen / t / T / "cells.npz")["column_rows"], R), f"{t} {T}: other rows"
        tok = S12.TK.build(T, device=S12.TK.DEVICE)
        sched = S12.CT.scheduler_like(tok, **S12.CT.VARIANTS["noclip"]) if S12.SETTING[T] == "unclamped" else None
        refs = [S12.P.center_crop(x[0] if x.ndim == 4 else x).astype(np.float32) for x in S12.D.read_rasters(T, R, "original")]
        pos = {int(r): i for i, r in enumerate(R)}
        if not (odir / "errors_ceiling.npz").exists():
            out = S12.decode(tok, T, S12.D.read_tokens(T, R, "original"), sched)
            e, eml, _ = S12.score(T, out, refs)
            extra = {f"ceiling|{T}|p1min": p1min(T, out),
                     f"ceiling|{T}|ref_p1min": np.array([float(np.nanpercentile(r.reshape(r.shape[0], -1), 1, axis=1).min()) for r in refs])} \
                if T == "S2L2A" else {f"ceiling|{T}|miou": np.float64(miou(lulc_conf(out, refs)))}
            np.savez(odir / "errors_ceiling.npz", **{f"ceiling|{T}|rows": R, f"ceiling|{T}|err": e,
                                                     f"ceiling|{T}|err_ml7": eml, **extra})
            print(f"ceiling {T}: error {e.mean():.4f} ml7 {eml.mean():.4f}", flush=True)
        for t in a.tags:
            f = odir / f"errors_{t}.npz"
            if f.exists():
                print(f"[skip] {f}")
                continue
            z = np.load(gen / t / T / "cells.npz")
            arrays, t0 = {}, time.time()
            for S in cells:
                rows, grids = z[f"{S}|rows"], z[f"{S}|grids"]
                out = S12.decode(tok, T, grids, sched)
                rr = [refs[pos[int(r)]] for r in rows]
                e, eml, _ = S12.score(T, out, rr)
                arrays.update({f"{t}|{S}|rows": rows, f"{t}|{S}|err": e, f"{t}|{S}|err_ml7": eml,
                               f"{t}|{S}|valid_frac": z[f"{S}|valid"].mean(1)})
                if T == "S2L2A":
                    arrays[f"{t}|{S}|p1min"] = p1min(T, out)
                else:
                    c = lulc_conf(out, rr)
                    arrays[f"{t}|{S}|confusion"], arrays[f"{t}|{S}|miou"] = c, np.float64(miou(c))
                print(f"  {t} {S:>6}->{T:<5} n {len(rows):3d} error {e.mean():.4f} ml7 {eml.mean():.4f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            tmp = odir / f"errors_{t}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, f)
    print("decode done", flush=True)


if __name__ == "__main__":
    main()
