#!/usr/bin/env python
"""Do decode_eo's clean references match the cleaned tokens? (next steps §3, 2026-10-11)

E5/E6 score decoded generations against raw rasters. On the cleaned token set the raw tars
are not cleaned, so decode_eo.read_rasters(..., "clean") corrects S2L2A and recomputes NDVI in
the offset rows. This checks that those references are what the val_clean/ tokens encode.

  A  S2L2A: offset rows are the stored raster - 1000 DN, clean rows unchanged (exact)
  B  NDVI: offset rows are ndvi_from_s2(corrected S2L2A) and differ from the stored NDVI;
     clean rows unchanged (exact)
  C  re-encoding the clean references (GTX Titan X, batch 32, tail-padded) reproduces the
     val_clean tokens (>= 0.999 per modality; the rest are FSQ-boundary flips)
CONTROLS
  C1 the ORIGINAL references re-encoded do NOT reproduce val_clean in the offset rows (<= 0.5)
  C2 ...but do reproduce the original tokens there (>= 0.99; the originals came from the TITAN V)
  C3 a stored NDVI that is not TerraMesh's formula on the stored bands is refused (RuntimeError)

`mor` env, GTX Titan X:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python eo/experiments/clean_refs_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import decode_eo as D  # noqa: E402
from terramesh_tok import clean as CL, preprocess as P, tokenizers as T  # noqa: E402

BATCH = 32
OUT = Path("/data/enric/reports/s2_offset/clean_refs_check.json")


@torch.no_grad()
def encode(tok, mod, arrs):
    xs = [P.prepare(a, mod, device="cpu")[0] for a in arrs]
    out = []
    for i in range(0, len(xs), BATCH):
        b = torch.cat(xs[i:i + BATCH])
        n = b.shape[0]
        if n < BATCH:
            b = torch.cat([b, b[-1:].repeat(BATCH - n, 1, 1, 1)])
        out.append(P.flatten_tokens(T.encode(tok, b.to("cuda:0"))).cpu().numpy()[:n])
    return np.concatenate(out).astype(np.int64)


def main() -> int:
    if torch.cuda.device_count() != 1 or "TITAN X" not in torch.cuda.get_device_name(0):
        raise SystemExit("pin the GTX Titan X: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0")
    from eo.data.eval_split import load_eval_rows  # noqa: E402  (pure numpy/pandas)
    lab = CL.load_labels(D.token_root("clean")).set_index("row")
    ev = [int(r) for r in load_eval_rows(root_dir=str(D.VAL))]
    present = {m: np.load(D.VAL / D.C.tok_dir_name(m) / "present.npy") for m in CL.CLEANED_MODALITIES}
    ok_rows = [r for r in ev if all(present[m][r] for m in CL.CLEANED_MODALITIES)]
    off = [r for r in ok_rows if lab.offset[r]][:24]
    cln = [r for r in ok_rows if not lab.offset[r]][:8]
    rows = np.array(off + cln)
    is_off = np.array([True] * len(off) + [False] * len(cln))
    res, ok = {"rows_offset": off, "rows_clean": cln}, True

    raw = {m: D._read_raw(m, rows) for m in CL.CLEANED_MODALITIES}
    ref = {m: D.read_rasters(m, rows, "clean") for m in CL.CLEANED_MODALITIES}
    orig = {m: D.read_rasters(m, rows, "original") for m in CL.CLEANED_MODALITIES}

    a = all(np.array_equal(ref["S2L2A"][i], (raw["S2L2A"][i].astype(np.int32) - 1000).astype(raw["S2L2A"][i].dtype))
            if is_off[i] else np.array_equal(ref["S2L2A"][i], raw["S2L2A"][i]) for i in range(len(rows)))
    b = all((np.array_equal(ref["NDVI"][i], CL.ndvi_from_s2(CL.correct_s2(raw["S2L2A"][i], True)).reshape(raw["NDVI"][i].shape))
             and not np.array_equal(ref["NDVI"][i], raw["NDVI"][i]))
            if is_off[i] else np.array_equal(ref["NDVI"][i], raw["NDVI"][i]) for i in range(len(rows)))
    print(f"A  S2L2A references: offset rows -1000 DN, clean rows unchanged = {a}")
    print(f"B  NDVI references: recomputed in offset rows, clean rows unchanged = {b}")
    res.update(A=a, B=b)
    ok &= a and b

    for m in CL.CLEANED_MODALITIES:
        tok = T.build(m, device="cuda:0")
        got = encode(tok, m, ref[m])
        got_o = encode(tok, m, [x for x, o in zip(orig[m], is_off) if o])
        want = D.read_tokens(m, rows, "clean").reshape(len(rows), -1)
        want_o = D.read_tokens(m, rows[is_off], "original").reshape(int(is_off.sum()), -1)
        agree = float((got == want).mean())
        c1 = float((got_o == want[is_off]).mean())
        c2 = float((got_o == want_o).mean())
        res[m] = {"clean_refs_vs_clean_tokens": agree, "orig_refs_vs_clean_tokens_offset_rows": c1,
                  "orig_refs_vs_orig_tokens_offset_rows": c2}
        print(f"C  {m}: clean references -> val_clean tokens {agree:.5f} (>= 0.999)")
        print(f"   C1 original references -> val_clean tokens, offset rows {c1:.4f} (<= 0.5)")
        print(f"   C2 original references -> original tokens, offset rows  {c2:.5f} (>= 0.99)")
        ok &= agree >= 0.999 and c1 <= 0.5 and c2 >= 0.99
        del tok
        torch.cuda.empty_cache()

    real = D._read_raw

    def broken(mod, rr):
        out = real(mod, rr)
        if mod == "NDVI":
            out[0] = (out[0].astype(np.float32) + 0.01).astype(out[0].dtype)
        return out
    D._read_raw = broken
    try:
        D.read_rasters("NDVI", rows[:1], "clean")
        c3 = False
    except RuntimeError:
        c3 = True
    finally:
        D._read_raw = real
    print(f"C3 a stored NDVI off TerraMesh's formula is refused = {c3}")
    res["C3"] = c3
    ok &= c3

    res["pass"] = bool(ok)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(("PASS" if ok else "FAIL") + f"  -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
