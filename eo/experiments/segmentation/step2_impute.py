"""A2 Step 2.3: the B01/B09 imputation for PASTIS, measured where the truth is known (a2seg env, GPU).

PASTIS ships 10 of the S2L2A tokenizer's 12 bands. Following Part 2 Step 1.1 (archived plan), each
candidate is measured on BEN S2 and TerraMesh S2L2A, which carry all 12 bands: drop B01 and B09,
impute them from the other ten, tokenize, and compare with the true-band tokens.

    --fit       the "linear" method: least squares of B01 and B09 on the ten real bands (DN, with an
                intercept), on 2,000 pixels from each of 300 TerraMesh val rows (cleaned S2L2A, 224 px
                centre crop; rows outside eval_rows_v1 and outside this script's evaluation rows)
    --eval      per method and set: token agreement with the true-band tokens (512 rows), the imputed
                bands' own error in z, and the round trip on the ten REAL bands (rmse_z_ml7 with B01
                and B09 excluded, decoder at 50 steps, seed 0) on 24 rows; the true-band round trip
                beside it. Also the identity check: the true-band tokens recomputed here equal the
                stored ones (BEN S2L2A_tok224, TerraMesh val_clean S2L2A_tok224).
                Control: "linear_swapped" (B01 and B09 exchanged) must lose agreement.
    --ben-root  BEN token roots for the functional test (eval_ben.py with BEN_ROOT): ft_valsel's S2
                tokens re-encoded with a method's imputed bands, everything else linked to the original

    python eo/experiments/segmentation/step2_impute.py --fit --eval --expect-gpu "GTX TITAN X"
    python eo/experiments/segmentation/step2_impute.py --ben-root linear,mean --expect-gpu "GTX TITAN X"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/enric/hf")

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo"))
sys.path.insert(0, str(REPO / "eo" / "scripts"))

from eo.ben import layout, preprocess as BP  # noqa: E402
from eo.geobench import s2 as G  # noqa: E402
from eo.terramesh_tok import contract as C, preprocess as P, tokenizers as T  # noqa: E402
import decode_eo as D  # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step2/impute")
TM_VAL = Path("/data/enric/data/TerraMesh/val")
TM_CLEAN = Path("/data/enric/data/TerraMesh/val_clean")
BATCH, TIMESTEPS, SEED = 32, 50, 0
N_TOK, N_DEC, N_FIT, PX_FIT = 512, 24, 300, 2000
IDX10 = [G.ORDER.index(b) for b in G.PASTIS_BANDS]       # the ten real bands in contract positions
IDX_MISSING = [G.ORDER.index(b) for b in G.MISSING]


def gpu_check(expect: str | None) -> str:
    name = torch.cuda.get_device_name(0)
    if expect and expect not in name:
        raise SystemExit(f"expected {expect}, CUDA returned {name}")
    torch.set_float32_matmul_precision("highest")     # terratorch sets "medium" at import (CLAUDE.md)
    return name


def tm_rows() -> dict:
    """TerraMesh val rows: evaluation (N_TOK, the first N_DEC decoded) and fit (N_FIT), disjoint,
    outside eval_rows_v1."""
    idx = pd.read_parquet(TM_VAL / "tok_index.parquet")
    present = np.load(TM_CLEAN / C.tok_dir_name("S2L2A") / "present.npy")
    held = set(json.loads((REPO / "eo/data/eval_rows_v1.json").read_text())["eval_rows"])
    pool = np.array([r for r in idx.row.values if present[r] and r not in held])
    perm = np.random.default_rng(20261119).permutation(pool)
    return {"eval": np.sort(perm[:N_TOK]), "fit": np.sort(perm[N_TOK:N_TOK + N_FIT])}


def tm_physical(rows) -> np.ndarray:
    raw = D.read_rasters("S2L2A", np.asarray(rows), token_set="clean")
    return np.stack([P.center_crop((r[0] if r.ndim == 4 else r).astype(np.float32)) for r in raw])


def fit() -> dict:
    rows = tm_rows()["fit"]
    rng = np.random.default_rng(0)
    X, Y = [], []
    for s in range(0, len(rows), 50):
        ph = tm_physical(rows[s:s + 50])                              # (n, 12, 224, 224) DN
        for a in ph:
            flat = a.reshape(12, -1)
            ok = np.isfinite(flat).all(0)
            sel = rng.choice(np.flatnonzero(ok), PX_FIT, replace=False)
            X.append(flat[IDX10][:, sel].T)
            Y.append(flat[IDX_MISSING][:, sel].T)
    X, Y = np.concatenate(X).astype(np.float64), np.concatenate(Y).astype(np.float64)
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)                      # (11, 2)
    pred = A @ coef
    std = np.asarray(C.V1_TOK_STD["S2L2A"])[IDX_MISSING]
    res = {"rows": rows.tolist(), "pixels": int(len(X)), "bands_in": list(G.PASTIS_BANDS),
           "coef": {b: coef[:, j].tolist() for j, b in enumerate(G.MISSING)},
           "r2": {b: float(1 - ((Y[:, j] - pred[:, j]) ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum())
                  for j, b in enumerate(G.MISSING)},
           "rmse_dn": {b: float(np.sqrt(((Y[:, j] - pred[:, j]) ** 2).mean())) for j, b in enumerate(G.MISSING)},
           "rmse_z": {b: float(np.sqrt(((Y[:, j] - pred[:, j]) ** 2).mean()) / std[j]) for j, b in enumerate(G.MISSING)},
           "source": "TerraMesh val S2L2A, cleaned (eo/terramesh_tok/clean.py), 224 px centre crop"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "linear_fit.json").write_text(json.dumps(res, indent=1))
    print("fit", {b: round(v, 4) for b, v in res["r2"].items()}, "rmse_z", {b: round(v, 4) for b, v in res["rmse_z"].items()})
    return res


def linear_coef() -> dict:
    return json.loads((OUT / "linear_fit.json").read_text())["coef"]


def make_input(phys12: torch.Tensor, method: str, lin: dict) -> torch.Tensor:
    """True 12-band DN (B, 12, H, W) -> the encoder input with B01/B09 replaced by `method`'s."""
    x10 = phys12[:, IDX10]
    if method == "true":
        x12 = phys12
    elif method == "linear_swapped":                  # control: the two imputed bands exchanged
        x12 = G.assemble_pastis(x10, "linear", lin)[:, [IDX_MISSING[1] if i == IDX_MISSING[0] else
                                                       IDX_MISSING[0] if i == IDX_MISSING[1] else i
                                                       for i in range(12)]]
    else:
        x12 = G.assemble_pastis(x10, method, lin)
    return G.standardize(x12)


def encode_fixed(tok, x: torch.Tensor) -> np.ndarray:
    """Batches of exactly BATCH rows, the tail padded by repeating the first row (tokenize_ben's)."""
    out = []
    for s in range(0, len(x), BATCH):
        xb = x[s:s + BATCH]
        n = len(xb)
        if n < BATCH:
            xb = torch.cat([xb, xb[:1].expand(BATCH - n, *xb.shape[1:])])
        out.append(P.flatten_tokens(T.encode(tok, xb)).cpu().numpy()[:n])
    return np.concatenate(out)


def roundtrip_10(tok, x_in: torch.Tensor, ref_phys: np.ndarray) -> dict:
    """Decode the tokens of x_in and score against the TRUE physical image on the ten real bands."""
    recs = []
    for i in range(0, len(x_in), 8):
        t = T.encode(tok, x_in[i:i + 8])
        recs.append(T.decode(tok, t, timesteps=TIMESTEPS, seed=SEED).float().cpu().numpy())
    rec_std = np.concatenate(recs)
    rec_phys = G.destandardize(torch.as_tensor(rec_std)).numpy()
    rows = []
    for ref, rs, rp in zip(ref_phys, rec_std, rec_phys):
        ref = ref.copy()
        ref[IDX_MISSING] = np.nan                     # excluded from the error: only the ten real bands
        rows.append(D.continuous_metrics(rs, rp, ref, "S2L2A"))
    df = pd.DataFrame(rows)
    return {k: {"median": float(df[k].median()), "mean": float(df[k].mean())} for k in ("rmse_z", "rmse_z_ml7")}


def ben_physical(rows: np.ndarray, packed, r2p) -> torch.Tensor:
    pp = r2p[rows]
    return BP.s2_physical(packed["S2_10m"][pp], packed["S2_20m"][pp], packed["S2_60m"][pp], "cuda")


def evaluate(gpu: str) -> dict:
    lin = linear_coef()
    tok = T.build("S2L2A", "cuda")
    methods = ["true", "mean", "copy", "linear", "linear_swapped"]
    res = {"gpu": gpu, "methods": methods, "n_tokens": N_TOK, "n_decoded": N_DEC, "timesteps": TIMESTEPS}

    # BEN: ft_train rows; the decoded 24 are step1_roundtrip's (PHASE6 Step 1.3)
    idx = pd.read_parquet(layout.INDEX)
    packed, r2p = layout.load_packed("S2", len(idx), layout.PACKED)
    pool = np.flatnonzero(idx.ft_train.values & (r2p >= 0))
    dec = np.array(json.loads(Path("/data/enric/reports/ben/step1_roundtrip.json").read_text())["ben_rows"])
    rest = np.setdiff1d(np.random.default_rng(1).choice(pool, N_TOK, replace=False), dec)[: N_TOK - len(dec)]
    ben_rows = np.concatenate([dec, np.sort(rest)])
    stored_ben = np.load(layout.ROOT / C.tok_dir_name("S2L2A") / "tokens.npy", mmap_mode="r")

    tmr = tm_rows()["eval"]
    stored_tm = np.load(TM_CLEAN / C.tok_dir_name("S2L2A") / "tokens.npy", mmap_mode="r")

    for name, rows, stored in (("BEN", ben_rows, stored_ben), ("TerraMesh", tmr, stored_tm)):
        t0 = time.time()
        if name == "BEN":
            phys = torch.cat([ben_physical(rows[s:s + 64], packed, r2p) for s in range(0, len(rows), 64)])
        else:
            phys = torch.cat([torch.as_tensor(tm_physical(rows[s:s + 64]), device="cuda")
                              for s in range(0, len(rows), 64)])
            phys = torch.nan_to_num(phys, nan=0.0)
        toks = {m: encode_fixed(tok, make_input(phys, m, lin)) for m in methods}
        r = {"rows": rows.tolist(),
             "identity_vs_stored": float((toks["true"] == np.asarray(stored[rows])).mean())}
        true_z = G.standardize(phys)
        for m in methods:
            agree = (toks[m] == toks["true"]).mean(1)
            imp_z = make_input(phys, m, lin)[:, IDX_MISSING]
            r[m] = {"token_agreement_mean": float(agree.mean()), "token_agreement_median": float(np.median(agree)),
                    "imputed_rmse_z": {b: float(((imp_z[:, j] - true_z[:, IDX_MISSING[j]]) ** 2).mean().sqrt())
                                       for j, b in enumerate(G.MISSING)}}
        ref = phys[:N_DEC].cpu().numpy()
        for m in methods:
            r[m]["roundtrip_10"] = roundtrip_10(tok, make_input(phys[:N_DEC], m, lin), ref)
        r["seconds"] = round(time.time() - t0, 1)
        res[name] = r
        print(name, f"identity {r['identity_vs_stored']:.6f}", flush=True)
        for m in methods:
            print(f"  {m:15s} agree {r[m]['token_agreement_mean']:.4f}  ml7(10 bands) "
                  f"{r[m]['roundtrip_10']['rmse_z_ml7']['median']:.4f}  imputed z "
                  + " ".join(f"{b} {v:.3f}" for b, v in r[m]["imputed_rmse_z"].items()), flush=True)
    (OUT / "eval.json").write_text(json.dumps(res, indent=1))
    return res


def ben_root(method: str, gpu: str) -> Path:
    """A BEN root whose S2L2A tokens on ft_valsel are re-encoded with `method`'s imputed B01/B09."""
    lin = linear_coef()
    root = OUT / f"ben_root_{method}"
    src = layout.ROOT
    (root / C.tok_dir_name("S2L2A")).mkdir(parents=True, exist_ok=True)
    for f in ("index_v1.parquet", "index_v1.json", C.tok_dir_name("S1GRD"), "splits"):
        if not (root / f).exists() and (src / f).exists():
            os.symlink(src / f, root / f)
    idx = pd.read_parquet(layout.INDEX)
    rows = np.flatnonzero(idx.ft_valsel.values)
    packed, r2p = layout.load_packed("S2", len(idx), layout.PACKED)
    order = rows[np.argsort(r2p[rows], kind="stable")]
    tok = T.build("S2L2A", "cuda")
    d = root / C.tok_dir_name("S2L2A")
    shutil.copyfile(src / C.tok_dir_name("S2L2A") / "present.npy", d / "present.npy")
    tokens = np.array(np.load(src / C.tok_dir_name("S2L2A") / "tokens.npy", mmap_mode="r"))
    orig = tokens[rows].copy()
    for s in range(0, len(order), 256):
        br = order[s:s + 256]
        tokens[br] = encode_fixed(tok, make_input(ben_physical(br, packed, r2p), method, lin)).astype(np.uint16)
    np.save(d / "tokens.npy", tokens)
    agree = float((tokens[rows] == orig).mean())
    meta = json.loads((src / C.tok_dir_name("S2L2A") / "metadata.json").read_text())   # the contract fields BEN's loader checks
    meta["a2_step2_imputation"] = {
        "what": f"BEN S2L2A tokens with ft_valsel re-encoded, B01/B09 imputed by '{method}' (A2 Step 2.3 functional test)",
        "rows_reencoded": int(len(rows)), "agreement_with_true_tokens": agree, "gpu": gpu,
        "batch": BATCH, "tail_padding": "first row repeated", "linear_fit": str(OUT / "linear_fit.json")}
    (d / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"ben root {method}: {len(rows)} ft_valsel rows, agreement with true tokens {agree:.4f}", flush=True)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ben-root", default=None, help="comma-separated methods")
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()
    if a.fit:
        fit()
    if a.eval or a.ben_root:
        gpu = gpu_check(a.expect_gpu)
        if a.eval:
            evaluate(gpu)
        for m in (a.ben_root.split(",") if a.ben_root else []):
            ben_root(m, gpu)


if __name__ == "__main__":
    main()
