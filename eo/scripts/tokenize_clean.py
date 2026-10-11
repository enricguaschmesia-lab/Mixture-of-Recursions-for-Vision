#!/usr/bin/env python
"""Build the cleaned token set: S2L2A without the BOA offset, NDVI recomputed (`mor` env).

The offset rows are labelled by label_s2_offset.py (terramesh_tok/clean.py has the evidence).
Only those rows are re-tokenized; every clean row keeps its original tokens, bit for bit, so
nothing changes where nothing was wrong (re-encoding them on another card would flip ~1 token
in 8,192 at FSQ bin boundaries).

    python eo/scripts/tokenize_clean.py verify-ndvi            # CPU, every row
    python eo/scripts/tokenize_clean.py build                  # GPU (GTX Titan X), offset rows
    python eo/scripts/tokenize_clean.py assemble               # CPU: val_clean/ + gate
    python eo/scripts/tokenize_clean.py roundtrip              # GPU (GTX Titan X): decode check

verify-ndvi  clean.ndvi_from_s2 on the stored bands must reproduce the stored NDVI bit-exactly
             (float16) on every row. Negative control: the unclipped formula must not, on rows
             with negative pixels; and NDVI from the corrected bands must differ on offset rows.
build        per offset row: the stored (uncorrected) S2 and NDVI are re-encoded too, and must
             reproduce the stored tokens -- the contract control, on exactly these rows -- and
             the corrected S2 and recomputed NDVI give the new tokens. Batch 32, tail padded.
             Resumable per shard.
assemble     <clean root>/<MOD>_tok224/ for S2L2A and NDVI (tokens.npy, present.npy,
             metadata.json, changed.npy), every other artifact symlinked, and the gate.
             Raw tars are NOT linked: anything reading pixels must apply clean.correct_s2 itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/enric/hf")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terramesh_tok import clean as CL, contract as C, io as tio, preprocess as P  # noqa: E402

VAL = Path("/data/enric/data/TerraMesh/val")
ROOT = CL.clean_root(VAL)
WORK = ROOT / "_build"
REPO = Path(__file__).resolve().parents[2]
BATCH = 32


def git_rev() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_stems(mod: str, shard: str, stems=None):
    """{stem: raw array} for the wanted stems of one shard (all if None), decoding only those."""
    out = {}
    with tarfile.open(VAL / mod / shard) as tf:
        for m in tf:
            if not m.name.endswith(".zarr.zip"):
                continue
            s = m.name[:-len(".zarr.zip")]
            if stems is None or s in stems:
                out[s] = tio._decode_zarr_zip(tf.extractfile(m).read())
    return out


def _stream(mod: str, shard: str):
    with tarfile.open(VAL / mod / shard) as tf:
        for m in tf:
            if m.name.endswith(".zarr.zip"):
                yield m.name[:-len(".zarr.zip")], tio._decode_zarr_zip(tf.extractfile(m).read())


# ---------------------------------------------------------------- verify-ndvi (CPU)
def _verify_shard(args):
    shard, offset_stems = args
    nd = read_stems("NDVI", shard)          # ~140 MB a shard; S2 (~1.7 GB) is streamed instead
    rows = []
    for stem, a in _stream("S2L2A", shard):
        b = nd[stem]
        got = CL.ndvi_from_s2(a)
        red, nir = a[..., 3, :, :].astype(np.float64), a[..., 7, :, :].astype(np.float64)
        unclipped = ((nir - red) / (nir + red + CL.NDVI_EPS)).astype(np.float16)[..., None, :, :]
        off = stem in offset_stems
        rows.append({"stem": stem, "exact": float(np.mean(got.view(np.uint16) == b.view(np.uint16))),
                     "neg_px": int((a[..., [3, 7], :, :] < 0).any(axis=-3).sum()),
                     "unclipped_exact": float(np.mean(unclipped.view(np.uint16) == b.view(np.uint16))),
                     "offset": off,
                     "corrected_exact": float(np.mean(CL.ndvi_from_s2(CL.correct_s2(a, True)).view(np.uint16)
                                                      == b.view(np.uint16))) if off else np.nan})
    return rows


def verify_ndvi(args) -> int:
    lab = CL.load_labels(ROOT)
    idx = pd.read_parquet(VAL / "tok_index.parquet")
    off = set(lab[lab.offset].stem)
    jobs = [(sh, off) for sh in sorted(idx.source_shard.unique())][:args.max_shards]
    t0, res = time.time(), []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_verify_shard, jobs), 1):
            res += r
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} shards, {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(res)
    d.to_parquet(WORK / ("verify_ndvi.parquet" if args.max_shards is None else "verify_ndvi_smoke.parquet"))
    neg = d[d.neg_px > 0]
    rep = {"rows": int(len(d)), "rows_exact": int((d.exact == 1).sum()), "min_exact": float(d.exact.min()),
           "control_unclipped_rows_with_negatives": int(len(neg)),
           "control_unclipped_exact_on_those": int((neg.unclipped_exact == 1).sum()),
           "control_corrected_exact_on_offset_rows": int((d[d.offset].corrected_exact == 1).sum()),
           "offset_rows": int(d.offset.sum())}
    full = args.max_shards is None
    rep["pass"] = bool((rep["rows"] == len(idx) or not full) and rep["rows_exact"] == rep["rows"]
                       and rep["control_unclipped_exact_on_those"] < rep["control_unclipped_rows_with_negatives"]
                       and rep["control_corrected_exact_on_offset_rows"] == 0)
    if full:
        (WORK / "verify_ndvi.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if rep["pass"] else 1


# ---------------------------------------------------------------- build (GPU)
def build(args) -> int:
    import torch
    from terramesh_tok import tokenizers as T
    if torch.cuda.device_count() != 1 or "TITAN X" not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected exactly the GTX Titan X visible (CUDA_DEVICE_ORDER=PCI_BUS_ID "
                         f"CUDA_VISIBLE_DEVICES=0); got {torch.cuda.device_count()} device(s)")
    lab = CL.load_labels(ROOT)
    off = lab[lab.offset]
    idx = pd.read_parquet(VAL / "tok_index.parquet").set_index("stem")
    toks = {m: T.build(m, device="cuda:0") for m in CL.CLEANED_MODALITIES}
    (WORK / "shards").mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def encode(mod, xs):
        out = []
        for i in range(0, len(xs), BATCH):
            b = torch.cat(xs[i:i + BATCH])
            n = b.shape[0]
            if n < BATCH:
                b = torch.cat([b, b[-1:].repeat(BATCH - n, 1, 1, 1)])
            out.append(P.flatten_tokens(T.encode(toks[mod], b.to("cuda:0"))).cpu().numpy()[:n].astype(np.uint16))
        return np.concatenate(out)

    shards = sorted(off.stem.map(idx.source_shard).unique())[:args.max_shards]
    t_all = time.time()
    for k, shard in enumerate(shards, 1):
        dst = WORK / "shards" / shard.replace(".tar", ".npz")
        if dst.exists():
            continue
        t0 = time.time()
        stems = [s for s in idx[idx.source_shard == shard].sort_values("row").index if s in set(off.stem)]
        s2 = read_stems("S2L2A", shard, set(stems))
        nd = read_stems("NDVI", shard, set(stems))
        x = {"S2L2A_unc": [], "S2L2A_cor": [], "NDVI_unc": [], "NDVI_cor": []}
        for s in stems:
            cor = CL.correct_s2(s2[s], True)
            x["S2L2A_unc"].append(P.prepare(s2[s], "S2L2A", device="cpu")[0])
            x["S2L2A_cor"].append(P.prepare(cor, "S2L2A", device="cpu")[0])
            x["NDVI_unc"].append(P.prepare(nd[s], "NDVI", device="cpu")[0])
            x["NDVI_cor"].append(P.prepare(CL.ndvi_from_s2(cor), "NDVI", device="cpu")[0])
        out = {k_: encode(k_.split("_")[0], v) for k_, v in x.items()}
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, rows=idx.loc[stems, "row"].values, **out)
        os.replace(tmp, dst)
        print(f"  [{k}/{len(shards)}] {shard}: {len(stems)} offset rows, {time.time() - t0:.0f}s", flush=True)
    print(f"build done: {len(shards)} shards, {time.time() - t_all:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------- assemble (CPU) + gate
def assemble(args) -> int:
    lab = CL.load_labels(ROOT)
    off_rows = np.sort(lab[lab.offset].row.values)
    parts = [np.load(f) for f in sorted((WORK / "shards").glob("*.npz"))]
    rows = np.concatenate([p["rows"] for p in parts])
    gate = {}
    gate["every offset row built exactly once"] = bool(np.array_equal(np.sort(rows), off_rows))
    order = np.argsort(rows)
    rep = {"labels_sha256": sha256(ROOT / CL.LABELS_FILE), "n_offset_rows": int(len(off_rows)),
           "git_rev": git_rev(), "device": "cuda:0 (GTX TITAN X, PCI_BUS_ID)", "batch_size": BATCH,
           "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "modalities": {}}
    for mod in CL.CLEANED_MODALITIES:
        unc = np.concatenate([p[f"{mod}_unc"] for p in parts])[order]
        cor = np.concatenate([p[f"{mod}_cor"] for p in parts])[order]
        src = VAL / C.tok_dir_name(mod)
        old = np.load(src / "tokens.npy")
        new = old.copy()
        new[off_rows] = cor
        changed = np.zeros(len(old), dtype=np.float32)
        changed[off_rows] = (cor != old[off_rows]).mean(1)
        repro = (unc == old[off_rows]).mean(1)
        clean_rows = np.setdiff1d(np.arange(len(old)), off_rows)
        m = {"contract_control_token_agreement": float(repro.mean()),
             "contract_control_rows_below_0.99": int((repro < 0.99).sum()),
             "corrected_changed_frac_median": float(np.median(changed[off_rows])),
             "corrected_changed_frac_p05": float(np.percentile(changed[off_rows], 5)),
             "clean_rows_identical": bool(np.array_equal(new[clean_rows], old[clean_rows])),
             "token_range": [int(new.min()), int(new.max())],
             "constant_token_rows": int((new[off_rows] == new[off_rows][:, :1]).all(1).sum())}
        rep["modalities"][mod] = m
        gate[f"{mod}: clean rows bit-identical"] = m["clean_rows_identical"]
        gate[f"{mod}: uncorrected input reproduces the stored tokens (>= 0.999)"] = m["contract_control_token_agreement"] >= 0.999
        gate[f"{mod}: corrected rows change (median >= 0.5)"] = m["corrected_changed_frac_median"] >= 0.5
        gate[f"{mod}: tokens in the codebook, no constant rows"] = (
            m["token_range"][1] < C.CODEBOOK[mod] and m["constant_token_rows"] == 0)

        dst = ROOT / C.tok_dir_name(mod)
        dst.mkdir(exist_ok=True)
        for name, arr in (("tokens.npy", new), ("present.npy", np.load(src / "present.npy")), ("changed.npy", changed)):
            with open(dst / (name + ".tmp"), "wb") as fh:
                np.save(fh, arr)
            os.replace(dst / (name + ".tmp"), dst / name)
        meta = json.loads((src / "metadata.json").read_text())
        meta["cleaning"] = {"what": "S2L2A BOA offset removed; NDVI recomputed from the corrected bands",
                            "source_tokens": str(src), **{k: rep[k] for k in ("labels_sha256", "n_offset_rows",
                                                                              "git_rev", "device", "date")}, **m}
        (dst / "metadata.json").write_text(json.dumps(meta, indent=1))

    for p in VAL.iterdir():            # everything else, linked; raw modality tars are NOT
        is_tok = p.name.endswith(C.TOK_DIR_SUFFIX) or p.name in ("Coords_tok", "tok_index.parquet")
        if is_tok and p.name not in {C.tok_dir_name(m) for m in CL.CLEANED_MODALITIES}:
            link = ROOT / p.name        # relative, so the tree is portable (RCP has ../val too)
            if link.is_symlink() and os.readlink(link) != os.path.relpath(p, ROOT):
                link.unlink()
            if not link.exists():
                link.symlink_to(os.path.relpath(p, ROOT))
    gate["tok_index.parquet identical"] = sha256(ROOT / "tok_index.parquet") == sha256(VAL / "tok_index.parquet")
    rep["gate"] = gate
    rep["pass"] = all(gate.values())
    (ROOT / "clean_build.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if rep["pass"] else 1


# ---------------------------------------------------------------- roundtrip (GPU)
def roundtrip(args) -> int:
    """Decode a stratified sample of offset rows (12.6b settings: unclamped DiVAE, 50 steps,
    batch 4, seed 0 per batch) and score in z units against the matching reference:
    new tokens vs the corrected image, old tokens vs the stored image. They must agree within
    tolerance; the control (new tokens vs the stored image) must be clearly worse."""
    import torch
    from terramesh_tok import tokenizers as T
    sys.path.insert(0, str(REPO / "eo/experiments"))
    import clamp_test as CT
    if torch.cuda.device_count() != 1 or "TITAN X" not in torch.cuda.get_device_name(0):
        raise SystemExit("expected exactly the GTX Titan X visible")
    lab = CL.load_labels(ROOT)
    rows = np.sort(lab[lab.offset].row.values)[:: max(1, int(lab.offset.sum()) // args.n)][:args.n]
    idx = pd.read_parquet(VAL / "tok_index.parquet").set_index("row").loc[rows]
    raw = {}
    for shard, g in idx.groupby("source_shard"):
        s2, nd = read_stems("S2L2A", shard, set(g.stem)), read_stems("NDVI", shard, set(g.stem))
        for r, st in zip(g.index, g.stem):
            raw[r] = (s2[st], nd[st])
    rep = {"rows": rows.tolist()}
    for mod in CL.CLEANED_MODALITIES:
        tok = T.build(mod, device="cuda:0")
        sched = CT.scheduler_like(tok, **CT.VARIANTS["noclip"])
        mean = np.array(C.V1_TOK_MEAN[mod], np.float32)[:, None, None]
        std = np.array(C.V1_TOK_STD[mod], np.float32)[:, None, None]

        def dec(tokens):
            out = []
            for i in range(0, len(tokens), 4):
                g = torch.from_numpy(tokens[i:i + 4].reshape(-1, C.GRID, C.GRID).astype(np.int64)).to("cuda:0")
                torch.manual_seed(0); torch.cuda.manual_seed_all(0)
                out.append(tok.decode_tokens(g, timesteps=50, scheduler=sched).float().cpu().numpy())
            return np.concatenate(out)

        def ref(r, corrected):
            s2, nd = raw[r]
            a = (CL.ndvi_from_s2(CL.correct_s2(s2, True)) if corrected else nd) if mod == "NDVI" \
                else CL.correct_s2(s2, corrected)
            a = P.center_crop(a[0]).astype(np.float32)
            return (a - mean) / std

        new = dec(np.load(ROOT / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")[rows])
        old = dec(np.load(VAL / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")[rows])
        mse = lambda out, corrected: np.array([float(((out[i] - ref(r, corrected)) ** 2).mean()) for i, r in enumerate(rows)])
        m = {"new_vs_corrected": mse(new, True), "old_vs_stored": mse(old, False), "control_new_vs_stored": mse(new, False)}
        rep[mod] = {k: {"mean": float(v.mean()), "median": float(np.median(v))} for k, v in m.items()}
        control_ok = np.median(m["control_new_vs_stored"]) > 2 * np.median(m["new_vs_corrected"])
        # v1, declared first: the same rows' stored image. It FAILS for NDVI by construction (2026-10-11):
        # the offset compressed those scenes' NDVI, so their stored version is an easier target
        # (corrected: median |z| 0.96 vs 0.66, spatial std 0.086 vs 0.062, error 2.2x). Kept on record.
        rep[mod]["criterion_v1_same_rows"] = bool(np.median(m["new_vs_corrected"]) <= 1.5 * np.median(m["old_vs_stored"]))
        # v2: clean rows matched on the corrected image's mean and spread, decoded the same way.
        refs = {r: ref(r, True) for r in rows}
        want = np.array([[refs[r].mean(), refs[r].std()] for r in rows])
        pool = np.sort(lab[~lab.offset & (lab.corpus == "majortom")].row.values)[::25]
        pidx = pd.read_parquet(VAL / "tok_index.parquet").set_index("row").loc[pool]
        stats, praw = {}, {}
        for shard, g in pidx.groupby("source_shard"):
            got = read_stems("S2L2A" if mod == "S2L2A" else "NDVI", shard, set(g.stem))
            for r, st in zip(g.index, g.stem):
                a = (P.center_crop(got[st][0]).astype(np.float32) - mean) / std
                stats[r], praw[r] = (a.mean(), a.std()), a
        prow = np.array(list(stats)); pst = np.array([stats[r] for r in prow])
        sc = pst.std(0)
        matched = []
        for w in want:                     # nearest unused clean row in (mean, std), scaled
            dist = (((pst - w) / sc) ** 2).sum(1)
            for k in np.argsort(dist):
                if prow[k] not in matched:
                    matched.append(int(prow[k])); break
        mt = dec(np.load(VAL / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")[np.array(matched)])
        mclean = np.array([float(((mt[i] - praw[r]) ** 2).mean()) for i, r in enumerate(matched)])
        rep[mod]["matched_clean_rows"] = matched
        rep[mod]["matched_clean_vs_stored"] = {"mean": float(mclean.mean()), "median": float(np.median(mclean))}
        rep[mod]["criterion_v2_matched_clean"] = bool(np.median(m["new_vs_corrected"]) <= 1.5 * np.median(mclean))
        rep[mod]["pass"] = bool(rep[mod]["criterion_v2_matched_clean"] and control_ok)
    rep["pass"] = all(rep[m]["pass"] for m in CL.CLEANED_MODALITIES)
    (ROOT / "clean_roundtrip.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if rep["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify-ndvi"); v.add_argument("--workers", type=int, default=2)
    b = sub.add_parser("build")
    for q in (v, b):
        q.add_argument("--max-shards", type=int, default=None, help="smoke: the first N shards only")
    sub.add_parser("assemble")
    rt = sub.add_parser("roundtrip"); rt.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    return {"verify-ndvi": verify_ndvi, "build": build, "assemble": assemble, "roundtrip": roundtrip}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
