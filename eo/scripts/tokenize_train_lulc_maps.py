#!/usr/bin/env python
"""A2 Step 2.6: TerraMesh LULC as map tokens of a frozen map tokenizer, row-aligned with a token root's
tok_index.parquet (`a2seg` env, GPU). For R1's mid-training (A2 Step 5.1): the train split's `ssl4eos12`
rows, and val's rows for R1's held-out map CE.

    train   the 96 `ssl4eos12` LULC tars (24.5 GB, 958,176 rows) fetched from the Hugging Face listing at the
            train root's pinned revision (plan.json `hf_revision`), size and sha256 checked on the fly
            (train_split._fetch); Major TOM's LULC (205 GB) is not needed and its rows stay absent.
    val     val's local raw LULC tars (90, both corpora, 89,088 rows): nothing is downloaded.

Per shard, in the index's order: read its members in tar order, check the tar's stems against the index's rows
for that shard, centre-crop each map to 224 (train_split.crop: the contract's window, = preprocess.center_crop,
which Step 1 read val's maps with), render it with the universal palette and LULC's fixed assignment (class c ->
colour c, no data on white), encode with the frozen arm at its fixed batch shape (tokenize_maps.encode_maps:
cuDNN deterministic, TF32 off), and write the codes at those rows. Output, in the TerraMesh token format:

    <out>/tokens.npy      (n_rows, tokens_per_map) uint16, row = the index's row (absent rows zero: read present)
         /present.npy     (n_rows,) bool
         /done/<shard>.json  rows, the stems' and the shard's tokens' sha256, label histogram, card, seconds
         /metadata.json   the arm's frozen record, palette and assignment, the index's path and sha256, the
                          revision and listing, counts, codes used, the card; written last

⚠ The output never goes inside a token root: the train root on g49 is gated by its build.json and read by live
runs. The artifact names the index it is aligned with (sha256), and the R1 dataset checks it against its root.
⚠ One card per artifact (codes can differ across cards at FSQ bin boundaries): a resume on another card is refused.
⚠ Raw tars are kept unless --delete-raw: Step 2.6 keeps them until the tokens are verified (verify_seg_r1.py Z14).

    python eo/scripts/tokenize_train_lulc_maps.py --split train --plan
    python eo/scripts/tokenize_train_lulc_maps.py --split train --arm aligned --work /data/enric/data/TerraMesh/train_lulc_raw \\
        --out /data/enric/data/TerraMesh/maps/train/MAP_aligned --expect-gpu "GTX TITAN X" [--shards N | --only a,b]
    python eo/scripts/tokenize_train_lulc_maps.py --split val --arm aligned --out /data/enric/data/TerraMesh/maps/val/MAP_aligned \\
        --expect-gpu "GTX TITAN X"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "scripts"))
from eo.maps import frozen, palette as P  # noqa: E402
from eo.terramesh_tok import contract as C  # noqa: E402
from eo.terramesh_tok import train_split as TS  # noqa: E402

TM = Path("/data/enric/data/TerraMesh")
LISTING = TM / "train_meta/hf_train_listing.json"
TRAIN_PLAN = TM / "train_tok/plan.json"                # the train root's plan (its copy on g49 is the same file)
SPLITS = {
    "train": {"index": TM / "train_coords/tok_index.parquet", "raw": None, "corpora": ("ssl4eos12",)},
    "val": {"index": TM / "val/tok_index.parquet", "raw": TM / "val", "corpora": ("majortom", "ssl4eos12")},
}
CHUNK = 2048                                             # maps rendered at once (tokenize_maps' size; Titan X fits it)
TASK = "lulc"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic_npy(arr: np.ndarray, path: Path) -> None:
    with open(str(path) + ".tmp", "wb") as fh:
        np.save(fh, arr)
    os.replace(str(path) + ".tmp", path)


def plan_shards(idx: pd.DataFrame, corpora, only=None, limit=None) -> list:
    shards = sorted(idx[idx.corpus.isin(corpora)].source_shard.unique())
    if only:
        bad = sorted(set(only) - set(shards))
        if bad:
            raise SystemExit(f"--only names shards outside the selection: {bad[:3]}")
        shards = [s for s in shards if s in set(only)]
    return shards[:limit] if limit else shards


def read_shard(raw_root: Path, shard: str):
    """(stems in tar order, (n, 224, 224) uint8 labels) of one LULC tar."""
    stems, maps = [], []
    for stem, arr in TS._iter_tar(raw_root, "LULC", shard):
        stems.append(stem)
        maps.append(np.asarray(TS.crop(arr)).reshape(C.CROP, C.CROP))
    return stems, np.stack(maps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--arm", choices=list(frozen.ARMS), default="aligned")
    ap.add_argument("--index", type=Path, default=None, help="the token root's tok_index.parquet (default: the split's)")
    ap.add_argument("--out", type=Path, default=None, help="the artifact directory (never inside a token root)")
    ap.add_argument("--work", type=Path, default=None, help="train: where the tars land (kept unless --delete-raw)")
    ap.add_argument("--listing", type=Path, default=LISTING)
    ap.add_argument("--revision", default=None, help="train: HF revision (default: the train root's plan.json hf_revision)")
    ap.add_argument("--plan", action="store_true", help="list what would be read and written; touch nothing")
    ap.add_argument("--only", default=None, help="comma-separated shard names (validation runs)")
    ap.add_argument("--shards", type=int, default=None, help="only the first N shards of the selection")
    ap.add_argument("--conns", type=int, default=8)
    ap.add_argument("--delete-raw", action="store_true", help="train: delete each tar once its shard is written")
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args()

    spec = SPLITS[a.split]
    index = a.index or spec["index"]
    idx = pd.read_parquet(index)
    if not (idx.row.values == np.arange(len(idx))).all():
        raise SystemExit(f"{index}: rows are not 0..n-1 in order")
    only = a.only.split(",") if a.only else None
    shards = plan_shards(idx, spec["corpora"], only, a.shards)
    n = len(idx)
    sel_rows = int(idx.source_shard.isin(shards).sum())
    if a.split == "train":
        files = json.loads(a.listing.read_text())["files"]["LULC"]
        revision = a.revision or json.loads(TRAIN_PLAN.read_text())["hf_revision"]
        size = sum(files[s]["size"] for s in shards)
        print(f"train: {len(shards)} LULC tars, {size / 1e9:.2f} GB at revision {revision}; {sel_rows} rows of {n}",
              flush=True)
    else:
        files, revision = None, None
        print(f"val: {len(shards)} local LULC tars under {spec['raw']}; {sel_rows} rows of {n}", flush=True)
    if a.plan:
        return
    if a.out is None:
        raise SystemExit("--out is required (and is never a token root)")
    if a.split == "train" and a.work is None:
        raise SystemExit("--work is required for the train split (where the tars land)")

    import torch
    from tokenize_maps import encode_maps
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    TS.strict_fp32()
    tok = frozen.load(a.arm)
    T, n_codes = tok.frozen["tokens"], tok.frozen["n_codes"]
    out = a.out
    (out / "done").mkdir(parents=True, exist_ok=True)
    index_sha = sha256_file(Path(index))
    card_file = out / "card.json"                        # one card per artifact, from its first written row on
    if card_file.exists() and any((out / "done").glob("*.json")):
        prev = json.loads(card_file.read_text())
        if prev["gpu"] != gpu:
            raise SystemExit(f"{out} was started on {prev['gpu']}; refusing to add rows from {gpu}")
        if prev["index_sha256"] != index_sha or prev["arm"] != a.arm or prev["split"] != a.split:
            raise SystemExit(f"{out} was started for {prev}; this run is {a.split}/{a.arm}/{index_sha[:12]}")
    else:
        TS.atomic_json({"gpu": gpu, "arm": a.arm, "split": a.split, "index": str(index), "index_sha256": index_sha,
                        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, card_file)
    tpath, ppath = out / "tokens.npy", out / "present.npy"
    if tpath.exists():
        tokens = np.load(tpath, mmap_mode="r+")
        assert tokens.shape == (n, T) and tokens.dtype == np.uint16, tokens.shape
    else:
        tokens = np.lib.format.open_memmap(tpath, mode="w+", dtype=np.uint16, shape=(n, T))
    present = np.load(ppath) if ppath.exists() else np.zeros(n, bool)

    todo = [s for s in shards if not (out / "done" / f"{s}.json").exists()]
    print(f"{a.split} MAP_{a.arm}: {len(shards)} shards selected, {len(todo)} to go, {T} tokens/map, on {gpu}", flush=True)

    # train: fetch the next tar while this one encodes (a download is ~4 s at 70 MB/s, an encode ~15-40 s)
    fetched: dict = {}

    def fetch(s):
        dest = a.work / "LULC" / s
        url = f"https://huggingface.co/datasets/{TS.HF_REPO}/resolve/{revision}/train/LULC/{s}"
        fetched[s] = TS._fetch(url, dest, files[s]["size"], files[s]["sha256"], conns=a.conns)

    pending = None
    t_all = time.time()
    for k, s in enumerate(todo):
        t0 = time.time()
        if a.split == "train":
            if pending is None:
                fetch(s)
            else:
                pending.join()
                if s not in fetched:
                    raise RuntimeError(f"{s}: the background fetch failed (see above)")
            if k + 1 < len(todo):
                pending = threading.Thread(target=fetch, args=(todo[k + 1],), daemon=True)
                pending.start()
            raw_root = a.work
        else:
            raw_root = spec["raw"]
        t_dl = time.time() - t0
        rows = idx[idx.source_shard == s]
        want = dict(zip(rows.stem, rows.row))
        stems, lab = read_shard(raw_root, s)
        if len(stems) != len(want) or set(stems) != set(want):
            raise RuntimeError(f"{s}: tar stems differ from the index ({len(set(stems) ^ set(want))} differ, "
                               f"{len(stems)} members against {len(want)} rows)")
        if lab.max() >= P.TASKS[TASK].n_classes:
            raise RuntimeError(f"{s}: a LULC label {int(lab.max())} outside 0..{P.TASKS[TASK].n_classes - 1}")
        r = np.array([want[st] for st in stems], dtype=np.int64)
        codes = np.empty((len(r), T), np.uint16)
        for c0 in range(0, len(lab), CHUNK):
            cc = encode_maps(tok, TASK, lab[c0:c0 + CHUNK])
            if cc.shape != (len(lab[c0:c0 + CHUNK]), T) or cc.min() < 0 or cc.max() >= n_codes:
                raise RuntimeError(f"{s}: codes of shape {cc.shape} in {cc.min()}..{cc.max()}, want (., {T}) in 0..{n_codes - 1}")
            codes[c0:c0 + CHUNK] = cc.astype(np.uint16)
        order = np.argsort(r)                            # write in row order: contiguous for a shard
        r_sorted, codes_sorted = r[order], codes[order]
        contiguous = bool(r_sorted[-1] - r_sorted[0] + 1 == len(r_sorted))
        if contiguous:
            tokens[r_sorted[0]:r_sorted[-1] + 1] = codes_sorted
        else:
            tokens[r_sorted] = codes_sorted
        tokens.flush()
        present[r] = True
        atomic_npy(present, ppath)
        stems_sorted = [stems[i] for i in order]
        hist = np.bincount(lab.reshape(-1), minlength=P.TASKS[TASK].n_classes)
        rec = {"shard": s, "rows": int(len(r)), "row_first": int(r_sorted[0]), "row_last": int(r_sorted[-1]),
               "contiguous": contiguous, "stems_sha256": sha256_bytes("\n".join(stems_sorted).encode()),
               "tokens_sha256": sha256_bytes(np.ascontiguousarray(codes_sorted).tobytes()),
               "label_hist": hist.tolist(), "codes_used": int(len(np.unique(codes))),
               "gpu": gpu, "seconds": round(time.time() - t0, 1), "fetch_wait_s": round(t_dl, 1)}
        if a.split == "train":
            rec.update({"tar_sha256": files[s]["sha256"], "tar_size": files[s]["size"], "revision": revision})
            if a.delete_raw:
                (a.work / "LULC" / s).unlink()
        TS.atomic_json(rec, out / "done" / f"{s}.json")
        el = time.time() - t_all
        print(f"  [{time.strftime('%F %T')}] {s}: {len(r)} maps in {rec['seconds']} s (fetch wait {t_dl:.1f} s); "
              f"{k + 1}/{len(todo)} in {el / 60:.1f} min", flush=True)
    if pending is not None:
        pending.join()

    done = sorted(p.stem for p in (out / "done").glob("*.json"))
    n_present = int(present.sum())
    expect = int(idx.source_shard.isin([d for d in done]).sum())
    if n_present != expect:
        raise RuntimeError(f"present {n_present} != rows of the done shards {expect}")
    all_done = set(done) >= set(plan_shards(idx, spec["corpora"]))
    meta = {"split": a.split, "arm": a.arm, "task": TASK, "frozen": tok.frozen, "tokenizer_record": tok.record(),
            "palette": "eo.maps.palette.UNIVERSAL", "palette_sha256": P.palette_record()["sha256"],
            "assignment": list(P.TASKS[TASK].assign), "render": "nearest to the tokenizer's input size",
            "crop": C.CROP, "crop_window": f"train_split.crop: offset {C.CROP_OFF} of {C.NATIVE}",
            "tokens_per_sample": T, "tokenizer_id_bound": n_codes, "dtype": "uint16", "flatten": "row-major",
            "batch": tok.batch, "run_fixed": "cuDNN deterministic, TF32 off, tail padded with zeros",
            "index": str(index), "index_sha256": index_sha, "row_order": "the index's row column",
            "n_rows": n, "n_present": n_present, "shards_done": len(done), "complete": all_done,
            "corpora": list(spec["corpora"]), "gpu": gpu, "revision": revision,
            "listing": str(a.listing) if a.split == "train" else None,
            "listing_sha256": sha256_file(a.listing) if a.split == "train" else None,
            "fork_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    assert meta["palette_sha256"] == tok.frozen["palette_sha256"], "palette differs from the frozen record"
    if all_done:                                         # the stats read every present row: only at the end
        tk = np.asarray(tokens[present])
        meta["codes_used"] = int(len(np.unique(tk)))
        meta["token_max"] = int(tk.max())
    TS.atomic_json(meta, out / "metadata.json")
    print(json.dumps({k: meta.get(k) for k in ("n_present", "shards_done", "complete", "codes_used", "token_max")}),
          flush=True)


if __name__ == "__main__":
    main()
