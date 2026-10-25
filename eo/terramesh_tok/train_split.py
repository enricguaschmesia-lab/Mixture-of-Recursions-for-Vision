"""Tokenizing the TerraMesh TRAIN split: plan, download, encode, consolidate (2026-10-25, mor env).

The val split was tokenized from a local copy (eo/scripts/tokenize_terramesh.py). The train split
needs 8.2 TB of tars, so it is streamed: one process downloads a shard's tars (sha256 against the
HF listing), another encodes them and deletes them. Every pixel goes through the same contract
(contract.py, and prepare()'s exact arithmetic replicated on the GPU in gpu_prepare); verify_val
re-tokenizes val shards through this path and compares with the stored val tokens.

Three differences from val, each measured first (worklog 2026-10-25):
  * NO S2 offset correction. clean.has_offset labels 22.5% of train rows, but those scenes are
    stored without the offset (eo/experiments/s2_offset_train_check.py). Every SCAN_EVERY-th S2
    scene is scanned here, so the whole split is checked rather than 3 shards (consolidate's C3).
  * NDVI is computed from the S2L2A bands (clean.ndvi_from_s2: bit-exact against TerraMesh's
    stored NDVI on 10,000/10,000 train rows, eo/experiments/ndvi_train_check.py), so the NDVI
    tars are never downloaded.
  * prepare() runs on the GPU. The encode is GPU-bound at batch 32 and the CPU prepare cost ~25%
    more on the workstation; the inputs are bit-identical (contiguous NCHW float32, same ops).

Layout under OUT:
    plan.json, order.json, tok_index.parquet, rule_labels.parquet
    <MOD>_tok224/shards/<shard>.npy, .nan.npy   per (shard, modality), as val
    s2_scan/<shard>.parquet                     every SCAN_EVERY-th S2 scene: floor, water_B12
    done/<shard>.json                           written LAST: the unit of completion
    control/STOP                                a worker stops before its next shard
Under WORK (raw, transient): <MOD>/<shard>, ready/<shard>.json
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean as CL
from . import contract as C

HF_REPO = "ibm-esa-geospatial/TerraMesh"
N_SHARDS = {"majortom": 798, "ssl4eos12": 96}
S1_OF = {"majortom": "S1RTC", "ssl4eos12": "S1GRD"}
#: Encoded per shard (NDVI from the S2L2A bands); S1 is the corpus's one.
OUT_MODS = ("S2L2A", "NDVI", "DEM", "LULC", "S1RTC", "S1GRD")
SCAN_EVERY = 10
BATCH = 32                     # the contract's batch shape (tokens are reproducible only at one)
REPO = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- helpers
def canonical_shards() -> list[str]:
    """All majortom shards, then all ssl4eos12 shards: the row order of tok_index.parquet."""
    return [f"{c}_shard_{i:06d}.tar" for c in N_SHARDS for i in range(1, N_SHARDS[c] + 1)]


def corpus(shard: str) -> str:
    return shard.split("_")[0]


def raw_mods(shard: str) -> tuple[str, ...]:
    return ("S2L2A", "DEM", "LULC", S1_OF[corpus(shard)])


def out_mods(shard: str) -> tuple[str, ...]:
    return ("S2L2A", "NDVI", "DEM", "LULC", S1_OF[corpus(shard)])


def git_rev() -> str:
    """The fork's commit (+ -dirty), or the CODE_VERSION file a copied checkout carries (RCP pods
    have no git), with this module's sha256 so a copy is still identifiable."""
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    try:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        r, dirty = "", ""
    if not r and (REPO / "CODE_VERSION").exists():
        r = (REPO / "CODE_VERSION").read_text().strip()
    return f"{r or 'unknown'}{'-dirty' if dirty else ''} train_split:{me}"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def atomic_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:          # np.save() would append '.npy' to a path
        np.save(fh, arr)
    os.replace(tmp, path)


def done_path(out: Path, shard: str) -> Path:
    return Path(out) / "done" / shard.replace(".tar", ".json")


def ready_path(work: Path, shard: str) -> Path:
    return Path(work) / "ready" / shard.replace(".tar", ".json")


def stop_requested(out: Path) -> bool:
    return (Path(out) / "control" / "STOP").exists()


def load_order(out: Path) -> list[str]:
    return json.loads((Path(out) / "order.json").read_text())["order"]


def parse_positions(spec: str, n: int) -> list[int]:
    """'0:894' forward, '894:0' backward (end exclusive), as order positions."""
    a, b = (int(x) for x in spec.split(":"))
    return list(range(a, b)) if a <= b else list(range(a - 1, b - 1, -1))


def resolve_train_root(raw, token_set: str, allow_partial: bool = False) -> Path:
    """The train-split token root a `multimodal.split: train` run reads (TERRAMESH_TRAIN_TOK_ROOT).

    Refuses rather than guesses:
      * unset, or not a root `consolidate` wrote (its build.json);
      * not offset-free by the in-line scan (C3);
      * partial (only the shards finished so far), unless the config allows it (smoke runs);
      * paired with the ORIGINAL val tokens. The train split is TerraMesh's fixed upstream
        version, which val_clean equals and val/ does not, so its runs must set
        multimodal.token_set: clean (the held-out rows then come from val_clean).
    """
    if not raw:
        raise ValueError("TERRAMESH_TRAIN_TOK_ROOT is unset: a multimodal.split: train run needs the "
                         "consolidated train root (eo/scripts/tokenize_train.py consolidate)")
    root = Path(raw)
    b = root / "build.json"
    if not b.is_file():
        raise FileNotFoundError(f"{b} missing: {root} is not a consolidated train root")
    build = json.loads(b.read_text())
    if build.get("split") != "train":
        raise RuntimeError(f"{b} records split {build.get('split')!r}, not 'train'")
    if build.get("offset_free") is not True:
        raise RuntimeError(f"{b}: the S2 offset scan (C3) did not pass; this root may carry the offset")
    if build.get("partial") and not allow_partial:
        raise RuntimeError(f"{b}: a PARTIAL root ({build.get('shards')} of {len(canonical_shards())} shards); "
                           f"set multimodal.allow_partial_train_root only for a smoke run")
    if token_set != "clean":
        raise ValueError(f"multimodal.token_set={token_set!r} with split train: the train split is the fixed "
                         f"upstream version, so its held-out rows must come from val_clean (token_set: clean)")
    return root


# ----------------------------------------------------------------- plan
def build_plan(out: Path, meta_path: Path, listing_path: Path, majortom_meta: Path,
               revision: str, seed: int) -> dict:
    """tok_index.parquet (canonical row order), the val-rule labels, and the seeded shard order."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shards = canonical_shards()
    rank = {s: i for i, s in enumerate(shards)}
    meta = pd.read_parquet(meta_path, columns=["tar", "zarr", "sample_id", "S2_time"])
    if set(meta.tar) != set(shards):
        raise RuntimeError("train_metadata.parquet's shards are not the canonical 798 + 96")
    meta["stem"] = meta.zarr.str[: -len(".zarr.zip")]
    meta["corpus"] = meta.tar.str.split("_").str[0]
    meta = meta.assign(_r=meta.tar.map(rank)).sort_values(["_r", "zarr"], kind="stable")
    meta = meta.reset_index(drop=True)
    if meta.stem.duplicated().any():
        raise RuntimeError("duplicate stems in train_metadata.parquet")
    index = pd.DataFrame({"row": np.arange(len(meta)), "stem": meta.stem.values,
                          "source_shard": meta.tar.values, "corpus": meta.corpus.values})
    tmp = out / "tok_index.parquet.tmp"
    index.to_parquet(tmp, index=False)
    os.replace(tmp, out / "tok_index.parquet")

    # clean.has_offset's labels, for the scan's read-out only: nothing is corrected (module doc).
    m = meta[meta.corpus == "majortom"].copy()
    m["grid_cell"] = m.sample_id.str.split("_").str[:2].str.join("_")
    m["timestamp"] = pd.to_datetime(m.S2_time).dt.strftime("%Y%m%dT%H%M%S")
    mt = pd.read_parquet(majortom_meta, columns=["grid_cell", "product_id", "timestamp"])
    j = m.merge(mt, on=["grid_cell", "timestamp"], how="left")
    if j.product_id.isna().any():
        raise RuntimeError(f"{int(j.product_id.isna().sum())} majortom rows have no Major TOM product")
    j["baseline"] = j.product_id.str.extract(r"_N(\d{4})_")[0].astype(int)
    j["rule_offset"] = (j.baseline.isin(CL.OFFSET_BASELINES)
                        & (pd.to_datetime(j.S2_time) < pd.Timestamp(CL.TERRAMESH_OFFSET_CUTOFF)))
    lab = j.groupby("stem").rule_offset.agg(["any", "all"]).reset_index()
    lab = lab.assign(rule_offset=lab["any"] & lab["all"], ambiguous=lab["any"] != lab["all"])
    lab[["stem", "rule_offset", "ambiguous"]].to_parquet(out / "rule_labels.parquet", index=False)

    perm = np.random.default_rng(seed).permutation(len(shards))
    order = [shards[i] for i in perm]
    atomic_json({"seed": seed, "order": order}, out / "order.json")
    plan = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "code": git_rev(),
        "hf_repo": HF_REPO, "hf_revision": revision,
        "train_metadata_sha256": sha256_file(Path(meta_path)),
        "hf_listing": str(listing_path), "hf_listing_sha256": sha256_file(Path(listing_path)),
        "majortom_meta": str(majortom_meta), "n_rows": int(len(index)),
        "rows_per_corpus": index.corpus.value_counts().to_dict(),
        "rule_offset_rows": int(lab.rule_offset.sum()), "rule_ambiguous_rows": int(lab.ambiguous.sum()),
        "order_seed": seed, "scan_every": SCAN_EVERY, "batch": BATCH,
        "note": "NO offset correction and NDVI from S2L2A; see eo/terramesh_tok/train_split.py",
    }
    atomic_json(plan, out / "plan.json")
    return plan


# ----------------------------------------------------------------- download
def _fetch(url: str, dest: Path, size: int, sha: str, tries: int = 6) -> str:
    """Stream url to dest, verifying size and sha256 on the fly. Returns the sha256."""
    if dest.exists() and dest.stat().st_size == size and sha256_file(dest) == sha:
        return sha
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for k in range(tries):
        try:
            h, n = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as fh:
                for b in iter(lambda: r.read(1 << 23), b""):
                    fh.write(b)
                    h.update(b)
                    n += len(b)
            if n != size or h.hexdigest() != sha:
                raise IOError(f"{dest.name}: got {n} B sha {h.hexdigest()[:12]}, "
                              f"want {size} B sha {sha[:12]}")
            os.replace(part, dest)
            return sha
        except Exception as e:                      # noqa: BLE001 -- retried, then re-raised
            if k == tries - 1:
                raise
            print(f"  retry {k + 1}/{tries - 1} {dest.parent.name}/{dest.name}: {e}", flush=True)
            time.sleep(10 * 2 ** k)
    raise AssertionError("unreachable")


def download_loop(out: Path, work: Path, positions: list[int], revision: str,
                  listing_path: Path, ahead: int = 4, streams: int = 8, poll: float = 20.0) -> None:
    """Keep up to `ahead` shards downloaded and verified ahead of the encoder, in order."""
    out, work = Path(out), Path(work)
    order = load_order(out)
    files = json.loads(Path(listing_path).read_text())["files"]
    base = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/train"
    pool = ThreadPoolExecutor(streams)
    inflight: dict[str, list] = {}
    todo = [order[p] for p in positions]
    t0 = time.time()
    n_bytes = 0

    def pending() -> int:
        return len(list((work / "ready").glob("*.json"))) if (work / "ready").exists() else 0

    while todo or inflight:
        if stop_requested(out):
            print("STOP requested; the downloader exits after its in-flight shards", flush=True)
            todo = []
        while todo and len(inflight) + pending() < ahead:
            sh = todo.pop(0)
            if done_path(out, sh).exists() or ready_path(work, sh).exists():
                continue
            inflight[sh] = [(m, pool.submit(_fetch, f"{base}/{m}/{sh}", work / m / sh,
                                            files[m][sh]["size"], files[m][sh]["sha256"]))
                            for m in raw_mods(sh)]
        if not inflight:
            if todo:
                time.sleep(poll)
            continue
        wait([f for fs in inflight.values() for _, f in fs], timeout=poll, return_when=FIRST_COMPLETED)
        for sh in [s for s, fs in inflight.items() if all(f.done() for _, f in fs)]:
            rec = {m: {"sha256": f.result(), "size": files[m][sh]["size"]} for m, f in inflight.pop(sh)}
            n_bytes += sum(v["size"] for v in rec.values())
            atomic_json({"shard": sh, "files": rec, "t": time.time()}, ready_path(work, sh))
            dt = time.time() - t0
            print(f"[{time.strftime('%F %T')}] ready {sh}  ({n_bytes / 1e9:.1f} GB in {dt / 60:.1f} min, "
                  f"{n_bytes / 1e6 / dt:.0f} MB/s)", flush=True)
    pool.shutdown()


# ----------------------------------------------------------------- encode
def _prefetch(gen, n: int = 96):
    """Run a generator in a thread (tar read + blosc decode release the GIL), bounded queue."""
    q: queue.Queue = queue.Queue(maxsize=n)
    end = object()
    err: list = []

    def run():
        try:
            for item in gen:
                q.put(item)
        except BaseException as e:               # noqa: BLE001 -- re-raised in the consumer
            err.append(e)
        finally:
            q.put(end)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = q.get()
        if item is end:
            if err:
                raise err[0]
            return
        yield item


def _iter_tar(root: Path, mod: str, shard: str, with_cloud_every: int = 0):
    """(stem, array[, cloud_mask or None]) per member, in tar order."""
    from . import io as tio
    k = 0
    with tarfile.open(Path(root) / mod / shard) as tf:
        for m in tf:
            if not m.name.endswith(".zarr.zip"):
                continue
            raw = tf.extractfile(m).read()
            stem = m.name[: -len(".zarr.zip")]
            arr = tio._decode_zarr_zip(raw, "bands")
            if with_cloud_every:
                cm = tio._decode_zarr_zip(raw, "cloud_mask") if k % with_cloud_every == 0 else None
                yield stem, arr, cm
            else:
                yield stem, arr
            k += 1


def crop(arr: np.ndarray, crop_off: int | None = None) -> np.ndarray:
    """(1, C, 264, 264) as stored -> (C, 224, 224), exactly prepare()'s window."""
    o = C.CROP_OFF if crop_off is None else crop_off
    a = arr[0] if arr.ndim == 4 and arr.shape[0] == 1 else arr
    return a[..., o:o + C.CROP, o:o + C.CROP]


def gpu_prepare(batch: np.ndarray, mod: str, device: str):
    """prepare() on a stacked (B, C, 224, 224) raw batch, on the GPU. Returns (x, n_nan per row).

    The same operations in the same order and precision as preprocess.prepare: fp32 cast,
    per-channel-mean NaN fill, (x - mean) / std; LULC as a 10-class one-hot with no
    standardization. Contiguous NCHW like torch.cat of prepare()'s outputs, because a
    channels-last tensor picks another cuDNN kernel and flips tokens (measured 2026-10-25).
    """
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
    if mod == "LULC":
        if int(x.max()) >= C.LULC_N_CLASSES:
            raise RuntimeError(f"LULC class {int(x.max())} >= {C.LULC_N_CLASSES}")
        x = F.one_hot(x[:, 0].long(), C.LULC_N_CLASSES).permute(0, 3, 1, 2).contiguous().float()
        return x, torch.zeros(x.shape[0], dtype=torch.int64)
    if x.shape[1] != C.N_CHANNELS[mod]:
        raise RuntimeError(f"{mod}: {x.shape[1]} channels, contract says {C.N_CHANNELS[mod]}")
    x = x.float()
    m = torch.tensor(C.V1_TOK_MEAN[mod], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    s = torch.tensor(C.V1_TOK_STD[mod], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    nan = torch.isnan(x)
    n_nan = nan.sum(dim=(1, 2, 3)).cpu()
    if bool(nan.any()):
        x = torch.where(nan, m.expand_as(x), x)
    return ((x - m) / s).contiguous(), n_nan


class Batcher:
    """Accumulates raw cropped arrays and encodes full BATCH-sized batches (tail-padded)."""

    def __init__(self, mod: str, tok, device: str):
        self.mod, self.tok, self.device = mod, tok, device
        self.buf: list[np.ndarray] = []
        self.stems: list[str] = []
        self.out: list[np.ndarray] = []
        self.nans: list[np.ndarray] = []
        self.secs = 0.0

    def add(self, stem: str, a: np.ndarray) -> None:
        self.stems.append(stem)
        self.buf.append(a)
        if len(self.buf) == BATCH:
            self._flush()

    def _flush(self) -> None:
        import torch
        from . import preprocess as P, tokenizers as T
        if not self.buf:
            return
        n = len(self.buf)
        b = np.stack(self.buf + [self.buf[-1]] * (BATCH - n))   # tail padding: the contract's shape
        t = time.time()
        x, n_nan = gpu_prepare(b, self.mod, self.device)
        f = P.flatten_tokens(T.encode(self.tok, x)).cpu().numpy()[:n]
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.secs += time.time() - t
        lo, hi = int(f.min()), int(f.max())
        if lo < 0 or hi >= C.CODEBOOK[self.mod]:
            raise RuntimeError(f"{self.mod}: token out of range [{lo},{hi}] vs {C.CODEBOOK[self.mod]}")
        self.out.append(f.astype(np.uint16))
        self.nans.append(n_nan.numpy()[:n].astype(np.int32))
        self.buf.clear()

    def finish(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        self._flush()
        return self.stems, np.concatenate(self.out), np.concatenate(self.nans)


def scan_scene(bands: np.ndarray, cm: np.ndarray) -> tuple[float, float]:
    """s2_offset_scan.py's statistics: floor (min over bands of the 1st percentile) and the
    median B12 over water pixels (cloud_mask 1, >= 50 of them), both on the 224 centre crop."""
    b = crop(bands).astype(np.float32)
    floor = float(np.percentile(b.reshape(b.shape[0], -1), 1, axis=1).min())
    w = crop(cm[None] if cm.ndim == 2 else cm)[0] == 1
    water = float(np.median(b[11][w])) if w.sum() >= 50 else float("nan")
    return floor, water


def check_tokens(mod: str, shard: str, stems: list[str], toks: np.ndarray, expect: list[str]) -> None:
    """tokenize_terramesh.check_invariants, for a train shard."""
    if stems != expect:
        bad = next((i for i, (a, b) in enumerate(zip(stems, expect)) if a != b), min(len(stems), len(expect)))
        raise RuntimeError(f"{mod}/{shard}: ROW MISALIGNMENT at {bad} ({len(stems)} read, {len(expect)} expected)")
    if toks.shape != (len(expect), C.TOKENS_PER_SAMPLE) or toks.dtype != np.uint16:
        raise RuntimeError(f"{mod}/{shard}: tokens {toks.shape} {toks.dtype}")
    const = (toks == toks[:, :1]).all(axis=1)
    if const.any():
        raise RuntimeError(f"{mod}/{shard}: {int(const.sum())} constant token sequence(s), the "
                           f"NaN-poisoning signature; first {stems[int(np.argmax(const))]}")


def encode_shard(root: Path, shard: str, toks: dict, device: str, expect: list[str],
                 crop_off: int | None = None) -> tuple[dict, pd.DataFrame, dict]:
    """Encode one shard's tars under root/<MOD>/<shard>: S2L2A and NDVI in one pass, then the rest.
    Returns ({mod: (tokens, nans)}, scan rows, {mod: seconds})."""
    res, secs = {}, {}
    s2, nd = Batcher("S2L2A", toks["S2L2A"], device), Batcher("NDVI", toks["NDVI"], device)
    scan = []
    for stem, bands, cm in _prefetch(_iter_tar(root, "S2L2A", shard, with_cloud_every=SCAN_EVERY)):
        s2.add(stem, crop(bands, crop_off))
        nd.add(stem, crop(CL.ndvi_from_s2(bands), crop_off))
        if cm is not None:
            scan.append((stem, *scan_scene(bands, cm)))
    for b in (s2, nd):
        st, tk, nn = b.finish()
        check_tokens(b.mod, shard, st, tk, expect)
        res[b.mod], secs[b.mod] = (tk, nn), b.secs
    for mod in ("DEM", "LULC", S1_OF[corpus(shard)]):
        b = Batcher(mod, toks[mod], device)
        for stem, arr in _prefetch(_iter_tar(root, mod, shard)):
            b.add(stem, crop(arr, crop_off))
        st, tk, nn = b.finish()
        check_tokens(mod, shard, st, tk, expect)
        res[mod], secs[mod] = (tk, nn), b.secs
    return res, pd.DataFrame(scan, columns=["stem", "floor", "water_B12"]), secs


def load_tokenizers(device: str, mods=OUT_MODS) -> dict:
    from . import tokenizers as T
    return {m: T.build(m, device) for m in mods}


def setup_gpu(expect_gpu: str | None) -> str:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False      # the contract is fp32 (TF32 would flip
    torch.backends.cudnn.allow_tf32 = False            # bins on an A100; no effect on Volta)
    name = torch.cuda.get_device_name(0)
    if expect_gpu and expect_gpu not in name:
        raise SystemExit(f"cuda:0 is {name!r}, expected {expect_gpu!r}")
    return name


def encode_loop(out: Path, work: Path, positions: list[int], device: str = "cuda:0",
                expect_gpu: str | None = None, delete_raw: bool = True, poll: float = 20.0) -> None:
    import importlib.metadata as md
    out, work = Path(out), Path(work)
    gpu = setup_gpu(expect_gpu)
    order = load_order(out)
    index = pd.read_parquet(out / "tok_index.parquet")
    by_shard = {s: g.sort_values("row").stem.tolist() for s, g in index.groupby("source_shard")}
    labels = pd.read_parquet(out / "rule_labels.parquet").set_index("stem")
    toks = load_tokenizers(device)
    versions = {p: md.version(p) for p in ("torch", "terratorch", "numcodecs", "numpy")}
    rev = git_rev()
    print(f"encode: {len(positions)} positions on {gpu}, code {rev}", flush=True)
    state = out / "control" / f"encoder_state.{os.getpid()}"   # read by the power watchdog:
    state.parent.mkdir(parents=True, exist_ok=True)              # a waiting encoder idles the GPU
    for pos in positions:
        shard = order[pos]
        if done_path(out, shard).exists():
            continue
        state.write_text(f"wait {shard}\n")
        while not ready_path(work, shard).exists():
            if stop_requested(out):
                print("STOP requested; exiting", flush=True)
                return
            time.sleep(poll)
        if stop_requested(out):
            print("STOP requested; exiting", flush=True)
            return
        state.write_text(f"encode {shard}\n")
        t0 = time.time()
        ready = json.loads(ready_path(work, shard).read_text())
        res, scan, secs = encode_shard(work, shard, toks, device, by_shard[shard])
        for mod, (tk, nn) in res.items():
            d = out / C.tok_dir_name(mod) / "shards"
            atomic_npy(tk, d / shard.replace(".tar", ".npy"))
            atomic_npy(nn, d / shard.replace(".tar", ".nan.npy"))
        (out / "s2_scan").mkdir(exist_ok=True)
        scan.to_parquet(out / "s2_scan" / shard.replace(".tar", ".parquet"), index=False)
        lab = labels.reindex(scan.stem)          # ssl4eos12 stems are absent: no product id, clean
        is_off = lab.rule_offset.fillna(False).astype(bool).values
        is_amb = lab.ambiguous.fillna(False).astype(bool).values
        ro, rc = scan[is_off & ~is_amb], scan[~is_off]
        rec = {
            "shard": shard, "position": pos, "n": len(by_shard[shard]), "code": rev, "gpu": gpu,
            "versions": versions, "seconds": round(time.time() - t0, 1),
            "encode_seconds": {m: round(v, 1) for m, v in secs.items()},
            "tokens": {m: {"min": int(tk.min()), "max": int(tk.max()), "n_nan": int(nn.sum())}
                       for m, (tk, nn) in res.items()},
            "tars": ready["files"],
            "scan": {"n": len(scan), "n_rule_offset": len(ro), "n_rule_clean": len(rc),
                     "floor_gt900_rule_offset": float((ro.floor > 900).mean()) if len(ro) else None,
                     "floor_gt900_rule_clean": float((rc.floor > 900).mean()) if len(rc) else None,
                     "water_B12_median_rule_offset": float(ro.water_B12.median()) if ro.water_B12.notna().any() else None,
                     "n_water_rule_offset": int(ro.water_B12.notna().sum())},
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
        atomic_json(rec, done_path(out, shard))
        if delete_raw:
            for m in raw_mods(shard):
                (work / m / shard).unlink(missing_ok=True)
            ready_path(work, shard).unlink(missing_ok=True)
        n = len(by_shard[shard])
        print(f"[{time.strftime('%F %T')}] {pos:4d} {shard:28s} {n:6d} rows {rec['seconds']:6.1f}s "
              f"({n / rec['seconds']:.0f}/s)  scan offset-rule floor>900 "
              f"{rec['scan']['floor_gt900_rule_offset']}", flush=True)


# ----------------------------------------------------------------- val equivalence (C1)
def verify_val(val_root: Path, shards: list[str], device: str = "cuda:0", expect_gpu: str | None = None,
               crop_off: int | None = None) -> dict:
    """Re-tokenize val shards through encode_shard and compare with the stored val tokens.

    NDVI is computed from S2L2A here and compared with val's stored-NDVI tokens (original set):
    the two inputs are bit-identical (clean.ndvi_from_s2 on every val row), so the tokens must be.
    crop_off != None is the deliberately-broken control: it must disagree massively.
    """
    gpu = setup_gpu(expect_gpu)
    val_root = Path(val_root)
    index = pd.read_parquet(val_root / "tok_index.parquet")
    toks = load_tokenizers(device)
    report = {"gpu": gpu, "crop_off": crop_off, "shards": {}}
    for shard in shards:
        rows = index[index.source_shard == shard].sort_values("row")
        res, _, secs = encode_shard(val_root, shard, toks, device, rows.stem.tolist(), crop_off=crop_off)
        rep = {}
        for mod, (tk, _) in res.items():
            ref = np.load(val_root / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")[rows.row.values]
            diff = tk != ref
            rep[mod] = {"tokens": int(diff.size), "differ": int(diff.sum()), "rate": float(diff.mean()),
                        "rows_differ": int(diff.any(axis=1).sum()), "scenes_per_s": round(len(rows) / secs[mod], 1)}
        report["shards"][shard] = rep
        print(shard, json.dumps(rep), flush=True)
    return report


# ----------------------------------------------------------------- consolidate
def consolidate(out: Path, val_root: Path, allow_partial: bool = False, dest: Path | None = None) -> dict:
    """Row-aligned <MOD>_tok224/{tokens,present}.npy + metadata.json, and build.json with C3/C4-lite.

    With allow_partial, the root holds only the finished shards (canonical order, rows renumbered):
    a uniform sample, because the shards are. dest defaults to out itself.
    """
    out = Path(out)
    dest = Path(dest) if dest else out
    index = pd.read_parquet(out / "tok_index.parquet")
    done = {p.name.replace(".json", ".tar") for p in (out / "done").glob("*.json")}
    missing = [s for s in canonical_shards() if s not in done]
    if missing and not allow_partial:
        raise RuntimeError(f"{len(missing)} shards not done (e.g. {missing[:3]}); pass allow_partial")
    keep = [s for s in canonical_shards() if s in done]
    idx = index[index.source_shard.isin(keep)].copy()
    idx["row_full"] = idx.row.values
    idx["row"] = np.arange(len(idx))
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "tok_index.parquet.tmp"
    idx[["row", "stem", "source_shard", "corpus"]].to_parquet(tmp, index=False)
    os.replace(tmp, dest / "tok_index.parquet")
    recs = {s: json.loads(done_path(out, s).read_text()) for s in keep}
    rows_of = {s: g.row.values for s, g in idx.groupby("source_shard")}
    n = len(idx)
    meta_mods = {}
    for mod in OUT_MODS:
        tokens = np.zeros((n, C.TOKENS_PER_SAMPLE), dtype=np.uint16)
        present = np.zeros(n, dtype=bool)
        n_nan, used = 0, []
        for s in keep:
            if mod not in out_mods(s):
                continue
            a = np.load(out / C.tok_dir_name(mod) / "shards" / s.replace(".tar", ".npy"))
            r = rows_of[s]
            if a.shape != (len(r), C.TOKENS_PER_SAMPLE):
                raise RuntimeError(f"{mod}/{s}: {a.shape} for {len(r)} rows")
            tokens[r] = a
            present[r] = True
            n_nan += recs[s]["tokens"][mod]["n_nan"]
            used.append(s)
        d = dest / C.tok_dir_name(mod)
        d.mkdir(parents=True, exist_ok=True)
        atomic_npy(tokens, d / "tokens.npy")
        atomic_npy(present, d / "present.npy")
        tmpl = json.loads((Path(val_root) / C.tok_dir_name(mod) / "metadata.json").read_text())
        tm = [recs[s]["tokens"][mod] for s in used]   # empty: e.g. S1GRD in a partial majortom-only root
        meta = {**tmpl,
                "split": "train", "n_samples_total_rows": n, "n_samples_present": int(present.sum()),
                "n_nan_total": int(n_nan), "shards": used,
                "token_min": min(t["min"] for t in tm) if tm else None,
                "token_max": max(t["max"] for t in tm) if tm else None,
                "device": sorted({recs[s]["gpu"] for s in used}),
                "contract_git_rev": sorted({recs[s]["code"] for s in used}),
                "versions": sorted({json.dumps(recs[s]["versions"], sort_keys=True) for s in used}),
                "source": ("computed from S2L2A with clean.ndvi_from_s2 (bit-exact vs TerraMesh's stored NDVI)"
                           if mod == "NDVI" else "TerraMesh train as released (no offset correction)"),
                "prepare": "train_split.gpu_prepare (prepare()'s arithmetic on the GPU)",
                "date": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        atomic_json(meta, d / "metadata.json")
        meta_mods[mod] = {"present": int(present.sum()), "token_max": meta["token_max"]}
        print(f"{mod}: {n} rows, present {int(present.sum())}, tok[{meta['token_min']},{meta['token_max']}]", flush=True)
        del tokens

    # C3: the whole-split offset scan
    sc = [r["scan"] for r in recs.values()]
    worst = max((r["scan"]["floor_gt900_rule_offset"] or 0.0, s) for s, r in recs.items())
    wmeds = [x["water_B12_median_rule_offset"] for x in sc if x["water_B12_median_rule_offset"] is not None]
    c3 = {"shards": len(sc), "scanned": sum(x["n"] for x in sc),
          "rule_offset_scanned": sum(x["n_rule_offset"] for x in sc),
          "floor_gt900_rule_offset_pooled": float(np.average(
              [x["floor_gt900_rule_offset"] or 0.0 for x in sc], weights=[x["n_rule_offset"] or 0 for x in sc]))
          if sum(x["n_rule_offset"] for x in sc) else None,
          "floor_gt900_rule_clean_pooled": float(np.average(
              [x["floor_gt900_rule_clean"] or 0.0 for x in sc], weights=[x["n_rule_clean"] or 0 for x in sc])),
          "worst_shard": {"shard": worst[1], "floor_gt900_rule_offset": worst[0]},
          "water_B12_median_of_shard_medians": float(np.median(wmeds)) if wmeds else None}
    c3["pass"] = bool((c3["floor_gt900_rule_offset_pooled"] or 0) <= 0.15 and worst[0] <= 0.30
                      and (c3["water_B12_median_of_shard_medians"] or 0) < 300)
    build = {"created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "split": "train", "rows": n,
             "shards": len(keep), "shards_missing": len(missing), "partial": bool(missing),
             "modalities": meta_mods, "offset_free": c3["pass"], "c3_offset_scan": c3,
             "plan": json.loads((out / "plan.json").read_text())}
    atomic_json(build, dest / "build.json")
    print("C3 offset scan:", json.dumps(c3), flush=True)
    return build


# ----------------------------------------------------------------- Coords
def coords_tokens(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """tokenize_coords.py's logic (0.25-degree snap, the -0.0 fix, ragged fallback), run on the
    UNIQUE snapped pairs and broadcast back. verify on val: identical to val/Coords_tok."""
    import torch
    from terratorch.models.backbones.terramind.tokenizer.tokenizer_register import (
        terramind_v1_coords_tokenizer)
    lon = (np.round(np.asarray(lon) * 4) / 4) + 0.0
    lat = (np.round(np.asarray(lat) * 4) / 4) + 0.0
    lon[lon == 0] = 0.0
    lat[lat == 0] = 0.0
    pairs, inv = np.unique(np.stack([lon, lat], 1), axis=0, return_inverse=True)
    ct = terramind_v1_coords_tokenizer(pretrained=True)
    coords = torch.tensor(pairs, dtype=torch.float32)
    rows, lens = [], set()
    for i in range(0, len(coords), 512):
        chunk = coords[i:i + 512]
        try:
            t = ct.encode(chunk)["tensor"]
            lens.add(t.shape[1])
            rows.append(t.cpu().numpy())
        except ValueError:                      # ragged -> per sample
            for j in range(len(chunk)):
                t = ct.encode(chunk[j:j + 1])["tensor"]
                lens.add(t.shape[1])
                rows.append(t.cpu().numpy())
    if len(lens) != 1:
        raise RuntimeError(f"ragged coordinate tokens after the -0.0 fix: lengths {lens}")
    return np.concatenate(rows).astype(np.uint16)[inv.reshape(-1)]


def write_coords(root: Path, meta_path: Path, val_root: Path) -> dict:
    idx = pd.read_parquet(Path(root) / "tok_index.parquet")
    meta = pd.read_parquet(meta_path, columns=["zarr", "center_lon", "center_lat"])
    meta["stem"] = meta.zarr.str[: -len(".zarr.zip")]
    m = idx.merge(meta[["stem", "center_lon", "center_lat"]], on="stem", how="left", validate="one_to_one")
    if len(m) != len(idx) or m.center_lon.isna().any():
        raise RuntimeError("index and metadata disagree")
    tok = coords_tokens(m.center_lon.values, m.center_lat.values)
    d = Path(root) / C.tok_dir_name("Coords")
    d.mkdir(parents=True, exist_ok=True)
    atomic_npy(tok, d / "tokens.npy")
    atomic_npy(np.ones(len(tok), dtype=bool), d / "present.npy")
    tmpl = json.loads((Path(val_root) / C.tok_dir_name("Coords") / "metadata.json").read_text())
    md = {**tmpl, "split": "train", "n_samples": int(len(tok)), "observed_id_min": int(tok.min()),
          "observed_id_max": int(tok.max()), "row_order": "tok_index.parquet of this root",
          "computed_on": "unique snapped pairs, broadcast (train_split.coords_tokens)",
          "date": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if int(tok.max()) >= int(tmpl["tokenizer_id_bound"]):
        raise RuntimeError(f"Coords id {int(tok.max())} >= tokenizer_id_bound {tmpl['tokenizer_id_bound']}")
    atomic_json(md, d / "metadata.json")
    return {"rows": int(len(tok)), "id_min": int(tok.min()), "id_max": int(tok.max())}
