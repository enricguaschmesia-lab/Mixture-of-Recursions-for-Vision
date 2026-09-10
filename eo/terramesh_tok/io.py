"""Reading TerraMesh WebDataset shards.

Each val/<MOD>/<shard>.tar holds one <stem>.zarr.zip per sample. The array is
a single blosc/zstd chunk with no filters, so it is decoded directly rather
than through zarr+fsspec -- verified byte-identical to
zarr.open_consolidated(..., zarr_format=2) on all 8 modalities, and much
faster (see notes/tokenizer_bringup.md, Step 3.9).

NOTE: IBM's own terramesh.py calls zarr.open_consolidated() WITHOUT
zarr_format=2, which raises GroupNotFoundError under zarr 3.x. If that loader
is ever used directly it must be patched.
"""
from __future__ import annotations
import io
import json
import tarfile
import zipfile

import numpy as np
from numcodecs import Blosc

ROOT = "/data/enric/data/TerraMesh/val"
_CODEC = Blosc()


def _decode_zarr_zip(raw: bytes, key: str = "bands") -> np.ndarray:
    z = zipfile.ZipFile(io.BytesIO(raw))
    meta = json.loads(z.read(f"{key}/.zarray"))
    chunks, shape = meta["chunks"], meta["shape"]
    if chunks != shape:
        raise NotImplementedError(f"multi-chunk array not handled: {meta}")
    name = f"{key}/" + ".".join("0" for _ in shape)
    buf = _CODEC.decode(z.read(name))
    return np.frombuffer(buf, dtype=np.dtype(meta["dtype"])).reshape(shape)


def read_sample(modality: str, shard: str, stem: str | None = None,
                root: str = ROOT) -> tuple[str, np.ndarray]:
    """Return (stem, array) for one sample. stem=None -> first member."""
    with tarfile.open(f"{root}/{modality}/{shard}") as tf:
        for m in tf:
            if not m.name.endswith(".zarr.zip"):
                continue
            s = m.name[: -len(".zarr.zip")]
            if stem is None or s == stem:
                return s, _decode_zarr_zip(tf.extractfile(m).read())
    raise KeyError(f"{stem} not found in {modality}/{shard}")


def iter_shard(modality: str, shard: str, limit: int | None = None,
               root: str = ROOT):
    """Yield (stem, array) for every sample in a shard, streaming."""
    n = 0
    with tarfile.open(f"{root}/{modality}/{shard}") as tf:
        for m in tf:
            if not m.name.endswith(".zarr.zip"):
                continue
            yield m.name[: -len(".zarr.zip")], _decode_zarr_zip(
                tf.extractfile(m).read())
            n += 1
            if limit is not None and n >= limit:
                return


def read_extras(modality: str, shard: str, stem: str, root: str = ROOT):
    """center_lon / center_lat / cloud_mask for one sample, when present."""
    out = {}
    with tarfile.open(f"{root}/{modality}/{shard}") as tf:
        for m in tf:
            if m.name != f"{stem}.zarr.zip":
                continue
            raw = tf.extractfile(m).read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            names = {n.split("/")[0] for n in z.namelist() if "/" in n}
            for k in ("center_lon", "center_lat", "cloud_mask"):
                if k in names:
                    out[k] = _decode_zarr_zip(raw, key=k)
            return out
    raise KeyError(stem)
