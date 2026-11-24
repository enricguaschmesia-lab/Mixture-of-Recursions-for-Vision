# eo/data/map_mix_dataset.py
"""A2 R1's mid-training data (A2 Step 5.1, `.venv` / `mor-eo`): TerraMesh train scenes, with each `ssl4eos12` row's
LULC rendered as an RGB map of the frozen aligned tokenizer, mixed with Major TOM scenes at half the tokens. The
design and its reading rule are docs/plans/a2_segmentation.md, Step 5.1.

THE MAP SEQUENCE (an `ssl4eos12` row). Its five image blocks (S2L2A, S1GRD, DEM, NDVI, LULC, the base dataset's chunks
token for token) and the map block

    [BO_MAP_LULC  81,296 + c_1 ... 81,296 + c_196  EO_MAP]          eo/data/map_vocab.py, the aligned slot

with c the row's codes in the artifact (eo/scripts/tokenize_train_lulc_maps.py, raster order on the 14 x 14 grid of the
image blocks). The six blocks come in a random order drawn from (seed, stream, row), as pretraining shuffles its blocks:
1,188 tokens, sequential positions, labels on every token. Modality ids are the image modalities' own, 10 for the map
codes (`loss_MAP`), and 0 for every BO/EO (lm_dataset.sequence_assembly), as in eo/data/seg_dataset.py.

THE BASE SEQUENCE (a Major TOM row): its five blocks (S2L2A, S1RTC, DEM, NDVI, LULC) in a random order, 990 tokens: the
base's own kind of sequence, every block token-identical to TerraMeshTokenDataset's (gate Z15). Only Major TOM rows
serve the base stream and only `ssl4eos12` rows the map stream, so no row appears in both.

ONE INDEX = ONE MICRO-BATCH of equal tokens: 10 map sequences (10 x 1,188 = 11,880) or 12 base sequences (12 x 990 =
11,880). Even indices are map micro-batches, odd ones base, and total_batch_size (micro-batches per step) is even, so
every step is exactly half map tokens and half base tokens. HF passes num_items_in_batch: each token weighs the same in
a step, so a stream's share of the gradient is its share of the tokens, one half (A1's temporal_dataset says why).
per_device_*_batch_size is 1 and `collate_fn` unwraps the micro-batch.

STREAMS. Each stream walks seeded permutations of its pool: position j of the walk is pool[perm_p[j mod n]], with
p = j // n and perm_p keyed on (seed, stream, p). Map micro-batch k holds walk positions 10 k .. 10 k + 9, base
micro-batch k positions 12 k .. 12 k + 11. So a micro-batch is a pure function of its index and the seed: no epoch, no
set_epoch, no worker id (CLAUDE.md: an epoch-dependent dataset must not expose set_epoch). The trainer reads it with a
SequentialSampler (`orders_itself`, util/trainer_pt.py), which a resume skips into exactly where the uninterrupted run
stood. The length is `passes` passes over the map pool, far past any budget, so HF never starts a second epoch (which
would replay index 0); the builder refuses a schedule longer than that.

THE MAP ARTIFACTS. A train artifact row-aligned with the train root, a val artifact with val's root. Each records the
sha256 of the tok_index.parquet it was written against, and `MapArtifact` refuses one whose index is not its root's,
whose arm, codes or tokens are not the picked tokenizer's (pinned below from Step 4.0's pick.json, so the check needs no
workstation path), or whose rows the pool needs are absent. They live outside the token roots: the train root on g49 is
gated by its build.json and read by live runs.

THE EVAL SETS (a dict, evaluated one by one, each logged as eval_<name>_loss with its per-modality CE):
    single   the base's held-out eval exactly: the 4,416 rows of eval_rows_v1 on val_clean at 990 tokens, through
             TerraMeshTokenDataset and temporal_dataset.SingleEval (4 rows a micro-batch, the base's eval batch: its
             orders are keyed on the worker id, so only that size gives the base's own tensors).
             Forgetting is read against the map base's own value of it.
    map      the same 4,416 rows with their val LULC map (the val artifact), six blocks in an order drawn from
             (eval_seed, row), 1,188 tokens, labels on every token: eval_map_loss_MAP is the held-out map CE.
    map_s2   the same rows as [S2L2A][MAP] (396 tokens; the PASTIS-1 / CloudSen12 layout), labels on the codes and
             EO_MAP only, as an R0 fine-tune's eval: S2 -> map at the downstream layout.
    `map` and `map_s2` come 16 rows a micro-batch (r1.eval_map_per): every sequence has the same labelled tokens, so
    HF's batch-weighted eval loss is their token mean at any size.
Val's Major TOM rows carry S1RTC beside a map, which training never shows (its map rows are all `ssl4eos12`); the map
CE is read per corpus afterwards (eo/experiments/segmentation/r1_read.py), with the pooled number deciding.

⚠ A RESUME MAY NOT CHANGE THE DATA. `data_settings` holds what defines the data (the dataset, the r1 block, the batch,
max_length, seed, multimodal's keys, and the artifacts' index and metadata hashes). pretrain.py's generic hook writes it
as temporal_data.json (the name it gives every data record) and refuses a resume whose settings differ; preflight
refuses it before a pod starts (check_map_data). Schedule keys are free.

Runs in `.venv`, and in `mor-eo` on RCP: numpy, torch, pandas; never terratorch.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from eo.data import map_vocab as MV
from eo.data.temporal_dataset import KEYS, SingleEval, microbatches, stack_trim, unwrap_microbatch
from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
from lm_dataset.sequence_assembly import assemble_sequence

DATASET = "terramesh_map"
MAP = MV.MAP
INPUT = "S2L2A"                                     # map_s2's input block
BLOCK = 198                                         # an image or map block with its BO/EO (196 + 2)
N_IMAGE = 5                                         # image blocks a row carries (S2L2A, S1, DEM, NDVI, LULC)
L_MAP = (N_IMAGE + 1) * BLOCK                       # 1,188
L_BASE = N_IMAGE * BLOCK                            # 990
L_S2 = 2 * BLOCK                                    # 396
STREAMS = {"map": 1, "base": 2, "eval_map": 3}      # the stream id in each draw's key

#: The picked tokenizer (A2 Step 4.0's pick.json, sha256 5ee4c885...): what a map artifact must have been written with.
PICKED = {"aligned": {"tokens": 196, "n_codes": 64_000,
                      "weights_sha256": "17be36cf219cc3aee926b5a4edcbf50871a2b73426d96f51e112dac1bab9e9f9",
                      "palette_sha256": "02605308c34bff172f63c6441203e00393028b85243fc9669ad3da9243110cfe"}}
#: The task token and LULC's assignment (class c -> colour c, eo/maps/palette.TASKS["lulc"], decision 6).
LULC_ASSIGN = list(range(10))


@dataclass(frozen=True)
class R1Config:
    """The `r1` block, defaults filled in. Its `record()` is part of the data record."""
    arm: str = "aligned"
    task: str = "lulc"
    train_maps: Optional[str] = None
    val_maps: Optional[str] = None
    map_per_microbatch: int = 10
    base_per_microbatch: int = 12
    eval_per: int = 4                   # `single`: the base's eval batch, so its tensors are the base's own
    eval_map_per: int = 16              # `map`, `map_s2`: orders keyed on (eval_seed, row), so any size serves the same
    eval_seed: int = 20261122
    passes: int = 4

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping]) -> "R1Config":
        raw = dict(raw or {})
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - names)
        if unknown:
            raise ValueError(f"r1: unknown keys {unknown}; known {sorted(names)}")
        out = cls(**{k: v for k, v in raw.items() if v is not None or k in ("train_maps", "val_maps")})
        out.validate()
        return out

    @classmethod
    def from_cfg(cls, cfg) -> "R1Config":
        from omegaconf import OmegaConf
        blk = cfg.get("r1", None)
        if blk is not None and OmegaConf.is_config(blk):
            blk = OmegaConf.to_container(blk, resolve=True)
        return cls.from_mapping(blk)

    def validate(self) -> None:
        if self.arm not in PICKED:
            raise ValueError(f"r1.arm {self.arm!r}: R1 runs on the picked tokenizer only ({sorted(PICKED)}, Step 4.0)")
        if MV.spec(self.arm)[1] != PICKED[self.arm]["tokens"] or MV.spec(self.arm)[1] != BLOCK - 2:
            raise ValueError(f"r1.arm {self.arm!r} does not write {BLOCK - 2} tokens a map")
        if self.task != "lulc":
            raise ValueError(f"r1.task {self.task!r}: R1 writes TerraMesh's LULC (BO_MAP_LULC)")
        for k in ("map_per_microbatch", "base_per_microbatch", "eval_per", "eval_map_per", "passes"):
            if int(getattr(self, k)) < 1:
                raise ValueError(f"r1.{k} must be a positive integer")
        if int(self.map_per_microbatch) * L_MAP != int(self.base_per_microbatch) * L_BASE:
            raise ValueError(f"r1: a map micro-batch ({self.map_per_microbatch} x {L_MAP}) and a base one "
                             f"({self.base_per_microbatch} x {L_BASE}) must hold the same tokens, or the streams are "
                             f"not half and half")

    def record(self) -> Dict:
        d = asdict(self)
        d.pop("train_maps")
        d.pop("val_maps")                        # paths differ by machine: the artifacts' hashes are recorded instead
        return d


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class MapArtifact:
    """A map-token artifact checked against its token root. Token arrays are memory-mapped lazily, per worker."""

    def __init__(self, path, root, arm: str = "aligned"):
        self.path, self.root, self.arm = Path(path), Path(root), arm
        mp = self.path / "metadata.json"
        if not mp.is_file():
            raise FileNotFoundError(f"{mp} missing: not a map artifact (eo/scripts/tokenize_train_lulc_maps.py), "
                                    f"or its writer has not finished")
        self.meta = json.loads(mp.read_text())
        self.meta_sha256 = sha256_file(mp)
        pk = PICKED[arm]
        problems = []
        if self.meta.get("arm") != arm:
            problems.append(f"arm {self.meta.get('arm')!r}, want {arm!r}")
        if int(self.meta.get("tokens_per_sample", -1)) != pk["tokens"]:
            problems.append(f"{self.meta.get('tokens_per_sample')} tokens a map, want {pk['tokens']}")
        if int(self.meta.get("tokenizer_id_bound", -1)) != pk["n_codes"]:
            problems.append(f"{self.meta.get('tokenizer_id_bound')} codes, want {pk['n_codes']}")
        fz = self.meta.get("frozen", {})
        if fz.get("weights_sha256") != pk["weights_sha256"]:
            problems.append("written with other tokenizer weights than Step 4.0's pick")
        if self.meta.get("palette_sha256") != pk["palette_sha256"]:
            problems.append("written with another palette than the frozen one")
        if list(self.meta.get("assignment", [])) != LULC_ASSIGN:
            problems.append(f"LULC assignment {self.meta.get('assignment')}, want {LULC_ASSIGN}")
        if not self.meta.get("complete"):
            problems.append("incomplete (its writer stopped before every shard was done)")
        if self.meta.get("token_max") is None or int(self.meta["token_max"]) >= pk["n_codes"]:
            problems.append(f"token_max {self.meta.get('token_max')} not below {pk['n_codes']}")
        idx_sha = sha256_file(self.root / "tok_index.parquet")
        if self.meta.get("index_sha256") != idx_sha:
            problems.append(f"written against index {str(self.meta.get('index_sha256'))[:12]}, but {self.root}'s "
                            f"tok_index.parquet is {idx_sha[:12]}: its rows are not this root's")
        if problems:
            raise RuntimeError(f"map artifact {self.path}: " + "; ".join(problems))
        self.index_sha256 = idx_sha
        self.present = np.load(self.path / "present.npy")
        self.n_rows = int(len(self.present))
        if int(self.meta["n_rows"]) != self.n_rows or int(self.meta["n_present"]) != int(self.present.sum()):
            raise RuntimeError(f"map artifact {self.path}: present.npy disagrees with its metadata")
        self._tokens = None

    def tokens(self) -> np.ndarray:
        if self._tokens is None:
            t = np.load(self.path / "tokens.npy", mmap_mode="r")
            if t.shape != (self.n_rows, PICKED[self.arm]["tokens"]) or t.dtype != np.uint16:
                raise RuntimeError(f"map artifact {self.path}: tokens.npy is {t.shape} {t.dtype}")
            self._tokens = t
        return self._tokens

    def codes(self, row: int) -> np.ndarray:
        if not bool(self.present[row]):
            raise KeyError(f"row {row} has no map in {self.path}")
        return np.asarray(self.tokens()[row]).astype(np.int64)    # uint16 -> int64 BEFORE any offset

    def record(self) -> Dict:
        return {"index_sha256": self.index_sha256, "metadata_sha256": self.meta_sha256,
                "n_present": int(self.present.sum()), "gpu": self.meta.get("gpu"), "split": self.meta.get("split")}

    def __getstate__(self):                       # workers re-open the memory map
        d = dict(self.__dict__)
        d["_tokens"] = None
        return d


class MapScenes:
    """The sequences of one token root's rows: with their map block (`with_map`) or the base's own."""

    def __init__(self, base: TerraMeshTokenDataset, maps: Optional[MapArtifact], cfg: R1Config, seed: int):
        if base.vocab != "nc":
            raise ValueError(f"R1 writes the nc layout (A2's base); the base dataset is {base.vocab!r}")
        self.base, self.maps, self.cfg, self.seed = base, maps, cfg, int(seed)
        self.lay = MV.LAYOUT                      # the map ids over A2's base (nc temporal, 81,296 -> 161,665)
        if self.lay.pad_id != base._V.PAD_ID:
            raise RuntimeError(f"the map layout's PAD {self.lay.pad_id} is not the base's {base._V.PAD_ID}")
        if maps is not None and maps.n_rows != base.n_rows:
            raise RuntimeError(f"map artifact has {maps.n_rows} rows, the root {base.n_rows}")

    def map_chunk(self, row: int) -> torch.Tensor:
        return torch.tensor(self.lay.map_block(self.cfg.arm, self.cfg.task, self.maps.codes(row)), dtype=torch.long)

    def modalities(self, row: int, with_map: bool) -> List[str]:
        return self.base.present_modalities(row) + ([MAP] if with_map else [])

    def order(self, row: int, with_map: bool, stream: str, seed: Optional[int] = None) -> List[str]:
        mods = self.modalities(row, with_map)
        rng = np.random.default_rng([int(self.seed if seed is None else seed), STREAMS[stream], int(row)])
        return [mods[i] for i in rng.permutation(len(mods))]

    def chunk(self, row: int, name: str) -> torch.Tensor:
        return self.map_chunk(row) if name == MAP else self.base._load_chunk(name, row)

    def modality_id(self, name: str) -> int:
        return MV.MAP_MODALITY_ID if name == MAP else int(self.base._V.MODALITY_TO_ID[name])

    def sequence(self, row: int, names: Sequence[str], max_length: int, labels: str = "all") -> Dict[str, torch.Tensor]:
        """The five-key example of `row` with blocks `names` in that order. labels "all" (pretraining) or "map" (the
        codes and EO_MAP only, as an R0 fine-tune's labels)."""
        chunks = [self.chunk(row, n) for n in names]
        out = assemble_sequence(chunks=chunks, chunk_modality_ids=[self.modality_id(n) for n in names],
                                chunk_shufflable=[False] * len(names), max_length=int(max_length),
                                pad_id=self.lay.pad_id, rng=np.random.default_rng(0), shuffle_image_patches=False)
        if labels == "map":
            lab = torch.full_like(out["labels"], -100)
            start = sum(int(c.shape[0]) for c in chunks[:names.index(MAP)])
            lo, hi = start + 1, start + int(chunks[names.index(MAP)].shape[0])      # codes and EO_MAP, not BO_MAP
            lab[lo:hi] = out["input_ids"][lo:hi]
            out["labels"] = lab
        elif labels != "all":
            raise ValueError(f"labels {labels!r}: all or map")
        return out


def full_rows(base: TerraMeshTokenDataset, rows: np.ndarray) -> np.ndarray:
    """The rows (of `rows`) carrying all five image blocks: only those make exact-length sequences."""
    n = np.zeros(base.n_rows, np.int64)
    for m in base.active_modalities:
        n += base._present[m].astype(np.int64)
    return rows[n[rows] == N_IMAGE]


class MapMixMicrobatches(Dataset):
    """R1's training set: even indices map micro-batches, odd ones base micro-batches (module docstring)."""

    collate_fn = staticmethod(unwrap_microbatch)
    orders_itself = True

    def __init__(self, scenes: MapScenes, map_rows: np.ndarray, base_rows: np.ndarray, cfg: R1Config,
                 total_batch_size: int, seed: int, max_length: int, data_settings: Optional[Dict] = None):
        super().__init__()
        if int(total_batch_size) % 2:
            raise ValueError(f"total_batch_size {total_batch_size} (micro-batches a step) must be even: half map, "
                             f"half base")
        if int(max_length) < L_MAP:
            raise ValueError(f"max_length {max_length} < {L_MAP}, a map sequence's length")
        if len(map_rows) < cfg.map_per_microbatch or len(base_rows) < cfg.base_per_microbatch:
            raise ValueError(f"pools too small: {len(map_rows)} map rows, {len(base_rows)} base rows")
        if len(np.intersect1d(map_rows, base_rows)):
            raise RuntimeError("a row is in both pools")
        self.scenes, self.cfg, self.seed = scenes, cfg, int(seed)
        self.pools = {"map": np.asarray(map_rows, np.int64), "base": np.asarray(base_rows, np.int64)}
        self.per = {"map": int(cfg.map_per_microbatch), "base": int(cfg.base_per_microbatch)}
        self.M = int(total_batch_size)
        self.steps_per_pass = -(-len(map_rows) // (self.per["map"] * self.M // 2))
        self.n = int(cfg.passes) * self.steps_per_pass * self.M
        self.max_length = int(max_length)
        self.data_settings = data_settings
        self._perm: Dict = {}

    def __len__(self) -> int:
        return self.n

    @property
    def max_steps(self) -> int:
        """The longest schedule this length serves without HF starting a second epoch."""
        return self.n // self.M

    def spec(self, i: int):
        """(stream, k): index i is the k-th micro-batch of its stream."""
        return ("map" if i % 2 == 0 else "base"), int(i) // 2

    def _permutation(self, stream: str, p: int) -> np.ndarray:
        key = (stream, int(p))
        if key not in self._perm:
            if len(self._perm) > 4:
                self._perm.clear()
            self._perm[key] = np.random.default_rng([self.seed, STREAMS[stream], 1000 + int(p)]).permutation(
                len(self.pools[stream]))
        return self._perm[key]

    def walk(self, stream: str, positions: np.ndarray) -> np.ndarray:
        pool = self.pools[stream]
        n = len(pool)
        out = np.empty(len(positions), np.int64)
        for p in np.unique(positions // n):
            sel = positions // n == p
            out[sel] = pool[self._permutation(stream, int(p))[positions[sel] % n]]
        return out

    def members(self, i: int):
        stream, k = self.spec(i)
        per = self.per[stream]
        return stream, self.walk(stream, np.arange(per * k, per * (k + 1), dtype=np.int64))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        stream, rows = self.members(int(i))
        with_map = stream == "map"
        L = L_MAP if with_map else L_BASE
        seqs = [self.scenes.sequence(int(r), self.scenes.order(int(r), with_map, stream), L) for r in rows]
        for s in seqs:
            if int(s["attention_mask"].sum()) != L:
                raise RuntimeError(f"row in a {stream} micro-batch is {int(s['attention_mask'].sum())} tokens, not {L}")
        return stack_trim(seqs, trim=False)

    def token_share(self) -> Dict[str, float]:
        t = {"map": self.per["map"] * L_MAP, "base": self.per["base"] * L_BASE}
        return {k: v / sum(t.values()) for k, v in t.items()}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_perm"] = {}
        return d


class MapEval(Dataset):
    """`map` (every block and the map, an order drawn from (eval_seed, row), labels on every token) or `map_s2`
    ([S2L2A][MAP], labels on the codes and EO_MAP), over fixed rows, `per` consecutive rows a micro-batch."""

    collate_fn = staticmethod(unwrap_microbatch)

    def __init__(self, scenes: MapScenes, rows: np.ndarray, per: int, layout: str, seed: int):
        super().__init__()
        if layout not in ("map", "map_s2"):
            raise ValueError(layout)
        microbatches(len(rows), per, "eval rows")
        self.scenes, self.rows, self.per, self.layout, self.seed = scenes, np.asarray(rows, np.int64), int(per), layout, int(seed)

    def __len__(self) -> int:
        return len(self.rows) // self.per

    def names(self, row: int) -> List[str]:
        if self.layout == "map_s2":
            return [INPUT, MAP]
        return self.scenes.order(row, True, "eval_map", seed=self.seed)

    def item(self, row: int) -> Dict[str, torch.Tensor]:
        if self.layout == "map_s2":
            return self.scenes.sequence(row, self.names(row), L_S2, labels="map")
        return self.scenes.sequence(row, self.names(row), L_MAP, labels="all")

    def __getitem__(self, m: int) -> Dict[str, torch.Tensor]:
        return stack_trim([self.item(int(r)) for r in self.rows[int(m) * self.per:(int(m) + 1) * self.per]], trim=False)


# --------------------------------------------------------------------------- builders
#: multimodal's keys that define the data, with the default the builders read when one is absent.
DATA_MULTIMODAL = {"split": "val", "token_set": "original", "eval_split": None, "active_modalities": None,
                   "modality_order": "fixed", "vocab": "v1", "allow_partial_train_root": False}


def _mm(cfg) -> Dict:
    from omegaconf import OmegaConf
    mm = cfg.get("multimodal", {}) or {}
    mm = OmegaConf.to_container(mm, resolve=True) if OmegaConf.is_config(mm) else dict(mm)
    return {k: mm.get(k, d) for k, d in DATA_MULTIMODAL.items()}


def validate_config(cfg) -> R1Config:
    """The refusals that do not need data: the r1 block, the layout, the batch. Preflight calls this."""
    rc = R1Config.from_cfg(cfg)
    mm = _mm(cfg)
    if cfg.get("dataset") != DATASET:
        raise ValueError(f"dataset {cfg.get('dataset')!r} is not {DATASET}")
    if mm["vocab"] != "nc":
        raise ValueError("R1 runs in the nc layout (multimodal.vocab: nc): A2's base is Coords-free")
    if mm["split"] != "train":
        raise ValueError("R1 trains on the train split (multimodal.split: train)")
    if mm["token_set"] != "clean":
        raise ValueError("multimodal.token_set must be clean (the train split is the fixed upstream version)")
    if mm["modality_order"] != "random":
        raise ValueError("multimodal.modality_order must be random, as pretraining shuffles its blocks")
    if mm["eval_split"] is None:
        raise ValueError("multimodal.eval_split must name the held-out split (v1): R1's eval sets read it")
    if int(cfg.model_config.vocab_size) != MV.TOTAL_VOCAB_SIZE:
        raise ValueError(f"model_config.vocab_size {cfg.model_config.vocab_size} is not the map layout's "
                         f"{MV.TOTAL_VOCAB_SIZE}")
    if int(cfg.get("max_length")) < L_MAP:
        raise ValueError(f"max_length {cfg.get('max_length')} < {L_MAP}")
    if int(cfg.get("total_batch_size")) % 2:
        raise ValueError("total_batch_size counts micro-batches and must be even (half map, half base)")
    for k in ("per_device_train_batch_size", "per_device_eval_batch_size"):
        if int(cfg.get(k)) != 1:
            raise ValueError(f"{k} must be 1: one index is one micro-batch")
    for k in ("train_maps", "val_maps"):
        if not getattr(rc, k):
            raise ValueError(f"r1.{k} must name a map artifact")
    return rc


def data_settings(cfg, train_maps: MapArtifact, val_maps: MapArtifact) -> Dict:
    rc = R1Config.from_cfg(cfg)
    return {"dataset": DATASET, "r1": rc.record(), "total_batch_size": int(cfg.get("total_batch_size")),
            "max_length": int(cfg.get("max_length")), "seed": int(cfg.get("seed", 42)), "multimodal": _mm(cfg),
            "train_maps": train_maps.record(), "val_maps": val_maps.record()}


def _base_dataset(root, split: str, mm: Dict, seed: int, rows=None, max_length: int = L_BASE) -> TerraMeshTokenDataset:
    return TerraMeshTokenDataset(root_dir=str(root), split=split, active_modalities=mm["active_modalities"],
                                 max_length=max_length, modality_order="random", seed=seed,
                                 shuffle_image_patches=False, rows=rows, vocab="nc")


def roots(cfg):
    """(train root, val root) as the run resolves them: TERRAMESH_TRAIN_TOK_ROOT and val_clean."""
    from eo.terramesh_tok.clean import resolve_token_root
    from eo.terramesh_tok.train_split import resolve_train_root
    from lm_dataset.load_dataset import TERRAMESH_TOK_ROOT
    mm = _mm(cfg)
    train = resolve_train_root(os.environ.get("TERRAMESH_TRAIN_TOK_ROOT"), mm["token_set"],
                               allow_partial=bool(mm["allow_partial_train_root"]))
    return Path(train), Path(resolve_token_root(TERRAMESH_TOK_ROOT, mm["token_set"]))


def eval_rows(cfg, val_root) -> np.ndarray:
    from eo.data.eval_split import load_eval_rows
    from eo.data.temporal_dataset import _eval_split_path
    return np.asarray(load_eval_rows(path=_eval_split_path(_mm(cfg)["eval_split"]), root_dir=str(val_root)), np.int64)


def build_train(cfg) -> MapMixMicrobatches:
    rc = validate_config(cfg)
    mm = _mm(cfg)
    seed = int(cfg.get("seed", 42))
    train_root, val_root = roots(cfg)
    base = _base_dataset(train_root, "train", mm, seed)
    tmaps = MapArtifact(rc.train_maps, train_root, rc.arm)
    vmaps = MapArtifact(rc.val_maps, val_root, rc.arm)
    import pandas as pd
    idx = pd.read_parquet(train_root / "tok_index.parquet", columns=["row", "corpus"])
    corpus = idx.sort_values("row").corpus.values
    ssl = np.nonzero(corpus == "ssl4eos12")[0]
    mt = np.nonzero(corpus == "majortom")[0]
    if not (tmaps.present[ssl].all() and not tmaps.present[mt].any()):
        raise RuntimeError(f"{rc.train_maps}: its maps are not exactly the root's ssl4eos12 rows "
                           f"({int(tmaps.present[ssl].sum())} of {len(ssl)} present, {int(tmaps.present[mt].sum())} "
                           f"Major TOM rows present)")
    map_rows, base_rows = full_rows(base, ssl), full_rows(base, mt)
    ds = MapMixMicrobatches(MapScenes(base, tmaps, rc, seed), map_rows, base_rows, rc,
                            int(cfg.get("total_batch_size")), seed, int(cfg.get("max_length")),
                            data_settings=data_settings(cfg, tmaps, vmaps))
    steps = int(cfg.get("num_train_steps"))
    if steps > ds.max_steps:
        raise ValueError(f"num_train_steps {steps} > {ds.max_steps}, the steps this dataset serves in one HF epoch "
                         f"(r1.passes {rc.passes}): a second epoch would replay index 0")
    print(f"R1 train: {len(map_rows)} map rows ({len(ssl) - len(map_rows)} short excluded), {len(base_rows)} base rows "
          f"({len(mt) - len(base_rows)} short excluded), {ds.M} micro-batches a step "
          f"({ds.M // 2 * rc.map_per_microbatch} map + {ds.M // 2 * rc.base_per_microbatch} base sequences, "
          f"{ds.M // 2 * (rc.map_per_microbatch * L_MAP + rc.base_per_microbatch * L_BASE)} tokens), "
          f"a map-pool pass {ds.steps_per_pass} steps, token share {ds.token_share()}")
    return ds


def build_eval(cfg) -> Dict[str, Dataset]:
    rc = validate_config(cfg)
    mm = _mm(cfg)
    seed = int(cfg.get("seed", 42))
    _, val_root = roots(cfg)
    rows = eval_rows(cfg, val_root)
    # single: exactly the base's eval set (TerraMeshTokenDataset over the held-out rows at 990, 4 rows a batch)
    single = SingleEval(_base_dataset(val_root, "val", mm, seed, rows=rows, max_length=L_BASE), int(rc.eval_per))
    vbase = _base_dataset(val_root, "val", mm, seed)
    vmaps = MapArtifact(rc.val_maps, val_root, rc.arm)
    missing = int((~vmaps.present[rows]).sum())
    if missing:
        raise RuntimeError(f"{rc.val_maps}: {missing} held-out rows have no map")
    scenes = MapScenes(vbase, vmaps, rc, seed)
    full = full_rows(vbase, rows)
    if len(full) != len(rows):
        raise RuntimeError(f"{len(rows) - len(full)} held-out rows lack an image block")
    return {"single": single,
            "map": MapEval(scenes, rows, rc.eval_map_per, "map", rc.eval_seed),
            "map_s2": MapEval(scenes, rows, rc.eval_map_per, "map_s2", rc.eval_seed)}
