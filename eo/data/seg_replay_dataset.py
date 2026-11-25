# eo/data/seg_replay_dataset.py
"""A2 Step 4.6's candidate RP (`.venv` / `mor-eo`): Step 4.1's segmentation fine-tune with TerraMesh's own sequences
replayed in every step. The design and its reading rule are docs/plans/a2_segmentation.md, Step 4, "4.6, the
forgetting-bounded recipe (item 90)", item 3.

ONE INDEX = ONE MICRO-BATCH. A step is M = S + B consecutive indices: S segmentation micro-batches, then B base ones.
  * Segmentation micro-batch j of step s holds the SegDataset's examples s * batch + j * per .. + per - 1 of the draw
    epoch, so the S of them together are exactly the batch R0's run reads at step s (same seed, same order, same views,
    gate Z28a). Its sequences are the task's own length (SegDataset.required: 394 tokens for PASTIS-1 and CloudSen12),
    labels on the map codes and EO_MAP only, as in R0.
  * Base micro-batch b of step s holds `base_per_microbatch` Major TOM rows of the TerraMesh train root, drawn with
    replacement from its full rows (every image block present) by a generator keyed on (seed, REPLAY, draw epoch, s, b).
    Each is the base's own kind of sequence: its five image blocks in a random order keyed on the row
    (map_mix_dataset.MapScenes, R1's base stream), 990 tokens, labels on every token (gate Z28b).
HF weighs every labelled token of a step the same (num_items_in_batch over the accumulation window, as R1 relies on), so
the base stream's share of the gradient is its share of the labelled tokens (gate Z28c): RP-1/2 (2 x 3 sequences) is
48.5 % of a PASTIS step, RP-1/4 (1 x 2) 23.9 %.

THE EPOCH. Every draw is a pure function of (seed, draw epoch, index); the draw epoch is set by SegEpochCallback from
global_step // steps_per_epoch (`set_draw_epoch`, never `set_epoch`, CLAUDE.md), and the trainer reads indices in order
(`orders_itself`). HF's epoch is len / M = the segmentation epoch's steps (gate Z28d), so the plateau stop's epochs are
R0's. A fine-tune is never resumed (the A2 cell restarts it from step 0).

THE EVAL SET is the run's unchanged segmentation eval (fold-4 or val map CE, the plateau stop's signal), at HF's usual
per-device batches: `collate` unwraps a micro-batch and stacks a list of examples as HF's default collator does.

Runs in `.venv`, and in `mor-eo` on RCP: numpy, torch, pandas; never terratorch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from eo.data.map_mix_dataset import L_BASE, MapScenes, R1Config, _base_dataset, _mm, full_rows, roots
from eo.data.seg_dataset import SegDataset, SegEpochCallback, spec_from_cfg
from eo.data.temporal_dataset import stack_trim, unwrap_microbatch

REPLAY_STREAM = 46                                   # the replay draw's key (A2 Step 4.6)


@dataclass(frozen=True)
class ReplayConfig:
    """The `replay` block."""
    enable: bool = False
    seg_per_microbatch: int = 8                      # R0's per-device batch: 4 micro-batches of 8 = its 32
    base_microbatches: int = 2
    base_per_microbatch: int = 3

    @classmethod
    def from_cfg(cls, cfg) -> "ReplayConfig":
        from omegaconf import OmegaConf
        raw = cfg.get("replay", None)
        if raw is not None and OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
        raw = dict(raw or {})
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"replay: unknown keys {unknown}")
        out = cls(**raw)
        for k in ("seg_per_microbatch", "base_per_microbatch"):
            if int(getattr(out, k)) < 1:
                raise ValueError(f"replay.{k} must be a positive integer")
        if int(out.base_microbatches) < 0:
            raise ValueError("replay.base_microbatches must be >= 0")
        return out

    def record(self) -> Dict:
        return asdict(self)


def collate(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """A micro-batch (training) is unwrapped; a list of single examples (the segmentation eval) is stacked."""
    if len(items) == 1 and items[0]["input_ids"].ndim == 2:
        return unwrap_microbatch(items)
    return stack_trim(items, trim=False)


class SegReplayMicrobatches(Dataset):
    """Step 4.6's RP training set (module docstring)."""

    orders_itself = True
    collate_fn = staticmethod(collate)

    def __init__(self, seg: SegDataset, scenes: Optional[MapScenes], base_rows: np.ndarray, rc: ReplayConfig,
                 seed: int):
        super().__init__()
        if seg.spec.split != "train":
            raise ValueError("the replay set wraps a training SegDataset")
        if seg.batch % int(rc.seg_per_microbatch):
            raise ValueError(f"the segmentation batch {seg.batch} is not a whole number of micro-batches of "
                             f"{rc.seg_per_microbatch}")
        if rc.base_microbatches and (scenes is None or len(base_rows) == 0):
            raise ValueError("replay needs the base rows of a TerraMesh root")
        self.seg, self.scenes, self.rc, self.seed = seg, scenes, rc, int(seed)
        self.pool = np.asarray(base_rows, np.int64)
        self.S = seg.batch // int(rc.seg_per_microbatch)
        self.B = int(rc.base_microbatches)
        self.M = self.S + self.B
        self.steps = seg.steps_per_epoch()

    # the epoch, as SegDataset's
    @property
    def draw_epoch(self) -> int:
        return self.seg.draw_epoch

    def set_draw_epoch(self, epoch: int) -> None:
        self.seg.set_draw_epoch(epoch)

    def steps_per_epoch(self) -> int:
        return self.steps

    def epoch_callback(self, steps_per_epoch: Optional[int] = None, verbose: bool = True) -> SegEpochCallback:
        return SegEpochCallback(self, steps_per_epoch or self.steps, verbose)

    def __len__(self) -> int:
        return self.steps * self.M

    def spec(self, i: int):
        """(kind, step, j): index i is the j-th segmentation or base micro-batch of its step."""
        s, j = divmod(int(i), self.M)
        return ("seg", s, j) if j < self.S else ("base", s, j - self.S)

    def seg_indices(self, s: int, j: int) -> np.ndarray:
        per = int(self.rc.seg_per_microbatch)
        return s * self.seg.batch + j * per + np.arange(per)

    def base_rows_at(self, s: int, b: int, epoch: Optional[int] = None) -> np.ndarray:
        e = self.draw_epoch if epoch is None else int(epoch)
        rng = np.random.default_rng([self.seed, REPLAY_STREAM, e, int(s), int(b)])
        return self.pool[rng.integers(len(self.pool), size=int(self.rc.base_per_microbatch))]

    def base_sequence(self, row: int) -> Dict[str, torch.Tensor]:
        out = self.scenes.sequence(int(row), self.scenes.order(int(row), False, "base"), L_BASE)
        if int(out["attention_mask"].sum()) != L_BASE:
            raise RuntimeError(f"replay row {row} is {int(out['attention_mask'].sum())} tokens, not {L_BASE}")
        return out

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        kind, s, j = self.spec(i)
        if kind == "seg":
            return stack_trim([self.seg[int(k)] for k in self.seg_indices(s, j)], trim=False)
        return stack_trim([self.base_sequence(int(r)) for r in self.base_rows_at(s, j)], trim=False)

    def labelled_share(self, s: int = 0) -> float:
        """The base stream's share of step s's labelled tokens (what HF weighs)."""
        n = {"seg": 0, "base": 0}
        for i in range(s * self.M, (s + 1) * self.M):
            n[self.spec(i)[0]] += int((self[i]["labels"] != -100).sum())
        return n["base"] / (n["seg"] + n["base"])


def build_train(cfg, base_root=None) -> SegReplayMicrobatches:
    """pretrain.py's builder (lm_dataset/load_dataset.py, geobench_seg with replay.enable). The base rows come from the
    TerraMesh train root (TERRAMESH_TRAIN_TOK_ROOT, map_mix_dataset.roots), never val: the forgetting read scores val's
    held-out rows. `base_root` is for the gate only."""
    rc = ReplayConfig.from_cfg(cfg)
    if not rc.enable:
        raise ValueError("replay.enable is false")
    for k in ("per_device_train_batch_size",):
        if int(cfg.get(k)) != 1:
            raise ValueError(f"{k} must be 1 under replay: one index is one micro-batch")
    spec = spec_from_cfg(cfg, "train")
    seed = int(cfg.get("seed", 42))
    S = int(cfg.get("total_batch_size")) - int(rc.base_microbatches)
    if S < 1:
        raise ValueError(f"total_batch_size {cfg.get('total_batch_size')} leaves no segmentation micro-batch")
    seg_batch = S * int(rc.seg_per_microbatch)
    seg = SegDataset(spec, max_length=None, seed=seed, batch=seg_batch)    # the task's own length (no padding)
    scenes, rows, split = None, np.zeros(0, np.int64), "train"
    if rc.base_microbatches:
        mm = _mm(cfg)
        if base_root is None:
            base_root = roots(cfg)[0]
        else:
            split = "val"                                                   # the gate's stand-in root
        base = _base_dataset(base_root, split, mm, seed)
        import pandas as pd
        from pathlib import Path
        idx = pd.read_parquet(Path(base_root) / "tok_index.parquet", columns=["row", "corpus"])
        corpus = idx.sort_values("row").corpus.values
        rows = full_rows(base, np.nonzero(corpus == "majortom")[0])
        scenes = MapScenes(base, None, R1Config(), seed)
    ds = SegReplayMicrobatches(seg, scenes, rows, rc, seed)
    share = ds.labelled_share(0)
    print(f"seg replay train: {spec.task} {spec.frames} arm {spec.arm}: {len(seg.samples)} samples, {ds.steps} steps "
          f"an epoch, {ds.S} segmentation micro-batches of {rc.seg_per_microbatch} + {ds.B} base micro-batches of "
          f"{rc.base_per_microbatch} a step; base pool {len(rows)} Major TOM rows of {base_root} ({split}); base share "
          f"of step 0's labelled tokens {share:.4f}")
    return ds
