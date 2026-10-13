# eo/data/ben_dataset.py
"""BigEarthNet S1/S2 tokens -> label block, for fine-tuning and testing (Phase 6 D6.5).

One example per patch:

    [BO_S2 s2... EO_S2] [BO_S1 s1... EO_S1]  BO_BEN c_1 ... c_k EO_BEN  <PAD>...
     \\_____ inputs: {S1}, {S2} or {S1, S2} _____/   \\__ label block, ALWAYS last __/

* Inputs use their EXISTING TerraMesh slots and BO/EO: BEN's S2 was tokenized with the
  S2L2A tokenizer and its S1 with the S1GRD one (`ben_vocab.INPUT_MODALITIES`), so their ids
  mean exactly what they meant in pretraining.
* The label block holds the patch's present classes in canonical order (`ben_vocab`).
  BEN is never an input (Gianfranco, 2026-09-30).
* Assembly goes through `lm_dataset/sequence_assembly.py`, shared on purpose with the CLEVR
  and TerraMesh datasets, so padding, masks and positions cannot drift.
* The batch dict is the closed five-key set.

Input condition. In training (`condition="random"`) each example draws its input set
uniformly from {S1}, {S2}, {S1, S2}, and a random order when both are present, from an RNG
keyed on (seed, epoch, row). The EPOCH is in the key so a patch is not pinned to one
condition for the whole fine-tune; `set_epoch` is called by a TrainerCallback before HF
spawns the epoch's dataloader workers (pretrain.py). Keyed on the ORIGINAL row id, never on
the position in a subset, and not on the worker id -- unlike the TerraMesh dataset, whose
order depends on `dataloader_num_workers` (CLAUDE.md); that dependence is not reproduced.

For testing, `condition` is one of "S1", "S2", "S1S2" and the order is FIXED (registry
order: S2L2A then S1GRD), as Phase 3 fixes its prompt order.

Loss coverage (PHASE6_PLAN §4.5). `loss_on="labels"` (the main runs) keeps `labels` only on
c_1..c_k and EO_BEN, i.e. the tokens the model must generate after the prompt's BO_BEN.
`loss_on="labels_inputs"` (ablation S6.b) also keeps every input token.

Runs in `.venv`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from eo.ben import labels as L
from eo.data import ben_vocab as V
from eo.terramesh_tok.contract import CROP, TOKENS_PER_SAMPLE, tok_dir_name
from lm_dataset.sequence_assembly import assemble_sequence

DEFAULT_ROOT = os.environ.get("BEN_ROOT", "/data/enric/data/BigEarthNet")
INDEX_NAME = "index_v1.parquet"
CONDITIONS = ("S1", "S2", "S1S2")
_COND_INPUTS = {"S1": ["S1"], "S2": ["S2"], "S1S2": ["S2", "S1"]}   # fixed order: S2L2A, S1GRD
INPUT_WIDTH = TOKENS_PER_SAMPLE + 2
# The absolute bound: both inputs plus a label block carrying all 19 classes.
MAX_LENGTH_BOUND = 2 * INPUT_WIDTH + V.N_CLASSES + 2          # 417


def load_index(root: str = DEFAULT_ROOT):
    import pandas as pd
    return pd.read_parquet(Path(root) / INDEX_NAME)


class BenTokenDataset(Dataset):
    def __init__(self, root: str = DEFAULT_ROOT, role: str = "ft_train",
                 condition: str = "random", seed: int = 42,
                 max_length: Optional[int] = MAX_LENGTH_BOUND, loss_on: str = "labels",
                 rows: Optional[Sequence[int]] = None, index=None):
        super().__init__()
        if condition != "random" and condition not in CONDITIONS:
            raise ValueError(f"condition must be 'random' or one of {CONDITIONS}")
        if loss_on not in ("labels", "labels_inputs"):
            raise ValueError(f"loss_on must be 'labels' or 'labels_inputs', got {loss_on!r}")
        self.root = Path(root)
        self.role, self.condition, self.seed, self.loss_on = role, condition, int(seed), loss_on
        self.epoch = 0

        idx = load_index(root) if index is None else index
        if "labels19" not in idx.columns:
            raise RuntimeError(f"{self.root / INDEX_NAME} has no labels yet "
                               f"(run build_ben_index.py labels after packing)")
        assert (idx["row"].values == np.arange(len(idx))).all(), "index rows must be 0..N-1"
        self.n_rows = len(idx)
        self.labels19 = idx["labels19"].to_numpy(np.int64)
        self.patch = idx["s2"].tolist()
        if rows is None:
            if role not in idx.columns:
                raise KeyError(f"no role column {role!r} in the index")
            rows = np.flatnonzero(idx[role].to_numpy(bool))
        self.rows = np.asarray(rows, dtype=np.int64)
        if self.rows.size == 0 or len(np.unique(self.rows)) != len(self.rows):
            raise ValueError("rows must be non-empty and unique")
        lab = self.labels19[self.rows]
        if (lab < 0).any():
            raise RuntimeError(f"{int((lab < 0).sum())} selected rows were never packed/labelled")
        # A patch whose only v1 labels are classes the 19-class grouping drops (e.g. Airports)
        # has an EMPTY label set: 25 of the 243,838 packed rows. SMARTIES keeps them with an
        # all-zero target, so they stay, as the block `BO_BEN EO_BEN`.
        n_lab = np.array([bin(int(m)).count("1") for m in lab])
        self.n_empty = int((n_lab == 0).sum())

        self._present: Dict[str, np.ndarray] = {}
        for src, mod in V.INPUT_MODALITIES.items():
            d = self.root / tok_dir_name(mod)
            meta = json.loads((d / "metadata.json").read_text())
            if int(meta.get("crop", -1)) != CROP or int(meta.get("tokens_per_sample", -1)) != TOKENS_PER_SAMPLE:
                raise RuntimeError(f"{d}: crop/tokens_per_sample {meta.get('crop')}/"
                                   f"{meta.get('tokens_per_sample')} != contract {CROP}/{TOKENS_PER_SAMPLE}")
            if meta.get("tokenizer_modality") != mod:
                raise RuntimeError(f"{d}: tokenized with {meta.get('tokenizer_modality')!r}, "
                                   f"expected {mod!r}")
            p = np.load(d / "present.npy")
            if p.shape[0] != self.n_rows:
                raise RuntimeError(f"{d}: {p.shape[0]} rows, index has {self.n_rows}")
            missing = int((~p[self.rows]).sum())
            if missing:
                raise RuntimeError(f"{d}: {missing} of the selected rows are not tokenized")
            self._present[src] = p
        self.required_length = int(2 * INPUT_WIDTH + n_lab.max() + 2) if condition in ("random", "S1S2") \
            else int(INPUT_WIDTH + n_lab.max() + 2)
        if max_length is None:
            self.max_length = self.required_length
        elif max_length < self.required_length:
            raise ValueError(f"max_length={max_length} < required {self.required_length}")
        else:
            self.max_length = int(max_length)
        self._tokens: Optional[Dict[str, np.ndarray]] = None

    # --- plumbing ----------------------------------------------------------
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _arrays(self) -> Dict[str, np.ndarray]:
        if self._tokens is None:       # memory-map lazily, inside each worker
            self._tokens = {}
            for src, mod in V.INPUT_MODALITIES.items():
                a = np.load(self.root / tok_dir_name(mod) / "tokens.npy", mmap_mode="r")
                assert a.shape == (self.n_rows, TOKENS_PER_SAMPLE), a.shape
                self._tokens[src] = a
        return self._tokens

    def _chunk(self, src: str, row: int) -> torch.Tensor:
        info = V.get_modality(V.INPUT_MODALITIES[src])
        raw = self._arrays()[src][row]
        body = torch.from_numpy(np.asarray(raw, dtype=np.int64)) + info.codebook_offset  # int64 FIRST
        return torch.cat([torch.tensor([info.bo_id]), body, torch.tensor([info.eo_id])])

    def chunk(self, src: str, row: int) -> torch.Tensor:
        """`BO body EO` of input `src` ("S1" | "S2") for index row `row`, in global ids."""
        return self._chunk(src, row)

    def classes(self, row: int) -> List[int]:
        return L.from_mask(int(self.labels19[row]))

    def inputs_for(self, pos: int) -> List[str]:
        """The input sources example `pos` gets at the current epoch, in sequence order."""
        row = int(self.rows[pos])
        if self.condition != "random":
            return list(_COND_INPUTS[self.condition])
        rng = np.random.default_rng((self.seed, self.epoch, row))
        cond = CONDITIONS[int(rng.integers(len(CONDITIONS)))]
        srcs = list(_COND_INPUTS[cond])
        if len(srcs) == 2 and rng.integers(2):
            srcs = srcs[::-1]
        return srcs

    # --- the sample --------------------------------------------------------
    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, pos: int) -> Dict[str, torch.Tensor]:
        row = int(self.rows[pos])
        srcs = self.inputs_for(pos)
        chunks = [self._chunk(s, row) for s in srcs]
        block = torch.tensor(V.label_block(self.classes(row)), dtype=torch.long)
        mids = [V.MODALITY_TO_ID[V.INPUT_MODALITIES[s]] for s in srcs] + [V.MODALITY_TO_ID["BEN"]]
        out = assemble_sequence(chunks=chunks + [block], chunk_modality_ids=mids,
                                chunk_shufflable=[False] * len(mids), max_length=self.max_length,
                                pad_id=V.PAD_ID, rng=np.random.default_rng(0),
                                shuffle_image_patches=False)
        if self.loss_on == "labels":
            n_in = sum(int(c.shape[0]) for c in chunks)
            keep = torch.zeros_like(out["labels"], dtype=torch.bool)
            keep[n_in + 1: n_in + int(block.shape[0])] = True          # c_1..c_k, EO_BEN
            out["labels"] = torch.where(keep, out["labels"], torch.full_like(out["labels"], -100))
        return out

    # --- prompts for generation (Step 5) ------------------------------------
    def prompt(self, pos: int, condition: str, input_row: Optional[int] = None) -> torch.Tensor:
        """Inputs in the fixed order, then BO_BEN. `input_row` substitutes another patch's
        inputs (the wrong-input control) while the truth stays this patch's."""
        row = int(self.rows[pos]) if input_row is None else int(input_row)
        chunks = [self._chunk(s, row) for s in _COND_INPUTS[condition]]
        return torch.cat(chunks + [torch.tensor([V.BO_BEN])])
