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

Two additions of PHASE6_PLAN §2c (2026-10-18), both off by default, so every earlier run's
examples are unchanged (gate X10b/X11a hold them bit-identical):

* L11b, TEACHER BLOCKS (`teacher=`). The role's rows must carry NO true label (labels19 = -1,
  i.e. ft_unlab); the label block is instead one of the K blocks a teacher sampled for the patch
  (`eo/scripts/ben_teacher.py`), drawn per epoch from an RNG keyed on (seed, epoch, row) -- a
  stream of its own, so it never shifts the condition draw. Sequence-level distillation: the
  pseudo-label block IS the label block, so the batch keeps its five keys. `BenMixDataset` puts
  such examples on top of a labelled set.
* L10, A GENERATED S2 (`gen_s2=`, condition "S1G2"). S2's slot, BO/EO and modality id hold a
  generated S2 body (S1 -> S2 by the extended base, stored like a token dir), UNMARKED: the model
  cannot tell it from a real S2 except by its content. `conditions=` adds it to the random draw.

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
#: L10: S1 with a generated S2 in S2's slot (needs gen_s2); never part of the default draw
GEN_CONDITION = "S1G2"
_COND_INPUTS = {"S1": ["S1"], "S2": ["S2"], "S1S2": ["S2", "S1"],   # fixed order: S2L2A, S1GRD
                GEN_CONDITION: ["G2", "S1"]}                        # "G2": the generated S2, S2's slot
INPUT_WIDTH = TOKENS_PER_SAMPLE + 2
# The absolute bound: both inputs plus a label block carrying all 19 classes.
MAX_LENGTH_BOUND = 2 * INPUT_WIDTH + V.N_CLASSES + 2          # 417
_TEACHER_STREAM = 1                                          # the teacher-block draw's RNG key tag


def load_index(root: str = DEFAULT_ROOT):
    import pandas as pd
    return pd.read_parquet(Path(root) / INDEX_NAME)


def load_teacher(root, name: str):
    """(rows (N,), masks (N, K) int64 19-bit label masks, provenance) of `<root>/<name>.npz/.json`,
    as written by eo/scripts/ben_teacher.py."""
    p = Path(root) / name
    z = np.load(p.with_suffix(".npz"))
    meta = json.loads(p.with_suffix(".json").read_text())
    rows, masks = z["rows"].astype(np.int64), z["masks"].astype(np.int64)
    if masks.ndim != 2 or len(masks) != len(rows) or len(np.unique(rows)) != len(rows):
        raise RuntimeError(f"{p}: masks {masks.shape} for {len(rows)} rows (unique {len(np.unique(rows))})")
    if (masks < 0).any() or (masks >= 1 << V.N_CLASSES).any():
        raise RuntimeError(f"{p}: masks outside 19 bits")
    if int(meta.get("k", -1)) != masks.shape[1] or int(meta.get("n", -1)) != len(rows):
        raise RuntimeError(f"{p}: provenance says k {meta.get('k')} n {meta.get('n')}, file has {masks.shape}")
    return rows, masks, meta


class BenTokenDataset(Dataset):
    def __init__(self, root: str = DEFAULT_ROOT, role: str = "ft_train",
                 condition: str = "random", seed: int = 42,
                 max_length: Optional[int] = MAX_LENGTH_BOUND, loss_on: str = "labels",
                 rows: Optional[Sequence[int]] = None, index=None,
                 conditions: Sequence[str] = CONDITIONS, gen_s2: Optional[str] = None,
                 teacher: Optional[str] = None, prompts_only: bool = False):
        super().__init__()
        conditions = tuple(conditions)
        if not conditions or len(set(conditions)) != len(conditions) \
                or any(c not in _COND_INPUTS for c in conditions):
            raise ValueError(f"conditions must be distinct, from {tuple(_COND_INPUTS)}, got {conditions}")
        if condition != "random" and condition not in _COND_INPUTS:
            raise ValueError(f"condition must be 'random' or one of {tuple(_COND_INPUTS)}")
        if loss_on not in ("labels", "labels_inputs"):
            raise ValueError(f"loss_on must be 'labels' or 'labels_inputs', got {loss_on!r}")
        active = conditions if condition == "random" else (condition,)
        if GEN_CONDITION in active and gen_s2 is None:
            raise ValueError(f"condition {GEN_CONDITION} needs gen_s2 (the generated S2 token dir)")
        self.root = Path(root)
        self.role, self.condition, self.seed, self.loss_on = role, condition, int(seed), loss_on
        self.conditions, self.gen_s2, self.teacher = conditions, gen_s2, teacher
        # prompts_only: a PROMPT source for any rows, labelled or not (ben_teacher.py samples the
        # teacher's blocks on ft_unlab from it); it serves no examples and no labels.
        self.prompts_only = bool(prompts_only)
        if self.prompts_only and teacher is not None:
            raise ValueError("prompts_only serves no labels, so it takes no teacher")
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
        if self.prompts_only:
            masks = np.zeros(1, dtype=np.int64)
        elif teacher is None:
            if (lab < 0).any():
                raise RuntimeError(f"{int((lab < 0).sum())} selected rows were never packed/labelled")
            masks = lab
        else:
            # L11b: distillation rows must carry NO true label, so none can reach the loss.
            if (lab >= 0).any():
                raise RuntimeError(f"teacher blocks on {int((lab >= 0).sum())} LABELLED rows: a distillation "
                                   f"role must be unlabelled (ft_unlab)")
            t_rows, t_masks, self.teacher_meta = load_teacher(self.root, teacher)
            pos_of = np.full(self.n_rows, -1, dtype=np.int64)
            pos_of[t_rows] = np.arange(len(t_rows))
            self._tpos = pos_of[self.rows]
            if (self._tpos < 0).any():
                raise RuntimeError(f"{teacher}: no blocks for {int((self._tpos < 0).sum())} of the role's rows")
            self._tmasks = t_masks
            masks = t_masks[self._tpos].ravel()
        # A patch whose only v1 labels are classes the 19-class grouping drops (e.g. Airports)
        # has an EMPTY label set: 25 of the 243,838 packed rows. SMARTIES keeps them with an
        # all-zero target, so they stay, as the block `BO_BEN EO_BEN`.
        n_lab = np.array([bin(int(m)).count("1") for m in masks])
        self.n_empty = int((n_lab == 0).sum())

        self._present: Dict[str, np.ndarray] = {}
        # both real inputs are required on every role, as before; the generated S2 only when used
        self._dirs = {src: tok_dir_name(mod) for src, mod in V.INPUT_MODALITIES.items()}
        if gen_s2 is not None:
            self._dirs["G2"] = gen_s2
        for src, dname in self._dirs.items():
            mod = V.INPUT_MODALITIES["S2" if src == "G2" else src]
            d = self.root / dname
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
        n_inputs = max(len(_COND_INPUTS[c]) for c in active)
        self.required_length = int(n_inputs * INPUT_WIDTH + n_lab.max() + 2)
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
            for src, dname in self._dirs.items():
                a = np.load(self.root / dname / "tokens.npy", mmap_mode="r")
                assert a.shape == (self.n_rows, TOKENS_PER_SAMPLE), a.shape
                self._tokens[src] = a
        return self._tokens

    def _chunk(self, src: str, row: int) -> torch.Tensor:
        info = V.get_modality(V.INPUT_MODALITIES["S2" if src == "G2" else src])   # G2: S2's own slot
        raw = self._arrays()[src][row]
        body = torch.from_numpy(np.asarray(raw, dtype=np.int64)) + info.codebook_offset  # int64 FIRST
        return torch.cat([torch.tensor([info.bo_id]), body, torch.tensor([info.eo_id])])

    def chunk(self, src: str, row: int) -> torch.Tensor:
        """`BO body EO` of input `src` ("S1" | "S2" | "G2") for index row `row`, in global ids."""
        return self._chunk(src, row)

    def classes(self, row: int) -> List[int]:
        """The patch's TRUE classes. A teacher-block or prompts-only dataset has none, and refuses."""
        if self.teacher is not None or self.prompts_only:
            raise RuntimeError("a teacher-block or prompts-only dataset serves no true labels")
        return L.from_mask(int(self.labels19[row]))

    def teacher_draw(self, pos: int) -> int:
        """Which of the K teacher blocks example `pos` gets at the current epoch."""
        row = int(self.rows[pos])
        rng = np.random.default_rng((self.seed, self.epoch, row, _TEACHER_STREAM))
        return int(rng.integers(self._tmasks.shape[1]))

    def block_classes(self, pos: int) -> List[int]:
        """The classes in example `pos`'s label block: the truth, or this epoch's teacher block."""
        if self.teacher is None:
            return self.classes(int(self.rows[pos]))
        return L.from_mask(int(self._tmasks[self._tpos[pos], self.teacher_draw(pos)]))

    def inputs_for(self, pos: int) -> List[str]:
        """The input sources example `pos` gets at the current epoch, in sequence order."""
        row = int(self.rows[pos])
        if self.condition != "random":
            return list(_COND_INPUTS[self.condition])
        rng = np.random.default_rng((self.seed, self.epoch, row))
        cond = self.conditions[int(rng.integers(len(self.conditions)))]
        srcs = list(_COND_INPUTS[cond])
        if len(srcs) == 2 and rng.integers(2):
            srcs = srcs[::-1]
        return srcs

    # --- the sample --------------------------------------------------------
    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, pos: int) -> Dict[str, torch.Tensor]:
        if self.prompts_only:
            raise RuntimeError("a prompts-only dataset serves no examples")
        row = int(self.rows[pos])
        srcs = self.inputs_for(pos)
        chunks = [self._chunk(s, row) for s in srcs]
        block = torch.tensor(V.label_block(self.block_classes(pos)), dtype=torch.long)
        mids = [V.MODALITY_TO_ID[V.INPUT_MODALITIES["S2" if s == "G2" else s]] for s in srcs] \
            + [V.MODALITY_TO_ID["BEN"]]
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


class BenMixDataset(Dataset):
    """L11b (PHASE6_PLAN §2c): a labelled fine-tuning set with unlabelled distillation examples ON TOP.

    Positions 0 .. n_lab-1 are the labelled dataset's own positions, so its examples (condition
    draw included) are exactly what the labelled-only run sees at the same epoch (gate X10b).
    The next n_unlab = round(ratio * n_lab) positions are teacher-block examples: epoch e takes
    the e-th consecutive slice of a fixed permutation of the unlabelled pool (keyed on the seed
    alone), so no patch repeats until the pool is used up, and then the slices wrap around.
    The HF sampler shuffles all positions together, so a step mixes the two at the ratio in
    expectation. Batch: the closed five-key set, from either source.
    """

    def __init__(self, labelled: BenTokenDataset, unlab: BenTokenDataset, ratio: float = 1.0, seed: int = 42):
        super().__init__()
        if unlab.teacher is None:
            raise ValueError("the unlabelled half must serve teacher blocks")
        if labelled.teacher is not None:
            raise ValueError("the labelled half must serve true labels")
        if labelled.max_length != unlab.max_length:
            raise ValueError(f"max_length differs: {labelled.max_length} vs {unlab.max_length}")
        if np.intersect1d(labelled.rows, unlab.rows).size:
            raise ValueError("a patch is in both halves")
        self.labelled, self.unlab = labelled, unlab
        self.n_lab = len(labelled)
        self.n_unlab = int(round(float(ratio) * self.n_lab))
        if not 0 <= self.n_unlab <= len(unlab):
            raise ValueError(f"ratio {ratio}: {self.n_unlab} unlabelled per epoch, pool {len(unlab)}")
        self.perm = np.random.default_rng((int(seed), 11)).permutation(len(unlab))
        self.max_length = labelled.max_length
        self.required_length = max(labelled.required_length, unlab.required_length)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self.labelled.set_epoch(epoch)
        self.unlab.set_epoch(epoch)

    def unlab_pos(self, j: int) -> int:
        """The unlabelled dataset's position at mixed position n_lab + j, this epoch."""
        return int(self.perm[(self.epoch * self.n_unlab + j) % len(self.unlab)])

    def __len__(self) -> int:
        return self.n_lab + self.n_unlab

    def __getitem__(self, pos: int) -> Dict[str, torch.Tensor]:
        pos = int(pos)
        if pos < self.n_lab:
            return self.labelled[pos]
        return self.unlab[self.unlab_pos(pos - self.n_lab)]
