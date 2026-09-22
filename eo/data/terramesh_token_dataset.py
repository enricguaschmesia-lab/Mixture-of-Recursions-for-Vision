# eo/data/terramesh_token_dataset.py
"""Map-style dataset over the Phase 1 tokenized TerraMesh val split.

Disk layout (Phase 1 Step 4, see docs/PHASE1_REPORT.md section 6):

    <root>/<MOD>_tok<CROP>/tokens.npy   (89088, T) uint16, T=196 images, 3 Coords
    <root>/<MOD>_tok<CROP>/present.npy  (89088,)  bool
    <root>/tok_index.parquet            row, stem, source_shard, corpus

The directory name carries the crop (contract.TOK_DIR_SUFFIX), so a dataset
built under one contract cannot silently read another crop's arrays. Coords is
crop-independent and keeps an unsuffixed 'Coords_tok' -- see contract.
CROP_INDEPENDENT.

Every modality's tokens.npy uses the same canonical row order, so joining
modalities is arr[i] with no lookup. Absent rows are zero-filled and 0 is a
valid token id, so present.npy is not optional.

Differs from the CLEVR MultimodalTokenizedDataset in how a body is obtained --
one memory-mapped matrix per modality rather than one file per sample, no
augmentation dimension, no text branch, plus presence masks. Everything from
BO/EO wrapping onward is shared via lm_dataset.sequence_assembly so the two
paths cannot drift; Phases 3–4 compare EO routing against the CLEVR results.

Runs in the repo's own .venv, NOT the `mor` conda env: it imports numpy, torch
and (lazily) pyarrow, but never terratorch. eo.terramesh_tok.contract is pure
constants and is safe to import from either env.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from eo.terramesh_tok.contract import CROP, GRID, TOKENS_PER_SAMPLE, tok_dir_name
from eo.data.eo_vocab import (
    assert_artifact_fits,
    DEFAULT_ACTIVE_MODALITIES,
    IMAGE_MODALITIES,
    MODALITY_TO_ID,
    PAD_ID,
    TOTAL_VOCAB_SIZE,
    get_modality,
)
from lm_dataset.sequence_assembly import assemble_sequence

DEFAULT_ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")


def assert_artifact_crop(root_dir, modalities) -> Dict[str, object]:
    """Every loaded artifact must have been tokenized at the contract crop.

    PHASE2_PLAN section 4.1 promised this: "make the dataloader ASSERT
    metadata.json's crop against contract.CROP at load. The assert is the real
    safety mechanism; the directory naming is just hygiene."

    Two token sets exist on disk -- <MOD>_tok224 and <MOD>_tok256 -- and training
    against the wrong one is the highest-consequence silent failure in this
    phase. tok_dir_name() already makes it structurally hard by deriving the
    directory from contract.CROP, but that only guards the PATH. This guards the
    CONTENT, so a renamed, copied or hand-placed directory is still caught.

    Crop-independent modalities (Coords) record `"crop": null`, which means
    "applies at every crop" -- NOT a missing field. Treating null as missing
    would make this fire on every run with Coords active.

    Returns {modality: recorded crop (None if crop-independent)}.
    """
    root = Path(root_dir)
    seen: Dict[str, object] = {}
    for name in modalities:
        meta_path = root / tok_dir_name(name) / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if "crop" not in meta:
            raise RuntimeError(
                f"'{name}' artifact records no crop at all ({meta_path}). Every "
                f"artifact must state the crop it was tokenized at, or null if it "
                f"is crop-independent. Re-run its tokenization script."
            )
        crop_block = meta["crop"]
        if crop_block is None:                       # crop-independent (Coords)
            seen[name] = None
            continue
        recorded = crop_block["crop"] if isinstance(crop_block, dict) else crop_block
        seen[name] = recorded
        if int(recorded) != CROP:
            raise RuntimeError(
                f"'{name}' was tokenized at crop {recorded}, but the contract is "
                f"{CROP} ({meta_path}).\n"
                f"  Training on a mismatched crop is silent: the token ids are all "
                f"in range and the sequence assembles fine, it is simply a different "
                f"scene geometry than every other modality in the batch.\n"
                f"  Either point TERRAMESH_TOK_ROOT at the {CROP} artifacts or "
                f"re-tokenize (eo/scripts/tokenize_terramesh.py)."
            )
        # grid/token-count must agree too, so a hand-edited crop field is caught
        tps = meta.get("tokens_per_sample")
        if tps is not None and int(tps) != TOKENS_PER_SAMPLE:
            raise RuntimeError(
                f"'{name}' records crop {recorded} but {tps} tokens/sample; the "
                f"contract's {CROP} crop gives {TOKENS_PER_SAMPLE} "
                f"({GRID}x{GRID}). The metadata is internally inconsistent."
            )
    return seen


class TerraMeshTokenDataset(Dataset):
    """One sample per TerraMesh scene: the present modalities, BO/EO-wrapped,
    concatenated into one sequence over the EO vocabulary (eo_vocab, D2.2).

    Pass `rows=` to restrict it to a subset of the canonical row order -- that
    is how the Phase 3 train/eval split is applied (eo/data/eval_split.py,
    D3.1). A scene's emitted sequence does not depend on which subset reached
    it; see _get_rng."""

    def __init__(
        self,
        root_dir: str = DEFAULT_ROOT,
        split: str = "val",
        active_modalities: Optional[Sequence[str]] = None,
        max_length: Optional[int] = None,
        modality_order: str = "fixed",
        seed: int = 42,
        shuffle_image_patches: bool = False,
        rows: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if modality_order not in ("fixed", "random"):
            raise ValueError(f"modality_order must be 'fixed' or 'random', got {modality_order}")

        # split is carried for symmetry with the CLEVR dataset and for when a
        # train split appears (worklog open item 5). Only 'val' exists today,
        # and the tokenized arrays sit directly under root_dir.
        self.root_dir = Path(root_dir)
        self.split = split
        # Defaults to the ratified training set -- all six image modalities PLUS
        # Coords. It used to default to IMAGE_MODALITIES, which excludes Coords,
        # so the dataset silently produced image-only sequences (990 tokens
        # instead of 995) even though Meeting 2 put Coords in the training set.
        self.active_modalities = (list(active_modalities) if active_modalities
                                  else list(DEFAULT_ACTIVE_MODALITIES))
        self.modality_order = modality_order
        self.seed = seed
        self.shuffle_image_patches = shuffle_image_patches

        if not self.active_modalities:
            raise ValueError("active_modalities must be non-empty.")
        for m in self.active_modalities:
            get_modality(m)  # validates against the EO registry

        # Does the DATA fit the vocabulary? eo_vocab's import-time invariants
        # prove the layout is self-consistent, but not that the artifact's ids
        # fit inside their slots -- and an id one past the end lands on a BO/EO
        # marker, so it corrupts silently instead of raising. O(1): reads the
        # max recorded by the run that wrote the tokens, not the arrays.
        assert_artifact_fits(self.root_dir, self.active_modalities)

        # Does the data match the CONTRACT? Separate concern from the slot check
        # above, and kept a separate function for that reason -- one name doing
        # two jobs is what let Coords go missing from the sequence (Step 4).
        assert_artifact_crop(self.root_dir, self.active_modalities)

        # Presence masks are tiny (89 KB each) and are needed up front to size
        # the sequence. Token matrices are memory-mapped lazily, per worker.
        self._present: Dict[str, np.ndarray] = {}
        n_rows = None
        for m in self.active_modalities:
            p = np.load(self._mod_dir(m) / "present.npy")
            if n_rows is None:
                n_rows = p.shape[0]
            elif p.shape[0] != n_rows:
                raise RuntimeError(
                    f"Row-count mismatch: '{m}' has {p.shape[0]} rows, expected {n_rows}. "
                    f"All modalities must share the tok_index.parquet row order."
                )
            self._present[m] = p
        self.n_rows = int(n_rows)

        rows_with_nothing = int((~np.stack([self._present[m] for m in self.active_modalities]).any(0)).sum())
        if rows_with_nothing:
            raise RuntimeError(
                f"{rows_with_nothing} rows have none of {self.active_modalities} present; "
                f"they would produce an empty sequence."
            )

        # The row subset (Phase 3 Step 1.4, D3.1). `rows` holds ORIGINAL row ids
        # into the canonical tok_index.parquet order; the dataset is indexed
        # positionally over it, and every array access and every RNG draw goes
        # through the original id.
        #
        # ⚠ Why an explicit argument rather than torch.utils.data.Subset:
        # _get_rng seeds on (seed, worker_id, idx). Under a Subset, `idx` is the
        # POSITION in the subset, so scene 12,345 would draw a different modality
        # order depending on which split it landed in -- silently breaking
        # reproducibility against the full-set runs and making train and eval
        # disagree about the same scene. Here `idx` stays the original row id, so
        # a scene's RNG stream is a property of the scene.
        if rows is None:
            self.rows = np.arange(self.n_rows, dtype=np.int64)
            self.is_subset = False
        else:
            self.rows = np.asarray(rows, dtype=np.int64)
            self.is_subset = True
            if self.rows.ndim != 1 or self.rows.size == 0:
                raise ValueError("rows must be a non-empty 1-D sequence of row ids.")
            if len(np.unique(self.rows)) != len(self.rows):
                raise ValueError("rows contains duplicate row ids.")
            if self.rows.min() < 0 or self.rows.max() >= self.n_rows:
                raise ValueError(
                    f"rows fall outside [0, {self.n_rows}); they do not index this artifact. "
                    f"Got [{self.rows.min()}, {self.rows.max()}]."
                )

        # Exact per-row sequence lengths: only present modalities contribute,
        # each as [BO, body, EO]. Computed over the SELECTED rows -- a subset
        # cannot need more room than the full set, but it can need less, and
        # max_length is validated against what this dataset will actually emit.
        widths = np.array(
            [get_modality(m).tokens_per_sample + 2 for m in self.active_modalities], dtype=np.int64
        )
        presence = np.stack([self._present[m] for m in self.active_modalities], axis=1)
        self._seq_lens = (presence.astype(np.int64) * widths[None, :]).sum(axis=1)
        self.required_length = int(self._seq_lens[self.rows].max())

        if max_length is None:
            self.max_length = self.required_length
        else:
            if max_length < self.required_length:
                raise ValueError(
                    f"max_length={max_length} would truncate: the longest sequence over "
                    f"{self.active_modalities} needs {self.required_length} tokens. "
                    f"Truncation would silently drop a whole modality -- raise max_length instead."
                )
            self.max_length = int(max_length)

        self._tokens: Optional[Dict[str, np.ndarray]] = None
        self._stems: Optional[List[str]] = None

    # -- paths and lazy handles ------------------------------------------
    def _mod_dir(self, modality: str) -> Path:
        return self.root_dir / tok_dir_name(modality)

    def _token_arrays(self) -> Dict[str, np.ndarray]:
        """Memory-map on first use inside each worker, not in __init__."""
        if self._tokens is None:
            self._tokens = {
                m: np.load(self._mod_dir(m) / "tokens.npy", mmap_mode="r")
                for m in self.active_modalities
            }
            for m, arr in self._tokens.items():
                expect = get_modality(m).tokens_per_sample
                if arr.shape != (self.n_rows, expect):
                    raise RuntimeError(
                        f"'{m}' tokens.npy is {arr.shape}, expected {(self.n_rows, expect)}"
                    )
        return self._tokens

    @property
    def stems(self) -> List[str]:
        """Sample stems for the SELECTED rows, in this dataset's positional
        order. Provenance only -- never part of a batch (the output dict is a
        closed five-key set). Identical to canonical row order when no subset
        is in force."""
        if self._stems is None:
            import pandas as pd

            idx = pd.read_parquet(self.root_dir / "tok_index.parquet")
            if len(idx) != self.n_rows:
                raise RuntimeError(f"tok_index.parquet has {len(idx)} rows, arrays have {self.n_rows}")
            all_stems = idx.sort_values("row")["stem"].tolist()
            self._stems = [all_stems[r] for r in self.rows]
        return self._stems

    def __len__(self) -> int:
        return int(len(self.rows))

    def present_modalities(self, row: int) -> List[str]:
        """Takes an ORIGINAL row id, not a position in this dataset."""
        return [m for m in self.active_modalities if bool(self._present[m][row])]

    def _get_rng(self, row: int) -> np.random.Generator:
        """Per-sample RNG -- deterministic given (seed, ORIGINAL row id). Workers
        don't collide. Same construction as the CLEVR dataset.

        ⚠ Keyed on the original row id, never on the position in a subset, so a
        scene draws the same modality order whether it is reached through the
        full set, the train split or the eval split."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        return np.random.default_rng(seed=(self.seed, worker_id, row))

    def _load_chunk(self, modality: str, row: int) -> torch.Tensor:
        info = get_modality(modality)
        raw = self._token_arrays()[modality][row]
        # uint16 -> int64 BEFORE the offset: 87,555 > 65,535, so adding the
        # offset in uint16 would wrap silently.
        body = torch.from_numpy(np.asarray(raw, dtype=np.int64)) + info.codebook_offset
        bo = torch.tensor([info.bo_id], dtype=torch.long)
        eo = torch.tensor([info.eo_id], dtype=torch.long)
        return torch.cat([bo, body, eo], dim=0)

    def __getitem__(self, pos: int) -> Dict[str, torch.Tensor]:
        # `pos` is a position in this dataset; `row` is the id in the canonical
        # tok_index.parquet order. They coincide only when no subset is in force.
        row = int(self.rows[pos])
        rng = self._get_rng(row)

        # Only modalities actually present in this row take a slot. An absent
        # modality is omitted entirely rather than emitted as an empty BO/EO
        # pair or a zero-filled body: 0 is a real token id, and a zero body
        # would train the model on content that does not exist.
        ordered = self.present_modalities(row)
        if self.modality_order == "random" and len(ordered) > 1:
            perm = rng.permutation(len(ordered))
            ordered = [ordered[i] for i in perm]

        chunks = [self._load_chunk(m, row) for m in ordered]

        total = sum(int(c.shape[0]) for c in chunks)
        if total > self.max_length:
            raise RuntimeError(
                f"row {row}: sequence of {total} exceeds max_length={self.max_length}. "
                f"This should have been caught in __init__."
            )

        return assemble_sequence(
            chunks=chunks,
            chunk_modality_ids=[MODALITY_TO_ID[m] for m in ordered],
            chunk_shufflable=[m in IMAGE_MODALITIES for m in ordered],
            max_length=self.max_length,
            pad_id=PAD_ID,
            rng=rng,
            shuffle_image_patches=self.shuffle_image_patches,
        )
