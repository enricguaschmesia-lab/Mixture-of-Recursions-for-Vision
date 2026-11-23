# eo/data/seg_head_dataset.py
"""A2 Step 4.2's Route C examples: the view's S2 block written twice, and the view's native label map (`.venv`).

    input_ids      [BO_S2L2A s1..s196 EO_S2L2A][BO_S2L2A s1..s196 EO_S2L2A]   396 tokens, the same block twice
    position_ids   0..395: the second copy at the base's 198-token lattice offset (temporal_dataset's lattice with no
                   date block is sequential)
    modality_ids   S2L2A's on the image tokens, 0 on every BO/EO (assemble_sequence's convention)
    attention_mask ones
    seg_labels     (native, native) int64: the view's label crop and dihedral variant (eo/maps/cells.view_labels, cut as
                   Step 2 cut the map tokens) from eo/maps/sets' train or val set; PASTIS 128 x 128 with void 19,
                   CloudSen12 224 x 224
No `labels`: the head's own loss reads `seg_labels` (eo/finetune/seg_head.py). Everything else is SegDataset's: the views,
the per-(seed, epoch, sample) view draw, the epoch order and its callback (`orders_itself`), the wrong-input derangement.
SegDataset is built with the config's map arm (aligned), whose artifact indexes the views; no map token is read.

Variants: `full`, and `wrong` (another sample's S2 under SegDataset's fixed derangement, the own view's labels: the
wrong-input control). A test split is refused (SegSpec.validate), and so are PASTIS-7 frames (one input block only).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from eo.data.seg_dataset import INPUT, SegDataset, SegSpec
from eo.terramesh_tok.contract import TOKENS_PER_SAMPLE

BLOCK = TOKENS_PER_SAMPLE + 2              # 198
LENGTH = 2 * BLOCK                         # 396


class SegHeadDataset(SegDataset):
    def __init__(self, spec: SegSpec, max_length: Optional[int] = None, seed: int = 42, batch: int = 1,
                 allow_test: bool = False):
        if spec.frames == "p7":
            raise ValueError("Route C reads one input block (PASTIS-1 or CloudSen12), not PASTIS-7")
        if spec.variant not in ("full", "wrong"):
            raise ValueError(f"Route C's variants are full and wrong, not {spec.variant!r}")
        super().__init__(spec, max_length=None, seed=seed, batch=batch, allow_test=allow_test)
        if max_length is not None and int(max_length) != LENGTH:
            raise ValueError(f"Route C's sequence is {LENGTH} tokens (the S2 block twice), not max_length {max_length}")
        self.max_length = LENGTH
        from eo.maps import cells, sets
        self.label_set, ids = sets.load(spec.task, spec.split)
        rows = cells.set_rows(spec.task, ids, self.views.sample_id.tolist())
        self._label_row = dict(zip(self.views.view.values.tolist(), rows.tolist()))
        self._geom = {int(r.view): (int(r.y0), int(r.x0), int(r.size), int(r.k)) for r in self.views.itertuples()}
        mid = self.V.MODALITY_TO_ID[INPUT]
        one = torch.tensor([0] + [mid] * TOKENS_PER_SAMPLE + [0], dtype=torch.long)
        self._modality_ids = torch.cat([one, one])

    def view_label(self, view: int) -> np.ndarray:
        from eo.maps import cells
        y0, x0, size, k = self._geom[int(view)]
        return cells.view_labels(self.label_set, [self._label_row[int(view)]], [y0], [x0], [size], [k])[0]

    def head_item(self, view: int, variant: Optional[str] = None) -> Dict[str, torch.Tensor]:
        variant = self.spec.variant if variant is None else variant
        if variant not in ("full", "wrong"):
            raise ValueError(f"Route C's variants are full and wrong, not {variant!r}")
        src = self.derangement()[int(view)] if variant == "wrong" else int(view)
        s2 = self._s2_chunk(int(self.in_rows[src][0]))
        ids = torch.cat([s2, s2])
        return {"input_ids": ids, "attention_mask": torch.ones(LENGTH, dtype=torch.long),
                "position_ids": torch.arange(LENGTH, dtype=torch.long), "modality_ids": self._modality_ids.clone(),
                "seg_labels": torch.from_numpy(self.view_label(int(view)).astype(np.int64))}

    def sequence(self, view: int, variant: Optional[str] = None):
        raise RuntimeError("Route C's examples are head_item(); a map sequence has no meaning here")

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.head_item(self.view_at(i))
