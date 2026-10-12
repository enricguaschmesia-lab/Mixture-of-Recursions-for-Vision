"""BigEarthNet's 19-class nomenclature, and the grouping from v1's 43 CORINE labels.

Copied from SMARTIES' released code, not from a paper: `eval_datasets.BigEarthNet.NEW_LABELS`
and `.GROUP_LABELS` at commit fc0df04415384f1cfce97f8db3c2cc320bf87d52 (the commit that pins
the eval splits). CLAUDE.md rule 3: released code over paper text. The order of NEW_LABELS is
the CANONICAL order: it is the column order of SMARTIES' multi-hot target, the order the label
block is written in, and the order of the BEN class ids in `eo/data/ben_vocab.py`.

SMARTIES' mapping (`filename_to_class_multi_hot`) does three things per v1 label:
  * in GROUP_LABELS          -> the grouped class;
  * already a NEW_LABELS name -> itself;
  * anything else             -> DROPPED (e.g. 'Airports', 'Road and rail networks',
                                 'Bare rock', 'Burnt areas', 'Intertidal flats').
`to_19` reproduces exactly that, including the silent drop, and counts the drops so they
are visible rather than lost.

SMARTIES reads the labels from the **S2** patch's `_labels_metadata.json`
(`BigEarthNetMM.__getitem__` takes `target` from `s2_ds`). The S1 patch carries its own copy;
`pack.py` records both and `splits.py` checks that they agree.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

SMARTIES_SHA = "fc0df04415384f1cfce97f8db3c2cc320bf87d52"

NEW_LABELS: List[str] = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters',
]

GROUP_LABELS: Dict[str, str] = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters',
}

N_CLASSES = len(NEW_LABELS)
assert N_CLASSES == 19, "the BEN-19 nomenclature has 19 classes"
assert len(set(NEW_LABELS)) == N_CLASSES
assert set(GROUP_LABELS.values()) <= set(NEW_LABELS), "a group target is not a 19-class name"
assert not set(GROUP_LABELS) & set(NEW_LABELS), "a 43-class name is also a 19-class name"

# Short names for tables and figure axes only. Never used as keys.
SHORT: List[str] = [
    'Urban', 'Industrial', 'Arable', 'Perm. crops', 'Pastures', 'Complex cult.',
    'Agri+natural', 'Agro-forestry', 'Broad-leaved', 'Coniferous', 'Mixed forest',
    'Grassland', 'Moors/heath', 'Trans. woodland', 'Beaches', 'Inland wetl.',
    'Coastal wetl.', 'Inland waters', 'Marine waters',
]
assert len(SHORT) == N_CLASSES


def to_19(labels43: Iterable[str]) -> Tuple[List[int], List[str]]:
    """v1 labels -> (sorted 19-class indices, dropped labels). SMARTIES' rule, exactly."""
    out, dropped = set(), []
    for lab in labels43:
        if lab in GROUP_LABELS:
            out.add(NEW_LABELS.index(GROUP_LABELS[lab]))
        elif lab in NEW_LABELS:
            out.add(NEW_LABELS.index(lab))
        else:
            dropped.append(lab)
    return sorted(out), dropped


def to_mask(indices: Iterable[int]) -> int:
    m = 0
    for i in indices:
        m |= 1 << int(i)
    return m


def from_mask(mask: int) -> List[int]:
    return [i for i in range(N_CLASSES) if (int(mask) >> i) & 1]
