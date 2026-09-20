"""Resolve which vocabulary registry a run uses, from its config.

WHY THIS EXISTS. There are two vocabulary registries in this repo and they are
NOT interchangeable:

    lm_dataset/multimodal_vocab_shared_caption_scene_desc.py   CLEVR / COCO
    eo/mor_data/eo_vocab.py                                    TerraMesh (EO)

They expose the same names (`MODALITY_TO_ID`, `ID_TO_MODALITY`, `get_modality`,
`TOTAL_VOCAB_SIZE`, `PAD_ID`) over completely different id spaces. Before this
module existed, `util/trainer_pt.py` imported `ID_TO_MODALITY` from the CLEVR
registry at *module scope*, so an EO run resolved its modality ids against
CLEVR's names. That does not crash — it silently mislabels, and partly drops:

    EO id 1 (S2L2A)  -> logged as 'tok_rgb@256'
    EO id 2 (S1GRD)  -> logged as 'tok_depth@256'
    EO id 3 (S1RTC)  -> logged as 'tok_normal@256'
    EO id 4 (DEM)    -> logged as 'caption'
    EO id 5 (NDVI)   -> logged as 'scene_desc'
    EO id 6 (LULC)   -> absent from CLEVR's dict: NEVER LOGGED AT ALL
    EO id 7 (Coords) -> absent from CLEVR's dict: NEVER LOGGED AT ALL

Per-modality losses feed Phase 4's routing analysis directly, so this is a
correctness bug in the measurement, not a cosmetic one. See docs worklog §4
item 15 and PHASE2_PLAN.md step 5.2.

The registry is a property of the DATASET (what the tokens were written
against), not of the model, so it is selected from `cfg.dataset`.

Imports are lazy and per-branch on purpose: a CLEVR run must not import the EO
registry and vice versa. `verify_step7.py` V0 holds CLEVR's dataset output
bit-identical, and the cheapest way to keep that true is to not touch the CLEVR
import path at all.
"""
from __future__ import annotations

from types import ModuleType
from typing import Dict, List, Optional

# Which registry each multimodal dataset's tokens were written against.
# clevr/coco both go through MultimodalTokenizedDataset, which imports the
# CLEVR registry directly (multimodal_tokenized_dataset.py:22).
_REGISTRY_BY_DATASET: Dict[str, str] = {
    "clevr_multimodal": "clevr",
    "coco_multimodal": "clevr",
    "terramesh_multimodal": "eo",
}

CLEVR_REGISTRY = "lm_dataset.multimodal_vocab_shared_caption_scene_desc"
EO_REGISTRY = "eo.mor_data.eo_vocab"


def dataset_names(cfg) -> List[str]:
    """The configured dataset names, comma-separated list allowed."""
    raw = cfg.get("dataset") if hasattr(cfg, "get") else getattr(cfg, "dataset", None)
    if raw is None:
        return []
    return [d.strip() for d in str(raw).split(",") if d.strip()]


def registry_kind(cfg) -> Optional[str]:
    """'clevr', 'eo', or None when this is not a multimodal run.

    Raises if a run somehow mixes datasets belonging to different registries --
    there is no single id space in that case, so every downstream consumer would
    be wrong in a way it could not detect.
    """
    kinds = {_REGISTRY_BY_DATASET[d] for d in dataset_names(cfg)
             if d in _REGISTRY_BY_DATASET}
    if not kinds:
        return None
    if len(kinds) > 1:
        raise ValueError(
            f"Datasets {dataset_names(cfg)} span more than one vocabulary registry "
            f"({sorted(kinds)}). Their token ids mean different things and cannot "
            f"share one embedding table."
        )
    return kinds.pop()


def is_eo(cfg) -> bool:
    return registry_kind(cfg) == "eo"


def get_registry(cfg) -> ModuleType:
    """The vocabulary registry module this run's tokens were written against."""
    import importlib

    kind = registry_kind(cfg)
    if kind is None:
        raise ValueError(
            f"No multimodal vocabulary registry applies to dataset(s) "
            f"{dataset_names(cfg)}. Known: {sorted(_REGISTRY_BY_DATASET)}."
        )
    return importlib.import_module(EO_REGISTRY if kind == "eo" else CLEVR_REGISTRY)


def get_id_to_modality(cfg) -> Dict[int, str]:
    """{modality_id: name} for per-modality loss logging.

    Empty dict for a non-multimodal run, so callers can treat "no per-modality
    breakdown" uniformly rather than special-casing.
    """
    if registry_kind(cfg) is None:
        return {}
    return dict(get_registry(cfg).ID_TO_MODALITY)


def assert_vocab_size(cfg) -> Optional[int]:
    """Fail at startup if `model_config.vocab_size` disagrees with the registry.

    Nothing in the repo derived or checked this: `vocab_size` is a hand-written
    literal in the YAML. Getting it wrong does not reliably crash --

      * too SMALL -> real token ids index past the embedding table. That one at
        least raises, though far from the cause.
      * too LARGE -> the extra rows are simply never reached. Training runs,
        converges, and silently carries dead parameters; the loss baseline
        ln(V) no longer matches the vocabulary actually in use.

    Returns the expected size, or None when the check does not apply (not a
    multimodal run, or no explicit model_config.vocab_size -- e.g. the
    use_pretrained_weights path, which takes vocab_size from the checkpoint).
    """
    if registry_kind(cfg) is None:
        return None
    model_config = cfg.get("model_config") if hasattr(cfg, "get") else None
    if not model_config or "vocab_size" not in model_config:
        return None

    expected = int(get_registry(cfg).TOTAL_VOCAB_SIZE)
    configured = int(model_config["vocab_size"])
    if configured != expected:
        raise ValueError(
            f"model_config.vocab_size={configured} disagrees with the "
            f"{registry_kind(cfg)} registry's TOTAL_VOCAB_SIZE={expected} "
            f"(dataset: {', '.join(dataset_names(cfg))}).\n"
            f"  Fix the config rather than the registry: the registry is derived "
            f"from the tokenizer codebooks and is the single source of truth.\n"
            f"  too small -> token ids index past the embedding table; "
            f"too large -> silently dead embedding rows and a wrong ln(V) baseline."
        )
    return expected
