import os
from pathlib import Path

from lm_dataset.multimodal_tokenized_dataset import MultimodalTokenizedDataset
from paths import PROJECT_ROOT

num_proc = 24


# Root of the pre-tokenized CLEVR dataset (expects <root>/<split>/<modality>/<stem>.{npy,json}).
# Override per-machine via: export CLEVR_ROOT=/path/to/clevr_dataset
CLEVR_ROOT = os.environ.get("CLEVR_ROOT", os.path.join(PROJECT_ROOT, "data", "clevr_dataset"))

COCO_ROOT = os.environ.get("COCO_ROOT", os.path.join(PROJECT_ROOT, "data", "coco_dataset"))

# Root of the Phase-1 tokenized TerraMesh split (contains <MOD>_tok/ and
# tok_index.parquet). Override with: export TERRAMESH_TOK_ROOT=/path/to/val
TERRAMESH_TOK_ROOT = os.environ.get(
    "TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val"
)

MULTIMODAL_DATASETS = {
    "clevr_multimodal": {
        "root_dir": CLEVR_ROOT,
    },
    "coco_multimodal": {
        "root_dir": COCO_ROOT,
    },
    "terramesh_multimodal": {
        "root_dir": TERRAMESH_TOK_ROOT,
    },
}



def _build_terramesh(cfg, mm_cfg, root_dir, want="train"):
    """Construct the EO dataset, applying the held-out split if one is configured.

    `multimodal.eval_split` names the committed split artifact (a path, or a
    bare tag resolved to eo/data/eval_rows_<tag>.json). When it is null the
    dataset is the full 89,088 rows and there is no eval set -- the Phase 2
    behaviour, kept so the smoke path and the CLEVR-comparison runs are
    unchanged.

    ⚠ Train is the COMPLEMENT of eval, computed here rather than stored, so the
    two cannot overlap by construction. A second committed row list would be
    one more thing that can drift.
    """
    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    from eo.data.eval_split import default_split_path, load_eval_rows, train_rows_from_eval

    split_ref = mm_cfg.get("eval_split", None)
    rows = None
    if split_ref is not None:
        path = Path(split_ref)
        if not path.suffix:                      # a bare tag, e.g. "v1"
            path = default_split_path(str(split_ref))
        elif not path.is_absolute():
            path = Path(PROJECT_ROOT) / path
        eval_rows = load_eval_rows(path=path, root_dir=root_dir)
        if want == "eval":
            rows = eval_rows
        else:
            import numpy as np

            n_total = len(np.load(Path(root_dir) / "Coords_tok" / "present.npy", mmap_mode="r"))
            rows = train_rows_from_eval(eval_rows, n_total)
    elif want == "eval":
        return None

    return TerraMeshTokenDataset(
        root_dir=root_dir,
        split=mm_cfg.get("split", "val"),
        active_modalities=mm_cfg.get("active_modalities", None),
        max_length=cfg.max_length,
        modality_order=mm_cfg.get("modality_order", "fixed"),
        seed=cfg.get("seed", 42),
        shuffle_image_patches=mm_cfg.get("shuffle_image_patches", False),
        rows=rows,
    )


def load_eval_dataset_from_config(cfg):
    """The held-out evaluation dataset, or None if this run has no split.

    Separate from load_dataset_from_config rather than returning a pair: the
    CLEVR path has no eval split and every existing caller expects exactly one
    dataset back.
    """
    dataset_name = [ds.strip() for ds in cfg.dataset.split(',')]
    if len(dataset_name) != 1 or dataset_name[0] != "terramesh_multimodal":
        return None
    mm_cfg = cfg.get("multimodal", {})
    return _build_terramesh(
        cfg, mm_cfg, MULTIMODAL_DATASETS["terramesh_multimodal"]["root_dir"], want="eval"
    )


def load_dataset_from_config(cfg):
    dataset_name = [ds.strip() for ds in cfg.dataset.split(',')]

    # Multimodal branch is ours
    if all(ds in MULTIMODAL_DATASETS for ds in dataset_name):
        if len(dataset_name) > 1:
            raise ValueError("Multimodal datasets cannot be combined via comma-separated list.")
        ds_name = dataset_name[0]
        ds_cfg = MULTIMODAL_DATASETS[ds_name]

        mm_cfg = cfg.get("multimodal", {})

        # EO branch: tokenized TerraMesh. Different storage model (one
        # memory-mapped matrix per modality, presence masks, no augmentations,
        # no text), shared sequence assembly.
        if ds_name == "terramesh_multimodal":
            return _build_terramesh(cfg, mm_cfg, ds_cfg["root_dir"], want="train")

        active_modalities = list(mm_cfg.get("active_modalities", ["tok_rgb@256"]))
        modality_order = mm_cfg.get("modality_order", "fixed")
        sample_from_k = mm_cfg.get("sample_from_k_augmentations", 10)
        text_tokenizer_path = mm_cfg.get("text_tokenizer_path", "gpt2")
        text_max_length = mm_cfg.get("text_max_length", 64)
        shuffle_image_patches = mm_cfg.get("shuffle_image_patches", False)

        return MultimodalTokenizedDataset(
            root_dir=ds_cfg["root_dir"],
            split=mm_cfg.get("split", "train"),
            active_modalities=active_modalities,
            max_length=cfg.max_length,
            modality_order=modality_order,
            sample_from_k_augmentations=sample_from_k,
            text_tokenizer_path=text_tokenizer_path,
            text_max_length=text_max_length,
            shuffle_image_patches=shuffle_image_patches,
        )


    raise ValueError(
        f"Unknown dataset(s): {dataset_name}. "
        f"Known: {list(MULTIMODAL_DATASETS.keys())}"
    )