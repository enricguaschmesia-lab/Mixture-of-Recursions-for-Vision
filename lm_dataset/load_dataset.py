from lm_dataset.multimodal_tokenized_dataset import MultimodalTokenizedDataset

num_proc = 24


# Multimodal datasets (CLEVR: per-modality tokenized .npy / .json files)
MULTIMODAL_DATASETS = {
    "clevr_multimodal": {
        "root_dir": "/home/gianfranco/projects/2025/Visual_Intelligence_Project/Dataset/clevr_com_304",
    },
}


def load_dataset_from_config(cfg):
    dataset_name = [ds.strip() for ds in cfg.dataset.split(',')]

    # Multimodal branch is ours
    if all(ds in MULTIMODAL_DATASETS for ds in dataset_name):
        if len(dataset_name) > 1:
            raise ValueError("Multimodal datasets cannot be combined via comma-separated list.")
        ds_name = dataset_name[0]
        ds_cfg = MULTIMODAL_DATASETS[ds_name]

        mm_cfg = cfg.get("multimodal", {})
        active_modalities = list(mm_cfg.get("active_modalities", ["tok_rgb@256"]))
        modality_order = mm_cfg.get("modality_order", "fixed")
        sample_from_k = mm_cfg.get("sample_from_k_augmentations", 10)
        text_tokenizer_path = mm_cfg.get("text_tokenizer_path", "gpt2")
        text_max_length = mm_cfg.get("text_max_length", 64)

        return MultimodalTokenizedDataset(
            root_dir=ds_cfg["root_dir"],
            split=mm_cfg.get("split", "train"),
            active_modalities=active_modalities,
            max_length=cfg.max_length,
            modality_order=modality_order,
            sample_from_k_augmentations=sample_from_k,
            text_tokenizer_path=text_tokenizer_path,
            text_max_length=text_max_length,
        )


    raise ValueError(
        f"Unknown dataset(s): {dataset_name}. "
        f"Known: {list(MULTIMODAL_DATASETS.keys())}"
    )