import os
import json
import warnings

import torch
from datasets import load_dataset, interleave_datasets

from lm_dataset.language_modeling_dataset import LanguageModelingDataset
from lm_dataset.tokenized_dataset import TokenizedCorpusDataset
from lm_dataset.data_preprocessing import AddLabels, RemoveIndex
from lm_dataset.multimodal_tokenized_dataset import MultimodalTokenizedDataset
from paths import DATA_DIR,MULTIMODAL_DATA_DIR

num_proc = 24

# Streaming language-modeling datasets (HF load_dataset)
LM_DATASETS = {
    "slimpajama": {"path": f"{DATA_DIR}/slimpajama", "split": "train"},
    "slimpajama_chunk1": {"path": "json", "data_files": f"{DATA_DIR}/slimpajama_chunk1/*.jsonl", "split": "train"},
    "cosmopedia": {"path": f"{DATA_DIR}/cosmopedia-v2", "split": "train"},
    "fineweb_edu": {"path": f"{DATA_DIR}/fineweb-edu-dedup", "split": "train"},
    "fineweb_test": {"path": f"{DATA_DIR}/fineweb-test", "split": "train"},
    "python_edu": {"path": f"{DATA_DIR}/python-edu", "split": "train"},
    "open_web_math": {"path": f"{DATA_DIR}/open-web-math", "split": "train"},
    "math_code_pile": {"path": f"{DATA_DIR}/math-code-pile", "split": "train"},
    "starcoderdata": {"path": f"{DATA_DIR}/starcoderdata", "split": "train"},
    "finemath": {"path": f"{DATA_DIR}/finemath", "split": "train"},
}

TOKENIZED_DATASETS = {
    "pythia_pile": "pythia",
}

# Multimodal datasets (CLEVR: per-modality tokenized .npy / .json files)
MULTIMODAL_DATASETS = {
    "clevr_multimodal": {
        "root_dir": "/home/gianfranco/projects/2025/Visual_Intelligence_Project/Dataset/clevr_com_304",
    },
}


def load_dataset_from_config(cfg, tokenizer):
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

    # Streaming LM branch
    if all(ds in LM_DATASETS for ds in dataset_name):
        if len(dataset_name) > 1:
            assert "weights" in cfg, "When combining datasets, weights must be provided"
            assert len(dataset_name) == len(cfg.weights.split(',')), \
                "Number of weights must match number of datasets"

        train_dataset = []
        for ds in dataset_name:
            _dataset = load_dataset(**LM_DATASETS[ds], streaming=True)
            if ds == "starcoderdata":
                _dataset.rename_column("content", "text")
            train_dataset.append(_dataset)

        if len(train_dataset) == 1:
            train_dataset = train_dataset[0]
        else:
            train_dataset = interleave_datasets(
                train_dataset, probabilities=cfg.weights.split(','), seed=42
            )

        transforms = [AddLabels(), RemoveIndex()]
        return LanguageModelingDataset(
            train_dataset, tokenizer,
            max_length=cfg.max_length,
            transforms=transforms,
            global_shuffling=cfg.get("global_shuffling", False),
            local_shuffling=cfg.get("local_shuffling", False),
            add_bos_token=cfg.get("add_bos_token", False),
        )

    # Pre-tokenized corpus branch
    if all(ds in TOKENIZED_DATASETS for ds in dataset_name):
        if "tokenizer" in cfg:
            tokenizer_used = TOKENIZED_DATASETS[cfg.dataset]
            if cfg.tokenizer != tokenizer_used:
                raise ValueError(f"Tokenizer {cfg.tokenizer} is not compatible with dataset {cfg.dataset}")

        if cfg.dataset == "pythia_pile":
            from lm_dataset.tokenized_dataset import PythiaPileTokenizedCorpus
            corpus = PythiaPileTokenizedCorpus(os.path.join(DATA_DIR, "pythia_pile_idxmaps"))
        else:
            raise ValueError(f"Unknown tokenized dataset: {cfg.dataset}")

        if cfg.dataloader_num_workers <= 1:
            warnings.warn(
                f"Using cfg.dataloader_num_workers={cfg.dataloader_num_workers} with TokenizedCorpusDataset. "
                f"You may want to increase this number to speed up data loading."
            )
        transforms = [AddLabels(), RemoveIndex()]
        return TokenizedCorpusDataset(
            corpus, length=cfg.max_length, eos_token=tokenizer.eos_token_id,
            add_bos_token=cfg.get("add_bos_token", False),
            bos_token=tokenizer.bos_token_id, transforms=transforms,
        )

    raise ValueError(
        f"Unknown dataset(s): {dataset_name}. "
        f"Known: {list(LM_DATASETS.keys()) + list(TOKENIZED_DATASETS.keys()) + list(MULTIMODAL_DATASETS.keys())}"
    )