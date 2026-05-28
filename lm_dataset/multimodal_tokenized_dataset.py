# lm_dataset/multimodal_tokenized_dataset.py
"""
Map-style dataset for multimodal tokenized data (CLEVR-style layout).

Expected disk layout:
    {root_dir}/{split}/{modality_name}/{stem}.{ext}

For 'tokens' modalities: file contains np.ndarray of shape (K, N_tokens) where
K = number of augmentations.

For 'text' modalities: file contains a JSON list of K strings.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from lm_dataset.multimodal_vocab_shared_caption_scene_desc import MODALITIES, MODALITY_TO_ID, PAD_ID, get_modality


class MultimodalTokenizedDataset(Dataset):
    """
    One sample per image, built by concatenating BO/EO-wrapped chunks from
    one or more pre-tokenized modalities.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        active_modalities: List[str],
        max_length: int,
        modality_order: str = 'fixed',  # 'fixed' or 'random'
        sample_from_k_augmentations: int = 10,
        text_tokenizer_path: str = 'gpt2',
        text_max_length: int = 64,
        seed: int = 42,
        shuffle_image_patches: bool = False,
    ):
        super().__init__()
        if modality_order not in ('fixed', 'random'):
            raise ValueError(f"modality_order must be 'fixed' or 'random', got {modality_order}")

        self.root_dir = root_dir
        self.split = split
        self.active_modalities = list(active_modalities)
        self.max_length = max_length
        self.modality_order = modality_order
        self.sample_from_k_augmentations = sample_from_k_augmentations
        self.text_tokenizer_path = text_tokenizer_path
        self.text_max_length = text_max_length
        self.seed = seed
        self.shuffle_image_patches = shuffle_image_patches
        self._augmentation_count_cache: Dict[str, int] = {}

        for m in self.active_modalities:
            get_modality(m)   # validates
        if len(self.active_modalities) == 0:
            raise ValueError("active_modalities must be non-empty.")
        if self.sample_from_k_augmentations <= 0:
            raise ValueError(
                "sample_from_k_augmentations must be positive, "
                f"got {self.sample_from_k_augmentations}"
            )

        # Use the first active modality to enumerate sample stems, matching
        # the convention in SimpleMultimodalDataset.
        anchor_dir = Path(root_dir) / split / self.active_modalities[0]
        if not anchor_dir.exists():
            raise FileNotFoundError(f"Modality directory not found: {anchor_dir}")
        self.file_stems = sorted(p.stem for p in anchor_dir.iterdir() if p.is_file())
        if len(self.file_stems) == 0:
            raise RuntimeError(f"No files found in {anchor_dir}")

        # We only instantiate if we have a text modality.
        self._needs_text_tokenizer = any(
            get_modality(m).data_type == 'text' for m in self.active_modalities
        )
        self._text_tokenizer = None   

    def __len__(self) -> int:
        return len(self.file_stems)
    
    
    def _get_text_tokenizer(self):
        if self._text_tokenizer is None:
            tok = AutoTokenizer.from_pretrained(self.text_tokenizer_path)
            tok.add_special_tokens({'pad_token': '[PAD]'})
            tok.add_special_tokens({'bos_token': '[SOS]', 'eos_token': '[EOS]'})
            # Design choice: text is wrapped with general [SOS]/[EOS] tokens insidethe modality-level <BO_text> ... <EO_text> wrapper. While we have also BO and EO for each modality.
            # In this case it will bound the text modality with SOS and EOS.
            tok._tokenizer.post_processor = TemplateProcessing(
                single="[SOS] $A [EOS]",
                special_tokens=[('[EOS]', tok.eos_token_id), ('[SOS]', tok.bos_token_id)],
            )

            actual_len = len(tok)
            for mod_name, info in MODALITIES.items():
                if info.data_type == 'text':
                    assert actual_len <= info.codebook_size, (
                        f"Tokenizer {self.text_tokenizer_path} has {actual_len} tokens, "
                        f"but modality '{mod_name}' declares codebook_size={info.codebook_size}. "
                        f"This will cause ID collisions. Fix the registry or use a different tokenizer."
                    )
            self._text_tokenizer = tok
        return self._text_tokenizer

    def _load_tokens_modality(self, modality: str, stem: str, aug_idx: int) -> torch.Tensor:
        info = get_modality(modality)
        path = Path(self.root_dir) / self.split / modality / f"{stem}{info.file_ext}"
        
        # sanity check
        if not path.exists():
            raise FileNotFoundError(
                f"Missing token file for modality='{modality}', split='{self.split}', "
                f"stem='{stem}': {path}"
            )

        arr = np.load(path) # shape (K, N)

        # sanity check
        if arr.ndim == 1:
            if aug_idx != 0:
                raise IndexError(
                    f"File {path} has no augmentation dimension, but aug_idx={aug_idx} was requested."
                )
            raw = arr
        elif arr.ndim == 2:
            if aug_idx >= arr.shape[0]:
                raise IndexError(
                    f"aug_idx={aug_idx} out of bounds for {path}, which has {arr.shape[0]} augmentations."
                )
            raw = arr[aug_idx]
        else:
            raise ValueError(
                f"Expected token array with shape (N,) or (K, N), got shape {arr.shape} in {path}"
            )

        tokens = torch.from_numpy(arr[aug_idx]).long().flatten()
        # Shift into unified vocab range.
        tokens = tokens + info.codebook_offset
        return tokens

    def _modality_augmentation_count(self, modality: str, stem: str) -> int:
        info = get_modality(modality)
        path = Path(self.root_dir) / self.split / modality / f"{stem}{info.file_ext}"
        if info.data_type == 'tokens':
            arr = np.load(path, mmap_mode='r')
            return int(arr.shape[0]) if arr.ndim > 1 else 1
        if info.data_type == 'text':
            with open(path, 'r', encoding='utf-8') as f:
                return len(json.load(f))
        raise ValueError(f"Unknown data_type: {info.data_type}")

    def _available_augmentations(self, stem: str) -> int:
        if stem not in self._augmentation_count_cache:
            counts = [
                self._modality_augmentation_count(modality, stem)
                for modality in self.active_modalities
            ]
            available = min(self.sample_from_k_augmentations, *counts)
            if available <= 0:
                raise ValueError(
                    f"Sample '{stem}' has no available augmentations for "
                    f"modalities {self.active_modalities}; counts={counts}"
                )
            self._augmentation_count_cache[stem] = available
        return self._augmentation_count_cache[stem]

    def _load_text_modality(self, modality: str, stem: str, aug_idx: int) -> torch.Tensor:
        info = get_modality(modality)
        path = Path(self.root_dir) / self.split / modality / f"{stem}{info.file_ext}"
        with open(path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        caption = captions[aug_idx]
        tok = self._get_text_tokenizer()
        out = tok(
            caption,
            max_length=self.text_max_length,
            padding=False,              # we handle padding globally
            truncation=True,
            return_tensors='pt',
        )
        tokens = out['input_ids'][0].long()
        # Shift into unified vocab range.
        tokens = tokens + info.codebook_offset
        return tokens

    def _load_modality_chunk(self, modality: str, stem: str, aug_idx: int) -> torch.Tensor:
        """Load a modality, wrap it in <BO_mod> ... <EO_mod>."""
        info = get_modality(modality)
        if info.data_type == 'tokens':
            body = self._load_tokens_modality(modality, stem, aug_idx)
        elif info.data_type == 'text':
            body = self._load_text_modality(modality, stem, aug_idx)
        else:
            raise ValueError(f"Unknown data_type: {info.data_type}")

        bo = torch.tensor([info.bo_id], dtype=torch.long)
        eo = torch.tensor([info.eo_id], dtype=torch.long)
        return torch.cat([bo, body, eo], dim=0)

    def _get_rng(self, idx: int) -> np.random.Generator:
        """Per-sample RNG — deterministic given (seed, idx). Workers don't collide."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        return np.random.default_rng(seed=(self.seed, worker_id, idx))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.file_stems[idx]
        rng = self._get_rng(idx)

        aug_idx = int(rng.integers(0, self._available_augmentations(stem)))

        if self.modality_order == 'random' and len(self.active_modalities) > 1:
            perm = rng.permutation(len(self.active_modalities))
            ordered = [self.active_modalities[i] for i in perm]
        else:
            ordered = self.active_modalities

        chunks = [self._load_modality_chunk(m, stem, aug_idx) for m in ordered]

        # Track which intra-sequence ranges are image-token bodies (exclude BO/EO),
        # so we can permute them while keeping their canonical positions in RoPE.
        # Also build a per-token modality id (0 = pad/BO/EO, >0 = body of a modality)
        # used downstream for per-modality loss logging.
        body_ranges: List[Tuple[int, int]] = []
        per_token_modality_ids: List[torch.Tensor] = []
        cursor = 0
        for m, chunk in zip(ordered, chunks):
            mid = MODALITY_TO_ID[m]
            ids = torch.zeros(chunk.shape[0], dtype=torch.long)
            if chunk.shape[0] > 2:
                ids[1:-1] = mid
            per_token_modality_ids.append(ids)
            if self.shuffle_image_patches and get_modality(m).data_type == 'tokens' and chunk.shape[0] > 2:
                body_ranges.append((cursor + 1, cursor + chunk.shape[0] - 1))
            cursor += chunk.shape[0]
        seq = torch.cat(chunks, dim=0)
        seq_modality_ids = torch.cat(per_token_modality_ids, dim=0)

        # Truncate or pad to max_length.
        if seq.shape[0] > self.max_length:
            seq = seq[: self.max_length]
            seq_modality_ids = seq_modality_ids[: self.max_length]
            body_ranges = [(s, min(e, self.max_length)) for s, e in body_ranges if s < self.max_length]
            body_ranges = [(s, e) for s, e in body_ranges if e - s >= 2]

        input_ids = torch.full((self.max_length,), PAD_ID, dtype=torch.long)
        input_ids[: seq.shape[0]] = seq
        modality_ids = torch.zeros(self.max_length, dtype=torch.long)
        modality_ids[: seq_modality_ids.shape[0]] = seq_modality_ids

        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[: seq.shape[0]] = 1

        position_ids = torch.arange(self.max_length, dtype=torch.long)

        # Permute image-patch bodies in-place; position_ids carries the pre-shuffle
        # index so RoPE encodes canonical raster position, not shuffled sequence position.
        for s, e in body_ranges:
            sigma = torch.from_numpy(rng.permutation(e - s)).long() + s
            input_ids[s:e] = input_ids[sigma]
            position_ids[s:e] = sigma

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'position_ids': position_ids,
            'modality_ids': modality_ids,
        }