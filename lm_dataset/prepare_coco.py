"""
prepare_coco.py

Extracts, formats, and tokenizes the COCO dataset to perfectly match 
the CLEVR dataset structure for the Mixture of Representations (MoR) model.
"""

import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T

# Import NVIDIA Cosmos Tokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 42

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR = PROJECT_ROOT / "data/coco_dataset"
RAW_DIR = BASE_DIR / "raw"

# Inputs
TRAIN_RAW_DIR = RAW_DIR / "train2017"
VAL_RAW_DIR = RAW_DIR / "val2017"

# Cosmos Checkpoint
COSMOS_CKPT_PATH = PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-DI16x16"

# ==========================================
# SEEDING & SETUP
# ==========================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# The exact formatting pipeline to ensure CLEVR parity
format_image = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(256),
    T.ToTensor(), 
    # Center pixel values to [-1, 1] for the tokenizer
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

# ==========================================
# COSMOS TOKENIZER WRAPPER
# ==========================================
class CosmosTokenizerWrapper:
    def __init__(self, ckpt_path):
        encoder_path = f"{ckpt_path}/encoder.jit"
        print(f"[*] Loading Cosmos Encoder from: {encoder_path}")
        
        # We only need the encoder for dataset preparation
        self.model = ImageTokenizer(checkpoint_enc=encoder_path)
        
    def process_and_tokenize(self, img_path):
        # A. Load and ensure standard RGB
        pil_image = Image.open(img_path).convert("RGB")
        
        # B. Format to 256x256 tensor
        img_tensor = format_image(pil_image)
        
        # C. Add batch dim, move to GPU, and cast to bfloat16
        tensor_batch = img_tensor.unsqueeze(0).cuda().to(torch.bfloat16)
        
        # D. Tokenize
        with torch.no_grad():
            indices, _ = self.model.encode(tensor_batch)
            
        # Extract integer indices and flatten to 1D array
        token_array = indices.cpu().numpy().astype(np.int32).flatten()
        
        # E. Expand dims to (1, N) to match the CLEVR sample_from_k logic
        return np.expand_dims(token_array, axis=0)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("--- Starting COCO Data Preparation ---")
    
    # 1. Validate raw data exists
    if not TRAIN_RAW_DIR.exists() or not VAL_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing raw data! Ensure images are in {TRAIN_RAW_DIR} and {VAL_RAW_DIR}")

    # 2. Grab all unzipped files
    all_train_files = sorted(list(TRAIN_RAW_DIR.glob("*.jpg")))
    all_val_files = sorted(list(VAL_RAW_DIR.glob("*.jpg")))
    
    print(f"[*] Found {len(all_train_files)} raw train images.")
    print(f"[*] Found {len(all_val_files)} raw val images.")

    # 3. Shuffle train files deterministically and slice
    random.shuffle(all_train_files)
    
    splits = {
        "train": all_train_files[:50000],
        "test": all_train_files[50000:55000],
        "val": all_val_files  # The official 5,000 val set
    }

    print("\n[*] Target Splits:")
    for split_name, files in splits.items():
        print(f"    - {split_name}: {len(files)} images")

    # 4. Initialize Tokenizer (loads model to GPU)
    tokenizer = CosmosTokenizerWrapper(COSMOS_CKPT_PATH)

    # 5. Process and Save
    for split_name, file_paths in splits.items():
        # Exact structure: coco_dataset/<split>/tok_rgb@256/
        split_out_dir = BASE_DIR / split_name / "tok_rgb@256"
        split_out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[*] Processing '{split_name}' split into {split_out_dir}...")
        
        success_count = 0
        error_count = 0
        
        for i, img_path in enumerate(tqdm(file_paths, desc=f"{split_name.capitalize()} progress")):
            try:
                # Tokenize the image
                tokens_to_save = tokenizer.process_and_tokenize(img_path)
                
                # Save as 5-digit padded string (00000.npy, 00001.npy, etc.)
                save_name = f"{i:05d}.npy"
                save_path = split_out_dir / save_name
                
                np.save(save_path, tokens_to_save)
                success_count += 1
                
            except Exception as e:
                # Catch corrupt images or OS file locks without crashing the 50,000 loop
                print(f"\n[!] Error processing {img_path.name}: {e}")
                error_count += 1

        print(f"[*] '{split_name}' complete! Success: {success_count} | Errors: {error_count}")

    print("\n--- All Data Successfully Prepared for MoR Training ---")

if __name__ == "__main__":
    main()