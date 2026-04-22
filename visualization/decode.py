import os

import numpy as np
import torch
from transformers import AutoTokenizer

from lm_dataset.multimodal_vocab_shared_caption_scene_desc import MODALITIES
from paths import SAVE_DIR


OUTPUT_DIR = os.path.join(SAVE_DIR, "infer/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _resolve_cosmos_decoder_jit(cosmos_model_path):
    """Return a local path to a Cosmos decoder.jit, downloading from HF if needed."""
    if os.path.isfile(cosmos_model_path):
        return cosmos_model_path
    if os.path.isdir(cosmos_model_path):
        candidate = os.path.join(cosmos_model_path, "decoder.jit")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"decoder.jit not found in {cosmos_model_path}")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=cosmos_model_path, filename="decoder.jit")


def decode_tokens_to_image(raw_tokens, cosmos_model_path, patch_grid_size, device):
    """
    Decode raw VQ codebook indices to an RGB image via a Cosmos ImageTokenizer.

    raw_tokens      : 1D int array of length patch_grid_size**2
    cosmos_model_path: HF repo id, directory, or path to decoder.jit
    Returns HxWx3 uint8 array, or None on failure.
    """
    expected = patch_grid_size * patch_grid_size
    if raw_tokens.size != expected:
        print(f"  [warn] Cosmos decode skipped: got {raw_tokens.size} tokens, "
              f"expected {expected} ({patch_grid_size}x{patch_grid_size}).")
        return None

    try:
        from cosmos_tokenizer.image_lib import ImageTokenizer
    except ImportError:
        print("  [warn] cosmos_tokenizer not installed. pip install cosmos-tokenizer to enable.")
        return None

    if not (device == "cuda" and torch.cuda.is_available()):
        print("  [warn] Cosmos decode requires CUDA; skipping image save.")
        return None

    try:
        decoder_jit = _resolve_cosmos_decoder_jit(cosmos_model_path)
        tokenizer = ImageTokenizer(checkpoint_dec=decoder_jit, device="cuda", dtype="bfloat16")
        indices = torch.from_numpy(raw_tokens.astype(np.int64)).reshape(
            1, patch_grid_size, patch_grid_size
        ).to("cuda")
        with torch.no_grad():
            decoded = tokenizer.decode(indices)         # [B, 3, H, W], range ~[-1, 1]
        decoded = decoded.float().cpu().numpy()[0]      # [3, H, W]
        decoded = np.clip((decoded + 1.0) / 2.0, 0.0, 1.0)
        return (decoded.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)  # HWC uint8
    except Exception as exc:
        print(f"  [warn] Cosmos decode failed: {exc}")
        return None


def decode_output(new_tokens, generate_modality, cosmos_model_path=None,
                  patch_grid_size=16, device="cpu"):
    """Print decoded output. For image modalities, saves tokens.npy and optionally a PNG."""
    info = MODALITIES[generate_modality]
    body = new_tokens[(new_tokens != info.bo_id) & (new_tokens != info.eo_id)]
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if info.data_type == "tokens":
        raw = (body - info.codebook_offset).cpu().numpy()
        print(f"Generated {len(raw)} image token(s) for '{generate_modality}'.")
        print(f"  First 16 codebook indices: {raw[:16]}")
        
        np.save(os.path.join(OUTPUT_DIR, f"generated_tokens_{timestamp}.npy"), raw)
        print(f"  Saved tokens to {os.path.join(OUTPUT_DIR, f'generated_tokens_{timestamp}.npy')}")

        if cosmos_model_path is not None:
            image = decode_tokens_to_image(raw[:patch_grid_size * patch_grid_size], cosmos_model_path, patch_grid_size, device)
            if image is not None:
                from PIL import Image
                Image.fromarray(image).save(os.path.join(OUTPUT_DIR, f"generated_image_{timestamp}.png"))
                print(f"  Saved decoded image to {os.path.join(OUTPUT_DIR, f'generated_image_{timestamp}.png')} ({image.shape[1]}x{image.shape[0]})")
    else:
        tok = AutoTokenizer.from_pretrained("gpt2")
        ids = (body - info.codebook_offset).cpu().tolist()
        print(f"Generated caption: {tok.decode(ids, skip_special_tokens=True)}")
