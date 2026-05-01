import os

import numpy as np
import torch
from transformers import AutoTokenizer

from lm_dataset.multimodal_vocab_shared_caption_scene_desc import MODALITIES
from paths import SAVE_DIR


OUTPUT_DIR = os.path.join(SAVE_DIR, "infer/")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_depth_grid_overlay(image, grid, max_depth, patch_grid_size, out_dir,
                             timestamp, alpha=0.45, title="Exit depth",
                             colorbar_label="# MoR layers"):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from PIL import Image
    except ImportError:
        print("  [warn] matplotlib/PIL not available; skipping depth overlay.")
        return

    cmap = cm.get_cmap("viridis", max_depth + 1)
    norm_grid = grid.astype(np.float32) / max(max_depth, 1)
    heat_rgba = (cmap(norm_grid) * 255.0).astype(np.uint8)

    heatmap_path = os.path.join(out_dir, f"generated_depth_{timestamp}.png")
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=max_depth + 0.5,
                   interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, ticks=list(range(max_depth + 1)))
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"  Saved depth heatmap to {heatmap_path}")

    if image is None:
        return

    H, W = image.shape[:2]
    heat_img = Image.fromarray(heat_rgba, mode="RGBA").resize((W, H), Image.NEAREST)
    heat_arr = np.asarray(heat_img).astype(np.float32) / 255.0
    base = image.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * heat_arr[..., :3]
    blended = np.clip(blended * 255.0 + 0.5, 0, 255).astype(np.uint8)

    overlay_path = os.path.join(out_dir, f"generated_depth_overlay_{timestamp}.png")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(blended)
    ax.set_title(f"Depth overlay (alpha={alpha:.2f})")
    ax.set_xticks([])
    ax.set_yticks([])
    sm = cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=-0.5, vmax=max_depth + 0.5))
    cbar = fig.colorbar(sm, ax=ax, ticks=list(range(max_depth + 1)))
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)
    print(f"  Saved depth overlay to {overlay_path}")


def save_depth_overlay(image, selected_tokens_per_layer, image_token_slice,
                       patch_grid_size, out_dir, timestamp, alpha=0.45):
    """
    Save a per-token exit-depth heatmap and an α-blended overlay on `image`.

    selected_tokens_per_layer : list of LongTensors (bs, k_i, 1) of absolute
        positions returned by each MoR expert-choice layer (one entry per layer).
        depth[p] = number of those lists that contain p, so values range
        0..num_mor_layers. A larger depth means the token kept being recursed on.
    image_token_slice : (start, end) absolute positions of the image tokens
        in the sequence these selected_tokens index into.
    image : HxWx3 uint8 decoded image (or None to skip the overlay file).
    """
    if not selected_tokens_per_layer:
        print("  [warn] no MoR layers captured; skipping depth overlay.")
        return

    num_layers = len(selected_tokens_per_layer)
    seq_len = int(max(t.max().item() for t in selected_tokens_per_layer)) + 1
    depths = torch.zeros(seq_len, dtype=torch.long)
    for sel in selected_tokens_per_layer:
        idx = sel.view(-1).to(torch.long)
        depths.scatter_add_(0, idx, torch.ones_like(idx))

    start, end = image_token_slice
    expected = patch_grid_size * patch_grid_size
    if end - start != expected or end > seq_len:
        print(f"  [warn] depth overlay skipped: image slice [{start}:{end}] "
              f"doesn't match {patch_grid_size}x{patch_grid_size}={expected} "
              f"or exceeds seq_len={seq_len}.")
        return

    grid = depths[start:end].view(patch_grid_size, patch_grid_size).numpy()
    _save_depth_grid_overlay(
        image=image,
        grid=grid,
        max_depth=num_layers,
        patch_grid_size=patch_grid_size,
        out_dir=out_dir,
        timestamp=timestamp,
        alpha=alpha,
        title=f"Exit depth (0..{num_layers}) - higher = more recursion",
        colorbar_label="# MoR layers the token passed through",
    )


def save_token_choice_depth_overlay(image, token_expert_indices_per_layer,
                                    image_token_slice, patch_grid_size, out_dir,
                                    timestamp, num_recursions, alpha=0.45):
    """
    Save token-choice MoR recursion depths.

    token_expert_indices_per_layer : list of LongTensors (bs, seq_len)
        Token-choice router assignments. An expert index i means the token
        was processed for i + 1 recursion passes.
    """
    if not token_expert_indices_per_layer:
        print("  [warn] no token-choice MoR layers captured; skipping depth overlay.")
        return

    token_expert_indices = token_expert_indices_per_layer[-1]
    if token_expert_indices.dim() == 2:
        token_expert_indices = token_expert_indices[0]
    depths = token_expert_indices.to(torch.long).cpu() + 1

    start, end = image_token_slice
    expected = patch_grid_size * patch_grid_size
    if end - start != expected or end > depths.numel():
        print(f"  [warn] depth overlay skipped: image slice [{start}:{end}] "
              f"doesn't match {patch_grid_size}x{patch_grid_size}={expected} "
              f"or exceeds seq_len={depths.numel()}.")
        return

    grid = depths[start:end].view(patch_grid_size, patch_grid_size).numpy()
    max_depth = int(num_recursions)
    _save_depth_grid_overlay(
        image=image,
        grid=grid,
        max_depth=max_depth,
        patch_grid_size=patch_grid_size,
        out_dir=out_dir,
        timestamp=timestamp,
        alpha=alpha,
        title=f"Token-choice recursion depth (1..{max_depth})",
        colorbar_label="# recursion passes",
    )

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
    """Print decoded output. For image modalities, saves tokens.npy and optionally a PNG.

    Returns (image, timestamp) where image is HxWx3 uint8 or None.
    """
    info = MODALITIES[generate_modality]
    body = new_tokens[(new_tokens != info.bo_id) & (new_tokens != info.eo_id)]
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image = None

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

    return image, timestamp
