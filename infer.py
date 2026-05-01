"""
infer.py — Inference entry point for the MoR vision-language model.

Two caching strategies:
  (default)        use_cache=false  — full sequence recomputed each step
  infer.use_cache=true              — KV cache; RecursiveDynamicCache if kv_sharing.enable=true

Usage (Hydra CLI):

  # Unconditional generation with default config:
  python infer.py

  # Override checkpoint or config file:
  python infer.py infer.checkpoint=results/pretrain/smoke_50steps
  python infer.py --config-name=infer/my_run

  # Image prefix completion (first 128 tokens as context):
  python infer.py infer.prompt_npy=data/clevr_dataset/test/tok_rgb@256/02645.npy infer.prefix_len=128

  # Diverse sampling:
  python infer.py infer.temperature=0.9 infer.seed=42

  # Decode to PNG with Cosmos:
  python infer.py infer.cosmos_model_path=nvidia/Cosmos-0.1-Tokenizer-DI16x16
"""

import os

# DTensor workaround — same as pretrain.py (needed for transformers 4.52.4 + single-GPU).
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    class DTensor:
        pass
import transformers.modeling_utils
transformers.modeling_utils.DTensor = DTensor

from util.env import load_dotenv
load_dotenv()

from paths import HF_CACHE_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from model.util import load_model_from_config, load_checkpoint
from model.sharing_strategy import SHARING_STRATEGY
from util.config import preprocess_config
from lm_dataset.multimodal_vocab_shared_caption_scene_desc import MODALITIES, PAD_ID
from visualization.decode import (
    decode_output,
    save_depth_overlay,
    save_token_choice_depth_overlay,
    OUTPUT_DIR,
)


class EoModStopping(StoppingCriteria):
    """Stop generation as soon as every sequence in the batch ends with eo_id."""

    def __init__(self, eo_id):
        self.eo_id = eo_id

    def __call__(self, input_ids, scores, **kwargs):
        return bool((input_ids[:, -1] == self.eo_id).all())


def build_prompt(icfg):
    """
    Build input_ids in the unified vocab from the inference sub-config.

    Prompt patterns:
      prompt_npy + prefix_len N  →  <BO_rgb> first_N_tokens          (image prefix completion)
      prompt_npy                 →  <BO_rgb> all_tokens <EO_rgb> <BO_target>
      prompt_text                →  <BO_caption> text <EO_caption> <BO_target>
      (none)                     →  <BO_target>                       (unconditional)
    """
    chunks = []
    rgb_info = MODALITIES["tok_rgb@256"]
    caption_info = MODALITIES["caption"]

    if icfg.prompt_npy is not None:
        arr = np.load(icfg.prompt_npy)
        if arr.ndim == 2:
            arr = arr[icfg.aug_idx]
        tokens = torch.from_numpy(arr.flatten()).long() + rgb_info.codebook_offset

        if icfg.prefix_len is not None:
            n = min(icfg.prefix_len, len(tokens))
            return torch.cat([torch.tensor([rgb_info.bo_id]), tokens[:n]], dim=0).unsqueeze(0)

        chunks += [torch.tensor([rgb_info.bo_id]), tokens, torch.tensor([rgb_info.eo_id])]

    if icfg.prompt_text is not None:
        tok = AutoTokenizer.from_pretrained("gpt2")
        text_ids = tok(
            icfg.prompt_text, truncation=True, max_length=64, return_tensors="pt"
        )["input_ids"][0].long() + caption_info.codebook_offset
        chunks += [torch.tensor([caption_info.bo_id]), text_ids, torch.tensor([caption_info.eo_id])]

    target_info = MODALITIES[icfg.generate_modality]
    chunks.append(torch.tensor([target_info.bo_id]))

    return torch.cat(chunks, dim=0).unsqueeze(0)  # (1, prompt_len)


@hydra.main(config_path="conf", config_name="infer/mor_19500_rgb", version_base=None)
def main(cfg: DictConfig):
    icfg = cfg.infer

    # Patch cfg fields that preprocess_config and load_model_from_config expect.
    with open_dict(cfg):
        cfg.wandb = False
        cfg.wandb_entity = ""
        cfg.wandb_project = "inference"
        cfg.wandb_run_name = "inference"
        cfg.resume_from_checkpoint = False
    cfg = preprocess_config(cfg)

    # Model
    print("Building model architecture...")
    model = load_model_from_config(cfg)

    if cfg.recursive.get("enable"):
        model, _ = SHARING_STRATEGY[cfg.model](cfg, model)

    if "kv_sharing" in cfg and cfg.kv_sharing.get("enable"):
        model.set_kv_sharing_config(cfg)

    if "mor" in cfg and cfg.mor.get("enable"):
        if cfg.mor.type == "expert":
            model.transform_layer_to_mor_expert(cfg)
        elif cfg.mor.type == "token":
            model.transform_layer_to_mor_token(cfg)

    print(f"Loading checkpoint: {icfg.checkpoint}")
    model = load_checkpoint(model, icfg.checkpoint)
    model.to(icfg.device)
    model.eval()

    # Sampling is implied by any sampling arg being non-default.
    do_sample = icfg.do_sample or icfg.temperature != 1.0 or icfg.top_k != 0 or icfg.top_p != 1.0

    # Prompt
    prompt_ids = build_prompt(icfg).to(icfg.device)
    prompt_len = prompt_ids.shape[1]
    cache_mode = (
        "RecursiveDynamicCache" if icfg.use_cache and cfg.get("kv_sharing") and cfg.kv_sharing.get("enable")
        else "DynamicCache" if icfg.use_cache
        else "no cache (recompute)"
    )
    print(f"Prompt length : {prompt_len} tokens")
    print(f"Target        : '{icfg.generate_modality}', up to {icfg.max_new_tokens} new tokens")
    print(f"Cache mode    : {cache_mode}")

    target_info = MODALITIES[icfg.generate_modality]
    stopping = StoppingCriteriaList([EoModStopping(target_info.eo_id)])

    if icfg.seed is not None:
        torch.manual_seed(icfg.seed)

    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=icfg.max_new_tokens,
            use_cache=icfg.use_cache,
            stopping_criteria=stopping,
            do_sample=do_sample,
            temperature=icfg.temperature if do_sample else 1.0,
            top_k=icfg.top_k if do_sample else 0,
            top_p=icfg.top_p if do_sample else 1.0,
            pad_token_id=PAD_ID,
        )

    new_tokens = out[0, prompt_len:].cpu()
    print(f"Generated {len(new_tokens)} token(s).")
    image, timestamp = decode_output(
        new_tokens,
        icfg.generate_modality,
        cosmos_model_path=icfg.cosmos_model_path,
        patch_grid_size=icfg.patch_grid_size,
        device=icfg.device,
    )

    # Depth overlay: collect MoR routing decisions via forward hooks, then run
    # one no-cache pass over the full generated sequence.
    want_overlay = (
        icfg.get("save_depth_overlay", False)
        and "mor" in cfg and cfg.mor.get("enable")
        and cfg.mor.get("type") in ("expert", "token")
        and target_info.data_type == "tokens"
    )
    if want_overlay:
        mor_selected: list = []
        token_expert_indices: list = []

        def _hook(_module, _inputs, output):
            sel = getattr(output, "selected_tokens", None)
            if sel is not None:
                mor_selected.append(sel.detach().cpu())
            tei = getattr(output, "token_expert_indices", None)
            if tei is not None:
                token_expert_indices.append(tei.detach().cpu())

        handles = [m.register_forward_hook(_hook)
                   for m in model.modules() if getattr(m, "mor", False)]
        try:
            with torch.no_grad():
                model(input_ids=out, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        bo_positions = (out[0] == target_info.bo_id).nonzero(as_tuple=False)
        if len(bo_positions) == 0:
            print("  [warn] depth overlay skipped: no BO token found in output.")
        else:
            bo_idx = int(bo_positions[0].item())
            start = bo_idx + 1
            end = start + icfg.patch_grid_size * icfg.patch_grid_size
            if cfg.mor.get("type") == "token":
                save_token_choice_depth_overlay(
                    image=image,
                    token_expert_indices_per_layer=token_expert_indices,
                    image_token_slice=(start, end),
                    patch_grid_size=icfg.patch_grid_size,
                    out_dir=OUTPUT_DIR,
                    timestamp=timestamp,
                    num_recursions=cfg.recursive.num_recursion,
                    alpha=icfg.get("depth_overlay_alpha", 0.45),
                )
            else:
                save_depth_overlay(
                    image=image,
                    selected_tokens_per_layer=mor_selected,
                    image_token_slice=(start, end),
                    patch_grid_size=icfg.patch_grid_size,
                    out_dir=OUTPUT_DIR,
                    timestamp=timestamp,
                    alpha=icfg.get("depth_overlay_alpha", 0.45),
                )


if __name__ == "__main__":
    main()
