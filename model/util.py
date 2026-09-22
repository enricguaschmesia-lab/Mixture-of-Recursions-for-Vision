import json
import os
import warnings

from omegaconf import DictConfig
import torch
from transformers import AutoConfig

from model.base_model.modeling_llama import LlamaForCausalLM
from model.recursive_model.modeling_llama import LlamaForCausalLM as RecursiveLlamaForCausalLM
from model.mor_model.modeling_llama import MoRLlamaForCausalLM

MODEL_CLS = {
    "smollm": LlamaForCausalLM,
    "smollm2": LlamaForCausalLM,
}

RECURSIVE_MODEL_CLS = {
    "smollm": RecursiveLlamaForCausalLM,
    "smollm2": RecursiveLlamaForCausalLM,
}

MOR_MODEL_CLS = {
    "smollm": MoRLlamaForCausalLM,
    "smollm2": MoRLlamaForCausalLM,
}

if "WANDB_MODE" not in os.environ:
    local_files_only = True
else:
    local_files_only = os.environ["WANDB_MODE"] == "offline"


def get_torch_dtype(cfg: DictConfig):
    """The COMPUTE dtype named by `precision`.

    ⚠ Duplicated from util.misc.get_torch_dtype (both predate this change).
    Kept so existing importers of this symbol are unaffected, but the model's
    parameters are built from get_param_dtype below -- under mixed precision
    the two differ.
    """
    if cfg.precision == "bf16":
        return torch.bfloat16
    elif cfg.precision == "fp16":
        return torch.float16
    elif cfg.precision == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Precision {cfg.precision} not supported.")


def load_model_from_config(cfg: DictConfig):
    if "mor" in cfg and cfg.mor.enable:
        model_cls = MOR_MODEL_CLS[cfg.model]
    elif cfg.recursive.enable or ("kv_sharing" in cfg and cfg.kv_sharing.enable):
        model_cls = RECURSIVE_MODEL_CLS[cfg.model]
    else:
        model_cls = MODEL_CLS[cfg.model]
        
    attn_implementation = cfg.get("attn_implementation", "flash_attention_2")
    # ⚠ PARAMETER dtype, not the compute dtype. Under `mixed_precision: true`
    # the weights stay fp32 and autocast handles the casting; without it this
    # is identical to get_torch_dtype. See util.misc.get_param_dtype.
    from util.misc import get_param_dtype
    torch_dtype = get_param_dtype(cfg)
    
    if cfg.use_pretrained_weights:
        print("Loading model from pretrained weights...")
        print(f"Loading model with {attn_implementation}...")
        return model_cls.from_pretrained(
            cfg.model_name_or_path,
            attn_implementation=attn_implementation, 
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
                
    else:
        print("Initializing model from scratch...")
        config = AutoConfig.from_pretrained(
            cfg.model_name_or_path,
            attn_implementation=attn_implementation, 
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        # Workaround for transformers 4.52.4 + newer SmolLM configs:
    # the tp_plan validation path iterates over ALL_PARALLEL_STYLES which
    # can be None in some torch/accelerate combinations. We don't use
    # tensor parallelism, so drop the field.
        if hasattr(config, 'base_model_tp_plan'):
            config.base_model_tp_plan = None

        
        if cfg.get("model_config") is not None:
            print("Using custom config for vanilla model...")
            for k, v in cfg.model_config.items():
                if not hasattr(config, k):
                    raise ValueError(f"Config key {k} not found in model config.")
                print(f" {k}: {v}")
                setattr(config, k, v)
        if cfg.get("max_length") and cfg.max_length != config.max_position_embeddings:
            warnings.warn(f"original max_position_embeddings of {config.max_position_embeddings} is changed to {cfg.max_length}")
            setattr(config, "max_position_embeddings", cfg.max_length)
        return model_cls._from_config(
            config, attn_implementation=attn_implementation, torch_dtype=torch_dtype,)


def load_checkpoint(model, checkpoint_path):
    """
    Load weights into model from a HuggingFace Trainer checkpoint directory.
    Accepts a local path or a HuggingFace model ID (owner/repo).
    Handles single safetensors, sharded safetensors, and pytorch_model.bin.
    """
    if not os.path.exists(checkpoint_path) and "/" in checkpoint_path and not checkpoint_path.startswith("/"):
        from huggingface_hub import snapshot_download
        print(f"  Downloading from HuggingFace Hub: {checkpoint_path}")
        checkpoint_path = snapshot_download(
            checkpoint_path,
            ignore_patterns=["optimizer.pt", "rng_state.pth", "training_args.bin", "scheduler.pt"],
        )
        print(f"  Cached at: {checkpoint_path}")

    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(index_path):
        from safetensors.torch import load_file
        with open(index_path) as f:
            index = json.load(f)
        state_dict = {}
        for shard in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(os.path.join(checkpoint_path, shard), device="cpu"))
    elif os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No checkpoint found in '{checkpoint_path}'. "
            "Expected model.safetensors, model.safetensors.index.json, or pytorch_model.bin."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected key(s): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    if missing:
        print(f"  [warn] {len(missing)} missing key(s): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    return model