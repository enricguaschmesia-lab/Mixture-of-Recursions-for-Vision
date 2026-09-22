import os
import builtins
import logging
import glob
import warnings
from pathlib import Path

import datetime
import logging
import torch.distributed as dist
import numpy as np
import torch

from paths import SAVE_DIR

original_print = builtins.print


def _print_rank_zero(*args, **kwargs):
    if os.environ.get("RANK") is not None and int(os.environ.get("RANK")) != 0:
        return
    original_print(*args, **kwargs)
    
    
def print_rank_zero():
    # overwrite the print function
    builtins.print = _print_rank_zero
    
    
def convert_to_serializable(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist() 
    elif isinstance(obj, complex):
        return [obj.real, obj.imag] 
    elif isinstance(obj, torch.dtype):
        return str(obj)
    else:
        raise TypeError(f"Type {type(obj)} is not serializable")
    

def check_saved_checkpoint(cfg, SAVE_DIR):
    """
    Checks if any folders starting with {SAVE_DIR}/pretrain/{cfg.output_dir}/checkpoint-
    exist. If they do, return True.

    Uses `output_dir`, not `name`: `output_dir` is what TrainingArguments actually
    writes to, and what get_latest_checkpoint_path() below already reads. The two
    coincide only because preprocess_config defaults output_dir to name -- so any
    run that sets them apart (eo/scripts/train_eo.sh puts runs under
    phase3/<run_id> while keeping a short name for W&B) made this look in a
    directory that does not exist, and silently report "no checkpoint".
    Measured 2026-09-21, Phase 3 Step 0.4.
    """

    checkpoint_dir_pattern = os.path.join(
        SAVE_DIR, "pretrain", cfg.get("output_dir") or cfg.name, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_dir_pattern)

    if checkpoint_dirs:
        print(f"Found checkpoint directories: {checkpoint_dirs}")
        return True
    else:
        print(f"No checkpoint directories found matching {checkpoint_dir_pattern}")
        return False


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    relaxation_param = 0
    
    is_peft_model = hasattr(model, "peft_config")
    
    for name, param in model.named_parameters():
        # if 'base_model' in name and 
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        
        if is_peft_model and ('default' in name or 'adaption' in name):
            relaxation_param += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            
    return trainable_params, all_param, relaxation_param


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.

    Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
    num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
    (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
    For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
    prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
    of trainable parameters of the backbone transformer model which can be different.
    """
    # note: same as PeftModel.print_trainable_parameters
    trainable_params, all_param, relaxation_param = get_nb_trainable_parameters(model)

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"relaxation params: {relaxation_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    
    
def get_torch_dtype(cfg):
    """The COMPUTE dtype named by `precision`.

    ⚠ This is not necessarily the dtype the parameters are stored in. For that,
    use get_param_dtype() -- see its docstring for why the two came apart.
    """
    if cfg.precision == "bf16":
        return torch.bfloat16
    elif cfg.precision == "fp16":
        return torch.float16
    elif cfg.precision == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Invalid precision: {cfg.precision}")


def get_param_dtype(cfg):
    """The dtype PARAMETERS are created in. Every site that allocates a weight
    must use this, not get_torch_dtype.

    ⚠ Why the two are different (Phase 3 Step 2). `precision` used to set both
    the parameter dtype AND `TrainingArguments(fp16=...)`, which enables a
    `GradScaler`. A scaler requires fp32 master weights, so `precision: fp16`
    made the parameters fp16 and then died at the FIRST optimizer step with
    "Attempting to unscale FP16 gradients" -- the fp16 path had therefore never
    run in this codebase. bf16 escaped it only because bf16 needs no scaler.

    Mixed precision means fp32 weights with autocast casting the *compute*. So
    under `mixed_precision: true` the parameters stay fp32 and `precision` is
    read as the AUTOCAST dtype only.

    ⚠ Opt-in, and deliberately so. Absent the key this returns exactly what
    get_torch_dtype returns, so every CLEVR config -- all of which say
    `precision: bf16` and none of which set this -- keeps pure-bf16 parameters
    and is bit-for-bit unaffected. Phases 3-4 compare EO routing against those
    CLEVR results; silently changing the reference path's training regime to
    mixed precision would confound that comparison, which is a far worse
    outcome than carrying one extra config key.
    """
    if cfg.get("mixed_precision", False):
        return torch.float32
    return get_torch_dtype(cfg)


def get_latest_checkpoint_path(cfg, resume_step=None):
    if resume_step is None:
        output_dir = os.path.join(SAVE_DIR, "pretrain", cfg.output_dir)
        checkpoint_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.name.split("-")[-1]))
    else:
        assert "_resume" in cfg.output_dir, "resume_step should be used with _resume in output_dir"
        output_dir = os.path.join(SAVE_DIR, "pretrain", cfg.output_dir.rsplit("_resume", 1)[0])
        latest_checkpoint = os.path.join(output_dir, f"checkpoint-{resume_step}")
    return latest_checkpoint


def get_iterator(param_dict): 
    for _name, param in param_dict.items():
        yield param
        

def get_launcher_type():
    is_accelerate_launch = any(var.startswith('ACCELERATE_') for var in os.environ)
    if is_accelerate_launch:
        return "accelerate"
    else:
        return "deepspeed"