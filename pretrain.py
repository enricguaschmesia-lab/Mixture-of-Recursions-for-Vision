"""
Training entry point for MoR vision and multimodal pretraining.

All the config files used with this script are in this folder:
    conf/pretrain_vision/

This script is used by scripts/pretrain.sh: please launch all your trainings from pretrain.sh,
not directly from this script.

Main outputs:
    SAVE_DIR/pretrain/<cfg.output_dir>/

Notes:
    Paths are resolved through .env, environment variables, and paths.py.
"""

import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Auto-load <repo>/.env into os.environ so Hydra's ${oc.env:...} interpolation
# and paths.py can see it. Must run before the paths import below.
from util.env import load_dotenv
from util.seeding import set_global_seed

load_dotenv()

from paths import SAVE_DIR, PROJECT_ROOT, HF_CACHE_DIR

os.environ["HF_HOME"] = HF_CACHE_DIR


import string

import random
import hydra
import torch
from omegaconf import DictConfig, open_dict
from transformers import TrainingArguments, Trainer


# Workaround for transformers 4.52.4: save_pretrained references DTensor
# without importing it in single-GPU contexts. Patch it in.
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    class DTensor: pass
import transformers.modeling_utils
transformers.modeling_utils.DTensor = DTensor

from lm_dataset.load_dataset import load_dataset_from_config, MULTIMODAL_DATASETS
from lm_dataset.modality_registry import assert_vocab_size
from model.util import load_model_from_config
from model.sharing_strategy import SHARING_STRATEGY
from util.config import preprocess_config
from util.tokenizer import load_tokenizer_from_config 
from util.trainer_pt import MoRTrainer
from util.callback import FixedStoppingCallback, ScalingLawsSaveCallback, MultimodalVisionEvalCallback
from util.misc import print_trainable_parameters, get_latest_checkpoint_path, print_rank_zero, get_launcher_type; print_rank_zero()

@hydra.main(config_path="conf/pretrain_vision", config_name="rgb_single_training/smoke_50steps", version_base=None)
def main(cfg: DictConfig):
    # Resolve derived config fields and normalize Hydra config before use.
    cfg = preprocess_config(cfg)

    if cfg.wandb and cfg.get("wandb_run_id") is None:
        characters = string.ascii_letters + string.digits
        with open_dict(cfg):
            cfg.wandb_run_id = "".join(random.choices(characters, k=8))
        print(f"Auto-generated wandb_run_id: {cfg.wandb_run_id}")
                
    # wandb settings
    if cfg.get("wandb"):
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity # name your W&B team
        os.environ["WANDB_PROJECT"] = cfg.wandb_project # name your W&B project
        if cfg.get("wandb_watch") is not None:
            os.environ["WANDB_WATCH"] = cfg.get("wandb_watch")
        os.environ ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = cfg.wandb_run_id
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = cfg.get("wandb_mode", "online")
        if os.environ["WANDB_MODE"] == "offline":
            os.environ["WANDB_DIR"] = PROJECT_ROOT
        os.environ["WANDB_SAVE_CODE"] = "false"
        os.environ["WANDB_LOG_MODEL"] = "false"
        
    set_global_seed(
        cfg.get("seed"),
        deterministic_cuda=cfg.get("deterministic_cuda", False),
    )

    launcher_type = get_launcher_type()
    
    print("Loading tokenizers...")
    # Load tokenizer early so tokenizer-related errors fail before training starts
    tokenizer = load_tokenizer_from_config(cfg)

    print("Loading dataset...")
    train_dataset = load_dataset_from_config(cfg)
    if cfg.resume_from_checkpoint:
        latest_checkpoint = get_latest_checkpoint_path(
            cfg, resume_step=cfg.resume_step if ("resume_step" in cfg and cfg.resume_step is not None) else None,
        )
        dataset_state_path = os.path.join(str(latest_checkpoint), "dataset.pt")
        if hasattr(train_dataset, "load_state_dict") and os.path.isfile(dataset_state_path):
            train_dataset.load_state_dict(torch.load(dataset_state_path))
        else:
            print("Skipping dataset state restore (map-style dataset or missing dataset.pt).")


    print ("Loading models...")
    # Check vocab_size against the registry BEFORE building the model: the
    # embedding table is the largest tensor in the model, and a mismatch here is
    # not reliably fatal later (too small raises far from the cause; too large
    # just carries dead rows and a wrong ln(V) baseline). No-op when the config
    # has no explicit model_config.vocab_size.
    _expected_vocab = assert_vocab_size(cfg)
    if _expected_vocab is not None:
        print(f"vocab_size checked against registry: {_expected_vocab}")

    # Build the base model first, then apply recursive parameter sharing,
    # optional KV-sharing settings, and finally MoR router transformations.
    model = load_model_from_config(cfg)
    
    if cfg.recursive.get("enable"):        
        model, lora_init_dict = SHARING_STRATEGY[cfg.model](cfg, model)
    
    if "kv_sharing" in cfg and cfg.kv_sharing.get("enable"):
        model.set_kv_sharing_config(cfg)
    if "mor" in cfg and cfg.mor.get("enable"):         
        # MoR models need a custom trainer for router-specific losses/logging.   
        if cfg.mor.type == "expert":
            model.transform_layer_to_mor_expert(cfg)
        elif cfg.mor.type == "token":
            model.transform_layer_to_mor_token(cfg)
        else:
            raise ValueError(f"Unknown MoR type {cfg.mor.type}.")
        
    print_trainable_parameters(model)
        
    report_to = []
    if cfg.wandb:
        report_to.append("wandb")
    if cfg.tensorboard:
        report_to.append("tensorboard")
    
    train_args = TrainingArguments(
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine_with_min_lr"),
        lr_scheduler_kwargs=dict(cfg.get("lr_scheduler_kwargs", {"min_lr_rate": 0.1,})),
        learning_rate=cfg.learning_rate,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        output_dir=os.path.join(SAVE_DIR, "pretrain", cfg.output_dir),
        max_steps=cfg.num_train_steps,
        warmup_steps=cfg.num_warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        # ⚠ safetensors cannot serialize SHARED storage, and recursive parameter
        # sharing is exactly that: every block_list entry is a literal alias of
        # one tensor, so save_pretrained raises
        #   "The weights trying to be saved contained shared tensors ..."
        # The condition used to be launcher-based alone, which meant any
        # non-accelerate launch of a recursive model trained fine and then died
        # at its FIRST checkpoint -- after the training time was already spent.
        # Found on the first EO smoke run, 2026-09-20.
        save_safetensors=False if (
            launcher_type == "accelerate"
            or cfg.recursive.get("enable")
            or ("mor" in cfg and cfg.mor.get("enable"))
        ) else True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
        dataloader_num_workers=cfg.dataloader_num_workers,
        bf16=cfg.precision == "bf16",
        fp16=cfg.precision == "fp16",
        overwrite_output_dir=True,
        report_to=report_to,
        run_name=cfg.wandb_run_name,
        logging_dir=cfg.tensorboard_dir,
        deepspeed=cfg.deepspeed if launcher_type == "deepspeed" else None,
        log_on_each_node=False,
        seed=cfg.get("seed", 42),
        data_seed=cfg.get("seed", 42),
        # ⚠ MUST be False. With the default True, Trainer.get_train_dataloader
        # wraps the collator in RemoveColumnsCollator for any dataset that is not
        # a datasets.Dataset -- ours are plain torch Datasets -- and that strips
        # every key absent from model.forward()'s signature. `modality_ids` is
        # exactly such a key: it is carried for per-modality loss logging and is
        # deliberately popped before the forward.
        #
        # The failure is silent. No warning, no error: modality_ids simply never
        # reaches compute_loss, per_modality_loss stays empty, and the
        # `loss_<modality>` entries are absent from every log line. Measured on
        # the first EO smoke run (2026-09-20), where 30 steps trained cleanly and
        # logged no per-modality loss at all. Those numbers feed Phase 4.
        remove_unused_columns=False,
    )
    
    callbacks = []
    fixed_save_steps = cfg.fixed_save_steps if ("fixed_save_steps" in cfg and cfg.fixed_save_steps) else None
    if cfg.stop_steps is not None:
        callbacks.append(FixedStoppingCallback(cfg.stop_steps))
    ds_names = [ds.strip() for ds in cfg.dataset.split(',')]
    if all(ds in MULTIMODAL_DATASETS for ds in ds_names):
        # Map-style dataset; no per-iteration state to save. Resume restarts at epoch boundary.
        if cfg.get("vision_eval") and cfg.vision_eval.get("enable"):
            callbacks.append(MultimodalVisionEvalCallback(cfg))


    if fixed_save_steps is not None:
        callbacks.append(ScalingLawsSaveCallback(fixed_save_steps,))
        
    if "mor" in cfg and cfg.mor.get("enable"):
        trainer = MoRTrainer(model=model, args=train_args, train_dataset=train_dataset, callbacks=callbacks, cfg=cfg,)
    else:
        trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, callbacks=callbacks,)
    
    train_result = trainer.train(
        resume_from_checkpoint=cfg.resume_from_checkpoint
    )
    metrics = train_result.metrics
    trainer.log_metrics("pretrain", metrics)
    trainer.save_metrics("pretrain", metrics)
    trainer.save_state()
    trainer.save_model()
    
    if cfg.get("relaxation") and cfg.relaxation.get("enable"):
        trainer.model.base_model.model.save_pretrained(train_args.output_dir, safe_serialization=False)
    
    
if __name__ == "__main__":
    main()