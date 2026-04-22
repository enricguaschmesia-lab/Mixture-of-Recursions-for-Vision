import os 

import pickle

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import CallbackHandler


class FixedStoppingCallback(TrainerCallback):
    """
    This callback is used when you want to set a certain num_train_steps for the learning rate scheduler 
    (e.g. "get linear schedule with warmup) but you want to stop training before that number of steps is reached.
    """
    def __init__(self, stop_steps: int):
        super().__init__()
        self.stop_steps = stop_steps
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.stop_steps:
            print(f"Stopping training at stop_steps={self.stop_steps}")
            control.should_training_stop = True

class PeftSaveCallback(TrainerCallback):
    """
    This callback is used to save the base model of peft model.
    """
    def __init__(self, save_steps: int, fixed_save_steps: str = None):
        super().__init__()
        self.save_steps = save_steps
        self.fixed_save_steps = []
        if fixed_save_steps is not None:
            self.fixed_save_steps = [int(step) for step in fixed_save_steps.split(",")]
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (state.global_step % self.save_steps == 0 or state.global_step in self.fixed_save_steps) and state.global_step != 0:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            output_dir = os.path.join(args.output_dir, checkpoint_folder)
            
            kwargs["model"].base_model.model.save_pretrained(output_dir, safe_serialization=False)
            
            
class DatasetSaveCallback(TrainerCallback):
    def __init__(self, save_steps: int, fixed_save_steps: str = None):
        super().__init__()
        self.save_steps = save_steps
        self.fixed_save_steps = []
        if fixed_save_steps is not None:
            self.fixed_save_steps = [int(step) for step in fixed_save_steps.split(",")]
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (state.global_step % self.save_steps == 0 or state.global_step in self.fixed_save_steps) and state.global_step != 0:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            output_dir = os.path.join(args.output_dir, checkpoint_folder)
            
            dataset = kwargs["train_dataloader"].dataset
            state_dict = dataset.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "dataset.pt"))


class MorSaveCallback(TrainerCallback):
    def __init__(self, save_steps: int, fixed_save_steps: str = None):
        super().__init__()
        self.save_steps = save_steps
        self.fixed_save_steps = []
        if fixed_save_steps is not None:
            self.fixed_save_steps = [int(step) for step in fixed_save_steps.split(",")]  # list
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (state.global_step % self.save_steps == 0 or state.global_step in self.fixed_save_steps) and state.global_step != 0:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            output_dir = os.path.join(args.output_dir, checkpoint_folder)
            
            training_step = None
            for layer_idx in range(len(kwargs['model'].model.layers)):
                if hasattr(kwargs['model'].model.layers[layer_idx], "training_step"):
                    training_step = kwargs['model'].model.layers[layer_idx].training_step
                    break
            if training_step is not None:
                with open(os.path.join(output_dir, "training_step.pickle"), 'wb') as f:
                    pickle.dump(training_step, f)
            
            if "sam_optimizer" in kwargs and kwargs["sam_optimizer"] is not None:
                torch.save(kwargs["sam_optimizer"].state_dict(), os.path.join(output_dir, "sam_optimizer.pt"))
            if "sam_lr_scheduler" in kwargs and kwargs["sam_lr_scheduler"] is not None:
                torch.save(kwargs["sam_lr_scheduler"].state_dict(), os.path.join(output_dir, "sam_lr_scheduler.pt"))
                    
                
class MoRCallbackHandler(CallbackHandler):
    def __init__(self, callbacks, model, processing_class, optimizer, lr_scheduler, sam_optimizer=None, sam_lr_scheduler=None,):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None
        
        self.sam_optimizer = sam_optimizer
        self.sam_lr_scheduler = sam_lr_scheduler
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = False
        kwargs["sam_optimizer"] = self.sam_optimizer
        kwargs["sam_lr_scheduler"] = self.sam_lr_scheduler
        return self.call_event("on_save", args, state, control, **kwargs)


class MultimodalVisionEvalCallback(TrainerCallback):
    """
    After every eval_epoch_interval epochs, forward 2 val images through the MoR model,
    count how many recursion passes each image patch received, render a heatmap, and
    log to WandB / save to disk.

    If vision_eval.cosmos_model_path is set, Cosmos VQ tokens are decoded to RGB before
    overlaying the heatmap. Otherwise a normalised token-ID grid is shown as the base image.

    Expected config block (under the top-level Hydra config):
        vision_eval:
          enable: true
          eval_epoch_interval: 10   # run every N completed epochs
          num_val_samples: 2
          patch_grid_size: 16       # sqrt(num_image_tokens); 16×16 = 256 tokens
          cosmos_model_path: null   # path / HF id for CausalVideoTokenizer (optional)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        vcfg = cfg.get("vision_eval", {})
        self.eval_epoch_interval = int(vcfg.get("eval_epoch_interval", 10))
        self.num_val_samples = int(vcfg.get("num_val_samples", 2))
        self.patch_grid_size = int(vcfg.get("patch_grid_size", 16))
        self.num_recursions = cfg.recursive.num_recursion
        self._cosmos_model_path = vcfg.get("cosmos_model_path", None)
        self._last_eval_epoch = -1
        self._val_dataset = None
        self._text_tok = None

    def _get_val_dataset(self):
        if self._val_dataset is not None:
            return self._val_dataset
        from lm_dataset.multimodal_tokenized_dataset import MultimodalTokenizedDataset
        from lm_dataset.load_dataset import MULTIMODAL_DATASETS
        ds_name = [ds.strip() for ds in self.cfg.dataset.split(",")][0]
        ds_cfg = MULTIMODAL_DATASETS[ds_name]
        mm_cfg = self.cfg.get("multimodal", {})
        active_modalities = list(mm_cfg.get("active_modalities", ["tok_rgb@256"]))
        self._val_dataset = MultimodalTokenizedDataset(
            root_dir=ds_cfg["root_dir"],
            split="val",
            active_modalities=active_modalities,
            max_length=self.cfg.max_length,
            modality_order="fixed",
            sample_from_k_augmentations=1,  # always aug_idx=0 for reproducibility
        )
        return self._val_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        epoch = round(state.epoch)
        if epoch <= 0 or epoch % self.eval_epoch_interval != 0 or epoch == self._last_eval_epoch:
            return
        self._last_eval_epoch = epoch

        try:
            val_dataset = self._get_val_dataset()
        except (FileNotFoundError, RuntimeError) as exc:
            warnings.warn(f"[MultimodalVisionEvalCallback] Cannot load val dataset: {exc}")
            return

        self._run_vision_eval(args, state, kwargs["model"], val_dataset)

    def _run_vision_eval(self, args, state, model, val_dataset):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            warnings.warn("[MultimodalVisionEvalCallback] matplotlib not installed; skipping.")
            return

        from lm_dataset.multimodal_vocab_shared_caption_scene_desc import get_modality
        mm_cfg = self.cfg.get("multimodal", {})
        active_modalities = list(mm_cfg.get("active_modalities", ["tok_rgb@256"]))
        n_mods = len(active_modalities)

        # Unwrap DDP / accelerate wrapper
        m = model.module if hasattr(model, "module") else model
        device = next(m.parameters()).device

        n = min(self.num_val_samples, len(val_dataset))
        samples = [val_dataset[i] for i in range(n)]
        input_ids = torch.stack([s["input_ids"] for s in samples]).to(device)
        attention_mask = torch.stack([s["attention_mask"] for s in samples]).to(device)

        # Hook every expert-MoR layer to capture selected_tokens per recursion.
        # Token-choice MoR uses a different selection mechanism; we record it via the
        # same hook since MoRLayerOutputWithPast always carries selected_tokens.
        selected_tokens_list = []
        hooks = []
        for layer in m.model.layers:
            if getattr(layer, "mor", False):
                def _hook(module, inp, out, _lst=selected_tokens_list):
                    if getattr(out, "selected_tokens", None) is not None:
                        _lst.append(out.selected_tokens.detach().cpu())
                hooks.append(layer.register_forward_hook(_hook))

        model.eval()
        with torch.no_grad():
            m(input_ids=input_ids, attention_mask=attention_mask)
        for h in hooks:
            h.remove()
        model.train()

        # Build per-position recursion-count map  [n, seq_len]
        seq_len = input_ids.shape[1]
        count_map = torch.zeros(n, seq_len)
        for st in selected_tokens_list:          # st: [n, top_k, 1]
            st_flat = st.squeeze(-1)             # [n, top_k]
            for b in range(n):
                count_map[b].scatter_add_(0, st_flat[b], torch.ones(st_flat.shape[1]))

        ids_cpu = input_ids.cpu()

        n_rows = n * n_mods
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows), squeeze=False)

        for i in range(n):
            for j, mod_name in enumerate(active_modalities):
                row = i * n_mods + j
                info = get_modality(mod_name)
                ids = ids_cpu[i]              # [seq_len]
                counts = count_map[i]         # [seq_len]

                # Locate BO and EO in the flattened sequence using their unique token IDs
                bo_matches = (ids == info.bo_id).nonzero(as_tuple=True)[0]
                eo_matches = (ids == info.eo_id).nonzero(as_tuple=True)[0]
                if len(bo_matches) == 0 or len(eo_matches) == 0:
                    for col in range(2):
                        axes[row, col].axis("off")
                        axes[row, col].set_title(f"[{mod_name}] not in sequence (sample {i})")
                    continue

                bo_pos = bo_matches[0].item()
                eo_pos = eo_matches[0].item()

                # Body is everything strictly between BO and EO.
                # For image modalities: body = [tok_0+offset, ..., tok_255+offset]
                # For text modalities:  body = [SOS+offset, tok_0+offset, ..., tok_N+offset, EOS+offset]
                body_counts = counts[bo_pos + 1 : eo_pos].numpy()
                body_tokens = ids[bo_pos + 1 : eo_pos]
                # Subtract codebook_offset to recover raw codebook / tokenizer indices
                raw_tokens = (body_tokens - info.codebook_offset).numpy()

                if info.data_type == "tokens":
                    self._plot_image_modality(
                        axes[row], mod_name, i, raw_tokens, body_counts, plt
                    )

                elif info.data_type == "text":
                    # Body starts with SOS and ends with EOS (added by TemplateProcessing).
                    # raw_tokens[0]  = tok.bos_token_id  (SOS)
                    # raw_tokens[-1] = tok.eos_token_id  (EOS)
                    # raw_tokens[1:-1] = actual content token IDs
                    self._plot_text_modality(
                        axes[row], mod_name, i, raw_tokens, body_counts, plt, np
                    )

        plt.suptitle(
            f"Vision MoR eval — epoch {round(state.epoch)}, step {state.global_step}",
            fontsize=12,
        )
        plt.tight_layout()

        out_dir = os.path.join(args.output_dir, "vision_eval")
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(
            out_dir, f"epoch_{round(state.epoch):04d}_step_{state.global_step}.png"
        )
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[MultimodalVisionEvalCallback] Saved to {fig_path}")

        if self.cfg.get("wandb"):
            wandb.log(
                {"vision_eval/recursion_heatmap": wandb.Image(fig_path)},
                step=state.global_step,
            )

    def _plot_image_modality(self, ax_row, mod_name, sample_idx, raw_tokens, body_counts, plt):
        """Two panels for an image modality: token-ID grid (or Cosmos decode) + 2D heatmap."""
        import numpy as np
        decoded = self._try_decode_cosmos(
            torch.from_numpy(raw_tokens).long().unsqueeze(0)
        )
        if decoded is not None:
            ax_row[0].imshow(np.clip(decoded[0], 0.0, 1.0))
            ax_row[0].set_title(f"[{mod_name}] Cosmos decoded (sample {sample_idx})")
        else:
            tg = raw_tokens.astype(float).reshape(self.patch_grid_size, self.patch_grid_size)
            tg = (tg - tg.min()) / (tg.max() - tg.min() + 1e-8)
            ax_row[0].imshow(tg, cmap="viridis")
            ax_row[0].set_title(f"[{mod_name}] Token-ID grid (sample {sample_idx})")
        ax_row[0].axis("off")

        # Right: 2D spatial recursion heatmap
        heatmap = body_counts.reshape(self.patch_grid_size, self.patch_grid_size)
        im = ax_row[1].imshow(heatmap, cmap="hot", vmin=0, vmax=self.num_recursions)
        plt.colorbar(im, ax=ax_row[1], label="# recursions")
        ax_row[1].set_title(f"[{mod_name}] Recursion map (sample {sample_idx})")
        ax_row[1].axis("off")

    def _plot_text_modality(self, ax_row, mod_name, sample_idx, raw_tokens, body_counts, plt, np):
        """
        Two panels for a text modality.

        raw_tokens layout (after subtracting codebook_offset):
            [SOS_id, content_0, content_1, ..., content_N, EOS_id]
        body_counts mirrors this positionally.

        Left:  1D colour strip (one cell per token, coloured by recursion count).
        Right: bar chart of recursion counts with decoded text as subtitle.
        """
        n_body = len(body_counts)
        x = np.arange(n_body)

        # x-axis labels: SOS / positional index / EOS
        labels = (
            ["SOS"]
            + [str(k) for k in range(1, n_body - 1)]
            + (["EOS"] if n_body > 1 else [])
        )
        tick_step = max(1, n_body // 12)

        # Try to decode content tokens for a human-readable subtitle
        content_raw = raw_tokens[1:-1].tolist() if n_body > 2 else []
        tok = self._get_text_tokenizer_for_eval()
        if tok is not None and content_raw:
            try:
                decoded_text = tok.decode(content_raw, skip_special_tokens=True)
            except Exception:
                decoded_text = ""
        else:
            decoded_text = ""
        subtitle = f'"{decoded_text}"' if decoded_text else "(no text decoded)"

        # Left: 1D colour strip
        ax_row[0].imshow(
            body_counts.reshape(1, -1),
            cmap="hot", aspect="auto",
            vmin=0, vmax=self.num_recursions,
        )
        ax_row[0].set_xticks(x[::tick_step])
        ax_row[0].set_xticklabels(labels[::tick_step], rotation=45, ha="right", fontsize=7)
        ax_row[0].set_yticks([])
        ax_row[0].set_title(
            f"[{mod_name}] Recursion strip (sample {sample_idx})\n{subtitle}",
            fontsize=8,
        )

        # Right: bar chart, bars coloured by recursion fraction
        bar_colors = plt.cm.hot(body_counts / max(self.num_recursions, 1))
        ax_row[1].bar(x, body_counts, color=bar_colors)
        ax_row[1].set_xticks(x[::tick_step])
        ax_row[1].set_xticklabels(labels[::tick_step], rotation=45, ha="right", fontsize=7)
        ax_row[1].set_ylim(0, self.num_recursions + 0.5)
        ax_row[1].set_ylabel("# recursions")
        ax_row[1].set_title(
            f"[{mod_name}] Recursion per token (sample {sample_idx})\n{subtitle}",
            fontsize=8,
        )

    def _get_text_tokenizer_for_eval(self):
        if self._text_tok is not None:
            return self._text_tok
        try:
            from transformers import AutoTokenizer
            mm_cfg = self.cfg.get("multimodal", {})
            path = mm_cfg.get("text_tokenizer_path", "gpt2")
            tok = AutoTokenizer.from_pretrained(path)
            tok.add_special_tokens({"pad_token": "[PAD]"})
            tok.add_special_tokens({"bos_token": "[SOS]", "eos_token": "[EOS]"})
            self._text_tok = tok
        except Exception as exc:
            warnings.warn(f"[MultimodalVisionEvalCallback] Could not load text tokenizer: {exc}")
            self._text_tok = None
        return self._text_tok

    def _try_decode_cosmos(self, tokens):
        """
        Decode raw VQ token indices (not offset-shifted) → RGB float arrays [H, W, 3] in [0, 1].
        Requires: pip install cosmos-tokenizer  (NVIDIA Cosmos weights needed).
        Enable by setting vision_eval.cosmos_model_path in the config.
        tokens: LongTensor of shape [B, n_patches] with raw codebook indices.
        """
        if self._cosmos_model_path is None:
            return None
        try:
            from cosmos_tokenizer.video_lib import CausalVideoTokenizer
            tokenizer = CausalVideoTokenizer.from_pretrained(self._cosmos_model_path)
            B = tokens.shape[0]
            # Reshape to spatial grid: [B, 1, H, W] (1 frame for image)
            indices = tokens.reshape(B, 1, self.patch_grid_size, self.patch_grid_size)
            with torch.no_grad():
                decoded = tokenizer.decode(indices).float().cpu().numpy()  # [B, C, H, W]
            decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min() + 1e-8)
            return [decoded[i].transpose(1, 2, 0) for i in range(B)]   # list of HWC
        except Exception as exc:
            warnings.warn(f"[MultimodalVisionEvalCallback] Cosmos decode failed: {exc}")
            return None


class ScalingLawsSaveCallback(TrainerCallback):
    """
    This callback is used to save the model during scaling laws experiments.
    """
    def __init__(self, fixed_save_steps: str):
        super().__init__()
        self.fixed_save_steps = [int(step) for step in fixed_save_steps.split(",")]  # list
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in self.fixed_save_steps and state.global_step != 0:
            control.should_save = True         