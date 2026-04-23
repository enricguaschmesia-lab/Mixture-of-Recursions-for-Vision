import os 

import pickle

import warnings
import wandb

import torch
import numpy as np
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import CallbackHandler
from paths import PROJECT_ROOT


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
        self._cosmos_decoder = None
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
        except Exception as exc:
            import traceback, sys
            print(f"[MultimodalVisionEvalCallback] Cosmos decode failed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            return None


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

        # Capture routing decisions from each MoR layer.
        # Expert-choice populates `selected_tokens` ([bs, top_k, 1]) once per recursion wrapper.
        # Token-choice populates `token_expert_indices` ([bs, seq_len]) once per forward.
        selected_tokens_list = []
        token_expert_indices_list = []

        hooks = []
        for layer in m.model.layers:
            if getattr(layer, "mor", False):
                def _hook(module, inp, out, _sel=selected_tokens_list, _tei=token_expert_indices_list):
                    if getattr(out, "selected_tokens", None) is not None:
                        _sel.append(out.selected_tokens.detach().cpu())
                    if getattr(out, "token_expert_indices", None) is not None:
                        _tei.append(out.token_expert_indices.detach().cpu())
                hooks.append(layer.register_forward_hook(_hook))

        model.eval()
        with torch.no_grad():
            m(input_ids=input_ids, attention_mask=attention_mask)
        for h in hooks:
            h.remove()
        model.train()


        # Build per-position recursion-count map  [n, seq_len].
        # Semantics: count_map[b, t] = number of recursion passes token t underwent in sample b.
        # Both branches produce values in {1, ..., Nr} for active tokens (and 0 if never selected,
        # which can happen under expert-choice but not under token-choice).
        seq_len = input_ids.shape[1]
        count_map = torch.zeros(n, seq_len)

        if len(token_expert_indices_list) > 0:
            # ---- Token-choice branch ----
            # Each token commits to one depth index i in {0, ..., Nr-1}, meaning i+1 recursions.
            # If multiple token-choice wrappers exist (e.g. per-modality routers in Milestone II),
            # we take the last non-zero assignment per position; adapt here if you split by modality.
            tei = token_expert_indices_list[-1]   # [n, seq_len], long tensor
            count_map = (tei + 1).float()
        elif len(selected_tokens_list) > 0:
            # ---- Expert-choice branch (original logic) ----
            for st in selected_tokens_list:           # st: [n, top_k, 1]
                st_flat = st.squeeze(-1)              # [n, top_k]
                for b in range(n):
                    count_map[b].scatter_add_(0, st_flat[b], torch.ones(st_flat.shape[1]))
        else:
            warnings.warn(
                "[MultimodalVisionEvalCallback] No routing info captured from MoR layers; "
                "count_map will be all zeros. Check that the model actually has MoR layers "
                "and that the router populates `selected_tokens` or `token_expert_indices`."
            )
            
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
        """Two panels: (left) clean Cosmos reconstruction, (right) reconstruction + recursion overlay."""
        import numpy as np
        from matplotlib.colors import BoundaryNorm, ListedColormap
        
        # ---- Base image: Cosmos decode, or normalised token-ID grid as fallback ----
        decoded = self._try_decode_cosmos(
            torch.from_numpy(raw_tokens).long().unsqueeze(0)
        )
        if decoded is not None:
            base_img = np.clip(decoded[0], 0.0, 1.0)        # [H, W, 3]
            base_label = f"[{mod_name}] Cosmos reconstruction (sample {sample_idx})"
        else:
            g = raw_tokens.astype(float).reshape(self.patch_grid_size, self.patch_grid_size)
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)
            base_img = np.stack([g, g, g], axis=-1)          # grayscale -> RGB for consistency
            base_label = f"[{mod_name}] Token-ID grid (sample {sample_idx})"

        H_img, W_img = base_img.shape[:2]
        extent = [0, W_img, H_img, 0]                        # align (0,0) at top-left

        # ---- Discrete colormap for recursion counts {0, 1, ..., num_recursions} ----
        n_levels = self.num_recursions + 1                   # e.g. 4 levels for num_recursion=3
        palette = plt.cm.viridis(np.linspace(0.15, 0.95, n_levels))
        cmap = ListedColormap(palette)
        bounds = np.arange(n_levels + 1) - 0.5               # [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = BoundaryNorm(bounds, cmap.N)

        heatmap = body_counts.reshape(self.patch_grid_size, self.patch_grid_size)

        # ---- Left: clean reconstruction ----
        ax_row[0].imshow(base_img, extent=extent)
        ax_row[0].set_title(base_label, fontsize=9)
        ax_row[0].axis("off")

        # ---- Right: reconstruction + semi-transparent recursion overlay ----
        ax_row[1].imshow(base_img, extent=extent)
        im = ax_row[1].imshow(
            heatmap, cmap=cmap, norm=norm,
            alpha=0.55, interpolation="nearest",
            extent=extent,
        )
        cbar = plt.colorbar(im, ax=ax_row[1], ticks=np.arange(n_levels))
        cbar.set_label("# recursions")
        ax_row[1].set_title(
            f"[{mod_name}] Reconstruction + recursion map (sample {sample_idx})",
            fontsize=9,
        )
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
        Decode raw VQ token indices → RGB arrays [H, W, 3] in [0, 1].
        Expects self._cosmos_model_path to be a *directory* containing decoder.jit.
        tokens: LongTensor of shape [B, n_patches] with raw codebook indices.
        """
        if self._cosmos_model_path is None:
            return None
        try:
            if self._cosmos_decoder is None:
                from cosmos_tokenizer.image_lib import ImageTokenizer
                decoder_jit = os.path.join(PROJECT_ROOT, self._cosmos_model_path, "decoder.jit")
                self._cosmos_decoder = ImageTokenizer(checkpoint_dec=decoder_jit)
            decoder = self._cosmos_decoder


            B = tokens.shape[0]
            # DI tokenizers expect [B, H_tok, W_tok]. cosmos decoder wants uint16 indices.
            indices = tokens.reshape(B, self.patch_grid_size, self.patch_grid_size).to(torch.int32).cuda()

            #indices = indices.to(torch.uint16).cuda()

            with torch.no_grad():
                # Decoder returns [B, C, H, W] in roughly [-1, 1].
                decoded = decoder.decode(indices).float().cpu().numpy()
            decoded = np.clip((decoded + 1.0) / 2.0, 0.0, 1.0)  # -> [0, 1]
            return [decoded[i].transpose(1, 2, 0) for i in range(B)]  # list of HWC
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