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
