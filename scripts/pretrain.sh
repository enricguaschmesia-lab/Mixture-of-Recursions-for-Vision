#!/bin/bash

# -----------------------------------------------------------------------------
# Launch script for MoR vision/multimodal pretraining.
# This script is the recommended entry point for training runs. It launches pretrain.py through either Hugging Face Accelerate or DeepSpeed and passes the
# selected Hydra config explicitly with:

# pretrain.py --config-name <config_name>

# Usage:

# bash scripts/pretrain.sh [accelerate|deepspeed] [online|offline] <gpu_ids> <config_name_1> [config_name_2 ...]

# Examples:

# bash scripts/pretrain.sh accelerate offline 0 my_config
# bash scripts/pretrain.sh accelerate online 0,1 config_a config_b
# bash scripts/pretrain.sh deepspeed offline 0,1,2,3 my_config

# Arguments:

# - launcher type: optional, either "accelerate" or "deepspeed".
# Defaults to "accelerate" if omitted.
# - W&B mode: optional, either "online" or "offline".
# Exported as WANDB_MODE before launching training.
# - gpu_ids: comma-separated GPU IDs, e.g. "0" or "0,1".
# - config_name(s): Hydra config names from conf/pretrain_vision/.

# All arguments after <gpu_ids> are interpreted as config names.

# -----------------------------------------------------------------------------

launcher_type="accelerate"

if [[ "$1" == "deepspeed" ]] || [[ "$1" == "accelerate" ]]; then
  launcher_type="$1"
  shift
fi

if [[ "$1" == "online" ]] || [[ "$1" == "offline" ]]; then
  user_specified_run_mode="$1"
  shift
fi

if [ -n "$user_specified_run_mode" ]; then
  export WANDB_MODE="$user_specified_run_mode"
else
  export WANDB_MODE="online" # Default value if not specified
fi
echo "INFO: WANDB_MODE is set to '$WANDB_MODE'"

# Use the first argument as GPU numbers.
gpu_numbers="$1"

# Use all arguments after the first one as config-names.
shift # Remove the first argument

# Check if GPU numbers are provided.
if [ -z "$gpu_numbers" ]; then
  echo "Usage: $0 <GPU numbers> <config name 1> <config name 2> ..."
  exit 1
fi
num_processes=$(echo "$gpu_numbers" | tr ',' '\n' | grep -c .)

# Check if at least one config-name argument is provided.
if [ $# -eq 0 ]; then
  echo "Usage: $0 <GPU numbers> <config name 1> <config name 2> ..."
  exit 1
fi

# Loop through each config-name argument and execute the command.
for config_name in "$@"; do
  # Generate a 5-digit random port.
  random_port=$((10000 + RANDOM % 90000))

  # Execute the command.
  echo "Running with config: $config_name, GPUs: $gpu_numbers, Port: $random_port"

  if [ "$launcher_type" == "deepspeed" ]; then
    echo "Launch with DeepSpeed..."
    HYDRA_FULL_ERROR=1 deepspeed --include "localhost:$gpu_numbers" --no_local_rank --master_port "$random_port" pretrain.py --config-name "$config_name"
  elif [ "$launcher_type" == "accelerate" ]; then
    echo "Launch with accelerate..."
    if [ "$num_processes" -eq 1 ]; then
      # Use single GPU config for single process.
      HYDRA_FULL_ERROR=1 accelerate launch --config_file acc_configs/single_gpu_config.yaml --gpu_ids $gpu_numbers --num_processes $num_processes --main_process_port "$random_port" pretrain.py --config-name "$config_name"
    else
      # Use default config for multiple processes.
      HYDRA_FULL_ERROR=1 accelerate launch --config_file acc_configs/default_config.yaml --gpu_ids $gpu_numbers --num_processes $num_processes --main_process_port "$random_port" pretrain.py --config-name "$config_name"
    fi
  fi
  # Optional: Add a delay after each config execution.
  # sleep 5 # Wait for 5 seconds.
done

echo "All configurations processed."