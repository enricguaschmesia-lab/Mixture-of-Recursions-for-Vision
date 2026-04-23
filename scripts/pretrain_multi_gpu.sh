#!/bin/bash
#SBATCH --job-name=mor_multinode
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1      # one srun task per node; accelerate spawns per-GPU procs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# === Args ===
CONFIG_NAME=$1        # Hydra config name, e.g. 250720_pretrain_smollm-135m_rec3_...
WANDB_KEY=$2

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: sbatch $0 <hydra-config-name> <wandb-api-key>"
    exit 1
fi

set -x
cat "$0"

# === Env ===
export WANDB_API_KEY=$WANDB_KEY
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Helpful on many clusters; uncomment/adjust if NCCL complains:
# export NCCL_SOCKET_IFNAME=^docker,lo
# export NCCL_IB_DISABLE=0

# === Rendezvous (first node in the allocation) ===
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=25678

GPUS_PER_NODE=2
NUM_NODES=$SLURM_NNODES
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

echo "MASTER_ADDR=$MASTER_ADDR  PORT=$MASTER_PORT  WORLD_SIZE=$WORLD_SIZE  NODES=$NUM_NODES"
mkdir -p logs

# === Launch: one task per node, accelerate fans out to local GPUs ===
# SLURM_PROCID is the node rank because ntasks-per-node=1.
srun --kill-on-bad-exit=1 bash -c '
  accelerate launch \
    --num_machines='"$NUM_NODES"' \
    --num_processes='"$WORLD_SIZE"' \
    --machine_rank=$SLURM_PROCID \
    --main_process_ip='"$MASTER_ADDR"' \
    --main_process_port='"$MASTER_PORT"' \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    pretrain.py --config-name='"$CONFIG_NAME"'
'