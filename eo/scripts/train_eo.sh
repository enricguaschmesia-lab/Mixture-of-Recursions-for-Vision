#!/usr/bin/env bash
# eo/scripts/train_eo.sh -- the ONLY supported way to start an EO training run.
#
# WHY NOT scripts/pretrain.sh: that one launches through accelerate or
# deepspeed. The EO path is deliberately plain single-GPU `python` (deepspeed
# must be null -- util.misc.get_launcher_type() returns "deepspeed" for anything
# that is not an accelerate launch, so a non-null value is honoured on every
# run and fails with a PackageNotFoundError that never mentions the config).
#
# WHAT THIS DOES, in order:
#   1. loads .env                    (values already in the shell win)
#   2. pins CUDA_DEVICE_ORDER=PCI_BUS_ID so nvidia-smi and CUDA agree
#   3. resolves --gpu BY NAME to an index          (eo/train/gpu.py)
#   4. runs every environment check, and REFUSES TO START if any fails
#                                                   (eo/train/preflight.py)
#   5. launches, optionally detached into tmux with a log on /data
#
# The checks are in Python, not here, so the phase gate can import them and
# prove each one can fail. This file stays thin on purpose.
#
# Usage:
#   bash eo/scripts/train_eo.sh --arm mor [options] [hydra overrides...]
#
#   --arm NAME        required. Names the run, the W&B run, and the output dir.
#   --gpu ALIAS       titanv (default, the training card) | titanx
#   --config NAME     Hydra config name under conf/pretrain_vision/
#                     (default: eo_terramesh/terramesh_mor_token)
#   --detach          run in tmux, logging to /data/enric/logs/<run>.log
#   --resume          continue an existing run directory
#   --force           overwrite an existing run directory (destructive)
#   --run-id ID       reuse an exact run id instead of minting a timestamp
#                     (required with --resume)
#
# Examples:
#   bash eo/scripts/train_eo.sh --arm smoke stop_steps=30 total_batch_size=4 wandb=false
#   bash eo/scripts/train_eo.sh --arm mor --detach
#   bash eo/scripts/train_eo.sh --arm mor --resume --run-id mor_20260921_143000
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ARM=""; GPU="titanv"; CONFIG="eo_terramesh/terramesh_mor_token"
DETACH=0; RESUME=0; FORCE=0; RUN_ID=""
HYDRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)     ARM="$2"; shift 2 ;;
    --gpu)     GPU="$2"; shift 2 ;;
    --config)  CONFIG="$2"; shift 2 ;;
    --run-id)  RUN_ID="$2"; shift 2 ;;
    --detach)  DETACH=1; shift ;;
    --resume)  RESUME=1; shift ;;
    --force)   FORCE=1; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)         HYDRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$ARM" ]] || { echo "ERROR: --arm is required (it names the run)." >&2; exit 2; }
if [[ $RESUME -eq 1 && -z "$RUN_ID" ]]; then
  echo "ERROR: --resume needs --run-id: which run directory should it continue?" >&2
  echo "       Available under \$MOR_SAVE_DIR/pretrain/phase3/:" >&2
  ls -1 "${MOR_SAVE_DIR:-/data/enric/runs}/pretrain/phase3" 2>/dev/null | sed 's/^/         /' >&2 || true
  exit 2
fi

# --- 1. .env, without overriding anything already exported -------------------
if [[ -f "$REPO/.env" ]]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" != *"="* ]] && continue
    key="${line%%=*}"; key="${key// /}"
    [[ -n "${!key:-}" ]] && continue          # shell wins over .env
    export "${key}=${line#*=}"
  done < "$REPO/.env"
else
  echo "WARNING: no $REPO/.env -- preflight will report whatever is missing." >&2
fi

# --- 2 & 3. device order, then resolve the GPU by NAME -----------------------
# Pinned BEFORE resolving so the index we compute is the index CUDA will use.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY not found. The EO training path runs in .venv." >&2; exit 2; }

if ! GPU_INDEX="$("$PY" -m eo.train.gpu "$GPU")"; then
  echo "ERROR: could not resolve --gpu '$GPU'." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

# --- run identity ------------------------------------------------------------
# Timestamped, so re-running an arm after a fix cannot silently destroy a week
# of training. The refusal in preflight is the real guard; this makes it rare.
[[ -n "$RUN_ID" ]] || RUN_ID="${ARM}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="phase3/${RUN_ID}"
RUN_DIR="${MOR_SAVE_DIR:-/data/enric/runs}/pretrain/${OUTPUT_DIR}"
LOG_DIR="/data/enric/logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

PRE_FLAGS=()
if [[ $RESUME -eq 1 || $FORCE -eq 1 ]]; then PRE_FLAGS+=(--allow-existing); fi
if [[ $RESUME -eq 1 ]]; then PRE_FLAGS+=(--resuming); fi
# Forward the Hydra overrides so the config check validates what actually runs,
# not the file on disk -- `mor.enable=false` on the command line changes the arm.
for _ov in "${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}"; do
  [[ "$_ov" == *"="* ]] && PRE_FLAGS+=(--override "$_ov")
done

# --- 4. preflight: refuse to start if anything is wrong ----------------------
"$PY" -m eo.train.preflight \
    --gpu "$GPU" \
    --run-dir "$RUN_DIR" \
    --config "conf/pretrain_vision/${CONFIG}.yaml" \
    "${PRE_FLAGS[@]+"${PRE_FLAGS[@]}"}"

# --- 5. launch ---------------------------------------------------------------
CMD=("$PY" pretrain.py
     --config-path conf/pretrain_vision
     --config-name "$CONFIG"
     "name=${RUN_ID}"
     "output_dir=${OUTPUT_DIR}"
     "wandb_run_name=${RUN_ID}")
if [[ $RESUME -eq 1 ]]; then CMD+=("resume_from_checkpoint=true"); fi
CMD+=("${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}")

echo
echo "run      : $RUN_ID"
echo "arm      : $ARM"
echo "gpu      : $GPU (CUDA_VISIBLE_DEVICES=$GPU_INDEX, order $CUDA_DEVICE_ORDER)"
echo "config   : $CONFIG"
echo "run dir  : $RUN_DIR"
echo "wandb    : ${WANDB_ENTITY:-?}/${WANDB_PROJECT:-?} (${WANDB_MODE:-online})"
echo "command  : ${CMD[*]}"
echo

if [[ $DETACH -eq 1 ]]; then
  # tmux + a log on /data is the project's long-job convention, and it is not
  # optional: it survived two dropped connections during the Phase 1 Step 4 run.
  # Doing it here means nobody has to remember it.
  mkdir -p "$LOG_DIR"
  command -v tmux >/dev/null || { echo "ERROR: tmux not installed." >&2; exit 2; }
  tmux has-session -t "$RUN_ID" 2>/dev/null && {
    echo "ERROR: tmux session '$RUN_ID' already exists." >&2; exit 2; }
  tmux new -d -s "$RUN_ID" \
    "cd '$REPO' && $(printf '%q ' "${CMD[@]}") 2>&1 | tee '$LOG_FILE'"
  echo "detached into tmux session '$RUN_ID'"
  echo "  follow : tail -f $LOG_FILE"
  echo "  attach : tmux attach -t $RUN_ID"
else
  exec "${CMD[@]}"
fi
