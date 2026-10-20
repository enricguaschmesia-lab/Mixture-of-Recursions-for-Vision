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
#                     | a100 (EPFL RCP only, inside a Run:AI job; no --detach)
#   --gpus N          cards to train on (default 1). N > 1 launches pretrain.py under
#                     torchrun (DDP, one process per card) and needs N identical
#                     cards on one node: an RCP pod submitted with `-g N`. Not on the
#                     workstation, not for a BEN fine-tune (preflight refuses both).
#                     The batch must divide: total_batch_size % (per-device x N) == 0.
#                     A resume must use the --gpus (and per-device batch) it was saved with.
#   --config NAME     Hydra config name under conf/pretrain_vision/
#                     (default: eo_terramesh/terramesh_mor_token)
#   --detach          run in tmux, logging to /data/enric/logs/<run>.log
#   --resume          continue an existing run directory
#   --force           overwrite an existing run directory (destructive)
#   --run-id ID       reuse an exact run id instead of minting a timestamp
#                     (required with --resume)
#   --init-from DIR   Phase 6 fine-tune: start from an EXTENDED checkpoint
#                     (eo/scripts/extend_checkpoint.py). NOT a resume: fresh optimizer
#                     and scheduler, a strict load, and preflight verifies the
#                     extension's identity artifact. Runs go under phase6/.
#
# Examples:
#   bash eo/scripts/train_eo.sh --arm smoke stop_steps=30 total_batch_size=4 wandb=false
#   bash eo/scripts/train_eo.sh --arm mor --detach
#   bash eo/scripts/train_eo.sh --arm mor --resume --run-id mor_20260921_143000
#   bash eo/scripts/train_eo.sh --arm v360 --config eo_terramesh/arm_b10c_vanilla \
#        --gpu a100 --gpus 2 per_device_train_batch_size=16        (inside a `-g 2` pod)
#   bash eo/scripts/train_eo.sh --arm b10full --config eo_ben/b10_full --gpu titanx \
#        --init-from /data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300 --detach
#
# Environment overrides (RCP; unset on the workstation): MOR_PYTHON (the
# interpreter, default .venv/bin/python), MOR_LOG_DIR (default /data/enric/logs).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ARM=""; GPU="titanv"; NGPU=1; CONFIG="eo_terramesh/terramesh_mor_token"
DETACH=0; RESUME=0; FORCE=0; RUN_ID=""; INIT_FROM=""
HYDRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)     ARM="$2"; shift 2 ;;
    --gpu)     GPU="$2"; shift 2 ;;
    --gpus)    NGPU="$2"; shift 2 ;;
    --config)  CONFIG="$2"; shift 2 ;;
    --run-id)  RUN_ID="$2"; shift 2 ;;
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --detach)  DETACH=1; shift ;;
    --resume)  RESUME=1; shift ;;
    --force)   FORCE=1; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)         HYDRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$ARM" ]] || { echo "ERROR: --arm is required (it names the run)." >&2; exit 2; }
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --gpus must be a positive integer, got '$NGPU'." >&2; exit 2; }
if [[ $RESUME -eq 1 && -z "$RUN_ID" ]]; then
  echo "ERROR: --resume needs --run-id: which run directory should it continue?" >&2
  echo "       Available under \$MOR_SAVE_DIR/pretrain/phase3/:" >&2
  ls -1 "${MOR_SAVE_DIR:-/data/enric/runs}/pretrain/phase3" 2>/dev/null | sed 's/^/         /' >&2 || true
  exit 2
fi

PHASE_DIR="phase3"
[[ "$CONFIG" == eo_ben/* ]] && PHASE_DIR="phase6"      # every BEN run, the from-scratch reference included
if [[ -n "$INIT_FROM" ]]; then
  if [[ $RESUME -eq 1 ]]; then
    echo "ERROR: --init-from and --resume do not combine: a fine-tune is rerun, not resumed." >&2; exit 2
  fi
  [[ -d "$INIT_FROM" ]] || { echo "ERROR: --init-from $INIT_FROM is not a directory." >&2; exit 2; }
  INIT_FROM="$(cd "$INIT_FROM" && pwd)"
  HYDRA_ARGS+=("finetune.init_from=$INIT_FROM")
  PHASE_DIR="phase6"
fi

# --- 1. .env, without overriding anything already exported -------------------
# Every key named here is handed to a detached run explicitly (step 5).
ENV_KEYS=(MOR_SAVE_DIR TERRAMESH_TOK_ROOT HF_HOME WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_DIR)
if [[ -f "$REPO/.env" ]]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" != *"="* ]] && continue
    key="${line%%=*}"; key="${key// /}"
    ENV_KEYS+=("$key")
    [[ -n "${!key:-}" ]] && continue          # shell wins over .env
    export "${key}=${line#*=}"
  done < "$REPO/.env"
else
  echo "WARNING: no $REPO/.env -- preflight will report whatever is missing." >&2
fi

# --- 2 & 3. device order, then resolve the GPU by NAME -----------------------
# Pinned BEFORE resolving so the index we compute is the index CUDA will use.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY="${MOR_PYTHON:-$REPO/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: $PY not found. The EO training path runs in .venv (MOR_PYTHON on RCP)." >&2; exit 2; }
# On RCP the Run:AI job is the supervisor (restarts, logs); a tmux inside a pod dies with it.
if [[ "$GPU" == "a100" && $DETACH -eq 1 ]]; then
  echo "ERROR: --detach is for the workstation. On RCP, submit the run as a Run:AI job." >&2; exit 2
fi

if ! GPU_INDEX="$("$PY" -m eo.train.gpu "$GPU" --count "$NGPU")"; then
  echo "ERROR: could not resolve --gpu '$GPU' x $NGPU." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

# --- run identity ------------------------------------------------------------
# Timestamped, so re-running an arm after a fix cannot silently destroy a week
# of training. The refusal in preflight is the real guard; this makes it rare.
[[ -n "$RUN_ID" ]] || RUN_ID="${ARM}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PHASE_DIR}/${RUN_ID}"
RUN_DIR="${MOR_SAVE_DIR:-/data/enric/runs}/pretrain/${OUTPUT_DIR}"
LOG_DIR="${MOR_LOG_DIR:-/data/enric/logs}"
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
    --gpus "$NGPU" \
    --run-dir "$RUN_DIR" \
    --config "conf/pretrain_vision/${CONFIG}.yaml" \
    "${PRE_FLAGS[@]+"${PRE_FLAGS[@]}"}"

# --- 5. launch ---------------------------------------------------------------
# One card: plain python, as every run before 2026-10-20. N cards: torchrun, one process
# per card on this node; not `accelerate launch`, which would change what pretrain.py
# derives from the launcher (save_safetensors). See eo/train/distributed.py.
LAUNCH=("$PY")
if [[ $NGPU -gt 1 ]]; then
  LAUNCH=("$PY" -m torch.distributed.run --standalone --nproc_per_node "$NGPU")
fi
CMD=("${LAUNCH[@]}" pretrain.py
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
echo "gpu      : $GPU x $NGPU (CUDA_VISIBLE_DEVICES=$GPU_INDEX, order $CUDA_DEVICE_ORDER)"
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
  # `=` matches the name exactly; a bare -t matches by PREFIX.
  tmux has-session -t "=$RUN_ID" 2>/dev/null && {
    echo "ERROR: tmux session '$RUN_ID' already exists." >&2; exit 2; }
  # ⚠ Hand the environment over EXPLICITLY. A detached session gets the tmux
  #   SERVER's environment, not this shell's, so when a server is already
  #   running (any other session open) the CUDA_VISIBLE_DEVICES resolved above
  #   and every .env value are silently replaced by whatever that server was
  #   started with: preflight validates one GPU and the run lands on another.
  #   Measured 2026-10-01 -- a variable exported before `tmux new -d` read back
  #   unset inside the session; arms A and B were right only because each
  #   launch happened to start a fresh server.
  TMUX_ENV=(-e "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER" -e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  for key in $(printf '%s\n' "${ENV_KEYS[@]}" | sort -u); do
    [[ -n "${!key+set}" ]] && TMUX_ENV+=(-e "$key=${!key}")
  done
  tmux new -d -s "$RUN_ID" "${TMUX_ENV[@]}" \
    "cd '$REPO' && $(printf '%q ' "${CMD[@]}") 2>&1 | tee '$LOG_FILE'"
  echo "detached into tmux session '$RUN_ID'"
  echo "  follow : tail -f $LOG_FILE"
  echo "  attach : tmux attach -t $RUN_ID"
else
  exec "${CMD[@]}"
fi
