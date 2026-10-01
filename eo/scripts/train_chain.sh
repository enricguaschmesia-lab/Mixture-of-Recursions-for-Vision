#!/usr/bin/env bash
# eo/scripts/train_chain.sh -- launch EO runs back to back on one GPU, unattended.
#
# Waits for a run that is already training to FINISH, then launches the next
# stage through train_eo.sh (still the only way a run starts), waits for that
# one, launches the next, and so on. Written for Phase 3 Step 11: A10 starts
# when the 50-epoch arm B releases the TITAN V, and B10 when A10 is done.
#
# WHAT "FINISHED" MEANS. The run's tmux session is gone AND its run directory
# holds both `checkpoint-<steps>` and the run-level `trainer_state.json` (HF
# writes the latter only at the end of training). A session that ended any
# other way is a crash, and the chain STOPS: launching the next arm onto the
# back of a failed one needs a human.
#
# ⚠ Sessions are matched EXACTLY (`tmux has-session -t =NAME`). A plain `-t`
#   matches by prefix; a watcher written that way once found itself and never
#   launched anything (worklog 2026-09-27/28).
#
# ⚠ It does not retry. If the card is not free within --gpu-wait minutes after
#   the previous run ends (the GPUs are shared with other projects), or if
#   preflight refuses, the chain logs why and stops. A retry loop around a
#   refusing preflight would eventually launch into whatever made it refuse.
#
# Usage:
#   bash eo/scripts/train_chain.sh --after RUN_ID:STEPS [--gpu titanv] \
#       [--gpu-wait MIN] ARM:CONFIG:STEPS [ARM:CONFIG:STEPS ...]
#
#   --after RUN_ID:STEPS   a run already training under $MOR_SAVE_DIR/pretrain/phase3/,
#                          and the step it ends at. Omit to launch the first stage now.
#   ARM:CONFIG:STEPS       one stage: train_eo.sh's --arm and --config, and the
#                          step the run ends at (its stop_steps).
#
# Example (Phase 3 Step 11), detached with a log:
#   tmux new -d -s chain_a10_b10 "bash eo/scripts/train_chain.sh \
#       --after vanilla_20260928_001221:16500 \
#       mor10:eo_terramesh/arm_a10_mor:3300 vanilla10:eo_terramesh/arm_b10_vanilla:3300 \
#       2>&1 | tee /data/enric/logs/chain_a10_b10.log"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

AFTER=""; GPU="titanv"; GPU_WAIT_MIN=60; POLL_S=60
STAGES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --after)    AFTER="$2"; shift 2 ;;
    --gpu)      GPU="$2"; shift 2 ;;
    --gpu-wait) GPU_WAIT_MIN="$2"; shift 2 ;;
    -h|--help)  sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *:*:*)      STAGES+=("$1"); shift ;;
    *)          echo "ERROR: unrecognised argument '$1'" >&2; exit 2 ;;
  esac
done
[[ ${#STAGES[@]} -gt 0 ]] || { echo "ERROR: at least one ARM:CONFIG:STEPS stage is required" >&2; exit 2; }

SAVE_ROOT="${MOR_SAVE_DIR:-/data/enric/runs}/pretrain/phase3"
PY="$REPO/.venv/bin/python"
# Same threshold preflight enforces (eo/train/preflight.py MIN_FREE_GPU_MIB).
NEED_MIB="$("$PY" -c 'from eo.train.preflight import MIN_FREE_GPU_MIB as m; print(m)')"
# Under PCI_BUS_ID ordering the CUDA index is the nvidia-smi index.
SMI_INDEX="$(CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" -m eo.train.gpu "$GPU")"

log() { echo "[$(date '+%F %T')] $*"; }

wait_for_run() {   # RUN_ID STEPS -- returns 1 if the run ended without finishing
  local run="$1" steps="$2" dir="$SAVE_ROOT/$1"
  log "waiting for $run to finish (checkpoint-$steps)"
  while tmux has-session -t "=$run" 2>/dev/null; do sleep "$POLL_S"; done
  if [[ -d "$dir/checkpoint-$steps" && -f "$dir/trainer_state.json" ]]; then
    log "$run finished: $dir/checkpoint-$steps"
    return 0
  fi
  log "STOP: session $run ended but $dir has no checkpoint-$steps + trainer_state.json -- crashed?"
  log "      see /data/enric/logs/$run.log. Not launching anything onto the back of it."
  return 1
}

wait_for_gpu() {
  local deadline=$(( $(date +%s) + GPU_WAIT_MIN * 60 )) used total free
  while :; do
    IFS=', ' read -r used total < <(nvidia-smi --id="$SMI_INDEX" \
        --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    if (( free >= NEED_MIB )); then log "$GPU free: $free MiB (need $NEED_MIB)"; return 0; fi
    if (( $(date +%s) >= deadline )); then
      log "STOP: $GPU still has only $free MiB free after $GPU_WAIT_MIN min (need $NEED_MIB)"
      nvidia-smi --id="$SMI_INDEX" --query-compute-apps=pid,used_memory,process_name --format=csv,noheader \
        | sed 's/^/        /'
      return 1
    fi
    sleep 30
  done
}

if [[ -n "$AFTER" ]]; then
  wait_for_run "${AFTER%%:*}" "${AFTER##*:}" || exit 1
fi

for stage in "${STAGES[@]}"; do
  IFS=: read -r arm config steps <<< "$stage"
  wait_for_gpu || exit 1
  run="${arm}_$(date +%Y%m%d_%H%M%S)"     # the id train_eo.sh would mint, made here so we can wait on it
  log "launching $run ($config, $steps steps) on $GPU"
  if ! bash eo/scripts/train_eo.sh --arm "$arm" --run-id "$run" --config "$config" \
        --gpu "$GPU" --detach; then
    log "STOP: train_eo.sh refused or failed for $run (preflight output above)"
    exit 1
  fi
  sleep 5
  tmux has-session -t "=$run" 2>/dev/null || { log "STOP: no tmux session $run after launch"; exit 1; }
  log "$run is training -- log /data/enric/logs/$run.log"
  wait_for_run "$run" "$steps" || exit 1
done
log "chain complete: ${#STAGES[@]} stage(s)"
