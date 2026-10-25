#!/usr/bin/env bash
# Workstation driver for the train-split tokenization (2026-10-25): a downloader and an encoder,
# both resumable per shard (eo/scripts/tokenize_train.py). The encoder runs on the TITAN V under the
# POWER WATCHDOG of run_generate_v2.sh: under 45 W for 180 s while it is encoding (not waiting for a
# download, see control/encoder_state.<pid>) means the stall signature, so it is aborted with a
# faulthandler stack and restarted; a restart redoes one shard at most.
#   tmux new -d -s tok_train "bash eo/scripts/run_tokenize_train.sh 0:894 2>&1 | tee -a /data/enric/logs/tok_train.log"
#   stop cleanly: touch /data/enric/data/TerraMesh/train_tok/control/STOP
set -uo pipefail
POS=${1:-0:894}
REPO=/home/enric/code/Mixture-of-Recursions-for-Vision
OUT=/data/enric/data/TerraMesh/train_tok
WORK=/data/enric/data/TerraMesh/train_raw
REV=01f5f9d84fe87f23e385a0507bf6d178fff21ca8
source /data/enric/miniforge3/etc/profile.d/conda.sh && conda activate mor || exit 2
cd "$REPO" || exit 2
# set INSIDE the session: a detached tmux does not inherit the launching shell's environment
export HF_HOME=/data/enric/hf CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1   # TITAN V; asserted below
log() { echo "[$(date '+%F %T')] $*"; }
log "train tokenization, positions $POS, code $(git rev-parse --short HEAD) + working tree"

downloader() {
  for attempt in $(seq 1 100); do
    python -u eo/scripts/tokenize_train.py download --out "$OUT" --work "$WORK" --positions "$POS" \
        --revision "$REV" --ahead 4 --streams 8 && return 0
    log "[dl] downloader exited rc=$?; restart $attempt in 60 s"; sleep 60
  done
}

watchdog() {   # nvidia-smi lists by PCI order; find the TITAN V by NAME, never by index
  local low=0
  while true; do
    sleep 15
    local pid; pid=$(cat "$OUT/control/encoder.pid" 2>/dev/null)
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || { low=0; continue; }
    local st; st=$(cat "$OUT/control/encoder_state.$pid" 2>/dev/null)
    local w; w=$(nvidia-smi --query-gpu=name,power.draw --format=csv,noheader,nounits | awk -F', ' '/TITAN V/{printf "%d", $2}')
    if [[ "$st" == encode* && -n "$w" && "$w" -lt 45 ]]; then low=$((low + 15)); else low=0; fi
    if [[ $low -ge 180 ]]; then
      log "WATCHDOG: TITAN V at ${w} W for ${low} s while encoding ($st); aborting pid $pid"
      kill -ABRT "$pid"; sleep 30; kill -KILL "$pid" 2>/dev/null; low=0
    fi
  done
}

mkdir -p "$OUT/control"
downloader > >(sed -u 's/^/[dl] /') 2>&1 &
DL=$!                      # the subshell running downloader(), so pkill -P reaches its python
watchdog &
WD=$!
trap 'kill $WD 2>/dev/null; pkill -P $DL 2>/dev/null; kill $DL 2>/dev/null' EXIT

for attempt in $(seq 1 50); do
  python -X faulthandler -u eo/scripts/tokenize_train.py encode --out "$OUT" --work "$WORK" \
      --positions "$POS" --expect-gpu "TITAN V" &
  EP=$!
  echo "$EP" > "$OUT/control/encoder.pid"
  wait "$EP"; rc=$?
  rm -f "$OUT/control/encoder_state.$EP"
  [[ $rc -eq 0 ]] && { log "encoder finished (all positions done, or STOP)"; break; }
  log "encoder rc=$rc (134/137 = watchdog or abort; stack above); restart $attempt"
  sleep 10
done
wait "$DL"
log "done"
