#!/usr/bin/env bash
# eo/scripts/sweep_d311.sh -- D3.11's y-axis: generate + decode every checkpoint.
#
# For each checkpoint and each of the six decodable targets, runs the three
# existing Step 6/7 stages in order, unchanged:
#
#   generate_eo.py   (.venv)  64 corpus-stratified held-out scenes, slot-masked
#   prepare_decode.py (.venv) global ids -> local codebook grids
#   decode_eo.py     (mor)    grids -> pixels -> metrics_<T>.json (+ ceiling
#                             and shuffled control over the SAME scenes)
#
# ⚠ THE PROTOCOL IS FIXED HERE SO BOTH ARMS GET THE SAME ONE. 64 scenes,
# slot-masked, sampled at temperature 1.0, seed 42, generation batch 8; decode
# seed 0, batch 4, 50 timesteps, |z| < 1.0. Arm B must be swept with this
# script and these defaults, or the D3.11 curves are not comparable.
#
# Resumable: a (checkpoint, target) whose metrics file exists is skipped, so a
# killed sweep is restarted by re-running the same command.
#
# Usage:
#   bash eo/scripts/sweep_d311.sh --gpu titanx --arm-tag arm_a \
#       --arm-config eo_terramesh/arm_a_mor \
#       --run-dir /data/enric/runs/pretrain/phase3/mor_20260923_140238 \
#       checkpoint-16500 checkpoint-2000 ...
#
# Outputs:
#   /data/enric/generations/d311/<arm-tag>/<ckpt>/<T>/slot_masked/
#   /data/enric/reports/d311/<arm-tag>/<ckpt>/<T>_<arm-tag>_<ckpt>/metrics_<T>.json
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

GPU=""; ARM_TAG=""; ARM_CONFIG=""; RUN_DIR=""
TARGETS="LULC DEM S2L2A NDVI S1RTC S1GRD"
N_SCENES=64
CKPTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)        GPU="$2"; shift 2 ;;
    --arm-tag)    ARM_TAG="$2"; shift 2 ;;
    --arm-config) ARM_CONFIG="$2"; shift 2 ;;
    --run-dir)    RUN_DIR="$2"; shift 2 ;;
    --targets)    TARGETS="$2"; shift 2 ;;
    -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            CKPTS+=("$1"); shift ;;
  esac
done
[[ -n "$GPU" && -n "$ARM_TAG" && -n "$ARM_CONFIG" && -n "$RUN_DIR" && ${#CKPTS[@]} -gt 0 ]] || {
  echo "ERROR: --gpu, --arm-tag, --arm-config, --run-dir and >=1 checkpoint are required" >&2; exit 2; }

# .env, without overriding anything already exported (same rule as train_eo.sh)
if [[ -f "$REPO/.env" ]]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# || "$line" =~ ^[[:space:]]*$ || "$line" != *"="* ]] && continue
    key="${line%%=*}"; key="${key// /}"
    [[ -n "${!key:-}" ]] && continue
    export "${key}=${line#*=}"
  done < "$REPO/.env"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
VENV="$REPO/.venv/bin/python"
MOR="/data/enric/miniforge3/envs/mor/bin/python"     # explicit: never trust PATH here
GPU_INDEX="$("$VENV" -m eo.train.gpu "$GPU")" || { echo "ERROR: cannot resolve --gpu $GPU" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
ALLOW=(); [[ "$GPU" == "titanv" ]] && ALLOW=(--allow-titanv)

echo "sweep: $ARM_TAG on $GPU (CUDA_VISIBLE_DEVICES=$GPU_INDEX)  targets: $TARGETS  scenes: $N_SCENES"
for ck in "${CKPTS[@]}"; do
  CK_PATH="$RUN_DIR/$ck"
  [[ -d "$CK_PATH" ]] || { echo "ERROR: no checkpoint $CK_PATH" >&2; exit 2; }
  for T in $TARGETS; do
    GEN="/data/enric/generations/d311/$ARM_TAG/$ck/$T"
    OUT_ROOT="/data/enric/reports/d311/$ARM_TAG/$ck"
    METRICS="$OUT_ROOT/${T}_${ARM_TAG}_${ck}/metrics_${T}.json"
    if [[ -f "$METRICS" ]]; then echo "[skip] $ck $T"; continue; fi
    echo "=== $(date '+%F %T')  $ck  $T"
    "$VENV" eo/scripts/generate_eo.py --arm-config "$ARM_CONFIG" --checkpoint "$CK_PATH" \
        --target "$T" --n-scenes "$N_SCENES" --slot-masked-only --out "$GEN" || { echo "generate FAILED $ck $T"; continue; }
    "$VENV" eo/scripts/prepare_decode.py --gen-dir "$GEN/slot_masked" --target "$T" || { echo "prepare FAILED $ck $T"; continue; }
    "$MOR" eo/scripts/decode_eo.py --decode-dir "$GEN/slot_masked" --target "$T" \
        --tag "${ARM_TAG}_${ck}" --out-root "$OUT_ROOT" "${ALLOW[@]}"
    rc=$?
    # 2 = the shuffled control did not collapse on this sample: recorded in the
    # artifact, not fatal to the sweep.
    [[ $rc -eq 0 || $rc -eq 2 ]] || echo "decode FAILED ($rc) $ck $T"
  done
done
echo "sweep done $(date '+%F %T')"
