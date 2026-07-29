#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${1:?usage: run_mps_model_controllability.sh MODEL_NAME SLUG [DTYPE] [PROTOCOL]}
SLUG=${2:?usage: run_mps_model_controllability.sh MODEL_NAME SLUG [DTYPE] [PROTOCOL]}
DTYPE=${3:-float16}
PROTOCOL=${4:-full}

export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export DEVICE_MAP=mps
export MONITOR_DEVICE=mps
export BATCH_SIZE=${BATCH_SIZE:-1}
export DIRECTION_BATCH_SIZE=${DIRECTION_BATCH_SIZE:-1}

exec bash scripts/run_model_controllability.sh \
  "${MODEL_NAME}" \
  "${SLUG}" \
  "${DTYPE}" \
  eager \
  "${PROTOCOL}"
