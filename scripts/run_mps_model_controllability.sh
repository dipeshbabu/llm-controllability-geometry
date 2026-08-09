#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${1:?usage: run_mps_model_controllability.sh MODEL_NAME SLUG [DTYPE] [PROTOCOL]}
SLUG=${2:?usage: run_mps_model_controllability.sh MODEL_NAME SLUG [DTYPE] [PROTOCOL]}
DTYPE=${3:-float16}
PROTOCOL=${4:-pilot}

export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export DEVICE_MAP=mps
export MONITOR_DEVICE=mps
export BATCH_SIZE=${BATCH_SIZE:-1}
if [[ -z "${DIRECTION_BATCH_SIZE:-}" ]]; then
  if [[ "${MODEL_NAME}" == "Qwen/Qwen3-8B" ]]; then
    export DIRECTION_BATCH_SIZE=2
  else
    export DIRECTION_BATCH_SIZE=1
  fi
fi

exec bash scripts/run_model_controllability.sh \
  "${MODEL_NAME}" \
  "${SLUG}" \
  "${DTYPE}" \
  eager \
  "${PROTOCOL}"
