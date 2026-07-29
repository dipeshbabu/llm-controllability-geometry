#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}

MODEL_NAME=${1:?usage: run_gemma_scope_study.sh MODEL_NAME SLUG}
SLUG=${2:?usage: run_gemma_scope_study.sh MODEL_NAME SLUG}
RUN_ROOT=${RUN_ROOT:-runs/controllability}
DEVICE=${DEVICE:-auto}

uv run llm-controllability gemma-scope \
  --states-dir "${RUN_ROOT}/${SLUG}/reachable" \
  --out-dir "${RUN_ROOT}/${SLUG}/gemma_scope" \
  --model-name "${MODEL_NAME}" \
  --direction-sweep \
    "${RUN_ROOT}/${SLUG}/directions/${SLUG}_eval_awareness_layer_sweep.csv" \
  --site resid_post_all \
  --width 16k \
  --l0 small \
  --device "${DEVICE}" \
  --top-k 128 \
  --analysis-features 2048 \
  --batch-size 32 \
  --max-samples 4096
