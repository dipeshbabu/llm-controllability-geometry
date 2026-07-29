#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}

MODEL_NAME=${1:?usage: run_token_monitor_study.sh MODEL_NAME SLUG [DTYPE] [ATTN_IMPLEMENTATION]}
SLUG=${2:?usage: run_token_monitor_study.sh MODEL_NAME SLUG [DTYPE] [ATTN_IMPLEMENTATION]}
DTYPE=${3:-bfloat16}
ATTN_IMPLEMENTATION=${4:-}
MODEL_REVISION=${MODEL_REVISION:-}
DEVICE_MAP=${DEVICE_MAP:-auto}
MONITOR_DEVICE=${MONITOR_DEVICE:-auto}

DATA_ROOT=${DATA_ROOT:-data/frontier}
RUN_ROOT=${RUN_ROOT:-runs/controllability}
DIRECTION_NAME="${SLUG}_eval_awareness"
SWEEP_CSV="${RUN_ROOT}/${SLUG}/directions/${DIRECTION_NAME}_layer_sweep.csv"
PROMPT_CONTROLS="${RUN_ROOT}/${SLUG}/optimized_prompt_controls.txt"
TOKEN_SPEC="${RUN_ROOT}/${SLUG}/token_monitor_study.json"
TOKEN_OUT="${RUN_ROOT}/${SLUG}/token_reachable"

ATTN_ARGS=()
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARGS=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi
REVISION_ARGS=()
if [[ -n "${MODEL_REVISION}" ]]; then
  REVISION_ARGS=(--revision "${MODEL_REVISION}")
fi

uv run llm-controllability build-study-spec \
  --out "${TOKEN_SPEC}" \
  --model-name "${MODEL_NAME}" \
  "${REVISION_ARGS[@]}" \
  "${ATTN_ARGS[@]}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${DTYPE}" \
  --data "${DATA_ROOT}/behavior/controllability_all.jsonl" \
  --example-limit 64 \
  --direction-sweep "${SWEEP_CSV}" \
  --layers target \
  --prompt-controls "${PROMPT_CONTROLS}" \
  --natural-controls "${DATA_ROOT}/controls/prompt_rewrites.json" \
  --semantic-model sentence-transformers/all-mpnet-base-v2 \
  --store-token-states

uv run llm-controllability collect-reachable \
  --spec "${TOKEN_SPEC}" \
  --out "${TOKEN_OUT}"

uv run llm-controllability monitor-invariance \
  --states-dir "${TOKEN_OUT}" \
  --out-dir "${RUN_ROOT}/${SLUG}/token_monitors" \
  --monitors last_linear mean_linear max_linear attention \
  --monitor-device "${MONITOR_DEVICE}" \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --seed 0
