#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}

MODEL_NAME=${1:?usage: run_model_controllability.sh MODEL_NAME SLUG [DTYPE] [ATTN_IMPLEMENTATION] [PROTOCOL]}
SLUG=${2:?usage: run_model_controllability.sh MODEL_NAME SLUG [DTYPE] [ATTN_IMPLEMENTATION] [PROTOCOL]}
DTYPE=${3:-bfloat16}
ATTN_IMPLEMENTATION=${4:-}
PROTOCOL=${5:-full}

DATA_ROOT=${DATA_ROOT:-data/frontier}
RUN_ROOT=${RUN_ROOT:-runs/controllability}

if [[ -z "${BATCH_SIZE:-}" ]]; then
  case "${MODEL_NAME}" in
    google/gemma-3-12b-*|google/gemma-4-12B*|microsoft/phi-4|microsoft/Phi-4-reasoning*)
      BATCH_SIZE=1
      ;;
    Qwen/Qwen3-8B)
      BATCH_SIZE=2
      ;;
    *)
      BATCH_SIZE=4
      ;;
  esac
fi
DIRECTION_BATCH_SIZE=${DIRECTION_BATCH_SIZE:-${BATCH_SIZE}}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-768}
MODEL_REVISION=${MODEL_REVISION:-}
DEVICE_MAP=${DEVICE_MAP:-auto}
MONITOR_DEVICE=${MONITOR_DEVICE:-auto}

case "${PROTOCOL}" in
  pilot)
    METHODS=${METHODS:-"gcg random minscan"}
    SEEDS=${SEEDS:-"0"}
    CONTEXT_COUNT=${CONTEXT_COUNT:-4}
    EXAMPLE_LIMIT=${EXAMPLE_LIMIT:-128}
    POPULATION_SIZE=${POPULATION_SIZE:-8}
    ITERS=${ITERS:-50}
    EXPLORE_PER_POP=${EXPLORE_PER_POP:-8}
    RANDOM_PROMPTS=${RANDOM_PROMPTS:-128}
    PROMPT_TOP_N=${PROMPT_TOP_N:-2}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
    JACOBIAN_EXAMPLE_LIMIT=${JACOBIAN_EXAMPLE_LIMIT:-4}
    CAUSAL_MAX_PAIRS=${CAUSAL_MAX_PAIRS:-8}
    ;;
  full|matched)
    if [[ "${PROTOCOL}" == "full" ]]; then
      METHODS=${METHODS:-"epo gcg random random_search minscan"}
    else
      METHODS=${METHODS:-"epo gcg"}
    fi
    SEEDS=${SEEDS:-"0 1 2 3 4"}
    CONTEXT_COUNT=${CONTEXT_COUNT:-16}
    EXAMPLE_LIMIT=${EXAMPLE_LIMIT:-512}
    POPULATION_SIZE=${POPULATION_SIZE:-24}
    ITERS=${ITERS:-150}
    EXPLORE_PER_POP=${EXPLORE_PER_POP:-16}
    RANDOM_PROMPTS=${RANDOM_PROMPTS:-256}
    PROMPT_TOP_N=${PROMPT_TOP_N:-8}
    MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
    JACOBIAN_EXAMPLE_LIMIT=${JACOBIAN_EXAMPLE_LIMIT:-16}
    CAUSAL_MAX_PAIRS=${CAUSAL_MAX_PAIRS:-32}
    ;;
  *)
    echo "PROTOCOL must be 'pilot', 'full', or 'matched'" >&2
    exit 2
    ;;
esac

SEQ_LEN=${SEQ_LEN:-32}
TOPK=${TOPK:-256}

read -r -a METHOD_ARGS <<< "${METHODS}"
read -r -a SEED_ARGS <<< "${SEEDS}"

DIRECTION_DIR="${RUN_ROOT}/${SLUG}/directions"
DIRECTION_NAME="${SLUG}_eval_awareness"
SWEEP_CSV="${DIRECTION_DIR}/${DIRECTION_NAME}_layer_sweep.csv"
TARGET_SPEC="${RUN_ROOT}/${SLUG}/prompt_search_spec.json"
PROMPT_RUN="${RUN_ROOT}/${SLUG}/prompt_search"
PROMPT_CONTROLS="${RUN_ROOT}/${SLUG}/optimized_prompt_controls.txt"
STUDY_SPEC="${RUN_ROOT}/${SLUG}/study.json"
STUDY_OUT="${RUN_ROOT}/${SLUG}/reachable"

ATTN_ARGS=()
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARGS=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi
REVISION_ARGS=()
if [[ -n "${MODEL_REVISION}" ]]; then
  REVISION_ARGS=(--revision "${MODEL_REVISION}")
fi

uv run llm-controllability fit-directions \
  --model-name "${MODEL_NAME}" \
  "${REVISION_ARGS[@]}" \
  "${ATTN_ARGS[@]}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${DTYPE}" \
  --contrast "${DATA_ROOT}/contrasts/eval_awareness_train.json" \
  --contrast-eval "${DATA_ROOT}/contrasts/eval_awareness_validation.json" \
  --layers all \
  --out-dir "${DIRECTION_DIR}" \
  --name "${DIRECTION_NAME}" \
  --spec-out "${TARGET_SPEC}" \
  --top-k 1 \
  --bidirectional \
  --max-length "${MAX_CONTEXT_LENGTH}" \
  --batch-size "${DIRECTION_BATCH_SIZE}" \
  --texts-path "${DATA_ROOT}/text_pools/frontier_train.txt"

uv run llm-controllability run \
  --spec "${TARGET_SPEC}" \
  --texts "${DATA_ROOT}/text_pools/frontier_train.txt" \
  --contexts "${DATA_ROOT}/behavior/controllability_train.jsonl" \
  --context-count "${CONTEXT_COUNT}" \
  --context-max-length "${MAX_CONTEXT_LENGTH}" \
  --out "${PROMPT_RUN}" \
  --methods "${METHOD_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${DTYPE}" \
  --seq-len "${SEQ_LEN}" \
  --population-size "${POPULATION_SIZE}" \
  --iters "${ITERS}" \
  --explore-per-pop "${EXPLORE_PER_POP}" \
  --batch-size "${BATCH_SIZE}" \
  --topk "${TOPK}" \
  --random-prompts "${RANDOM_PROMPTS}"

uv run llm-controllability export-prompt-controls \
  --records "${PROMPT_RUN}/candidates.csv" \
  --direction-sweep "${SWEEP_CSV}" \
  --methods epo gcg \
  --top-n "${PROMPT_TOP_N}" \
  --bidirectional \
  --out "${PROMPT_CONTROLS}"

uv run llm-controllability build-study-spec \
  --out "${STUDY_SPEC}" \
  --model-name "${MODEL_NAME}" \
  "${REVISION_ARGS[@]}" \
  "${ATTN_ARGS[@]}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${DTYPE}" \
  --data "${DATA_ROOT}/behavior/controllability_all.jsonl" \
  --example-limit "${EXAMPLE_LIMIT}" \
  --direction-sweep "${SWEEP_CSV}" \
  --layers sweep \
  --prompt-controls "${PROMPT_CONTROLS}" \
  --natural-controls "${DATA_ROOT}/controls/prompt_rewrites.json" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --semantic-model sentence-transformers/all-mpnet-base-v2 \
  --minimum-semantic-similarity 0.80 \
  --maximum-quality-drop 0.75 \
  --maximum-control-cost 64

uv run llm-controllability collect-reachable \
  --spec "${STUDY_SPEC}" \
  --out "${STUDY_OUT}"

uv run llm-controllability analyze-control \
  --spec "${STUDY_SPEC}" \
  --states-dir "${STUDY_OUT}" \
  --out-dir "${RUN_ROOT}/${SLUG}/control" \
  --target-metric target_projection \
  --tolerance-fraction 0.10

uv run llm-controllability analyze-jacobians \
  --spec "${STUDY_SPEC}" \
  --out "${RUN_ROOT}/${SLUG}/jacobians.csv" \
  --example-limit "${JACOBIAN_EXAMPLE_LIMIT}" \
  --epsilon 0.25 \
  --seed 0

uv run llm-controllability analyze-transfer \
  --states-dir "${STUDY_OUT}" \
  --group-tag source \
  --target-metric target_projection \
  --out "${RUN_ROOT}/${SLUG}/transfer_source.csv"

uv run llm-controllability analyze-transfer \
  --states-dir "${STUDY_OUT}" \
  --group-tag category \
  --target-metric target_projection \
  --out "${RUN_ROOT}/${SLUG}/transfer_category.csv"

uv run llm-controllability causal-study \
  --spec "${STUDY_SPEC}" \
  --states-dir "${STUDY_OUT}" \
  --out-dir "${RUN_ROOT}/${SLUG}/causal" \
  --patch-layers spec \
  --prompt-prefix optimized_prompt \
  --activation-prefix activation_addition \
  --target-metric target_projection \
  --max-pairs "${CAUSAL_MAX_PAIRS}" \
  --seed 0

uv run llm-controllability monitor-invariance \
  --states-dir "${STUDY_OUT}" \
  --out-dir "${RUN_ROOT}/${SLUG}/monitors" \
  --monitors linear nonlinear multilayer_linear random_linear \
  --monitor-device "${MONITOR_DEVICE}" \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --reachable-weight 1.0 \
  --seed 0

uv run llm-controllability render-study-figures \
  --run-dir "${RUN_ROOT}/${SLUG}" \
  --out-dir "${RUN_ROOT}/${SLUG}/figures"
