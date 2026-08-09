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
HOST_IS_DARWIN=0
if [[ "$(uname -s)" == "Darwin" ]]; then
  HOST_IS_DARWIN=1
fi
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if [[ "${HOST_IS_DARWIN}" -eq 1 ]]; then
    case "${MODEL_NAME}" in
      google/gemma-3-12b-*|Qwen/Qwen3-8B)
        BATCH_SIZE=1
        ;;
      google/gemma-3-4b-*|microsoft/Phi-4-mini-instruct|Qwen/Qwen3-4B)
        BATCH_SIZE=2
        ;;
      *)
        BATCH_SIZE=4
        ;;
    esac
  else
    case "${MODEL_NAME}" in
      google/gemma-3-12b-*)
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
fi
if [[ -z "${DIRECTION_BATCH_SIZE:-}" ]]; then
  if [[ "${HOST_IS_DARWIN}" -eq 1 && "${MODEL_NAME}" == "Qwen/Qwen3-8B" ]]; then
    DIRECTION_BATCH_SIZE=2
  else
    DIRECTION_BATCH_SIZE=${BATCH_SIZE}
  fi
fi
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-768}
MODEL_REVISION=${MODEL_REVISION:-}
DEVICE_MAP=${DEVICE_MAP:-auto}
MONITOR_DEVICE=${MONITOR_DEVICE:-auto}

DEFAULT_SEEDS="0 1 2 3 4"
DEFAULT_CONTEXT_COUNT=16
DEFAULT_EXAMPLE_LIMIT=512
DEFAULT_CMAP_DIRECTIONS=8
DEFAULT_CMAP_QUERY_BUDGET=512
DEFAULT_CMAP_VALIDATION_EXAMPLES=8
DEFAULT_CMAP_TEST_EXAMPLES=16
DEFAULT_POPULATION_SIZE=24
DEFAULT_ITERS=150
DEFAULT_EXPLORE_PER_POP=16
DEFAULT_RANDOM_PROMPTS=256
DEFAULT_PROMPT_TOP_N=8
DEFAULT_MAX_NEW_TOKENS=128
DEFAULT_JACOBIAN_EXAMPLE_LIMIT=16
DEFAULT_CAUSAL_MAX_PAIRS=32

case "${PROTOCOL}" in
  pilot)
    METHODS=${METHODS:-"gcg random minscan"}
    DEFAULT_SEEDS="0"
    DEFAULT_CONTEXT_COUNT=4
    DEFAULT_EXAMPLE_LIMIT=128
    DEFAULT_CMAP_DIRECTIONS=2
    DEFAULT_CMAP_QUERY_BUDGET=64
    DEFAULT_CMAP_VALIDATION_EXAMPLES=2
    DEFAULT_CMAP_TEST_EXAMPLES=4
    DEFAULT_POPULATION_SIZE=8
    DEFAULT_ITERS=50
    DEFAULT_EXPLORE_PER_POP=8
    DEFAULT_RANDOM_PROMPTS=128
    DEFAULT_PROMPT_TOP_N=2
    DEFAULT_MAX_NEW_TOKENS=64
    DEFAULT_JACOBIAN_EXAMPLE_LIMIT=4
    DEFAULT_CAUSAL_MAX_PAIRS=8
    ;;
  full)
    METHODS=${METHODS:-"epo gcg random random_search minscan"}
    ;;
  matched)
    METHODS=${METHODS:-"epo gcg"}
    ;;
  scaling)
    METHODS=${METHODS:-"epo gcg"}
    DEFAULT_SEEDS="0 1 2"
    DEFAULT_CONTEXT_COUNT=8
    DEFAULT_EXAMPLE_LIMIT=128
    DEFAULT_CMAP_DIRECTIONS=4
    DEFAULT_CMAP_QUERY_BUDGET=192
    DEFAULT_CMAP_VALIDATION_EXAMPLES=4
    DEFAULT_CMAP_TEST_EXAMPLES=8
    ;;
  *)
    echo "PROTOCOL must be 'pilot', 'full', 'matched', or 'scaling'" >&2
    exit 2
    ;;
esac

SEEDS=${SEEDS:-${DEFAULT_SEEDS}}
CONTEXT_COUNT=${CONTEXT_COUNT:-${DEFAULT_CONTEXT_COUNT}}
EXAMPLE_LIMIT=${EXAMPLE_LIMIT:-${DEFAULT_EXAMPLE_LIMIT}}
CMAP_DIRECTIONS=${CMAP_DIRECTIONS:-${DEFAULT_CMAP_DIRECTIONS}}
CMAP_QUERY_BUDGET=${CMAP_QUERY_BUDGET:-${DEFAULT_CMAP_QUERY_BUDGET}}
CMAP_VALIDATION_EXAMPLES=${CMAP_VALIDATION_EXAMPLES:-${DEFAULT_CMAP_VALIDATION_EXAMPLES}}
CMAP_TEST_EXAMPLES=${CMAP_TEST_EXAMPLES:-${DEFAULT_CMAP_TEST_EXAMPLES}}
POPULATION_SIZE=${POPULATION_SIZE:-${DEFAULT_POPULATION_SIZE}}
ITERS=${ITERS:-${DEFAULT_ITERS}}
EXPLORE_PER_POP=${EXPLORE_PER_POP:-${DEFAULT_EXPLORE_PER_POP}}
RANDOM_PROMPTS=${RANDOM_PROMPTS:-${DEFAULT_RANDOM_PROMPTS}}
PROMPT_TOP_N=${PROMPT_TOP_N:-${DEFAULT_PROMPT_TOP_N}}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-${DEFAULT_MAX_NEW_TOKENS}}
JACOBIAN_EXAMPLE_LIMIT=${JACOBIAN_EXAMPLE_LIMIT:-${DEFAULT_JACOBIAN_EXAMPLE_LIMIT}}
CAUSAL_MAX_PAIRS=${CAUSAL_MAX_PAIRS:-${DEFAULT_CAUSAL_MAX_PAIRS}}
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
  --maximum-control-cost 64 \
  --cmap-directions "${CMAP_DIRECTIONS}" \
  --cmap-query-budget "${CMAP_QUERY_BUDGET}" \
  --cmap-validation-examples "${CMAP_VALIDATION_EXAMPLES}" \
  --cmap-test-examples "${CMAP_TEST_EXAMPLES}"

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
  --basis-dimensions 8 16 32 \
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
