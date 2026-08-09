#!/usr/bin/env bash
set -euo pipefail

export RUN_ROOT=${RUN_ROOT:-runs/controllability}

uv run llm-controllability run-study-matrix \
  --analysis controllability \
  --matrix configs/scaling_matrix.json

uv run llm-controllability render-matrix-figures \
  --matrix-dir "${RUN_ROOT}/scaling_matrix" \
  --out-dir "${RUN_ROOT}/scaling_matrix/figures"
