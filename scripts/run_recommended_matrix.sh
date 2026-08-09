#!/usr/bin/env bash
set -euo pipefail

uv run llm-controllability run-study-matrix \
  --analysis controllability \
  --matrix configs/recommended_matrix.json

RUN_ROOT=${RUN_ROOT:-runs/controllability}
uv run llm-controllability render-matrix-figures \
  --matrix-dir "${RUN_ROOT}/matrix" \
  --out-dir "${RUN_ROOT}/matrix/figures"
