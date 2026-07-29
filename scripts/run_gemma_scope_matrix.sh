#!/usr/bin/env bash
set -euo pipefail

uv run llm-controllability run-study-matrix \
  --analysis gemma_scope \
  --matrix configs/recommended_matrix.json
