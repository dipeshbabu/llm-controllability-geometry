#!/usr/bin/env bash
set -euo pipefail

uv run llm-controllability run-study-matrix \
  --analysis token_monitors \
  --matrix configs/recommended_matrix.json
