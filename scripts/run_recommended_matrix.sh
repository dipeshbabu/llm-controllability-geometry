#!/usr/bin/env bash
set -euo pipefail

uv run llm-controllability run-study-matrix \
  --analysis controllability \
  --matrix configs/recommended_matrix.json
