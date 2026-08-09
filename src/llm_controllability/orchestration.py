"""Configuration-driven launch orchestration for the declared model matrix."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_controllability.evaluation.matrix import aggregate_matrix

ANALYSES = ("controllability", "token_monitors", "gemma_scope")


def load_matrix(path: str | Path) -> dict[str, Any]:
    matrix = json.loads(Path(path).read_text(encoding="utf-8"))
    models = matrix.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("matrix must declare a nonempty models list")
    slugs = [str(model.get("slug", "")) for model in models]
    if any(not slug for slug in slugs) or len(slugs) != len(set(slugs)):
        raise ValueError("matrix model slugs must be nonempty and unique")
    return matrix


def matrix_commands(
    matrix: dict[str, Any],
    analysis: str,
    *,
    selected_models: Sequence[str] = (),
) -> list[list[str]]:
    if analysis not in ANALYSES:
        raise ValueError(f"unknown matrix analysis: {analysis}")
    selected = set(selected_models)
    defaults = matrix.get("defaults", {})
    dtype_default = str(defaults.get("dtype", "bfloat16"))
    attention_default = str(defaults.get("attn_implementation", "eager"))
    commands: list[list[str]] = []
    for model in matrix["models"]:
        slug = str(model["slug"])
        if selected and slug not in selected:
            continue
        if analysis not in model.get("analyses", []):
            continue
        model_name = str(model["model_name"])
        dtype = str(model.get("dtype", dtype_default))
        attention = str(model.get("attn_implementation", attention_default))
        if analysis == "controllability":
            commands.append(
                [
                    "bash",
                    "scripts/run_model_controllability.sh",
                    model_name,
                    slug,
                    dtype,
                    attention,
                    str(model["protocol"]),
                ]
            )
        elif analysis == "token_monitors":
            commands.append(
                [
                    "bash",
                    "scripts/run_token_monitor_study.sh",
                    model_name,
                    slug,
                    dtype,
                    attention,
                ]
            )
        else:
            commands.append(
                [
                    "bash",
                    "scripts/run_gemma_scope_study.sh",
                    model_name,
                    slug,
                ]
            )
    missing = selected - {str(model["slug"]) for model in matrix["models"]}
    if missing:
        raise ValueError(f"unknown selected model slugs: {', '.join(sorted(missing))}")
    if not commands:
        raise ValueError(f"matrix selects no models for analysis {analysis!r}")
    return commands


def run_declared_matrix(
    matrix_path: str | Path,
    analysis: str,
    *,
    project_root: str | Path,
    selected_models: Sequence[str] = (),
    dry_run: bool = False,
    aggregate: bool = True,
) -> list[list[str]]:
    root = Path(project_root).resolve()
    matrix_path = Path(matrix_path)
    if not matrix_path.is_absolute():
        matrix_path = root / matrix_path
    matrix = load_matrix(matrix_path)
    commands = matrix_commands(
        matrix,
        analysis,
        selected_models=selected_models,
    )
    if dry_run:
        return commands

    for command in commands:
        script = root / command[1]
        if not script.is_file():
            raise FileNotFoundError(f"matrix launcher does not exist: {script}")
        subprocess.run(command, cwd=root, check=True)

    if analysis == "controllability" and aggregate and not selected_models:
        run_root = Path(os.environ.get("RUN_ROOT", "runs/controllability"))
        if not run_root.is_absolute():
            run_root = root / run_root
        aggregate_subdir = str(matrix.get("aggregate_subdir", "matrix"))
        aggregate_matrix(run_root, matrix_path, run_root / aggregate_subdir)
    return commands
