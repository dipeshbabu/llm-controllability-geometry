"""Configuration-driven behavior-preserving controllability study."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from llm_controllability.constraints import (
    BehaviorGate,
    BudgetConstraint,
    OutputQualityConstraint,
    PromptSemanticEquivalenceConstraint,
    SemanticEquivalenceConstraint,
    TaskPreservationConstraint,
)
from llm_controllability.constraints.verification import TransformerSentenceEmbedder
from llm_controllability.controllability.types import ControlChannel
from llm_controllability.interventions import (
    ActivationAddition,
    AdaptiveActivationController,
    DirectionalAblation,
    HybridIntervention,
    MappedPromptIntervention,
    NoIntervention,
    PromptIntervention,
)
from llm_controllability.interventions.core import Intervention
from llm_controllability.models.adapters import model_device, model_dtype
from llm_controllability.models.loading import load_model
from llm_controllability.models.runtime import (
    model_runtime_config,
    runtime_capabilities,
)
from llm_controllability.reachability.boundaries import (
    controllability_boundary_rows,
    detection_control_gap_rows,
    directed_accessibility_rows,
    representation_control_gap_rows,
    summarize_controllability_boundaries,
    summarize_directed_accessibility,
)
from llm_controllability.reachability.cmap import (
    CMapConfig,
    discover_controllability_manifold,
)
from llm_controllability.reachability.collection import (
    collect_reachable_states,
    load_examples,
)
from llm_controllability.reachability.discovery import (
    boundary_survival_rows,
    controllability_atlas_rows,
    phase_transition_candidate_rows,
)
from llm_controllability.reachability.geometry import (
    budget_growth,
    layerwise_propagation,
    principal_angle_rows,
    split_half_stability_rows,
    summarize_reachability,
    summarize_split_half_stability,
    summarize_trajectories,
    target_orthogonal_decomposition,
)
from llm_controllability.reachability.io import save_state_samples


def _resolve(path: str, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base_dir / value


def _load_direction(config: Mapping[str, Any], base_dir: Path) -> torch.Tensor:
    path = _resolve(str(config["direction"]), base_dir)
    if path.suffix == ".npy":
        values = np.load(path)
    else:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        values = (
            loaded.detach().cpu().numpy()
            if isinstance(loaded, torch.Tensor)
            else loaded
        )
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1)


def build_interventions(
    configs: list[Mapping[str, Any]],
    *,
    base_dir: Path,
) -> list[Intervention]:
    interventions: list[Intervention] = [NoIntervention()]
    for config in configs:
        kind = str(config["type"])
        name = str(config["name"])
        if kind == "prompt":
            texts: list[str]
            if "texts_path" in config:
                texts = [
                    line
                    for line in _resolve(str(config["texts_path"]), base_dir)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
            else:
                texts = [str(config["text"])]
            for index, text in enumerate(texts):
                suffix = f"_{index:04d}" if len(texts) > 1 else ""
                interventions.append(
                    PromptIntervention(
                        name=f"{name}{suffix}",
                        text=text,
                        mode=str(config.get("mode", "suffix")),
                        separator=str(config.get("separator", "\n")),
                    )
                )
        elif kind == "mapped_prompt":
            rows = json.loads(
                _resolve(str(config["path"]), base_dir).read_text(encoding="utf-8")
            )
            if not isinstance(rows, list):
                raise TypeError("mapped prompt file must contain a JSON list")
            control_names = config.get("control_names")
            if control_names is None:
                control_names = sorted(
                    {
                        control_name
                        for row in rows
                        for control_name in row.get("controls", {})
                    }
                )
            for control_name in control_names:
                rewrites = {
                    str(row["source_prompt"]): str(row["controls"][control_name])
                    for row in rows
                    if control_name in row.get("controls", {})
                }
                if not rewrites:
                    raise ValueError(
                        f"mapped prompt file has no rewrites for {control_name!r}"
                    )
                interventions.append(
                    MappedPromptIntervention(
                        name=f"{name}_{control_name}",
                        rewrites=rewrites,
                        rewrite_kind=str(control_name),
                        strict=bool(config.get("strict", True)),
                    )
                )
        elif kind == "activation_addition":
            direction = _load_direction(config, base_dir)
            strengths = config.get("strengths", [config.get("strength", 1.0)])
            for strength in strengths:
                interventions.append(
                    ActivationAddition(
                        name=f"{name}_a{float(strength):g}",
                        layer=int(config["layer"]),
                        direction=direction,
                        strength=float(strength),
                        token_scope=str(config.get("token_scope", "last")),
                        channel=ControlChannel(
                            str(config.get("channel", ControlChannel.ACTIVATION.value))
                        ),
                    )
                )
        elif kind == "directional_ablation":
            direction = _load_direction(config, base_dir)
            fractions = config.get("fractions", [config.get("fraction", 1.0)])
            for fraction in fractions:
                interventions.append(
                    DirectionalAblation(
                        name=f"{name}_f{float(fraction):g}",
                        layer=int(config["layer"]),
                        direction=direction,
                        fraction=float(fraction),
                        token_scope=str(config.get("token_scope", "last")),
                    )
                )
        elif kind == "adaptive":
            direction = _load_direction(config, base_dir)
            setpoints = (
                config["setpoints"] if "setpoints" in config else [config["setpoint"]]
            )
            for setpoint in setpoints:
                interventions.append(
                    AdaptiveActivationController(
                        name=f"{name}_s{float(setpoint):g}",
                        layer=int(config["layer"]),
                        direction=direction,
                        setpoint=float(setpoint),
                        kp=float(config.get("kp", 1.0)),
                        ki=float(config.get("ki", 0.0)),
                        kd=float(config.get("kd", 0.0)),
                        max_update=(
                            float(config["max_update"])
                            if config.get("max_update") is not None
                            else None
                        ),
                        token_scope=str(config.get("token_scope", "last")),
                    )
                )
        elif kind == "hybrid":
            prompt_parts = build_interventions([config["prompt"]], base_dir=base_dir)[
                1:
            ]
            activation_parts = build_interventions(
                [config["activation"]], base_dir=base_dir
            )[1:]
            for prompt_part in prompt_parts:
                for activation_part in activation_parts:
                    if not isinstance(prompt_part, PromptIntervention):
                        raise TypeError(
                            "hybrid prompt component must have type 'prompt'"
                        )
                    interventions.append(
                        HybridIntervention(
                            name=f"{name}_{prompt_part.name}_{activation_part.name}",
                            prompt=prompt_part,
                            activation=activation_part,
                        )
                    )
        else:
            raise ValueError(f"unknown intervention type: {kind}")
    return interventions


def build_behavior_gate(config: Mapping[str, Any]) -> BehaviorGate:
    constraints = []
    task = config.get("task")
    if task is not None:
        constraints.append(
            TaskPreservationConstraint(
                minimum_score=task.get("minimum_score"),
                maximum_drop=task.get("maximum_drop", 0.0),
                require_correct=bool(task.get("require_correct", True)),
                require_reference_correct=bool(
                    task.get("require_reference_correct", True)
                ),
            )
        )
    semantic = config.get("semantic")
    if semantic is not None:
        constraints.append(
            SemanticEquivalenceConstraint(
                float(semantic.get("minimum_similarity", 0.85))
            )
        )
        if bool(semantic.get("check_prompt", True)):
            constraints.append(
                PromptSemanticEquivalenceConstraint(
                    float(
                        semantic.get(
                            "minimum_prompt_similarity",
                            semantic.get("minimum_similarity", 0.85),
                        )
                    )
                )
            )
    quality = config.get("quality")
    if quality is not None:
        constraints.append(
            OutputQualityConstraint(
                maximum_drop=float(quality.get("maximum_drop", 0.5)),
                minimum_score=quality.get("minimum_score"),
            )
        )
    budget = config.get("budget")
    if budget is not None:
        constraints.append(BudgetConstraint(float(budget["maximum_cost"])))
    if not constraints:
        raise ValueError(
            "at least one behavior or budget constraint must be configured"
        )
    return BehaviorGate(constraints)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _limit_examples(
    examples: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Limit a declared split without breaking paired question groups."""

    if limit is None or len(examples) <= limit:
        return examples
    if not all(
        example.get("split") in {"train", "validation", "test"} for example in examples
    ):
        return examples[:limit]

    by_split: dict[str, list[list[dict[str, Any]]]] = {}
    for split in ("train", "validation", "test"):
        grouped: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for index, example in enumerate(examples):
            if example["split"] != split:
                continue
            group = str(example.get("pair_id", example.get("id", index)))
            if group not in grouped:
                grouped[group] = []
                order.append(group)
            grouped[group].append(example)
        by_split[split] = [grouped[group] for group in order]

    split_counts = {
        split: sum(len(group) for group in groups) for split, groups in by_split.items()
    }
    total = sum(split_counts.values())
    raw = {split: limit * count / total for split, count in split_counts.items()}
    quotas = {split: int(value) for split, value in raw.items()}
    for split in sorted(
        raw,
        key=lambda value: raw[value] - quotas[value],
        reverse=True,
    )[: limit - sum(quotas.values())]:
        quotas[split] += 1

    selected: list[dict[str, Any]] = []
    deferred: list[list[dict[str, Any]]] = []
    for split in ("train", "validation", "test"):
        used = 0
        for group in by_split[split]:
            if used + len(group) <= quotas[split]:
                selected.extend(group)
                used += len(group)
            else:
                deferred.append(group)
    for group in deferred:
        if len(selected) + len(group) <= limit:
            selected.extend(group)
    return selected


def _runtime_metadata(
    model: torch.nn.Module,
    tokenizer,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    repository_root = Path(__file__).resolve().parents[3]
    lock_path = repository_root / "uv.lock"
    try:
        repository_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        repository_commit = None
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "model_device": str(model_device(model)),
        "model_dtype": str(model_dtype(model)).removeprefix("torch."),
        "accelerator": runtime_capabilities(),
        "model_revision": getattr(config, "_commit_hash", None),
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "repository_commit": repository_commit,
        "uv_lock_sha256": (
            hashlib.sha256(lock_path.read_bytes()).hexdigest()
            if lock_path.exists()
            else None
        ),
    }
    if torch.cuda.is_available():
        metadata["gpu"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    index
                ).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    else:
        metadata["gpu"] = []
    runtime = model_runtime_config(model)
    metadata["resolved_runtime"] = runtime.manifest() if runtime is not None else None
    return metadata


def run_study(spec_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    spec_path = Path(spec_path)
    base_dir = spec_path.resolve().parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model_config = spec["model"]
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[model_config.get("dtype", "bfloat16")]
    model, tokenizer = load_model(
        model_name=model_config["name"],
        tokenizer_name=model_config.get("tokenizer_name"),
        attn_implementation=model_config.get("attn_implementation"),
        device_map=model_config.get("device_map", "auto"),
        torch_dtype=dtype,
        prompt_format=model_config.get("prompt_format", "auto"),
        enable_thinking=model_config.get("enable_thinking"),
        trust_remote_code=model_config.get("trust_remote_code"),
        revision=model_config.get("revision"),
    )
    examples = load_examples(_resolve(spec["data"]["path"], base_dir))
    examples = _limit_examples(
        examples,
        (int(spec["data"]["limit"]) if spec["data"].get("limit") is not None else None),
    )
    interventions = build_interventions(spec["interventions"], base_dir=base_dir)
    gate = build_behavior_gate(spec["constraints"])

    semantic_embedder = None
    semantic = spec["constraints"].get("semantic")
    if semantic is not None:
        semantic_embedder = TransformerSentenceEmbedder(
            model_name=semantic["model_name"],
            device=semantic.get("device", "cpu"),
            max_length=int(semantic.get("max_length", 512)),
            batch_size=int(semantic.get("batch_size", 16)),
        )

    target_directions = {
        int(layer): np.load(_resolve(path, base_dir))
        for layer, path in spec.get("target_directions", {}).items()
    }
    samples = collect_reachable_states(
        model,
        tokenizer,
        model_name=model_config["name"],
        examples=examples,
        interventions=interventions,
        layers=[int(layer) for layer in spec["layers"]],
        behavior_gate=gate,
        generation=spec.get("generation", {}),
        pooling=spec.get("pooling", "last"),
        max_length=int(spec.get("max_length", 2048)),
        semantic_embedder=semantic_embedder,
        target_directions=target_directions,
        store_token_states=bool(spec.get("store_token_states", False)),
        seed=int(spec.get("seed", 0)),
    )
    out_dir = Path(out_dir)
    save_state_samples(samples, out_dir)
    split_values = {sample.tags.get("split") for sample in samples}
    analysis_samples = (
        [sample for sample in samples if sample.tags.get("split") == "test"]
        if "test" in split_values
        else samples
    )
    geometry = summarize_reachability(analysis_samples)
    _write_csv(geometry, out_dir / "geometry.csv")
    if analysis_samples is not samples:
        _write_csv(summarize_reachability(samples), out_dir / "geometry_all.csv")
    trajectories = summarize_trajectories(analysis_samples)
    _write_csv(trajectories, out_dir / "trajectory_geometry.csv")
    _write_csv(
        target_orthogonal_decomposition(analysis_samples),
        out_dir / "target_geometry.csv",
    )
    _write_csv(
        budget_growth(analysis_samples),
        out_dir / "budget_growth.csv",
    )
    _write_csv(
        layerwise_propagation(analysis_samples),
        out_dir / "layer_propagation.csv",
    )
    _write_csv(
        principal_angle_rows(analysis_samples),
        out_dir / "principal_angles.csv",
    )
    split_half = split_half_stability_rows(analysis_samples)
    _write_csv(split_half, out_dir / "split_half_stability.csv")
    _write_csv(
        summarize_split_half_stability(split_half),
        out_dir / "split_half_stability_summary.csv",
    )
    boundaries = controllability_boundary_rows(analysis_samples)
    _write_csv(boundaries, out_dir / "controllability_boundaries.csv")
    _write_csv(
        summarize_controllability_boundaries(boundaries),
        out_dir / "controllability_boundary_summary.csv",
    )
    accessibility = directed_accessibility_rows(analysis_samples)
    _write_csv(accessibility, out_dir / "directed_accessibility.csv")
    _write_csv(
        summarize_directed_accessibility(accessibility),
        out_dir / "directed_accessibility_summary.csv",
    )
    _write_csv(
        detection_control_gap_rows(analysis_samples),
        out_dir / "detection_control_gap.csv",
    )
    _write_csv(
        representation_control_gap_rows(analysis_samples),
        out_dir / "representation_control_gap.csv",
    )
    _write_csv(
        controllability_atlas_rows(analysis_samples),
        out_dir / "controllability_atlas.csv",
    )
    boundary_survival = boundary_survival_rows(analysis_samples)
    _write_csv(boundary_survival, out_dir / "boundary_survival.csv")
    _write_csv(
        phase_transition_candidate_rows(boundary_survival),
        out_dir / "phase_transition_candidates.csv",
    )
    cmap_artifacts: list[str] = []
    cmap_manifest: dict[str, Any] | None = None
    cmap_values = spec.get("cmap")
    if cmap_values is not None and bool(cmap_values.get("enabled", False)):
        cmap_config = CMapConfig.from_mapping(cmap_values)
        validation_examples = [
            example for example in examples if example.get("split") == "validation"
        ]
        test_examples = [
            example for example in examples if example.get("split") == "test"
        ]
        cmap_result = discover_controllability_manifold(
            model,
            tokenizer,
            model_name=model_config["name"],
            validation_examples=validation_examples,
            test_examples=test_examples,
            behavior_gate=gate,
            generation=spec.get("generation", {}),
            config=cmap_config,
            max_length=int(spec.get("max_length", 2048)),
            semantic_embedder=semantic_embedder,
            target_direction=target_directions.get(cmap_config.layer),
        )
        save_state_samples(cmap_result.samples, out_dir / "cmap")
        np.savez_compressed(
            out_dir / "cmap_directions.npz",
            **cmap_result.directions,
        )
        _write_csv(cmap_result.query_rows, out_dir / "cmap_queries.csv")
        _write_csv(cmap_result.direction_rows, out_dir / "cmap_directions.csv")
        _write_csv(cmap_result.summary_rows, out_dir / "cmap_summary.csv")
        cmap_artifacts = [
            "cmap/states.npz",
            "cmap/samples.jsonl",
            "cmap_directions.npz",
            "cmap_queries.csv",
            "cmap_directions.csv",
            "cmap_summary.csv",
        ]
        cmap_manifest = {
            "config": {
                key: value
                for key, value in cmap_values.items()
                if key != "enabled"
            },
            "n_state_samples": len(cmap_result.samples),
            "n_queries": len(cmap_result.query_rows),
            "n_discovered_directions": len(cmap_result.directions),
        }
    manifest = {
        "spec": str(spec_path),
        "model": model_config["name"],
        "n_examples": len(examples),
        "n_interventions": len(interventions),
        "layers": spec["layers"],
        "n_state_samples": len(samples),
        "n_behavior_preserved": sum(sample.behavior_preserved for sample in samples),
        "analysis_split": "test" if "test" in split_values else "all",
        "n_analysis_samples": len(analysis_samples),
        "cmap": cmap_manifest,
        "runtime": _runtime_metadata(model, tokenizer),
        "artifacts": [
            "states.npz",
            "samples.jsonl",
            "geometry.csv",
            "trajectory_geometry.csv",
            "target_geometry.csv",
            "budget_growth.csv",
            "layer_propagation.csv",
            "principal_angles.csv",
            "split_half_stability.csv",
            "split_half_stability_summary.csv",
            "controllability_boundaries.csv",
            "controllability_boundary_summary.csv",
            "directed_accessibility.csv",
            "directed_accessibility_summary.csv",
            "detection_control_gap.csv",
            "representation_control_gap.csv",
            "controllability_atlas.csv",
            "boundary_survival.csv",
            "phase_transition_candidates.csv",
            *cmap_artifacts,
        ],
    }
    if analysis_samples is not samples:
        manifest["artifacts"].append("geometry_all.csv")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
