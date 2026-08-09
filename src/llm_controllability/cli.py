"""Command line tools for language model controllability experiments."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import torch

from llm_controllability.controllability.specs import (
    best_direction_target,
    build_study_spec,
    export_bidirectional_prompt_controls,
    export_prompt_controls,
)
from llm_controllability.controllability.study import run_study
from llm_controllability.data.behavior import (
    load_behavior_evals,
    score_continuations,
    write_behavior_rows,
    write_behavior_templates,
)
from llm_controllability.data.frontier import SOURCE_SPECS, build_frontier_data
from llm_controllability.evaluation.causal_study import run_causal_study
from llm_controllability.evaluation.control_study import run_control_study
from llm_controllability.evaluation.figures import (
    render_matrix_figures,
    render_study_figures,
)
from llm_controllability.evaluation.jacobian_study import run_jacobian_study
from llm_controllability.evaluation.matrix import aggregate_matrix
from llm_controllability.evaluation.monitor_study import run_monitor_study
from llm_controllability.evaluation.transfer_study import run_transfer_study
from llm_controllability.features.gemma_scope import run_gemma_scope_study
from llm_controllability.models.architecture import get_layers
from llm_controllability.models.loading import load_model
from llm_controllability.models.runtime import (
    resolve_runtime,
    runtime_capabilities,
)
from llm_controllability.optimization.benchmarks import (
    epo_suppression_run,
    gcg_suppression_run,
    minscan_baseline,
    random_search_baseline,
    random_token_baseline,
)
from llm_controllability.optimization.contextual import ContextualTargetRunner
from llm_controllability.optimization.robustness import (
    evaluate_robustness,
    robustness_rows,
    robustness_summary_rows,
)
from llm_controllability.orchestration import ANALYSES, run_declared_matrix
from llm_controllability.reachability import (
    boundary_survival_rows,
    controllability_atlas_rows,
    controllability_boundary_rows,
    detection_control_gap_rows,
    directed_accessibility_rows,
    phase_transition_candidate_rows,
    split_half_stability_rows,
    summarize_controllability_boundaries,
    summarize_directed_accessibility,
    summarize_split_half_stability,
)
from llm_controllability.reachability.geometry import summarize_reachability
from llm_controllability.reachability.io import load_state_samples
from llm_controllability.reporting.latex import rows_from_csv, rows_to_latex_table
from llm_controllability.reporting.plotting import plot_method_bars, plot_scatter
from llm_controllability.reporting.results import (
    records_from_csv,
    records_to_csv,
    rows_to_csv,
    summarize_by_method,
)
from llm_controllability.targets.directions import (
    fit_direction_sweep,
    top_direction_specs,
)
from llm_controllability.targets.generation import (
    logit_specs,
    neuron_specs,
    parse_int_list,
    residual_specs,
    write_spec,
)
from llm_controllability.targets.specs import build_runner_from_spec, target_name


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def runtime_info(args) -> None:
    runtime = resolve_runtime(
        device_map=args.device_map,
        dtype=args.torch_dtype,
        attention_implementation=args.attn_implementation,
    )
    print(
        json.dumps(
            {
                **runtime_capabilities(),
                "runtime": runtime.manifest(),
            },
            indent=2,
        )
    )


def _model_profile_kwargs(spec: dict) -> dict:
    profile = spec.get("model_profile", {})
    return {
        "prompt_format": profile.get("prompt_format", "auto"),
        "enable_thinking": profile.get("enable_thinking"),
        "trust_remote_code": profile.get("trust_remote_code"),
    }


def _load_texts(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
            texts = [
                str(row.get("prompt", row.get("text", ""))).strip() for row in rows
            ]
            return [text for text in texts if text]
        return [line.strip() for line in f if line.strip()]


def run_experiments(args) -> None:
    spec = _load_json(args.spec)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model, tokenizer = load_model(
        model_size=spec.get("model_size", args.model_size),
        model_name=spec.get("model_name", args.model_name),
        tokenizer_name=spec.get("tokenizer_name"),
        attn_implementation=spec.get("attn_implementation", args.attn_implementation),
        device_map=spec.get("device_map", args.device_map),
        torch_dtype=dtype,
        revision=spec.get("revision", args.revision),
        **_model_profile_kwargs(spec),
    )
    texts = _load_texts(args.texts or spec.get("texts_path"))
    contexts = _load_texts(args.contexts)
    if args.context_count > 0 and not contexts:
        raise ValueError("--context-count requires a nonempty --contexts file")
    methods = set(args.methods)
    records = []

    for target_spec in spec["targets"]:
        name = target_name(target_spec)
        minimize = bool(target_spec.get("minimize", True))
        base_runner = build_runner_from_spec(model, tokenizer, target_spec)
        for seed in args.seeds:
            runner = (
                ContextualTargetRunner.from_texts(
                    base_runner,
                    model,
                    tokenizer,
                    contexts,
                    context_count=args.context_count,
                    max_length=args.context_max_length,
                    seed=seed,
                )
                if args.context_count > 0
                else base_runner
            )
            if "epo" in methods:
                records.extend(
                    epo_suppression_run(
                        runner,
                        model,
                        tokenizer,
                        target_name=name,
                        seed=seed,
                        seq_len=args.seq_len,
                        population_size=args.population_size,
                        iters=args.iters,
                        explore_per_pop=args.explore_per_pop,
                        batch_size=args.batch_size,
                        topk=args.topk,
                        minimize=minimize,
                    )
                )
            if "gcg" in methods:
                records.extend(
                    gcg_suppression_run(
                        runner,
                        model,
                        tokenizer,
                        target_name=name,
                        seed=seed,
                        seq_len=args.seq_len,
                        iters=args.iters,
                        explore_per_iter=args.explore_per_pop,
                        batch_size=args.batch_size,
                        topk=args.topk,
                        x_penalty=args.gcg_x_penalty,
                        minimize=minimize,
                    )
                )
            if "random" in methods:
                records.extend(
                    random_token_baseline(
                        runner,
                        model,
                        tokenizer,
                        target_name=name,
                        seed=seed,
                        n_prompts=args.random_prompts,
                        seq_len=args.seq_len,
                        batch_size=args.batch_size,
                    )
                )
            if "random_search" in methods:
                records.extend(
                    random_search_baseline(
                        runner,
                        model,
                        tokenizer,
                        target_name=name,
                        seed=seed,
                        population_size=args.population_size,
                        iters=args.iters,
                        explore_per_pop=args.explore_per_pop,
                        seq_len=args.seq_len,
                        batch_size=args.batch_size,
                    )
                )
            if "minscan" in methods:
                if not texts:
                    raise ValueError(
                        "minscan requires --texts or texts_path in the spec"
                    )
                records.extend(
                    minscan_baseline(
                        runner,
                        model,
                        tokenizer,
                        texts,
                        target_name=name,
                        seed=seed,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        fluency_quantile=args.minscan_fluency_quantile,
                        minimize=minimize,
                    )
                )

    out = Path(args.out)
    records_to_csv(records, out / "candidates.csv")
    rows_to_csv(summarize_by_method(records), out / "summary.csv")
    resolved_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": spec.get("model_name", args.model_name),
                "requested_revision": spec.get("revision", args.revision),
                "resolved_revision": resolved_revision,
                "methods": sorted(methods),
                "seeds": args.seeds,
                "sequence_length": args.seq_len,
                "population_size": args.population_size,
                "iterations": args.iters,
                "explore_per_population": args.explore_per_pop,
                "context_count": args.context_count,
                "artifacts": ["candidates.csv", "summary.csv"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def summarize(args) -> None:
    records = records_from_csv(args.records)
    rows = summarize_by_method(
        records,
        threshold=args.threshold,
        fluent_quantile=args.fluent_quantile,
    )
    rows_to_csv(rows, args.out)


def plot(args) -> None:
    records = records_from_csv(args.records)
    out_dir = Path(args.out_dir)
    for target in sorted({r.target_name for r in records}):
        group = [r for r in records if r.target_name == target]
        minimize = not target.endswith("_increase")
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in target)
        plot_scatter(
            group,
            out_dir / f"{safe}_scatter.png",
            title=target,
            minimize=minimize,
        )
        plot_method_bars(
            group,
            out_dir / f"{safe}_bars.png",
            title=target,
            minimize=minimize,
        )


def robustness(args) -> None:
    spec = _load_json(args.spec)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model, tokenizer = load_model(
        model_size=spec.get("model_size", args.model_size),
        model_name=spec.get("model_name", args.model_name),
        tokenizer_name=spec.get("tokenizer_name"),
        attn_implementation=spec.get("attn_implementation", args.attn_implementation),
        device_map=spec.get("device_map", args.device_map),
        torch_dtype=dtype,
        revision=spec.get("revision", args.revision),
        **_model_profile_kwargs(spec),
    )
    records = records_from_csv(args.records)
    by_name = {target_name(t): t for t in spec["targets"]}
    robust_records = []
    for name, target_spec in by_name.items():
        group = [r for r in records if r.target_name == name]
        if args.top_n:
            group = sorted(group, key=lambda r: (r.target, r.xentropy))[: args.top_n]
        if not group:
            continue
        runner = build_runner_from_spec(model, tokenizer, target_spec)
        robust_records.extend(
            evaluate_robustness(
                runner,
                model,
                tokenizer,
                group,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        )
    records_to_csv(robust_records, args.out)
    rows_to_csv(robustness_rows(robust_records), args.rows_out)
    if args.summary_out:
        rows_to_csv(
            robustness_summary_rows(
                robust_records,
                target_tolerance=args.target_tolerance,
            ),
            args.summary_out,
        )


def behavior(args) -> None:
    spec = _load_json(args.spec) if args.spec else {}
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model, tokenizer = load_model(
        model_size=spec.get("model_size", args.model_size),
        model_name=spec.get("model_name", args.model_name),
        tokenizer_name=spec.get("tokenizer_name"),
        attn_implementation=spec.get("attn_implementation", args.attn_implementation),
        device_map=spec.get("device_map", args.device_map),
        torch_dtype=dtype,
        revision=spec.get("revision", args.revision),
        **_model_profile_kwargs(spec),
    )
    evals = load_behavior_evals(args.evals)
    rows = score_continuations(model, tokenizer, evals)
    write_behavior_rows(rows, args.out)


def generate_targets(args) -> None:
    targets = []
    if args.tokens:
        targets.extend(logit_specs(args.tokens, prefix=args.logit_prefix))
    if args.token_file:
        targets.extend(
            logit_specs(_load_texts(args.token_file), prefix=args.logit_prefix)
        )
    if args.layers and args.neurons:
        targets.extend(
            neuron_specs(
                parse_int_list(args.layers),
                parse_int_list(args.neurons),
                prefix=args.neuron_prefix,
            )
        )
    if args.vector:
        layer_by_file = None
        if args.vector_layers:
            layer_by_file = {}
            for part in args.vector_layers.split(","):
                if part.strip():
                    name, layer = part.split("=", 1)
                    layer_by_file[name.strip()] = int(layer)
        targets.extend(
            residual_specs(
                args.vector,
                layer_by_file=layer_by_file,
                default_layer=args.default_vector_layer,
                prefix=args.residual_prefix,
            )
        )
    write_spec(
        targets,
        args.out,
        model_name=args.model_name,
        model_size=args.model_size,
        texts_path=args.texts_path,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        revision=args.revision,
    )


def fit_directions(args) -> None:
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model, tokenizer = load_model(
        model_size=args.model_size,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        torch_dtype=dtype,
        revision=args.revision,
    )
    rows = fit_direction_sweep(
        model,
        tokenizer,
        args.contrast,
        (
            list(range(len(get_layers(model))))
            if args.layers == "all"
            else parse_int_list(args.layers)
        ),
        args.out_dir,
        name=args.name,
        eval_contrast_path=args.contrast_eval,
        pooling=args.pooling,
        max_len=args.max_length,
        batch_size=args.batch_size,
    )
    resolved_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "direction_manifest.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "requested_revision": args.revision,
                "resolved_revision": resolved_revision,
                "contrast": args.contrast,
                "contrast_validation": args.contrast_eval,
                "pooling": args.pooling,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "layers": args.layers,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.spec_out:
        write_spec(
            top_direction_specs(
                rows,
                top_k=args.top_k,
                bidirectional=args.bidirectional,
            ),
            args.spec_out,
            model_name=args.model_name,
            model_size=args.model_size,
            texts_path=args.texts_path,
            attn_implementation=args.attn_implementation,
            device_map=args.device_map,
            revision=resolved_revision or args.revision,
        )


def latex_table(args) -> None:
    rows_to_latex_table(
        rows_from_csv(args.csv),
        args.out,
        columns=args.columns.split(",") if args.columns else None,
        caption=args.caption or "",
        label=args.label or "",
    )


def behavior_templates(args) -> None:
    write_behavior_templates(args.out)


def frontier_data(args) -> None:
    manifest = build_frontier_data(
        args.sources,
        args.out_dir,
        max_items_per_source=args.max_items_per_source,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        behavior_limit=args.behavior_limit,
        allow_gated=args.allow_gated,
    )
    print(json.dumps(manifest, indent=2))


def collect_reachable(args) -> None:
    manifest = run_study(args.spec, args.out)
    print(json.dumps(manifest, indent=2))


def analyze_reachability(args) -> None:
    samples = load_state_samples(args.states_dir)
    out_path = Path(args.out)
    rows_to_csv(summarize_reachability(samples), out_path)
    boundaries = controllability_boundary_rows(
        samples,
        target_metric=args.target_metric,
    )
    accessibility = directed_accessibility_rows(samples, seed=args.seed)
    split_half = split_half_stability_rows(samples, seed=args.seed)
    boundary_survival = boundary_survival_rows(
        samples,
        target_metric=args.target_metric,
        seed=args.seed,
    )
    artifacts = {
        "controllability_boundaries.csv": boundaries,
        "controllability_boundary_summary.csv": (
            summarize_controllability_boundaries(boundaries, seed=args.seed)
        ),
        "directed_accessibility.csv": accessibility,
        "directed_accessibility_summary.csv": (
            summarize_directed_accessibility(accessibility, seed=args.seed)
        ),
        "detection_control_gap.csv": detection_control_gap_rows(
            samples,
            target_metric=args.target_metric,
        ),
        "split_half_stability.csv": split_half,
        "split_half_stability_summary.csv": summarize_split_half_stability(split_half),
        "controllability_atlas.csv": controllability_atlas_rows(
            samples,
            target_metric=args.target_metric,
        ),
        "boundary_survival.csv": boundary_survival,
        "phase_transition_candidates.csv": phase_transition_candidate_rows(
            boundary_survival
        ),
    }
    for filename, rows in artifacts.items():
        rows_to_csv(rows, out_path.parent / filename)
    print(
        json.dumps(
            {
                "geometry": str(out_path),
                "extended_artifacts": sorted(artifacts),
            },
            indent=2,
        )
    )


def monitor_invariance_study(args) -> None:
    manifest = run_monitor_study(
        args.states_dir,
        args.out_dir,
        label_key=args.label_key,
        monitor_kinds=args.monitors,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        reachable_weight=args.reachable_weight,
        monitor_device=args.monitor_device,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


def export_controls(args) -> None:
    target = args.target
    if target is None and args.direction_sweep is not None:
        target = best_direction_target(args.direction_sweep)
    if args.bidirectional:
        if target is None:
            raise ValueError(
                "bidirectional export requires --target or --direction-sweep"
            )
        texts = export_bidirectional_prompt_controls(
            args.records,
            args.out,
            target_base=target,
            methods=args.methods,
            top_n_per_direction=args.top_n,
        )
        print(
            json.dumps(
                {
                    "out": args.out,
                    "n_controls": len(texts),
                    "per_direction": args.top_n,
                },
                indent=2,
            )
        )
        return
    texts = export_prompt_controls(
        args.records,
        args.out,
        target_name=target,
        methods=args.methods,
        top_n=args.top_n,
        minimize=not args.maximize,
    )
    print(json.dumps({"out": args.out, "n_controls": len(texts)}, indent=2))


def build_controllability_spec(args) -> None:
    spec = build_study_spec(
        out_path=args.out,
        model_name=args.model_name,
        data_path=args.data,
        direction_sweep=args.direction_sweep,
        capture_layers=(
            None
            if args.layers == "sweep"
            else []
            if args.layers == "target"
            else parse_int_list(args.layers)
        ),
        prompt_controls_path=args.prompt_controls,
        natural_controls_path=args.natural_controls,
        example_limit=args.example_limit,
        dtype=args.torch_dtype,
        device_map=args.device_map,
        revision=args.revision,
        attn_implementation=args.attn_implementation,
        strengths=args.strengths,
        ablation_fractions=args.ablation_fractions,
        max_new_tokens=args.max_new_tokens,
        maximum_quality_drop=args.maximum_quality_drop,
        maximum_control_cost=args.maximum_control_cost,
        semantic_model=args.semantic_model,
        minimum_semantic_similarity=args.minimum_semantic_similarity,
        store_token_states=args.store_token_states,
        cmap_direction_budget=args.cmap_directions,
        cmap_query_budget=args.cmap_query_budget,
        cmap_validation_examples=args.cmap_validation_examples,
        cmap_test_examples=args.cmap_test_examples,
        seed=args.seed,
    )
    print(json.dumps({"out": args.out, "layers": spec["layers"]}, indent=2))


def transfer_study(args) -> None:
    summary = run_transfer_study(
        args.states_dir,
        args.out,
        group_tag=args.group_tag,
        target_metric=args.target_metric,
    )
    print(json.dumps(summary, indent=2))


def causal_study(args) -> None:
    manifest = run_causal_study(
        args.spec,
        args.states_dir,
        args.out_dir,
        selection_layer=args.selection_layer,
        patch_layers=(
            None if args.patch_layers == "spec" else parse_int_list(args.patch_layers)
        ),
        prompt_prefix=args.prompt_prefix,
        activation_prefix=args.activation_prefix,
        target_metric=args.target_metric,
        max_pairs=args.max_pairs,
        max_heads_per_layer=args.max_heads_per_layer,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


def control_study(args) -> None:
    manifest = run_control_study(
        args.states_dir,
        args.spec,
        args.out_dir,
        target_metric=args.target_metric,
        tolerance_fraction=args.tolerance_fraction,
    )
    print(json.dumps(manifest, indent=2))


def jacobian_study(args) -> None:
    manifest = run_jacobian_study(
        args.spec,
        args.out,
        example_limit=args.example_limit,
        epsilon=args.epsilon,
        basis_dimensions=tuple(args.basis_dimensions),
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


def render_figures(args) -> None:
    manifest = render_study_figures(args.run_dir, args.out_dir)
    print(json.dumps(manifest, indent=2))


def render_discovery_figures(args) -> None:
    manifest = render_matrix_figures(args.matrix_dir, args.out_dir)
    print(json.dumps(manifest, indent=2))


def aggregate_study_matrix(args) -> None:
    manifest = aggregate_matrix(
        args.run_root,
        args.matrix,
        args.out_dir,
    )
    print(json.dumps(manifest, indent=2))


def run_study_matrix(args) -> None:
    commands = run_declared_matrix(
        args.matrix,
        args.analysis,
        project_root=args.project_root,
        selected_models=args.models or (),
        dry_run=args.dry_run,
        aggregate=not args.no_aggregate,
    )
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))


def gemma_scope_study(args) -> None:
    manifest = run_gemma_scope_study(
        args.states_dir,
        args.out_dir,
        model_name=args.model_name,
        layer=args.layer,
        direction_sweep=args.direction_sweep,
        release=args.release,
        sae_id=args.sae_id,
        site=args.site,
        width=args.width,
        l0=args.l0,
        device=args.device,
        top_k=args.top_k,
        analysis_features=args.analysis_features,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        include_unpreserved=args.include_unpreserved,
    )
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-controllability")
    sub = parser.add_subparsers(dest="command", required=True)

    runtime = sub.add_parser(
        "runtime-info",
        help="report resolved CUDA, MPS, or CPU runtime settings",
    )
    runtime.add_argument("--device-map", default="auto")
    runtime.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    runtime.add_argument("--attn-implementation")
    runtime.set_defaults(func=runtime_info)

    run = sub.add_parser("run", help="optimize prompt controls from a JSON target spec")
    run.add_argument("--spec", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--texts")
    run.add_argument("--contexts")
    run.add_argument("--context-count", type=int, default=0)
    run.add_argument("--context-max-length", type=int, default=256)
    run.add_argument("--methods", nargs="+", default=["epo", "random", "minscan"])
    run.add_argument("--seeds", nargs="+", type=int, default=[0])
    run.add_argument("--model-size", default="70m")
    run.add_argument("--model-name")
    run.add_argument("--revision")
    run.add_argument("--attn-implementation", default=None)
    run.add_argument("--device-map", default="auto")
    run.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    run.add_argument("--seq-len", type=int, default=24)
    run.add_argument("--population-size", type=int, default=16)
    run.add_argument("--iters", type=int, default=100)
    run.add_argument("--explore-per-pop", type=int, default=16)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--topk", type=int, default=256)
    run.add_argument("--random-prompts", type=int, default=256)
    run.add_argument("--max-length", type=int, default=128)
    run.add_argument("--minscan-fluency-quantile", type=float, default=0.2)
    run.add_argument("--gcg-x-penalty", type=float, default=1.0)
    run.set_defaults(func=run_experiments)

    summary = sub.add_parser("summarize", help="summarize candidate CSV output")
    summary.add_argument("--records", required=True)
    summary.add_argument("--out", required=True)
    summary.add_argument("--threshold", type=float)
    summary.add_argument("--fluent-quantile", type=float, default=0.25)
    summary.set_defaults(func=summarize)

    plots = sub.add_parser("plot", help="generate standard figures")
    plots.add_argument("--records", required=True)
    plots.add_argument("--out-dir", required=True)
    plots.set_defaults(func=plot)

    robust = sub.add_parser("robustness", help="evaluate deterministic prompt variants")
    robust.add_argument("--spec", required=True)
    robust.add_argument("--records", required=True)
    robust.add_argument("--out", required=True)
    robust.add_argument("--rows-out", required=True)
    robust.add_argument("--summary-out")
    robust.add_argument("--target-tolerance", type=float, default=0.0)
    robust.add_argument("--top-n", type=int, default=10)
    robust.add_argument("--model-size", default="70m")
    robust.add_argument("--model-name")
    robust.add_argument("--revision")
    robust.add_argument("--attn-implementation", default=None)
    robust.add_argument("--device-map", default="auto")
    robust.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    robust.add_argument("--batch-size", type=int, default=64)
    robust.add_argument("--max-length", type=int, default=128)
    robust.set_defaults(func=robustness)

    beh = sub.add_parser("behavior", help="score continuation preferences")
    beh.add_argument("--evals", required=True)
    beh.add_argument("--out", required=True)
    beh.add_argument("--spec")
    beh.add_argument("--model-size", default="70m")
    beh.add_argument("--model-name")
    beh.add_argument("--revision")
    beh.add_argument("--attn-implementation", default=None)
    beh.add_argument("--device-map", default="auto")
    beh.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    beh.set_defaults(func=behavior)

    gen = sub.add_parser("generate-targets", help="write target specs")
    gen.add_argument("--out", required=True)
    gen.add_argument("--tokens", nargs="*")
    gen.add_argument("--token-file")
    gen.add_argument("--layers")
    gen.add_argument("--neurons")
    gen.add_argument("--vector", nargs="*")
    gen.add_argument("--vector-layers")
    gen.add_argument("--default-vector-layer", type=int)
    gen.add_argument("--logit-prefix", default="logit")
    gen.add_argument("--neuron-prefix", default="neuron")
    gen.add_argument("--residual-prefix", default="residual")
    gen.add_argument("--model-name")
    gen.add_argument("--revision")
    gen.add_argument("--model-size")
    gen.add_argument("--texts-path")
    gen.add_argument("--attn-implementation")
    gen.add_argument("--device-map")
    gen.set_defaults(func=generate_targets)

    dirs = sub.add_parser(
        "fit-directions", help="fit residual directions across layers"
    )
    dirs.add_argument("--contrast", required=True)
    dirs.add_argument("--contrast-eval")
    dirs.add_argument("--layers", required=True, help="layer list/range or 'all'")
    dirs.add_argument("--out-dir", required=True)
    dirs.add_argument("--name", required=True)
    dirs.add_argument("--spec-out")
    dirs.add_argument("--top-k", type=int, default=3)
    dirs.add_argument(
        "--bidirectional",
        action="store_true",
        help="write paired decrease and increase targets for each selected layer",
    )
    dirs.add_argument("--texts-path")
    dirs.add_argument("--pooling", choices=["last", "mean"], default="last")
    dirs.add_argument("--max-length", type=int, default=256)
    dirs.add_argument("--batch-size", type=int, default=16)
    dirs.add_argument("--model-size", default="70m")
    dirs.add_argument("--model-name")
    dirs.add_argument("--revision")
    dirs.add_argument("--tokenizer-name")
    dirs.add_argument("--attn-implementation", default=None)
    dirs.add_argument("--device-map", default="auto")
    dirs.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    dirs.set_defaults(func=fit_directions)

    table = sub.add_parser("latex-table", help="convert a CSV summary to a LaTeX table")
    table.add_argument("--csv", required=True)
    table.add_argument("--out", required=True)
    table.add_argument("--columns")
    table.add_argument("--caption")
    table.add_argument("--label")
    table.set_defaults(func=latex_table)

    beh_tpl = sub.add_parser(
        "behavior-templates", help="write starter behavioral eval templates"
    )
    beh_tpl.add_argument("--out", required=True)
    beh_tpl.set_defaults(func=behavior_templates)

    frontier = sub.add_parser(
        "build-frontier-data",
        help="build local prompt pools and contrast pairs from frontier benchmark datasets",
    )
    frontier.add_argument("--out-dir", default="data/frontier")
    frontier.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCE_SPECS),
        default=["mmlu_pro", "math500"],
    )
    frontier.add_argument("--max-items-per-source", type=int, default=500)
    frontier.add_argument("--train-fraction", type=float, default=0.6)
    frontier.add_argument("--validation-fraction", type=float, default=0.2)
    frontier.add_argument("--behavior-limit", type=int, default=300)
    frontier.add_argument("--seed", type=int, default=0)
    frontier.add_argument(
        "--allow-gated",
        action="store_true",
        help="allow gated sources after accepting their Hugging Face terms",
    )
    frontier.set_defaults(func=frontier_data)

    reachable = sub.add_parser(
        "collect-reachable",
        help="collect behavior-preserving states under prompt and activation controls",
    )
    reachable.add_argument("--spec", required=True)
    reachable.add_argument("--out", required=True)
    reachable.set_defaults(func=collect_reachable)

    geometry = sub.add_parser(
        "analyze-reachability",
        help="recompute reachable-set geometry from a state archive",
    )
    geometry.add_argument("--states-dir", required=True)
    geometry.add_argument("--out", required=True)
    geometry.add_argument("--target-metric", default="target_projection")
    geometry.add_argument("--seed", type=int, default=0)
    geometry.set_defaults(func=analyze_reachability)

    monitors = sub.add_parser(
        "monitor-invariance",
        help="train natural and reachable-augmented monitors on a state archive",
    )
    monitors.add_argument("--states-dir", required=True)
    monitors.add_argument("--out-dir", required=True)
    monitors.add_argument("--label-key", default="monitor_label")
    monitors.add_argument(
        "--monitors",
        nargs="+",
        choices=[
            "linear",
            "random_linear",
            "last_linear",
            "mean_linear",
            "max_linear",
            "attention",
            "nonlinear",
            "multilayer_linear",
        ],
        default=["linear", "nonlinear"],
    )
    monitors.add_argument("--train-fraction", type=float, default=0.6)
    monitors.add_argument("--validation-fraction", type=float, default=0.2)
    monitors.add_argument("--reachable-weight", type=float, default=1.0)
    monitors.add_argument("--monitor-device", default="auto")
    monitors.add_argument("--seed", type=int, default=0)
    monitors.set_defaults(func=monitor_invariance_study)

    controls = sub.add_parser(
        "export-prompt-controls",
        help="export unique Pareto prompt controls from candidate records",
    )
    controls.add_argument("--records", required=True)
    controls.add_argument("--out", required=True)
    controls.add_argument("--target")
    controls.add_argument("--direction-sweep")
    controls.add_argument("--methods", nargs="+")
    controls.add_argument("--top-n", type=int, default=32)
    controls.add_argument(
        "--bidirectional",
        action="store_true",
        help="export balanced decrease and increase controls",
    )
    controls.add_argument("--maximize", action="store_true")
    controls.set_defaults(func=export_controls)

    study_spec = sub.add_parser(
        "build-study-spec",
        help="build a full controllability study spec from a fitted direction",
    )
    study_spec.add_argument("--out", required=True)
    study_spec.add_argument("--model-name", required=True)
    study_spec.add_argument("--revision")
    study_spec.add_argument("--data", required=True)
    study_spec.add_argument("--direction-sweep", required=True)
    study_spec.add_argument(
        "--layers",
        default="sweep",
        help="capture list/range, 'target', or 'sweep' for five depth-spaced layers",
    )
    study_spec.add_argument("--prompt-controls", required=True)
    study_spec.add_argument("--natural-controls", required=True)
    study_spec.add_argument("--example-limit", type=int)
    study_spec.add_argument("--attn-implementation")
    study_spec.add_argument("--device-map", default="auto")
    study_spec.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    study_spec.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[
            -16.0,
            -8.0,
            -4.0,
            -2.0,
            -1.0,
            -0.5,
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
        ],
    )
    study_spec.add_argument(
        "--ablation-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
    )
    study_spec.add_argument("--max-new-tokens", type=int, default=128)
    study_spec.add_argument("--maximum-quality-drop", type=float, default=0.75)
    study_spec.add_argument("--maximum-control-cost", type=float, default=64.0)
    study_spec.add_argument("--semantic-model")
    study_spec.add_argument("--minimum-semantic-similarity", type=float, default=0.8)
    study_spec.add_argument("--store-token-states", action="store_true")
    study_spec.add_argument(
        "--cmap-directions",
        type=int,
        default=0,
        help="enable C-MAP with this active direction budget",
    )
    study_spec.add_argument("--cmap-query-budget", type=int, default=512)
    study_spec.add_argument("--cmap-validation-examples", type=int, default=8)
    study_spec.add_argument("--cmap-test-examples", type=int, default=16)
    study_spec.add_argument("--seed", type=int, default=0)
    study_spec.set_defaults(func=build_controllability_spec)

    transfer = sub.add_parser(
        "analyze-transfer",
        help="test whether subspace overlap predicts intervention transfer",
    )
    transfer.add_argument("--states-dir", required=True)
    transfer.add_argument("--out", required=True)
    transfer.add_argument("--group-tag", default="source")
    transfer.add_argument("--target-metric", default="target_projection")
    transfer.set_defaults(func=transfer_study)

    causal = sub.add_parser(
        "causal-study",
        help="run matched cross-channel activation patching",
    )
    causal.add_argument("--spec", required=True)
    causal.add_argument("--states-dir", required=True)
    causal.add_argument("--out-dir", required=True)
    causal.add_argument("--selection-layer", type=int)
    causal.add_argument(
        "--patch-layers",
        default="spec",
        help="layer list/range or 'spec' to use the study capture layers",
    )
    causal.add_argument("--prompt-prefix", default="optimized_prompt")
    causal.add_argument("--activation-prefix", default="activation_addition")
    causal.add_argument("--target-metric", default="target_projection")
    causal.add_argument("--max-pairs", type=int, default=32)
    causal.add_argument("--max-heads-per-layer", type=int, default=8)
    causal.add_argument("--seed", type=int, default=0)
    causal.set_defaults(func=causal_study)

    control = sub.add_parser(
        "analyze-control",
        help="measure channel-specific setpoint reachability and minimum cost",
    )
    control.add_argument("--spec", required=True)
    control.add_argument("--states-dir", required=True)
    control.add_argument("--out-dir", required=True)
    control.add_argument("--target-metric", default="target_projection")
    control.add_argument("--tolerance-fraction", type=float, default=0.1)
    control.set_defaults(func=control_study)

    jacobians = sub.add_parser(
        "analyze-jacobians",
        help="estimate local residual-control Jacobians by finite differences",
    )
    jacobians.add_argument("--spec", required=True)
    jacobians.add_argument("--out", required=True)
    jacobians.add_argument("--example-limit", type=int, default=16)
    jacobians.add_argument("--epsilon", type=float, default=0.25)
    jacobians.add_argument(
        "--basis-dimensions",
        nargs="+",
        type=int,
        default=[8, 16, 32],
        help="nested orthonormal residual-control dimensions",
    )
    jacobians.add_argument("--seed", type=int, default=0)
    jacobians.set_defaults(func=jacobian_study)

    figures = sub.add_parser(
        "render-study-figures",
        help="render predeclared figures from completed study artifacts",
    )
    figures.add_argument("--run-dir", required=True)
    figures.add_argument("--out-dir", required=True)
    figures.set_defaults(func=render_figures)

    matrix_figures = sub.add_parser(
        "render-matrix-figures",
        help="render predeclared cross-model discovery figures",
    )
    matrix_figures.add_argument("--matrix-dir", required=True)
    matrix_figures.add_argument("--out-dir", required=True)
    matrix_figures.set_defaults(func=render_discovery_figures)

    matrix = sub.add_parser(
        "aggregate-study-matrix",
        help="validate and aggregate the declared multi-model study matrix",
    )
    matrix.add_argument("--run-root", default="runs/controllability")
    matrix.add_argument(
        "--matrix",
        default="configs/recommended_matrix.json",
    )
    matrix.add_argument("--out-dir", required=True)
    matrix.set_defaults(func=aggregate_study_matrix)

    matrix_run = sub.add_parser(
        "run-study-matrix",
        help="launch model studies declared in the matrix configuration",
    )
    matrix_run.add_argument(
        "--matrix",
        default="configs/recommended_matrix.json",
    )
    matrix_run.add_argument("--analysis", choices=ANALYSES, required=True)
    matrix_run.add_argument("--project-root", default=".")
    matrix_run.add_argument("--models", nargs="*")
    matrix_run.add_argument("--dry-run", action="store_true")
    matrix_run.add_argument("--no-aggregate", action="store_true")
    matrix_run.set_defaults(func=run_study_matrix)

    scope = sub.add_parser(
        "gemma-scope",
        help="analyze Gemma 3 reachable states with a Gemma Scope 2 SAE",
    )
    scope.add_argument("--states-dir", required=True)
    scope.add_argument("--out-dir", required=True)
    scope.add_argument("--model-name", required=True)
    layer_group = scope.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layer", type=int)
    layer_group.add_argument("--direction-sweep")
    scope.add_argument("--release")
    scope.add_argument("--sae-id")
    scope.add_argument("--site", default="resid_post_all")
    scope.add_argument("--width", choices=["16k", "262k"], default="16k")
    scope.add_argument("--l0", choices=["small", "big"], default="small")
    scope.add_argument("--device", default="auto")
    scope.add_argument("--top-k", type=int, default=128)
    scope.add_argument("--analysis-features", type=int, default=2048)
    scope.add_argument("--batch-size", type=int, default=32)
    scope.add_argument("--max-samples", type=int, default=4096)
    scope.add_argument("--include-unpreserved", action="store_true")
    scope.set_defaults(func=gemma_scope_study)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
