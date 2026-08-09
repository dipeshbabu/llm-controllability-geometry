"""Matched cross-channel activation patching study."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm_controllability.causal import (
    ActivationCache,
    AttentionHeadAblation,
    ComponentAblation,
    ModuleOutputCache,
    ModuleOutputPatching,
    StatePatching,
)
from llm_controllability.controllability.study import build_interventions
from llm_controllability.controllability.types import ControlChannel, StateSample
from llm_controllability.interventions.core import Intervention
from llm_controllability.models.adapters import (
    ensure_padding_token,
    model_device,
    tokenize_prompts,
)
from llm_controllability.models.architecture import (
    get_attention_head_layout,
    get_attention_module,
    get_layers,
    get_mlp_output_projection,
)
from llm_controllability.models.loading import load_model
from llm_controllability.reachability.collection import load_examples
from llm_controllability.reachability.io import load_state_samples


def _resolve(path: str, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base_dir / value


def matched_control_pairs(
    samples: Sequence[StateSample],
    *,
    selection_layer: int,
    prompt_prefix: str,
    activation_prefix: str,
    target_metric: str,
) -> list[tuple[StateSample, StateSample]]:
    """Choose the nearest target matched prompt and activation pair per example."""

    grouped: dict[str, list[StateSample]] = defaultdict(list)
    for sample in samples:
        if (
            sample.layer == selection_layer
            and sample.behavior_preserved
            and target_metric in sample.metrics
        ):
            grouped[sample.example_id].append(sample)

    pairs = []
    for example_id in sorted(grouped):
        group = grouped[example_id]
        prompts = [
            sample
            for sample in group
            if sample.intervention.channel is ControlChannel.PROMPT
            and sample.intervention.name.startswith(prompt_prefix)
        ]
        activations = [
            sample
            for sample in group
            if sample.intervention.channel is ControlChannel.ACTIVATION
            and sample.intervention.name.startswith(activation_prefix)
        ]
        if not prompts or not activations:
            continue
        pairs.append(
            min(
                ((prompt, activation) for prompt in prompts for activation in activations),
                key=lambda pair: abs(
                    float(pair[0].metrics[target_metric])
                    - float(pair[1].metrics[target_metric])
                ),
            )
        )
    return pairs


def matched_prompt_pairs(
    samples: Sequence[StateSample],
    *,
    selection_layer: int,
) -> list[tuple[str, StateSample, StateSample]]:
    """Pair neutral and evaluation baseline states for the same question."""

    grouped: dict[str, list[StateSample]] = defaultdict(list)
    for sample in samples:
        pair_id = sample.tags.get("pair_id")
        if (
            pair_id is not None
            and sample.layer == selection_layer
            and sample.behavior_preserved
            and sample.intervention.channel is ControlChannel.BASELINE
            and "monitor_label" in sample.metrics
        ):
            grouped[pair_id].append(sample)
    pairs = []
    for pair_id, group in sorted(grouped.items()):
        neutral = [
            sample
            for sample in group
            if int(sample.metrics["monitor_label"]) == 0
        ]
        evaluation = [
            sample
            for sample in group
            if int(sample.metrics["monitor_label"]) == 1
        ]
        if neutral and evaluation:
            pairs.append((pair_id, neutral[0], evaluation[0]))
    return pairs


def _generate_ids(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    intervention: Intervention,
    generation: Mapping[str, Any],
    *,
    max_length: int,
) -> torch.Tensor:
    prepared = intervention.prepare_prompt(prompt)
    max_new_tokens = int(generation.get("max_new_tokens", 64))
    prompt_limit = max(1, max_length - max_new_tokens)
    tokens = tokenize_prompts(
        tokenizer,
        prepared,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_limit,
    ).to(model_device(model))
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(generation.get("do_sample", False)),
        "num_beams": int(generation.get("num_beams", 1)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(generation.get("temperature", 1.0))
        kwargs["top_p"] = float(generation.get("top_p", 1.0))
    intervention.reset()
    with intervention.apply(model), torch.no_grad():
        return model.generate(**tokens, **kwargs)


def _forward_with_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    intervention: Intervention,
    layers: Sequence[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    cache = ActivationCache(layers, detach_to_cpu=True)
    intervention.reset()
    with intervention.apply(model), cache.capture(model), torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )
    logits = output.logits[:, -1].detach().float().cpu()
    return logits, cache.values


def _patched_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target: Intervention,
    *,
    layer: int,
    source_state: torch.Tensor,
) -> torch.Tensor:
    target.reset()
    patch = StatePatching(layer, source_state, token_scope="last")
    with target.apply(model), patch.apply(model), torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )
    return output.logits[:, -1].detach().float().cpu()


def _ablated_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    intervention: Intervention,
    module: torch.nn.Module,
) -> torch.Tensor:
    intervention.reset()
    with intervention.apply(model), ComponentAblation(module).apply(), torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )
    return output.logits[:, -1].detach().float().cpu()


def _head_ablated_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    intervention: Intervention,
    block: torch.nn.Module,
    head: int,
) -> torch.Tensor:
    intervention.reset()
    with (
        intervention.apply(model),
        AttentionHeadAblation(block, head).apply(),
        torch.no_grad(),
    ):
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )
    return output.logits[:, -1].detach().float().cpu()


def _receiver_modules(
    model: torch.nn.Module,
    layers: Sequence[int],
) -> dict[str, torch.nn.Module]:
    blocks = get_layers(model)
    modules = {}
    for layer in layers:
        modules[f"{layer}:attention"] = get_attention_module(blocks[layer])
        modules[f"{layer}:mlp"] = get_mlp_output_projection(blocks[layer])
    return modules


def _capture_module_outputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    intervention: Intervention,
    modules: Mapping[str, torch.nn.Module],
) -> dict[str, torch.Tensor]:
    cache = ModuleOutputCache(modules)
    intervention.reset()
    with intervention.apply(model), cache.capture(), torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )
    return cache.values


def _receiver_blocked_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target: Intervention,
    *,
    source_layer: int,
    source_state: torch.Tensor,
    receiver_module: torch.nn.Module,
    receiver_reference: torch.Tensor,
) -> torch.Tensor:
    target.reset()
    sender_patch = StatePatching(source_layer, source_state, token_scope="last")
    receiver_patch = ModuleOutputPatching(
        receiver_module,
        receiver_reference,
    )
    with (
        target.apply(model),
        sender_patch.apply(model),
        receiver_patch.apply(),
        torch.no_grad(),
    ):
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )
    return output.logits[:, -1].detach().float().cpu()


def _js_divergence(first_logits: torch.Tensor, second_logits: torch.Tensor) -> float:
    first_log = torch.log_softmax(first_logits, dim=-1)
    second_log = torch.log_softmax(second_logits, dim=-1)
    first = first_log.exp()
    second = second_log.exp()
    mixture = 0.5 * (first + second)
    mixture_log = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    value = 0.5 * (
        (first * (first_log - mixture_log)).sum(dim=-1)
        + (second * (second_log - mixture_log)).sum(dim=-1)
    )
    return float(value.mean())


def _patch_rows(
    model: torch.nn.Module,
    *,
    example_id: str,
    source_name: str,
    target_name: str,
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    source_cache: Mapping[int, torch.Tensor],
    target_ids: torch.Tensor,
    target_intervention: Intervention,
    layers: Sequence[int],
    direction: str,
) -> list[dict[str, Any]]:
    denominator = float(torch.linalg.vector_norm(source_logits - target_logits))
    source_target_js = _js_divergence(source_logits, target_logits)
    rows = []
    for layer in layers:
        patched = _patched_logits(
            model,
            target_ids,
            target_intervention,
            layer=layer,
            source_state=source_cache[layer],
        )
        source_distance = float(torch.linalg.vector_norm(source_logits - patched))
        rows.append(
            {
                "example_id": example_id,
                "direction": direction,
                "source_intervention": source_name,
                "target_intervention": target_name,
                "layer": layer,
                "source_target_logit_distance": denominator,
                "patched_source_logit_distance": source_distance,
                "logit_recovery": (
                    1.0 - source_distance / denominator
                    if denominator > 0
                    else float("nan")
                ),
                "patch_effect": float(
                    torch.linalg.vector_norm(patched - target_logits)
                ),
                "source_target_js": source_target_js,
                "patched_source_js": _js_divergence(patched, source_logits),
                "source_top1": int(source_logits.argmax(dim=-1)[0]),
                "target_top1": int(target_logits.argmax(dim=-1)[0]),
                "patched_top1": int(patched.argmax(dim=-1)[0]),
            }
        )
    return rows


def _ablation_rows(
    model: torch.nn.Module,
    *,
    example_id: str,
    intervention: Intervention,
    input_ids: torch.Tensor,
    baseline_logits: torch.Tensor,
    layers: Sequence[int],
    channel: str,
) -> list[dict[str, Any]]:
    rows = []
    blocks = get_layers(model)
    for layer in layers:
        modules = {
            "attention": get_attention_module(blocks[layer]),
            "mlp": get_mlp_output_projection(blocks[layer]),
        }
        for component, module in modules.items():
            ablated = _ablated_logits(
                model,
                input_ids,
                intervention,
                module,
            )
            rows.append(
                {
                    "example_id": example_id,
                    "intervention": intervention.name,
                    "channel": channel,
                    "layer": layer,
                    "component": component,
                    "ablation_effect": float(
                        torch.linalg.vector_norm(ablated - baseline_logits)
                    ),
                    "ablation_js": _js_divergence(ablated, baseline_logits),
                    "baseline_top1": int(baseline_logits.argmax(dim=-1)[0]),
                    "ablated_top1": int(ablated.argmax(dim=-1)[0]),
                    "top1_changed": bool(
                        baseline_logits.argmax(dim=-1)[0]
                        != ablated.argmax(dim=-1)[0]
                    ),
                }
            )
    return rows


def _head_ablation_rows(
    model: torch.nn.Module,
    *,
    example_id: str,
    intervention: Intervention,
    input_ids: torch.Tensor,
    baseline_logits: torch.Tensor,
    layers: Sequence[int],
    channel: str,
    max_heads_per_layer: int,
) -> list[dict[str, Any]]:
    rows = []
    blocks = get_layers(model)
    for layer in layers:
        head_count, _ = get_attention_head_layout(blocks[layer])
        selected = np.unique(
            np.rint(
                np.linspace(
                    0,
                    head_count - 1,
                    num=min(max_heads_per_layer, head_count),
                )
            ).astype(int)
        )
        for head in selected:
            ablated = _head_ablated_logits(
                model,
                input_ids,
                intervention,
                blocks[layer],
                int(head),
            )
            rows.append(
                {
                    "example_id": example_id,
                    "intervention": intervention.name,
                    "channel": channel,
                    "layer": layer,
                    "head": int(head),
                    "n_heads": head_count,
                    "ablation_effect": float(
                        torch.linalg.vector_norm(ablated - baseline_logits)
                    ),
                    "ablation_js": _js_divergence(
                        ablated,
                        baseline_logits,
                    ),
                    "top1_changed": bool(
                        baseline_logits.argmax(dim=-1)[0]
                        != ablated.argmax(dim=-1)[0]
                    ),
                }
            )
    return rows


def _path_mediation_rows(
    model: torch.nn.Module,
    *,
    example_id: str,
    source_name: str,
    target_name: str,
    source_cache: Mapping[int, torch.Tensor],
    target_ids: torch.Tensor,
    target_intervention: Intervention,
    target_logits: torch.Tensor,
    layers: Sequence[int],
    direction: str,
) -> list[dict[str, Any]]:
    rows = []
    ordered = sorted(set(layers))
    receiver_layers = ordered[1:]
    modules = _receiver_modules(model, receiver_layers)
    references = _capture_module_outputs(
        model,
        target_ids,
        target_intervention,
        modules,
    )
    for source_layer, receiver_layer in zip(ordered, receiver_layers):
        sender_patched = _patched_logits(
            model,
            target_ids,
            target_intervention,
            layer=source_layer,
            source_state=source_cache[source_layer],
        )
        total_effect = float(
            torch.linalg.vector_norm(sender_patched - target_logits)
        )
        for component in ("attention", "mlp"):
            name = f"{receiver_layer}:{component}"
            blocked = _receiver_blocked_logits(
                model,
                target_ids,
                target_intervention,
                source_layer=source_layer,
                source_state=source_cache[source_layer],
                receiver_module=modules[name],
                receiver_reference=references[name],
            )
            mediated = float(
                torch.linalg.vector_norm(sender_patched - blocked)
            )
            rows.append(
                {
                    "example_id": example_id,
                    "direction": direction,
                    "source_intervention": source_name,
                    "target_intervention": target_name,
                    "source_layer": source_layer,
                    "receiver_layer": receiver_layer,
                    "receiver_component": component,
                    "sender_patch_effect": total_effect,
                    "receiver_blocked_effect": float(
                        torch.linalg.vector_norm(blocked - target_logits)
                    ),
                    "mediated_effect": mediated,
                    "mediated_fraction": (
                        mediated / total_effect
                        if total_effect > 0
                        else float("nan")
                    ),
                    "sender_patch_js": _js_divergence(
                        sender_patched,
                        target_logits,
                    ),
                    "receiver_blocked_js": _js_divergence(
                        blocked,
                        target_logits,
                    ),
                }
            )
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_causal_study(
    spec_path: str | Path,
    states_dir: str | Path,
    out_dir: str | Path,
    *,
    selection_layer: int | None = None,
    patch_layers: Sequence[int] | None = None,
    prompt_prefix: str = "optimized_prompt",
    activation_prefix: str = "activation_addition",
    target_metric: str = "target_projection",
    max_pairs: int = 32,
    max_heads_per_layer: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    spec_path = Path(spec_path)
    base_dir = spec_path.resolve().parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    target_layers = [int(layer) for layer in spec.get("target_directions", {})]
    if selection_layer is None:
        if len(target_layers) != 1:
            raise ValueError("selection layer is ambiguous; pass --selection-layer")
        selection_layer = target_layers[0]
    if patch_layers is None:
        patch_layers = [int(layer) for layer in spec["layers"]]

    samples = load_state_samples(states_dir)
    if any(sample.tags.get("split") == "test" for sample in samples):
        samples = [
            sample for sample in samples if sample.tags.get("split") == "test"
        ]
    pairs = matched_control_pairs(
        samples,
        selection_layer=selection_layer,
        prompt_prefix=prompt_prefix,
        activation_prefix=activation_prefix,
        target_metric=target_metric,
    )
    pairs.sort(
        key=lambda pair: (
            abs(
                float(pair[0].metrics[target_metric])
                - float(pair[1].metrics[target_metric])
            ),
            pair[0].example_id,
        )
    )
    pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("no matched behavior-preserving control pairs were found")
    prompt_pairs = matched_prompt_pairs(
        samples,
        selection_layer=selection_layer,
    )
    prompt_pairs = prompt_pairs[:max_pairs]

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
    tokenizer, _ = ensure_padding_token(tokenizer)
    interventions = {
        intervention.name: intervention
        for intervention in build_interventions(
            spec["interventions"],
            base_dir=base_dir,
        )
    }
    examples = {
        str(example.get("id", index)): example
        for index, example in enumerate(
            load_examples(_resolve(spec["data"]["path"], base_dir))
        )
    }

    rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    cross_prompt_rows: list[dict[str, Any]] = []
    for prompt_sample, activation_sample in pairs:
        example_id = prompt_sample.example_id
        if example_id not in examples:
            raise ValueError(f"state archive example {example_id!r} is absent from the spec data")
        prompt_control = interventions[prompt_sample.intervention.name]
        activation_control = interventions[activation_sample.intervention.name]
        prompt = str(examples[example_id]["prompt"])
        prompt_ids = _generate_ids(
            model,
            tokenizer,
            prompt,
            prompt_control,
            spec.get("generation", {}),
            max_length=int(spec.get("max_length", 2048)),
        )
        activation_ids = _generate_ids(
            model,
            tokenizer,
            prompt,
            activation_control,
            spec.get("generation", {}),
            max_length=int(spec.get("max_length", 2048)),
        )
        prompt_logits, prompt_cache = _forward_with_cache(
            model,
            prompt_ids,
            prompt_control,
            patch_layers,
        )
        activation_logits, activation_cache = _forward_with_cache(
            model,
            activation_ids,
            activation_control,
            patch_layers,
        )
        pair_gap = abs(
            float(prompt_sample.metrics[target_metric])
            - float(activation_sample.metrics[target_metric])
        )
        forward_rows = _patch_rows(
            model,
            example_id=example_id,
            source_name=prompt_control.name,
            target_name=activation_control.name,
            source_logits=prompt_logits,
            target_logits=activation_logits,
            source_cache=prompt_cache,
            target_ids=activation_ids,
            target_intervention=activation_control,
            layers=patch_layers,
            direction="prompt_to_activation",
        )
        reverse_rows = _patch_rows(
            model,
            example_id=example_id,
            source_name=activation_control.name,
            target_name=prompt_control.name,
            source_logits=activation_logits,
            target_logits=prompt_logits,
            source_cache=activation_cache,
            target_ids=prompt_ids,
            target_intervention=prompt_control,
            layers=patch_layers,
            direction="activation_to_prompt",
        )
        for row in forward_rows + reverse_rows:
            row["selection_layer"] = selection_layer
            row["target_match_gap"] = pair_gap
            rows.append(row)
        pair_path_rows = _path_mediation_rows(
            model,
            example_id=example_id,
            source_name=prompt_control.name,
            target_name=activation_control.name,
            source_cache=prompt_cache,
            target_ids=activation_ids,
            target_intervention=activation_control,
            target_logits=activation_logits,
            layers=patch_layers,
            direction="prompt_to_activation",
        ) + _path_mediation_rows(
            model,
            example_id=example_id,
            source_name=activation_control.name,
            target_name=prompt_control.name,
            source_cache=activation_cache,
            target_ids=prompt_ids,
            target_intervention=prompt_control,
            target_logits=prompt_logits,
            layers=patch_layers,
            direction="activation_to_prompt",
        )
        for row in pair_path_rows:
            row["selection_layer"] = selection_layer
            row["target_match_gap"] = pair_gap
            path_rows.append(row)
        pair_ablation_rows = _ablation_rows(
            model,
            example_id=example_id,
            intervention=prompt_control,
            input_ids=prompt_ids,
            baseline_logits=prompt_logits,
            layers=patch_layers,
            channel="prompt",
        ) + _ablation_rows(
            model,
            example_id=example_id,
            intervention=activation_control,
            input_ids=activation_ids,
            baseline_logits=activation_logits,
            layers=patch_layers,
            channel="activation",
        )
        for row in pair_ablation_rows:
            row["selection_layer"] = selection_layer
            row["target_match_gap"] = pair_gap
            ablation_rows.append(row)
        pair_head_rows = _head_ablation_rows(
            model,
            example_id=example_id,
            intervention=prompt_control,
            input_ids=prompt_ids,
            baseline_logits=prompt_logits,
            layers=patch_layers,
            channel="prompt",
            max_heads_per_layer=max_heads_per_layer,
        ) + _head_ablation_rows(
            model,
            example_id=example_id,
            intervention=activation_control,
            input_ids=activation_ids,
            baseline_logits=activation_logits,
            layers=patch_layers,
            channel="activation",
            max_heads_per_layer=max_heads_per_layer,
        )
        for row in pair_head_rows:
            row["selection_layer"] = selection_layer
            row["target_match_gap"] = pair_gap
            head_rows.append(row)

    baseline_control = interventions["baseline"]
    for pair_id, neutral_sample, evaluation_sample in prompt_pairs:
        neutral_prompt = str(examples[neutral_sample.example_id]["prompt"])
        evaluation_prompt = str(
            examples[evaluation_sample.example_id]["prompt"]
        )
        neutral_ids = _generate_ids(
            model,
            tokenizer,
            neutral_prompt,
            baseline_control,
            spec.get("generation", {}),
            max_length=int(spec.get("max_length", 2048)),
        )
        evaluation_ids = _generate_ids(
            model,
            tokenizer,
            evaluation_prompt,
            baseline_control,
            spec.get("generation", {}),
            max_length=int(spec.get("max_length", 2048)),
        )
        neutral_logits, neutral_cache = _forward_with_cache(
            model,
            neutral_ids,
            baseline_control,
            patch_layers,
        )
        evaluation_logits, evaluation_cache = _forward_with_cache(
            model,
            evaluation_ids,
            baseline_control,
            patch_layers,
        )
        pair_rows = _patch_rows(
            model,
            example_id=pair_id,
            source_name=neutral_sample.example_id,
            target_name=evaluation_sample.example_id,
            source_logits=neutral_logits,
            target_logits=evaluation_logits,
            source_cache=neutral_cache,
            target_ids=evaluation_ids,
            target_intervention=baseline_control,
            layers=patch_layers,
            direction="neutral_to_evaluation",
        ) + _patch_rows(
            model,
            example_id=pair_id,
            source_name=evaluation_sample.example_id,
            target_name=neutral_sample.example_id,
            source_logits=evaluation_logits,
            target_logits=neutral_logits,
            source_cache=evaluation_cache,
            target_ids=neutral_ids,
            target_intervention=baseline_control,
            layers=patch_layers,
            direction="evaluation_to_neutral",
        )
        for row in pair_rows:
            row["pair_id"] = pair_id
            row["selection_layer"] = selection_layer
            cross_prompt_rows.append(row)

    out_dir = Path(out_dir)
    _write_csv(rows, out_dir / "patching.csv")
    _write_csv(ablation_rows, out_dir / "component_ablation.csv")
    _write_csv(head_rows, out_dir / "head_ablation.csv")
    _write_csv(path_rows, out_dir / "path_mediation.csv")
    _write_csv(
        cross_prompt_rows,
        out_dir / "cross_prompt_patching.csv",
    )
    manifest = {
        "model": model_config["name"],
        "selection_layer": selection_layer,
        "patch_layers": list(patch_layers),
        "n_pairs": len(pairs),
        "n_rows": len(rows),
        "n_ablation_rows": len(ablation_rows),
        "n_head_ablation_rows": len(head_rows),
        "n_path_rows": len(path_rows),
        "n_cross_prompt_pairs": len(prompt_pairs),
        "n_cross_prompt_rows": len(cross_prompt_rows),
        "max_heads_per_layer": max_heads_per_layer,
        "prompt_prefix": prompt_prefix,
        "activation_prefix": activation_prefix,
        "target_metric": target_metric,
        "selection_policy": "closest_target_match_then_example_id",
        "artifacts": [
            "patching.csv",
            "component_ablation.csv",
            "head_ablation.csv",
            "path_mediation.csv",
            "cross_prompt_patching.csv",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
