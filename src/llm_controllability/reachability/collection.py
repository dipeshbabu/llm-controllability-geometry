"""Execute interventions and collect behavior-gated residual states."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from llm_controllability.constraints import BehaviorGate, BehaviorRecord
from llm_controllability.constraints.verification import (
    TransformerSentenceEmbedder,
    verify_output,
)
from llm_controllability.controllability.types import InterventionMetadata, StateSample
from llm_controllability.data.behavior import continuation_logprob
from llm_controllability.interventions.core import Intervention, _hidden_from_output
from llm_controllability.models.adapters import (
    ensure_padding_token,
    last_nonpadding_indices,
    model_device,
    tokenize_prompt_with_continuation,
    tokenize_prompts,
)
from llm_controllability.models.architecture import get_layers


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("examples", data.get("evals"))
    if not isinstance(data, list):
        raise TypeError("example file must contain a list or an 'examples' list")
    return data


def generate_completion(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    intervention: Intervention,
    generation: Mapping[str, Any],
) -> tuple[str, str]:
    """Generate deterministically under one control setting."""

    prepared_prompt = intervention.prepare_prompt(prompt)
    tokens = tokenize_prompts(
        tokenizer,
        prepared_prompt,
        return_tensors="pt",
    ).to(model_device(model))
    kwargs = {
        "max_new_tokens": int(generation.get("max_new_tokens", 64)),
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
        generated = model.generate(**tokens, **kwargs)
    continuation_ids = generated[0, tokens["input_ids"].shape[1] :]
    return prepared_prompt, tokenizer.decode(continuation_ids, skip_special_tokens=True)


def run_and_capture(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    intervention: Intervention,
    layers: Sequence[int],
    generation: Mapping[str, Any],
    *,
    pooling: Literal["last", "mean", "max"] = "last",
    max_length: int = 2048,
    store_token_states: bool = False,
) -> tuple[
    str,
    str,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[str, float],
]:
    """Generate once, then recapture that exact sequence under the same control."""

    intervention.reset()
    prepared_prompt = intervention.prepare_prompt(prompt)
    max_new_tokens = int(generation.get("max_new_tokens", 64))
    prompt_limit = max(1, max_length - max_new_tokens)
    prompt_tokens = tokenize_prompts(
        tokenizer,
        prepared_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_limit,
    ).to(model_device(model))
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(generation.get("do_sample", False)),
        "num_beams": int(generation.get("num_beams", 1)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(generation.get("temperature", 1.0))
        generation_kwargs["top_p"] = float(generation.get("top_p", 1.0))

    with intervention.apply(model):
        with torch.no_grad():
            generated = model.generate(**prompt_tokens, **generation_kwargs)
        generation_diagnostics = intervention.diagnostics()
        generation_control_cost = intervention.control_cost(tokenizer)
        # Recapture from the controller's declared initial state. Carrying
        # feedback history from generation into a full-sequence replay would
        # create a state that occurred in neither execution.
        intervention.reset()

        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:

            def hook(module, inputs, block_output, layer=layer):
                captured[layer] = _hidden_from_output(block_output)

            handles.append(get_layers(model)[layer].register_forward_hook(hook))
        try:
            full_mask = torch.ones_like(generated, dtype=torch.long)
            with torch.no_grad():
                model(input_ids=generated, attention_mask=full_mask)
        finally:
            for handle in handles:
                handle.remove()

    continuation_ids = generated[0, prompt_tokens["input_ids"].shape[1] :]
    output = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    result: dict[int, np.ndarray] = {}
    token_states: dict[int, np.ndarray] = {}
    for layer in layers:
        hidden = captured[layer]
        if pooling == "last":
            pooled = hidden[:, -1]
        elif pooling == "mean":
            pooled = hidden.mean(dim=1)
        elif pooling == "max":
            pooled = hidden.max(dim=1).values
        else:
            raise ValueError(f"unknown pooling method: {pooling}")
        result[layer] = pooled[0].detach().float().cpu().numpy()
        if store_token_states:
            token_states[layer] = hidden[0].detach().float().cpu().numpy()
    execution = {
        "control_cost": generation_control_cost,
        **generation_diagnostics,
    }
    return prepared_prompt, output, result, token_states, execution


def capture_residual_states(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    output: str,
    layers: Sequence[int],
    intervention: Intervention,
    *,
    pooling: Literal["last", "mean", "max"] = "last",
    max_length: int = 2048,
) -> dict[int, np.ndarray]:
    """Teacher-force a generated trajectory and pool each requested layer."""

    tokens = tokenize_prompt_with_continuation(
        tokenizer,
        prompt,
        output,
        max_length=max_length,
    )
    tokens = {key: value.to(model_device(model)) for key, value in tokens.items()}
    captured: dict[int, torch.Tensor] = {}
    handles = []
    intervention.reset()
    with intervention.apply(model):
        for layer in layers:

            def hook(module, inputs, block_output, layer=layer):
                captured[layer] = _hidden_from_output(block_output)

            handles.append(get_layers(model)[layer].register_forward_hook(hook))
        try:
            with torch.no_grad():
                model(**tokens)
        finally:
            for handle in handles:
                handle.remove()

    mask = tokens.get("attention_mask")
    result = {}
    for layer in layers:
        hidden = captured[layer]
        if pooling == "last":
            if mask is None:
                pooled = hidden[:, -1]
            else:
                index = last_nonpadding_indices(mask)
                pooled = hidden[
                    torch.arange(hidden.shape[0], device=hidden.device), index
                ]
        elif pooling == "mean":
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                weights = mask.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        elif pooling == "max":
            if mask is None:
                pooled = hidden.max(dim=1).values
            else:
                invalid = ~mask.bool().unsqueeze(-1)
                pooled = (
                    hidden.masked_fill(invalid, torch.finfo(hidden.dtype).min)
                    .max(dim=1)
                    .values
                )
        else:
            raise ValueError(f"unknown pooling method: {pooling}")
        result[layer] = pooled[0].detach().float().cpu().numpy()
    return result


def _quality_score(model, tokenizer, prompt: str, output: str) -> float | None:
    if not output:
        return None
    try:
        return float(
            continuation_logprob(model, tokenizer, prompt, output)["avg_logprob"]
        )
    except ValueError:
        return None


def collect_reachable_states(
    model: torch.nn.Module,
    tokenizer,
    *,
    model_name: str,
    examples: Sequence[Mapping[str, Any]],
    interventions: Sequence[Intervention],
    layers: Sequence[int],
    behavior_gate: BehaviorGate,
    generation: Mapping[str, Any],
    pooling: Literal["last", "mean", "max"] = "last",
    max_length: int = 2048,
    semantic_embedder: TransformerSentenceEmbedder | None = None,
    target_directions: Mapping[int, np.ndarray] | None = None,
    store_token_states: bool = False,
    seed: int = 0,
) -> list[StateSample]:
    """Collect baseline and intervention states with hard behavior verdicts."""

    tokenizer, _ = ensure_padding_token(tokenizer)
    samples: list[StateSample] = []
    target_directions = target_directions or {}
    for index, example in enumerate(examples):
        example_id = str(example.get("id", index))
        prompt = str(example["prompt"])
        baseline_intervention = next(
            (
                intervention
                for intervention in interventions
                if intervention.channel.value == "baseline"
            ),
            None,
        )
        if baseline_intervention is None:
            raise ValueError("intervention list must include a baseline")

        (
            base_prompt,
            base_output,
            base_states,
            base_token_states,
            base_execution,
        ) = run_and_capture(
            model,
            tokenizer,
            prompt,
            baseline_intervention,
            layers,
            generation,
            pooling=pooling,
            max_length=max_length,
            store_token_states=store_token_states,
        )
        base_task_score, base_correct = verify_output(base_output, example)
        if semantic_embedder is not None:
            base_prompt_embedding, base_embedding = semantic_embedder.encode(
                [base_prompt, base_output]
            )
        else:
            base_prompt_embedding = None
            base_embedding = None
        reference = BehaviorRecord(
            output=base_output,
            task_score=base_task_score,
            task_correct=base_correct,
            quality_score=_quality_score(model, tokenizer, prompt, base_output),
            embedding=base_embedding,
            metadata={"prompt_embedding": base_prompt_embedding},
        )

        for intervention in interventions:
            if intervention is baseline_intervention:
                prepared_prompt, output = base_prompt, base_output
                record = reference
                states = base_states
                token_states = base_token_states
                execution = base_execution
            else:
                (
                    prepared_prompt,
                    output,
                    states,
                    token_states,
                    execution,
                ) = run_and_capture(
                    model,
                    tokenizer,
                    prompt,
                    intervention,
                    layers,
                    generation,
                    pooling=pooling,
                    max_length=max_length,
                    store_token_states=store_token_states,
                )
                task_score, task_correct = verify_output(output, example)
                if semantic_embedder is not None:
                    prompt_embedding, embedding = semantic_embedder.encode(
                        [prepared_prompt, output]
                    )
                else:
                    prompt_embedding = None
                    embedding = None
                record = BehaviorRecord(
                    output=output,
                    task_score=task_score,
                    task_correct=task_correct,
                    quality_score=_quality_score(
                        model,
                        tokenizer,
                        prepared_prompt,
                        output,
                    ),
                    embedding=embedding,
                    metadata={"prompt_embedding": prompt_embedding},
                )

            metadata = InterventionMetadata(
                name=intervention.name,
                channel=intervention.channel,
                control_cost=float(execution["control_cost"]),
                parameters=intervention.parameters(),
            )
            preserved, verdicts = behavior_gate.evaluate(
                reference,
                record,
                control_cost=metadata.control_cost,
            )
            for layer, state in states.items():
                metrics: dict[str, float] = {}
                if record.task_score is not None:
                    metrics["task_score"] = float(record.task_score)
                if record.quality_score is not None:
                    metrics["quality_score"] = float(record.quality_score)
                if "monitor_label" in example:
                    metrics["monitor_label"] = float(example["monitor_label"])
                if layer in target_directions:
                    direction = np.asarray(
                        target_directions[layer], dtype=np.float64
                    ).reshape(-1)
                    direction /= max(np.linalg.norm(direction), 1e-12)
                    metrics["target_projection"] = float(np.dot(state, direction))
                metrics.update(
                    {
                        key: float(value)
                        for key, value in execution.items()
                        if key != "control_cost"
                    }
                )
                samples.append(
                    StateSample(
                        example_id=example_id,
                        model_name=model_name,
                        layer=layer,
                        intervention=metadata,
                        state=state,
                        token_states=token_states[layer]
                        if store_token_states
                        else None,
                        prompt=prepared_prompt,
                        output=output,
                        behavior_preserved=preserved,
                        constraint_results={
                            name: result.passed for name, result in verdicts.items()
                        },
                        metrics=metrics,
                        tags={
                            key: str(example[key])
                            for key in (
                                "source",
                                "category",
                                "split",
                                "concept",
                                "pair_id",
                            )
                            if example.get(key) is not None
                        },
                        seed=seed,
                    )
                )
    return samples
