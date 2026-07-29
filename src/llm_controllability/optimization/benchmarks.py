"""Experiment runners and baselines for suppression studies."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from llm_controllability.models.adapters import (
    allowed_candidate_token_ids,
    model_device,
)
from llm_controllability.optimization.epo import History, epo, evaluate_fitness, gcg
from llm_controllability.reporting.results import CandidateRecord


def _decode(tokenizer, ids) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def score_input_ids(
    cache_run: Callable,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    target_name: str,
    method: str,
    seed: int,
    batch_size: int = 64,
    source: str = "",
) -> list[CandidateRecord]:
    with torch.no_grad():
        state = evaluate_fitness(
            model,
            cache_run,
            input_ids.to(model_device(model)),
            batch_size,
        )
    records = []
    ids_cpu = input_ids.detach().cpu()
    for i in range(ids_cpu.shape[0]):
        records.append(
            CandidateRecord(
                target_name=target_name,
                method=method,
                seed=seed,
                text=_decode(tokenizer, ids_cpu[i]),
                target=float(state.target[i].detach().cpu()),
                xentropy=float(state.xentropy[i].detach().cpu()),
                source=source,
            )
        )
    return records


def score_texts(
    cache_run: Callable,
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    target_name: str,
    method: str,
    seed: int,
    batch_size: int = 64,
    max_length: int = 128,
    source: str = "",
) -> list[CandidateRecord]:
    encoded_items = []
    for idx, text in enumerate(texts):
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        if len(ids) >= 2:
            encoded_items.append((idx, text, ids))

    by_len: dict[int, list[tuple[int, str, list[int]]]] = {}
    for item in encoded_items:
        by_len.setdefault(len(item[2]), []).append(item)

    scored_by_idx = {}
    for length_group in by_len.values():
        for start in range(0, len(length_group), batch_size):
            chunk = length_group[start : start + batch_size]
            input_ids = torch.tensor([ids for _, _, ids in chunk], dtype=torch.long)
            batch_records = score_input_ids(
                cache_run,
                model,
                tokenizer,
                input_ids,
                target_name=target_name,
                method=method,
                seed=seed,
                batch_size=batch_size,
                source=source,
            )
            for record, (idx, original, _) in zip(batch_records, chunk):
                scored_by_idx[idx] = CandidateRecord(
                    target_name=record.target_name,
                    method=record.method,
                    seed=record.seed,
                    text=original,
                    target=record.target,
                    xentropy=record.xentropy,
                    source=record.source,
                    extra={"encoded_text": record.text},
                )

    records = []
    for idx in sorted(scored_by_idx):
        records.append(scored_by_idx[idx])
    return records


def random_token_baseline(
    cache_run: Callable,
    model,
    tokenizer,
    *,
    target_name: str,
    seed: int,
    n_prompts: int = 256,
    seq_len: int = 24,
    batch_size: int = 64,
) -> list[CandidateRecord]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    allowed_ids = allowed_candidate_token_ids(
        tokenizer,
        embedding_size=model.get_input_embeddings().num_embeddings,
        device="cpu",
    )
    sampled = torch.randint(
        low=0,
        high=allowed_ids.numel(),
        size=(n_prompts, seq_len),
        generator=generator,
        device="cpu",
    )
    input_ids = allowed_ids[sampled].to(model_device(model))
    return score_input_ids(
        cache_run,
        model,
        tokenizer,
        input_ids,
        target_name=target_name,
        method="random",
        seed=seed,
        batch_size=batch_size,
        source="uniform_tokens",
    )


def random_search_baseline(
    cache_run: Callable,
    model,
    tokenizer,
    *,
    target_name: str,
    seed: int,
    population_size: int = 16,
    iters: int = 100,
    explore_per_pop: int = 16,
    seq_len: int = 24,
    batch_size: int = 64,
) -> list[CandidateRecord]:
    n_prompts = population_size * max(1, iters) * max(1, explore_per_pop)
    records = random_token_baseline(
        cache_run,
        model,
        tokenizer,
        target_name=target_name,
        seed=seed,
        n_prompts=n_prompts,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    return [
        CandidateRecord(
            target_name=r.target_name,
            method="random_search",
            seed=r.seed,
            text=r.text,
            target=r.target,
            xentropy=r.xentropy,
            source="uniform_tokens_equal_budget",
            extra={
                "population_size": population_size,
                "iters": iters,
                "explore_per_pop": explore_per_pop,
            },
        )
        for r in records
    ]


def minscan_baseline(
    cache_run: Callable,
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    target_name: str,
    seed: int,
    batch_size: int = 64,
    max_length: int = 128,
    fluency_quantile: float = 0.2,
    minimize: bool = True,
) -> list[CandidateRecord]:
    scored = score_texts(
        cache_run,
        model,
        tokenizer,
        texts,
        target_name=target_name,
        method="minscan_pool",
        seed=seed,
        batch_size=batch_size,
        max_length=max_length,
        source="text_pool",
    )
    if not scored:
        return []
    threshold = sorted(r.xentropy for r in scored)[
        max(0, min(len(scored) - 1, math.floor((len(scored) - 1) * fluency_quantile)))
    ]
    fluent = [r for r in scored if r.xentropy <= threshold]
    best = min(fluent, key=lambda r: r.target) if minimize else max(fluent, key=lambda r: r.target)
    return [
        CandidateRecord(
            target_name=target_name,
            method="minscan",
            seed=seed,
            text=best.text,
            target=best.target,
            xentropy=best.xentropy,
            source=best.source,
            extra={"pool_size": len(scored), "fluency_quantile": fluency_quantile},
        )
    ]


def history_to_records(
    history: History,
    tokenizer,
    *,
    target_name: str,
    method: str,
    seed: int,
) -> list[CandidateRecord]:
    ids = history.ids.reshape((-1, history.ids.shape[-1]))
    target = history.target.reshape(-1)
    xentropy = history.xentropy.reshape(-1)
    records = []
    for i in range(ids.shape[0]):
        records.append(
            CandidateRecord(
                target_name=target_name,
                method=method,
                seed=seed,
                text=_decode(tokenizer, ids[i]),
                target=float(target[i]),
                xentropy=float(xentropy[i]),
                source="search_history",
            )
        )
    return records


def epo_suppression_run(
    cache_run: Callable,
    model,
    tokenizer,
    *,
    target_name: str,
    seed: int,
    seq_len: int = 24,
    population_size: int = 16,
    iters: int = 100,
    explore_per_pop: int = 16,
    batch_size: int = 64,
    topk: int = 256,
    x_penalty_min: float = 0.1,
    x_penalty_max: float = 10.0,
    method: str = "epo",
    minimize: bool = True,
) -> list[CandidateRecord]:
    cache_run.minimize = minimize
    history = epo(
        cache_run,
        model,
        tokenizer,
        seq_len=seq_len,
        population_size=population_size,
        iters=iters,
        explore_per_pop=explore_per_pop,
        batch_size=batch_size,
        topk=topk,
        x_penalty_min=x_penalty_min,
        x_penalty_max=x_penalty_max,
        seed=seed,
        callback=False,
    )
    return history_to_records(history, tokenizer, target_name=target_name, method=method, seed=seed)


def gcg_suppression_run(
    cache_run: Callable,
    model,
    tokenizer,
    *,
    target_name: str,
    seed: int,
    seq_len: int = 24,
    iters: int = 100,
    explore_per_iter: int = 16,
    batch_size: int = 64,
    topk: int = 256,
    x_penalty: float = 1.0,
    minimize: bool = True,
) -> list[CandidateRecord]:
    cache_run.minimize = minimize
    history = gcg(
        cache_run,
        model,
        tokenizer,
        seq_len=seq_len,
        iters=iters,
        explore_per_iter=explore_per_iter,
        batch_size=batch_size,
        topk=topk,
        x_penalty_min=x_penalty,
        x_penalty_max=x_penalty,
        seed=seed,
        callback=False,
    )
    return history_to_records(history, tokenizer, target_name=target_name, method="gcg", seed=seed)
