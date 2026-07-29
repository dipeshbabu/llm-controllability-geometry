"""Context conditioned target runners for prompt control optimization."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from llm_controllability.models.adapters import prompt_control_parts


class ContextualTargetRunner:
    """Average a target runner over fixed prompt contexts.

    The optimized tokens remain the only mutable input. Each call prepends the
    same sampled contexts, evaluates the target on every resulting sequence,
    and averages the target and suffix logits across contexts.
    """

    def __init__(
        self,
        base_runner: Callable[..., dict[str, torch.Tensor]],
        model: torch.nn.Module,
        context_parts: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        if not context_parts:
            raise ValueError("context conditioned optimization requires at least one context")
        self.base_runner = base_runner
        self.model = model
        self.context_parts = tuple(
            (
                prefix.detach().long().reshape(-1),
                suffix.detach().long().reshape(-1),
            )
            for prefix, suffix in context_parts
        )
        if any(prefix.numel() == 0 for prefix, _ in self.context_parts):
            raise ValueError("prompt contexts must contain at least one token")
        self.minimize = bool(getattr(base_runner, "minimize", False))
        self.accepts_candidate_distribution = True

    @classmethod
    def from_texts(
        cls,
        base_runner: Callable[..., dict[str, torch.Tensor]],
        model: torch.nn.Module,
        tokenizer,
        texts: Sequence[str],
        *,
        context_count: int,
        max_length: int,
        seed: int,
    ) -> ContextualTargetRunner:
        if context_count <= 0:
            raise ValueError("context_count must be positive")
        candidates = [text.strip() for text in texts if text.strip()]
        if not candidates:
            raise ValueError("no nonempty optimization contexts were provided")
        rng = random.Random(seed)
        if len(candidates) > context_count:
            candidates = rng.sample(candidates, context_count)
        context_parts = [
            prompt_control_parts(
                tokenizer,
                text,
                max_length=max_length,
            )
            for text in candidates
        ]
        return cls(base_runner, model, context_parts)

    def __call__(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        candidate_distribution: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        for result in self.iter_context_results(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            candidate_distribution=candidate_distribution,
        ):
            for key, value in result.items():
                totals[key] = value if key not in totals else totals[key] + value

        scale = 1.0 / len(self.context_parts)
        return {key: value * scale for key, value in totals.items()}

    def iter_context_results(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        candidate_distribution: torch.Tensor | None = None,
    ):
        """Yield one context result so callers can release each graph promptly."""

        if input_ids is None and inputs_embeds is None:
            if candidate_distribution is None:
                raise ValueError(
                    "provide input_ids, inputs_embeds, or candidate_distribution"
                )
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError("provide only one of input_ids or inputs_embeds")

        candidate = (
            input_ids
            if input_ids is not None
            else inputs_embeds
            if inputs_embeds is not None
            else candidate_distribution
        )
        assert candidate is not None
        batch_size = candidate.shape[0]
        candidate_length = candidate.shape[1]
        device = candidate.device
        embedding = self.model.get_input_embeddings()

        for prefix_ids, suffix_ids in self.context_parts:
            prefix_ids = prefix_ids.to(device)
            suffix_ids = suffix_ids.to(device)
            prefix_length = prefix_ids.shape[0]
            if input_ids is not None:
                prefix = prefix_ids.view(1, -1).expand(batch_size, -1)
                suffix = suffix_ids.view(1, -1).expand(batch_size, -1)
                result = self.base_runner(
                    input_ids=torch.cat([prefix, input_ids, suffix], dim=1)
                )
            else:
                current_embeds = inputs_embeds
                if current_embeds is None:
                    assert candidate_distribution is not None
                    current_embeds = torch.matmul(
                        candidate_distribution,
                        embedding.weight,
                    )
                prefix = embedding(prefix_ids).detach()
                prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)
                suffix = embedding(suffix_ids).detach()
                suffix = suffix.unsqueeze(0).expand(batch_size, -1, -1)
                result = self.base_runner(
                    inputs_embeds=torch.cat(
                        [prefix, current_embeds, suffix],
                        dim=1,
                    )
                )

            processed: dict[str, torch.Tensor] = {}
            for key, value in result.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"contextual target output {key!r} is not a tensor")
                processed[key] = value
            if "logits" in processed:
                logits = processed["logits"][
                    :,
                    prefix_length : prefix_length + candidate_length,
                ]
                if logits.shape[1] != candidate_length:
                    raise ValueError(
                        "context and candidate exceed the model sequence limit"
                    )
                processed["logits"] = logits
                if input_ids is not None:
                    processed["xentropy"] = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                        input_ids[:, 1:].reshape(-1),
                        reduction="none",
                    ).view(batch_size, -1).mean(dim=1)
                elif candidate_distribution is not None:
                    processed["xentropy"] = (
                        -(
                            torch.log_softmax(logits[:, :-1], dim=-1)
                            * candidate_distribution[:, 1:]
                        )
                        .sum(dim=-1)
                        .mean(dim=-1)
                    )
            yield processed

    @property
    def context_count(self) -> int:
        return len(self.context_parts)
