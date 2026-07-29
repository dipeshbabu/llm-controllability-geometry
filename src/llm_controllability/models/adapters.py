"""Model-family compatibility for text-only controllability experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

PromptFormat = Literal["plain", "chat"]
LoaderKind = Literal["causal", "multimodal"]

_PROFILE_ATTRIBUTE = "_llm_controllability_profile"
_CONTROL_SLOT = "<|llm_controllability_control_slot|>"


@dataclass(frozen=True)
class ModelProfile:
    """Resolved loading and prompt semantics for one checkpoint."""

    family: str
    loader: LoaderKind = "causal"
    prompt_format: PromptFormat = "plain"
    enable_thinking: bool | None = None
    trust_remote_code: bool = False


def resolve_model_profile(
    model_name: str,
    *,
    prompt_format: str = "auto",
    enable_thinking: bool | None = None,
    trust_remote_code: bool | None = None,
) -> ModelProfile:
    """Infer the supported checkpoint family without querying model metadata."""

    normalized = model_name.lower()
    if normalized.startswith("google/gemma-3-"):
        profile = ModelProfile(
            family="gemma3",
            loader="multimodal",
            prompt_format="chat" if normalized.endswith("-it") else "plain",
        )
    elif normalized.startswith("google/gemma-4-"):
        profile = ModelProfile(
            family="gemma4",
            loader="multimodal",
            prompt_format="chat" if normalized.endswith("-it") else "plain",
            enable_thinking=False,
        )
    elif normalized == "microsoft/phi-4-mini-instruct":
        profile = ModelProfile(
            family="phi4",
            prompt_format="chat",
            trust_remote_code=True,
        )
    elif normalized in {
        "microsoft/phi-4",
        "microsoft/phi-4-reasoning",
        "microsoft/phi-4-reasoning-plus",
    }:
        profile = ModelProfile(family="phi4", prompt_format="chat")
    elif normalized == "qwen/qwen3-8b":
        profile = ModelProfile(
            family="qwen3",
            prompt_format="chat",
            enable_thinking=False,
        )
    elif normalized == "qwen/qwen3-8b-base":
        profile = ModelProfile(family="qwen3")
    elif normalized.endswith(("-instruct", "-it")):
        profile = ModelProfile(family="generic", prompt_format="chat")
    else:
        profile = ModelProfile(family="generic")

    if prompt_format != "auto":
        if prompt_format not in {"plain", "chat"}:
            raise ValueError("prompt_format must be 'auto', 'plain', or 'chat'")
        profile = replace(profile, prompt_format=prompt_format)
    if enable_thinking is not None:
        profile = replace(profile, enable_thinking=enable_thinking)
    if trust_remote_code is not None:
        profile = replace(profile, trust_remote_code=trust_remote_code)
    return profile


def attach_model_profile(model: Any, tokenizer: Any, profile: ModelProfile) -> None:
    """Attach immutable experiment semantics to the loaded objects."""

    setattr(model, _PROFILE_ATTRIBUTE, profile)
    setattr(tokenizer, _PROFILE_ATTRIBUTE, profile)


def model_profile(value: Any) -> ModelProfile:
    """Return an attached profile or conservative plain-text defaults."""

    return getattr(value, _PROFILE_ATTRIBUTE, ModelProfile(family="generic"))


def ensure_padding_token(tokenizer):
    """Ensure batching with padding works for decoder-only tokenizers."""

    added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added = True
    return tokenizer, added


def allowed_candidate_token_ids(
    tokenizer,
    *,
    embedding_size: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return token ids that cannot alter the surrounding prompt protocol."""

    try:
        tokenizer_size = len(tokenizer)
    except (TypeError, AttributeError):
        tokenizer_size = int(tokenizer.vocab_size)
    upper = (
        min(tokenizer_size, int(embedding_size))
        if embedding_size is not None
        else tokenizer_size
    )
    blocked = {
        int(token_id)
        for token_id in getattr(tokenizer, "all_special_ids", [])
        if 0 <= int(token_id) < upper
    }
    allowed = [token_id for token_id in range(upper) if token_id not in blocked]
    if not allowed:
        raise ValueError("tokenizer has no ordinary tokens available for optimization")
    return torch.as_tensor(allowed, dtype=torch.long, device=device)


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the device used by the text embedding table."""

    embedding = model.get_input_embeddings()
    if embedding is not None and hasattr(embedding, "weight"):
        return embedding.weight.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Return the dtype used by the text embedding table."""

    embedding = model.get_input_embeddings()
    if embedding is not None and hasattr(embedding, "weight"):
        return embedding.weight.dtype
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def last_nonpadding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Locate the final real token for either left or right padded batches."""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
    ).expand_as(attention_mask)
    indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    if bool((indices < 0).any()):
        raise ValueError("every sequence must contain at least one nonpadding token")
    return indices


def _apply_chat_template(
    tokenizer,
    content: str,
    *,
    add_generation_prompt: bool,
) -> str:
    profile = model_profile(tokenizer)
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if profile.enable_thinking is not None:
        kwargs["enable_thinking"] = profile.enable_thinking
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_prompt(
    tokenizer,
    text: str,
    *,
    add_generation_prompt: bool = True,
) -> tuple[str, bool]:
    """Render one logical prompt and report whether a chat template was used."""

    if model_profile(tokenizer).prompt_format == "plain":
        return text, False
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("chat-formatted checkpoint tokenizer has no chat template")
    return (
        _apply_chat_template(
            tokenizer,
            text,
            add_generation_prompt=add_generation_prompt,
        ),
        True,
    )


def tokenize_prompts(
    tokenizer,
    texts: str | Sequence[str],
    *,
    add_generation_prompt: bool = True,
    **kwargs: Any,
):
    """Tokenize logical prompts with one consistent family-specific template."""

    scalar = isinstance(texts, str)
    values = [texts] if scalar else list(texts)
    rendered = [
        render_prompt(
            tokenizer,
            str(text),
            add_generation_prompt=add_generation_prompt,
        )
        for text in values
    ]
    use_chat = any(item[1] for item in rendered)
    if use_chat and not all(item[1] for item in rendered):
        raise ValueError("a tokenization batch cannot mix plain and chat prompts")
    payload: str | list[str] = [item[0] for item in rendered]
    if scalar:
        payload = payload[0]
    kwargs.setdefault("add_special_tokens", not use_chat)
    return tokenizer(payload, **kwargs)


def encode_prompt(
    tokenizer,
    text: str,
    *,
    add_generation_prompt: bool = True,
    **kwargs: Any,
) -> list[int]:
    """Encode one logical prompt without duplicating chat special tokens."""

    rendered, used_chat = render_prompt(
        tokenizer,
        text,
        add_generation_prompt=add_generation_prompt,
    )
    kwargs.setdefault("add_special_tokens", not used_chat)
    return list(tokenizer.encode(rendered, **kwargs))


def tokenize_prompt_with_continuation(
    tokenizer,
    prompt: str,
    continuation: str,
    *,
    max_length: int | None = None,
):
    """Teacher-force a continuation after the model's generation prefix."""

    prompt_ids = encode_prompt(
        tokenizer,
        prompt,
        add_generation_prompt=True,
    )
    continuation_ids = list(
        tokenizer.encode(continuation, add_special_tokens=False)
    )
    input_ids = prompt_ids + continuation_ids
    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
    values = torch.as_tensor([input_ids], dtype=torch.long)
    return {
        "input_ids": values,
        "attention_mask": torch.ones_like(values),
    }


def prompt_control_parts(
    tokenizer,
    context: str,
    *,
    max_length: int,
    separator: str = "\n",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tokenized text before and after a discrete user-control slot."""

    profile = model_profile(tokenizer)
    if profile.prompt_format == "plain":
        prefix = encode_prompt(
            tokenizer,
            f"{context}{separator}",
            add_generation_prompt=False,
            truncation=True,
            max_length=max_length,
        )
        return torch.as_tensor(prefix, dtype=torch.long), torch.empty(0, dtype=torch.long)

    rendered = _apply_chat_template(
        tokenizer,
        f"{context}{separator}{_CONTROL_SLOT}",
        add_generation_prompt=True,
    )
    if rendered.count(_CONTROL_SLOT) != 1:
        raise ValueError("chat template did not preserve the prompt-control slot")
    prefix_text, suffix_text = rendered.split(_CONTROL_SLOT)
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
    if len(prefix) + len(suffix) > max_length:
        keep = max(1, max_length - len(suffix))
        prefix = prefix[-keep:]
    return torch.as_tensor(prefix, dtype=torch.long), torch.as_tensor(
        suffix,
        dtype=torch.long,
    )


def profile_manifest(profile: ModelProfile) -> dict[str, Any]:
    return {
        "family": profile.family,
        "loader": profile.loader,
        "prompt_format": profile.prompt_format,
        "enable_thinking": profile.enable_thinking,
        "trust_remote_code": profile.trust_remote_code,
    }
