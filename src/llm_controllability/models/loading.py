"""Load supported Hugging Face checkpoints with recorded prompt semantics."""

from __future__ import annotations

import torch
import transformers

from llm_controllability.models.adapters import (
    attach_model_profile,
    ensure_padding_token,
    resolve_model_profile,
)


def load_tokenizer(
    model_name: str = "EleutherAI/pythia-70m-deduped",
    revision: str | None = None,
):
    """Load and configure a tokenizer for decoder-only batching."""

    profile = resolve_model_profile(model_name)
    tokenizer, _ = ensure_padding_token(
        transformers.AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=profile.trust_remote_code,
        )
    )
    tokenizer._llm_controllability_profile = profile
    return tokenizer


def load_model(
    model_size: str = "12b",
    model_name: str | None = None,
    tokenizer_name: str | None = None,
    requires_grad: bool = False,
    attn_implementation: str | None = "flash_attention_2",
    device_map: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    prompt_format: str = "auto",
    enable_thinking: bool | None = None,
    trust_remote_code: bool | None = None,
    revision: str | None = None,
):
    """Load a text-capable checkpoint for controllability experiments."""

    model_name = model_name or f"EleutherAI/pythia-{model_size}-deduped"
    profile = resolve_model_profile(
        model_name,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "use_cache": False,
        "device_map": device_map,
        "trust_remote_code": profile.trust_remote_code,
        "revision": revision,
    }
    transformers_major = int(transformers.__version__.split(".", 1)[0])
    model_kwargs["dtype" if transformers_major >= 5 else "torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model_class = (
        transformers.AutoModelForMultimodalLM
        if profile.loader == "multimodal"
        else transformers.AutoModelForCausalLM
    )
    model = model_class.from_pretrained(model_name, **model_kwargs)

    if not requires_grad:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    tokenizer, added_tokens = ensure_padding_token(
        transformers.AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            revision=revision,
            trust_remote_code=profile.trust_remote_code,
        )
    )
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    attach_model_profile(model, tokenizer, profile)
    return model, tokenizer
