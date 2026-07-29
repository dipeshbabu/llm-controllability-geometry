"""Small compatibility helpers for Hugging Face causal LM architectures."""


def _candidate_text_models(model):
    """Yield likely decoder-only submodules from outermost to innermost."""

    queue = [model]
    seen = set()
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for name in ("model", "language_model", "text_model", "transformer"):
            child = getattr(candidate, name, None)
            if child is not None:
                queue.append(child)


def get_layers(model):
    """Return the transformer block list for common decoder-only models."""
    for candidate in _candidate_text_models(model):
        if hasattr(candidate, "gpt_neox") and hasattr(candidate.gpt_neox, "layers"):
            return candidate.gpt_neox.layers
        if hasattr(candidate, "layers"):
            return candidate.layers
        if hasattr(candidate, "h"):
            return candidate.h
    raise AttributeError("Could not locate transformer layers on model")


def get_mlp_output_projection(block):
    """Return the MLP output projection whose input is the post-activation feature."""
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise AttributeError("Transformer block has no mlp module")

    for name in ("dense_4h_to_h", "fc2", "down_proj", "c_proj"):
        if hasattr(mlp, name):
            return getattr(mlp, name)
    raise AttributeError("Could not locate MLP output projection on block")


def get_attention_module(block):
    """Return the attention module for common decoder-only model blocks."""
    for name in ("attention", "self_attn", "attn"):
        if hasattr(block, name):
            return getattr(block, name)
    raise AttributeError("Could not locate attention module on block")


def get_attention_output_projection(block):
    """Return the projection that combines attention heads."""

    attention = get_attention_module(block)
    for name in ("o_proj", "dense", "out_proj", "c_proj"):
        if hasattr(attention, name):
            return getattr(attention, name)
    raise AttributeError("Could not locate attention output projection on block")


def get_attention_head_layout(block) -> tuple[int, int]:
    """Return query-head count and head width for common decoder blocks."""

    attention = get_attention_module(block)
    head_count = None
    for name in ("num_heads", "num_attention_heads", "n_head"):
        value = getattr(attention, name, None)
        if value is not None:
            head_count = int(value)
            break
    config = getattr(attention, "config", None)
    if head_count is None and config is not None:
        value = getattr(config, "num_attention_heads", None)
        if value is not None:
            head_count = int(value)
    projection = get_attention_output_projection(block)
    input_width = getattr(projection, "in_features", None)
    if input_width is None:
        weight = getattr(projection, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) == 2:
            input_width = int(weight.shape[1])
    head_width = getattr(attention, "head_dim", None)
    if head_count is None or input_width is None:
        raise AttributeError("Could not infer attention head layout")
    if head_width is None:
        if int(input_width) % head_count:
            raise ValueError("attention projection width is not divisible by head count")
        head_width = int(input_width) // head_count
    if int(head_width) * head_count != int(input_width):
        raise ValueError("attention head layout does not match output projection input")
    return head_count, int(head_width)
