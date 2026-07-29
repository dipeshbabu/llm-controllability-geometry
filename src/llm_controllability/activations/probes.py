import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from llm_controllability.models.adapters import (
    ensure_padding_token,
    last_nonpadding_indices,
    model_device,
    model_dtype,
    tokenize_prompts,
)
from llm_controllability.models.architecture import get_layers


def _to_numpy_float32(tensor):
    return tensor.detach().float().cpu().numpy()


def _pool_hidden(hidden, attention_mask, pooling):
    if pooling == "last":
        last_idx = last_nonpadding_indices(attention_mask)
        return hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            last_idx,
        ]
    if pooling == "mean":
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    raise ValueError("pooling must be 'last' or 'mean'")


@torch.no_grad()
def collect_residual_states_many(
    model,
    hook_layers,
    tok,
    texts,
    max_len=256,
    pooling="last",
    batch_size=16,
):
    """Collect pooled states from several layers in the same model passes."""

    layers = tuple(dict.fromkeys(int(layer) for layer in hook_layers))
    if not layers:
        raise ValueError("hook_layers must be nonempty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tok, _ = ensure_padding_token(tok)
    captured = {}
    outputs = {layer: [] for layer in layers}
    handles = []

    for layer in layers:
        def hook(module, inputs, output, layer=layer):
            captured[layer] = output[0] if isinstance(output, tuple) else output

        handles.append(get_layers(model)[layer].register_forward_hook(hook))

    try:
        for batch in [
            texts[i:i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]:
            ids = tokenize_prompts(
                tok,
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(model_device(model))
            model(**ids)
            for layer in layers:
                pooled = _pool_hidden(
                    captured[layer],
                    ids["attention_mask"],
                    pooling,
                )
                outputs[layer].append(_to_numpy_float32(pooled))
            captured.clear()
    finally:
        for handle in handles:
            handle.remove()
    return {
        layer: np.vstack(values)
        for layer, values in outputs.items()
    }


@torch.no_grad()
def collect_residual_states(
    model,
    hook_layer: int,
    tok,
    texts,
    max_len=256,
    pooling="last",
    batch_size=16,
):
    outs = []
    tok, _ = ensure_padding_token(tok)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def hk(mod, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hk.h = hidden

    handle = get_layers(model)[hook_layer].register_forward_hook(hk)
    for batch in [
        texts[i:i + batch_size]
        for i in range(0, len(texts), batch_size)
    ]:
        ids = tokenize_prompts(
            tok,
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(model_device(model))
        _ = model(**ids)
        H = _pool_hidden(hk.h, ids["attention_mask"], pooling)
        outs.append(_to_numpy_float32(H))  # [B, d]
    handle.remove()
    return np.vstack(outs)  # [N, d]


@torch.no_grad()
def collect_residual_means(
    model,
    hook_layer: int,
    tok,
    texts,
    max_len=256,
    batch_size=16,
):
    outs = []
    tok, _ = ensure_padding_token(tok)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def hk(mod, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hk.h = hidden

    handle = get_layers(model)[hook_layer].register_forward_hook(hk)
    for batch in [
        texts[i:i + batch_size]
        for i in range(0, len(texts), batch_size)
    ]:
        ids = tokenize_prompts(
            tok,
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(model_device(model))
        _ = model(**ids)
        mask = ids["attention_mask"].to(hk.h.dtype).unsqueeze(-1)
        H = (hk.h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        H = _to_numpy_float32(H)
        outs.append(H)
    handle.remove()
    return np.vstack(outs)  # [N, d]


def fit_direction(
    model,
    tok,
    layer,
    eval_texts,
    noneval_texts,
    pooling="last",
    batch_size=16,
):
    Xe = collect_residual_states(
        model,
        layer,
        tok,
        eval_texts,
        pooling=pooling,
        batch_size=batch_size,
    )
    Xn = collect_residual_states(
        model,
        layer,
        tok,
        noneval_texts,
        pooling=pooling,
        batch_size=batch_size,
    )
    X = np.vstack([Xe, Xn])
    y = np.hstack([np.ones(len(Xe)), np.zeros(len(Xn))])
    clf = LogisticRegression(max_iter=500, C=1.0).fit(X, y)
    w = torch.tensor(
        clf.coef_[0] / (np.linalg.norm(clf.coef_[0]) + 1e-8),
        dtype=model_dtype(model),
        device=model_device(model),
    )
    return w  # unit vector
