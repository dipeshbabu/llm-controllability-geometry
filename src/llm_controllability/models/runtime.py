"""Resolve model runtime settings across CUDA, Apple MPS, and CPU hosts."""

from __future__ import annotations

import os
import platform
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_DTYPE_NAMES = {value: key for key, value in _DTYPES.items()}
_RUNTIME_ATTRIBUTE = "_llm_controllability_runtime"


def mps_is_built() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built())


def mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve one compute device and reject unavailable explicit backends."""

    requested = str(device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if resolved.type == "mps" and not mps_is_available():
        detail = (
            "this PyTorch build includes MPS, but no compatible device is available"
            if mps_is_built()
            else "this PyTorch build does not include MPS"
        )
        raise RuntimeError(f"MPS was requested, but {detail}")
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")
    return resolved


def resolve_device_map(device_map: str = "auto") -> tuple[str, torch.device]:
    """Resolve a Hugging Face device map and its primary compute backend."""

    requested = str(device_map).lower()
    device = resolve_device(requested)
    if requested == "auto" and device.type == "cuda":
        return "auto", device
    return str(device), device


def dtype_name(dtype: str | torch.dtype) -> str:
    if isinstance(dtype, str):
        normalized = dtype.lower()
        if normalized not in _DTYPES:
            raise ValueError(f"unsupported torch dtype: {dtype}")
        return normalized
    try:
        return _DTYPE_NAMES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported torch dtype: {dtype}") from error


def _macos_supports_bfloat16() -> bool:
    version = platform.mac_ver()[0]
    if not version:
        return False
    major = int(version.split(".", 1)[0])
    return major >= 14


def resolve_dtype(
    dtype: str | torch.dtype,
    device: str | torch.device,
) -> tuple[torch.dtype, str]:
    """Resolve a supported model dtype for the selected backend."""

    requested = dtype_name(dtype)
    resolved_device = torch.device(device)
    if (
        resolved_device.type == "mps"
        and requested == "bfloat16"
        and not _macos_supports_bfloat16()
    ):
        warnings.warn(
            "MPS bfloat16 requires macOS 14 or newer; using float16 instead",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.float16, "float16"
    if resolved_device.type == "cpu" and requested == "float16":
        warnings.warn(
            "CPU float16 is not reliable for this workload; using float32 instead",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.float32, "float32"
    return _DTYPES[requested], requested


def resolve_attention_implementation(
    implementation: str | None,
    device: str | torch.device,
) -> str | None:
    """Replace CUDA-only FlashAttention with a portable implementation."""

    if implementation != "flash_attention_2":
        return implementation
    if torch.device(device).type == "cuda":
        return implementation
    warnings.warn(
        "flash_attention_2 requires CUDA; using eager attention instead",
        RuntimeWarning,
        stacklevel=2,
    )
    return "eager"


@dataclass(frozen=True)
class RuntimeConfig:
    requested_device_map: str
    device_map: str
    device: torch.device
    requested_dtype: str
    dtype: torch.dtype
    dtype_name: str
    requested_attention_implementation: str | None
    attention_implementation: str | None

    def manifest(self) -> dict[str, Any]:
        return {
            "requested_device_map": self.requested_device_map,
            "device_map": self.device_map,
            "device": str(self.device),
            "requested_dtype": self.requested_dtype,
            "dtype": self.dtype_name,
            "requested_attention_implementation": (
                self.requested_attention_implementation
            ),
            "attention_implementation": self.attention_implementation,
        }


def attach_runtime_config(model: Any, runtime: RuntimeConfig) -> None:
    setattr(model, _RUNTIME_ATTRIBUTE, runtime)


def model_runtime_config(model: Any) -> RuntimeConfig | None:
    return getattr(model, _RUNTIME_ATTRIBUTE, None)


def resolve_runtime(
    *,
    device_map: str = "auto",
    dtype: str | torch.dtype = torch.float16,
    attention_implementation: str | None = None,
) -> RuntimeConfig:
    resolved_map, device = resolve_device_map(device_map)
    resolved_dtype, resolved_dtype_name = resolve_dtype(dtype, device)
    resolved_attention = resolve_attention_implementation(
        attention_implementation,
        device,
    )
    return RuntimeConfig(
        requested_device_map=str(device_map),
        device_map=resolved_map,
        device=device,
        requested_dtype=dtype_name(dtype),
        dtype=resolved_dtype,
        dtype_name=resolved_dtype_name,
        requested_attention_implementation=attention_implementation,
        attention_implementation=resolved_attention,
    )


def runtime_capabilities() -> dict[str, Any]:
    selected = resolve_device("auto")
    try:
        transformers_version = version("transformers")
    except PackageNotFoundError:
        transformers_version = None
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "selected_device": str(selected),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_built": mps_is_built(),
        "mps_available": mps_is_available(),
        "macos_version": platform.mac_ver()[0] or None,
        "mps_fallback_enabled": (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
        ),
    }
