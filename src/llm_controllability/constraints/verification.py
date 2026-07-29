"""Deterministic task verifiers and optional semantic embeddings."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import transformers


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\\(", "").replace("\\)", "").replace("$", "")
    text = " ".join(text.split())
    return text.strip(" \t\r\n.,;:!?\"'")


def _extract_final_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1]
    marked = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*([^\r\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if marked:
        return marked[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def _multiple_choice_answer(output: str) -> str | None:
    explicit = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*\(?([A-Z])\)?(?=\s|[.,;:!?]|$)",
        output,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()
    boxed = re.findall(r"\\BOXED\{\s*([A-Z])\s*\}", output.upper())
    if boxed:
        return boxed[-1]
    leading = re.match(r"\s*\(?([A-Z])\)?(?:[.):]|\s)", output.upper())
    if leading:
        return leading.group(1)
    standalone = re.findall(r"\b([A-Z])\b", output.upper())
    return standalone[-1] if standalone else None


def verify_output(output: str, example: Mapping[str, Any]) -> tuple[float | None, bool | None]:
    """Score common benchmark answer formats without relying on a model judge."""

    if "answer" not in example:
        return None, None
    expected = str(example["answer"])
    verifier = str(example.get("verifier", "contains"))
    normalized_output = normalize_answer(output)
    normalized_expected = normalize_answer(expected)

    if verifier == "exact":
        correct = normalized_output == normalized_expected
    elif verifier == "contains":
        correct = normalized_expected in normalized_output
    elif verifier == "multiple_choice":
        selected = _multiple_choice_answer(output)
        correct = selected is not None and selected == expected.strip().upper()
    elif verifier == "numeric":
        expected_numbers = re.findall(
            r"[-+]?(?:\d*\.)?\d+",
            _extract_final_answer(expected).replace(",", ""),
        )
        output_numbers = re.findall(
            r"[-+]?(?:\d*\.)?\d+",
            _extract_final_answer(output).replace(",", ""),
        )
        tolerance = float(example.get("numeric_tolerance", 1e-6))
        correct = bool(expected_numbers and output_numbers) and abs(
            float(output_numbers[-1]) - float(expected_numbers[-1])
        ) <= tolerance
    elif verifier == "final_answer":
        extracted_output = normalize_answer(_extract_final_answer(output))
        extracted_expected = normalize_answer(_extract_final_answer(expected))
        correct = extracted_output == extracted_expected
    else:
        raise ValueError(f"unknown task verifier: {verifier}")
    return float(correct), correct


@dataclass
class TransformerSentenceEmbedder:
    """Mean-pooled transformer embeddings for a hard semantic gate."""

    model_name: str
    device: str = "cpu"
    max_length: int = 512
    batch_size: int = 16

    def __post_init__(self) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            outputs.append(pooled.cpu().numpy())
        return np.concatenate(outputs, axis=0)
