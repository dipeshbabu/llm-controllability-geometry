"""LaTeX table export for experiment summaries."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path


def _escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return _escape(value)
    return f"{parsed:.3g}"


def rows_from_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_to_latex_table(
    rows: Sequence[dict],
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    caption: str = "",
    label: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("% No rows to render.\n", encoding="utf-8")
        return
    columns = list(columns or rows[0].keys())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(columns) + r"}",
        r"\toprule",
        " & ".join(_escape(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_format_value(row.get(c, "")) for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if caption:
        lines.append(r"\caption{" + _escape(caption) + r"}")
    if label:
        lines.append(r"\label{" + _escape(label) + r"}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
