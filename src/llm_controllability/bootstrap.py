"""Initialize cross-platform runtime settings before importing PyTorch."""

from __future__ import annotations

import os


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from llm_controllability.cli import main as cli_main

    cli_main()
