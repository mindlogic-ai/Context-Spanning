"""The on-disk locations the runtime reads, resolved in one place.

| variable           | holds                                                  | default            |
| ------------------ | ------------------------------------------------------ | ------------------ |
| `DUETASPAN_ROOT`   | the `contextspan` package directory                    | auto-detected      |
| `DUETASPAN_DATA`   | the tool bank and its world / geo databases            | `$ROOT/datasets`   |
| `DUETASPAN_MODELS` | weights obtained separately (the Qwen3-ASR checkpoint) | `$ROOT/models`     |
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["ROOT", "DATA", "MODELS"]


def _env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


#: The `contextspan` package directory: this file sits two levels below it.
ROOT: Path = _env("DUETASPAN_ROOT", Path(__file__).resolve().parents[2])

#: The tool bank (`moshicp/mcp_tool_bank.json`) and the databases its tools answer from.
DATA: Path = _env("DUETASPAN_DATA", ROOT / "datasets")

#: Model weights obtained separately.
MODELS: Path = _env("DUETASPAN_MODELS", ROOT / "models")
