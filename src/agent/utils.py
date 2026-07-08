"""Shared utility functions for the agentic pipeline."""
from __future__ import annotations

import json as _json
from typing import Any


def as_text(data: Any, cap: int = 2000) -> str:
    if isinstance(data, (dict, list)):
        return _json.dumps(data, ensure_ascii=False, default=str)[:cap]
    return str(data)[:cap]
