"""Lightweight reporting utilities for article-oriented outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any


def to_serializable(obj: Any) -> Any:
    """Convert dataclasses and paths into JSON-friendly values."""
    if is_dataclass(obj):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def write_json_report(path: str | Path, payload: Any) -> None:
    """Write a JSON report with stable formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
