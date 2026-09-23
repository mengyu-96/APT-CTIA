"""Small formatting helpers shared by user-facing pages."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


def format_percentage(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "—"


def safe_artifact_name(path: str | None) -> str:
    if not path:
        return ""
    text = str(path).strip()
    windows_name = PureWindowsPath(text).name
    posix_name = PurePosixPath(text.replace("\\", "/")).name
    return posix_name or windows_name


def compact_identifier(value: Any, head: int = 11, tail: int = 5) -> str:
    text = str(value or "")
    if len(text) <= head + tail + 1:
        return text
    return f"{text[:head]}…{text[-tail:]}"


def report_option_label(item: dict[str, Any]) -> str:
    created = item.get("created") or "时间未知"
    samples = int(item.get("total_samples", 0) or 0)
    return f"{created} · {samples} 个样本 · {compact_identifier(item.get('id'))}"
