"""Validation for user-selected report generation options."""

from __future__ import annotations

from typing import Any


REPORT_TYPES = {"full", "technical", "executive"}
REPORT_SECTIONS = {
    "executive_summary",
    "sample_analysis",
    "feature_analysis",
    "evidence_graph",
    "attribution_conclusion",
    "iocs",
}

DEFAULT_SECTIONS = {
    "full": list(REPORT_SECTIONS),
    "technical": [
        "feature_analysis",
        "evidence_graph",
        "sample_analysis",
        "attribution_conclusion",
        "iocs",
    ],
    "executive": ["executive_summary", "attribution_conclusion"],
}


def normalize_report_options(raw: Any) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    report_type = str(payload.get("report_type", "full"))
    if report_type not in REPORT_TYPES:
        report_type = "full"

    requested = payload.get("sections")
    if isinstance(requested, list):
        sections = [str(item) for item in requested if str(item) in REPORT_SECTIONS]
    else:
        sections = list(DEFAULT_SECTIONS[report_type])
    if not sections:
        sections = list(DEFAULT_SECTIONS[report_type])

    return {
        "report_type": report_type,
        "sections": sections,
        "redact_identifiers": bool(payload.get("redact_identifiers", False)),
        "language": "zh-CN",
    }
