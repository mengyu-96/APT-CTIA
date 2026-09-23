"""Decision policy for accepting or escalating model attribution output."""

from __future__ import annotations


def assess_attribution(
    *,
    confidence: float,
    runner_up_confidence: float,
    evidence_count: int,
    evidence_reliable: bool,
    min_confidence: float = 0.6,
    min_margin: float = 0.1,
    min_evidence_count: int = 1,
) -> dict[str, object]:
    reasons: list[str] = []
    if confidence < min_confidence:
        reasons.append("置信度低于阈值")
    if confidence - runner_up_confidence < min_margin:
        reasons.append("候选组织得分过于接近")
    if evidence_count < min_evidence_count:
        reasons.append("有效证据数量不足")
    if not evidence_reliable:
        reasons.append("解释证据区分度不足")
    return {
        "status": "review_required" if reasons else "accepted",
        "reasons": reasons,
    }
