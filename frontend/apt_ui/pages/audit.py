from __future__ import annotations

import pandas as pd
import streamlit as st

from apt_ui.services import ui
from apt_ui.services.api_client import get_json, invalidate


EVENT_LABELS = {
    "review.saved": "归因复核",
}
DECISION_LABELS = {
    "confirmed": "确认",
    "rejected": "驳回",
    "corrected": "改判",
    "insufficient": "证据不足",
}


def render_audit_log() -> None:
    ui.page_header("审计日志", "追踪分析师复核和关键研判变更", icon="fa-clipboard-list")

    if ui.refresh_button("refresh_audit_logs"):
        invalidate("audit_logs")
        st.rerun()

    rows = get_json("/api/audit_logs", params={"limit": 300}, timeout=5, default=[])
    if not rows:
        ui.empty_state("暂无审计记录。保存分析师复核结论后会在此显示。", icon="fa-clipboard-list")
        return

    table_rows = []
    for item in rows:
        table_rows.append(
            {
                "时间": item.get("timestamp", ""),
                "事件": EVENT_LABELS.get(item.get("event"), item.get("event", "")),
                "操作人": item.get("actor") or "未填写",
                "结果 ID": item.get("result_id", ""),
                "样本 ID": item.get("sample_id", ""),
                "结论": DECISION_LABELS.get(item.get("decision"), item.get("decision", "")),
                "改判组织": item.get("corrected_label", ""),
                "说明": item.get("note", ""),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True, height=520)
