from __future__ import annotations

import os

import streamlit as st

from apt_ui.services import ui
from apt_ui.services.api_client import get_binary, get_json, get_runtime_config, request
from apt_ui.services.presentation import compact_identifier, report_option_label, safe_artifact_name
from apt_ui.services.tasks import get_task_detail, list_tasks


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")
MODEL_DISPLAY_NAMES = {
    "RGAT": "GRACE",
    "GAT": "APT-ATT",
    "Hybrid": "APT-MMF",
    "GCN": "MLDSJ",
    "Transformer": "Mead",
    "GraphSAGE": "TRAIL",
}
REPORT_TYPES = {
    "完整归因报告": "full",
    "技术细节报告": "technical",
    "高管摘要": "executive",
}
REPORT_SECTIONS = {
    "执行摘要": "executive_summary",
    "样本分析": "sample_analysis",
    "特征提取结果": "feature_analysis",
    "证据图谱": "evidence_graph",
    "归因结论": "attribution_conclusion",
    "IOC 清单": "iocs",
}
OUTPUT_FORMATS = {
    "PDF 报告": "pdf",
    "JSON 数据": "json",
    "STIX 2.1 情报包": "stix",
}


def render_report() -> None:
    ui.page_header("报告生成与导出", "根据归因结果或训练记录生成 PDF、JSON 或 STIX 2.1", icon="fa-file-alt")

    config_col, preview_col = st.columns([1, 2], gap="large")

    with config_col:
        with ui.section_card("报告配置", icon="fa-gear"):
            _render_report_config()

    with preview_col:
        with ui.section_card("报告下载", icon="fa-file-pdf"):
            _render_report_preview()


def _render_report_config() -> None:
    runtime_config = get_runtime_config()
    training_source_enabled = runtime_config.get("training_report_source_enabled", True)

    source_options = ["归因结果"]
    if training_source_enabled:
        source_options.append("训练记录")
    source_type = st.radio("数据来源", source_options, horizontal=True)

    selected_task: dict | None = None
    task_id: str | None = None

    if source_type == "归因结果":
        attr_results = get_json("/api/attribution_results", timeout=5, default=[])
        if attr_results:
            options = {
                report_option_label(item): item
                for item in attr_results
            }
            selected_label = st.selectbox("选择归因结果", options=list(options.keys()))
            if selected_label:
                selected_task = options[selected_label]
                task_id = selected_task["id"]
        else:
            st.info("暂无归因结果。")
    else:
        completed_tasks = [
            task
            for task in list_tasks(task_type="train", limit=40, ttl="default", timeout=5)
            if task.get("status") == "completed"
        ]
        if completed_tasks:
            options = {
                f"{MODEL_DISPLAY_NAMES.get(task.get('model'), task.get('model', '未知模型'))} · "
                f"{task.get('dataset', '未知数据集')} · {compact_identifier(task.get('id'))}": task
                for task in completed_tasks
            }
            selected_label = st.selectbox("选择训练任务", options=list(options.keys()))
            if selected_label:
                selected_task = options[selected_label]
                task_id = selected_task["id"]
        else:
            st.info("暂无已完成的训练任务。")

    if not task_id:
        task_id = st.text_input("或手动输入任务/结果 ID", value="").strip()

    report_type_label = st.selectbox("报告类型", list(REPORT_TYPES), index=0)
    output_format_label = st.selectbox("导出格式", list(OUTPUT_FORMATS), index=0)
    selected_section_labels = st.multiselect(
        "包含章节",
        list(REPORT_SECTIONS),
        default=["执行摘要", "样本分析", "归因结论", "IOC 清单"],
    )
    redact_identifiers = st.checkbox("脱敏样本与任务标识", value=False)

    st.divider()
    if st.button("生成报告", type="primary", width="stretch"):
        if not task_id:
            st.error("请选择或输入任务/结果 ID。")
            return
        if not selected_section_labels:
            st.error("请至少选择一个报告章节。")
            return
        if source_type == "训练记录" and OUTPUT_FORMATS[output_format_label] == "stix":
            st.error("STIX 2.1 仅适用于归因结果，请选择 PDF 或 JSON。")
            return

        with st.spinner("正在生成报告..."):
            analysis_results = _build_analysis_results(source_type, task_id, selected_task)
            payload = {
                "task_id": task_id,
                "analysis_results": analysis_results,
                "output_format": OUTPUT_FORMATS[output_format_label],
                "report_options": {
                    "report_type": REPORT_TYPES[report_type_label],
                    "sections": [REPORT_SECTIONS[label] for label in selected_section_labels],
                    "redact_identifiers": redact_identifiers,
                },
            }
            try:
                response = request(
                    "POST",
                    "/api/generate_report",
                    json_body=payload,
                    timeout=30,
                )
            except Exception as exc:
                st.error(f"连接后端失败: {exc}")
                return

        if response.status_code != 200:
            st.error(f"生成失败: {response.text}")
            return

        result = response.json()
        st.session_state["report_generated"] = result
        st.success("报告生成成功。")


def _build_analysis_results(source_type: str, task_id: str, selected_task: dict | None) -> dict:
    if source_type == "归因结果" and selected_task:
        distribution = selected_task.get("label_distribution", {})
        sorted_dist = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        total = max(int(selected_task.get("total_samples", 0) or 0), 1)
        attributions = [
            {"name": name, "score": count / total, "risk": "High" if idx == 0 else "Medium"}
            for idx, (name, count) in enumerate(sorted_dist)
        ]
        return {
            "top_attribution": sorted_dist[0][0] if sorted_dist else "Unknown",
            "attributions": attributions,
            "total_samples": total,
            "source_type": "inference",
        }

    if source_type == "训练记录" and selected_task:
        task_detail = get_task_detail(task_id) if task_id else {}
        result = task_detail.get("result") or selected_task.get("result") or {}
        report = result.get("classification_report") or {}
        top_attr = "Unknown"
        best_f1 = -1.0
        attributions = []
        for cls_name, metrics in report.items():
            if cls_name in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if not isinstance(metrics, dict) or "f1-score" not in metrics:
                continue
            f1 = float(metrics.get("f1-score", 0.0) or 0.0)
            if f1 > best_f1:
                best_f1 = f1
                top_attr = cls_name
            attributions.append(
                {
                    "name": cls_name,
                    "score": float(metrics.get("precision", 0.0) or 0.0),
                    "risk": "High" if f1 > 0.8 else "Medium",
                }
            )
        return {
            "top_attribution": top_attr,
            "attributions": attributions,
            "confusion_matrix_plot": result.get("confusion_matrix_plot"),
            "history_plot": result.get("history_plot"),
            "source_type": "training",
        }

    return {}


def _render_report_preview() -> None:
    report_info = st.session_state.get("report_generated")
    if not report_info:
        ui.empty_state("请完成报告配置并点击“生成报告”。", icon="fa-file-pdf")
        return

    report_name = safe_artifact_name(report_info.get("report_name")) or "report.pdf"
    download_path = report_info.get("download_path") or report_info.get("report_url")
    report_format = str(report_info.get("report_format") or report_name.rsplit(".", 1)[-1]).lower()
    format_label = {"pdf": "PDF", "json": "JSON", "stix": "STIX 2.1"}.get(report_format, report_format.upper())
    mime_type = "application/pdf" if report_format == "pdf" else "application/json"

    st.success(f"{format_label} 文件已就绪。")
    st.markdown(f"**文件名：** `{report_name}`")

    report_bytes = get_binary(download_path, timeout=30) if download_path else None

    if report_bytes:
        st.download_button(
            f"下载 {format_label}",
            data=report_bytes,
            file_name=report_name,
            mime=mime_type,
            width="stretch",
        )
    elif download_path:
        st.link_button(f"打开 {format_label}", f"{BACKEND_URL}{download_path}", width="stretch")
        st.caption("当前环境无法直接读取文件字节，已回退为后端下载链接。")
    else:
        st.warning("报告已生成，但未获取到可下载文件。")
