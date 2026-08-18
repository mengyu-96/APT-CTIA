from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from apt_ui.services import ui
from apt_ui.services.api_client import get_binary, get_json, get_runtime_config, request
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


def render_report() -> None:
    ui.page_header("报告生成与导出", "Report Generation & Export", icon="fa-file-alt")

    config_col, preview_col = st.columns([1, 2], gap="large")

    with config_col:
        with ui.section_card("报告配置", icon="fa-gear"):
            _render_report_config()

    with preview_col:
        with ui.section_card("报告预览与下载", icon="fa-file-pdf"):
            _render_report_preview()


def _render_report_config() -> None:
    runtime_config = get_runtime_config()
    training_source_enabled = runtime_config.get("training_report_source_enabled", True)

    source_options = ["归因任务 (Inference)"]
    if training_source_enabled:
        source_options.append("训练任务 (Training)")
    source_type = st.radio("数据来源", source_options, horizontal=True)

    selected_task: dict | None = None
    task_id: str | None = None

    if source_type == "归因任务 (Inference)":
        attr_results = get_json("/api/attribution_results", timeout=5, default=[])
        if attr_results:
            options = {
                f"{item['created']} - {item['id']} ({item['total_samples']} samples)": item
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
                f"{task['id'][:8]} - {MODEL_DISPLAY_NAMES.get(task.get('model'), task.get('model', 'Unknown'))} ({task['dataset']})": task
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

    st.selectbox("报告类型", ["完整归因报告", "技术细节报告", "高管摘要"], index=0)
    st.caption("导出格式：PDF")
    st.multiselect(
        "包含章节",
        ["执行摘要", "样本分析", "特征提取结果", "模型训练图谱", "归因结论", "IOCs"],
        default=["执行摘要", "归因结论", "IOCs"],
    )

    st.divider()
    if st.button("生成报告", type="primary", width="stretch"):
        if not task_id:
            st.error("请选择或输入任务/结果 ID。")
            return

        with st.spinner("正在生成报告..."):
            analysis_results = _build_analysis_results(source_type, task_id, selected_task)
            try:
                response = request(
                    "POST",
                    "/api/generate_report",
                    json_body={"task_id": task_id, "analysis_results": analysis_results},
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
    if source_type == "归因任务 (Inference)" and selected_task:
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

    if source_type == "训练任务 (Training)" and selected_task:
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
        ui.empty_state("请先在左侧配置并点击“生成报告”。", icon="fa-file-pdf")
        return

    report_path = str(report_info.get("report_path", ""))
    report_name = report_info.get("report_name") or Path(report_path).name or "report.pdf"
    download_path = report_info.get("download_path") or report_info.get("report_url")

    st.success("PDF 报告已就绪。")
    if report_path:
        st.markdown(f"**文件路径:** `{report_path}`")

    pdf_bytes = None
    if report_path:
        try:
            local_path = Path(report_path)
            if local_path.exists() and local_path.is_file():
                pdf_bytes = local_path.read_bytes()
        except Exception:
            pdf_bytes = None
    if pdf_bytes is None and download_path:
        pdf_bytes = get_binary(download_path, timeout=30)

    if pdf_bytes:
        st.download_button(
            "下载 PDF 报告",
            data=pdf_bytes,
            file_name=report_name,
            mime="application/pdf",
            width="stretch",
        )
    elif download_path:
        st.link_button("打开 PDF 报告", f"{BACKEND_URL}{download_path}", width="stretch")
        st.caption("当前环境无法直接读取文件字节，已回退为后端下载链接。")
    else:
        st.warning("报告已生成，但未获取到可下载文件。")
