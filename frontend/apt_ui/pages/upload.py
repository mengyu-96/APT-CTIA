from __future__ import annotations

import requests
import streamlit as st

from apt_ui.services.api_client import build_upload_payload, invalidate, request
from apt_ui.services import ui
from apt_ui.services.task_ui import render_task_panel


POLL_STATE_KEY = "upload_polling_active"


def _task_title(task: dict) -> str:
    name = task.get("name") or "预处理任务"
    if name == "preprocess task":
        file_count = task.get("files", 0)
        return f"批量预处理 ({file_count} 个文件)" if file_count else "批量预处理"
    return name


def render_upload() -> None:
    ui.page_header("分析任务管理", "Analysis Task Management", icon="fa-tasks")

    left, right = st.columns([1, 1], gap="large")

    with left:
        with ui.section_card("新建预处理任务", icon="fa-plus-circle"):
            st.caption("支持上传 `TXT`、`JSON`、`PDF` 文件。")
            with st.form("create_preprocess_task", clear_on_submit=True):
                dataset_name = st.text_input(
                    "输出数据集名称",
                    placeholder="留空则自动生成时间戳目录",
                )
                uploaded_files = st.file_uploader(
                    "选择文件",
                    accept_multiple_files=True,
                    type=["txt", "json", "pdf"],
                )
                submitted = st.form_submit_button("提交任务", type="primary", width="stretch")

            if submitted:
                _submit(dataset_name, uploaded_files)

    with right:
        with ui.section_card("任务记录", icon="fa-list-check"):
            render_task_panel(
                "preprocess",
                title_fn=_task_title,
                key_prefix="preprocess",
                poll_state_key=POLL_STATE_KEY,
                empty_message="暂无预处理任务。",
                active_message="正在执行预处理。",
            )


def _submit(dataset_name: str, uploaded_files) -> None:
    if not uploaded_files:
        st.error("请至少选择一个文件。")
        return
    try:
        payload = {"output_name": dataset_name} if dataset_name else {}
        with st.spinner("正在上传并提交任务..."):
            response = request(
                "POST",
                "/api/preprocess",
                files=build_upload_payload("files", uploaded_files),
                data=payload,
                timeout=(10, 120),
            )
        if response.status_code == 202:
            invalidate("tasks", "datasets")
            st.session_state[POLL_STATE_KEY] = True
            st.success(f"任务已提交：{response.json().get('task_id')}")
            st.rerun()
        else:
            st.error(f"提交失败: {response.text}")
    except requests.exceptions.ReadTimeout:
        invalidate("tasks")
        st.session_state[POLL_STATE_KEY] = True
        st.warning("请求超时，但任务可能已进入队列。")
    except requests.exceptions.ConnectionError:
        st.error("无法连接后端服务。")
    except Exception as exc:
        st.error(f"提交异常: {exc}")
