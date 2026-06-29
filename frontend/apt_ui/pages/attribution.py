from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from apt_ui.services.api_client import invalidate, get_json, request
from apt_ui.services.charting import PLOTLY_CHART_CONFIG, BRAND_SEQUENCE, apply_layout
from apt_ui.services.task_ui import render_task_panel
from apt_ui.services import ui


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")
POLL_STATE_KEY = "inference_polling_active"
VIEW_STATE_KEY = "attribution_page_view"

TASK_VIEW_LABEL = "新建归因任务"
HISTORY_VIEW_LABEL = "历史归因结果"


def _get_models() -> list[dict]:
    return get_json("/api/models", timeout=4, default=[])


def _get_datasets() -> list[dict]:
    return get_json("/api/datasets", timeout=3, default=[])


def _get_attribution_results(*, ttl: str = "default") -> list[dict]:
    return get_json("/api/attribution_results", timeout=4, default=[], ttl=ttl)


def _get_attribution_detail(result_id: str) -> dict | None:
    return get_json(f"/api/attribution_results/{result_id}", timeout=20, default=None, ttl="slow")


def _get_attribution_sample(result_id: str, report_id: str) -> dict | None:
    if not result_id or report_id is None:
        return None
    return get_json(
        f"/api/attribution_results/{result_id}/sample/{report_id}",
        timeout=20,
        default=None,
        ttl="slow",
    )


@st.cache_data(ttl=300, show_spinner=False)
def _distribution_df(distribution: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(list(distribution.items()), columns=["APT 组织", "数量"])


@st.cache_data(ttl=300, show_spinner=False)
def _result_rows(raw_results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in raw_results:
        top3 = ", ".join(
            f"{candidate.get('label', 'Unknown')} ({candidate.get('score', 0):.2%})"
            for candidate in item.get("top3", [])
        )
        rows.append(
            {
                "report_id": item.get("report_id"),
                "predicted_label": item.get("predicted_label"),
                "confidence": item.get("confidence"),
                "top3": top3,
            }
        )
    return pd.DataFrame(rows)


def _delete_result(result_id: str) -> bool:
    if not result_id:
        return False
    try:
        response = request("DELETE", f"/api/attribution_results/{result_id}", timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def _render_explanation_panel(sample_result: dict) -> None:
    explanation = sample_result.get("explanation") or {}
    if not explanation:
        st.info("当前样本没有可展示的解释证据。")
        return

    decision_mode = explanation.get("decision_mode") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("预测组织", sample_result.get("predicted_label", "Unknown"))
    c2.metric("置信度", f"{sample_result.get('confidence', 0):.2%}")
    c3.metric("主导信号", decision_mode.get("dominant_signal", "unknown"))

    if decision_mode:
        st.caption(
            f"语义权重 {decision_mode.get('semantic_gate', 0):.2%} | "
            f"结构权重 {decision_mode.get('structural_gate', 0):.2%} | "
            f"{decision_mode.get('description', '')}"
        )

    key_nodes = explanation.get("key_nodes") or []
    if key_nodes:
        st.markdown("**关键证据节点**")
        node_df = pd.DataFrame(key_nodes)
        st.dataframe(node_df, width="stretch", hide_index=True)
        if {"text", "attention"}.issubset(node_df.columns):
            fig_nodes = px.bar(
                node_df.head(8),
                x="text",
                y="attention",
                color="type" if "type" in node_df.columns else None,
            )
            apply_layout(fig_nodes)
            st.plotly_chart(fig_nodes, width="stretch", config=PLOTLY_CHART_CONFIG)

    key_edges = explanation.get("key_edges") or []
    if key_edges:
        st.markdown("**关键证据边**")
        st.dataframe(pd.DataFrame(key_edges), width="stretch", hide_index=True)

    evidence_paths = explanation.get("evidence_paths") or []
    if evidence_paths:
        st.markdown("**核心证据路径**")
        for item in evidence_paths[:5]:
            st.markdown(f"- `{' -> '.join(item.get('path_texts', []))}`")

    mitre_attack = explanation.get("mitre_attack") or {}
    techniques = mitre_attack.get("techniques") or []
    if techniques:
        st.markdown("**MITRE ATT&CK 摘要**")
        st.dataframe(pd.DataFrame(techniques), width="stretch", hide_index=True)


def _inference_task_title(task: dict) -> str:
    return task.get("name") or "归因任务"


def _render_task_view() -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with ui.section_card("新建归因任务", icon="fa-plus-circle"):
            st.caption("仅在当前视图加载模型与数据集，避免历史视图产生无效请求。")
            with st.form("inference_form"):
                models = _get_models()
                model_options = {item["name"]: item["id"] for item in models} if models else {}
                if model_options:
                    selected_model_name = st.selectbox("选择模型", list(model_options.keys()))
                    selected_model_id = model_options[selected_model_name]
                else:
                    st.warning("暂无可用模型，请先完成训练。")
                    selected_model_id = None

                datasets = [item for item in _get_datasets() if "Graph" in item.get("type", "")]
                dataset_options = {item["name"]: item["id"] for item in datasets} if datasets else {}
                if dataset_options:
                    selected_dataset_name = st.selectbox("选择待归因数据集", list(dataset_options.keys()))
                    selected_dataset_id = dataset_options[selected_dataset_name]
                else:
                    st.warning("暂无可用图数据集，请先完成预处理。")
                    selected_dataset_id = None

                submitted = st.form_submit_button("提交任务", type="primary", width="stretch")

            if submitted:
                if not selected_model_id or not selected_dataset_id:
                    st.error("请先选择有效的模型和数据集。")
                else:
                    _submit_inference(selected_model_id, selected_dataset_id)

    with right:
        with ui.section_card("任务记录", icon="fa-list-check"):
            render_task_panel(
                "inference",
                title_fn=_inference_task_title,
                key_prefix="inference",
                poll_state_key=POLL_STATE_KEY,
                limit=20,
                empty_message="暂无归因任务记录。",
                active_message="正在进行归因分析。",
            )


def _submit_inference(model_id: str, dataset_id: str) -> None:
    payload = {"model_id": model_id, "dataset_id": dataset_id}
    try:
        with st.spinner("正在提交归因任务..."):
            response = request("POST", "/api/inference", json_body=payload, timeout=(5, 30))
        if response.status_code == 202:
            invalidate("tasks")
            st.session_state[POLL_STATE_KEY] = True
            st.success(f"任务已提交：{response.json().get('task_id')}")
            st.rerun()
        else:
            st.error(f"提交失败: {response.text}")
    except requests.exceptions.ReadTimeout:
        invalidate("tasks")
        st.session_state[POLL_STATE_KEY] = True
        st.warning("请求超时，但任务可能已经进入队列。")
    except Exception as exc:
        st.error(f"提交异常: {exc}")


def _render_history_view() -> None:
    if ui.refresh_button("refresh_attr_results"):
        invalidate("attribution_results")
        st.rerun()

    results = _get_attribution_results(ttl="default")
    if not results:
        st.info("暂无历史归因结果。")
        return

    if "selected_attr_result_id" not in st.session_state or st.session_state["selected_attr_result_id"] not in {item["id"] for item in results}:
        st.session_state["selected_attr_result_id"] = results[0]["id"]

    st.caption("历史结果仅在当前视图加载；删除或手动刷新后更新。")
    list_col, detail_col = st.columns([1, 2.25], gap="large")

    with list_col:
        with ui.section_card("结果列表", icon="fa-clock-rotate-left"):
            labels = {
                f"{item['created']} · {item['total_samples']} 样本": item["id"]
                for item in results
            }
            current_id = st.session_state.get("selected_attr_result_id")
            current_label = next((label for label, rid in labels.items() if rid == current_id), next(iter(labels)))
            selected_label = st.radio(
                "结果列表",
                list(labels.keys()),
                index=list(labels.keys()).index(current_label),
                label_visibility="collapsed",
            )
            st.session_state["selected_attr_result_id"] = labels[selected_label]

    with detail_col:
        selected_result_id = st.session_state.get("selected_attr_result_id")
        detail = _get_attribution_detail(selected_result_id) if selected_result_id else None
        if not detail:
            st.warning("无法加载所选归因结果。")
            return

        head_col, delete_col = st.columns([4, 1])
        with head_col:
            st.markdown(
                '<div class="section-title"><i class="fas fa-bullseye"></i>'
                '<span>归因结果详情</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"结果 ID: `{selected_result_id}`")
        with delete_col:
            if st.button(f"🗑️ {ui.ACTION_LABELS['delete']}", key="delete_selected_attr_result", width="stretch"):
                if _delete_result(selected_result_id):
                    invalidate("attribution_results")
                    st.toast("结果已删除", icon="✅")
                    remaining = [item for item in results if item["id"] != selected_result_id]
                    st.session_state["selected_attr_result_id"] = remaining[0]["id"] if remaining else None
                    st.rerun()
                st.warning("删除失败")

        distribution = detail.get("label_distribution") or {}
        raw_results = detail.get("results") or []
        total_samples = detail.get("total_samples", len(raw_results))
        top_group = next(iter(sorted(distribution.items(), key=lambda item: item[1], reverse=True)), ("Unknown", 0))

        metric_cols = st.columns(3)
        metric_cols[0].metric("样本数", total_samples)
        metric_cols[1].metric("归因组织数", len(distribution))
        metric_cols[2].metric("首要归因", top_group[0])

        if distribution:
            chart_col, summary_col = st.columns([1, 1], gap="large")
            with chart_col:
                fig = px.pie(
                    _distribution_df(distribution),
                    values="数量",
                    names="APT 组织",
                    hole=0.55,
                    color_discrete_sequence=BRAND_SEQUENCE,
                )
                # The summary column already lists every group, so the legend is
                # redundant and only overlapped the chart — hide it and label
                # slices directly on the donut instead.
                fig.update_traces(
                    textposition="inside",
                    texttemplate="%{percent}",
                    textfont_size=12,
                    hovertemplate="%{label}<br>%{value} 样本 (%{percent})<extra></extra>",
                )
                apply_layout(fig, showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                             height=300, uniformtext_minsize=10, uniformtext_mode="hide",
                             annotations=[dict(
                                 text=f"<b>{total_samples}</b><br><span style='font-size:11px'>样本</span>",
                                 x=0.5, y=0.5, font_size=18, font_color="#e6edf3", showarrow=False,
                             )])
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CHART_CONFIG)

            with summary_col:
                st.markdown("**归因组织分布**")
                ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
                for name, count in ordered[:6]:
                    ratio = (count / total_samples) if total_samples else 0.0
                    st.progress(ratio, text=f"{name} · {count} / {total_samples}")
                if len(ordered) > 6:
                    st.caption(f"另有 {len(ordered) - 6} 个组织未列出。")

        if raw_results:
            st.subheader("样本明细")
            result_df = _result_rows(raw_results)
            st.dataframe(
                result_df,
                column_config={
                    "report_id": "报告 ID",
                    "predicted_label": "预测归因",
                    "confidence": st.column_config.ProgressColumn(
                        "置信度",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                    "top3": "Top-3 候选",
                },
                width="stretch",
                height=360,
            )

            st.subheader("解释证据")
            selector_col, export_col, report_col = st.columns([2.8, 1, 1], gap="small")
            with selector_col:
                sample_ids = [item.get("report_id") for item in raw_results]
                selected_sample_id = st.selectbox("选择样本", sample_ids, key="selected_attr_sample_id")
            with export_col:
                csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "导出 CSV",
                    csv_bytes,
                    "attribution_results.csv",
                    "text/csv",
                    width="stretch",
                )
            with report_col:
                if st.button("PDF 报告", width="stretch", key="generate_attr_pdf"):
                    with st.spinner("正在生成报告..."):
                        attrs = [
                            {
                                "name": name,
                                "score": count / total_samples if total_samples else 0.0,
                                "risk": "High" if idx == 0 else "Medium",
                            }
                            for idx, (name, count) in enumerate(
                                sorted(distribution.items(), key=lambda item: item[1], reverse=True)
                            )
                        ]
                        payload = {
                            "task_id": selected_result_id,
                            "analysis_results": {
                                "top_attribution": attrs[0]["name"] if attrs else "Unknown",
                                "attributions": attrs or [{"name": "Unknown", "score": 0.0, "risk": "Low"}],
                                "total_samples": total_samples,
                            },
                        }
                        response = request("POST", "/api/generate_report", json_body=payload, timeout=30)
                        if response.status_code == 200:
                            report_info = response.json()
                            report_url = f"{BACKEND_URL}{report_info['report_url']}"
                            st.success("报告生成成功。")
                            st.markdown(f"[打开 PDF 报告]({report_url})")
                        else:
                            st.error(f"生成失败: {response.text}")

            if selected_sample_id is not None:
                with st.expander("解释证据详情", expanded=True):
                    with st.spinner("正在加载该样本的解释证据…"):
                        selected_sample = _get_attribution_sample(selected_result_id, selected_sample_id)
                    if selected_sample:
                        _render_explanation_panel(selected_sample)
                    else:
                        st.info("当前样本没有可展示的解释证据。")


def render_attribution() -> None:
    ui.page_header("APT 归因结果", "APT Attribution Results", icon="fa-bullseye")

    if VIEW_STATE_KEY not in st.session_state:
        st.session_state[VIEW_STATE_KEY] = TASK_VIEW_LABEL

    current_view = st.segmented_control(
        "视图",
        [TASK_VIEW_LABEL, HISTORY_VIEW_LABEL],
        selection_mode="single",
        key=VIEW_STATE_KEY,
    )

    if current_view == TASK_VIEW_LABEL:
        _render_task_view()
    else:
        _render_history_view()
