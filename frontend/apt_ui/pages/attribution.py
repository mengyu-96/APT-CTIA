from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from apt_ui.services import ui
from apt_ui.services.api_client import get_binary, get_json, invalidate, request
from apt_ui.services.charting import BRAND_SEQUENCE, PLOTLY_CHART_CONFIG, apply_layout
from apt_ui.services.presentation import compact_identifier
from apt_ui.services.task_ui import render_task_panel


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")
POLL_STATE_KEY = "inference_polling_active"
VIEW_STATE_KEY = "attribution_page_view"

TASK_VIEW_LABEL = "新建归因任务"
HISTORY_VIEW_LABEL = "历史归因结果"
MODEL_DISPLAY_NAMES = {
    "RGAT": "HERA",
    "GAT": "APT-ATT",
    "Hybrid": "APT-MMF",
    "GCN": "MLDSJ",
    "Transformer": "Mead",
    "GraphSAGE": "TRAIL",
}
REVIEW_DECISIONS = {
    "确认模型结论": "confirmed",
    "驳回模型结论": "rejected",
    "改判其他组织": "corrected",
    "证据不足": "insufficient",
}


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


def _get_reviews(result_id: str) -> list[dict]:
    return get_json(
        f"/api/attribution_results/{result_id}/reviews",
        timeout=5,
        default=[],
        ttl="default",
    )


@st.cache_data(ttl=300, show_spinner=False)
def _distribution_df(distribution: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(list(distribution.items()), columns=["APT组织", "数量"])


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
                "model_prediction": item.get("model_prediction", item.get("predicted_label")),
                "confidence": float(item.get("confidence", 0) or 0) * 100,
                "decision": "需人工复核"
                if (item.get("decision") or {}).get("status") == "review_required"
                else "已通过自动判定",
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


def _load_report_bytes(report_info: dict) -> bytes | None:
    download_path = report_info.get("download_path") or report_info.get("report_url")
    if download_path:
        return get_binary(download_path, timeout=30)
    return None


def _render_explanation_panel(sample_result: dict) -> None:
    explanation = sample_result.get("explanation") or {}
    if not explanation:
        st.info("当前样本没有可展示的解释证据。")
        return

    decision_mode = explanation.get("decision_mode") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("模型首选", sample_result.get("model_prediction", sample_result.get("predicted_label", "Unknown")))
    c2.metric("置信度", f"{sample_result.get('confidence', 0):.2%}")
    signal_labels = {
        "semantic": "语义证据",
        "structural": "结构证据",
        "balanced": "语义与结构均衡",
    }
    dominant_signal = decision_mode.get("dominant_signal", "unknown")
    c3.metric("主导信号", signal_labels.get(dominant_signal, "暂无判断"))

    decision = sample_result.get("decision") or {}
    if decision.get("status") == "review_required":
        reasons = "；".join(decision.get("reasons") or ["未达到自动归因条件"])
        st.warning(f"系统已拒绝自动归因并转人工复核：{reasons}")
    elif decision:
        st.success("当前结果达到自动归因阈值，仍可由分析师复核。")

    if decision_mode:
        st.caption(
            f"语义权重 {decision_mode.get('semantic_gate', 0):.2%} | "
            f"结构权重 {decision_mode.get('structural_gate', 0):.2%}。"
        )
    st.caption("置信度是当前模型的输出得分，需结合证据完整性、情报时效和人工研判使用。")

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

    techniques = (explanation.get("mitre_attack") or {}).get("techniques") or []
    if techniques:
        st.markdown("**MITRE ATT&CK 摘要**")
        st.dataframe(pd.DataFrame(techniques), width="stretch", hide_index=True)


def _render_analyst_review(result_id: str, sample_id: str, reviews: list[dict]) -> None:
    st.markdown("**分析师复核**")
    sample_reviews = [item for item in reviews if str(item.get("sample_id")) == str(sample_id)]
    if sample_reviews:
        latest = sample_reviews[-1]
        decision_label = next(
            (label for label, value in REVIEW_DECISIONS.items() if value == latest.get("decision")),
            latest.get("decision", "未知"),
        )
        detail = f"最近结论：{decision_label}"
        if latest.get("corrected_label"):
            detail += f" → {latest['corrected_label']}"
        if latest.get("reviewer"):
            detail += f"；复核人：{latest['reviewer']}"
        st.info(detail)
        if latest.get("note"):
            st.caption(f"复核说明：{latest['note']}")

    with st.form(f"analyst_review_{result_id}_{sample_id}"):
        decision_label = st.selectbox("复核结论", list(REVIEW_DECISIONS))
        corrected_label = st.text_input("改判组织", placeholder="仅在选择“改判其他组织”时必填")
        note = st.text_area("复核说明", max_chars=2000)
        reviewer = st.text_input(
            "复核人",
            value=str(st.session_state.get("auth_username", "")),
            max_chars=100,
        )
        submitted = st.form_submit_button("保存复核结论", type="primary", width="stretch")

    if not submitted:
        return
    decision = REVIEW_DECISIONS[decision_label]
    if decision == "corrected" and not corrected_label.strip():
        st.error("改判时必须填写新的 APT 组织名称。")
        return
    try:
        response = request(
            "POST",
            f"/api/attribution_results/{result_id}/reviews",
            json_body={
                "sample_id": str(sample_id),
                "decision": decision,
                "corrected_label": corrected_label.strip(),
                "note": note.strip(),
                "reviewer": reviewer.strip(),
            },
            timeout=10,
        )
    except Exception as exc:
        st.error(f"保存失败：{exc}")
        return
    if response.status_code != 201:
        st.error(f"保存失败：{response.text}")
        return
    invalidate("attribution_results")
    st.success("复核结论已保存。")
    st.rerun()


def _inference_task_title(task: dict) -> str:
    name = task.get("name") or "归因任务"
    return "归因任务" if name == "Attribution Inference" else name


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


def _render_task_view() -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with ui.section_card("新建归因任务", icon="fa-plus-circle"):
            st.caption("仅在当前视图加载模型与数据集，避免历史视图触发无效请求。")
            with st.form("inference_form"):
                models = _get_models()
                model_options = {
                    MODEL_DISPLAY_NAMES.get(item.get("type"), item["name"]): item["id"]
                    for item in models
                } if models else {}
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


def _render_result_report_download(selected_result_id: str, distribution: dict[str, int], total_samples: int) -> None:
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
    if response.status_code != 200:
        st.error(f"生成失败: {response.text}")
        return

    report_info = response.json()
    pdf_bytes = _load_report_bytes(report_info)
    st.success("报告生成成功。")
    if pdf_bytes:
        st.download_button(
            "下载 PDF 报告",
            data=pdf_bytes,
            file_name=report_info.get("report_name", f"report_{selected_result_id}.pdf"),
            mime="application/pdf",
            width="stretch",
            key=f"download_attr_pdf_{selected_result_id}",
        )
        return

    download_path = report_info.get("download_path") or report_info.get("report_url")
    if download_path:
        st.link_button("打开 PDF 报告", f"{BACKEND_URL}{download_path}", width="stretch")
    else:
        st.warning("报告已生成，但暂时无法获取下载文件。")


def _render_history_view() -> None:
    if ui.refresh_button("refresh_attr_results"):
        invalidate("attribution_results")
        st.rerun()

    results = _get_attribution_results(ttl="default")
    if not results:
        st.info("暂无历史归因结果。")
        return

    valid_ids = {item["id"] for item in results}
    if st.session_state.get("selected_attr_result_id") not in valid_ids:
        st.session_state["selected_attr_result_id"] = results[0]["id"]

    st.caption("历史结果仅在当前视图加载；删除或手动刷新后更新。")
    list_col, detail_col = st.columns([1, 2.25], gap="large")

    with list_col:
        with ui.section_card("结果列表", icon="fa-clock-rotate-left"):
            labels = {
                f"{item['created']} - {item['total_samples']} 样本": item["id"]
                for item in results
            }
            current_id = st.session_state["selected_attr_result_id"]
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
                '<div class="section-title"><i class="fas fa-bullseye"></i><span>归因结果详情</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"结果编号：`{compact_identifier(selected_result_id)}`")
        with delete_col:
            def delete_selected_result() -> None:
                if _delete_result(selected_result_id):
                    invalidate("attribution_results")
                    st.toast("结果已删除", icon="✅")
                    remaining = [item for item in results if item["id"] != selected_result_id]
                    st.session_state["selected_attr_result_id"] = remaining[0]["id"] if remaining else None
                else:
                    st.warning("删除失败。")

            ui.confirm_delete(
                "delete_selected_attr_result",
                f"归因结果“{compact_identifier(selected_result_id)}”",
                delete_selected_result,
            )

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
                    names="APT组织",
                    hole=0.55,
                    color_discrete_sequence=BRAND_SEQUENCE,
                )
                fig.update_traces(
                    textposition="inside",
                    texttemplate="%{percent}",
                    textfont_size=12,
                    hovertemplate="%{label}<br>%{value} 样本 (%{percent})<extra></extra>",
                )
                apply_layout(
                    fig,
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=300,
                    uniformtext_minsize=10,
                    uniformtext_mode="hide",
                    annotations=[
                        dict(
                            text=f"<b>{total_samples}</b><br><span style='font-size:11px'>样本</span>",
                            x=0.5,
                            y=0.5,
                            font_size=18,
                            font_color="#e6edf3",
                            showarrow=False,
                        )
                    ],
                )
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CHART_CONFIG)

            with summary_col:
                st.markdown("**归因组织分布**")
                ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
                for name, count in ordered[:6]:
                    ratio = (count / total_samples) if total_samples else 0.0
                    st.progress(ratio, text=f"{name} - {count} / {total_samples}")
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
                        max_value=100.0,
                        format="%.2f%%",
                    ),
                    "top3": "Top-3 候选",
                },
                width="stretch",
                height=360,
            )

            st.subheader("解释证据")
            selector_col, csv_col, json_col, stix_col, report_col = st.columns([2.4, 1, 1, 1, 1], gap="small")
            with selector_col:
                sample_ids = [item.get("report_id") for item in raw_results]
                selected_sample_id = st.selectbox("选择样本", sample_ids, key="selected_attr_sample_id")
            with csv_col:
                csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "导出 CSV",
                    csv_bytes,
                    f"attribution_results_{selected_result_id}.csv",
                    "text/csv",
                    width="stretch",
                )
            with json_col:
                json_bytes = get_binary(
                    f"/api/attribution_results/{selected_result_id}/export",
                    params={"format": "json"},
                    timeout=20,
                )
                st.download_button(
                    "导出 JSON",
                    data=json_bytes or b"",
                    file_name=f"attribution_{selected_result_id}.json",
                    mime="application/json",
                    width="stretch",
                    disabled=not json_bytes,
                )
            with stix_col:
                stix_bytes = get_binary(
                    f"/api/attribution_results/{selected_result_id}/export",
                    params={"format": "stix"},
                    timeout=20,
                )
                st.download_button(
                    "导出 STIX",
                    data=stix_bytes or b"",
                    file_name=f"attribution_{selected_result_id}.stix.json",
                    mime="application/json",
                    width="stretch",
                    disabled=not stix_bytes,
                )
            with report_col:
                if st.button("PDF 报告", width="stretch", key="generate_attr_pdf"):
                    with st.spinner("正在生成报告..."):
                        _render_result_report_download(selected_result_id, distribution, total_samples)

            if selected_sample_id is not None:
                with st.expander("解释证据详情", expanded=True):
                    with st.spinner("正在加载该样本的解释证据..."):
                        selected_sample = _get_attribution_sample(selected_result_id, selected_sample_id)
                    if selected_sample:
                        _render_explanation_panel(selected_sample)
                        _render_analyst_review(
                            selected_result_id,
                            str(selected_sample_id),
                            _get_reviews(selected_result_id),
                        )
                    else:
                        st.info("当前样本没有可展示的解释证据。")


def render_attribution() -> None:
    ui.page_header("APT 归因结果", "创建归因任务并核验证据", icon="fa-bullseye")

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
