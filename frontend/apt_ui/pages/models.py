from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from apt_ui.services.api_client import invalidate, get_json, request
from apt_ui.services.charting import PLOTLY_CHART_CONFIG, apply_layout
from apt_ui.services import ui


MODEL_DISPLAY_NAMES = {
    "RGAT": "GRACE",
    "GAT": "APT-ATT",
    "Hybrid": "APT-MMF",
    "GCN": "MLDSJ",
    "Transformer": "Mead",
    "GraphSAGE": "TRAIL",
}

def _display_model(model: dict) -> dict:
    """Apply product-facing names without changing backend model identifiers."""
    item = dict(model)
    model_type = str(item.get("type", ""))
    item["display_name"] = MODEL_DISPLAY_NAMES.get(model_type, item.get("name", "Unknown"))
    item["display_type"] = MODEL_DISPLAY_NAMES.get(model_type, model_type or "Unknown")
    return item


def _get_models() -> list[dict]:
    return get_json(
        "/api/models",
        params={"include_report": "true"},
        timeout=5,
        default=[],
    )


def _get_model_detail(model_id: str) -> dict | None:
    return get_json(f"/api/models/{model_id}", timeout=5, default=None, ttl="slow")


def _rename_model(model_id: str, new_name: str) -> tuple[bool, str]:
    try:
        response = request("PUT", f"/api/models/{model_id}/rename", json_body={"new_name": new_name}, timeout=5)
        if response.status_code == 200:
            return True, "模型已重命名。"
        payload = response.json()
        return False, payload.get("error", "未知错误")
    except Exception as exc:
        return False, str(exc)


def _delete_model(model_id: str) -> tuple[bool, str]:
    try:
        response = request("DELETE", f"/api/models/{model_id}", timeout=5)
        if response.status_code == 200:
            return True, "模型已删除。"
        payload = response.json()
        return False, payload.get("error", "未知错误")
    except Exception as exc:
        return False, str(exc)


@st.cache_data(ttl=300, show_spinner=False)
def _performance_df(models: list[dict]) -> pd.DataFrame:
    rows = []
    for model in models:
        report = model.get("classification_report") or {}
        weighted = report.get("weighted avg") or {}
        rows.append(
            {
                "模型": model.get("display_name", model.get("name", "Unknown")),
                "准确率": float(model.get("accuracy", 0.0) or 0.0),
                "加权精确率": float(weighted.get("precision", 0.0) or 0.0),
                "加权召回率": float(weighted.get("recall", 0.0) or 0.0),
                "加权F1": float(weighted.get("f1-score", model.get("f1_score", 0.0)) or 0.0),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def _class_report_df(report: dict) -> pd.DataFrame:
    rows = []
    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"} or not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "Class": label,
                "Precision": metrics.get("precision", 0.0),
                "Recall": metrics.get("recall", 0.0),
                "F1-Score": metrics.get("f1-score", 0.0),
                "Support": metrics.get("support", 0),
            }
        )
    return pd.DataFrame(rows)


def render_models() -> None:
    ui.page_header("模型管理", "Model Management", icon="fa-microchip")

    if ui.refresh_button("refresh_models", label="刷新模型列表"):
        invalidate("models")
        st.rerun()

    models = _get_models()
    if not models:
        st.info("暂无已训练模型，请先在「模型训练」页面提交训练任务。")
        return

    display_models = [_display_model(model) for model in models]
    latest_model = display_models[0]
    ui.metric_row([
        ("模型数量", len(models)),
        ("最新模型 Accuracy", f"{latest_model.get('accuracy', 0):.3f}"),
        ("最新模型 F1", f"{latest_model.get('f1_score', 0):.3f}"),
    ])

    chart_col, info_col = st.columns([2, 1], gap="large")
    with chart_col:
        with ui.section_card("APT-CTI 数据集模型性能对比", icon="fa-chart-column"):
            df_perf = _performance_df(display_models)
            metric_columns = ["准确率", "加权精确率", "加权召回率", "加权F1"]
            fig = px.bar(df_perf, x="模型", y=metric_columns, barmode="group")
            fig.update_yaxes(range=[0, 1], tickformat=".2f")
            apply_layout(fig, xaxis_title="模型", yaxis_title="性能得分")
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CHART_CONFIG)

    with info_col:
        with ui.section_card("最新模型", icon="fa-star"):
            st.metric("名称", latest_model["display_name"])
            st.metric("数据集", latest_model.get("dataset_name", "Unknown"))
            st.caption(f"Batch Size: {latest_model.get('batch_size', 32)} · 创建时间: {latest_model.get('created', '-')}")

    search_col, _ = st.columns([2, 3])
    with search_col:
        keyword = st.text_input("搜索模型", placeholder="按模型名称筛选")

    filtered_models = [
        item for item in display_models
        if keyword.lower() in item["display_name"].lower()
    ]

    selected_model_id = st.session_state.get("selected_model_id")
    if selected_model_id:
        detail = _get_model_detail(selected_model_id)
        if detail:
            with ui.section_card():
                detail_title_col, close_col = st.columns([5, 1])
                with detail_title_col:
                    st.markdown(
                        f'<div class="section-title"><i class="fas fa-circle-info"></i>'
                        f'<span>模型详情：{MODEL_DISPLAY_NAMES.get(detail.get("type", ""), detail["name"])}</span></div>',
                        unsafe_allow_html=True,
                    )
                with close_col:
                    if st.button("关闭详情", width="stretch"):
                        st.session_state["selected_model_id"] = None
                        st.rerun()

                report = detail.get("classification_report") or {}
                report_df = _class_report_df(report)
                left, right = st.columns([1.3, 1], gap="large")
                with left:
                    if not report_df.empty:
                        fig_cls = px.bar(report_df, x="Class",
                                         y=["Precision", "Recall", "F1-Score"], barmode="group")
                        apply_layout(fig_cls)
                        st.plotly_chart(fig_cls, width="stretch", config=PLOTLY_CHART_CONFIG)
                    else:
                        st.info("当前模型没有详细分类报告。")
                with right:
                    st.caption(f"Epochs: {detail.get('epochs', 0)} · Batch Size: {detail.get('batch_size', 32)}")
                    macro = report.get("macro avg", {})
                    weighted = report.get("weighted avg", {})
                    radar_df = pd.DataFrame({
                        "metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"],
                        "value": [
                            report.get("accuracy", 0.0),
                            macro.get("precision", 0.0),
                            macro.get("recall", 0.0),
                            macro.get("f1-score", 0.0),
                            weighted.get("f1-score", 0.0),
                        ],
                    })
                    fig_radar = px.line_polar(radar_df, r="value", theta="metric", line_close=True)
                    fig_radar.update_traces(fill="toself")
                    apply_layout(fig_radar, polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                    st.plotly_chart(fig_radar, width="stretch", config=PLOTLY_CHART_CONFIG)

                if not report_df.empty:
                    st.dataframe(report_df, width="stretch", hide_index=True)

    st.markdown('<div class="section-title"><i class="fas fa-layer-group"></i><span>模型列表</span></div>', unsafe_allow_html=True)
    if not filtered_models:
        st.info("没有匹配的模型。")
        return

    for idx, model in enumerate(filtered_models):
        with ui.section_card():
            title_col, stat_col, action_col = st.columns([3, 2, 2], gap="large")

            with title_col:
                st.markdown(f"**{model['display_name']}**")
                st.caption(
                    f"类型: {model.get('display_type', 'Unknown')} · 数据集: {model.get('dataset_name', 'Unknown')} · "
                    f"轮数: {model.get('epochs', 0)} · Batch Size: {model.get('batch_size', 32)} · 创建: {model.get('created', '-')}"
                )

            with stat_col:
                c1, c2 = st.columns(2)
                c1.metric("Accuracy", f"{model.get('accuracy', 0):.3f}")
                c2.metric("F1", f"{model.get('f1_score', 0):.3f}")

            with action_col:
                if st.button(f"👁️ {ui.ACTION_LABELS['view']}", key=f"view_model_{idx}", width="stretch"):
                    st.session_state["selected_model_id"] = model["id"]
                    st.rerun()

                rename_key = f"rename_value_{model['id']}"
                if rename_key not in st.session_state:
                    st.session_state[rename_key] = model["display_name"]
                st.text_input("新名称", key=rename_key, label_visibility="collapsed",
                              placeholder="输入新名称…")

                rename_col, delete_col = st.columns(2)
                with rename_col:
                    if st.button(f"✏️ {ui.ACTION_LABELS['rename']}", key=f"rename_model_{idx}", width="stretch"):
                        ok, message = _rename_model(model["id"], st.session_state[rename_key].strip())
                        if ok:
                            invalidate("models")
                            st.toast(message, icon="✅")
                            st.rerun()
                        st.error(message)
                with delete_col:
                    if st.button(f"🗑️ {ui.ACTION_LABELS['delete']}", key=f"delete_model_{idx}", width="stretch"):
                        ok, message = _delete_model(model["id"])
                        if ok:
                            invalidate("models")
                            if st.session_state.get("selected_model_id") == model["id"]:
                                st.session_state["selected_model_id"] = None
                            st.toast(message, icon="✅")
                            st.rerun()
                        st.error(message)
