from __future__ import annotations

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from apt_ui.services.api_client import (
    get_artifact_bytes,
    get_json,
    get_runtime_config,
    invalidate,
    request,
)
from apt_ui.services.charting import PLOTLY_CHART_CONFIG, apply_layout
from apt_ui.services.tasks import get_task_detail
from apt_ui.services.task_ui import TaskAction, default_delete_action, render_task_panel
from apt_ui.services import ui


POLL_STATE_KEY = "training_polling_active"
RESULT_STATE_KEY = "current_model_result"


def _get_datasets() -> list[dict]:
    return get_json("/api/datasets", timeout=3, default=[])


def _load_artifact_image(path: str) -> bytes | None:
    return get_artifact_bytes(path, timeout=5)


def _train_task_title(task: dict) -> str:
    return task.get("model") or task.get("name") or "训练任务"


def _train_task_subtitle(task: dict) -> str:
    return task.get("dataset") or "-"


def _view_result_action() -> TaskAction:
    def _handler(task: dict) -> bool:
        if task.get("status") != "completed" or not task.get("id"):
            return False
        detail = get_task_detail(task.get("id", ""), timeout=10, ttl="slow")
        if detail.get("result"):
            st.session_state[RESULT_STATE_KEY] = detail["result"]
            return True
        return False

    return TaskAction(
        label=ui.ACTION_LABELS["view"],
        handler=_handler,
        enabled=lambda task: task.get("status") == "completed",
        icon="👁️",
    )


@st.cache_data(ttl=300, show_spinner=False)
def _history_frames(history: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    loss_len = min(len(history.get("train_loss", [])), len(history.get("val_loss", [])))
    acc_len = min(len(history.get("train_acc", [])), len(history.get("val_acc", [])))
    loss_df = pd.DataFrame(
        {
            "Epoch": range(1, loss_len + 1),
            "Train Loss": history.get("train_loss", [])[:loss_len],
            "Val Loss": history.get("val_loss", [])[:loss_len],
        }
    )
    acc_df = pd.DataFrame(
        {
            "Epoch": range(1, acc_len + 1),
            "Train Acc": history.get("train_acc", [])[:acc_len],
            "Val Acc": history.get("val_acc", [])[:acc_len],
        }
    )
    return loss_df, acc_df


@st.cache_data(ttl=300, show_spinner=False)
def _classification_report_df(report: dict) -> pd.DataFrame:
    return pd.DataFrame(report).transpose()


def _render_result_panel() -> None:
    result = st.session_state.get(RESULT_STATE_KEY)
    if not result:
        ui.empty_state("点击任务记录中的「查看」后，这里会显示训练指标与图表。", icon="fa-chart-line")
        return

    ui.metric_row([
        ("Accuracy", f"{result.get('test_accuracy', 0):.2%}"),
        ("Weighted F1", f"{result.get('test_f1_weighted', 0):.2%}"),
        ("Temperature", f"{float(result.get('temperature', 1.0)):.3f}"),
    ])

    if st.button("清空结果", key="clear_train_result"):
        st.session_state.pop(RESULT_STATE_KEY, None)
        st.rerun()

    with st.expander("训练配置", expanded=False):
        st.json(result.get("config") or {})

    history = result.get("history") or {}
    if history:
        loss_df, acc_df = _history_frames(history)
        tab_loss, tab_acc = st.tabs(["损失曲线", "准确率曲线"])

        with tab_loss:
            fig_loss = px.line(loss_df, x="Epoch", y=["Train Loss", "Val Loss"], markers=True)
            apply_layout(fig_loss)
            st.plotly_chart(fig_loss, width="stretch", config=PLOTLY_CHART_CONFIG)

        with tab_acc:
            fig_acc = px.line(acc_df, x="Epoch", y=["Train Acc", "Val Acc"], markers=True)
            apply_layout(fig_acc)
            st.plotly_chart(fig_acc, width="stretch", config=PLOTLY_CHART_CONFIG)

    confusion_matrix_plot = result.get("confusion_matrix_plot")
    if confusion_matrix_plot:
        st.subheader("混淆矩阵")
        image_bytes = _load_artifact_image(confusion_matrix_plot)
        if image_bytes:
            st.image(image_bytes, caption="Confusion Matrix", width="stretch")
        else:
            st.warning("无法加载混淆矩阵图片。")

    report = result.get("classification_report")
    if report:
        with st.expander("分类报告", expanded=False):
            st.dataframe(_classification_report_df(report), width="stretch")

    model_path = result.get("model_path")
    if model_path:
        st.success(f"模型已保存至: `{model_path}`")


def render_clustering() -> None:
    runtime_config = get_runtime_config()
    if not runtime_config.get("training_enabled", True):
        ui.page_header("模型训练", "Server Resource Limited", icon="fa-project-diagram")
        st.warning("当前服务器算力不足，已关闭在线模型训练功能。")
        st.markdown(
            "建议前往 GitHub 本地部署并运行训练流程："
            " [https://github.com/mengyu-96/APT-CTIA](https://github.com/mengyu-96/APT-CTIA)"
        )
        return

    ui.page_header("模型训练", "Model Training", icon="fa-project-diagram")

    datasets = _get_datasets()
    graph_datasets = [item for item in datasets if "Graph" in item.get("type", "")]

    if "latest_preprocessing_run" in st.session_state:
        latest = st.session_state["latest_preprocessing_run"]
        latest_path = latest.get("output_path")
        if latest_path and not any(item.get("path") == latest_path for item in graph_datasets):
            graph_datasets.insert(
                0,
                {
                    "id": "latest",
                    "name": "最近一次预处理结果",
                    "path": latest_path,
                    "type": "Graph Collection",
                },
            )

    left, right = st.columns([0.82, 1.18], gap="large")

    with left:
        with ui.section_card("训练配置", icon="fa-sliders"):
            st.caption("仅展示可直接用于训练的图数据集。")
            with st.form("training_form"):
                if not graph_datasets:
                    st.warning("暂无可用图数据集，请先完成预处理。")
                    selected_dataset = None
                else:
                    dataset_options = {item["name"]: item for item in graph_datasets}
                    dataset_name = st.selectbox("选择数据集", list(dataset_options.keys()))
                    selected_dataset = dataset_options[dataset_name]

                algorithm = st.selectbox(
                    "模型架构",
                    ["GAT", "RGAT", "GCN", "GraphSAGE", "Transformer", "GIN", "Hybrid"],
                    index=0,
                )

                use_temporal = False
                temporal_hidden = 128
                if algorithm == "RGAT":
                    use_temporal = st.checkbox("启用时序模块（GRU）", value=False)
                    if use_temporal:
                        temporal_hidden = st.number_input("时序隐藏层维度", min_value=64, max_value=512, value=128)

                st.divider()
                epochs = st.number_input("训练轮数", min_value=10, max_value=5000, value=100)
                lr = st.number_input("学习率", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
                batch_size = st.number_input("Batch Size", min_value=1, max_value=1024, value=32, step=1)
                hidden_dim = st.select_slider("隐藏层维度", options=[64, 128, 256, 512], value=128)
                dropout = st.slider("Dropout", min_value=0.0, max_value=0.9, value=0.5)
                gradient_accumulation_steps = st.number_input("梯度累积步数", min_value=1, max_value=64, value=1, step=1)
                auto_memory_guard = st.checkbox("自动显存保护", value=True)
                submitted = st.form_submit_button("提交任务", type="primary", width="stretch")

            if submitted:
                if not selected_dataset:
                    st.error("请先选择一个有效的数据集。")
                else:
                    _submit_training(selected_dataset, algorithm, epochs, lr, batch_size,
                                     hidden_dim, dropout, gradient_accumulation_steps, auto_memory_guard,
                                     use_temporal, temporal_hidden)

        with ui.section_card("任务记录", icon="fa-list-check"):
            render_task_panel(
                "train",
                title_fn=_train_task_title,
                subtitle_fn=_train_task_subtitle,
                actions=[_view_result_action(), default_delete_action()],
                key_prefix="train",
                poll_state_key=POLL_STATE_KEY,
                empty_message="暂无训练记录。",
                active_message="训练进行中。",
            )

    with right:
        with ui.section_card("训练结果", icon="fa-chart-line"):
            _render_result_panel()


def _submit_training(selected_dataset, algorithm, epochs, lr, batch_size,
                     hidden_dim, dropout, gradient_accumulation_steps, auto_memory_guard,
                     use_temporal, temporal_hidden) -> None:
    payload = {
        "processed_data_path": selected_dataset["path"],
        "dataset_id": selected_dataset["id"],
        "dataset_name": selected_dataset["name"],
        "model_type": algorithm,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "auto_shrink_batch_size": bool(auto_memory_guard),
        "auto_scale_accumulation": bool(auto_memory_guard),
        "dynamic_batch_by_graph_size": bool(auto_memory_guard),
        "mixed_precision": True,
        "seed": 42,
        "use_temporal": use_temporal,
        "temporal_hidden_dim": temporal_hidden,
    }
    try:
        with st.spinner("正在提交训练任务..."):
            response = request("POST", "/api/train", json_body=payload, timeout=(5, 30))
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
        st.warning("请求超时，但任务可能已进入队列。")
    except Exception as exc:
        st.error(f"提交异常: {exc}")
