import streamlit as st
import pandas as pd
import plotly.express as px

from apt_ui.services.api_client import invalidate, get_json
from apt_ui.services.charting import PLOTLY_CHART_CONFIG, BRAND_SEQUENCE, apply_layout
from apt_ui.services import ui


def get_datasets():
    return get_json("/api/datasets", timeout=2, default=[])


def get_dataset_stats(dataset_id):
    return get_json(f"/api/datasets/{dataset_id}/stats", timeout=5, default=[], ttl="slow")


ENTITY_CATEGORIES = {
    "静态特征": ["FILE_PATH", "REGISTRY", "FILE_EXT", "HASH_MD5", "HASH_SHA1", "HASH_SHA256", "SSL_CERT"],
    "动态特征": ["PROCESS", "SERVICE", "USER_AGENT", "USER_ACCOUNT", "MUTEX", "PIPE", "CMD_LINE"],
    "网络特征": ["IP", "DOMAIN", "URL", "EMAIL", "PORT", "HOSTNAME", "PROTOCOL"],
    "威胁情报": ["MITRE_TECH", "CVE", "CWE", "MALWARE", "TOOL", "THREAT_INTEL_SOURCE", "APT_GROUP", "CAMPAIGN", "ORG", "INDUSTRY", "COUNTRY", "CITY"],
}


@st.cache_data(ttl=300, show_spinner=False)
def _aggregate_feature_stats(stats):
    total_entities = 0
    total_reports = len(stats)
    category_counts = {k: 0 for k in ENTITY_CATEGORIES}
    category_counts["其他"] = 0
    entity_type_counts = {}

    avg_nodes = avg_edges = 0
    if total_reports > 0:
        avg_nodes = sum(s.get("num_nodes", 0) for s in stats) / total_reports
        avg_edges = sum(s.get("num_edges", 0) for s in stats) / total_reports

    for report in stats:
        for etype, count in report.get("entity_counts", {}).items():
            entity_type_counts[etype] = entity_type_counts.get(etype, 0) + count
            total_entities += count
            placed = False
            for cat, types in ENTITY_CATEGORIES.items():
                if etype in types:
                    category_counts[cat] += count
                    placed = True
                    break
            if not placed:
                category_counts["其他"] += count

    return total_entities, total_reports, category_counts, entity_type_counts, avg_nodes, avg_edges


@st.cache_data(ttl=300, show_spinner=False)
def _feature_dataframes(category_counts, entity_type_counts, stats):
    df_cat = pd.DataFrame({"Category": list(category_counts), "Count": list(category_counts.values())})
    df_cat = df_cat[df_cat["Count"] > 0]

    sorted_types = sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    df_top = pd.DataFrame(sorted_types, columns=["Type", "Count"])

    all_types = []
    for etype, count in entity_type_counts.items():
        cat = "其他"
        for c, ts in ENTITY_CATEGORIES.items():
            if etype in ts:
                cat = c
                break
        all_types.append({"Type": etype, "Count": count, "Category": cat})

    df_detail = pd.DataFrame(all_types).sort_values("Count", ascending=False)
    df_graphs = pd.DataFrame(stats)
    return df_cat, df_top, df_detail, df_graphs


def render_features():
    ui.page_header("特征提取与可视化", "Feature Extraction & Visualization", icon="fa-chart-bar")

    datasets = get_datasets()
    if not datasets:
        st.info("暂无数据集。请先前往 [分析任务管理] 上传并预处理数据。")
        return

    processed_datasets = [d for d in datasets if "Graph" in d.get("type", "")]
    if not processed_datasets:
        st.warning("暂无已处理的图数据集。")
        return

    dataset_options = {d["name"]: d["id"] for d in processed_datasets}

    with ui.section_card():
        c1, c2 = st.columns([4, 1], gap="medium")
        with c1:
            selected_name = st.selectbox("选择数据集", list(dataset_options))
        with c2:
            st.write("")
            if ui.refresh_button("refresh_features"):
                invalidate("datasets")
                st.rerun()

    selected_id = dataset_options[selected_name]
    stats = get_dataset_stats(selected_id)
    if not stats:
        st.warning("该数据集暂无详细统计信息（可能是旧版本生成的，或预处理尚未完成）。")
        return

    total_entities, total_reports, category_counts, entity_type_counts, avg_nodes, avg_edges = _aggregate_feature_stats(stats)
    df_cat, df_top, df_detail, df_graphs = _feature_dataframes(category_counts, entity_type_counts, stats)

    ui.metric_row([
        ("样本报告数", total_reports),
        ("平均图节点数", f"{avg_nodes:.1f}"),
        ("平均图边数", f"{avg_edges:.1f}"),
        ("提取实体总数", total_entities),
    ])

    st.write("")
    tab_all, tab_detail, tab_insight = st.tabs(["总体分布", "详细统计", "深度洞察"])

    with tab_all:
        with ui.section_card("特征类别分布", icon="fa-chart-pie"):
            c_pie, c_bar = st.columns(2)
            with c_pie:
                fig_pie = px.pie(df_cat, values="Count", names="Category",
                                 color_discrete_sequence=BRAND_SEQUENCE, hole=0.5)
                # Keep labels inside the slices so percentages never overflow the
                # plot area; show the category name + percent on hover.
                fig_pie.update_traces(
                    textposition="inside",
                    texttemplate="%{percent}",
                    textfont_size=12,
                    insidetextorientation="horizontal",
                    hovertemplate="%{label}<br>%{value} (%{percent})<extra></extra>",
                )
                apply_layout(fig_pie, margin=dict(t=10, b=10, l=10, r=10), height=300,
                             uniformtext_minsize=10, uniformtext_mode="hide",
                             legend=dict(orientation="h", yanchor="top", y=-0.05,
                                         bgcolor="rgba(0,0,0,0)", font=dict(color="#93a4b3", size=11)))
                st.plotly_chart(fig_pie, width="stretch", config=PLOTLY_CHART_CONFIG)
            with c_bar:
                fig_bar = px.bar(df_top, x="Count", y="Type", orientation="h",
                                 color="Count", color_continuous_scale="Tealgrn")
                apply_layout(fig_bar, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_bar, width="stretch", config=PLOTLY_CHART_CONFIG)

    with tab_detail:
        with ui.section_card("各类特征详细统计", icon="fa-table"):
            max_count = int(df_detail["Count"].max()) if not df_detail.empty else 100
            st.dataframe(
                df_detail,
                column_config={
                    "Type": "实体类型",
                    "Count": st.column_config.ProgressColumn("数量", format="%d", min_value=0, max_value=max_count),
                    "Category": "所属类别",
                },
                width="stretch",
                hide_index=True,
                height=420,
            )

    with tab_insight:
        with ui.section_card("图结构洞察", icon="fa-diagram-project"):
            has_groups = "apt_group" in df_graphs.columns and df_graphs["apt_group"].notna().any()
            if has_groups:
                st.markdown("**各组织样本规模分布**")
                c_box1, c_box2 = st.columns(2)
                with c_box1:
                    fig_box_n = px.box(df_graphs, x="apt_group", y="num_nodes", color="apt_group")
                    apply_layout(fig_box_n, showlegend=False)
                    st.plotly_chart(fig_box_n, width="stretch", config=PLOTLY_CHART_CONFIG)
                with c_box2:
                    fig_box_e = px.box(df_graphs, x="apt_group", y="num_edges", color="apt_group")
                    apply_layout(fig_box_e, showlegend=False)
                    st.plotly_chart(fig_box_e, width="stretch", config=PLOTLY_CHART_CONFIG)
                st.divider()

            if {"num_nodes", "num_edges"}.issubset(df_graphs.columns):
                c1, c2 = st.columns(2)
                with c1:
                    fig_scatter = px.scatter(
                        df_graphs, x="num_nodes", y="num_edges",
                        color="apt_group" if has_groups else None,
                        hover_data=["report_id"] if "report_id" in df_graphs.columns else None,
                        render_mode="webgl", title="节点数 vs 边数（图复杂度）",
                    )
                    apply_layout(fig_scatter)
                    st.plotly_chart(fig_scatter, width="stretch", config=PLOTLY_CHART_CONFIG)
                with c2:
                    fig_hist = px.histogram(df_graphs, x="num_nodes", nbins=20, title="图规模分布")
                    apply_layout(fig_hist, bargap=0.1)
                    st.plotly_chart(fig_hist, width="stretch", config=PLOTLY_CHART_CONFIG)
            else:
                st.info("数据中缺少图结构统计信息。")
