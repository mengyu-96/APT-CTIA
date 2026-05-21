from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from apt_ui.state import get_samples


def render_dataset() -> None:
    st.header("6. 数据集管理")
    samples = get_samples()
    if not samples:
        st.warning("当前会话没有样本。请先上传一些样本。")
        return

    st.subheader("样本库浏览与搜索")
    q = st.text_input("按文件名/类型/哈希搜索", value="")
    rows = []
    for s in samples:
        rows.append(
            {
                "文件名": s["name"],
                "类型": s["file_type"],
                "SHA256": s["sha256"],
                "MD5": s["md5"],
                "大小(KB)": round(s["size_bytes"] / 1024, 2),
                "标签": s.get("tags", []),
            }
        )
    df = pd.DataFrame(rows)
    if q.strip():
        ql = q.lower()
        df = df[
            df["文件名"].str.lower().str.contains(ql)
            | df["类型"].str.lower().str.contains(ql)
            | df["SHA256"].str.lower().str.contains(ql)
            | df["MD5"].str.lower().str.contains(ql)
        ]
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("标签编辑与标准化（演示）")
    sha = st.selectbox("选择样本（SHA256）", [s["sha256"] for s in samples])
    tags = st.text_input("标签（逗号分隔）", value="")
    if st.button("保存标签", type="primary"):
        for s in samples:
            if s["sha256"] == sha:
                s["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
                st.success("已保存。")
                break

    st.divider()
    st.subheader("数据统计仪表板（示例）")
    fig = px.histogram(df, x="类型")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

