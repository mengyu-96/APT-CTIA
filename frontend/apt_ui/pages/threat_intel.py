from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from apt_ui.state import get_samples


def render_threat_intel() -> None:
    st.header("4. 威胁情报关联")
    samples = get_samples()
    if not samples:
        st.warning("尚未上传样本。请先上传后再做情报关联。")
        return

    tab_mitre, tab_ext, tab_sim = st.tabs(["MITRE ATT&CK 映射", "外部威胁情报集成", "相似样本检索"])

    with tab_mitre:
        st.subheader("MITRE ATT&CK 映射（示例）")
        st.caption("演示版用矩阵热力图占位；后续可对接 ATT&CK STIX / TAXII / 本地库。")
        data = pd.DataFrame(
            [
                {"战术": "Initial Access", "技术": "T1566 Phishing", "强度": 0.8},
                {"战术": "Execution", "技术": "T1059 Command and Scripting", "强度": 0.6},
                {"战术": "Persistence", "技术": "T1547 Boot or Logon Autostart", "强度": 0.5},
                {"战术": "Command and Control", "技术": "T1071 Application Layer Protocol", "强度": 0.7},
            ]
        )
        fig = px.density_heatmap(data, x="战术", y="技术", z="强度", color_continuous_scale="Reds")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_ext:
        st.subheader("关联已知 APT 组织（示例）")
        org = st.selectbox("选择组织", ["APT29", "Lazarus", "Gamaredon", "Sidewinder"])
        st.markdown("**相关 CVE**")
        st.write(["CVE-2021-26855", "CVE-2022-30190"])
        st.markdown("**攻击报告链接**")
        st.write([f"{org} 报告 1（占位）", f"{org} 报告 2（占位）"])

    with tab_sim:
        st.subheader("相似样本检索（演示）")
        q = st.text_input("输入样本哈希（SHA256 或 MD5）", value="")
        if st.button("检索", type="primary"):
            if not q.strip():
                st.warning("请输入哈希。")
            else:
                # 演示：在会话 samples 中做“包含匹配”
                hits = []
                for s in samples:
                    if q.lower() in s["sha256"].lower() or q.lower() in s["md5"].lower():
                        hits.append(s)
                if not hits:
                    st.info("未找到相似样本（演示检索）。")
                else:
                    st.success(f"找到 {len(hits)} 个候选样本：")
                    st.dataframe(
                        [
                            {"文件名": h["name"], "类型": h["file_type"], "SHA256": h["sha256"], "MD5": h["md5"]}
                            for h in hits
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

