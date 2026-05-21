from __future__ import annotations

import streamlit as st


CASES = {
    "Sidewinder：多平台攻击": [
        "导入样本（Windows/Android/Linux）并提取静态与连接特征",
        "以活动聚类聚合样本，观察共享基础设施与代码模式",
        "组织聚类将多活动合并为同一组织视图",
        "输出报告：证据链（IOC/TTP/样本谱系）",
    ],
    "Gamaredon：2017 vs 2022 活动对比": [
        "分别导入两时期样本并提取特征",
        "对比 TTP 分布（宏 → PowerShell → 下载器链）",
        "分析基础设施迁移（域名/证书/AS）与聚类漂移",
        "输出时间轴与关键证据摘要",
    ],
    "APT29：WellMess / WellMail 聚类": [
        "导入 WellMess/WellMail 样本集",
        "聚类依据：共享代码模式、C2 指标、字符串与编译信息",
        "映射到 ATT&CK 技术集合并生成矩阵",
        "生成组织级报告并导出 STIX 2.0",
    ],
    "Lazarus：加密货币恶意软件分析": [
        "导入钱包窃取与交易所入侵相关样本",
        "提取网络/钱包地址/证书等连接特征",
        "组织级聚类观察跨活动复用组件",
        "报告导出：IOC 列表、网络图、时间线",
    ],
}


def render_case_studies() -> None:
    st.header("5. 案例研究展示")
    st.caption("用于演示与教学：点击案例查看分析步骤（可替换为论文/数据集的真实过程）。")

    case = st.selectbox("选择案例", list(CASES.keys()))
    st.subheader(case)

    steps = CASES[case]
    for i, s in enumerate(steps, start=1):
        st.write(f"{i}. {s}")

