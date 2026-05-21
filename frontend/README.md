# APT 归因分析（Python 前端页面 / Streamlit）

本项目用 **Python + Streamlit** 实现一个面向 ADAPT/APT 归因流程的交互式“前端页面”，覆盖：

- 样本上传与预处理
- 特征提取与可视化
- 模型训练（活动/组织两级）
- 威胁情报关联（MITRE ATT&CK / 外部情报）
- 案例研究展示
- 数据集管理
- 报告生成与导出（PDF / JSON / STIX 2.0）

## 运行

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 启动

```bash
streamlit run app.py
```

> 说明：目前提供 UI 与演示数据/流程占位；后续可对接你的后端（ADAPT 特征提取、聚类、情报接口）替换 `services` 层即可。

