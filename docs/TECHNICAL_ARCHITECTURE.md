# APT归因系统技术架构文档

本文档详细描述了APT归因系统的全链路技术实现，从原始数据预处理到最终模型训练与推理。

## 1. 系统全链路概览

```mermaid
graph TD
    A[原始数据 (dataset_TXT/)] -->|preprocess_apt_dataset.py| B(数据清洗与实体提取)
    B --> C{中间数据 (results_archive/)}
    C -->|Graph Construction| D[异构图数据 (PyG Data)]
    D -->|train_gnn.py| E[图神经网络模型]
    E --> F[归因结果]
    E -->|utils/analyze_errors.py| G[错误分析与可视化]
    H[基线模型] -->|train_baseline_models.py| I[对比分析]
```

---

## 2. 阶段一：数据预处理 (Preprocessing)

**核心脚本**: `preprocess_apt_dataset.py`

此阶段负责将非结构化的威胁情报报告（CTI Reports）转化为结构化的实体与文本数据。

### 2.1 数据摄入 (Data Ingestion)
- **输入**: 包含PDF或TXT文件的目录结构（通常按APT组织分类）。
- **元数据解析**: 脚本自动解析文件名以提取关键元数据。
  - 格式约定: `Group_Date_Vendor_Title.ext` (例如 `APT28_20230101_FireEye_CampaignX.pdf`)
  - 提取字段: `apt_group`, `published_date`, `source_vendor`。
- **文件读取**:
  - **TXT**: 直接读取 UTF-8 文本。
  - **PDF**: 采用混合策略。首选 `pdfplumber` 以保留文本布局和准确性；若失败则回退到 `fitz` (PyMuPDF) 保证鲁棒性。

### 2.2 实体提取 (Entity Extraction)
采用 **规则+字典** 的混合提取引擎 (`EntityExtractor` 类)：
1.  **正则提取 (Regex)**: 针对格式固定的IoC（入侵指标）。
    - `IPv4`, `IPv6`, `Email`, `URL`, `MD5`, `SHA1`, `SHA256`, `CVE`, `Registry Key`, `File Path`.
2.  **字典匹配 (Dictionary)**: 针对命名实体（需加载 `vocabulary_dir` 下的JSON词表）。
    - `Malware` (恶意软件名), `Tool` (黑客工具), `Actor` (组织名), `Country`, `Industry`, `MITRE ATT&CK Techniques`.
    - 匹配算法: 自动处理大小写，支持多关键词匹配。

### 2.3 中间产物
处理结果暂存为 JSONL 格式，便于调试和分步处理：
- `paragraphs.jsonl`: 包含清洗后的段落文本及其元数据。
- `entities.jsonl`: 包含每个段落中提取到的实体及其位置信息。

---

## 3. 阶段二：图构建 (Graph Construction)

**核心逻辑**: `GraphBuilder.build_graph()`

此阶段是将“文本信息”转化为“图结构知识”的关键步骤。系统采用 **领域知识驱动 (Domain-Knowledge Driven)** 的构图策略。

### 3.1 节点构建 (Node Construction)
- **实体节点**: 上一步提取的所有实体（IP, Malware, etc.）。
- **报告节点** (可选): 一个代表整篇报告的超级节点 (`__root__`)，用于汇聚全局信息。
- **节点特征 (Node Features)**:
  - 初始特征: 词频/TF-IDF 向量（反映实体在文档中的重要性）。
  - 类型编码: One-hot 编码实体的类型（如 `IP` vs `Malware`）。

### 3.2 边构建策略 (Edge Construction Strategies)
系统实现了6层渐进式的连边策略，从统计共现到语义关联：

1.  **段落内共现 (Intra-paragraph Co-occurrence)**: 同一段落内的实体两两相连（假设它们在语义上相关）。
2.  **跨段落连接 (Inter-paragraph Connection)**: 若同一实体出现在不同段落，连接这些段落的其他实体（构建文档级上下文）。
3.  **报告根节点连接**: 所有实体连接到 `__root__` 节点（如果启用）。
4.  **领域语义规则 (Semantic Rules)**: **(核心创新点)**
    - *归属关系*: `Operation` -> `APT Group`
    - *使用关系*: `Malware` -> `MITRE_Tech` (技术), `Operation` -> `CVE` (漏洞)
    - *基础设施*: `Hash` -> `IP/Domain` (C2通信)
    - *投放关系*: `Email` -> `Hash` (钓鱼附件)
    - *持久化*: `Hash` -> `Registry/FilePath`
5.  **小图增强**: 若提取实体极少 (<5)，通过全连接增强信息流，防止图卷积失效。
6.  **自环 (Self-loops)**: 保证每个节点都能聚合自身特征。

### 3.3 文档级特征注入 (Document Embedding)
- 使用 `sentence-transformers` (如 `all-MiniLM-L6-v2`) 对整篇报告文本进行编码。
- 生成 `doc_emb` (维度如 384)，作为图的全局属性 (`data.doc_emb`)。
- **用途**: 在 GraphSAGE 模型中与图特征融合，解决图稀疏问题。

---

## 4. 阶段三：模型训练 (Model Training)

**核心脚本**: `train_gnn.py`

### 4.1 数据加载与增强
- **分层划分**: 保证训练/验证/测试集中各 APT 组织的比例一致。
- **类别不平衡处理**:
  - **Oversampling**: 使用 `WeightedRandomSampler` 对少样本类别进行过采样。
  - **Class Weights**: 计算类别频率倒数，作为 Loss 权重。

### 4.2 模型架构
支持多种 GNN 变体，推荐架构如下：

#### GraphSAGE (With Text Fusion)
```python
# 伪代码逻辑
x = SageConv(x, edge_index)  # 图卷积提取结构特征
x_graph = GlobalMeanPool(x)  # 聚合为图向量
x_final = Concat(x_graph, doc_emb) # 拼接文档文本特征
logits = Classifier(x_final)
```
此架构结合了 GNN 的结构推理能力和 BERT 的语义理解能力。

#### GAT (Graph Attention Network)
利用 Attention 机制自动学习不同实体对归因的重要性（例如：恶意软件 Hash 的权重 > 普通 Country 实体）。

### 4.3 训练策略
- **损失函数**:
  - `CrossEntropy`: 标准多分类损失。
  - `Focal Loss`: (`--focal`) 降低易分类样本权重，专注于难样本。
- **Logit Adjustment**: 减去类别先验的对数 (`log(p_class)`)，从理论上修正长尾分布带来的偏差。
- **优化器**: AdamW + ReduceLROnPlateau (动态调整学习率)。

---

## 5. 阶段四：推理与评估 (Inference & Evaluation)

**核心脚本**: `analyze_errors.py`

### 5.1 评估指标
- **Accuracy**: 总体准确率。
- **Balanced Accuracy**: 各类别准确率的平均值（对不平衡数据更公平）。
- **Top-K Accuracy**: 真实标签在前 K 个预测中的比例（通常 K=3）。
- **F1-Macro**: 宏平均 F1 分数。

### 5.2 错误分析
脚本会生成详细的 `misclassification_analysis.json`，包含：
- 误判样本的详细信息（文件名、真实标签、预测标签、置信度）。
- 混淆矩阵：识别哪些组织容易混淆（例如 APT28 和 APT29）。
- **置信度校准**: 分析模型对错误预测是否过于自信。

---

## 总结
该架构通过 **"规则提取 + 知识构图 + 深度学习"** 的范式，解决了传统基于特征工程方法的局限性，特别是在处理非结构化文本和长尾分布数据时展现了较强的鲁棒性。
