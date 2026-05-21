# 2024-2025学年第一学期 APT归因研究进展汇报

## 1. 研究背景与核心内容 (Research Context & Core Content)
本学期致力于基于**非结构化文本数据 (TXT)** 的 APT (Advanced Persistent Threat, 高级持续性威胁) 组织归因技术研究。我们旨在突破传统基于 IOC (Indicators of Compromise, 失陷指标) 简单匹配的局限性，利用**图神经网络 (GNN, Graph Neural Network)** 自动挖掘 CTI (Cyber Threat Intelligence, 网络威胁情报) 报告中的深层语义与隐式结构化特征。

*   **核心任务**: 构建端到端的 GNN 归因模型。
    *   **输入**: 原始 CTI 文本报告 (TXT格式)。
    *   **处理**: 实体抽取与图构建。
    *   **输出**: 预测该报告描述的攻击活动所属的 APT 组织 (如 APT28, Lazarus, FIN7 等)。
*   **数据集**: `d:\git\APT归因\dataset_TXT` (纯文本报告)，包含多源异构的 APT 分析文档，覆盖 12 个主要 APT 组织。
*   **技术架构**:
    1.  **预处理 (Preprocessing)**: 文本清洗 -> 正则表达式/NLP 实体提取 -> 语义向量化 -> 异构图构建。
    2.  **图构建 (Graph Construction)**:
        *   **节点 (Nodes)**: 提取的实体 (IP, Domain, MD5, FilePath, Email, CVE 等) 及关键术语。
        *   **边 (Edges)**:
            *   **共现边 (Co-occurrence)**: 同一段落或滑动窗口内出现的实体相连。
            *   **语义边 (Semantic)**: 基于实体类型规则 (如 Malware -> 使用 -> Vulnerability)。
        *   **特征 (Features)**: 节点特征融合了 **Type One-Hot** (类型编码) + **Text Hashing** (文本哈希) + **Sentence Embedding** (句向量) + **TF-IDF** (词频逆文档频率)。
    3.  **模型 (Model)**: **GraphSAGE** (Sample and Aggregate, 归纳式图学习框架) / **GAT** (Graph Attention Network, 图注意力网络) + **Global Pooling** (全局池化)。

## 2. 现有方法不足与项目现状 (Limitations & Current Status)
*   **传统方法局限**:
    *   **缺乏结构信息**: 传统的 **TF-IDF** (Term Frequency-Inverse Document Frequency, 词频-逆文档频率) + **Random Forest** (随机森林) 方法仅将文档视为词袋 (Bag-of-Words)，忽略了实体间的拓扑关联 (如: 域名A 解析到 IP-B, IP-B 下发了 文件C)。
    *   **语义理解浅薄**: 无法理解 "Dropper" (投放器) 与 "Backdoor" (后门) 在攻击链中的语义联系。
*   **项目早期问题 (已识别并正在解决)**:
    *   **代码冗余与干扰**: 旧有的基于统计机器学习的脚本不适应图神经网络路线，且路径硬编码，维护成本高。
    *   **伪实现风险**: 早期实验缺乏严格的交叉验证，且缺乏对图构建质量的监控，可能存在"垃圾进垃圾出" (Garbage In, Garbage Out) 的风险。

## 3. 采用的创新点 (Innovations)
1.  **图结构化建模与语义融合 (Graph-Semantic Fusion)**:
    *   **双重特征驱动**: 结合结构信息 (Graph Topology) 与 语义信息 (Semantic Vectors)。我们引入了预训练模型 `sentence-transformers/all-MiniLM-L6-v2` (**Sentence-BERT**, 一种基于孪生网络的语义嵌入模型)，将实体文本映射为 **384维** 的高维语义向量，使模型能理解 "phishing" (网络钓鱼) 和 "spear-phishing" (鱼叉式钓鱼) 的语义相似性。
    *   **动态上下文构图**: 实现了基于 **Context Window** (上下文窗口) 的实体共现连边策略，相比简单的段落共现，更能捕捉紧密的实体关系。
2.  **抗类别不平衡的训练策略 (Class Imbalance Mitigation)**:
    *   **Hard Class Weighting (困难样本加权)**: 针对 **DEEPPANDA**, **MENUPASS**, **ROCKETKITTEN** 等易混淆且样本少的组织，在 Loss 计算时给予 **2.0x - 3.0x** 的额外权重惩罚。
    *   **损失函数优化**:
        *   **Focal Loss**: $FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$，通过 $\gamma$ 参数降低简单样本权重，迫使模型关注难分类样本。
        *   **Logit Adjustment**: 在模型输出层根据类别先验概率调整 Logits (对数几率)，从理论上修正长尾分布带来的偏差。
3.  **严格的防伪验证体系 (Anti-Pseudo Implementation Verification)**:
    *   **5折分层交叉验证 (Stratified 5-Fold CV)**: 确保测试集与训练集严格隔离，消除单次划分的随机性。
    *   **全流程监控**: 增加 `graph_stats.json` 和 `per_class_metrics.csv` 输出，监控图规模、节点度分布及逐类指标，确保模型并未学习到伪特征。

## 4. 研究进展与实验数据 (Progress & Results)
### 4.1 实验环境
*   **模型架构**: **GraphSAGE** (2 layers, 128 hidden dim)。
*   **特征维度**: 353维 (Type + Hash + Embedding + Stats)
*   **训练配置**: 60 Epochs, **AdamW** (Adam with Weight Decay, 带权重衰减的Adam优化器), Weighted Random Sampler (加权随机采样器)。

### 4.2 最新结果 (5-Fold CV Average)
*   **平均准确率 (Accuracy)**: **71%** (±3.9%)
    *   *分析*: 证明模型学到了有效特征，但距离 90% 目标仍有较大差距。
*   **Top-3 准确率**: **91.5%**
    *   *分析*: 真实标签常出现在预测概率最高的前 3 个候选列表中。
*   **加权 F1 (Weighted F1)**: **49.9%**

### 4.3 遇到的主要问题 (Critical Issues) - 基于 `per_class_metrics.csv` 分析
1.  **难例混淆 (Confusion)**:
    *   **DEEPPANDA**: 准确率仅 ~33%，常被误判为 **APT28** 或 **Equation Group**。
    *   **FIN7**: 虽然 Recall (召回率) 高，但与 **Carbanak** 存在特征重叠。
    *   **样本极少类**: 如 **APT17**, **WINNTI** (Support < 5)，F1 Score 常为 0.0，模型因样本不足无法有效学习。
2.  **维度灾难风险**: 当前 353维 的特征相对于部分仅有 10-20 个节点的图来说可能过大，导致 **Overfitting** (过拟合)。
3.  **图稀疏性**: 监控数据 `graph_stats.json` 显示，部分简短报告生成的图节点数 < 5，且 `avg_degree` (平均度) 较低，导致 GNN 无法有效进行多跳 **Message Passing** (消息传递)。

## 5. 寒假及后续改进计划 (Improvement Plan)
针对上述问题及 90% 准确率的目标，制定以下 **四大改进方向**：

### 5.1 模型参数调优 (Parameter Tuning)
*   **行动**: 启动大规模 **网格搜索 (Grid Search)**。
*   **核查点**: 确认 `run_gnn_grid.py` 能够正确记录每组实验的 `val_acc`, `val_f1` 以及训练曲线。
*   **参数空间**:
    *   **Feature**: [Hash64, Hash64+Embed, Hash64+Embed+TFIDF]
    *   **Model**: [GraphSAGE, GAT, GIN]
    *   **Loss**: [CrossEntropy, FocalLoss, LabelSmoothing]

### 5.2 移除不必要及错误的技术手段 (Removing Redundant/Error-Prone Techniques)
*   **行动**: **彻底移除**旧有的、不适用于图神经网络的技术堆栈，防止其作为"僵尸代码"或错误的技术路径干扰模型性能评估与开发。

### 5.3 数据增强与特征加权 (Data Augmentation & Weighting)
*   **行动**: 针对数据分布问题进行干预。
*   **细节**:
    *   **特征加权**: 对关键实体（如 C2 域名、特有互斥量）的节点特征进行加权。
    *   **图数据增强**: 尝试 **DropEdge** (随机删边) 防止过拟合，以及 **SMOTE** (Synthetic Minority Over-sampling Technique, 合成少数类过采样技术) for Graphs 在嵌入空间对少数类样本进行过采样。

### 5.4 全链路输出核查与调试 (Pipeline Debugging)
*   **行动**: 深度审计项目各环节输出，确保无"伪实现"。
*   **具体核查内容**:
    1.  **预处理核查 (`entities.jsonl`)**:
        *   *检查项*: 随机抽取 5 个报告，人工核对提取的实体是否准确，是否存在大量无意义的数字或乱码被误识别为实体。
    2.  **图结构核查 (`graph_stats.json`)**:
        *   *检查项*: 检查 `num_nodes` 分布。若大量图 `num_nodes < 5`，需调整预处理逻辑（如增大上下文窗口、放宽正则过滤）。
        *   *检查项*: 检查 `type_counts`，确认是否某些关键类型（如 `MALWARE`）缺失。
    3.  **训练过程核查 (`errors.csv` & `confusion_matrix.csv`)**:
        *   *检查项*: 分析 `errors.csv` 中的具体误判案例，回溯其文本报告，人工判断是否由于报告内容本身模糊导致。
        *   *检查项*: 检查 `confusion_matrix.csv` 对角线元素，确认哪些类别是"绝对无法识别"的。
    4.  **中间层可视化 (t-SNE)**:
        *   *检查项*: 输出 GNN 最后一层 Embedding 的 **t-SNE** (t-Distributed Stochastic Neighbor Embedding, t-分布随机邻域嵌入) 降维图，观察不同 APT 组织的样本在特征空间是否通过 GNN 聚合后变得线性可分。
