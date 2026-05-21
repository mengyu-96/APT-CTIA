# APT归因系统实验指南

本指南旨在帮助研究人员使用本项目代码库进行系统性的实验，以验证不同模型架构、特征融合策略及数据处理方法对APT归因任务的影响。

## 0. 环境与数据准备

在开始实验前，请确保已正确安装环境依赖并生成了基础图数据。

```bash
# 安装依赖
pip install -r requirements.txt

# 生成完整数据集（包含TXT和PDF）
python preprocess_apt_dataset.py --output-dir processed_data_mixed_v1
```

生成的图数据位于 `processed_data_mixed_v1/graphs.pt`。

---

## 实验一：基准模型架构对比 (Graph vs Non-Graph)

### 1.0 传统机器学习基线 (XGBoost/SVM/RF)
验证图结构是否带来了增量收益。

```bash
python train_baseline_models.py --data-dir results_archive/diagnose_output --output-dir results_archive/baselines
```

### 1.1 GCN (Graph Convolutional Network)
最基础的图卷积网络。

```bash
python train_gnn.py ^
    --graphs-path results_archive/diagnose_output/graphs.pt ^
    --label-mapping-path results_archive/diagnose_output/label_mapping.json ^
    --output-dir results_archive/exp1_gcn ^
    --model-type GCN ^
    --hidden-dim 128 --num-layers 3 --dropout 0.5 --epochs 100
```

### 1.2 GAT (Graph Attention Network) - **推荐**
引入注意力机制，允许模型关注更重要的邻居节点。

```bash
python train_gnn.py ^
    --graphs-path results_archive/diagnose_output/graphs.pt ^
    --label-mapping-path results_archive/diagnose_output/label_mapping.json ^
    --output-dir results_archive/exp1_gat ^
    --model-type GAT ^
    --hidden-dim 128 --num-layers 2 --heads 8 --dropout 0.6 --epochs 100
```

### 1.3 GraphSAGE (Sample and Aggregate)
适合大规模图的归纳式学习框架。

```bash
python train_gnn.py ^
    --graphs-path results_archive/diagnose_output/graphs.pt ^
    --label-mapping-path results_archive/diagnose_output/label_mapping.json ^
    --output-dir results_archive/exp1_sage ^
    --model-type GraphSAGE ^
    --hidden-dim 128 --num-layers 2 --dropout 0.5 --epochs 100
```

---

## 实验二：多模态文本融合的影响

**目的**：验证将文档级的文本语义特征（如TF-IDF或BERT Embedding）与图结构特征融合是否能提升性能。

**注意**：此实验主要适用于 **GraphSAGE** 模型，因为代码中针对该模型实现了显式的特征拼接逻辑。

### 2.1 启用文本融合 (默认)
如果预处理阶段生成了 `doc_emb`（默认行为），GraphSAGE 会自动使用它。

```bash
python train_gnn.py ^
    --graphs-path results_archive/diagnose_output/graphs.pt ^
    --label-mapping-path results_archive/diagnose_output/label_mapping.json ^
    --output-dir results_archive/exp2_sage_fusion ^
    --model-type GraphSAGE ^
    --epochs 100
```

### 2.2 禁用文本融合 (Ablation Study)
使用 `--no-text-fusion` 参数强制忽略文档Embedding，仅使用图结构特征。

```bash
python train_gnn.py ^
    --graphs-path results_archive/diagnose_output/graphs.pt ^
    --label-mapping-path results_archive/diagnose_output/label_mapping.json ^
    --output-dir results_archive/exp2_sage_no_fusion ^
    --model-type GraphSAGE ^
    --no-text-fusion ^
    --epochs 100
```

---

## 实验三：类别不平衡处理策略

**目的**：APT数据集通常存在严重的类别不平衡（头部组织样本多，尾部少），探究不同策略的有效性。

### 3.1 基准 (无特殊处理)
使用标准交叉熵损失。

```bash
python train_gnn.py ^
    --graphs-path processed_data_mixed_v1/graphs.pt ^
    --label-mapping-path processed_data_mixed_v1/label_mapping.json ^
    --output-dir models/exp3_baseline ^
    --model-type GAT
```

### 3.2 类别加权 (Class Weights)
根据类别样本数量的倒数对Loss进行加权。

```bash
python train_gnn.py ^
    --graphs-path processed_data_mixed_v1/graphs.pt ^
    --label-mapping-path processed_data_mixed_v1/label_mapping.json ^
    --output-dir models/exp3_weighted ^
    --model-type GAT ^
    --class-weights
```

### 3.3 Focal Loss
专注于难分类样本的损失函数。

```bash
python train_gnn.py ^
    --graphs-path processed_data_mixed_v1/graphs.pt ^
    --label-mapping-path processed_data_mixed_v1/label_mapping.json ^
    --output-dir models/exp3_focal ^
    --model-type GAT ^
    --focal --focal-gamma 2.0
```

### 3.4 过采样 (Oversampling)
使用 `WeightedRandomSampler` 对少数类进行过采样。

```bash
python train_gnn.py ^
    --graphs-path processed_data_mixed_v1/graphs.pt ^
    --label-mapping-path processed_data_mixed_v1/label_mapping.json ^
    --output-dir models/exp3_oversample ^
    --model-type GAT ^
    --oversample
```

---

## 实验四：数据源影响 (TXT vs Mixed PDF)

**目的**：验证引入PDF格式的报告是否能增加有效样本量并提升模型泛化能力。

### 4.1 仅生成 TXT 数据集
重新运行预处理，使用 `--only-txt` 参数。

```bash
python preprocess_apt_dataset.py --output-dir processed_data_txt_only --only-txt
```

### 4.2 训练并对比
使用 TXT 数据集训练模型，并与使用 Mixed 数据集（实验一的结果）进行对比。

```bash
python train_gnn.py ^
    --graphs-path processed_data_txt_only/graphs.pt ^
    --label-mapping-path processed_data_txt_only/label_mapping.json ^
    --output-dir models/exp4_txt_only ^
    --model-type GAT
```

---

## 结果分析

所有实验结果均会自动保存到对应的 `output-dir` 中，包含：
- `test_results.json`: 最终测试集指标 (Accuracy, F1, etc.)
- `classification_report.txt`: 每个类别的详细指标。
- `confusion_matrix.csv`: 混淆矩阵。
- `errors.csv`: 具体的误判样本列表。

可以使用 `analyze_errors.py` 加载特定模型进行深入分析：
*(注意：需要修改 `analyze_errors.py` 中的 `model_path`、`graphs_path` 以及**模型类定义**（如将 `APTAttributionGAT` 改为 `APTAttributionGraphSAGE`）以匹配训练时的设置)*
