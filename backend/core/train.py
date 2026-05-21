"""
基于图神经网络的APT归因训练脚本
支持GCN、GAT、Transformer等多种模型架构
"""

from __future__ import annotations

import json
import logging
import sys
import io
from pathlib import Path
from typing import Dict, List, Optional, Any
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.nn import Linear
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from torch.utils.data import WeightedRandomSampler
import os
import datetime
from torch_geometric.nn import GATConv, GCNConv, TransformerConv, SAGEConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import dropout_adj

# Import new model
try:
    from backend.core.models.rgat import RelationAwareGAT
except ImportError:
    # Fallback if running from backend directory
    try:
        from core.models.rgat import RelationAwareGAT
    except ImportError:
        try:
            from models.rgat import RelationAwareGAT
        except ImportError:
            # Only warn if not found
            RelationAwareGAT = None

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    # Only wrap if not already utf-8
    if getattr(sys.stdout, 'encoding', '') != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger(__name__)

if RelationAwareGAT is None:
    LOGGER.warning("Could not import RelationAwareGAT. RGAT model will not be available.")


# ============================================================================
# 模型定义
# ============================================================================


class APTAttributionGCN(torch.nn.Module):
    """基于GCN的APT归因模型"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


class APTAttributionGAT(torch.nn.Module):
    """基于GAT的APT归因模型（推荐）"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))
        self.norms.append(torch.nn.BatchNorm1d(hidden_dim * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim * heads))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False))
        
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.elu(x)
                x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


class APTAttributionTransformer(torch.nn.Module):
    """基于Transformer的APT归因模型"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        self.convs.append(TransformerConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        self.norms.append(torch.nn.BatchNorm1d(hidden_dim * heads))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim * heads))
        if num_layers > 1:
            self.convs.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
        
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


class APTAttributionHybrid(torch.nn.Module):
    """混合池化策略的模型"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        model_type: str = "GAT",
        heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "GCN":
            self.convs = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        elif model_type == "GAT":
            self.convs = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
                self.norms.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            if num_layers > 1:
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False))
        else:  # Transformer
            self.convs = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.convs.append(TransformerConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim * heads))
            for _ in range(num_layers - 1):
                self.convs.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
                self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        self.dropout = torch.nn.Dropout(dropout)
        self.attn_pool = AttentionalAggregation(Linear(hidden_dim, 1))
        pool_dim = hidden_dim * 3
        self.classifier = Linear(pool_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                if self.model_type == "GAT":
                    x = F.elu(x)
                else:
                    x = F.relu(x)
                x = self.norms[i](x)
                x = self.dropout(x)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_attn = self.attn_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_attn], dim=1)
        x = self.classifier(x)
        return x


class APTAttributionGraphSAGE(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.5, text_emb_dim: int = 0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.dropout = torch.nn.Dropout(dropout)
        
        clf_input_dim = hidden_dim
        if text_emb_dim > 0:
            clf_input_dim += text_emb_dim
            
        self.classifier = Linear(clf_input_dim, num_classes)
        self.text_emb_dim = text_emb_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, doc_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        x = global_mean_pool(x, batch)
        
        if self.text_emb_dim > 0 and doc_emb is not None:
             if doc_emb.size(0) == x.size(0):
                 x = torch.cat([x, doc_emb], dim=1)
             
        x = self.classifier(x)
        return x


class APTAttributionGIN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        nn1 = torch.nn.Sequential(Linear(input_dim, hidden_dim), torch.nn.ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(nn1))
        self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            nnk = torch.nn.Sequential(Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(nnk))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(inputs, dim=1)
        p = logp.exp()
        idx = torch.arange(inputs.size(0), device=inputs.device)
        logp_t = logp[idx, targets]
        p_t = p[idx, targets]
        loss = -(1 - p_t) ** self.gamma * logp_t
        if self.weight is not None:
            alpha_t = self.weight[targets]
            loss = loss * alpha_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================================
# 训练与评估函数
# ============================================================================

def load_graph_dataset(dataset_path: Path, label_mapping_path: Optional[Path] = None) -> tuple[List[Data], Dict[str, int], Dict[int, str]]:
    """加载图数据集 (支持目录或单文件)"""
    LOGGER.info("加载图数据: %s", dataset_path)
    graphs: List[Data] = []
    
    try:
        if dataset_path.is_dir():
            # Check for 'graphs' subdirectory first (new format)
            graphs_dir = dataset_path / "graphs"
            if graphs_dir.exists() and graphs_dir.is_dir():
                for pt_file in graphs_dir.glob("*.pt"):
                    try:
                        g = torch.load(pt_file, map_location="cpu", weights_only=False)
                        graphs.append(g)
                    except Exception as e:
                        LOGGER.warning(f"Failed to load graph {pt_file}: {e}")
            elif (dataset_path / "graphs.pt").exists():
                 # Old format in directory
                 graphs = torch.load(dataset_path / "graphs.pt", map_location="cpu", weights_only=False)
            else:
                 # Maybe the directory IS the graphs directory? Unlikely based on structure, but possible.
                 # Or maybe user passed the root dir and we need to look for graphs.pt
                 pass
        elif dataset_path.is_file():
            graphs = torch.load(dataset_path, map_location="cpu", weights_only=False)
        
        if not graphs:
             # Try to glob if it's a directory but no 'graphs' subdir
             if dataset_path.is_dir():
                 for pt_file in dataset_path.glob("*.pt"):
                     if pt_file.name == "best_model.pt": continue
                     try:
                        g = torch.load(pt_file, map_location="cpu", weights_only=False)
                        if isinstance(g, Data) or (isinstance(g, list) and len(g)>0 and isinstance(g[0], Data)):
                             if isinstance(g, list): graphs.extend(g)
                             else: graphs.append(g)
                     except: pass

    except Exception as e:
        raise RuntimeError(f"无法加载图数据: {e}")
    
    if not graphs:
        raise RuntimeError(f"未找到有效的图数据在: {dataset_path}")

    label_to_idx = {}
    if label_mapping_path and label_mapping_path.exists():
        LOGGER.info("加载标签映射: %s", label_mapping_path)
        with label_mapping_path.open("r", encoding="utf-8") as f:
            label_to_idx = json.load(f)
    else:
        # 如果没有映射文件，尝试从图中恢复（不推荐）
        labels = set()
        for g in graphs:
            if hasattr(g, 'apt_group'):
                labels.add(g.apt_group)
        label_to_idx = {l: i for i, l in enumerate(sorted(labels))}

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    valid_graphs = []
    for graph in graphs:
        if graph.x is not None and graph.edge_index is not None and graph.y is not None:
            if graph.x.size(0) > 0:
                valid_graphs.append(graph)
    
    LOGGER.info("有效图数量: %d / %d", len(valid_graphs), len(graphs))
    return valid_graphs, label_to_idx, idx_to_label


def split_dataset_stratified(graphs: List[Data], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    labels = np.array([int(g.y.item()) for g in graphs])
    indices = np.arange(len(graphs))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss.split(indices, labels))
    labels_train_val = labels[train_val_idx]
    
    # Check if we have enough samples for split
    if len(train_val_idx) < 2:
         # Fallback to simple split
         train_graphs = [graphs[i] for i in train_val_idx]
         val_graphs = []
         test_graphs = [graphs[i] for i in test_idx]
         return train_graphs, val_graphs, test_graphs

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio / (train_ratio + val_ratio), random_state=seed)
    train_idx, val_idx = next(sss2.split(train_val_idx, labels_train_val))
    
    train_graphs = [graphs[i] for i in train_val_idx[train_idx]]
    val_graphs = [graphs[i] for i in train_val_idx[val_idx]]
    test_graphs = [graphs[i] for i in test_idx]
    LOGGER.info("数据集划分(分层): 训练集=%d, 验证集=%d, 测试集=%d", len(train_graphs), len(val_graphs), len(test_graphs))
    return train_graphs, val_graphs, test_graphs


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None, clip_norm: float = 0.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        doc_emb = None
        if hasattr(batch, 'doc_emb'):
            doc_emb = batch.doc_emb

        # Use torch.amp.autocast for newer PyTorch versions to avoid warnings
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            # Only enable if on CUDA for now (unless using bfloat16 on CPU which requires more setup)
            amp_ctx = torch.amp.autocast(device_type=device_type, enabled=(device.type == 'cuda'))
        else:
            amp_ctx = autocast(enabled=(device.type == 'cuda'))

        with amp_ctx:
            if isinstance(model, APTAttributionGraphSAGE):
                out = model(batch.x, batch.edge_index, batch.batch, doc_emb=doc_emb)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(out, batch.y)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            doc_emb = None
            if hasattr(batch, 'doc_emb'):
                doc_emb = batch.doc_emb
            
            if isinstance(model, APTAttributionGraphSAGE):
                out = model(batch.x, batch.edge_index, batch.batch, doc_emb=doc_emb)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(out, batch.y)
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, average="weighted") if all_labels else 0.0
    return avg_loss, accuracy, f1


def plot_training_history(history: Dict[str, List[float]], output_path: Path):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels: List[str], output_path: Path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_training_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the model training and evaluation process based on a config dict.
    """
    base_output_dir = Path(config.get('base_output_dir', 'results_archive/training_runs'))
    processed_data_path = Path(config.get('processed_data_path'))
    
    # Try to find label mapping
    # Priority 1: Inside the dataset directory (Standard)
    label_mapping_path = processed_data_path / "label_mapping.json"
    
    # Priority 2: Parent directory (Legacy or shared)
    if not label_mapping_path.exists():
        label_mapping_path = processed_data_path.parent / "label_mapping.json"
    
    if not label_mapping_path.exists():
        LOGGER.warning(f"Label mapping file not found at {label_mapping_path}. This may cause label mismatch errors.")
        label_mapping_path = None
    else:
        LOGGER.info(f"Using label mapping from: {label_mapping_path}")

    model_type = config.get('model_type', 'GAT')
    epochs = int(config.get('epochs', 100))
    lr = float(config.get('lr', 0.001))
    batch_size = int(config.get('batch_size', 32))
    seed = int(config.get('seed', 42))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_output_dir = base_output_dir / f"{model_type}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        LOGGER.info("CUDA is available. Using GPU for training.")
        LOGGER.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        LOGGER.warning("CUDA is not available. Using CPU for training. This may be slow.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    LOGGER.info(f"Starting training run. Output: {run_output_dir}")
    
    graphs, label_to_idx, idx_to_label = load_graph_dataset(processed_data_path, label_mapping_path)
    if not graphs:
        raise ValueError("No valid graphs loaded.")
    
    # Validation: Ensure num_classes covers all labels in the dataset
    max_y = 0
    for g in graphs:
        if g.y is not None:
            max_y = max(max_y, int(g.y.item()))
            
    num_classes = len(label_to_idx)
    LOGGER.info(f"Detected {num_classes} classes from mapping. Max label index in data: {max_y}")
    
    if max_y >= num_classes:
        LOGGER.error(f"Data contains label index {max_y} which is >= num_classes {num_classes}. Adjusting num_classes to {max_y + 1} to prevent crash.")
        num_classes = max_y + 1
        
    input_dim = graphs[0].x.size(1)
    
    text_emb_dim = 0
    if hasattr(graphs[0], 'doc_emb') and graphs[0].doc_emb is not None:
        text_emb_dim = graphs[0].doc_emb.size(1)

    train_graphs, val_graphs, test_graphs = split_dataset_stratified(
        graphs, 0.7, 0.15, 0.15, seed=seed
    )
    
    # Dataloaders with optimizations
    # Increase num_workers for faster data loading if on Windows/Linux (but be careful on Windows with spawn)
    # pin_memory=True for faster transfer to GPU
    num_workers = 4 if os.name != 'nt' else 0 # 0 on Windows to avoid pickling issues, or tune carefully
    
    # Only use pin_memory if CUDA is available to avoid warnings on CPU
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    # Model
    if model_type == "GCN":
        model = APTAttributionGCN(input_dim, 128, num_classes)
    elif model_type == "GAT":
        model = APTAttributionGAT(input_dim, 128, num_classes)
    elif model_type == "Transformer":
        model = APTAttributionTransformer(input_dim, 128, num_classes)
    elif model_type == "GraphSAGE":
        model = APTAttributionGraphSAGE(input_dim, 128, num_classes, text_emb_dim=text_emb_dim)
    elif model_type == "GIN":
        model = APTAttributionGIN(input_dim, 128, num_classes)
    elif model_type == "RGAT":
        if RelationAwareGAT is None:
            raise ValueError("RGAT model is not available (ImportError).")
        # RGAT specific config
        ablation_mode = config.get('ablation_mode', "dual")
        model = RelationAwareGAT(
            num_node_features=input_dim,
            num_classes=num_classes,
            hidden_dim=128,
            num_heads=4,
            dropout=0.5,
            num_entity_types=40, # Default safe upper bound
            ablation_mode=ablation_mode,
            use_gatv2=True 
        )
    else:
        model = APTAttributionHybrid(input_dim, 128, num_classes)
    
    model = model.to(device)
    
    # Compile model if supported (PyTorch 2.0+)
    if hasattr(torch, 'compile') and os.name != 'nt': # Windows support for torch.compile is still experimental/limited
         try:
             model = torch.compile(model)
             LOGGER.info("Model compiled with torch.compile() for faster training.")
         except Exception as e:
             LOGGER.warning(f"Failed to compile model: {e}")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    
    # Initialize GradScaler for AMP
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_loss = float('inf')
    best_model_path = run_output_dir / "best_model.pt"
    
    # Early Stopping Config
    early_stop_patience = 25
    no_improve_epochs = 0
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        scheduler.step(val_loss)
        
        # Save best model logic: Priority 1 = Accuracy (Max), Priority 2 = Loss (Min)
        is_best = False
        if val_acc > best_val_acc:
            is_best = True
        elif val_acc == best_val_acc:
            if val_loss < best_model_loss:
                is_best = True
        
        if is_best:
            best_val_acc = val_acc
            best_model_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            LOGGER.info(f"Epoch {epoch+1}: New best model saved (Acc={val_acc:.4f}, Loss={val_loss:.4f})")

        # Early Stopping based on Loss (Prevent Overfitting)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        LOGGER.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f} | Patience={no_improve_epochs}/{early_stop_patience}")
        
        if no_improve_epochs >= early_stop_patience:
            LOGGER.info(f"Early stopping triggered at epoch {epoch+1} (Val Loss did not improve for {early_stop_patience} epochs)")
            break

    # Plot History
    history_plot_path = run_output_dir / "training_history.png"
    try:
        plot_training_history(history, history_plot_path)
    except Exception as e:
        LOGGER.warning(f"Failed to plot history: {e}")

    # Final Eval
    model.load_state_dict(torch.load(best_model_path))
    
    # Custom validate for test to get full preds
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            doc_emb = None
            if hasattr(batch, 'doc_emb'):
                doc_emb = batch.doc_emb
            
            if isinstance(model, APTAttributionGraphSAGE):
                out = model(batch.x, batch.edge_index, batch.batch, doc_emb=doc_emb)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    # Classification Report
    all_class_ids = list(range(len(idx_to_label)))
    target_names = [idx_to_label[i] for i in all_class_ids]
    
    cls_report = classification_report(
        all_labels, 
        all_preds, 
        labels=all_class_ids,
        target_names=target_names, 
        output_dict=True,
        zero_division=0
    )
    
    # Confusion Matrix Plot
    cm_plot_path = run_output_dir / "confusion_matrix.png"
    try:
        plot_confusion_matrix(all_labels, all_preds, target_names, cm_plot_path)
    except Exception as e:
        LOGGER.warning(f"Failed to plot confusion matrix: {e}")

    results = {
        "test_accuracy": test_acc,
        "test_f1_weighted": test_f1,
        "config": config,
        "model_path": str(best_model_path),
        "history_plot": str(history_plot_path) if history_plot_path.exists() else None,
        "confusion_matrix_plot": str(cm_plot_path) if cm_plot_path.exists() else None,
        "classification_report": cls_report,
        "history": history
    }
    
    with (run_output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results
