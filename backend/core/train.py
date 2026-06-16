"""
基于图神经网络的APT归因训练脚本
支持GCN、GAT、Transformer等多种模型架构
"""

from __future__ import annotations

import json
import logging
import sys
import io
import copy
import hashlib
import re
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Any
import random
import numpy as np
import torch
import torch.nn.functional as F
import os
import datetime
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPL_CACHE_DIR = _REPO_ROOT / ".cache" / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from torch.nn import Linear
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from torch_geometric.nn import GATConv, GCNConv, TransformerConv, SAGEConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import dropout_adj

# Timing & resource utils
try:
    from utils.timing import TimeLogger, CudaTimer, count_params, count_flops_safe, peak_vram_mb, peak_ram_mb, reset_vram_peak
    from utils.splits import build_report_level_split
except ImportError:
    # Allow running from backend/ subdir
    _here = Path(__file__).resolve().parents[2]
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from utils.timing import TimeLogger, CudaTimer, count_params, count_flops_safe, peak_vram_mb, peak_ram_mb, reset_vram_peak
    from utils.splits import build_report_level_split

# Import new model
try:
    from backend.core.models.rgat import RelationAwareGAT
    from backend.core.models.hgt import HGTClassifier
except ImportError:
    # Fallback if running from backend directory
    try:
        from core.models.rgat import RelationAwareGAT
        from core.models.hgt import HGTClassifier
    except ImportError:
        try:
            from models.rgat import RelationAwareGAT
            from models.hgt import HGTClassifier
        except ImportError:
            # Only warn if not found
            RelationAwareGAT = None
            HGTClassifier = None

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
if HGTClassifier is None:
    LOGGER.warning("Could not import HGTClassifier. HGT model will not be available.")


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

def _coerce_batch_metadata_value(value: Any) -> Any:
    if value is None:
        return 0
    if isinstance(value, (int, float, bool)):
        return int(value) if isinstance(value, bool) else value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _normalize_graph_dict_metadata_for_batch(graphs: List[Data]) -> None:
    """Make custom dict metadata collatable without changing model tensors."""
    dict_keys_by_attr: Dict[str, set[str]] = defaultdict(set)
    for graph in graphs:
        for attr_name in graph.keys():
            value = getattr(graph, attr_name, None)
            if isinstance(value, dict):
                dict_keys_by_attr[str(attr_name)].update(str(key) for key in value.keys())

    for attr_name, keys in dict_keys_by_attr.items():
        ordered_keys = sorted(keys)
        for graph in graphs:
            value = getattr(graph, attr_name, None)
            value = value if isinstance(value, dict) else {}
            normalized = {
                key: _coerce_batch_metadata_value(value.get(key, value.get(str(key), 0)))
                for key in ordered_keys
            }
            setattr(graph, attr_name, normalized)


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

    _normalize_graph_dict_metadata_for_batch(graphs)

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


def _clone_graph(graph: Data) -> Data:
    return copy.deepcopy(graph)


def _find_report_node_index(graph: Data) -> Optional[int]:
    labels = getattr(graph, 'node_labels', None)
    paragraph_indices = getattr(graph, 'node_paragraph_indices', None)
    if isinstance(labels, list) and labels:
        if str(labels[0]).upper() == 'REPORT':
            return 0
    if isinstance(paragraph_indices, list) and paragraph_indices:
        if int(paragraph_indices[0]) < 0:
            return 0
    return None


def _remove_node(graph: Data, node_idx: int) -> Data:
    g = _clone_graph(graph)
    num_nodes = int(g.x.size(0))
    if node_idx < 0 or node_idx >= num_nodes:
        return g

    keep_mask = torch.ones(num_nodes, dtype=torch.bool)
    keep_mask[node_idx] = False
    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
    old_to_new[keep_mask] = torch.arange(int(keep_mask.sum()))

    g.x = g.x[keep_mask]
    edge_mask = (g.edge_index[0] != node_idx) & (g.edge_index[1] != node_idx)
    filtered_edge_index = g.edge_index[:, edge_mask]
    g.edge_index = old_to_new[filtered_edge_index]

    for attr_name in ('node_texts', 'node_labels', 'node_paragraph_indices'):
        attr = getattr(g, attr_name, None)
        if isinstance(attr, list) and len(attr) == num_nodes:
            setattr(g, attr_name, [attr[i] for i in range(num_nodes) if i != node_idx])
    return g


def _make_edge_index_from_paragraphs(
    graph: Data,
    mode: str,
    keep_root_edges: bool,
    paragraph_window: int = 1,
) -> torch.Tensor:
    num_nodes = int(graph.x.size(0))
    root_idx = _find_report_node_index(graph)
    paragraph_indices = getattr(graph, 'node_paragraph_indices', None)
    if not isinstance(paragraph_indices, list) or len(paragraph_indices) != num_nodes:
        return graph.edge_index.clone()

    node_to_para = {idx: int(paragraph_indices[idx]) for idx in range(num_nodes)}
    edge_set = set()

    if mode == 'paragraph':
        para_to_nodes: Dict[int, List[int]] = {}
        for idx in range(num_nodes):
            if idx == root_idx:
                continue
            para_idx = node_to_para.get(idx, -1)
            if para_idx < 0:
                continue
            para_to_nodes.setdefault(para_idx, []).append(idx)
        for node_ids in para_to_nodes.values():
            unique_ids = sorted(set(node_ids))
            if len(unique_ids) == 1:
                edge_set.add((unique_ids[0], unique_ids[0]))
            else:
                for i in unique_ids:
                    for j in unique_ids:
                        if i != j:
                            edge_set.add((i, j))
    elif mode == 'sliding_window':
        valid_nodes = [idx for idx in range(num_nodes) if idx != root_idx and node_to_para.get(idx, -1) >= 0]
        for i in valid_nodes:
            for j in valid_nodes:
                if i == j:
                    continue
                if abs(node_to_para[i] - node_to_para[j]) <= paragraph_window:
                    edge_set.add((i, j))

    if keep_root_edges and root_idx is not None:
        for idx in range(num_nodes):
            if idx == root_idx:
                continue
            edge_set.add((root_idx, idx))
            edge_set.add((idx, root_idx))

    if not edge_set:
        fallback_idx = 1 if root_idx == 0 and num_nodes > 1 else 0
        edge_set.add((fallback_idx, fallback_idx))
        if keep_root_edges and root_idx is not None and fallback_idx != root_idx:
            edge_set.add((root_idx, fallback_idx))
            edge_set.add((fallback_idx, root_idx))

    edges = sorted(edge_set)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


_TYPE_VECTOR_DIM = 30
_HASH_DIM = 64
_STATS_DIM = 3
_WINDOW_TOKENS = 50

_HASH_LABELS = {"HASH", "HASH_MD5", "HASH_SHA1", "HASH_SHA256"}
_INFRA_LABELS = {"IP", "DOMAIN", "URL", "HOSTNAME", "PORT"}
_PROVENANCE_LABELS = {
    "PROCESS",
    "SERVICE",
    "FILE_PATH",
    "REGISTRY",
    "HASH",
    "HASH_MD5",
    "HASH_SHA1",
    "HASH_SHA256",
    "IP",
    "DOMAIN",
    "URL",
    "HOSTNAME",
    "PORT",
}
_SEMANTIC_RELATION_RULES: tuple[tuple[set[str], set[str]], ...] = (
    ({"MALWARE", "TOOL"}, {"MITRE_TECH"}),
    ({"OPERATION", "CAMPAIGN"}, {"CVE"}),
    (_HASH_LABELS, {"IP", "DOMAIN", "URL"}),
    ({"MALWARE"}, _HASH_LABELS),
    ({"CVE"}, _HASH_LABELS),
    ({"DOMAIN", "HOSTNAME"}, {"IP"}),
    (_HASH_LABELS, {"REGISTRY", "FILE_PATH"}),
    ({"OPERATION", "CAMPAIGN"}, {"ORG", "INDUSTRY", "COUNTRY"}),
)


@lru_cache(maxsize=16)
def _load_variant_paragraphs(paragraphs_path: str) -> Dict[str, Dict[int, str]]:
    grouped: Dict[str, Dict[int, str]] = defaultdict(dict)
    path = Path(paragraphs_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            report_id = str(record.get("report_id", "")).strip()
            if not report_id:
                continue
            grouped[report_id][int(record.get("paragraph_index", 0))] = str(record.get("text", ""))
    return dict(grouped)


@lru_cache(maxsize=16)
def _load_variant_entities(entities_path: str) -> Dict[str, Dict[int, List[dict]]]:
    grouped: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    path = Path(entities_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            report_id = str(record.get("report_id", "")).strip()
            if not report_id:
                continue
            grouped[report_id][int(record.get("paragraph_index", 0))].append(record)
    return {report_id: dict(paragraphs) for report_id, paragraphs in grouped.items()}


def _load_variant_artifacts(processed_data_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if processed_data_path is None:
        return None
    paragraphs_path = processed_data_path / "paragraphs.jsonl"
    entities_path = processed_data_path / "entities.jsonl"
    if not paragraphs_path.exists() or not entities_path.exists():
        return None
    return {
        "paragraphs": _load_variant_paragraphs(str(paragraphs_path.resolve())),
        "entities": _load_variant_entities(str(entities_path.resolve())),
    }


def _require_variant_artifacts(
    variant: str,
    processed_data_path: Optional[Path],
    strict_repro: bool,
) -> Optional[Dict[str, Any]]:
    artifacts = _load_variant_artifacts(processed_data_path)
    if artifacts is None and strict_repro:
        raise FileNotFoundError(
            f"Variant '{variant}' requires paragraphs.jsonl and entities.jsonl under {processed_data_path} in strict reproduction mode."
        )
    return artifacts


def _token_spans(text: str) -> List[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text or "")]


def _entity_token_position(token_spans: List[tuple[int, int]], start: int, end: int) -> Optional[float]:
    overlapping: List[int] = []
    for token_idx, (token_start, token_end) in enumerate(token_spans):
        if token_end <= start:
            continue
        if token_start >= end:
            break
        overlapping.append(token_idx)
    if not overlapping:
        for token_idx, (token_start, token_end) in enumerate(token_spans):
            if token_start <= start < token_end or token_start < end <= token_end:
                overlapping.append(token_idx)
                break
    if not overlapping:
        return None
    return float(sum(overlapping) / len(overlapping))


def _build_graph_edges_from_artifacts(
    graph: Data,
    artifacts: Optional[Dict[str, Any]],
    mode: str,
    keep_root_edges: bool,
    window_tokens: int = _WINDOW_TOKENS,
) -> Optional[torch.Tensor]:
    if artifacts is None:
        return None
    report_id = str(getattr(graph, "report_id", "")).strip()
    if not report_id:
        return None

    report_entities = artifacts["entities"].get(report_id, {})
    report_paragraphs = artifacts["paragraphs"].get(report_id, {})
    node_texts = list(getattr(graph, "node_texts", []) or [])
    node_labels = list(getattr(graph, "node_labels", []) or [])
    num_nodes = int(graph.x.size(0))
    if len(node_texts) != num_nodes or len(node_labels) != num_nodes:
        return None

    root_idx = _find_report_node_index(graph)
    key_to_node_idx: Dict[tuple[str, str], int] = {}
    for node_idx, (text, label) in enumerate(zip(node_texts, node_labels)):
        if node_idx == root_idx:
            continue
        key_to_node_idx[(str(text).strip().lower(), str(label).strip().upper())] = node_idx

    edge_set: set[tuple[int, int]] = set()
    if mode == "paragraph":
        for para_idx, paragraph_entities in report_entities.items():
            node_ids = []
            for record in paragraph_entities:
                key = (str(record.get("text", "")).strip().lower(), str(record.get("label", "")).strip().upper())
                node_idx = key_to_node_idx.get(key)
                if node_idx is not None:
                    node_ids.append(node_idx)
            unique_ids = sorted(set(node_ids))
            if len(unique_ids) == 1:
                edge_set.add((unique_ids[0], unique_ids[0]))
            else:
                for src, dst in combinations(unique_ids, 2):
                    edge_set.add((src, dst))
                    edge_set.add((dst, src))
    elif mode == "sliding_window":
        occurrences: List[tuple[int, float]] = []
        token_offset = 0
        for para_idx in sorted(report_paragraphs):
            paragraph_text = report_paragraphs.get(para_idx, "")
            token_spans = _token_spans(paragraph_text)
            paragraph_entities = report_entities.get(para_idx, [])
            for record in paragraph_entities:
                key = (str(record.get("text", "")).strip().lower(), str(record.get("label", "")).strip().upper())
                node_idx = key_to_node_idx.get(key)
                if node_idx is None:
                    continue
                start = record.get("start")
                end = record.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                token_pos = _entity_token_position(token_spans, start, end)
                if token_pos is None:
                    continue
                occurrences.append((node_idx, token_offset + token_pos))
            token_offset += len(token_spans)

        for i, (src_node, src_pos) in enumerate(occurrences):
            for dst_node, dst_pos in occurrences[i + 1:]:
                if abs(dst_pos - src_pos) >= window_tokens:
                    continue
                if src_node == dst_node:
                    edge_set.add((src_node, src_node))
                    continue
                edge_set.add((src_node, dst_node))
                edge_set.add((dst_node, src_node))

    if keep_root_edges and root_idx is not None:
        for node_idx in range(num_nodes):
            if node_idx == root_idx:
                continue
            edge_set.add((root_idx, node_idx))
            edge_set.add((node_idx, root_idx))

    if not edge_set:
        fallback_idx = 1 if root_idx == 0 and num_nodes > 1 else 0
        edge_set.add((fallback_idx, fallback_idx))
        if keep_root_edges and root_idx is not None and fallback_idx != root_idx:
            edge_set.add((root_idx, fallback_idx))
            edge_set.add((fallback_idx, root_idx))

    edges = sorted(edge_set)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _hash_text_feature(text: str, dim: int = _HASH_DIM) -> torch.Tensor:
    vector = torch.zeros(dim, dtype=torch.float)
    if not text:
        return vector
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    for token in tokens:
        if not token:
            continue
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vector[digest % dim] += 1.0
    norm = torch.norm(vector, p=2)
    if norm > 0:
        vector = vector / norm
    return vector


def _text_stats_feature(text: str) -> torch.Tensor:
    length = float(len(text))
    if length == 0:
        return torch.zeros(_STATS_DIM, dtype=torch.float)
    digits = sum(1 for ch in text if ch.isdigit())
    alphabetic = sum(1 for ch in text if ch.isalpha())
    return torch.tensor(
        [
            min(length / 100.0, 1.0),
            digits / length,
            alphabetic / length,
        ],
        dtype=torch.float,
    )


def _infer_type_dim(graph: Data) -> int:
    if hasattr(graph, "node_labels"):
        unique_labels = {str(label).upper() for label in list(getattr(graph, "node_labels", []) or [])}
        if len(unique_labels) + 1 <= _TYPE_VECTOR_DIM:
            return _TYPE_VECTOR_DIM
    return min(_TYPE_VECTOR_DIM, int(graph.x.size(1)))


def _build_raw_knowledge_graph(graph: Data, artifacts: Optional[Dict[str, Any]]) -> Optional[Data]:
    if artifacts is None:
        return None
    report_id = str(getattr(graph, "report_id", "")).strip()
    if not report_id:
        return None
    report_entities = artifacts["entities"].get(report_id, {})
    mention_records: List[dict] = []
    for para_idx in sorted(report_entities):
        for record in report_entities[para_idx]:
            mention_records.append(
                {
                    "paragraph_index": int(para_idx),
                    "label": str(record.get("label", "UNKNOWN")).strip().upper(),
                    "text": str(record.get("text", "")).strip(),
                }
            )
    mention_records = [record for record in mention_records if record["text"]]
    if not mention_records:
        return None

    feature_dim = int(graph.x.size(1))
    type_dim = _infer_type_dim(graph)
    hash_start = type_dim
    para_hash_start = min(feature_dim, hash_start + _HASH_DIM)
    ctx_hash_start = min(feature_dim, para_hash_start + _HASH_DIM)
    stats_start = min(feature_dim, ctx_hash_start + _HASH_DIM)
    root_idx = _find_report_node_index(graph)
    node_labels = list(getattr(graph, "node_labels", []) or [])
    known_labels = [str(label).upper() for label in node_labels if str(label).upper() != "REPORT"]
    type_to_idx = {label: idx for idx, label in enumerate(dict.fromkeys(known_labels))}
    unknown_type_idx = min(type_dim - 1, len(type_to_idx)) if type_dim > 0 else 0

    node_features: List[torch.Tensor] = []
    node_texts: List[str] = []
    rebuilt_labels: List[str] = []
    paragraph_indices: List[int] = []

    if root_idx is not None:
        root_feature = torch.zeros(feature_dim, dtype=torch.float)
        if type_dim > 0:
            root_feature[min(type_dim - 1, unknown_type_idx)] = 1.0
        if feature_dim >= hash_start + _HASH_DIM:
            root_feature[hash_start:hash_start + _HASH_DIM] = _hash_text_feature(report_id)
        if feature_dim >= stats_start + _STATS_DIM:
            root_feature[stats_start:stats_start + _STATS_DIM] = _text_stats_feature(report_id)
        node_features.append(root_feature)
        node_texts.append("REPORT")
        rebuilt_labels.append("REPORT")
        paragraph_indices.append(-1)

    for record in mention_records:
        feature = torch.zeros(feature_dim, dtype=torch.float)
        if type_dim > 0:
            type_idx = type_to_idx.get(record["label"], unknown_type_idx)
            feature[min(type_dim - 1, type_idx)] = 1.0
        if feature_dim >= hash_start + _HASH_DIM:
            feature[hash_start:hash_start + _HASH_DIM] = _hash_text_feature(record["text"])
        if feature_dim >= stats_start + _STATS_DIM:
            feature[stats_start:stats_start + _STATS_DIM] = _text_stats_feature(record["text"])
        node_features.append(feature)
        node_texts.append(record["text"])
        rebuilt_labels.append(record["label"])
        paragraph_indices.append(record["paragraph_index"])

    edge_set: set[tuple[int, int]] = set()
    start_idx = 1 if root_idx is not None else 0
    para_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for node_idx, para_idx in enumerate(paragraph_indices[start_idx:], start=start_idx):
        para_to_nodes[int(para_idx)].append(node_idx)
    for node_ids in para_to_nodes.values():
        unique_ids = sorted(set(node_ids))
        if len(unique_ids) == 1:
            edge_set.add((unique_ids[0], unique_ids[0]))
        else:
            for src, dst in combinations(unique_ids, 2):
                edge_set.add((src, dst))
                edge_set.add((dst, src))
    if root_idx is not None:
        for node_idx in range(1, len(node_features)):
            edge_set.add((0, node_idx))
            edge_set.add((node_idx, 0))
    if not edge_set:
        fallback_idx = 1 if len(node_features) > 1 else 0
        edge_set.add((fallback_idx, fallback_idx))
        if root_idx is not None and fallback_idx != 0:
            edge_set.add((0, fallback_idx))
            edge_set.add((fallback_idx, 0))

    rebuilt = Data(
        x=torch.stack(node_features),
        edge_index=torch.tensor(sorted(edge_set), dtype=torch.long).t().contiguous(),
        y=graph.y.clone(),
    )
    if hasattr(graph, "doc_emb") and graph.doc_emb is not None:
        rebuilt.doc_emb = torch.zeros_like(graph.doc_emb)
    rebuilt.report_id = getattr(graph, "report_id", report_id)
    rebuilt.apt_group = getattr(graph, "apt_group", None)
    rebuilt.node_texts = node_texts
    rebuilt.node_labels = rebuilt_labels
    rebuilt.node_paragraph_indices = paragraph_indices
    return rebuilt


def _edge_index_from_set(edge_set: set[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    if not edge_set:
        fallback_idx = 0 if num_nodes else 0
        edge_set.add((fallback_idx, fallback_idx))
    return torch.tensor(sorted(edge_set), dtype=torch.long).t().contiguous()


def _edge_set_from_graph(graph: Data) -> set[tuple[int, int]]:
    if graph.edge_index.numel() == 0:
        return set()
    return set(map(tuple, graph.edge_index.t().tolist()))


def _upper_node_labels(graph: Data) -> List[str]:
    labels = list(getattr(graph, "node_labels", []) or [])
    num_nodes = int(graph.x.size(0))
    if len(labels) != num_nodes:
        return ["UNKNOWN"] * num_nodes
    return [str(label).strip().upper() for label in labels]


def _add_root_edges(edge_set: set[tuple[int, int]], graph: Data) -> None:
    root_idx = _find_report_node_index(graph)
    if root_idx is None:
        return
    num_nodes = int(graph.x.size(0))
    for node_idx in range(num_nodes):
        if node_idx == root_idx:
            continue
        edge_set.add((root_idx, node_idx))
        edge_set.add((node_idx, root_idx))


def _add_sparse_completion_edges(edge_set: set[tuple[int, int]], graph: Data) -> None:
    root_idx = _find_report_node_index(graph)
    node_ids = [idx for idx in range(int(graph.x.size(0))) if idx != root_idx]
    if len(node_ids) >= 5:
        return
    for src in node_ids:
        for dst in node_ids:
            if src != dst:
                edge_set.add((src, dst))


def _semantic_relation_edges(graph: Data) -> set[tuple[int, int]]:
    labels = _upper_node_labels(graph)
    root_idx = _find_report_node_index(graph)
    edge_set: set[tuple[int, int]] = set()
    for src_idx, src_label in enumerate(labels):
        if src_idx == root_idx:
            continue
        for dst_idx, dst_label in enumerate(labels):
            if dst_idx == root_idx or src_idx == dst_idx:
                continue
            for source_types, target_types in _SEMANTIC_RELATION_RULES:
                if src_label in source_types and dst_label in target_types:
                    edge_set.add((src_idx, dst_idx))
                    edge_set.add((dst_idx, src_idx))
                    break
    return edge_set


def _augment_full_hetero_graph(graph: Data) -> Data:
    g = _clone_graph(graph)
    edge_set = _edge_set_from_graph(g)
    edge_set.update(_semantic_relation_edges(g))
    _add_root_edges(edge_set, g)
    _add_sparse_completion_edges(edge_set, g)
    g.edge_index = _edge_index_from_set(edge_set, int(g.x.size(0)))
    return g


def _build_cskg4apt_graph(graph: Data) -> Data:
    g = _clone_graph(graph)
    edge_set = _semantic_relation_edges(g)
    _add_root_edges(edge_set, g)
    _add_sparse_completion_edges(edge_set, g)
    if not edge_set:
        edge_set = _edge_set_from_graph(_augment_full_hetero_graph(g))
    g.edge_index = _edge_index_from_set(edge_set, int(g.x.size(0)))
    return g


def _build_attackg_graph(graph: Data) -> Data:
    g = _clone_graph(graph)
    labels = _upper_node_labels(g)
    paragraph_indices = list(getattr(g, "node_paragraph_indices", []) or [])
    root_idx = _find_report_node_index(g)
    attack_labels = {"MALWARE", "TOOL", "MITRE_TECH", "CVE", "FILE_PATH", "REGISTRY"} | _HASH_LABELS | _INFRA_LABELS
    edge_set = _semantic_relation_edges(g)
    if len(paragraph_indices) == int(g.x.size(0)):
        by_para: Dict[int, List[int]] = defaultdict(list)
        for idx, para_idx in enumerate(paragraph_indices):
            if idx == root_idx:
                continue
            if labels[idx] in attack_labels and int(para_idx) >= 0:
                by_para[int(para_idx)].append(idx)
        for node_ids in by_para.values():
            ordered = sorted(set(node_ids))
            for src, dst in zip(ordered, ordered[1:]):
                edge_set.add((src, dst))
                edge_set.add((dst, src))
    _add_root_edges(edge_set, g)
    _add_sparse_completion_edges(edge_set, g)
    g.edge_index = _edge_index_from_set(edge_set, int(g.x.size(0)))
    return g


def _build_apt_kgl_graph(graph: Data) -> Data:
    g = _clone_graph(graph)
    labels = _upper_node_labels(g)
    paragraph_indices = list(getattr(g, "node_paragraph_indices", []) or [])
    root_idx = _find_report_node_index(g)
    edge_set = _semantic_relation_edges(g)
    if len(paragraph_indices) == int(g.x.size(0)):
        by_para: Dict[int, List[int]] = defaultdict(list)
        for idx, para_idx in enumerate(paragraph_indices):
            if idx == root_idx:
                continue
            if labels[idx] in _PROVENANCE_LABELS and int(para_idx) >= 0:
                by_para[int(para_idx)].append(idx)
        for node_ids in by_para.values():
            unique_ids = sorted(set(node_ids))
            for src, dst in combinations(unique_ids, 2):
                edge_set.add((src, dst))
                edge_set.add((dst, src))
    _add_root_edges(edge_set, g)
    _add_sparse_completion_edges(edge_set, g)
    g.edge_index = _edge_index_from_set(edge_set, int(g.x.size(0)))
    return g


def _build_syntax_aware_graph(graph: Data, artifacts: Optional[Dict[str, Any]]) -> Data:
    g = _clone_graph(graph)
    root_idx = _find_report_node_index(g)
    edge_set: set[tuple[int, int]] = set()
    report_id = str(getattr(g, "report_id", "")).strip()
    node_texts = [str(item).strip().lower() for item in list(getattr(g, "node_texts", []) or [])]
    node_labels = _upper_node_labels(g)

    if artifacts is not None and report_id:
        key_to_node_idx: Dict[tuple[str, str], int] = {}
        for node_idx, (text, label) in enumerate(zip(node_texts, node_labels)):
            if node_idx == root_idx:
                continue
            key_to_node_idx[(text, label)] = node_idx

        report_entities = artifacts["entities"].get(report_id, {})
        for paragraph_entities in report_entities.values():
            ordered: List[tuple[int, int, int]] = []
            for record in paragraph_entities:
                text = str(record.get("text", "")).strip().lower()
                label = str(record.get("label", "")).strip().upper()
                node_idx = key_to_node_idx.get((text, label))
                if node_idx is None:
                    continue
                start = record.get("start", 0)
                end = record.get("end", start)
                ordered.append((int(start) if isinstance(start, int) else 0, int(end) if isinstance(end, int) else 0, node_idx))
            ordered = sorted(dict.fromkeys(ordered))
            node_order = [node_idx for _start, _end, node_idx in ordered]
            for src, dst in zip(node_order, node_order[1:]):
                if src != dst:
                    edge_set.add((src, dst))
                    edge_set.add((dst, src))
            for src, mid, dst in zip(node_order, node_order[1:], node_order[2:]):
                if src != dst:
                    edge_set.add((src, dst))
                    edge_set.add((dst, src))

    if not edge_set:
        paragraph_indices = list(getattr(g, "node_paragraph_indices", []) or [])
        if len(paragraph_indices) == int(g.x.size(0)):
            by_para: Dict[int, List[int]] = defaultdict(list)
            for node_idx, para_idx in enumerate(paragraph_indices):
                if node_idx == root_idx or int(para_idx) < 0:
                    continue
                by_para[int(para_idx)].append(node_idx)
            for node_ids in by_para.values():
                ordered = sorted(set(node_ids))
                for src, dst in zip(ordered, ordered[1:]):
                    edge_set.add((src, dst))
                    edge_set.add((dst, src))

    _add_root_edges(edge_set, g)
    _add_sparse_completion_edges(edge_set, g)
    g.edge_index = _edge_index_from_set(edge_set, int(g.x.size(0)))
    return g


def _build_cti_thinker_reasoning_graph(graph: Data) -> Data:
    g = _clone_graph(graph)
    labels = _upper_node_labels(g)
    edge_set = _semantic_relation_edges(g)
    root_idx = _find_report_node_index(g)

    reasoning_labels = {"MALWARE", "TOOL", "MITRE_TECH", "CVE", "OPERATION", "CAMPAIGN"} | _INFRA_LABELS | _HASH_LABELS
    reasoning_nodes = [idx for idx, label in enumerate(labels) if idx != root_idx and label in reasoning_labels]
    for src in reasoning_nodes:
        for dst in reasoning_nodes:
            if src != dst:
                edge_set.add((src, dst))

    _add_root_edges(edge_set, g)
    _add_sparse_completion_edges(edge_set, g)
    g.edge_index = _edge_index_from_set(edge_set, int(g.x.size(0)))
    return g


def _basic_feature_graph(graph: Data) -> Data:
    g = _clone_graph(graph)
    feature_dim = int(g.x.size(1))
    type_dim = min(_infer_type_dim(g), feature_dim)
    keep_until = min(feature_dim, type_dim + _HASH_DIM)
    reduced = torch.zeros_like(g.x)
    reduced[:, :keep_until] = g.x[:, :keep_until]
    g.x = reduced
    if hasattr(g, "doc_emb") and g.doc_emb is not None:
        g.doc_emb = torch.zeros_like(g.doc_emb)
    return g


def transform_graph_for_variant(
    graph: Data,
    graph_variant: str,
    processed_data_path: Optional[Path] = None,
    strict_repro: bool = False,
) -> Data:
    variant = str(graph_variant or 'full').lower()
    if variant in {'default', 'full', 'hetero_graph_full'}:
        return _augment_full_hetero_graph(graph)
    if variant == 'no_root_node':
        g = _augment_full_hetero_graph(graph)
        root_idx = _find_report_node_index(g)
        return _remove_node(g, root_idx) if root_idx is not None else g
    if variant == 'no_bridge_edges':
        g = _clone_graph(graph)
        artifacts = _require_variant_artifacts(variant, processed_data_path, strict_repro)
        edge_index = _build_graph_edges_from_artifacts(g, artifacts, mode='paragraph', keep_root_edges=True)
        g.edge_index = edge_index if edge_index is not None else _make_edge_index_from_paragraphs(g, mode='paragraph', keep_root_edges=True)
        return g
    if variant == 'no_semantic_edges':
        g = _clone_graph(graph)
        semantic_edges = _semantic_relation_edges(g)
        edge_set = _edge_set_from_graph(g) - semantic_edges
        _add_root_edges(edge_set, g)
        _add_sparse_completion_edges(edge_set, g)
        g.edge_index = _edge_index_from_set(edge_set, int(g.x.size(0)))
        return g
    if variant == 'paragraph_cooccurrence':
        g = _clone_graph(graph)
        artifacts = _require_variant_artifacts(variant, processed_data_path, strict_repro)
        edge_index = _build_graph_edges_from_artifacts(g, artifacts, mode='paragraph', keep_root_edges=False)
        if edge_index is not None:
            g.edge_index = edge_index
            if _find_report_node_index(g) is not None:
                g = _remove_node(g, 0)
        else:
            if _find_report_node_index(g) is not None:
                g = _remove_node(g, 0)
            g.edge_index = _make_edge_index_from_paragraphs(g, mode='paragraph', keep_root_edges=False)
        return g
    if variant == 'sliding_window_cooccurrence':
        g = _clone_graph(graph)
        artifacts = _require_variant_artifacts(variant, processed_data_path, strict_repro)
        edge_index = _build_graph_edges_from_artifacts(g, artifacts, mode='sliding_window', keep_root_edges=False, window_tokens=_WINDOW_TOKENS)
        if edge_index is not None:
            g.edge_index = edge_index
            if _find_report_node_index(g) is not None:
                g = _remove_node(g, 0)
        else:
            if _find_report_node_index(g) is not None:
                g = _remove_node(g, 0)
            g.edge_index = _make_edge_index_from_paragraphs(g, mode='sliding_window', keep_root_edges=False, paragraph_window=1)
        return g
    if variant == 'cskg4apt':
        return _build_cskg4apt_graph(graph)
    if variant == 'attackg':
        return _build_attackg_graph(graph)
    if variant in {'apt_kgl', 'apt_kgl_provenance'}:
        return _build_apt_kgl_graph(graph)
    if variant in {'syntax_aware_graph_network', 'syntax_dependency'}:
        artifacts = _require_variant_artifacts(variant, processed_data_path, strict_repro)
        return _build_syntax_aware_graph(graph, artifacts)
    if variant in {'cti_thinker', 'cti_thinker_reasoning'}:
        return _build_cti_thinker_reasoning_graph(graph)
    return _clone_graph(graph)


def transform_graphs_for_variant(
    graphs: List[Data],
    graph_variant: str,
    processed_data_path: Optional[Path] = None,
    strict_repro: bool = False,
) -> List[Data]:
    variant = str(graph_variant or 'full').lower()
    if variant in {'default'}:
        return graphs
    transformed = [
        transform_graph_for_variant(
            graph,
            variant,
            processed_data_path=processed_data_path,
            strict_repro=strict_repro,
        )
        for graph in graphs
    ]
    LOGGER.info("Applied graph variant transform: %s (%d graphs)", variant, len(transformed))
    return transformed


def transform_graph_features_for_variant(
    graphs: List[Data],
    feature_variant: str,
    processed_data_path: Optional[Path] = None,
    strict_repro: bool = False,
) -> List[Data]:
    variant = str(feature_variant or 'full').lower()
    if variant in {'default', 'full', 'apt_dshg_preprocess'}:
        return graphs

    artifacts = None
    if variant == 'knowledge_extraction_framework':
        artifacts = _require_variant_artifacts(variant, processed_data_path, strict_repro)

    transformed: List[Data] = []
    for graph in graphs:
        g = _clone_graph(graph)
        if variant == 'knowledge_extraction_framework':
            rebuilt = _build_raw_knowledge_graph(graph, artifacts)
            if rebuilt is not None:
                g = rebuilt
        elif variant in {'basic_features', 'no_multigranularity_features'}:
            g = _basic_feature_graph(graph)
        transformed.append(g)

    LOGGER.info("Applied feature variant transform: %s (%d graphs)", variant, len(transformed))
    return transformed


def split_graph_dataset(
    graphs: List[Data],
    seed: int,
    split_mode: str = "stratified",
    raw_index_path: Optional[Path] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
):
    labels = [int(g.y.item()) for g in graphs]
    report_ids = [str(getattr(g, "report_id", f"graph_{idx}")) for idx, g in enumerate(graphs)]
    train_idx, val_idx, test_idx, split_meta = build_report_level_split(
        report_ids=report_ids,
        labels=labels,
        seed=seed,
        split_mode=split_mode,
        raw_index_path=raw_index_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_graphs = [graphs[int(i)] for i in train_idx.tolist()]
    val_graphs = [graphs[int(i)] for i in val_idx.tolist()]
    test_graphs = [graphs[int(i)] for i in test_idx.tolist()]
    LOGGER.info("数据集划分(分层): 训练集=%d, 验证集=%d, 测试集=%d", len(train_graphs), len(val_graphs), len(test_graphs))
    LOGGER.info(
        "Dataset split (%s -> %s): train=%d val=%d test=%d",
        split_meta.get("split_mode_requested", split_mode),
        split_meta.get("split_mode_resolved", split_mode),
        len(train_graphs),
        len(val_graphs),
        len(test_graphs),
    )
    return train_graphs, val_graphs, test_graphs, split_meta


def _filter_graphs_by_min_class_samples(graphs: List[Data], min_class_samples: int) -> tuple[List[Data], Dict[str, Any]]:
    threshold = int(min_class_samples or 1)
    if threshold <= 1:
        return graphs, {
            "min_class_samples": threshold,
            "class_filter_removed_graphs": 0,
            "class_filter_removed_classes": 0,
            "class_filter_kept_classes": len({int(g.y.item()) for g in graphs if g.y is not None}),
        }
    counts: Dict[int, int] = defaultdict(int)
    for graph in graphs:
        if graph.y is not None:
            counts[int(graph.y.item())] += 1
    kept_labels = {label for label, count in counts.items() if count >= threshold}
    filtered = [graph for graph in graphs if graph.y is not None and int(graph.y.item()) in kept_labels]
    return filtered, {
        "min_class_samples": threshold,
        "class_filter_removed_graphs": int(len(graphs) - len(filtered)),
        "class_filter_removed_classes": int(len(counts) - len(kept_labels)),
        "class_filter_kept_classes": int(len(kept_labels)),
    }


def _graph_structure_summary(graphs: List[Data]) -> Dict[str, Any]:
    if not graphs:
        return {
            "graph_count": 0,
            "graph_nodes_mean": 0.0,
            "graph_edges_mean": 0.0,
            "graph_avg_degree_mean": 0.0,
            "graph_density_mean": 0.0,
            "graph_cross_paragraph_edge_ratio_mean": 0.0,
        }
    node_counts: List[float] = []
    edge_counts: List[float] = []
    avg_degrees: List[float] = []
    densities: List[float] = []
    cross_ratios: List[float] = []
    for graph in graphs:
        n_nodes = int(graph.x.size(0))
        n_edges = int(graph.edge_index.size(1)) if graph.edge_index is not None else 0
        node_counts.append(float(n_nodes))
        edge_counts.append(float(n_edges))
        avg_degrees.append(float(n_edges) / max(1.0, float(n_nodes)))
        densities.append(float(n_edges) / max(1.0, float(n_nodes * max(1, n_nodes - 1))))
        paragraph_indices = getattr(graph, "node_paragraph_indices", None)
        if isinstance(paragraph_indices, list) and len(paragraph_indices) == n_nodes and n_edges:
            cross_edges = 0
            valid_edges = 0
            for src, dst in graph.edge_index.t().tolist():
                src_para = int(paragraph_indices[src])
                dst_para = int(paragraph_indices[dst])
                if src_para < 0 or dst_para < 0:
                    continue
                valid_edges += 1
                if src_para != dst_para:
                    cross_edges += 1
            cross_ratios.append(float(cross_edges) / max(1.0, float(valid_edges)))
        else:
            cross_ratios.append(0.0)

    return {
        "graph_count": int(len(graphs)),
        "graph_nodes_mean": round(float(np.mean(node_counts)), 6),
        "graph_nodes_std": round(float(np.std(node_counts)), 6),
        "graph_edges_mean": round(float(np.mean(edge_counts)), 6),
        "graph_edges_std": round(float(np.std(edge_counts)), 6),
        "graph_avg_degree_mean": round(float(np.mean(avg_degrees)), 6),
        "graph_density_mean": round(float(np.mean(densities)), 6),
        "graph_cross_paragraph_edge_ratio_mean": round(float(np.mean(cross_ratios)), 6),
    }


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


def _forward_model(model, batch):
    doc_emb = batch.doc_emb if hasattr(batch, 'doc_emb') else None
    if isinstance(model, APTAttributionGraphSAGE):
        return model(batch.x, batch.edge_index, batch.batch, doc_emb=doc_emb)
    return model(batch.x, batch.edge_index, batch.batch)


def _class_weight_tensor(graphs: List[Data], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float)
    for graph in graphs:
        if graph.y is None:
            continue
        label = int(graph.y.item())
        if 0 <= label < num_classes:
            counts[label] += 1.0
    safe_counts = torch.clamp(counts, min=1.0)
    weights = counts.sum().clamp(min=1.0) / (safe_counts * max(1, num_classes))
    weights[counts == 0] = 0.0
    positive = weights[weights > 0]
    if positive.numel() > 0:
        weights = weights / positive.mean()
    return weights.to(device)


def _collect_logits_and_labels(model, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    logits: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits.append(_forward_model(model, batch).detach().cpu())
            labels.append(batch.y.detach().cpu())
    if not logits:
        return torch.empty((0, 0), dtype=torch.float), torch.empty((0,), dtype=torch.long)
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0).long()


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0 or labels.numel() == 0 or logits.size(0) < 2:
        return 1.0
    log_temperature = torch.zeros((), dtype=torch.float, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 10.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        return float(torch.exp(log_temperature).clamp(0.05, 10.0).detach().cpu())
    except Exception:
        return 1.0


def _nll_for_temperature(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> float:
    if logits.numel() == 0 or labels.numel() == 0:
        return 0.0
    with torch.no_grad():
        return float(F.cross_entropy(logits / max(temperature, 1e-6), labels).cpu())


def _topk_records(
    logits: torch.Tensor,
    labels: torch.Tensor,
    report_ids: List[str],
    idx_to_label: Dict[int, str],
    temperature: float,
    k: int = 3,
) -> List[Dict[str, Any]]:
    if logits.numel() == 0:
        return []
    probs = F.softmax(logits / max(temperature, 1e-6), dim=1)
    top_k = min(k, probs.size(1))
    values, indices = torch.topk(probs, k=top_k, dim=1)
    records: List[Dict[str, Any]] = []
    for row_idx in range(probs.size(0)):
        records.append(
            {
                "report_id": report_ids[row_idx] if row_idx < len(report_ids) else f"test_{row_idx}",
                "true_label": idx_to_label.get(int(labels[row_idx]), str(int(labels[row_idx]))),
                "top_k": [
                    {
                        "label": idx_to_label.get(int(indices[row_idx, rank]), str(int(indices[row_idx, rank]))),
                        "probability": round(float(values[row_idx, rank]), 6),
                    }
                    for rank in range(top_k)
                ],
            }
        )
    return records


def _flatten_label_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        labels: List[str] = []
        for item in value:
            labels.extend(_flatten_label_list(item))
        return labels
    return []


def _safe_tensor_mean(value: Any) -> Optional[float]:
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return float(value.detach().float().mean().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summarize_rgat_gate_attention(model, loader, device: torch.device, output_path: Path) -> Optional[Dict[str, Any]]:
    gate_values: List[float] = []
    node_attention_values: List[float] = []
    edge_attention_values: List[float] = []
    node_type_gate_values: Dict[str, List[float]] = defaultdict(list)
    semantic_gate_means: List[float] = []
    structural_gate_means: List[float] = []
    batch_count = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            try:
                _, att = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
            except TypeError:
                return None
            if not isinstance(att, dict):
                continue
            batch_count += 1

            semantic_mean = _safe_tensor_mean(att.get("semantic_gate_mean"))
            structural_mean = _safe_tensor_mean(att.get("structural_gate_mean"))
            if semantic_mean is not None:
                semantic_gate_means.append(semantic_mean)
            if structural_mean is not None:
                structural_gate_means.append(structural_mean)

            gate_alpha = att.get("gate_alpha")
            if isinstance(gate_alpha, torch.Tensor) and gate_alpha.numel() > 0:
                gate_flat = gate_alpha.detach().float().view(-1).cpu().numpy().tolist()
                gate_values.extend(float(v) for v in gate_flat)
                node_labels = _flatten_label_list(getattr(batch, "node_labels", None))
                if len(node_labels) == len(gate_flat):
                    for label, value in zip(node_labels, gate_flat):
                        node_type_gate_values[str(label)].append(float(value))

            node_attention = att.get("node_attention")
            if isinstance(node_attention, torch.Tensor) and node_attention.numel() > 0:
                node_attention_values.extend(
                    float(v) for v in node_attention.detach().float().view(-1).cpu().numpy().tolist()
                )

            edge_attention = att.get("edge_attention")
            if isinstance(edge_attention, torch.Tensor) and edge_attention.numel() > 0:
                edge_attention_values.extend(
                    float(v) for v in edge_attention.detach().float().view(-1).cpu().numpy().tolist()
                )

    if batch_count == 0:
        return None

    def _stats(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0}
        arr = np.asarray(values, dtype=float)
        return {
            "count": int(arr.size),
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "p25": round(float(np.quantile(arr, 0.25)), 6),
            "median": round(float(np.quantile(arr, 0.5)), 6),
            "p75": round(float(np.quantile(arr, 0.75)), 6),
            "max": round(float(np.max(arr)), 6),
        }

    gate_stats = _stats(gate_values)
    if gate_values:
        gate_arr = np.asarray(gate_values, dtype=float)
        gate_stats["semantic_dominant_ratio"] = round(float(np.mean(gate_arr > 0.5)), 6)
        gate_stats["structural_dominant_ratio"] = round(float(np.mean(gate_arr < 0.5)), 6)

    summary = {
        "batch_count": batch_count,
        "semantic_gate_mean": round(float(np.mean(semantic_gate_means)), 6) if semantic_gate_means else None,
        "structural_gate_mean": round(float(np.mean(structural_gate_means)), 6) if structural_gate_means else None,
        "gate_alpha": gate_stats,
        "node_attention": _stats(node_attention_values),
        "edge_attention": _stats(edge_attention_values),
        "gate_by_node_type": {
            label: _stats(values)
            for label, values in sorted(node_type_gate_values.items())
        },
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


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


def _resolve_dataloader_workers(config: Dict[str, Any]) -> int:
    """Choose a sandbox-safe DataLoader worker count.

    Restricted environments often forbid the socket-based resource sharing used
    by multiprocessing DataLoader workers. Default to single-process loading on
    CPU and allow explicit override for full experimental machines.
    """
    configured = config.get("num_workers", None)
    if configured is not None:
        return max(0, int(configured))
    if torch.cuda.is_available():
        return 4 if os.name != "nt" else 0
    return 0

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
    raw_index_path = processed_data_path / "raw_index.csv"

    model_type = config.get('model_type', 'GAT')
    graph_variant = str(config.get('graph_variant', 'full'))
    feature_variant = str(config.get('feature_variant', 'full'))
    split_mode = str(config.get('split_mode', 'stratified'))
    strict_repro = bool(config.get('strict_repro', True))
    fusion_mode = str(config.get('fusion_mode', 'adaptive'))
    pooling_mode = str(config.get('pooling_mode', 'attention'))
    rgcn_num_bases = config.get('rgcn_num_bases', 30)
    raw_loss_type = config.get('loss_type', None)
    if raw_loss_type is None:
        raw_loss_type = 'focal' if model_type == "RGAT" else 'cross_entropy'
    loss_type = str(raw_loss_type).strip().lower()
    focal_gamma = float(config.get('focal_gamma', 2.0))
    use_temperature_calibration = bool(config.get('use_temperature_calibration', True))
    min_class_samples = int(config.get('min_class_samples', 1) or 1)
    enable_micro_benchmark = bool(config.get('enable_micro_benchmark', True))
    enable_flops = bool(config.get('enable_flops', True))
    inference_benchmark_samples = int(config.get('inference_benchmark_samples', 100) or 0)
    epochs = int(config.get('epochs', 100))
    lr = float(config.get('lr', 0.001))
    batch_size = int(config.get('batch_size', 32))
    seed = int(config.get('seed', 42))
    gpu_id = int(config.get('gpu_id', -1))
    dataset_id = str(config.get('dataset_id', config.get('dataset_name', 'unknown')))
    require_cuda = bool(config.get('require_cuda', True))

    if strict_repro and model_type == "HGT":
        raise ValueError("HGT in this repository is an HGT-inspired approximation over homogeneous graphs. Use --allow-approximate to run it.")

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this experiment, but PyTorch cannot access a GPU in the current runtime."
        )

    # Honor explicit gpu_id (the runner sets CUDA_VISIBLE_DEVICES so cuda:0 is the right card)
    if gpu_id >= 0 and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 若调用方提供了 run_dir，则直接用（便于实验 runner 控制目录）
    if config.get('run_dir'):
        run_output_dir = Path(config['run_dir'])
    else:
        run_output_dir = base_output_dir / f"{model_type}_{timestamp}_seed{seed}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        LOGGER.info("CUDA is available. Using GPU for training.")
        LOGGER.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        LOGGER.warning("CUDA is not available. Using CPU for training. This may be slow.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_vram_peak()

    # TimeLogger: 实验全过程时间与资源采集
    time_logger = TimeLogger(
        experiment_id=f"{model_type}_{dataset_id}_seed{seed}",
        dataset=dataset_id,
        model=model_type,
        seed=seed,
    )
    
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    LOGGER.info(f"Starting training run. Output: {run_output_dir}")

    with time_logger.section("data_load"):
        graphs, label_to_idx, idx_to_label = load_graph_dataset(processed_data_path, label_mapping_path)
    with time_logger.section("graph_variant_transform"):
        graphs = transform_graphs_for_variant(
            graphs,
            graph_variant,
            processed_data_path=processed_data_path,
            strict_repro=strict_repro,
        )
    with time_logger.section("feature_variant_transform"):
        graphs = transform_graph_features_for_variant(
            graphs,
            feature_variant,
            processed_data_path=processed_data_path,
            strict_repro=strict_repro,
        )
    graphs, class_filter_meta = _filter_graphs_by_min_class_samples(graphs, min_class_samples)
    time_logger.update(**class_filter_meta)
    time_logger.update(**_graph_structure_summary(graphs))
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

    train_graphs, val_graphs, test_graphs, split_meta = split_graph_dataset(
        graphs,
        seed=seed,
        split_mode=split_mode,
        raw_index_path=raw_index_path,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    time_logger.update(**split_meta)
    
    # Use CPU-safe defaults locally; explicit overrides can re-enable workers on full lab machines.
    num_workers = _resolve_dataloader_workers(config)

    # Only use pin_memory if CUDA is available to avoid warnings on CPU
    use_pin_memory = torch.cuda.is_available()
    time_logger.update(dataloader_num_workers=num_workers, dataloader_pin_memory=use_pin_memory)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    # Model
    if model_type == "GCN":
        model = APTAttributionGCN(input_dim, 128, num_classes)
    elif model_type == "GAT":
        model = APTAttributionGAT(input_dim, 128, num_classes)
    elif model_type == "HGT":
        if HGTClassifier is None:
            raise ValueError("HGT model is not available (ImportError).")
        model = HGTClassifier(input_dim, 128, num_classes, num_node_types=40, heads=4, dropout=0.5)
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
            num_entity_types=40,  # Default safe upper bound
            ablation_mode=ablation_mode,
            use_gatv2=True,
            fusion_mode=fusion_mode,
            pooling_mode=pooling_mode,
            rgcn_num_bases=rgcn_num_bases,
        )
    else:
        model = APTAttributionHybrid(input_dim, 128, num_classes)
    
    model = model.to(device)

    # Compile model if supported (PyTorch 2.0+) - 默认关闭，便于多 seed 并行 + 调试
    if config.get('use_compile', False) and hasattr(torch, 'compile') and os.name != 'nt':
        try:
            model = torch.compile(model)
            LOGGER.info("Model compiled with torch.compile() for faster training.")
        except Exception as e:
            LOGGER.warning(f"Failed to compile model: {e}")

    # 多 GPU DataParallel：仅在显式启用、batch 足够大、设备数 >= 2 时启用
    if config.get('multi_gpu_dp', False) and torch.cuda.device_count() >= 2 and batch_size >= 64:
        model = torch.nn.DataParallel(model)
        LOGGER.info(f"DataParallel enabled across {torch.cuda.device_count()} GPUs.")

    # 记录参数量和关键实验配置
    time_logger.update(**count_params(model))
    time_logger.update(
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        fusion_mode=fusion_mode,
        pooling_mode=pooling_mode,
        rgcn_num_bases=rgcn_num_bases,
        enable_micro_benchmark=enable_micro_benchmark,
        enable_flops=enable_flops,
        inference_benchmark_samples=inference_benchmark_samples,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    class_weights = _class_weight_tensor(train_graphs, num_classes, device)
    time_logger.update(
        class_weighting="inverse_frequency",
        class_weights=[round(float(value), 6) for value in class_weights.detach().cpu()],
    )
    if loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
    elif loss_type in {"cross_entropy", "ce"}:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
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

    best_epoch_idx = 0
    early_stop_epoch = None
    train_total_t0 = time.perf_counter()

    for epoch in range(epochs):
        _t_train_start = time.perf_counter()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        _t_train_end = time.perf_counter()
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        _t_val_end = time.perf_counter()

        # 记录每轮耗时
        time_logger.add_epoch_time(
            train=_t_train_end - _t_train_start,
            val=_t_val_end - _t_train_end,
            data_load=0.0,  # PyG DataLoader 内联在 train_epoch 中，无法准确分离
        )

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
            best_epoch_idx = epoch + 1
            # 保存 state_dict；若被 DataParallel 包裹则取 module
            state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, best_model_path)
            LOGGER.info(f"Epoch {epoch+1}: New best model saved (Acc={val_acc:.4f}, Loss={val_loss:.4f})")

        # Early Stopping based on Loss (Prevent Overfitting)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        LOGGER.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f} | Patience={no_improve_epochs}/{early_stop_patience}")

        if no_improve_epochs >= early_stop_patience:
            early_stop_epoch = epoch + 1
            LOGGER.info(f"Early stopping triggered at epoch {epoch+1} (Val Loss did not improve for {early_stop_patience} epochs)")
            break

    train_total_t1 = time.perf_counter()
    time_logger.update(
        best_epoch=best_epoch_idx,
        early_stop_epoch=early_stop_epoch,
        train_total_wall_s=round(train_total_t1 - train_total_t0, 4),
    )

    # Plot History
    history_plot_path = run_output_dir / "training_history.png"
    try:
        plot_training_history(history, history_plot_path)
    except Exception as e:
        LOGGER.warning(f"Failed to plot history: {e}")

    # ========================================================================
    # Final Evaluation
    # ========================================================================
    model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_load.load_state_dict(torch.load(best_model_path, map_location=device))
    eval_model = model_to_load

    eval_model.eval()
    with time_logger.section("test_eval"):
        val_logits, val_labels = _collect_logits_and_labels(eval_model, val_loader, device)
        temperature = _fit_temperature(val_logits, val_labels) if use_temperature_calibration else 1.0
        test_logits, test_labels = _collect_logits_and_labels(eval_model, test_loader, device)
        calibrated_logits = test_logits / max(temperature, 1e-6) if test_logits.numel() else test_logits
        all_preds = calibrated_logits.argmax(dim=1).cpu().numpy().tolist() if calibrated_logits.numel() else []
        all_labels = test_labels.cpu().numpy().tolist()
        time_logger.update(
            temperature=round(float(temperature), 6),
            use_temperature_calibration=use_temperature_calibration,
            val_nll_uncalibrated=round(_nll_for_temperature(val_logits, val_labels, 1.0), 6),
            val_nll_calibrated=round(_nll_for_temperature(val_logits, val_labels, temperature), 6),
            test_nll_calibrated=round(_nll_for_temperature(test_logits, test_labels, temperature), 6),
        )

    # 7 项分类指标（论文公式 27–33）
    test_acc = float(accuracy_score(all_labels, all_preds))
    test_w_p = float(precision_score(all_labels, all_preds, average="weighted", zero_division=0))
    test_w_r = float(recall_score(all_labels, all_preds, average="weighted", zero_division=0))
    test_w_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    test_m_p = float(precision_score(all_labels, all_preds, average="macro", zero_division=0))
    test_m_r = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))
    test_m_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    # Classification report
    all_class_ids = list(range(len(idx_to_label)))
    target_names = [idx_to_label[i] for i in all_class_ids]
    cls_report = classification_report(
        all_labels, all_preds, labels=all_class_ids, target_names=target_names,
        output_dict=True, zero_division=0,
    )

    # Confusion Matrix
    cm_plot_path = run_output_dir / "confusion_matrix.png"
    try:
        plot_confusion_matrix(all_labels, all_preds, target_names, cm_plot_path)
    except Exception as e:
        LOGGER.warning(f"Failed to plot confusion matrix: {e}")

    topk_path = run_output_dir / "topk_predictions.json"
    try:
        report_ids = [str(getattr(graph, "report_id", f"test_{idx}")) for idx, graph in enumerate(test_graphs)]
        topk_path.write_text(
            json.dumps(
                _topk_records(test_logits, test_labels, report_ids, idx_to_label, temperature, k=3),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        LOGGER.warning(f"Failed to write top-k predictions: {e}")

    gate_summary_path = run_output_dir / "gate_summary.json"
    gate_summary = None
    if model_type == "RGAT":
        try:
            gate_summary = _summarize_rgat_gate_attention(eval_model, test_loader, device, gate_summary_path)
            if gate_summary:
                gate_alpha_stats = gate_summary.get("gate_alpha", {})
                time_logger.update(
                    gate_semantic_mean=gate_summary.get("semantic_gate_mean"),
                    gate_structural_mean=gate_summary.get("structural_gate_mean"),
                    gate_node_count=gate_alpha_stats.get("count", 0),
                    gate_semantic_dominant_ratio=gate_alpha_stats.get("semantic_dominant_ratio"),
                    gate_structural_dominant_ratio=gate_alpha_stats.get("structural_dominant_ratio"),
                )
        except Exception as e:
            LOGGER.warning(f"Failed to write RGAT gate summary: {e}")

    # ========================================================================
    # 时间细化：forward/backward micro-benchmark + 推理延迟
    # ========================================================================
    eval_model.train()
    try:
        # 取一个 batch 做 5 次前向 / 反向计时
        sample_batches = []
        for b in train_loader:
            sample_batches.append(b)
            if len(sample_batches) >= 1:
                break
        if sample_batches:
            sample_batch = sample_batches[0].to(device)
            # warmup
            for _ in range(2):
                _doc = sample_batch.doc_emb if hasattr(sample_batch, 'doc_emb') else None
                _o = eval_model(sample_batch.x, sample_batch.edge_index, sample_batch.batch, doc_emb=_doc) \
                    if isinstance(eval_model, APTAttributionGraphSAGE) else eval_model(sample_batch.x, sample_batch.edge_index, sample_batch.batch)
                _loss = criterion(_o, sample_batch.y)
                _loss.backward()
            # measure
            for _ in range(5):
                ft = CudaTimer(device)
                ft.start()
                _doc = sample_batch.doc_emb if hasattr(sample_batch, 'doc_emb') else None
                _o = eval_model(sample_batch.x, sample_batch.edge_index, sample_batch.batch, doc_emb=_doc) \
                    if isinstance(eval_model, APTAttributionGraphSAGE) else eval_model(sample_batch.x, sample_batch.edge_index, sample_batch.batch)
                f_ms = ft.stop()
                time_logger.add_forward_ms(f_ms)

                bt = CudaTimer(device)
                bt.start()
                _loss = criterion(_o, sample_batch.y)
                _loss.backward()
                b_ms = bt.stop()
                time_logger.add_backward_ms(b_ms)
    except Exception as e:
        LOGGER.warning(f"forward/backward micro-bench failed: {e}")

    # 推理延迟：单样本 100 次（不足则取全部测试集）
    eval_model.eval()
    try:
        inf_samples = test_graphs[: min(max(0, inference_benchmark_samples), len(test_graphs))]
        inf_ms_list: List[float] = []
        with torch.no_grad():
            for g in inf_samples:
                gd = g.to(device)
                ct = CudaTimer(device)
                ct.start()
                _doc = gd.doc_emb if hasattr(gd, 'doc_emb') and gd.doc_emb is not None else None
                _batch_idx = torch.zeros(gd.x.size(0), dtype=torch.long, device=device)
                if isinstance(eval_model, APTAttributionGraphSAGE):
                    _ = eval_model(gd.x, gd.edge_index, _batch_idx, doc_emb=_doc)
                else:
                    _ = eval_model(gd.x, gd.edge_index, _batch_idx)
                inf_ms_list.append(ct.stop())
        if inf_ms_list:
            time_logger.update(
                inference_per_sample_ms=round(sum(inf_ms_list) / len(inf_ms_list), 4),
                inference_samples_used=len(inf_ms_list),
            )
    except Exception as e:
        LOGGER.warning(f"inference latency benchmark failed: {e}")

    # FLOPs (best-effort)
    try:
        def _sample_input_fn():
            g = test_graphs[0].to(device)
            bidx = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
            return (g.x, g.edge_index, bidx)
        flops = count_flops_safe(eval_model, _sample_input_fn)
        if flops is not None:
            time_logger.update(flops_per_forward=int(flops))
    except Exception as e:
        LOGGER.debug(f"FLOPs estimation skipped: {e}")

    # 保存 time_log.json + metrics.json
    time_logger.finalize()
    time_log_path = run_output_dir / "time_log.json"
    time_logger.save(time_log_path)

    metrics = {
        "accuracy": test_acc,
        "weighted_precision": test_w_p,
        "weighted_recall": test_w_r,
        "weighted_f1": test_w_f1,
        "macro_precision": test_m_p,
        "macro_recall": test_m_r,
        "macro_f1": test_m_f1,
    }
    with (run_output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    results = {
        # 兼容旧字段
        "test_accuracy": test_acc,
        "test_f1_weighted": test_w_f1,
        # 完整 7 项
        "metrics": metrics,
        "config": config,
        "model_path": str(best_model_path),
        "time_log_path": str(time_log_path),
        "temperature": round(float(temperature), 6),
        "topk_predictions_path": str(topk_path) if topk_path.exists() else None,
        "gate_summary_path": str(gate_summary_path) if gate_summary_path.exists() else None,
        "gate_summary": gate_summary,
        "history_plot": str(history_plot_path) if history_plot_path.exists() else None,
        "confusion_matrix_plot": str(cm_plot_path) if cm_plot_path.exists() else None,
        "classification_report": cls_report,
        "history": history,
    }

    with (run_output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results
