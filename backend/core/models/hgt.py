from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import TransformerConv, global_mean_pool


class HGTClassifier(nn.Module):
    """HGT-inspired baseline over homogeneous PyG Data.

    This repository preprocesses reports into homogeneous `torch_geometric.data.Data`
    objects. A full native HGT implementation would need `HeteroData`, so this model
    injects node-type and relation-type signals into TransformerConv as a pragmatic
    approximation suitable for the existing pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_node_types: int = 40,
        heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_node_types = int(num_node_types)
        self.num_relations = (self.num_node_types + 1) ** 2
        self.dropout = float(dropout)

        self.input_proj = Linear(input_dim, hidden_dim)
        self.node_type_emb = nn.Embedding(self.num_node_types + 1, hidden_dim)
        self.edge_type_emb = nn.Embedding(self.num_relations, hidden_dim)

        self.conv1 = TransformerConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=hidden_dim,
            beta=True,
        )
        self.conv2 = TransformerConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=hidden_dim,
            beta=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _infer_node_types(self, x: torch.Tensor) -> torch.Tensor:
        type_dim = min(self.num_node_types + 1, x.size(1))
        return x[:, :type_dim].argmax(dim=1).clamp(max=self.num_node_types)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        node_types = self._infer_node_types(x)
        h = self.input_proj(x) + self.node_type_emb(node_types)

        src_type = node_types[edge_index[0]]
        dst_type = node_types[edge_index[1]]
        edge_type = (src_type * (self.num_node_types + 1) + dst_type).clamp(max=self.num_relations - 1)
        edge_attr = self.edge_type_emb(edge_type)

        h = self.conv1(h, edge_index, edge_attr)
        h = self.norm1(F.relu(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(F.relu(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        pooled = global_mean_pool(h, batch)
        return self.classifier(pooled)
