import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, RGCNConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

class RGAPTiveFusion(nn.Module):
    """
    RGAPTive Fusion Module (Innovation Point 3)
    Dynamically balances the contribution of semantic (GAT) and structural (RGCN) features
    using a learnable gating mechanism.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Gate network: Input is concatenation of both branches -> Output is gate value [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_semantic, x_structural, return_gate=False):
        # Concatenate features
        combined = torch.cat([x_semantic, x_structural], dim=-1)
        # Compute gate coefficient alpha
        alpha = self.gate_net(combined)
        # Weighted sum: alpha * Semantic + (1 - alpha) * Structural
        fused = alpha * x_semantic + (1 - alpha) * x_structural
        if return_gate:
            return fused, alpha
        return fused

class RelationAwareGAT(torch.nn.Module):
    """
    Relation-aware Graph Attention Network (RGAT) for APT Attribution.
    
    Key Innovations:
    1. Schema-Free Dynamic Heterogeneous Learning: Dynamic edge type inference without predefined schema.
    2. Structural-Semantic Dual-Stream Complementarity: Parallel GAT (Semantic) and RGCN (Structural) branches.
    3. RGAPTive Fusion: Learnable gating to balance semantic and structural information.
    4. Hierarchical/Global Attention Pooling: Improved graph readout using attention.
    
    Ablation Support:
    - mode="dual": Full model (GAT + RGCN + Fusion)
    - mode="gat_only": Only Semantic branch (Standard GAT)
    - mode="rgcn_only": Only Structural branch (Standard RGCN)
    """
    def __init__(
        self, 
        num_node_features: int, 
        num_classes: int, 
        hidden_dim: int = 128, 
        num_heads: int = 4, 
        dropout: float = 0.5,
        num_entity_types: int = 40,
        ablation_mode: str = "dual",  # "dual", "gat_only", "rgcn_only"
        num_layers: int = 2,
        use_gatv2: bool = True,
        fusion_mode: str = "adaptive",
        pooling_mode: str = "attention",
        rgcn_num_bases: int | None = 30,
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_entity_types = num_entity_types
        self.ablation_mode = ablation_mode
        self.fusion_mode = str(fusion_mode).strip().lower()
        self.pooling_mode = str(pooling_mode).strip().lower()
        
        # Calculate number of possible relations (Source Type -> Target Type)
        self.num_relations = (num_entity_types + 1) ** 2
        
        GATLayer = GATv2Conv if use_gatv2 else GATConv
        
        # --- Branch 1: Semantic Stream (GAT) ---
        if ablation_mode in ["dual", "gat_only"]:
            # Output dim is hidden_dim (if heads>1, we project or concat. Here we concat then project)
            # Actually GATConv with concat=True outputs heads*out_channels
            # We want final output to be hidden_dim for fusion
            self.gat_conv = GATLayer(num_node_features, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            # Projection to align dimensions if needed, but here heads * (hidden/heads) = hidden.
            
        # --- Branch 2: Structural Stream (RGCN) ---
        if ablation_mode in ["dual", "rgcn_only"]:
            # Uses Basis Decomposition for parameter efficiency with many relations
            num_bases = rgcn_num_bases
            if isinstance(num_bases, int) and num_bases <= 0:
                num_bases = None
            self.rgcn_conv = RGCNConv(
                num_node_features,
                hidden_dim,
                num_relations=self.num_relations,
                num_bases=num_bases,
            )

        # --- RGAPTive Fusion ---
        if ablation_mode == "dual":
            if self.fusion_mode == "adaptive":
                self.fusion = RGAPTiveFusion(hidden_dim)
            elif self.fusion_mode == "mean":
                self.fusion = None
            else:
                raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        
        # --- Layer 2: Deep Processing (Shared or Specific) ---
        # We use a shared GAT layer for deeper processing after fusion
        self.conv2 = GATLayer(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)

        # --- Readout: Global Attention Pooling (Innovation Point 4) ---
        # Computes importance of each node for the graph representation
        if self.pooling_mode == "attention":
            self.attention_pool = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_dim // 2, 1)
                )
            )
        elif self.pooling_mode == "mean":
            self.attention_pool = None
        else:
            raise ValueError(f"Unsupported pooling_mode: {pooling_mode}")

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch, return_attention=False):
        # --- Dynamic Relation Inference (Innovation Point 1) ---
        # Infer edge types on the fly based on node entity types
        type_dim = min(self.num_entity_types, x.size(1))
        src_type = x[edge_index[0], :type_dim].argmax(dim=1)
        dst_type = x[edge_index[1], :type_dim].argmax(dim=1)
        edge_type = src_type * (type_dim + 1) + dst_type
        edge_type = edge_type.clamp(max=self.num_relations - 1)

        # --- Layer 1: Dual-Stream Processing ---
        x_semantic = None
        x_structural = None
        
        att_weights = None

        if self.ablation_mode in ["dual", "gat_only"]:
            if return_attention:
                # GATConv returns (out, (edge_index, alpha)) when return_attention_weights=True
                x_semantic, (att_edge_index, att_alpha) = self.gat_conv(x, edge_index, return_attention_weights=True)
                att_weights = {"edge_index": att_edge_index, "edge_attention": att_alpha}
            else:
                x_semantic = self.gat_conv(x, edge_index) # [N, Hidden]
        
        if self.ablation_mode in ["dual", "rgcn_only"]:
            x_structural = self.rgcn_conv(x, edge_index, edge_type) # [N, Hidden]

        # --- Feature Fusion ---
        gate_alpha = None
        if self.ablation_mode == "dual":
            if self.fusion_mode == "adaptive":
                if return_attention:
                    x, gate_alpha = self.fusion(x_semantic, x_structural, return_gate=True)
                else:
                    x = self.fusion(x_semantic, x_structural)
            else:
                x = 0.5 * (x_semantic + x_structural)
                if return_attention:
                    gate_alpha = torch.full((x.size(0), 1), 0.5, dtype=x.dtype, device=x.device)
        elif self.ablation_mode == "gat_only":
            x = x_semantic
        elif self.ablation_mode == "rgcn_only":
            x = x_structural
        
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # --- Layer 2: Deep Refinement ---
        x = self.conv2(x, edge_index)
        
        # --- Readout: Global Attention Pooling ---
        if return_attention:
            if att_weights is None:
                att_weights = {}
            if self.attention_pool is not None:
                # Calculate node attention weights manually for visualization.
                node_scores = self.attention_pool.gate_nn(x).view(-1, 1)
                from torch_geometric.utils import softmax
                node_att_weights = softmax(node_scores, batch)
                att_weights["node_attention"] = node_att_weights
            att_weights["fusion_mode"] = self.fusion_mode
            att_weights["pooling_mode"] = self.pooling_mode
            if gate_alpha is not None:
                att_weights["gate_alpha"] = gate_alpha
                att_weights["semantic_gate_mean"] = gate_alpha.mean()
                att_weights["structural_gate_mean"] = 1.0 - gate_alpha.mean()
            elif self.ablation_mode == "gat_only":
                att_weights["semantic_gate_mean"] = torch.tensor(1.0, device=x.device)
                att_weights["structural_gate_mean"] = torch.tensor(0.0, device=x.device)
            elif self.ablation_mode == "rgcn_only":
                att_weights["semantic_gate_mean"] = torch.tensor(0.0, device=x.device)
                att_weights["structural_gate_mean"] = torch.tensor(1.0, device=x.device)

        if self.attention_pool is not None:
            x = self.attention_pool(x, batch)  # [Batch_Size, Hidden_Dim]
        else:
            x = global_mean_pool(x, batch)
            
        # --- Classifier ---
        out = self.classifier(x)
        
        if return_attention:
            return out, att_weights
            
        return out
