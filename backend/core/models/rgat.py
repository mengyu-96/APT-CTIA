import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, RGCNConv, GlobalAttention
from torch_geometric.data import Data, Batch

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

    def forward(self, x_semantic, x_structural):
        # Concatenate features
        combined = torch.cat([x_semantic, x_structural], dim=-1)
        # Compute gate coefficient alpha
        alpha = self.gate_net(combined)
        # Weighted sum: alpha * Semantic + (1 - alpha) * Structural
        return alpha * x_semantic + (1 - alpha) * x_structural

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
        ablation_mode: str = "dual", # "dual", "gat_only", "rgcn_only"
        num_layers: int = 2,
        use_gatv2: bool = True
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_entity_types = num_entity_types
        self.ablation_mode = ablation_mode
        
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
            self.rgcn_conv = RGCNConv(num_node_features, hidden_dim, num_relations=self.num_relations, num_bases=30)

        # --- RGAPTive Fusion ---
        if ablation_mode == "dual":
            self.fusion = RGAPTiveFusion(hidden_dim)
        
        # --- Layer 2: Deep Processing (Shared or Specific) ---
        # We use a shared GAT layer for deeper processing after fusion
        self.conv2 = GATLayer(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)

        # --- Readout: Global Attention Pooling (Innovation Point 4) ---
        # Computes importance of each node for the graph representation
        self.attention_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        )

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
        if self.ablation_mode == "dual":
            x = self.fusion(x_semantic, x_structural)
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
             # Calculate node attention weights manually for visualization
             # GlobalAttention uses gate_nn to compute scores, then softmax
             node_scores = self.attention_pool.gate_nn(x).view(-1, 1)
             from torch_geometric.utils import softmax
             node_att_weights = softmax(node_scores, batch)
             if att_weights is None: att_weights = {}
             att_weights["node_attention"] = node_att_weights
        
        x = self.attention_pool(x, batch)  # [Batch_Size, Hidden_Dim]
            
        # --- Classifier ---
        out = self.classifier(x)
        
        if return_attention:
            return F.log_softmax(out, dim=1), att_weights
            
        return F.log_softmax(out, dim=1)
