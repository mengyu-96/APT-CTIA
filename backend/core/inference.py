
from __future__ import annotations

import json
import logging
import sys
import io
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Import models from train.py
# Assuming backend is in python path or relative import works
try:
    from core.train import (
        APTAttributionGCN, 
        APTAttributionGAT, 
        APTAttributionTransformer, 
        APTAttributionGraphSAGE, 
        APTAttributionGIN, 
        APTAttributionHybrid,
        load_graph_dataset
    )
except ImportError:
    # Fallback for direct execution
    from train import (
        APTAttributionGCN, 
        APTAttributionGAT, 
        APTAttributionTransformer, 
        APTAttributionGraphSAGE, 
        APTAttributionGIN, 
        APTAttributionHybrid,
        load_graph_dataset
    )

try:
    from core.models.rgat import RelationAwareGAT
except ImportError:
    try:
        from models.rgat import RelationAwareGAT
    except ImportError:
        RelationAwareGAT = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger(__name__)

def load_model_for_inference(model_path: Path, config: Dict[str, Any], num_classes: int, input_dim: int, text_emb_dim: int = 0):
    """
    Load the trained model architecture and weights.
    """
    model_type = config.get('model_type', 'GAT')
    hidden_dim = int(config.get('hidden_dim', 128))
    dropout = float(config.get('dropout', 0.5))
    # heads is not always in config, default to 8 if GAT/Transformer
    heads = 8 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "GCN":
        model = APTAttributionGCN(input_dim, hidden_dim, num_classes, dropout=dropout)
    elif model_type == "GAT":
        model = APTAttributionGAT(input_dim, hidden_dim, num_classes, heads=heads, dropout=dropout)
    elif model_type == "Transformer":
        model = APTAttributionTransformer(input_dim, hidden_dim, num_classes, heads=heads, dropout=dropout)
    elif model_type == "GraphSAGE":
        model = APTAttributionGraphSAGE(input_dim, hidden_dim, num_classes, dropout=dropout, text_emb_dim=text_emb_dim)
    elif model_type == "GIN":
        model = APTAttributionGIN(input_dim, hidden_dim, num_classes, dropout=dropout)
    elif model_type == "RGAT":
        if RelationAwareGAT is None:
             raise RuntimeError("RGAT model code not available.")
        ablation_mode = config.get('ablation_mode', "dual")
        model = RelationAwareGAT(
            num_node_features=input_dim, 
            num_classes=num_classes, 
            hidden_dim=hidden_dim,
            dropout=dropout,
            ablation_mode=ablation_mode
        )
    else:
        model = APTAttributionHybrid(input_dim, hidden_dim, num_classes, model_type=model_type, heads=heads, dropout=dropout)
    
    def _extract_state_dict(blob):
        # Support different checkpoint formats
        if isinstance(blob, dict):
            for key in ["state_dict", "model_state_dict", "weights"]:
                if key in blob and isinstance(blob[key], dict):
                    return blob[key]
        return blob

    def _compat_remap_rgat_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Backward-compat remapping for older checkpoints whose module names differ:
        - 'conv1.*'         -> 'gat_conv.*'
        - 'rgcn_branch.*'   -> 'rgcn_conv.*'
        - 'fusion_gate.*'   -> skipped (layout differs; will use randomly initialised gate_net)
        Other keys are kept as-is.
        """
        remapped: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            nk = k
            if k.startswith("conv1."):
                nk = "gat_conv." + k[len("conv1."):]
            elif k.startswith("rgcn_branch."):
                nk = "rgcn_conv." + k[len("rgcn_branch."):]
            elif k.startswith("fusion_gate."):
                # Skip mapping legacy single-layer gate to new two-layer gate_net
                continue
            remapped[nk] = v
        return remapped
    
    def _filter_compatible_keys(sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Filter out keys whose shapes don't match current model parameters/buffers.
        This prevents size-mismatch RuntimeError while still using compatible weights.
        """
        model_sd = model.state_dict()
        kept: Dict[str, torch.Tensor] = {}
        dropped: list[tuple[str, tuple, tuple | None]] = []
        for k, v in sd.items():
            if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape):
                kept[k] = v
            else:
                exp_shape = tuple(model_sd[k].shape) if k in model_sd else None
                dropped.append((k, tuple(v.shape), exp_shape))
        if dropped:
            msg_preview = ", ".join([f"{k}: {src}->{dst}" for k, src, dst in dropped[:10]])
            LOGGER.warning("Dropped %d incompatible weight(s): %s%s",
                           len(dropped), msg_preview, " ..." if len(dropped) > 10 else "")
        return kept

    try:
        raw_blob = torch.load(model_path, map_location=device)
        state_dict = _extract_state_dict(raw_blob)
        # RGAT compatibility mapping
        if config.get("model_type", "GAT") == "RGAT":
            state_dict = _compat_remap_rgat_keys(state_dict)
        # Filter by shape to avoid size mismatch errors
        state_dict = _filter_compatible_keys(state_dict, model)
        incompatible = model.load_state_dict(state_dict, strict=False)
        try:
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
        except Exception:
            missing, unexpected = [], []
        if missing or unexpected:
            LOGGER.warning("Loaded model with non-strict keys. Missing: %s | Unexpected: %s", missing, unexpected)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {model_path}: {e}")

    model = model.to(device)
    model.eval()
    return model, device

def run_inference_pipeline(
    model_dir: str | Path,
    dataset_dir: str | Path,
    output_dir: Optional[str | Path] = None
) -> Dict[str, Any]:
    """
    Run inference on a dataset using a trained model.
    
    Args:
        model_dir: Directory containing the trained model (results.json, best_model.pt).
        dataset_dir: Directory containing the dataset to infer (graphs.pt).
        output_dir: Optional directory to save inference results.
        
    Returns:
        Dict containing inference results and statistics.
    """
    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    
    # 1. Load Model Config
    results_path = model_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Model config not found at {results_path}")
        
    with open(results_path, 'r', encoding='utf-8') as f:
        train_results = json.load(f)
        config = train_results.get('config', {})
        
    # 2. Load Dataset (Graphs)
    # Note: We need to use the SAME label mapping as training if we want to map back to names.
    # Usually, the label mapping is stored with the PROCESSED data that was used for training.
    # But here we are inferring on potentially NEW data.
    # The model outputs indices 0..N-1. We need to know what class '0' corresponds to.
    # This information should have been saved during training.
    # Let's check if `train.py` saved `idx_to_label`.
    # It seems `train.py` saves `classification_report` which has target names, but maybe not a direct map file.
    # However, `load_graph_dataset` returns `idx_to_label`.
    # We should look for `label_mapping.json` in the training data directory, OR rely on what's in `results.json` keys.
    # `results.json` has `classification_report` keys which are label names.
    
    # To be safe, we need the mapping used during training.
    # We will assume the model was trained on classes sorted alphabetically (default behavior of load_graph_dataset).
    # So if we have the list of class names, we can reconstruct the mapping.
    
    class_names = []
    if 'classification_report' in train_results:
        # keys like "APT28", "APT29", ... and "accuracy", "macro avg"...
        # We need to filter.
        keys = list(train_results['classification_report'].keys())
        class_names = [k for k in keys if k not in ['accuracy', 'macro avg', 'weighted avg']]
        class_names.sort() # Ensure sorted order as per train.py logic
    
    # 3. Load Graphs to Infer
    # Pass the dataset directory directly to load_graph_dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")
        
    # We don't need label mapping for the INPUT dataset (we ignore its labels if any), 
    # but we need to know the input dimension.
    try:
        graphs, _, _ = load_graph_dataset(dataset_dir) # We ignore the label mapping of the new dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load graphs from {dataset_dir}: {e}")
    
    if not graphs:
        raise ValueError("No valid graphs found in dataset.")

    input_dim = graphs[0].x.size(1)
    
    text_emb_dim = 0
    if hasattr(graphs[0], 'doc_emb') and graphs[0].doc_emb is not None:
        text_emb_dim = graphs[0].doc_emb.size(1)
        
    num_classes = len(class_names) if class_names else 0
    
    # If we couldn't determine num_classes from results.json, we might have a problem.
    # But let's assume we can. If class_names is empty, we might need to fallback or error.
    if num_classes == 0:
        # Try to infer from model weights shape if possible, but that requires loading model first.
        # Let's try to load model and check classifier output size.
        pass

    # 4. Load Model
    model_path = model_dir / "best_model.pt"
    # We need to instantiate the model first.
    # If we don't know num_classes, we can't instantiate the final layer correctly.
    # Let's assume standard logic: 
    # If we really can't find class names, we check if there is a `label_mapping.json` in the processed data path used for training.
    train_data_path = config.get('processed_data_path')
    if not class_names and train_data_path:
        train_data_path = Path(train_data_path)
        # Priority 1: Inside dataset folder
        train_map_path = train_data_path / "label_mapping.json"
        if not train_map_path.exists():
            # Priority 2: Parent folder
            train_map_path = train_data_path.parent / "label_mapping.json"
            
        if train_map_path.exists():
             with open(train_map_path, 'r', encoding='utf-8') as f:
                 mapping = json.load(f)
                 # Sort by index to get correct order of names
                 # mapping is {label: idx}
                 sorted_items = sorted(mapping.items(), key=lambda x: x[1])
                 class_names = [item[0] for item in sorted_items]
                 num_classes = len(class_names)
    
    if num_classes == 0:
         raise RuntimeError("Could not determine number of classes from model config or results.")

    model, device = load_model_for_inference(model_path, config, num_classes, input_dim, text_emb_dim)
    
    # 5. Run Inference
    batch_size = 32
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_attentions = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            doc_emb = None
            if hasattr(batch, 'doc_emb'):
                doc_emb = batch.doc_emb
            
            att_data = None
            
            if isinstance(model, APTAttributionGraphSAGE):
                out = model(batch.x, batch.edge_index, batch.batch, doc_emb=doc_emb)
            elif RelationAwareGAT is not None and isinstance(model, RelationAwareGAT):
                out, att_data = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
                
            probs = F.softmax(out, dim=1)
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Process attention for this batch
            if att_data:
                edge_idx_full = att_data.get("edge_index")
                edge_att_full = att_data.get("edge_attention")
                node_att_full = att_data.get("node_attention")
                
                for i in range(batch.num_graphs):
                    node_mask = (batch.batch == i)
                    if node_att_full is not None:
                        g_node_att = node_att_full[node_mask].cpu().numpy().flatten().tolist()
                    else:
                        g_node_att = []
                        
                    if edge_idx_full is not None and edge_att_full is not None:
                         src_nodes = edge_idx_full[0]
                         edge_graph_indices = batch.batch[src_nodes]
                         edge_mask = (edge_graph_indices == i)
                         
                         # Mean over heads
                         g_edge_att = edge_att_full[edge_mask].mean(dim=1).cpu().numpy().tolist()
                         
                         min_node_idx = torch.where(node_mask)[0].min()
                         g_edge_index = (edge_idx_full[:, edge_mask] - min_node_idx).cpu().numpy().tolist()
                    else:
                         g_edge_att = []
                         g_edge_index = []
                         
                    all_attentions.append({
                        "node_attention": g_node_att,
                        "edge_attention": g_edge_att,
                        "edge_index": g_edge_index
                    })
            else:
                all_attentions.extend([{}] * batch.num_graphs)

    # Re-align report IDs
    # We iterate graphs in order
    inference_results = []
    for i, graph in enumerate(graphs):
        rid = getattr(graph, 'report_id', f"unknown_{i}")
        pred_idx = all_preds[i]
        prob_vec = all_probs[i]
        confidence = float(prob_vec[pred_idx])
        pred_label = class_names[pred_idx]
        
        # Top 3
        top3_indices = prob_vec.argsort()[-3:][::-1]
        top3 = [
            {"label": class_names[idx], "confidence": float(prob_vec[idx])}
            for idx in top3_indices
        ]
        
        result_entry = {
            "report_id": rid,
            "predicted_label": pred_label,
            "confidence": confidence,
            "top3": top3,
            "attention_data": all_attentions[i] if i < len(all_attentions) else {}
        }
        inference_results.append(result_entry)
        
    # 6. Aggregate Stats
    label_counts = {}
    for res in inference_results:
        lbl = res['predicted_label']
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
    summary = {
        "total_samples": len(inference_results),
        "label_distribution": label_counts,
        "results": inference_results
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "inference_results.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
    return summary
