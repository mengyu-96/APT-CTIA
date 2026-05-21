
import torch
import json
import os
import sys
from pathlib import Path
from torch_geometric.data import Data, Batch
import pandas as pd
import torch.nn.functional as F

# Add project root
sys.path.append(r"d:\git\APT归因")

# Import model class
from train_gnn import APTAttributionGraphSAGE, SAGEConv, APTAttributionGAT, GATConv

def analyze_errors():
    graphs_path = r"d:\git\APT归因\processed_data_mixed_v1\graphs.pt"
    model_path = r"d:\git\APT归因\models_v3_pdf_mixed\best_model.pt"
    label_mapping_path = r"d:\git\APT归因\processed_data_mixed_v1\label_mapping.json"
    raw_index_path = r"d:\git\APT归因\processed_data_mixed_v1\raw_index.csv"

    # Load Data
    print("Loading data...")
    try:
        graphs = torch.load(graphs_path, weights_only=False)
    except:
        graphs = torch.load(graphs_path)
    
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    # Load Metadata for filenames
    df = pd.read_csv(raw_index_path)
    report_map = {row['report_id']: row for _, row in df.iterrows()}

    # Load Model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model structure first
    input_dim = graphs[0].x.size(1)
    num_classes = len(label_map)
    # Hyperparams from training command (GAT defaults)
    hidden_dim = 128
    # text_emb_dim = 384 # GAT doesn't use this in current impl
    
    # Try GAT first as it was default
    model = APTAttributionGAT(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        num_classes=num_classes, 
        heads=8,
        num_layers=2, 
        dropout=0.5
    )
    
    try:
        # Try loading as full model
        loaded_obj = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(loaded_obj, torch.nn.Module):
            model = loaded_obj
        elif isinstance(loaded_obj, dict):
            model.load_state_dict(loaded_obj)
        else:
            print(f"Unknown model format: {type(loaded_obj)}")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()

    print("Running inference...")
    misclassified = []
    
    with torch.no_grad():
        for i, graph in enumerate(graphs):
            # Prepare batch of 1
            batch = Batch.from_data_list([graph]).to(device)
            
            # Forward
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Pred
            probs = F.softmax(out, dim=1)
            pred_idx = out.argmax(dim=1).item()
            true_idx = graph.y.item()
            
            if pred_idx != true_idx:
                conf = probs[0][pred_idx].item()
                true_conf = probs[0][true_idx].item()
                
                rid = getattr(graph, 'report_id', 'unknown')
                meta = report_map.get(rid, {})
                filename = meta.get('raw_name', 'unknown')
                file_type = meta.get('file_type', 'unknown')
                
                misclassified.append({
                    "report_id": rid,
                    "filename": filename,
                    "file_type": file_type,
                    "true_label": idx_to_label.get(true_idx, str(true_idx)),
                    "pred_label": idx_to_label.get(pred_idx, str(pred_idx)),
                    "pred_conf": f"{conf:.4f}",
                    "true_conf": f"{true_conf:.4f}",
                    "num_nodes": graph.num_nodes,
                    "num_edges": graph.num_edges
                })

    print(f"\nFound {len(misclassified)} misclassified samples out of {len(graphs)} total.")
    
    print("\nDetailed Misclassification Report:")
    print("="*80)
    print(f"{'True Label':<15} | {'Pred Label':<15} | {'Conf':<8} | {'Type':<5} | {'Filename'}")
    print("-" * 80)
    
    for item in misclassified:
        print(f"{item['true_label']:<15} | {item['pred_label']:<15} | {item['pred_conf']:<8} | {item['file_type']:<5} | {item['filename']}")
        
    # Save to file
    with open("misclassification_analysis.json", "w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=2, ensure_ascii=False)
        
    print("\nAnalysis saved to misclassification_analysis.json")

if __name__ == "__main__":
    analyze_errors()
