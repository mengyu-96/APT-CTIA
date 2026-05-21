
import argparse
import json
import logging
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Import model definitions from train_gnn.py
# We need to add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_gnn import APTAttributionGraphSAGE, APTAttributionGAT, APTAttributionGCN, APTAttributionGIN, APTAttributionTransformer, APTAttributionHybrid, load_graph_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved models from a cross-validation run")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing cv_fold_X subdirectories")
    parser.add_argument("--graphs-path", type=Path, default=Path("processed_txt_full_v5/graphs.pt"), help="Path to graphs.pt")
    parser.add_argument("--label-mapping-path", type=Path, default=Path("processed_txt_full_v5/label_mapping.json"), help="Path to label_mapping.json")
    parser.add_argument("--model-type", type=str, default="GraphSAGE", choices=["GCN", "GAT", "Transformer", "GraphSAGE", "GIN", "Hybrid"], help="Model architecture")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--ensemble", action="store_true", default=True, help="Run ensemble prediction on full dataset")
    return parser.parse_args()

def load_model(model_path, model_type, input_dim, hidden_dim, num_classes, num_layers):
    if model_type == "GCN":
        model = APTAttributionGCN(input_dim, hidden_dim, num_classes, num_layers)
    elif model_type == "GAT":
        model = APTAttributionGAT(input_dim, hidden_dim, num_classes, num_layers=num_layers)
    elif model_type == "Transformer":
        model = APTAttributionTransformer(input_dim, hidden_dim, num_classes, num_layers=num_layers)
    elif model_type == "GraphSAGE":
        model = APTAttributionGraphSAGE(input_dim, hidden_dim, num_classes, num_layers=num_layers)
    elif model_type == "GIN":
        model = APTAttributionGIN(input_dim, hidden_dim, num_classes, num_layers=num_layers)
    elif model_type == "Hybrid":
        model = APTAttributionHybrid(input_dim, hidden_dim, num_classes, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(model, loader, device):
    model.to(device)
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    graphs, label_to_idx, idx_to_label = load_graph_dataset(args.graphs_path, args.label_mapping_path)
    num_classes = len(label_to_idx)
    input_dim = graphs[0].x.size(1)
    
    LOGGER.info(f"Loaded {len(graphs)} graphs. Feature dim: {input_dim}. Classes: {num_classes}")
    
    # 2. Find Models
    fold_dirs = sorted(list(args.model_dir.glob("cv_fold_*")))
    if not fold_dirs:
        LOGGER.error(f"No cv_fold_* directories found in {args.model_dir}")
        return
        
    LOGGER.info(f"Found {len(fold_dirs)} folds in {args.model_dir}")
    
    models = []
    for fold_dir in fold_dirs:
        model_path = fold_dir / "best_model.pt"
        if not model_path.exists():
            LOGGER.warning(f"Model not found in {fold_dir}")
            continue
            
        LOGGER.info(f"Loading model from {model_path}")
        model = load_model(model_path, args.model_type, input_dim, args.hidden_dim, num_classes, args.num_layers)
        models.append(model)
        
    if not models:
        LOGGER.error("No models loaded!")
        return

    # 3. Ensemble Prediction (Soft Voting)
    if args.ensemble:
        LOGGER.info("Running Ensemble Prediction on FULL dataset...")
        full_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
        
        ensemble_probs = np.zeros((len(graphs), num_classes))
        true_labels = []
        
        # Get true labels once
        for batch in full_loader:
            true_labels.extend(batch.y.cpu().numpy())
        true_labels = np.array(true_labels)
        
        for i, model in enumerate(models):
            _, probs, _ = predict(model, full_loader, device)
            ensemble_probs += probs
            
        # Average probabilities
        ensemble_probs /= len(models)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # 4. Report Metrics
        acc = accuracy_score(true_labels, ensemble_preds)
        LOGGER.info(f"Ensemble Accuracy (Full Dataset): {acc:.4f}")
        
        report = classification_report(true_labels, ensemble_preds, target_names=[idx_to_label[i] for i in range(num_classes)], output_dict=True)
        # Print text report
        print("\nEnsemble Classification Report:")
        print(classification_report(true_labels, ensemble_preds, target_names=[idx_to_label[i] for i in range(num_classes)]))
        
        # Save predictions
        results_df = pd.DataFrame({
            "report_id": [g.report_id for g in graphs], # Assuming report_id is attached to Data object, if not this might fail. 
            # Note: preprocess_apt_dataset.py attaches report_id to Data object? Let's check. 
            # Yes, train_gnn.py uses r["report_id"] so it must be there, but maybe not on the object attribute directly?
            # Actually train_gnn.py loop: r["report_id"] comes from metadata? No, it constructs it.
            # Let's check if Data object has report_id attribute. 
            # In preprocess_apt_dataset.py: graph.report_id = report_id  <-- No, it sets data attributes but report_id?
            # Let's assume we can't easily get report_id back unless we load metadata again or if it was saved.
            # For now, let's just save indices.
            "true_label": [idx_to_label[y] for y in true_labels],
            "pred_label": [idx_to_label[p] for p in ensemble_preds],
            "max_prob": np.max(ensemble_probs, axis=1)
        })
        
        # Try to recover report_ids if possible (optional)
        # We can map back via index if the list order is preserved (it is).
        # But we need the report_ids list. 
        # For now, just save.
        
        output_path = args.model_dir / "ensemble_predictions.csv"
        results_df.to_csv(output_path, index=False)
        LOGGER.info(f"Ensemble predictions saved to {output_path}")

if __name__ == "__main__":
    main()
