
import torch
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
import sys
import os

# Add parent directory to path to import train_gnn classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_gnn import APTAttributionGraphSAGE, load_graph_dataset

def main():
    # Paths
    base_dir = Path("d:/git/APT归因/diagnose_output")
    graphs_path = base_dir / "graphs.pt"
    label_mapping_path = base_dir / "label_mapping.json"
    model_path = base_dir / "training/best_model.pt"
    output_plot = base_dir / "training/tsne_visualization.png"

    # Load data
    graphs, label_to_idx, idx_to_label = load_graph_dataset(graphs_path, label_mapping_path)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = graphs[0].x.size(1)
    num_classes = len(label_to_idx)
    
    model = APTAttributionGraphSAGE(input_dim, 128, num_classes, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get embeddings
    embeddings = []
    labels = []
    
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # We need the embedding before the final classifier
            # GraphSAGE forward: convs -> global_pool -> classifier
            # We want the output of global_pool
            
            x = batch.x
            edge_index = batch.edge_index
            
            for i, conv in enumerate(model.convs):
                x = conv(x, edge_index)
                if i < len(model.convs) - 1:
                    x = model.norms[i](x)
                    x = torch.nn.functional.relu(x)
                    # x = model.dropout(x) # No dropout in eval
            
            x = torch.nn.functional.avg_pool1d(x.unsqueeze(0).transpose(1, 2), x.size(0)).squeeze(2)
            # Wait, global_mean_pool usage in model:
            # x = global_mean_pool(x, batch)
            # Replicating logic exactly:
            from torch_geometric.nn import global_mean_pool
            # Reset x
            x = batch.x
            for i, conv in enumerate(model.convs):
                x = conv(x, edge_index)
                if i < len(model.convs) - 1:
                    x = model.norms[i](x)
                    x = torch.nn.functional.relu(x)
            
            embed = global_mean_pool(x, batch.batch)
            embeddings.append(embed.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        indices = labels == label
        class_name = idx_to_label[label]
        plt.scatter(reduced[indices, 0], reduced[indices, 1], c=[color], label=class_name, alpha=0.7, s=50)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE Visualization of GraphSAGE Embeddings")
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Saved t-SNE plot to {output_plot}")

if __name__ == "__main__":
    main()
