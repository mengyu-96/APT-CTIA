
import torch
from torch_geometric.data import Data
import json
from pathlib import Path

def verify_graphs(graphs_path, label_mapping_path):
    print(f"Loading graphs from {graphs_path}...")
    try:
        graphs = torch.load(graphs_path, weights_only=False)
    except TypeError:
        graphs = torch.load(graphs_path)
    
    print(f"Loaded {len(graphs)} graphs.")
    
    if len(graphs) == 0:
        print("Error: No graphs loaded.")
        return

    sample_graph = graphs[0]
    print("\nSample Graph Properties:")
    print(f"  Nodes: {sample_graph.num_nodes}")
    print(f"  Edges: {sample_graph.num_edges}")
    print(f"  Node Features (x) shape: {sample_graph.x.shape}")
    
    if hasattr(sample_graph, 'doc_emb'):
        print(f"  Document Embedding (doc_emb) shape: {sample_graph.doc_emb.shape}")
    else:
        print("  Document Embedding (doc_emb) NOT FOUND!")

    # Check for empty graphs
    empty_graphs = [i for i, g in enumerate(graphs) if g.num_nodes == 0]
    if empty_graphs:
        print(f"\nWarning: Found {len(empty_graphs)} empty graphs at indices: {empty_graphs}")
    
    # Check label distribution
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    idx_to_label = {v: k for k, v in label_mapping.items()}
    
    label_counts = {}
    for g in graphs:
        label_idx = g.y.item()
        label_name = idx_to_label.get(label_idx, f"Unknown-{label_idx}")
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
    print("\nLabel Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")

if __name__ == "__main__":
    graphs_file = r"d:\git\APT归因\processed_data_txt_enhanced\graphs.pt"
    mapping_file = r"d:\git\APT归因\processed_data_txt_enhanced\label_mapping.json"
    verify_graphs(graphs_file, mapping_file)
