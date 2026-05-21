
import torch
from torch_geometric.data import Data
import json
import os
import sys

def verify_graphs(graphs_path, label_mapping_path):
    print(f"Loading graphs from {graphs_path}...")
    if not os.path.exists(graphs_path):
        print(f"Error: File not found: {graphs_path}")
        return

    try:
        graphs = torch.load(graphs_path, weights_only=False)
    except TypeError:
        try:
            graphs = torch.load(graphs_path)
        except Exception as e:
            print(f"Error loading graphs: {e}")
            return
    except Exception as e:
        print(f"Error loading graphs: {e}")
        return
    
    print(f"Loaded {len(graphs)} graphs.")
    
    if len(graphs) == 0:
        print("Error: No graphs loaded.")
        return

    sample_graph = graphs[0]
    print("\nSample Graph Properties:")
    print(f"  Nodes: {sample_graph.num_nodes}")
    print(f"  Edges: {sample_graph.num_edges}")
    if sample_graph.x is not None:
        print(f"  Node Features (x) shape: {sample_graph.x.shape}")
    else:
        print("  Node Features (x) is None")
    
    if hasattr(sample_graph, 'doc_emb') and sample_graph.doc_emb is not None:
        print(f"  Document Embedding (doc_emb) shape: {sample_graph.doc_emb.shape}")
    else:
        print("  Document Embedding (doc_emb) NOT FOUND or None!")

    # Check for empty graphs
    empty_graphs = [i for i, g in enumerate(graphs) if g.num_nodes == 0]
    if empty_graphs:
        print(f"\nWarning: Found {len(empty_graphs)} empty graphs at indices: {empty_graphs}")
    
    # Check label distribution
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        
        idx_to_label = {v: k for k, v in label_mapping.items()}
        
        label_counts = {}
        for g in graphs:
            if g.y is not None:
                label_idx = g.y.item()
                label_name = idx_to_label.get(label_idx, f"Unknown-{label_idx}")
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
            else:
                label_counts["No Label"] = label_counts.get("No Label", 0) + 1
            
        print("\nLabel Distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")
    else:
        print(f"Warning: Label mapping file not found at {label_mapping_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        graphs_file = sys.argv[1]
        mapping_file = os.path.join(os.path.dirname(graphs_file), "label_mapping.json")
    else:
        graphs_file = r"d:\git\APT归因\processed_data_txt_only_v2\graphs.pt"
        mapping_file = r"d:\git\APT归因\processed_data_txt_only_v2\label_mapping.json"
    verify_graphs(graphs_file, mapping_file)
