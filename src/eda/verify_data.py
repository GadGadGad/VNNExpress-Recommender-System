import torch
import torch_geometric
from torch_geometric.data import HeteroData
import numpy as np
import networkx as nx
import argparse
from pathlib import Path

def verify_gnn_data(data_path: str):
    """
    Verify the correctness of the GNN data file with explicit checks.
    """
    path = Path(data_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return

    print(f"Loading data from {path}...")
    try:
        data = torch.load(path, weights_only=False)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    print("\n" + "="*50)
    print("DATA VERIFICATION REPORT")
    print("="*50)

    # 1. Structure Check
    is_hetero = isinstance(data, HeteroData)
    print(f"Type: {'Heterogeneous' if is_hetero else 'Homogeneous'}")
    if not is_hetero:
        print("[WARNING] Expected HeteroData for this project!")

    # 2. Node Features Check
    print("\n--- Node Features ---")
    for node_type in data.node_types:
        x = data[node_type].x
        num_nodes, num_features = x.shape
        print(f"Node Type: '{node_type}'")
        print(f"   Count: {num_nodes}")
        print(f"   Dim:   {num_features}")
        
        # Check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"   [FAIL] Features contain NaNs or Infs!")
        else:
            print(f"   [PASS] No NaNs/Infs")
            
        # Check for Zero Rows (dead nodes)
        zero_rows = (x.abs().sum(dim=1) == 0).sum().item()
        if zero_rows > 0:
            print(f"   [WARN] {zero_rows} nodes have all-zero features")
        else:
            print(f"   [PASS] All nodes have non-zero features")

    # 3. Edge & Degree Check (5-Core Verification)
    print("\n--- 5-Core Compliance Check ---")
    edge_type = ('user', 'comments', 'article')
    if edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        src, dst = edge_index[0], edge_index[1]
        
        num_users = data['user'].num_nodes
        num_articles = data['article'].num_nodes
        
        user_deg = torch.bincount(src, minlength=num_users)
        art_deg = torch.bincount(dst, minlength=num_articles)
        
        min_u_deg = user_deg.min().item()
        min_a_deg = art_deg.min().item()
        
        print(f"User Min Degree:    {min_u_deg}")
        print(f"Article Min Degree: {min_a_deg}")
        
        if min_u_deg >= 5:
            print("   [PASS] User 5-core check passed")
        else:
            print(f"   [FAIL] User 5-core check FAILED (Found users with degree {min_u_deg})")
            
        if min_a_deg >= 5:
            print("   [PASS] Article 5-core check passed")
        else:
            print(f"   [FAIL] Article 5-core check FAILED (Found articles with degree {min_a_deg})")
            
        # Check isolation
        isolated_users = (user_deg == 0).sum().item()
        isolated_articles = (art_deg == 0).sum().item()
        if isolated_users > 0 or isolated_articles > 0:
            print(f"   [WARN] Isolated nodes found: Users={isolated_users}, Articles={isolated_articles}")

    # 4. Connectivity Check
    print("\n--- Connectivity Check ---")
    try:
        # Convert to homogeneous and then undirected for connectivity check
        homo_data = data.to_homogeneous()
        g = torch_geometric.utils.to_networkx(homo_data, to_undirected=True)
        
        num_components = nx.number_connected_components(g)
        print(f"Connected Components: {num_components}")
        
        if num_components > 1:
            largest_cc = max(nx.connected_components(g), key=len)
            print(f"Size of Largest Component: {len(largest_cc)} nodes ({len(largest_cc)/g.number_of_nodes():.1%})")
            print("   [INFO] Graph is fragmented. This is common if the graph is sparse.")
        else:
            print("   [PASS] Graph is fully connected.")
    except Exception as e:
        print(f"   [SKIP] Connectivity check skipped (Error: {e})")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/processed/user_article_graph.pt')
    args = parser.parse_args()
    verify_gnn_data(args.data_path)
