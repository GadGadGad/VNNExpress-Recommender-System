import torch
import pandas as pd
from pathlib import Path
from torch_geometric.data import HeteroData
import argparse

def check_leakage(data_dir):
    data_path = Path(data_dir)
    print(f"\n[{data_path}] Checking for data leakage...")
    
    # 1. Load the Split info (Ground Truth)
    split_path = data_path / 'graph_with_negatives.pt'
    if not split_path.exists():
        print(f"Skipping: Split file not found at {split_path}")
        return

    print(f"Loading splits from {split_path}...")
    try:
        splits = torch.load(split_path, weights_only=False)
    except Exception as e:
        print(f"Error loading splits: {e}")
        return

    # Extract disjoint test pairs from the split file
    test_edges = None
    
    try:
        # Case 1: splits is a dictionary (common in G1/Regular G2 if dictionary wrapper used)
        if isinstance(splits, dict):
             # Try to find test split in splits['splits']['test']
             if 'splits' in splits and 'test' in splits['splits']:
                 test_data = splits['splits']['test']
                 if isinstance(test_data, dict):
                     if 'edge_index' in test_data:
                         test_edges = test_data['edge_index']
                     elif 'pos_users' in test_data and 'pos_articles' in test_data:
                         # Reconstruct edge_index from separate tensors (G1 format)
                         u = test_data['pos_users']
                         v = test_data['pos_articles']
                         test_edges = torch.stack([u, v], dim=0)
                 elif hasattr(test_data, 'edge_index'):
                     test_edges = test_data.edge_index

             # Try to find test split in splits['graph'] (G2 format sometimes)
             if test_edges is None and 'graph' in splits:
                  g = splits['graph']
                  if hasattr(g, 'edge_index_dict') and ('user', 'comments', 'article') in g.edge_index_dict:
                      qt = g['user', 'comments', 'article']
                      if hasattr(qt, 'test_mask') and qt.test_mask is not None:
                          test_edges = qt.edge_index[:, qt.test_mask]

        # Case 2: splits is a HeteroData object (standard PyG)
        else:
             if hasattr(splits, 'node_types') and 'user' in splits.node_types:
                 # Check strict location
                 if ('user', 'comments', 'article') in splits.edge_index_dict: 
                      qt = splits['user', 'comments', 'article']
                      if hasattr(qt, 'test_mask') and qt.test_mask is not None:
                          test_edges = qt.edge_index[:, qt.test_mask]
                      
    except Exception as e:
        print(f"Error extracting test edges: {e}")

    if test_edges is None:
        print("   [Error] Could not find test edges or test_mask in split file.")
        if isinstance(splits, dict):
            print(f"   [Debug] Keys: {list(splits.keys())}")
        return

    test_pairs = set(zip(test_edges[0].tolist(), test_edges[1].tolist()))
    print(f"Found {len(test_pairs)} test pairs.")
    
    # 2. Check Graph Files
    graph_files = ['full_hetero_graph.pt', 'category_graph.pt', 'graph_with_negatives.pt']
    
    for g_file in graph_files:
        g_path = data_path / g_file
        if not g_path.exists():
            continue
            
        print(f"Checking graph: {g_file}...")
        try:
            graph = torch.load(g_path, weights_only=False)
        except Exception as e:
            print(f"   [Error] Failed to load {g_file}: {e}")
            continue
        
        target_edges = None
        
        # Unpack if wrapper dict
        if isinstance(graph, dict) and 'graph' in graph:
             graph = graph['graph'] 

        if hasattr(graph, 'edge_index_dict'):
             if ('user', 'comments', 'article') in graph.edge_index_dict:
                store = graph['user', 'comments', 'article']
                
                # If checking the split file itself (graph_with_negatives.pt)
                if g_file == 'graph_with_negatives.pt':
                     if hasattr(store, 'train_mask') and store.train_mask is not None:
                         target_edges = store.edge_index[:, store.train_mask]
                     else:
                         print("   [INFO] graph_with_negatives contains all edges. Checking Train Mask intersection...")
                         # Can't easily verify without mask
                         continue
                else:
                     target_edges = store.edge_index
        
        if target_edges is not None:
            leak_count = 0
            for i in range(target_edges.shape[1]):
                u, i_item = target_edges[0, i].item(), target_edges[1, i].item()
                if (u, i_item) in test_pairs:
                    leak_count += 1
            
            leak_rate = (leak_count / len(test_pairs)) * 100
            if leak_count > 0:
                print(f"   [CRITICAL] LEAKAGE DETECTED in {g_file}!")
                print(f"   {leak_count} / {len(test_pairs)} test pairs found in graph ({leak_rate:.2f}%)")
            else:
                print(f"   [OK] 0 leakage in user-article edges.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to check")
    args = parser.parse_args()
    
    check_leakage(args.data_path)
