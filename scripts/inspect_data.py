
import torch
import os

data_path = 'data/processed/strict_g1'
graph_path = os.path.join(data_path, 'user_article_graph.pt')
split_path = os.path.join(data_path, 'split_indices_random.pt')

print(f"Loading {graph_path}...")
try:
    graph_data = torch.load(graph_path, weights_only=False)
    print("Graph Data Type:", type(graph_data))
    if isinstance(graph_data, dict) and 'graph' in graph_data:
        data = graph_data['graph']
        print("HeteroData Keys:", data.keys) # Node/Edge types
        
        # Access specific edge storage
        # HeteroData keys are usually tuples or strings
        try:
            edge_store = data[('user', 'comments', 'article')]
            print("Edge Store Keys:", edge_store.keys())
            print("edge_index shape:", edge_store.edge_index.shape)
            if 'test_edge_index' in edge_store:
                 print("test_edge_index shape:", edge_store.test_edge_index.shape)
        except Exception as e:
            print("Could not access ('user', 'comments', 'article'):", e)
            
        splits = graph_data.get('splits')
        if isinstance(splits, dict):
            print("Splits Keys:", splits.keys())
            for k, v in splits.items():
               if hasattr(v, 'shape'): print(f"  Split {k}: {v.shape}")
               elif isinstance(v, list): print(f"  Split {k}: len={len(v)}")

    else:
        print("Graph Data:", graph_data)
except Exception as e:
    print(f"Error loading graph: {e}")

print(f"\nLoading {split_path}...")
try:
    split_data = torch.load(split_path, weights_only=False)
    print("Split Data Type:", type(split_data))
    if isinstance(split_data, dict):
        print("Split Keys:", split_data.keys())
        for k, v in split_data.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
            elif isinstance(v, list):
                print(f"  {k}: len={len(v)}")
            else:
                print(f"  {k}: {type(v)}")
except Exception as e:
    print(f"Error loading split: {e}")
