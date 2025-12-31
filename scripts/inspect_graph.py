import torch
import sys

path = 'data/processed/regular_g2/graph_with_negatives.pt'
print(f"Inspecting {path}...")
try:
    data = torch.load(path, weights_only=False)
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        if 'splits' in data:
            print(f"Splits keys: {data['splits'].keys()}")
            if 'train' in data['splits']:
                print(f"Train split type: {type(data['splits']['train'])}")
                print(f"Train split keys/attrs: {dir(data['splits']['train']) if not isinstance(data['splits']['train'], dict) else data['splits']['train'].keys()}")
            if 'test' in data['splits']:
                print(f"Test split type: {type(data['splits']['test'])}")
    else:
        print("Not a dict.")
        print(data)
except Exception as e:
    print(e)
