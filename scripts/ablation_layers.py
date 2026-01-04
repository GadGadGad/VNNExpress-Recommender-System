import os
import subprocess
import sys

# Configuration
model = "ma_hgn"
graph_type = "hetero"
data_path = "data/processed/strict_g2"
protocol = "full"
epochs = 50
layers_list = [1, 2, 3, 4]

print(f"Starting Layer Ablation for {model.upper()} on {protocol.upper()} protocol...")
print(f"Data: {data_path}")
print(f"Layers to test: {layers_list}")

results = {}

for layers in layers_list:
    print(f"\n\n{'='*60}")
    print(f"Training with {layers} LAYERS")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "scripts/train_cf_models.py",
        "--model", model,
        "--data-path", data_path,
        "--graph-type", graph_type,
        "--eval-protocol", protocol,
        "--epochs", str(epochs),
        "--n-layers", str(layers),
        "--save-results", f"results/ablation_layer_{layers}.json",
        "--batch-size", "2048"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished {layers} layers.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to train with {layers} layers.")
        print(e)

print("\nAblation Study Completed!")
