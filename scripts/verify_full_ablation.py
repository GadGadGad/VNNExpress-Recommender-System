
import os
import subprocess
import shutil
from pathlib import Path

# Config matching run_ablation.sh
CF_MODELS = ["simgcl", "xsimgcl", "lightgcl", "ma-hcl", "ma_hgn", "sim-mahgn", "ngcf"]
# Skipping CB_MODELS for this test as they are different script
GRAPHS = ["test_g1", "test_g2", "test_g3"]
PROTOCOLS = ["cold"] # simplified to just 'cold' for quick check, or ['cold', 'full']

def setup_test_data():
    print(">>> Setting up test data...")
    # Clean up
    for g in GRAPHS:
        shutil.rmtree(f"data/processed/{g}", ignore_errors=True)
        os.makedirs(f"data/processed/{g}", exist_ok=True)
    
    # We rely on train_cf_models loading raw data fallback if files don't exist
    # So we just ensure the directories exist.
    # Actually, for 'test_g2', we want to simulate proper hetero cache if possible,
    # but the script will generate it from raw csvs if missing.
    pass

def run_command(cmd):
    try:
        # Run with timeout and capture output
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr + "\n" + e.stdout
    except Exception as e:
        return False, str(e)

def main():
    setup_test_data()
    
    failures = []
    successes = 0
    
    print(f"Testing {len(CF_MODELS) * len(GRAPHS) * len(PROTOCOLS)} combinations...")
    
    for protocol in PROTOCOLS:
        for model in CF_MODELS:
            for graph in GRAPHS:
                print(f"Testing {model} on {graph} ({protocol})...", end=" ", flush=True)
                
                # Mock data path
                data_path = f"data/processed/{graph}"
                
                # Command
                cmd = (
                    f"python scripts/train_cf_models.py "
                    f"--model {model} "
                    f"--data-path {data_path} "
                    f"--epochs 1 "
                    f"--patience 1 "
                    f"--batch-size 4096 " # Large batch to finish quickly
                    f"--eval-protocol {protocol} "
                )
                
                passed, output = run_command(cmd)
                
                if passed:
                    print("✅ PASSED")
                    successes += 1
                else:
                    print("❌ FAILED")
                    failures.append(f"{model} on {graph} ({protocol}):\n{output[-1000:]}")
                    
    print("\n" + "="*50)
    print(f"Verification Complete: {successes} Passed, {len(failures)} Failed")
    
    if len(failures) > 0:
        print("\nFailures:")
        for f in failures:
            print("-" * 20)
            print(f)
            
if __name__ == "__main__":
    main()
