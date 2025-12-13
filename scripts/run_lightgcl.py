"""
Run LightGCL Training
=====================

Usage:
    python scripts/run_lightgcl.py
    python scripts/run_lightgcl.py --config configs/lightgcl_config.yaml
    python scripts/run_lightgcl.py --force_reload --device cuda
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import argparse
import yaml
import numpy as np

from src.data.dataloader_lightgcl import load_data
from src.models.lightgcl import LightGCL
from src.training.trainer_lightgcl import LightGCLTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='LightGCL Training')
    parser.add_argument('--config', type=str, 
                        default='configs/lightgcl_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--force_reload', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    return parser.parse_args()


def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Config not found: {config_path}, using defaults")
        return {}


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    loss_config = config.get('loss', {})
    train_config = config.get('training', {})
    eval_config = config.get('evaluation', {})
    
    if args.data_path:
        data_config['path'] = args.data_path
    if args.force_reload:
        data_config['force_reload'] = True
    if args.epochs:
        train_config['n_epochs'] = args.epochs
    if args.batch_size:
        train_config['batch_size'] = args.batch_size
    if args.lr:
        train_config['lr'] = args.lr
        
    # Device
    device_str = args.device or config.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("LightGCL for VnExpress Recommendation")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Set seed
    seed = train_config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load data
    n_users, n_items, train_data, train_dict, test_data, loader = load_data(
        data_path=data_config.get('path', 'data'),
        min_user_interactions=data_config.get('min_user_interactions', 2),
        min_article_interactions=data_config.get('min_article_interactions', 2),
        test_ratio=data_config.get('test_ratio', 0.2),
        force_reload=data_config.get('force_reload', False),
        seed=seed
    )
    
    print(f"\nDataset Summary:")
    print(f"  Users: {n_users:,}")
    print(f"  Items: {n_items:,}")
    print(f"  Train: {len(train_data):,}")
    print(f"  Test users: {len(test_data)}")
    
    # Initialize model
    model = LightGCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=model_config.get('embedding_dim', 64),
        n_layers=model_config.get('n_layers', 3),
        svd_q=model_config.get('svd_q', 5),
        dropout=model_config.get('dropout', 0.0),
        reg_weight=loss_config.get('reg_weight', 1e-4),
        ssl_weight=loss_config.get('ssl_weight', 0.1),
        temp=loss_config.get('temperature', 0.2)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.get('lr', 0.001)
    )
    
    # Trainer
    trainer = LightGCLTrainer(model, optimizer, device, n_users, n_items)
    
    # Create adjacency matrix
    trainer.create_adj_matrix(train_data)
    
    # Save path
    save_path = config.get('save_path', 'checkpoints/lightgcl_best.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Train
    best_metrics = trainer.train(
        train_data=train_data,
        train_dict=train_dict,
        test_data=test_data,
        n_epochs=train_config.get('n_epochs', 100),
        batch_size=train_config.get('batch_size', 2048),
        eval_every=train_config.get('eval_every', 5),
        patience=train_config.get('patience', 100),
        save_path=save_path
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    if save_path and os.path.exists(save_path):
        try:
            trainer.load_model(save_path)
        except Exception as e:
            print(f"  Warning: Could not load saved model: {e}")
            print("  Using current model state for evaluation.")
        
    final_metrics = trainer.evaluate(
        test_data, train_dict,
        k_list=eval_config.get('k_list', [10, 20, 50])
    )
    
    
    print("\nFinal Results:")
    for k, v in sorted(final_metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()