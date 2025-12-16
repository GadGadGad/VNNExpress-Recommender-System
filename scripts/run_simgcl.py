"""
Run SimGCL Training
===================

Usage:
    python scripts/run_simgcl.py
    python scripts/run_simgcl.py --config configs/simgcl_config.yaml
    python scripts/run_simgcl.py --eps 0.2 --ssl_weight 0.3
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import argparse
import yaml
import numpy as np

from src.data.dataloader_lightgcl import load_data
from src.models.simgcl import SimGCL
from src.training.trainer_simgcl import SimGCLTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='SimGCL Training')
    parser.add_argument('--config', type=str, 
                        default='configs/simgcl_config.yaml')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--force_reload', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--eps', type=float, default=None, help='Noise magnitude')
    parser.add_argument('--ssl_weight', type=float, default=None)
    return parser.parse_args()


def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    print(f"Config not found: {config_path}, using defaults")
    return {}


def main():
    args = parse_args()
    config = load_config(args.config)
    
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    loss_config = config.get('loss', {})
    train_config = config.get('training', {})
    eval_config = config.get('evaluation', {})
    
    # Override with command line args
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
    if args.patience:
        train_config['patience'] = args.patience
    if args.eval_every:
        train_config['eval_every'] = args.eval_every
    if args.eps:
        model_config['eps'] = args.eps
    if args.ssl_weight:
        loss_config['ssl_weight'] = args.ssl_weight
        
    device_str = args.device or config.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("SimGCL for VnExpress Recommendation")
    print("=" * 60)
    print(f"Device: {device}")
    
    seed = train_config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load data (reuse from LightGCL)
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
    model = SimGCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=model_config.get('embedding_dim', 64),
        n_layers=model_config.get('n_layers', 3),
        eps=model_config.get('eps', 0.1),
        dropout=model_config.get('dropout', 0.0),
        reg_weight=loss_config.get('reg_weight', 1e-4),
        ssl_weight=loss_config.get('ssl_weight', 0.2),
        temp=loss_config.get('temperature', 0.2)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.get('lr', 0.001)
    )
    
    trainer = SimGCLTrainer(model, optimizer, device, n_users, n_items)
    trainer.create_adj_matrix(train_data)
    
    save_path = config.get('save_path', 'checkpoints/simgcl_best.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    best_metrics = trainer.train(
        train_data=train_data,
        train_dict=train_dict,
        test_data=test_data,
        n_epochs=train_config.get('n_epochs', 200),
        batch_size=train_config.get('batch_size', 2048),
        eval_every=train_config.get('eval_every', 5),
        patience=train_config.get('patience', 50),
        save_path=save_path
    )
    
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    if save_path and os.path.exists(save_path):
        try:
            trainer.load_model(save_path)
        except Exception as e:
            print(f"  Warning: Could not load saved model: {e}")
        
    final_metrics = trainer.evaluate(
        test_data, train_dict,
        k_list=eval_config.get('k_list', [10, 20, 50])
    )
    
    print("\nFinal Results:")
    for k, v in sorted(final_metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()