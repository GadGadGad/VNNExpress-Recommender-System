# src/training - Training and evaluation utilities
from .trainer import train_model
from .metrics import evaluate_top_k

__all__ = ['train_model', 'evaluate_top_k']
