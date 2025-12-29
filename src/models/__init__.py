# src/models - GNN and Content-based models
from .gnn import GNNBaseline
from .base_gcl import BaseGCL
from .ngcf import NGCF
from .lightgcn import LightGCN
from .lightgcl import LightGCL
from .simgcl import SimGCL
from .xsimgcl import XSimGCL
from .content_based import (
    UniversalEncoder,
    ContentBasedRecommender,
    HybridRecommender,
    TFIDFRecommender
)

from .ma_hgn import MAHGN
from .ma_hcl import MAHCL

__all__ = [
    'GNNBaseline', 
    'BaseGCL',
    'NGCF',
    'LightGCN',
    'LightGCL', 
    'SimGCL',
    'XSimGCL',
    'MAHGN',
    'MAHCL',
    'UniversalEncoder',
    'ContentBasedRecommender',
    'HybridRecommender',
    'TFIDFRecommender',
]