import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

class ResidualVectorQuantizer(nn.Module):
    """
    Simplified Residual Vector Quantizer for Semantic ID generation.
    """
    def __init__(self, input_dim, n_codebooks=3, codebook_size=32):
        super(ResidualVectorQuantizer, self).__init__()
        self.input_dim = input_dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        
        # Codebooks for each residue stage
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, input_dim))
            for _ in range(n_codebooks)
        ])
        for cb in self.codebooks:
            nn.init.xavier_uniform_(cb)

    def forward(self, x):
        """
        x: (n_items, input_dim) - continuous embeddings
        Returns: 
            indices: (n_items, n_codebooks) - discrete IDs
            quantized: (n_items, input_dim) - reconstructed embeddings
        """
        n_items = x.size(0)
        indices = []
        quantized = torch.zeros_like(x)
        residual = x
        
        for i in range(self.n_codebooks):
            cb = self.codebooks[i]
            # Compute distances: (n_items, 1, dim) - (1, size, dim)
            dists = torch.cdist(residual.unsqueeze(1), cb.unsqueeze(0)).squeeze(1)
            
            # Find closest codewords
            idx = torch.argmin(dists, dim=1)
            indices.append(idx)
            
            # Get quantized vectors
            q = cb[idx]
            quantized = quantized + q
            residual = residual - q
            
        return torch.stack(indices, dim=1), quantized

    def fit_kmeans(self, x_np):
        """Initialize codebooks using KMeans for better starting point."""
        residual = x_np
        for i in range(self.n_codebooks):
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10)
            kmeans.fit(residual)
            self.codebooks[i].data.copy_(torch.from_numpy(kmeans.cluster_centers_))
            
            # Update residual
            labels = kmeans.labels_
            q = kmeans.cluster_centers_[labels]
            residual = residual - q
        print(f"Initialized {self.n_codebooks} codebooks with KMeans.")

def generate_semantic_ids(embeddings, n_codebooks=3, codebook_size=32):
    """
    Convenience function to generate semantic IDs for items.
    """
    input_dim = embeddings.shape[1]
    rvq = ResidualVectorQuantizer(input_dim, n_codebooks, codebook_size)
    
    # Fit with KMeans for stability
    if isinstance(embeddings, torch.Tensor):
        emb_np = embeddings.detach().cpu().numpy()
    else:
        emb_np = embeddings
        
    rvq.fit_kmeans(emb_np)
    
    # Forward pass to get indices
    with torch.no_grad():
        indices, _ = rvq(torch.from_numpy(emb_np).float())
        
    return indices, rvq
