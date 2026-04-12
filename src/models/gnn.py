import torch
from torch_geometric.nn import SAGEConv

class GNNBaseline(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        # SAGEConv((-1, -1)) cho phép input dimension linh hoạt (lazy initialization)
        # Layer 1: Input to Hidden
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        # Layer 2: Hidden to Output Embedding
        self.conv2 = SAGEConv((-1, -1), out_dim)

    def forward(self, x, edge_index):
        # ReLU activation cho tính phi tuyến
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
