import os
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import to_hetero

# Import modules từ folder src
from src.dataset import load_data, load_processed_data
from src.models import GNNBaseline
from src.trainer import train_model
from src.metrics import evaluate_top_k

# ================= CONFIG =================
# Cấu hình đường dẫn và tham số
ARTICLES_PATH = 'data/raw/articles.csv'
REPLIES_PATH = 'data/raw/replies.csv'
HIDDEN_DIM = 64
OUT_DIM = 32
EPOCHS = 100
LR = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ==========================================

def main():
    print(f"🚀 Gad's AlgoVerse Recommender - Baseline v0.1")
    print(f"Running on: {DEVICE}")

    # 1. Load Data
    # data, user_map, article_map = load_data(ARTICLES_PATH, REPLIES_PATH, HIDDEN_DIM)
    data = load_processed_data()
    data = data.to(DEVICE)

    # 2. Split Data (Train/Val/Test)
    # RandomLinkSplit tự động tách edge và sinh nhãn
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        edge_types=[('user', 'comments', 'article')],
        rev_edge_types=[('article', 'rev_comments', 'user')]
    )
    train_data, val_data, test_data = transform(data)

    # 3. Initialize Model
    base_model = GNNBaseline(HIDDEN_DIM, OUT_DIM)

    # Chuyển đổi sang Heterogeneous GNN (tự động xử lý các loại node khác nhau)
    model = to_hetero(base_model, data.metadata(), aggr='sum').to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 4. Training Loop
    edge_type = ('user', 'comments', 'article')
    train_model(model, train_data, optimizer, EPOCHS, DEVICE, edge_type)

    # 5. Evaluation
    print("\n--- Starting Evaluation ---")
    k = 5
    p, r, n = evaluate_top_k(model, test_data, edge_type, k=k)

    print(f"\n📊 FINAL RESULTS (Top-{k}):")
    print(f"   Precision : {p:.4f}")
    print(f"   Recall    : {r:.4f}")
    print(f"   NDCG      : {n:.4f}")

    # 6. Save Model
    save_path = 'baseline_v0.1.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Model saved to {save_path}")

if __name__ == "__main__":
    main()
