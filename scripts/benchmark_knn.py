import os
import torch
import numpy as np
import scipy.sparse as sp
import argparse
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from collections import defaultdict
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import compute_metrics, print_metrics

def load_data(data_path):
    print(f"Loading data from {data_path}...")
    graph_path = os.path.join(data_path, 'user_article_graph.pt')
    # Load with weights_only=False as verified safe local file
    graph_data = torch.load(graph_path, weights_only=False)
    
    # Load data
    hetero_data = graph_data['graph']
    n_users = graph_data['n_users']
    n_items = graph_data['n_items']
    
    # Extract training edges
    edge_store = hetero_data[('user', 'comments', 'article')]
    train_edge_index = edge_store.edge_index.numpy()
    
    # Extract edge weights if available (implicit ratings)
    if 'edge_weight' in edge_store:
        train_ratings = edge_store.edge_weight.numpy()
    else:
        train_ratings = np.ones(train_edge_index.shape[1], dtype=np.float32)
        
    # Extract test edges
    if 'test_edge_index' in edge_store:
        test_edge_index = edge_store.test_edge_index.numpy()
    else:
        raise ValueError("No test_edge_index found in graph data!")

    return n_users, n_items, train_edge_index, train_ratings, test_edge_index

def build_sparse_matrix(n_users, n_items, edge_index, ratings):
    row = edge_index[0]
    col = edge_index[1]
    return sp.csr_matrix((ratings, (row, col)), shape=(n_users, n_items))

def get_test_dict(test_edge_index, protocol='full'):
    test_dict = defaultdict(list)
    # Group all test items by user
    temp_dict = defaultdict(list)
    for u, i in zip(test_edge_index[0], test_edge_index[1]):
        temp_dict[u].append(i)
        
    if protocol == 'full':
        return temp_dict
    elif protocol == 'loo':
        # Leave-One-Out: Take ONLY the last item (assuming sorted by time)
        # However, since we don't have timestamps here, we assume the input order 
        # preserves time (which is true for our data processing pipeline).
        # OR better: Randomly pick one if time is unknown, but here we trust the split order.
        loo_dict = {}
        for u, items in temp_dict.items():
            if len(items) > 0:
                loo_dict[u] = [items[-1]] # Take the last one
        return loo_dict
    else:
        return temp_dict

def get_train_dict(train_edge_index, test_edge_index=None, protocol='full'):
    train_dict = defaultdict(set)
    for u, i in zip(train_edge_index[0], train_edge_index[1]):
        train_dict[u].add(i)
        
    if protocol == 'loo' and test_edge_index is not None:
        # IN LOO: The "Train" set for the model should arguably include 
        # the "other" test items that were not picked as the evaluation target.
        # But to be safe and consistent with GNN training (which used strict split),
        # we will strictly use the provided train_edge_index.
        # So no changes here. The "test" set in GNN was strictly held out.
        # If we move items from test->train, we change the model's knowledge compared to GNN.
        # So we keep it strict.
        pass
        
    return train_dict

def sparse_similarity(R, dense_output=False, method='cosine'):
    # R is sparse csr
    if method == 'jaccard':
        print("Computing Sparse Jaccard Similarity...")
        # 1. Binarize R (treat all ratings as 1)
        R_bin = R.copy()
        R_bin.data = np.ones_like(R_bin.data)
        
        # 2. Intersection (common items)
        # I[u, v] = |Items(u) AND Items(v)|
        intersection = R_bin.dot(R_bin.T)
        
        # 3. Union
        # |Union| = |A| + |B| - |Intersection|
        # sizes[u] = |Items(u)|
        sizes = np.array(R_bin.sum(axis=1)).flatten()
        
        rows, cols = intersection.nonzero()
        inter_vals = intersection.data
        
        # Calculate Union for non-zero intersections only (Sparse-safe)
        union_vals = sizes[rows] + sizes[cols] - inter_vals
        
        # Avoid division by zero
        valid = union_vals > 0
        
        # Jaccard = I / U
        intersection.data[valid] = inter_vals[valid] / union_vals[valid]
        intersection.data[~valid] = 0.0
        
        return intersection

    elif method == 'pearson':
        # Pearson = Cosine on centered data
        print("Note: Approximating Pearson for sparse data (centering non-zeros)...")
        R_copy = R.copy()
        R_copy.data = R_copy.data - np.mean(R_copy.data)
        # Re-use cosine logic on centered data
        R = R_copy

    # Cosine Logic
    # Normalize rows
    row_norms = np.array(np.sqrt(R.multiply(R).sum(axis=1)))[:, 0]
    row_norms[row_norms == 0] = 1.0
    div = sp.diags(1 / row_norms)
    R_norm = div.dot(R)
    
    # Cosine sim = R_norm * R_norm.T
    sim = R_norm.dot(R_norm.T)
    return sim

def run_user_knn(R, n_users, n_items, similarity='cosine'):
    print(f"Computing User-User Similarity ({similarity})...")
    sim_matrix = sparse_similarity(R, method=similarity)
    
    # Zero out diagonal
    sim_matrix.setdiag(0)
    sim_matrix.eliminate_zeros()
    
    print("Predicting scores (Sparse)...")
    # Scores = Sim * R
    scores = sim_matrix.dot(R)
    
    return scores

def run_item_knn(R, n_users, n_items, similarity='cosine'):
    print(f"Computing Item-Item Similarity ({similarity})...")
    sim_matrix = sparse_similarity(R.T, method=similarity)
    
    # Zero out diagonal
    sim_matrix.setdiag(0)
    sim_matrix.eliminate_zeros()
    
    print("Predicting scores (Sparse)...")
    # Scores = R * Sim
    scores = R.dot(sim_matrix)
    
    return scores

def run_slope_one(R, n_users, n_items):
    print("Computing Slope One Deviations (Dense)...")
    
    # 1. Convert R to dense for deviation calculation (N_Users x N_Items)
    # Note: R is (13k x 3.6k). Dense is ~180MB. Feasible.
    R_dense = R.toarray()
    
    # 2. Compute Deviation Matrix D[j,i]
    
    # Let R_bin be binary incidence matrix
    R_bin = (R > 0).astype(np.float32)
    
    # Common Counts: C = R_bin.T @ R_bin
    # C[j,i] is count of users who rated both j and i
    C = (R_bin.T @ R_bin).toarray()
    
    # Numerator part 1: A[j,i] = Sum_u (R[u,j] * I(u has i))
    A = R.T @ R_bin 
    A = A.toarray()
    
    # Numerator part 2: B[j,i] = Sum_u (R[u,i] * I(u has j)) = A.T
    B = A.T 
    
    # DiffSum = Sum (r_uj - r_ui) over shared users
    DiffSum = A - B
    
    print("Deviations computed implicitly. Predicting...")
    
    # Weighted Slope One Prediction Formula:
    # P(u,j) = ( Sum_{i in R_u} (dev[j,i] + r_ui) * C[j,i] ) / Sum_{i in R_u} C[j,i]
    #
    # Numerator:
    # = Sum_i (dev[j,i]*C[j,i] + r_ui*C[j,i])
    # dev[j,i]*C[j,i] = DiffSum[j,i]
    # -> Sum_i (DiffSum[j,i] + r_ui*C[j,i])
    #
    # Part 1: Sum over i in R_u of DiffSum[j,i]
    # Note: DiffSum[j,i] corresponds to target j.
    # We want result[u, j].
    # R_bin[u, :] selects the items i in user's history.
    # Product: R_bin @ DiffSum.T 
    # (n_users, n_items) x (n_items, n_items) -> (n_users, n_items)
    # Element [u, j] is sum_i (R_bin[u,i] * DiffSum.T[i,j])
    # DiffSum.T[i,j] = DiffSum[j,i] (Wait. Transpose indices)
    # Let's verify: Result[u,j] = Sum_k (M[u,k] * N[k,j])
    # Here k is item i.
    # We want Sum_i (R_bin[u,i] * DiffSum[j,i]).
    # DiffSum[j,i] is the (j,i) element.
    # We need matrix where column j contains DiffSum[j, :].
    # No, we need Sum_i (R_bin[u,i] * Matrix[i,j]).
    # So Matrix[i,j] must be DiffSum[j,i].
    # So Matrix must be DiffSum.T.
    # Yes. R_bin @ DiffSum.T
    
    Num1 = R_bin.dot(DiffSum.T)
    
    # Part 2: Sum over i in R_u of (r_ui * C[j,i])
    # Note r_ui is continuous rating. C[j,i] is count (symmetric).
    # We want Sum_i (R[u,i] * C[i,j]).
    # This is R @ C.
    Num2 = R.dot(C)
    
    # Denom: Sum over i in R_u of C[j,i]
    # Sum_i (I(u has i) * C[i,j])
    # This is R_bin @ C.
    Denom = R_bin.dot(C)
    
    # Combine
    with np.errstate(divide='ignore', invalid='ignore'):
        Pred = (Num1 + Num2) / Denom
        Pred[np.isnan(Pred)] = 0.0
        Pred[np.isinf(Pred)] = 0.0
    
    return Pred

def run_svd(R, n_users, n_items, k=64):
    print(f"Running SVD (Matrix Factorization) with k={k}...")
    # TruncatedSVD is optimized for sparse input
    svd = TruncatedSVD(n_components=k, random_state=42)
    
    # Fit and transform users
    print("Fitting SVD...")
    user_factors = svd.fit_transform(R) # (n_users, k)
    item_factors = svd.components_.T    # (n_items, k)
    
    print(f"Factors learned. Predicting...")
    # Scores = U @ V.T
    scores = user_factors.dot(item_factors.T)
    
    return scores

def run_nmf(R, n_users, n_items, k=64):
    print(f"Running NMF (Non-negative Matrix Factorization) with k={k}...")
    # NMF handles non-negative data (implicit feedback).
    # 'cd' solver is Coordinate Descent (similar to ALS).
    # init='nndsvd' is better for sparsity.
    model = NMF(n_components=k, init='nndsvd', solver='cd', random_state=42, max_iter=200)
    
    print("Fitting NMF (this might take a minute)...")
    user_factors = model.fit_transform(R) # (n_users, k)
    item_factors = model.components_.T    # (n_items, k)
    
    print("Factors learned. Predicting...")
    scores = user_factors.dot(item_factors.T)
    
    return scores

def main():
    parser = argparse.ArgumentParser(description="Benchmark Model-Based CF (MF)")
    parser.add_argument('--data-path', type=str, required=True, help='Path to processed data directory')
    parser.add_argument('--model', type=str, choices=['user_knn', 'item_knn', 'slope_one', 'svd', 'nmf'], required=True, help='Model type')
    parser.add_argument('--similarity', type=str, choices=['cosine', 'pearson', 'jaccard'], default='cosine', help='Similarity metric')
    parser.add_argument('--protocol', type=str, choices=['full', 'loo'], default='full', help='Evaluation protocol')
    parser.add_argument('--output', type=str, default=None, help='Path to save JSON results')
    
    args = parser.parse_args()
    
    print(f"Running {args.model} with {args.similarity} sim on {args.protocol} protocol.")

    # 1. Load Data
    n_users, n_items, train_idx, train_ratings, test_idx = load_data(args.data_path)
    print(f"Data loaded: {n_users} users, {n_items} items.")
    print(f"Train edges: {train_idx.shape[1]}, Test edges: {test_idx.shape[1]}")
    
    # 2. Build Sparse Interaction Matrix
    R = build_sparse_matrix(n_users, n_items, train_idx, train_ratings)
    
    # 3. Helpers for Evaluation
    test_dict = get_test_dict(test_idx, protocol=args.protocol)
    train_dict = get_train_dict(train_idx, test_idx, protocol=args.protocol)
    
    # 4. Run Model
    if args.model == 'user_knn':
        scores = run_user_knn(R, n_users, n_items, similarity=args.similarity)
    elif args.model == 'item_knn':
        scores = run_item_knn(R, n_users, n_items, similarity=args.similarity)
    elif args.model == 'slope_one':
        scores = run_slope_one(R, n_users, n_items)
    elif args.model == 'svd':
        scores = run_svd(R, n_users, n_items, k=64)
    elif args.model == 'nmf':
        scores = run_nmf(R, n_users, n_items, k=64)
    
    # 5. Evaluate
    print("Evaluating...")
    # metrics.py expects scores as dict or numpy array. 
    # scores is currently (n_users, n_items) sparse csr.
    # Convert to dense for evaluation (approx 200MB, safe)
    if sp.issparse(scores):
        scores = scores.toarray()
        
    metrics = compute_metrics(scores, test_dict, train_dict)
    
    # 6. Print and Save
    print_metrics(metrics)
    
    if args.output:
        # Create directory if needed
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results = {
            "model": args.model,
            "data_path": args.data_path,
            "metrics": metrics
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
