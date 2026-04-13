#!/usr/bin/env python3
"""
Training loop for CF/CL models.
Handles BPR loss, contrastive SSL losses, denoising, and leakage detection.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.cf_data_loader import sample_batch
from src.training.cf_evaluator import evaluate, compute_entropy
from src.models.ma_hcl import MAHCL
from src.models import XSimGCL, MAHGN, LightGCN
from src.models.bigcf import BIGCF


def train_model(model, data, args, device, item_content=None, semantic_ids=None, user_priors=None, 
                re_ranker=None, cold_users=None):

    """Train a CF model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    edge_index = data['edge_index'].to(device)
    edge_index_dict = data.get('edge_index_dict')
    if edge_index_dict is None and isinstance(data, dict):
        if 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
            edge_index_dict = data['graph'].edge_index_dict
    train_pairs = data['train_pairs']
    train_dict = data['train_dict']
    test_dict = data['test_dict']
    n_items = data['n_items']

    is_hetero_model = args.model in ['ma_hgn', 'sim-mahgn']
    graph_to_check = edge_index_dict if (is_hetero_model and edge_index_dict is not None) else edge_index
    
    if graph_to_check is not None:
        if isinstance(graph_to_check, dict):
            # Hetero graph: Look for user-article or user-item edges
            ua_edges = None
            for key, val in graph_to_check.items():
                if isinstance(key, tuple) and len(key) == 3:
                    if 'user' in str(key[0]).lower() and ('article' in str(key[2]).lower() or 'item' in str(key[2]).lower()):
                        ua_edges = val
                        break
            if ua_edges is not None:
                n_graph_interactions = ua_edges.size(1)
        else:
            # Bipartite graph: Count edges (might be symmetric 2*N or directed N)
            n_edges = graph_to_check.size(1)
            
            # If edges == 2 * train_pairs, it's symmetric (count half)
            if n_edges == len(train_pairs) * 2:
                n_graph_interactions = n_edges // 2
            elif n_edges == len(train_pairs):
                n_graph_interactions = n_edges
            else:
                # Unknown format: use unique pairs
                src, dst = graph_to_check[0].cpu(), graph_to_check[1].cpu()
                pairs = set(zip(src.tolist(), dst.tolist()))
                n_graph_interactions = len(pairs)
                
        n_train_interactions = len(train_pairs)
        
        if n_graph_interactions > n_train_interactions:
            print(f"\n" + "!"*60)
            print(f"CRITICAL LEAKAGE DETECTED!")
            print(f"   Message Passing Graph has {n_graph_interactions:,} user-item interactions.")
            print(f"   Training Set has only {n_train_interactions:,} interactions.")
            print(f"   Leakage: {n_graph_interactions - n_train_interactions:,} test edges are visible to the model!")
            print(f"   REASON: Graph was likely built with '--min-user-interactions' on ALL data.")
            print(f"   FIX: Regenerate graph using leakage-fixed converter.")
            print("!"*60 + "\n")

    
    best_recall = 0
    best_metrics = {}
    best_state = None
    patience_counter = 0
    
    pbar = tqdm(range(args.epochs), desc=f"Training {args.model.upper()}", ncols=100)
    
    if args.epochs == 0:
        print("Epochs set to 0. Exiting after data preparation.")
        return {'status': 'consolidated'}
        
    item_interaction_counts = np.zeros(n_items)
    for _, item in train_pairs:
        item_interaction_counts[item] += 1
    item_probs = (item_interaction_counts + 1e-6) / (item_interaction_counts.sum() + 1e-6 * n_items)
    sampling_strategy = 'popular' if args.denoise_ratio > 0 else 'random' # Default to popular if denoising
    
    for epoch in pbar:
        model.train()
        total_loss = 0
        n_batches = len(train_pairs) // args.batch_size + 1
        
        for _ in range(n_batches):
            users, pos_items, neg_items = sample_batch(train_pairs, train_dict, n_items, args.batch_size, args.neg_ratio, 
                                                      sampling=sampling_strategy, item_probs=item_probs)
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            
            optimizer.zero_grad()
            
            # Different models have different loss signatures
            if hasattr(model, 'calculate_loss'):
                if args.model in ['simgcl', 'bigcf', 'igcl', 'xsimgcl', 'lightgcn']:
                    graph_structure = data.get('adj_norm')
                    if graph_structure is None:
                        print("Warning: adj_norm missing in train loop, falling back to edge_index")
                        graph_structure = edge_index
                elif args.model == 'lightgcl':
                     graph_structure = data.get('adj_norm')
                elif args.model in ['ma_hgn', 'ma-hcl']:
                    graph_structure = getattr(data, 'edge_index_dict', None)
                    if graph_structure is None and isinstance(data, dict):
                        graph_structure = data.get('edge_index_dict')
                    
                    if graph_structure is None:
                         src, dst = edge_index
                         n_users_limit = data['n_users']
                         
                         mask = (src < n_users_limit) & (dst >= n_users_limit)
                         u_i_src = src[mask]
                         u_i_dst = dst[mask] - n_users_limit # Remove offset
                         
                         u_i_edges = torch.stack([u_i_src, u_i_dst], dim=0)
                         i_u_edges = torch.stack([u_i_dst, u_i_src], dim=0)
                         
                         graph_structure = {
                             ('user', 'interacts', 'item'): u_i_edges,
                             ('item', 'rev_interacts', 'user'): i_u_edges
                         }
                else:
                    graph_structure = edge_index
                
                
                if isinstance(model, XSimGCL):
                    if args.denoise_ratio > 0 and epoch >= 5: # Burn-in 5 epochs
                        loss, bpr_sample, ssl, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items, 
                                                                         semantic_ids=semantic_ids, user_priors=user_priors,
                                                                         return_per_sample=True)
                        # Truncated Loss Denoising
                        n_to_prune = int(len(bpr_sample) * args.denoise_ratio)
                        if n_to_prune > 0:
                            _, indices = torch.topk(bpr_sample, k=n_to_prune, largest=True)
                            mask = torch.ones_like(bpr_sample)
                            mask[indices] = 0
                            bpr = (bpr_sample * mask).sum() / mask.sum()
                            loss = bpr + model.ssl_weight * ssl + model.lambda_reg * reg
                        else:
                            bpr = bpr_sample.mean()
                    else:
                        loss, bpr, ssl, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items, 
                                                                  semantic_ids=semantic_ids, user_priors=user_priors)
                elif isinstance(model, BIGCF):
                    loss, bpr, cl, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items)

                elif isinstance(model, MAHGN):
                    # MAHGN logic for heterogeneous graph structure
                    hetero_graph_structure = getattr(data, 'edge_index_dict', None)
                    if hetero_graph_structure is None and isinstance(data, dict):
                        if 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
                             hetero_graph_structure = data['graph'].edge_index_dict
                        else:
                             hetero_graph_structure = data.get('edge_index_dict')
                             
                    # If still None, construct from bipartite edge_index
                    if hetero_graph_structure is None and edge_index is not None:
                         src, dst = edge_index
                         n_users_limit = data['n_users']
                         
                         # Filter user - item edges (src < n_users, dst >= n_users)
                         mask = (src < n_users_limit) & (dst >= n_users_limit)
                         u_i_src = src[mask]
                         u_i_dst = dst[mask] - n_users_limit
                         
                         u_i_edges = torch.stack([u_i_src, u_i_dst], dim=0)
                         i_u_edges = torch.stack([u_i_dst, u_i_src], dim=0)
                         
                         hetero_graph_structure = {
                             ('user', 'interacts', 'item'): u_i_edges,
                             ('item', 'rev_interacts', 'user'): i_u_edges
                         }
                    
                    loss, bpr, cl, reg = model.calculate_loss(hetero_graph_structure, users, pos_items, neg_items)

                elif isinstance(model, MAHCL):
                     # MAHCL uses graph_structure computed above
                     loss, bpr, cl, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items)
                else:
                    loss, bpr, reg, ssl = model.calculate_loss(graph_structure, users, pos_items, neg_items)
            elif hasattr(model, 'bpr_loss'):
                # NGCF/NCL style
                loss, reg = model.bpr_loss(users, pos_items, neg_items, edge_index)
                loss = loss + args.weight_decay * reg
            else:
                user_emb, item_emb = model(edge_index)
                
                pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
                neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        else:
            pbar.set_postfix({'loss': f"{total_loss:.4f}"})
        
        # Evaluate only at the final epoch
        if epoch == args.epochs - 1:
            edge_index_dict = None
            if isinstance(data, dict):
                edge_index_dict = data.get('edge_index_dict')
                if edge_index_dict is None and 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
                    edge_index_dict = data['graph'].edge_index_dict
            elif hasattr(data, 'edge_index_dict'):
                edge_index_dict = data.edge_index_dict
            
            adj_norm = data.get('adj_norm') if isinstance(data, dict) else getattr(data, 'adj_norm', None)
            
            # Fallback for Hetero models on Homogeneous Data
            if edge_index_dict is None and args.model in ['ma_hgn', 'sim-mahgn', 'ma-hcl']:
                 if edge_index is not None:
                     src, dst = edge_index
                     n_users_limit = data['n_users']
                     
                     mask = (src < n_users_limit) & (dst >= n_users_limit)
                     u_i_src = src[mask]
                     u_i_dst = dst[mask] - n_users_limit # Remove offset
                     
                     u_i_edges = torch.stack([u_i_src, u_i_dst], dim=0)
                     i_u_edges = torch.stack([u_i_dst, u_i_src], dim=0)
                     
                     edge_index_dict = {
                         ('user', 'interacts', 'item'): u_i_edges,
                         ('item', 'rev_interacts', 'user'): i_u_edges
                     }

            metrics = evaluate(model, test_dict, train_dict, n_items, edge_index, device=device, adj_norm=adj_norm,
                               re_ranker=re_ranker, rerank_strategy=args.rerank, eval_protocol=args.eval_protocol,
                               cold_users=cold_users, edge_index_dict=edge_index_dict)

            best_metrics = metrics.copy()
            best_state = model.state_dict().copy()
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_metrics
