"""
EDA for Graph Data
==================
Exploratory Data Analysis on the GNN-ready graph data.

Analyzes:
- Node statistics (users, articles)
- Edge statistics and connectivity
- Degree distribution
- Graph properties (density, clustering)
- Feature analysis

Usage:
    python src/eda_graph_data.py
    python src/eda_graph_data.py --data-path data/processed/user_article_graph.pt
"""

import argparse
import os
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class GraphDataEDA:
    """EDA for PyTorch Geometric graph data."""
    
    def __init__(self, data_path: str, output_dir: str = 'plots'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("=" * 60)
        print("Graph Data - Exploratory Data Analysis")
        print("=" * 60)
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self):
        """Load graph data."""
        print(f"\nLoading graph from {self.data_path}...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Graph not found: {self.data_path}")
        
        data = torch.load(self.data_path, weights_only=False)
        print(f"   [OK] Graph loaded successfully")
        return data
    
    def print_basic_stats(self):
        """Print basic graph statistics."""
        print("\n" + "=" * 60)
        print("BASIC GRAPH STATISTICS")
        print("=" * 60)
        
        # Check if heterogeneous
        is_hetero = hasattr(self.data, 'node_types')
        
        if is_hetero:
            print("\nGraph Type: Heterogeneous")
            print(f"Node Types: {self.data.node_types}")
            print(f"Edge Types: {self.data.edge_types}")
            
            print("\n--- Node Statistics ---")
            total_nodes = 0
            for node_type in self.data.node_types:
                num_nodes = self.data[node_type].x.shape[0]
                feat_dim = self.data[node_type].x.shape[1]
                total_nodes += num_nodes
                print(f"  {node_type}: {num_nodes:,} nodes, {feat_dim} features")
            print(f"  Total: {total_nodes:,} nodes")
            
            print("\n--- Edge Statistics ---")
            total_edges = 0
            for edge_type in self.data.edge_types:
                edge_index = self.data[edge_type].edge_index
                num_edges = edge_index.shape[1]
                total_edges += num_edges
                print(f"  {edge_type}: {num_edges:,} edges")
            print(f"  Total: {total_edges:,} edges")
            
            # Graph density
            if 'user' in self.data.node_types and 'article' in self.data.node_types:
                num_users = self.data['user'].x.shape[0]
                num_articles = self.data['article'].x.shape[0]
                max_edges = num_users * num_articles
                edge_type = ('user', 'comments', 'article')
                if edge_type in self.data.edge_types:
                    actual_edges = self.data[edge_type].edge_index.shape[1]
                    density = actual_edges / max_edges
                    print(f"\n--- Graph Density ---")
                    print(f"  Bipartite density: {density:.6f} ({density*100:.4f}%)")
                    print(f"  Sparsity: {(1-density)*100:.4f}%")
        else:
            print("\nGraph Type: Homogeneous")
            print(f"Nodes: {self.data.x.shape[0]:,}")
            print(f"Features: {self.data.x.shape[1]}")
            print(f"Edges: {self.data.edge_index.shape[1]:,}")
    
    def analyze_degree_distribution(self):
        """Analyze and plot degree distributions."""
        print("\nAnalyzing degree distributions...")
        
        is_hetero = hasattr(self.data, 'node_types')
        
        if is_hetero and 'user' in self.data.node_types:
            # User-Article bipartite graph
            edge_type = ('user', 'comments', 'article')
            if edge_type not in self.data.edge_types:
                print("   No user-article edges found")
                return
            
            edge_index = self.data[edge_type].edge_index
            num_users = self.data['user'].x.shape[0]
            num_articles = self.data['article'].x.shape[0]
            
            # User degrees (how many articles each user commented on)
            user_degrees = torch.bincount(edge_index[0], minlength=num_users).numpy()
            
            # Article degrees (how many users commented on each article)
            article_degrees = torch.bincount(edge_index[1], minlength=num_articles).numpy()
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # User degree distribution
            ax1 = axes[0, 0]
            ax1.hist(user_degrees, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
            ax1.set_xlabel('Degree (Articles Commented)')
            ax1.set_ylabel('Number of Users')
            ax1.set_title('User Degree Distribution')
            ax1.set_yscale('log')
            
            # User degree stats
            ax2 = axes[0, 1]
            ax2.hist(user_degrees[user_degrees > 0], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
            ax2.set_xlabel('Degree')
            ax2.set_ylabel('Frequency')
            ax2.set_title('User Degree (Excluding Zero-Degree)')
            
            # User degree log-log (power law check)
            ax3 = axes[0, 2]
            degree_counts = Counter(user_degrees)
            degrees = sorted(degree_counts.keys())
            counts = [degree_counts[d] for d in degrees]
            ax3.loglog(degrees[1:], counts[1:], 'o', markersize=3, alpha=0.6)
            ax3.set_xlabel('Degree (log)')
            ax3.set_ylabel('Frequency (log)')
            ax3.set_title('User Degree Power Law Check')
            ax3.grid(True, alpha=0.3)
            
            # Article degree distribution
            ax4 = axes[1, 0]
            ax4.hist(article_degrees, bins=50, color='coral', edgecolor='white', alpha=0.8)
            ax4.set_xlabel('Degree (Number of Commenters)')
            ax4.set_ylabel('Number of Articles')
            ax4.set_title('Article Degree Distribution')
            ax4.set_yscale('log')
            
            # Article degree stats
            ax5 = axes[1, 1]
            ax5.hist(article_degrees[article_degrees > 0], bins=50, color='coral', edgecolor='white', alpha=0.8)
            ax5.set_xlabel('Degree')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Article Degree (Excluding Zero-Degree)')
            
            # Article degree log-log
            ax6 = axes[1, 2]
            degree_counts = Counter(article_degrees)
            degrees = sorted(degree_counts.keys())
            counts = [degree_counts[d] for d in degrees]
            ax6.loglog(degrees[1:], counts[1:], 'o', markersize=3, alpha=0.6, color='coral')
            ax6.set_xlabel('Degree (log)')
            ax6.set_ylabel('Frequency (log)')
            ax6.set_title('Article Degree Power Law Check')
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.output_dir / 'graph_eda_01_degree_distribution.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   [OK] Saved: {save_path}")
            plt.close()
            
            # Print statistics
            print("\n--- Degree Statistics ---")
            print(f"  User Degrees:")
            print(f"    Mean: {user_degrees.mean():.2f}")
            print(f"    Std: {user_degrees.std():.2f}")
            print(f"    Min: {user_degrees.min()}, Max: {user_degrees.max()}")
            print(f"    Median: {np.median(user_degrees):.0f}")
            print(f"    Zero-degree users: {(user_degrees == 0).sum():,}")
            
            print(f"\n  Article Degrees:")
            print(f"    Mean: {article_degrees.mean():.2f}")
            print(f"    Std: {article_degrees.std():.2f}")
            print(f"    Min: {article_degrees.min()}, Max: {article_degrees.max()}")
            print(f"    Median: {np.median(article_degrees):.0f}")
            print(f"    Zero-degree articles: {(article_degrees == 0).sum():,}")
    
    def analyze_features(self):
        """Analyze node feature distributions."""
        print("\nAnalyzing node features...")
        
        is_hetero = hasattr(self.data, 'node_types')
        
        if is_hetero:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            for idx, node_type in enumerate(['user', 'article']):
                if node_type not in self.data.node_types:
                    continue
                
                features = self.data[node_type].x.numpy()
                
                ax = axes[idx]
                
                # Feature magnitude distribution
                magnitudes = np.linalg.norm(features, axis=1)
                ax.hist(magnitudes, bins=50, color='green' if node_type == 'user' else 'purple',
                       edgecolor='white', alpha=0.8)
                ax.set_xlabel('Feature Vector Magnitude')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{node_type.capitalize()} Feature Magnitudes\n(dim={features.shape[1]})')
                
                # Add stats
                stats_text = f"Mean: {magnitudes.mean():.2f}\nStd: {magnitudes.std():.2f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            save_path = self.output_dir / 'graph_eda_02_feature_distribution.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   [OK] Saved: {save_path}")
            plt.close()
    
    def analyze_connectivity(self):
        """Analyze graph connectivity patterns."""
        print("\nAnalyzing connectivity patterns...")
        
        is_hetero = hasattr(self.data, 'node_types')
        
        if not is_hetero:
            return
        
        edge_type = ('user', 'comments', 'article')
        if edge_type not in self.data.edge_types:
            return
        
        edge_index = self.data[edge_type].edge_index
        num_users = self.data['user'].x.shape[0]
        num_articles = self.data['article'].x.shape[0]
        
        user_degrees = torch.bincount(edge_index[0], minlength=num_users).numpy()
        article_degrees = torch.bincount(edge_index[1], minlength=num_articles).numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top users by activity
        ax1 = axes[0, 0]
        top_k = 20
        top_user_indices = np.argsort(user_degrees)[-top_k:][::-1]
        top_user_degrees = user_degrees[top_user_indices]
        ax1.barh(range(top_k), top_user_degrees, color='steelblue')
        ax1.set_yticks(range(top_k))
        ax1.set_yticklabels([f'User {i}' for i in top_user_indices])
        ax1.set_xlabel('Number of Articles Commented')
        ax1.set_title(f'Top {top_k} Most Active Users')
        ax1.invert_yaxis()
        
        # Top articles by engagement
        ax2 = axes[0, 1]
        top_article_indices = np.argsort(article_degrees)[-top_k:][::-1]
        top_article_degrees = article_degrees[top_article_indices]
        ax2.barh(range(top_k), top_article_degrees, color='coral')
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels([f'Article {i}' for i in top_article_indices])
        ax2.set_xlabel('Number of Commenters')
        ax2.set_title(f'Top {top_k} Most Commented Articles')
        ax2.invert_yaxis()
        
        # User activity distribution (cumulative)
        ax3 = axes[1, 0]
        sorted_user_degrees = np.sort(user_degrees)[::-1]
        cumsum = np.cumsum(sorted_user_degrees) / sorted_user_degrees.sum()
        x_pct = np.arange(1, len(cumsum) + 1) / len(cumsum) * 100
        ax3.plot(x_pct, cumsum * 100, color='steelblue', linewidth=2)
        ax3.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Percentage of Users')
        ax3.set_ylabel('Percentage of Total Interactions')
        ax3.set_title('User Activity Pareto Analysis')
        ax3.grid(True, alpha=0.3)
        
        # Identify Pareto point
        pareto_idx = np.searchsorted(cumsum, 0.8)
        pareto_pct = (pareto_idx / len(cumsum)) * 100
        ax3.annotate(f'{pareto_pct:.1f}% users -> 80% activity',
                    xy=(pareto_pct, 80), xytext=(pareto_pct + 10, 70),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        # Article engagement distribution
        ax4 = axes[1, 1]
        sorted_article_degrees = np.sort(article_degrees)[::-1]
        cumsum = np.cumsum(sorted_article_degrees) / sorted_article_degrees.sum()
        x_pct = np.arange(1, len(cumsum) + 1) / len(cumsum) * 100
        ax4.plot(x_pct, cumsum * 100, color='coral', linewidth=2)
        ax4.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Percentage of Articles')
        ax4.set_ylabel('Percentage of Total Comments')
        ax4.set_title('Article Engagement Pareto Analysis')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'graph_eda_03_connectivity.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {save_path}")
        plt.close()
    
    def analyze_edge_weights(self):
        """Analyze edge weights if present."""
        print("\nAnalyzing edge weights...")
        
        is_hetero = hasattr(self.data, 'node_types')
        
        if is_hetero:
            edge_type = ('user', 'comments', 'article')
            if edge_type in self.data.edge_types:
                edge_data = self.data[edge_type]
                if hasattr(edge_data, 'edge_weight') and edge_data.edge_weight is not None:
                    weights = edge_data.edge_weight.numpy()
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1 = axes[0]
                    ax1.hist(weights, bins=50, color='teal', edgecolor='white', alpha=0.8)
                    ax1.set_xlabel('Edge Weight')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title('Edge Weight Distribution')
                    ax1.set_yscale('log')
                    
                    ax2 = axes[1]
                    ax2.boxplot(weights, vert=True, patch_artist=True,
                               boxprops=dict(facecolor='teal', alpha=0.8))
                    ax2.set_ylabel('Edge Weight')
                    ax2.set_title('Edge Weight Box Plot')
                    
                    plt.tight_layout()
                    save_path = self.output_dir / 'graph_eda_04_edge_weights.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"   [OK] Saved: {save_path}")
                    plt.close()
                    
                    print(f"\n--- Edge Weight Statistics ---")
                    print(f"  Mean: {weights.mean():.2f}")
                    print(f"  Std: {weights.std():.2f}")
                    print(f"  Min: {weights.min():.2f}, Max: {weights.max():.2f}")
                else:
                    print("   No edge weights found (using uniform weights)")
    
    def generate_summary(self):
        """Generate a summary plot."""
        print("\nGenerating summary...")
        
        is_hetero = hasattr(self.data, 'node_types')
        
        if not is_hetero:
            return
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Graph Data Summary', fontsize=16, fontweight='bold')
        
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Key metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        num_users = self.data['user'].x.shape[0]
        num_articles = self.data['article'].x.shape[0]
        edge_type = ('user', 'comments', 'article')
        num_edges = self.data[edge_type].edge_index.shape[1] if edge_type in self.data.edge_types else 0
        
        metrics_text = f"""
Graph Statistics:
-----------------
Users: {num_users:,}
Articles: {num_articles:,}
Edges: {num_edges:,}

Feature Dims:
  User: {self.data['user'].x.shape[1]}
  Article: {self.data['article'].x.shape[1]}

Avg Interactions:
  Per User: {num_edges/num_users:.1f}
  Per Article: {num_edges/num_articles:.1f}
        """
        ax1.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                family='monospace', transform=ax1.transAxes)
        
        # User degree
        ax2 = fig.add_subplot(gs[0, 1])
        edge_index = self.data[edge_type].edge_index
        user_degrees = torch.bincount(edge_index[0], minlength=num_users).numpy()
        ax2.hist(user_degrees, bins=30, color='steelblue', edgecolor='white')
        ax2.set_title('User Degree Distribution')
        ax2.set_yscale('log')
        ax2.set_xlabel('Degree')
        
        # Article degree
        ax3 = fig.add_subplot(gs[0, 2])
        article_degrees = torch.bincount(edge_index[1], minlength=num_articles).numpy()
        ax3.hist(article_degrees, bins=30, color='coral', edgecolor='white')
        ax3.set_title('Article Degree Distribution')
        ax3.set_yscale('log')
        ax3.set_xlabel('Degree')
        
        # Power law check
        ax4 = fig.add_subplot(gs[1, 0])
        degree_counts = Counter(user_degrees)
        degrees = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees]
        ax4.loglog(degrees[1:], counts[1:], 'o', markersize=3, alpha=0.6)
        ax4.set_title('User Degree (Log-Log)')
        ax4.set_xlabel('Degree')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Feature magnitudes
        ax5 = fig.add_subplot(gs[1, 1])
        user_mags = np.linalg.norm(self.data['user'].x.numpy(), axis=1)
        ax5.hist(user_mags, bins=30, color='green', alpha=0.7, label='Users')
        article_mags = np.linalg.norm(self.data['article'].x.numpy(), axis=1)
        ax5.hist(article_mags, bins=30, color='purple', alpha=0.7, label='Articles')
        ax5.set_title('Feature Magnitudes')
        ax5.set_xlabel('L2 Norm')
        ax5.legend()
        
        # Pareto
        ax6 = fig.add_subplot(gs[1, 2])
        sorted_degrees = np.sort(user_degrees)[::-1]
        cumsum = np.cumsum(sorted_degrees) / sorted_degrees.sum()
        x_pct = np.arange(1, len(cumsum) + 1) / len(cumsum) * 100
        ax6.plot(x_pct, cumsum * 100, color='steelblue', linewidth=2)
        ax6.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
        ax6.set_title('User Pareto (Cumulative)')
        ax6.set_xlabel('% Users')
        ax6.set_ylabel('% Interactions')
        ax6.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / 'graph_eda_00_summary.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/graph_eda_00_summary.png")
        plt.close()
    
    def run_all(self):
        """Run all EDA analyses."""
        self.print_basic_stats()
        self.analyze_degree_distribution()
        self.analyze_features()
        self.analyze_connectivity()
        self.analyze_edge_weights()
        self.generate_summary()
        
        print("\n" + "=" * 60)
        print("Graph EDA Complete!")
        print(f"   Plots saved to: {self.output_dir.absolute()}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='EDA for Graph Data')
    parser.add_argument('--data-path', '-d',
                       default='data/processed/user_article_graph.pt',
                       help='Path to graph data')
    parser.add_argument('--output-dir', '-o',
                       default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    eda = GraphDataEDA(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    eda.run_all()


if __name__ == "__main__":
    main()
