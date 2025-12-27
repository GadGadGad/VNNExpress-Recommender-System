#!/usr/bin/env python3
"""
Visualize Graph Variants G1-G4 (Enhanced Version)
==================================================
Creates beautiful visual diagrams with clear weight representation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palette
COLORS = {
    'user': '#3498DB',      # Blue
    'article': '#2ECC71',   # Green
    'category': '#E74C3C',  # Red
    'author': '#9B59B6',    # Purple
    'edge_ua': '#7F8C8D',   # Gray
    'edge_social': '#E67E22', # Orange
    'edge_category': '#F1C40F', # Yellow
    'edge_author': '#1ABC9C',  # Teal
    'bg': '#F8F9FA',        # Light gray background
}


def draw_node(ax, x, y, label, node_type, size=0.4):
    """Draw a styled node with shadow effect."""
    color = COLORS.get(node_type, '#95A5A6')
    
    # Shadow
    shadow = FancyBboxPatch((x - size/2 + 0.02, y - size/2 - 0.02), size, size,
                            boxstyle="round,pad=0.05,rounding_size=0.1",
                            facecolor='#2C3E50', alpha=0.3, linewidth=0)
    ax.add_patch(shadow)
    
    # Main box
    if node_type == 'category':
        box = FancyBboxPatch((x - size/2, y - size/2), size, size,
                            boxstyle="round,pad=0.05,rounding_size=0.05",
                            facecolor=color, edgecolor='white', linewidth=2)
    elif node_type == 'author':
        # Diamond shape for authors
        diamond = plt.Polygon([(x, y + size/2), (x + size/2, y), (x, y - size/2), (x - size/2, y)],
                             facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        return
    else:
        box = FancyBboxPatch((x - size/2, y - size/2), size, size,
                            boxstyle="round,pad=0.05,rounding_size=0.15",
                            facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(box)
    
    # Label
    ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')


def draw_weighted_edge(ax, x1, y1, x2, y2, weight, color, show_weight=True, style='-'):
    """Draw an edge with thickness proportional to weight."""
    # Line width proportional to weight
    lw = 1 + weight * 1.5
    alpha = 0.4 + weight * 0.15
    
    if style == '--':
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, 
               linestyle='--', solid_capstyle='round', zorder=1)
    else:
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha,
               solid_capstyle='round', zorder=1)
    
    # Weight label
    if show_weight and weight > 0:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, f'w={weight}', fontsize=7, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))


def create_g1_bipartite():
    """G1: Simple User-Article Bipartite Graph"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(COLORS['bg'])
    
    # Node positions
    users = {'U1': (1, 6), 'U2': (1, 4.5), 'U3': (1, 3), 'U4': (1, 1.5)}
    articles = {'A1': (5, 7), 'A2': (5, 5.5), 'A3': (5, 4), 'A4': (5, 2.5), 'A5': (5, 1)}
    
    # Edges (all weight=1, unweighted)
    edges = [
        ('U1', 'A1', 1), ('U1', 'A2', 1),
        ('U2', 'A2', 1), ('U2', 'A3', 1),
        ('U3', 'A1', 1), ('U3', 'A3', 1), ('U3', 'A4', 1),
        ('U4', 'A4', 1), ('U4', 'A5', 1),
    ]
    
    # Draw edges
    for u, a, w in edges:
        x1, y1 = users[u]
        x2, y2 = articles[a]
        draw_weighted_edge(ax, x1 + 0.2, y1, x2 - 0.2, y2, w, COLORS['edge_ua'], show_weight=False)
    
    # Draw nodes
    for name, (x, y) in users.items():
        draw_node(ax, x, y, name, 'user')
    for name, (x, y) in articles.items():
        draw_node(ax, x, y, name, 'article')
    
    # Title and labels
    ax.set_title("G1: Bipartite Graph", fontsize=16, fontweight='bold', pad=20)
    ax.text(1, 7.5, "Users", ha='center', fontsize=12, fontweight='bold', color=COLORS['user'])
    ax.text(5, 7.5, "Articles", ha='center', fontsize=12, fontweight='bold', color=COLORS['article'])
    
    # Info box
    info = "• Binary edges (w=1)\n• No social signals\n• ~52k edges"
    ax.text(3, 0, info, ha='center', fontsize=10, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7'))
    
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-1, 8.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "g1_bipartite.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("✓ Saved: plots/g1_bipartite.png")


def create_g2_heterogeneous():
    """G2: Heterogeneous Graph with Weighted Edges and Social"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor(COLORS['bg'])
    
    # Node positions
    users = {'U1': (1, 6), 'U2': (1, 4), 'U3': (1, 2)}
    articles = {'A1': (6, 6.5), 'A2': (6, 4.5), 'A3': (6, 2.5), 'A4': (6, 0.5)}
    
    # User-Article edges with WEIGHTS
    ua_edges = [
        ('U1', 'A1', 3),  # High engagement
        ('U1', 'A2', 1),  # Low engagement
        ('U2', 'A2', 2),  # Medium
        ('U2', 'A3', 1),
        ('U3', 'A3', 3),
        ('U3', 'A4', 2),
    ]
    
    # User-User social edges
    social_edges = [('U1', 'U2'), ('U2', 'U3')]
    
    # Draw User-Article edges with weights
    for u, a, w in ua_edges:
        x1, y1 = users[u]
        x2, y2 = articles[a]
        draw_weighted_edge(ax, x1 + 0.2, y1, x2 - 0.2, y2, w, COLORS['edge_ua'], show_weight=True)
    
    # Draw Social edges
    for u1, u2 in social_edges:
        x1, y1 = users[u1]
        x2, y2 = users[u2]
        draw_weighted_edge(ax, x1, y1 - 0.2, x2, y2 + 0.2, 2, COLORS['edge_social'], 
                          show_weight=False, style='--')
    
    # Draw nodes
    for name, (x, y) in users.items():
        draw_node(ax, x, y, name, 'user')
    for name, (x, y) in articles.items():
        draw_node(ax, x, y, name, 'article')
    
    # Title
    ax.set_title("G2: Heterogeneous Graph (Weighted + Social)", fontsize=16, fontweight='bold', pad=20)
    ax.text(1, 7.5, "Users", ha='center', fontsize=12, fontweight='bold', color=COLORS['user'])
    ax.text(6, 7.5, "Articles", ha='center', fontsize=12, fontweight='bold', color=COLORS['article'])
    
    # Weight legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['edge_ua'], linewidth=2, label='w=1 (view)'),
        Line2D([0], [0], color=COLORS['edge_ua'], linewidth=4, label='w=2 (comment)'),
        Line2D([0], [0], color=COLORS['edge_ua'], linewidth=6, label='w=3 (reply)'),
        Line2D([0], [0], color=COLORS['edge_social'], linewidth=3, linestyle='--', label='Social (reply-to)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
    
    # Info box
    info = "• Weighted edges: w = (1 + log(reactions)) × decay(t)\n• Social edges from reply relationships\n• ~65k edges"
    ax.text(3.5, -0.8, info, ha='center', fontsize=10, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7'))
    
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-1.5, 8.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "g2_heterogeneous.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("✓ Saved: plots/g2_heterogeneous.png")


def create_g3_category_hubs():
    """G3: Category Hub Graph"""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor(COLORS['bg'])
    
    # Node positions - 3 columns
    users = {'U1': (1, 6), 'U2': (1, 4), 'U3': (1, 2)}
    categories = {'Sports': (5, 5.5), 'Tech': (5, 3), 'Biz': (5, 0.5)}
    articles = {'A1': (9, 6.5), 'A2': (9, 5), 'A3': (9, 3.5), 'A4': (9, 2), 'A5': (9, 0.5)}
    
    # User-Category edges (from reading history)
    uc_edges = [
        ('U1', 'Sports', 3), ('U1', 'Tech', 1),
        ('U2', 'Tech', 2), ('U2', 'Biz', 2),
        ('U3', 'Sports', 1), ('U3', 'Biz', 3),
    ]
    
    # Category-Article edges
    ca_edges = [
        ('Sports', 'A1', 1), ('Sports', 'A2', 1),
        ('Tech', 'A3', 1), ('Tech', 'A4', 1),
        ('Biz', 'A4', 1), ('Biz', 'A5', 1),
    ]
    
    # Draw edges
    for u, c, w in uc_edges:
        x1, y1 = users[u]
        x2, y2 = categories[c]
        draw_weighted_edge(ax, x1 + 0.2, y1, x2 - 0.3, y2, w, COLORS['edge_social'], show_weight=True)
    
    for c, a, w in ca_edges:
        x1, y1 = categories[c]
        x2, y2 = articles[a]
        draw_weighted_edge(ax, x1 + 0.3, y1, x2 - 0.2, y2, w, COLORS['edge_category'], show_weight=False)
    
    # Draw nodes
    for name, (x, y) in users.items():
        draw_node(ax, x, y, name, 'user')
    for name, (x, y) in categories.items():
        draw_node(ax, x, y, name, 'category', size=0.5)
    for name, (x, y) in articles.items():
        draw_node(ax, x, y, name, 'article')
    
    # Labels
    ax.set_title("G3: Category Hub Graph", fontsize=16, fontweight='bold', pad=20)
    ax.text(1, 7.5, "Users", ha='center', fontsize=12, fontweight='bold', color=COLORS['user'])
    ax.text(5, 7.5, "Categories\n(Hub Nodes)", ha='center', fontsize=11, fontweight='bold', color=COLORS['category'])
    ax.text(9, 7.5, "Articles", ha='center', fontsize=12, fontweight='bold', color=COLORS['article'])
    
    # Path illustration
    ax.annotate('', xy=(8, -0.5), xytext=(2, -0.5),
               arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    ax.text(5, -0.8, "User → Category → Article (Semantic Shortcut)", ha='center', fontsize=10, style='italic')
    
    # Info box
    info = "• Categories as semantic hubs (only 6 nodes)\n• Implicit User-Category from history\n• ~64k edges (efficient!)"
    ax.text(5, -1.5, info, ha='center', fontsize=10, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7'))
    
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-2.2, 8.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "g3_category_hubs.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("✓ Saved: plots/g3_category_hubs.png")


def create_g4_full_heterogeneous():
    """G4: Full Heterogeneous Graph with all node types"""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_facecolor(COLORS['bg'])
    
    # Node positions - arranged in layers
    users = {'U1': (1, 6), 'U2': (1, 3.5), 'U3': (1, 1)}
    articles = {'A1': (5, 7), 'A2': (5, 4.5), 'A3': (5, 2), 'A4': (5, -0.5)}
    categories = {'Sports': (9, 5.5), 'Tech': (9, 2)}
    authors = {'Auth1': (13, 5.5), 'Auth2': (13, 2)}
    
    # All edge types with weights
    ua_edges = [('U1', 'A1', 3), ('U1', 'A2', 1), ('U2', 'A2', 2), ('U2', 'A3', 2), ('U3', 'A3', 1), ('U3', 'A4', 3)]
    social_edges = [('U1', 'U2'), ('U2', 'U3')]
    ac_edges = [('A1', 'Sports'), ('A2', 'Sports'), ('A3', 'Tech'), ('A4', 'Tech')]
    aa_edges = [('A1', 'Auth1'), ('A2', 'Auth1'), ('A3', 'Auth2'), ('A4', 'Auth2')]
    
    # Draw all edges
    for u, a, w in ua_edges:
        x1, y1 = users[u]
        x2, y2 = articles[a]
        draw_weighted_edge(ax, x1 + 0.2, y1, x2 - 0.2, y2, w, COLORS['edge_ua'], show_weight=True)
    
    for u1, u2 in social_edges:
        x1, y1 = users[u1]
        x2, y2 = users[u2]
        draw_weighted_edge(ax, x1, y1 - 0.25, x2, y2 + 0.25, 2, COLORS['edge_social'], show_weight=False, style='--')
    
    for a, c in ac_edges:
        x1, y1 = articles[a]
        x2, y2 = categories[c]
        draw_weighted_edge(ax, x1 + 0.2, y1, x2 - 0.3, y2, 1, COLORS['edge_category'], show_weight=False)
    
    for a, auth in aa_edges:
        x1, y1 = articles[a]
        x2, y2 = authors[auth]
        draw_weighted_edge(ax, x1 + 0.2, y1 - 0.1, x2 - 0.25, y2, 1, COLORS['edge_author'], show_weight=False)
    
    # Draw nodes
    for name, (x, y) in users.items():
        draw_node(ax, x, y, name, 'user')
    for name, (x, y) in articles.items():
        draw_node(ax, x, y, name, 'article')
    for name, (x, y) in categories.items():
        draw_node(ax, x, y, name, 'category', size=0.5)
    for name, (x, y) in authors.items():
        draw_node(ax, x, y, name, 'author', size=0.5)
    
    # Labels
    ax.set_title("G4: Full Heterogeneous Graph (All Node & Edge Types)", fontsize=16, fontweight='bold', pad=20)
    ax.text(1, 8, "Users", ha='center', fontsize=12, fontweight='bold', color=COLORS['user'])
    ax.text(5, 8, "Articles", ha='center', fontsize=12, fontweight='bold', color=COLORS['article'])
    ax.text(9, 8, "Categories", ha='center', fontsize=11, fontweight='bold', color=COLORS['category'])
    ax.text(13, 8, "Authors", ha='center', fontsize=11, fontweight='bold', color=COLORS['author'])
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['user'], label='User'),
        mpatches.Patch(color=COLORS['article'], label='Article'),
        mpatches.Patch(color=COLORS['category'], label='Category'),
        mpatches.Patch(color=COLORS['author'], label='Author'),
        Line2D([0], [0], color=COLORS['edge_ua'], linewidth=3, label='User→Article (weighted)'),
        Line2D([0], [0], color=COLORS['edge_social'], linewidth=3, linestyle='--', label='User↔User (social)'),
        Line2D([0], [0], color=COLORS['edge_category'], linewidth=3, label='Article→Category'),
        Line2D([0], [0], color=COLORS['edge_author'], linewidth=3, label='Article→Author'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95, ncol=2)
    
    # Info
    info = "• 4 node types: User, Article, Category, Author\n• 4 edge types with semantic meaning\n• Full multi-aspect representation"
    ax.text(7, -1.5, info, ha='center', fontsize=10, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7'))
    
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(-2.5, 9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "g4_full_heterogeneous.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("✓ Saved: plots/g4_full_heterogeneous.png")


def create_comparison_summary():
    """Create a beautiful summary comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['bg'])
    
    graphs = [
        ("G1: Bipartite", "User ↔ Article", "Binary edges (w=1)", "52k", COLORS['user'], "Baseline"),
        ("G2: Heterogeneous", "G1 + Social", "Weighted + Time decay", "65k", COLORS['article'], "+248%"),
        ("G3: Category Hubs", "U ↔ Cat ↔ A", "Semantic shortcuts", "64k", COLORS['category'], "+984%"),
        ("G4: Full Hetero", "All types", "Multi-aspect fusion", "70k", COLORS['author'], "Best"),
    ]
    
    for ax, (title, structure, weights, edges, color, result) in zip(axes.flat, graphs):
        ax.set_facecolor('white')
        
        # Main title
        ax.text(0.5, 0.85, title, ha='center', va='center', fontsize=20, fontweight='bold',
               transform=ax.transAxes, color='#2C3E50')
        
        # Structure
        ax.text(0.5, 0.65, f"Structure: {structure}", ha='center', va='center', fontsize=13,
               transform=ax.transAxes, color='#7F8C8D')
        
        # Weights
        ax.text(0.5, 0.50, f"Weights: {weights}", ha='center', va='center', fontsize=12,
               transform=ax.transAxes, color='#95A5A6', style='italic')
        
        # Edges count
        ax.text(0.5, 0.35, f"Edges: ~{edges}", ha='center', va='center', fontsize=11,
               transform=ax.transAxes, color='#BDC3C7')
        
        # Result badge
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor='white', alpha=0.9)
        ax.text(0.5, 0.15, result, ha='center', va='center', fontsize=14, fontweight='bold',
               transform=ax.transAxes, color='white', bbox=bbox_props)
        
        # Colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle("Graph Variants Comparison (G1 → G4)", fontsize=18, fontweight='bold', 
                y=0.98, color='#2C3E50')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "graph_comparison_summary.png", dpi=150, bbox_inches='tight', 
               facecolor=COLORS['bg'])
    plt.close()
    print("✓ Saved: plots/graph_comparison_summary.png")


if __name__ == "__main__":
    print("=" * 50)
    print("Generating Enhanced Graph Visualizations...")
    print("=" * 50)
    
    create_g1_bipartite()
    create_g2_heterogeneous()
    create_g3_category_hubs()
    create_g4_full_heterogeneous()
    create_comparison_summary()
    
    print("\n" + "=" * 50)
    print("All visualizations saved to plots/")
    print("=" * 50)
