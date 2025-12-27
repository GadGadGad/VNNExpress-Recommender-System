"""
Comprehensive EDA for VnExpress Crawled Data
---------------------------------------------
This script performs Exploratory Data Analysis on:
- articles.csv: Article metadata
- replies.csv: Comments and replies
- user_profiles.csv: User profile information

Run: python src/eda_crawled_data.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'palette': 'viridis'
}


class CrawledDataEDA:
    """Main class for performing EDA on crawled VnExpress data."""
    
    def __init__(self, data_dir='data/raw', output_dir='plots'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        print("=" * 60)
        print("VnExpress Crawled Data - Exploratory Data Analysis")
        print("=" * 60)
        
        self.articles = self._load_articles()
        self.replies = self._load_replies()
        self.users = self._load_users()
        self.metadata = self._load_metadata()
        
    def _load_articles(self):
        """Load and preprocess articles data."""
        path = self.data_dir / 'articles.csv'
        print(f"\nLoading articles from {path}...")
        df = pd.read_csv(path)
        print(f"   → {len(df):,} articles loaded")
        return df
    
    def _load_replies(self):
        """Load and preprocess replies data."""
        path = self.data_dir / 'replies.csv'
        print(f"Loading replies from {path}...")
        df = pd.read_csv(path)
        # Filter out NO_COMMENT markers
        df_clean = df[df['parent_user_id'] != 'NO_COMMENT'].copy()
        print(f"   → {len(df):,} total rows, {len(df_clean):,} actual comments")
        return df_clean
    
    def _load_users(self):
        """Load user profiles data."""
        path = self.data_dir / 'user_profiles.csv'
        print(f"Loading user profiles from {path}...")
        df = pd.read_csv(path)
        print(f"   → {len(df):,} users loaded")
        return df
    
    def _load_metadata(self):
        """Load metadata (categories and tags)."""
        path = self.data_dir / 'metadata.csv'
        if path.exists():
            print(f"Loading metadata from {path}...")
            df = pd.read_csv(path)
            
            # Filter outliers (count < 10)
            if 'category' in df.columns:
                counts = df['category'].value_counts()
                to_keep = counts[counts >= 10].index
                initial_len = len(df)
                df = df[df['category'].isin(to_keep)]
                print(f"   → {len(df):,} metadata entries loaded (Filtered {initial_len - len(df)} outliers)")
            else:
                print(f"   → {len(df):,} metadata entries loaded")
            return df
        else:
            print(f"   [SKIP] metadata.csv not found")
            return None
    
    def print_basic_stats(self):
        """Print basic statistics about the datasets."""
        print("\n" + "=" * 60)
        print("BASIC STATISTICS")
        print("=" * 60)
        
        # Articles stats
        print("\n--- Articles Dataset ---")
        print(f"Total articles: {len(self.articles):,}")
        print(f"Columns: {list(self.articles.columns)}")
        print(f"Unique authors: {self.articles['author'].nunique()}")
        
        # Replies stats
        print("\n--- Replies Dataset ---")
        print(f"Total comment rows: {len(self.replies):,}")
        print(f"Unique articles with comments: {self.replies['article_url'].nunique():,}")
        print(f"Unique parent commenters: {self.replies['parent_user_id'].nunique():,}")
        
        # Count distinct reply users
        reply_users = self.replies['reply_user_id'].dropna()
        reply_users = reply_users[reply_users != '']
        print(f"Unique reply users: {reply_users.nunique():,}")
        
        # Users stats
        print("\n--- Users Dataset ---")
        print(f"Total user profiles: {len(self.users):,}")
    
    def analyze_tags(self):
        """Analyze article tags distribution."""
        # Check if tags column exists in articles
        if 'tags' not in self.articles.columns:
            print("\n[SKIP] Tags analysis - 'tags' column not in articles.csv")
            print("       (Tags may be in metadata.csv instead)")
            return
        
        print("\nAnalyzing tags distribution...")
        
        # Parse tags (they are stored as string representation of lists)
        import ast
        
        all_tags = []
        for tags_str in self.articles['tags'].fillna('[]'):
            try:
                tags_list = ast.literal_eval(tags_str) if isinstance(tags_str, str) else []
                all_tags.extend(tags_list)
            except:
                pass
        
        if not all_tags:
            print("   No tags found to analyze.")
            return
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Top 20 tags
        top_tags = dict(tag_counts.most_common(20))
        
        bars = ax.barh(list(top_tags.keys())[::-1], list(top_tags.values())[::-1], 
                      color=plt.cm.viridis(np.linspace(0, 0.8, len(top_tags))))
        ax.set_xlabel('Number of Articles')
        ax.set_title('Top 20 Article Tags')
        ax.bar_label(bars, fmt='%d', padding=3, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_01_tags_distribution.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_01_tags_distribution.png")
        plt.close()
    
    def analyze_metadata_categories(self):
        """Analyze category distribution from metadata.csv."""
        if self.metadata is None:
            print("   [SKIP] No metadata available")
            return
        
        print("\nAnalyzing category distribution (from metadata)...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Top 15 categories
        top_cats = self.metadata['category'].value_counts().head(15)
        
        ax1 = axes[0]
        bars = ax1.barh(top_cats.index[::-1], top_cats.values[::-1], 
                       color=plt.cm.viridis(np.linspace(0, 0.8, 15)))
        ax1.set_xlabel('Number of Articles')
        ax1.set_title('Top 15 Article Categories')
        ax1.bar_label(bars, fmt='%d', padding=3, fontsize=9)
        
        # Category pie chart (top 10 + Others)
        ax2 = axes[1]
        top_10 = self.metadata['category'].value_counts().head(10)
        others = self.metadata['category'].value_counts()[10:].sum()
        pie_data = pd.concat([top_10, pd.Series({'Others': others})])
        
        wedges, texts, autotexts = ax2.pie(
            pie_data.values, 
            labels=pie_data.index, 
            autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
            colors=plt.cm.viridis(np.linspace(0, 0.9, len(pie_data))),
            startangle=90
        )
        ax2.set_title('Category Distribution (Top 10 + Others)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_09_metadata_categories.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_09_metadata_categories.png")
        plt.close()
    
    def analyze_metadata_tags(self):
        """Analyze tags distribution from metadata.csv."""
        if self.metadata is None:
            print("   [SKIP] No metadata available")
            return
        
        print("Analyzing tags distribution (from metadata)...")
        
        import ast
        from collections import Counter
        
        all_tags = []
        for tags_str in self.metadata['tags'].fillna('[]'):
            try:
                tags_list = ast.literal_eval(tags_str) if isinstance(tags_str, str) else []
                all_tags.extend(tags_list)
            except:
                pass
        
        if not all_tags:
            print("   No tags found in metadata.")
            return
        
        tag_counts = Counter(all_tags)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Top 25 tags bar chart
        ax1 = axes[0]
        top_tags = dict(tag_counts.most_common(25))
        bars = ax1.barh(list(top_tags.keys())[::-1], list(top_tags.values())[::-1], 
                       color=plt.cm.plasma(np.linspace(0.2, 0.8, len(top_tags))))
        ax1.set_xlabel('Number of Articles')
        ax1.set_title('Top 25 Article Tags (from Metadata)')
        ax1.bar_label(bars, fmt='%d', padding=3, fontsize=8)
        
        # Tag frequency distribution
        ax2 = axes[1]
        tag_freq = list(tag_counts.values())
        ax2.hist(tag_freq, bins=50, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
        ax2.set_xlabel('Number of Articles per Tag')
        ax2.set_ylabel('Number of Tags')
        ax2.set_title(f'Tag Frequency Distribution\n({len(tag_counts):,} unique tags)')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_10_metadata_tags.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_10_metadata_tags.png")
        plt.close()
    
    def analyze_engagement_by_category(self):
        """Analyze engagement (comments) by category using metadata.csv."""
        if self.metadata is None:
            print("   [SKIP] No metadata available")
            return
        
        print("Analyzing engagement by category (from metadata)...")
        
        # Merge metadata with replies to get category info
        comments_per_article = self.replies.groupby('article_url').size().reset_index(name='comment_count')
        
        merged = self.metadata.merge(
            comments_per_article, left_on='article_url', right_on='article_url', how='left'
        )
        merged['comment_count'] = merged['comment_count'].fillna(0)
        
        # Calculate average comments per category
        category_engagement = merged.groupby('category').agg({
            'comment_count': ['mean', 'sum', 'count']
        }).round(2)
        category_engagement.columns = ['avg_comments', 'total_comments', 'num_articles']
        category_engagement = category_engagement.sort_values('avg_comments', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Top 15 categories by average comments
        ax1 = axes[0]
        top_15 = category_engagement.head(15)
        bars = ax1.barh(range(len(top_15)), top_15['avg_comments'].values,
                        color=plt.cm.viridis(np.linspace(0, 0.8, 15)))
        ax1.set_yticks(range(len(top_15)))
        ax1.set_yticklabels(top_15.index)
        ax1.set_xlabel('Average Comments per Article')
        ax1.set_title('Most Engaging Categories (by Avg Comments)')
        ax1.invert_yaxis()
        ax1.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
        
        # Top 15 by total comments
        ax2 = axes[1]
        top_15_total = category_engagement.nlargest(15, 'total_comments')
        bars2 = ax2.barh(range(len(top_15_total)), top_15_total['total_comments'].values,
                         color=plt.cm.plasma(np.linspace(0.2, 0.8, 15)))
        ax2.set_yticks(range(len(top_15_total)))
        ax2.set_yticklabels(top_15_total.index)
        ax2.set_xlabel('Total Comments')
        ax2.set_title('Categories with Most Total Comments')
        ax2.invert_yaxis()
        ax2.bar_label(bars2, fmt='%d', padding=3, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_11_engagement_by_category.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_11_engagement_by_category.png")
        plt.close()
    
    def analyze_comments_per_article(self):
        """Analyze comment distribution across articles."""
        print("Analyzing comments per article...")
        
        comments_per_article = self.replies.groupby('article_url').size()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram with log scale
        ax1 = axes[0]
        ax1.hist(comments_per_article, bins=50, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
        ax1.set_xlabel('Number of Comments')
        ax1.set_ylabel('Number of Articles')
        ax1.set_title('Distribution of Comments per Article')
        ax1.set_yscale('log')
        
        # Add statistics box
        stats_text = f"Mean: {comments_per_article.mean():.1f}\n"
        stats_text += f"Median: {comments_per_article.median():.1f}\n"
        stats_text += f"Max: {comments_per_article.max()}\n"
        stats_text += f"Std: {comments_per_article.std():.1f}"
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Box plot
        ax2 = axes[1]
        bp = ax2.boxplot(comments_per_article, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['secondary'])
        ax2.set_ylabel('Number of Comments')
        ax2.set_title('Comments per Article (Box Plot)')
        ax2.set_xticklabels(['All Articles'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_02_comments_per_article.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_02_comments_per_article.png")
        plt.close()
    
    def analyze_user_activity(self):
        """Analyze user activity patterns (Power Law)."""
        print("👥 Analyzing user activity patterns...")
        
        # Combine parent and reply user activity
        parent_activity = self.replies['parent_user_id'].value_counts()
        reply_users = self.replies['reply_user_id'].dropna()
        reply_users = reply_users[reply_users != '']
        reply_activity = reply_users.value_counts()
        
        # Total activity per user
        all_users = pd.concat([
            self.replies['parent_user_id'],
            reply_users
        ])
        total_activity = all_users.value_counts()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Log-log plot (Power Law)
        ax1 = axes[0, 0]
        ranks = np.arange(1, len(total_activity) + 1)
        ax1.loglog(ranks, total_activity.values, 'o', markersize=2, alpha=0.5, color=COLORS['primary'])
        ax1.set_xlabel('User Rank (log scale)')
        ax1.set_ylabel('Number of Interactions (log scale)')
        ax1.set_title('User Activity: Rank-Frequency Plot (Power Law)')
        ax1.grid(True, alpha=0.3)
        
        # Top 20 most active users
        ax2 = axes[0, 1]
        top_20 = total_activity.head(20)
        bars = ax2.barh(range(len(top_20)), top_20.values, color=plt.cm.plasma(np.linspace(0.2, 0.8, 20)))
        ax2.set_yticks(range(len(top_20)))
        ax2.set_yticklabels([str(uid)[:15] + '...' if len(str(uid)) > 15 else str(uid) for uid in top_20.index])
        ax2.set_xlabel('Number of Interactions')
        ax2.set_title('Top 20 Most Active Users')
        ax2.invert_yaxis()
        
        # Activity distribution histogram
        ax3 = axes[1, 0]
        ax3.hist(total_activity.values, bins=50, color=COLORS['tertiary'], edgecolor='white', alpha=0.8)
        ax3.set_xlabel('Number of Interactions per User')
        ax3.set_ylabel('Number of Users')
        ax3.set_title('User Activity Distribution')
        ax3.set_yscale('log')
        
        # Cumulative distribution
        ax4 = axes[1, 1]
        sorted_activity = np.sort(total_activity.values)[::-1]
        cumsum = np.cumsum(sorted_activity) / sorted_activity.sum()
        ax4.plot(np.arange(1, len(cumsum) + 1) / len(cumsum) * 100, cumsum * 100, 
                 color=COLORS['quaternary'], linewidth=2)
        ax4.set_xlabel('Percentage of Users')
        ax4.set_ylabel('Percentage of Total Activity')
        ax4.set_title('Cumulative Activity Distribution (Pareto)')
        ax4.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% activity')
        ax4.axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='20% users')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_03_user_activity.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_03_user_activity.png")
        plt.close()
    
    def analyze_reactions(self):
        """Analyze reaction patterns."""
        print("Analyzing reaction patterns...")
        
        # Convert reactions to numeric
        self.replies['parent_reactions_num'] = pd.to_numeric(self.replies['parent_reactions'], errors='coerce').fillna(0)
        self.replies['reply_reactions_num'] = pd.to_numeric(self.replies['reply_reactions'], errors='coerce').fillna(0)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Parent reactions distribution
        ax1 = axes[0]
        parent_reactions = self.replies['parent_reactions_num']
        parent_reactions = parent_reactions[parent_reactions > 0]
        if len(parent_reactions) > 0:
            ax1.hist(parent_reactions, bins=50, color=COLORS['primary'], edgecolor='white', alpha=0.8)
            ax1.set_xlabel('Number of Reactions')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Parent Comment Reactions Distribution')
            ax1.set_yscale('log')
        
        # Reply reactions distribution
        ax2 = axes[1]
        reply_reactions = self.replies['reply_reactions_num']
        reply_reactions = reply_reactions[reply_reactions > 0]
        if len(reply_reactions) > 0:
            ax2.hist(reply_reactions, bins=50, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
            ax2.set_xlabel('Number of Reactions')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Reply Reactions Distribution')
            ax2.set_yscale('log')
        
        # Top reacted comments
        ax3 = axes[2]
        top_reacted = self.replies.nlargest(15, 'parent_reactions_num')[['parent_author', 'parent_reactions_num']]
        bars = ax3.barh(range(len(top_reacted)), top_reacted['parent_reactions_num'].values,
                        color=plt.cm.Reds(np.linspace(0.3, 0.9, 15)))
        ax3.set_yticks(range(len(top_reacted)))
        ax3.set_yticklabels([str(a)[:15] + '...' if len(str(a)) > 15 else str(a) for a in top_reacted['parent_author']])
        ax3.set_xlabel('Reactions')
        ax3.set_title('Top 15 Most Reacted Comments')
        ax3.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_04_reactions.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_04_reactions.png")
        plt.close()
    
    def analyze_text_length(self):
        """Analyze text length distribution."""
        print("Analyzing text lengths...")
        
        # Calculate text lengths
        self.articles['title_len'] = self.articles['title'].fillna('').str.len()
        self.articles['content_len'] = self.articles['content'].fillna('').str.len()
        self.replies['comment_len'] = self.replies['parent_text'].fillna('').str.len()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Article title length
        ax1 = axes[0]
        ax1.hist(self.articles['title_len'], bins=50, color=COLORS['primary'], edgecolor='white', alpha=0.8)
        ax1.set_xlabel('Title Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Article Title Length Distribution\nMean: {self.articles["title_len"].mean():.0f} chars')
        ax1.axvline(self.articles['title_len'].mean(), color='red', linestyle='--', label='Mean')
        ax1.legend()
        
        # Article content length
        ax2 = axes[1]
        content_len = self.articles['content_len'][self.articles['content_len'] > 0]
        ax2.hist(content_len, bins=50, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
        ax2.set_xlabel('Content Length (characters)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Article Content Length Distribution\nMean: {content_len.mean():.0f} chars')
        ax2.axvline(content_len.mean(), color='red', linestyle='--', label='Mean')
        ax2.legend()
        
        # Comment length
        ax3 = axes[2]
        comment_len = self.replies['comment_len'][self.replies['comment_len'] > 0]
        ax3.hist(comment_len, bins=50, color=COLORS['tertiary'], edgecolor='white', alpha=0.8)
        ax3.set_xlabel('Comment Length (characters)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Comment Length Distribution\nMean: {comment_len.mean():.0f} chars')
        ax3.axvline(comment_len.mean(), color='red', linestyle='--', label='Mean')
        ax3.set_yscale('log')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_05_text_lengths.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_05_text_lengths.png")
        plt.close()
    
    def analyze_reply_depth(self):
        """Analyze reply patterns - comments with vs without replies."""
        print("Analyzing reply patterns...")
        
        # Count replies per parent comment
        has_reply = self.replies['reply_text'].fillna('').str.len() > 0
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart: comments with replies vs without
        ax1 = axes[0]
        reply_counts = [has_reply.sum(), (~has_reply).sum()]
        labels = ['Has Replies', 'No Replies']
        colors = [COLORS['primary'], COLORS['secondary']]
        wedges, texts, autotexts = ax1.pie(
            reply_counts, labels=labels, autopct='%1.1f%%', 
            colors=colors, startangle=90
        )
        ax1.set_title('Comments with vs without Replies')
        
        # Replies per parent comment
        ax2 = axes[1]
        replies_per_parent = self.replies.groupby(['article_url', 'parent_user_id']).size()
        ax2.hist(replies_per_parent, bins=30, color=COLORS['tertiary'], edgecolor='white', alpha=0.8)
        ax2.set_xlabel('Number of Reply Records per Comment Thread')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Thread Size Distribution')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_06_reply_patterns.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_06_reply_patterns.png")
        plt.close()
    
    def analyze_author_productivity(self):
        """Analyze article author productivity."""
        print("Analyzing author productivity...")
        
        author_counts = self.articles['author'].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top 20 authors
        ax1 = axes[0]
        top_20 = author_counts.head(20)
        bars = ax1.barh(range(len(top_20)), top_20.values, 
                        color=plt.cm.coolwarm(np.linspace(0.2, 0.8, 20)))
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels([str(a)[:20] + '...' if len(str(a)) > 20 else str(a) for a in top_20.index])
        ax1.set_xlabel('Number of Articles')
        ax1.set_title('Top 20 Most Prolific Authors')
        ax1.invert_yaxis()
        ax1.bar_label(bars, fmt='%d', padding=3, fontsize=8)
        
        # Author productivity distribution
        ax2 = axes[1]
        ax2.hist(author_counts.values, bins=50, color=COLORS['quaternary'], edgecolor='white', alpha=0.8)
        ax2.set_xlabel('Number of Articles per Author')
        ax2.set_ylabel('Number of Authors')
        ax2.set_title('Author Productivity Distribution')
        ax2.set_yscale('log')
        
        # Add statistics
        stats_text = f"Total Authors: {len(author_counts):,}\n"
        stats_text += f"Mean: {author_counts.mean():.1f}\n"
        stats_text += f"Median: {author_counts.median():.1f}\n"
        stats_text += f"Max: {author_counts.max()}"
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_07_author_productivity.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_07_author_productivity.png")
        plt.close()
    
    def analyze_engagement_overall(self):
        """Analyze overall engagement patterns."""
        print("Analyzing engagement patterns...")
        
        comments_per_article = self.replies.groupby('article_url').size().reset_index(name='comment_count')
        
        # Merge with articles to get titles
        merged = self.articles[['url', 'title']].merge(
            comments_per_article, left_on='url', right_on='article_url', how='left'
        )
        merged['comment_count'] = merged['comment_count'].fillna(0)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Top 20 most commented articles
        ax1 = axes[0]
        top_20 = merged.nlargest(20, 'comment_count')
        bars = ax1.barh(range(len(top_20)), top_20['comment_count'].values,
                        color=plt.cm.viridis(np.linspace(0, 0.8, 20)))
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels([t[:40] + '...' if len(str(t)) > 40 else t for t in top_20['title']])
        ax1.set_xlabel('Number of Comments')
        ax1.set_title('Top 20 Most Commented Articles')
        ax1.invert_yaxis()
        ax1.bar_label(bars, fmt='%d', padding=3, fontsize=9)
        
        # Comment count distribution
        ax2 = axes[1]
        ax2.hist(merged['comment_count'], bins=50, color=plt.cm.plasma(0.5), edgecolor='white', alpha=0.8)
        ax2.set_xlabel('Number of Comments')
        ax2.set_ylabel('Number of Articles')
        ax2.set_title('Distribution of Comments per Article')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_08_engagement_patterns.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_08_engagement_patterns.png")
        plt.close()
    
    def generate_summary_dashboard(self):
        """Generate a summary dashboard with key metrics."""
        print("Generating summary dashboard...")
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('VnExpress Crawled Data - Summary Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. Key Metrics (text)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        metrics = f"""
        Total Articles: {len(self.articles):,}
        Total Comments: {len(self.replies):,}
        Total Users: {len(self.users):,}
        
        Authors: {self.articles['author'].nunique()}
        Commented Articles: {self.replies['article_url'].nunique():,}
        """
        ax1.text(0.1, 0.5, metrics, fontsize=12, verticalalignment='center', 
                 family='monospace', transform=ax1.transAxes)
        ax1.set_title('Key Metrics', fontweight='bold', fontsize=12)
        
        # 2. Top 10 Authors
        ax2 = fig.add_subplot(gs[0, 1:])
        top_authors = self.articles['author'].value_counts().head(10)
        ax2.barh(top_authors.index[::-1], top_authors.values[::-1], color=plt.cm.viridis(np.linspace(0, 0.8, 10)))
        ax2.set_title('Top 10 Authors', fontweight='bold')
        ax2.set_xlabel('Articles')
        
        # 3. Comments per Article
        ax3 = fig.add_subplot(gs[1, 0])
        comments_per_article = self.replies.groupby('article_url').size()
        ax3.hist(comments_per_article, bins=30, color=COLORS['secondary'], edgecolor='white')
        ax3.set_title('Comments/Article', fontweight='bold')
        ax3.set_yscale('log')
        ax3.set_xlabel('Comments')
        ax3.set_ylabel('Frequency')
        
        # 4. User Activity Power Law
        ax4 = fig.add_subplot(gs[1, 1])
        all_users = pd.concat([
            self.replies['parent_user_id'],
            self.replies['reply_user_id'].dropna()
        ])
        user_activity = all_users.value_counts()
        ax4.loglog(range(1, len(user_activity) + 1), user_activity.values, 'o', 
                   markersize=1, alpha=0.5, color=COLORS['primary'])
        ax4.set_title('User Activity (Power Law)', fontweight='bold')
        ax4.set_xlabel('User Rank')
        ax4.set_ylabel('Interactions')
        
        # 5. Top 10 Authors
        ax5 = fig.add_subplot(gs[1, 2])
        top_authors = self.articles['author'].value_counts().head(10)
        ax5.barh(top_authors.index[::-1], top_authors.values[::-1], 
                 color=plt.cm.coolwarm(np.linspace(0.2, 0.8, 10)))
        ax5.set_title('Top 10 Authors', fontweight='bold')
        ax5.set_xlabel('Articles')
        
        # 6. Text Length Distributions
        ax6 = fig.add_subplot(gs[2, 0])
        title_lens = self.articles['title'].fillna('').str.len()
        ax6.hist(title_lens, bins=30, color=COLORS['tertiary'], edgecolor='white')
        ax6.set_title('Title Lengths', fontweight='bold')
        ax6.set_xlabel('Characters')
        
        ax7 = fig.add_subplot(gs[2, 1])
        comment_lens = self.replies['parent_text'].fillna('').str.len()
        comment_lens = comment_lens[comment_lens > 0]
        ax7.hist(comment_lens, bins=30, color=COLORS['quaternary'], edgecolor='white')
        ax7.set_title('Comment Lengths', fontweight='bold')
        ax7.set_xlabel('Characters')
        ax7.set_yscale('log')
        
        # 7. Reactions Distribution  
        ax8 = fig.add_subplot(gs[2, 2])
        reactions = pd.to_numeric(self.replies['parent_reactions'], errors='coerce').fillna(0)
        reactions = reactions[reactions > 0]
        if len(reactions) > 0:
            ax8.hist(reactions, bins=30, color=COLORS['primary'], edgecolor='white')
            ax8.set_yscale('log')
        ax8.set_title('Reaction Distribution', fontweight='bold')
        ax8.set_xlabel('Reactions')
        
        plt.savefig(self.output_dir / 'eda_00_summary_dashboard.png', dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {self.output_dir}/eda_00_summary_dashboard.png")
        plt.close()
    
    def run_all(self):
        """Run all EDA analyses."""
        self.print_basic_stats()
        
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        self.generate_summary_dashboard()
        self.analyze_tags()
        self.analyze_comments_per_article()
        self.analyze_user_activity()
        self.analyze_reactions()
        self.analyze_text_length()
        self.analyze_reply_depth()
        self.analyze_author_productivity()
        self.analyze_engagement_overall()
        
        # Metadata-based analysis (if metadata.csv exists)
        self.analyze_metadata_categories()
        self.analyze_metadata_tags()
        self.analyze_engagement_by_category()
        
        print("\n" + "=" * 60)
        print("EDA COMPLETE!")
        print(f"   All plots saved to: {self.output_dir.absolute()}")
        print("=" * 60)


def main():
    """Main entry point."""
    eda = CrawledDataEDA(
        data_dir='data/raw',
        output_dir='plots'
    )
    eda.run_all()


if __name__ == "__main__":
    main()
