#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for VNExpress Crawler Data
Analyzes merged data and individual categories from data_small_* folders
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import ast
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# Base directory
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Categories
CATEGORIES = ['giaoduc', 'khcn', 'kinhdoanh', 'thegioi', 'thethao', 'thoisu']
CATEGORY_NAMES = {
    'giaoduc': 'Giáo dục',
    'gocnhin': 'Góc nhìn', 
    'khcn': 'Khoa học CN',
    'kinhdoanh': 'Kinh doanh',
    'thegioi': 'Thế giới',
    'thethao': 'Thể thao',
    'thoisu': 'Thời sự'
}


def load_category_data(category: str) -> dict:
    """Load all CSV files for a category"""
    folder = BASE_DIR / f"data_small_{category}"
    data = {}
    
    for file_name in ['articles.csv', 'replies.csv', 'user_profiles.csv', 'metadata.csv']:
        file_path = folder / file_name
        if file_path.exists():
            try:
                data[file_name.replace('.csv', '')] = pd.read_csv(file_path, on_bad_lines='skip')
            except Exception as e:
                print(f"  Warning: Error loading {file_path}: {e}")
                data[file_name.replace('.csv', '')] = pd.DataFrame()
        else:
            data[file_name.replace('.csv', '')] = pd.DataFrame()
    
    return data


def merge_all_categories() -> dict:
    """Merge data from all categories"""
    merged = {
        'articles': [],
        'replies': [],
        'user_profiles': [],
        'metadata': []
    }
    
    for category in CATEGORIES:
        print(f"Loading {CATEGORY_NAMES.get(category, category)}...")
        data = load_category_data(category)
        
        for key in merged:
            if not data[key].empty:
                df = data[key].copy()
                df['category_folder'] = category
                merged[key].append(df)
    
    result = {}
    for key in merged:
        if merged[key]:
            result[key] = pd.concat(merged[key], ignore_index=True)
        else:
            result[key] = pd.DataFrame()
    
    return result


def basic_stats(df: pd.DataFrame, name: str) -> dict:
    """Get basic statistics for a dataframe"""
    stats = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    return stats


def analyze_articles(df: pd.DataFrame, category_name: str = "All"):
    """Analyze articles dataframe"""
    print(f"\n{'='*60}")
    print(f"ARTICLES ANALYSIS - {category_name}")
    print(f"{'='*60}")
    
    if df.empty:
        print("  No data available")
        return {}
    
    stats = {
        'total_articles': len(df),
        'unique_articles': df['article_id'].nunique() if 'article_id' in df.columns else len(df),
        'unique_authors': df['author'].nunique() if 'author' in df.columns else 0,
    }
    
    print(f"  Total articles: {stats['total_articles']:,}")
    print(f"  Unique articles: {stats['unique_articles']:,}")
    print(f"  Unique authors: {stats['unique_authors']:,}")
    
    # Content length analysis
    if 'content' in df.columns:
        df['content_length'] = df['content'].fillna('').str.len()
        stats['avg_content_length'] = df['content_length'].mean()
        stats['median_content_length'] = df['content_length'].median()
        print(f"  Avg content length: {stats['avg_content_length']:,.0f} chars")
        print(f"  Median content length: {stats['median_content_length']:,.0f} chars")
    
    # Title length analysis
    if 'title' in df.columns:
        df['title_length'] = df['title'].fillna('').str.len()
        stats['avg_title_length'] = df['title_length'].mean()
        print(f"  Avg title length: {stats['avg_title_length']:.1f} chars")
    
    # Top authors
    if 'author' in df.columns:
        top_authors = df['author'].value_counts().head(10)
        stats['top_authors'] = top_authors.to_dict()
        print(f"\n  Top 5 Authors:")
        for author, count in top_authors.head(5).items():
            print(f"    - {author}: {count:,} articles")
    
    # Missing values
    print(f"\n  Missing values:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"    - {col}: {missing:,} ({missing/len(df)*100:.1f}%)")
    
    return stats


def analyze_replies(df: pd.DataFrame, category_name: str = "All"):
    """Analyze replies/comments dataframe"""
    print(f"\n{'='*60}")
    print(f"REPLIES ANALYSIS - {category_name}")
    print(f"{'='*60}")
    
    if df.empty:
        print("  No data available")
        return {}
    
    stats = {
        'total_comments': len(df),
        'unique_articles': df['article_url'].nunique() if 'article_url' in df.columns else 0,
        'unique_users': df['parent_user_id'].nunique() if 'parent_user_id' in df.columns else 0,
    }
    
    print(f"  Total comments: {stats['total_comments']:,}")
    print(f"  Articles with comments: {stats['unique_articles']:,}")
    print(f"  Unique users commenting: {stats['unique_users']:,}")
    
    # Comments per article
    if 'article_url' in df.columns:
        comments_per_article = df.groupby('article_url').size()
        stats['avg_comments_per_article'] = comments_per_article.mean()
        stats['max_comments_per_article'] = comments_per_article.max()
        print(f"  Avg comments per article: {stats['avg_comments_per_article']:.1f}")
        print(f"  Max comments on an article: {stats['max_comments_per_article']:,}")
    
    # Reactions analysis
    if 'parent_reactions' in df.columns:
        df['parent_reactions'] = pd.to_numeric(df['parent_reactions'], errors='coerce')
        stats['avg_reactions'] = df['parent_reactions'].mean()
        stats['max_reactions'] = df['parent_reactions'].max()
        print(f"  Avg reactions per comment: {stats['avg_reactions']:.1f}")
        print(f"  Max reactions: {stats['max_reactions']:,.0f}")
    
    # Comment length
    if 'parent_text' in df.columns:
        df['comment_length'] = df['parent_text'].fillna('').str.len()
        stats['avg_comment_length'] = df['comment_length'].mean()
        print(f"  Avg comment length: {stats['avg_comment_length']:.0f} chars")
    
    # Reply analysis (nested comments)
    if 'reply_text' in df.columns:
        has_replies = df['reply_text'].notna().sum()
        stats['comments_with_replies'] = has_replies
        print(f"  Comments with nested replies: {has_replies:,}")
    
    # Top commenters
    if 'parent_author' in df.columns:
        top_commenters = df['parent_author'].value_counts().head(10)
        stats['top_commenters'] = top_commenters.to_dict()
        print(f"\n  Top 5 Commenters:")
        for user, count in top_commenters.head(5).items():
            print(f"    - {user}: {count:,} comments")
    
    return stats


def analyze_users(df: pd.DataFrame, category_name: str = "All"):
    """Analyze user profiles dataframe"""
    print(f"\n{'='*60}")
    print(f"USER PROFILES ANALYSIS - {category_name}")
    print(f"{'='*60}")
    
    if df.empty:
        print("  No data available")
        return {}
    
    stats = {
        'total_users': len(df),
        'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else len(df),
    }
    
    print(f"  Total user profiles: {stats['total_users']:,}")
    print(f"  Unique users: {stats['unique_users']:,}")
    
    # Join date analysis
    if 'join_date' in df.columns:
        # Parse join dates (format: MM/YYYY)
        df['join_date_parsed'] = pd.to_datetime(df['join_date'], format='%m/%Y', errors='coerce')
        valid_dates = df['join_date_parsed'].notna().sum()
        if valid_dates > 0:
            stats['earliest_join'] = df['join_date_parsed'].min()
            stats['latest_join'] = df['join_date_parsed'].max()
            print(f"  Valid join dates: {valid_dates:,}")
            print(f"  Earliest join: {stats['earliest_join']}")
            print(f"  Latest join: {stats['latest_join']}")
            
            # Join date distribution by year
            df['join_year'] = df['join_date_parsed'].dt.year
            year_dist = df['join_year'].value_counts().sort_index()
            stats['join_year_dist'] = year_dist.to_dict()
    
    return stats


def analyze_metadata(df: pd.DataFrame, category_name: str = "All"):
    """Analyze metadata dataframe"""
    print(f"\n{'='*60}")
    print(f"METADATA ANALYSIS - {category_name}")
    print(f"{'='*60}")
    
    if df.empty:
        print("  No data available")
        return {}
    
    stats = {
        'total_records': len(df),
        'unique_articles': df['article_url'].nunique() if 'article_url' in df.columns else 0,
    }
    
    print(f"  Total metadata records: {stats['total_records']:,}")
    print(f"  Unique articles: {stats['unique_articles']:,}")
    
    # Category distribution
    if 'category' in df.columns:
        cat_dist = df['category'].value_counts()
        stats['category_distribution'] = cat_dist.to_dict()
        print(f"\n  Category distribution:")
        for cat, count in cat_dist.head(10).items():
            print(f"    - {cat}: {count:,}")
    
    # Tag analysis
    if 'tags' in df.columns:
        all_tags = []
        for tags_str in df['tags'].dropna():
            try:
                tags = ast.literal_eval(tags_str)
                if isinstance(tags, list):
                    all_tags.extend(tags)
            except:
                pass
        
        if all_tags:
            tag_counts = Counter(all_tags)
            stats['total_unique_tags'] = len(tag_counts)
            stats['top_tags'] = dict(tag_counts.most_common(20))
            print(f"\n  Total unique tags: {stats['total_unique_tags']:,}")
            print(f"  Top 10 tags:")
            for tag, count in tag_counts.most_common(10):
                print(f"    - {tag}: {count:,}")
    
    return stats


def create_visualizations(merged_data: dict, category_stats: dict):
    """Create and save visualizations"""
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Articles per category
    if category_stats:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Articles count
        categories = list(category_stats.keys())
        articles_counts = [category_stats[c]['articles'].get('total_articles', 0) for c in categories]
        category_labels = [CATEGORY_NAMES.get(c, c) for c in categories]
        
        ax1 = axes[0, 0]
        bars1 = ax1.bar(category_labels, articles_counts, color=sns.color_palette("husl", len(categories)))
        ax1.set_title('Số lượng bài viết theo danh mục', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Danh mục')
        ax1.set_ylabel('Số bài viết')
        ax1.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars1, articles_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        # Comments count
        comments_counts = [category_stats[c]['replies'].get('total_comments', 0) for c in categories]
        ax2 = axes[0, 1]
        bars2 = ax2.bar(category_labels, comments_counts, color=sns.color_palette("husl", len(categories)))
        ax2.set_title('Số lượng bình luận theo danh mục', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Danh mục')
        ax2.set_ylabel('Số bình luận')
        ax2.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars2, comments_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        # Users count
        users_counts = [category_stats[c]['users'].get('total_users', 0) for c in categories]
        ax3 = axes[1, 0]
        bars3 = ax3.bar(category_labels, users_counts, color=sns.color_palette("husl", len(categories)))
        ax3.set_title('Số lượng người dùng theo danh mục', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Danh mục')
        ax3.set_ylabel('Số người dùng')
        ax3.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars3, users_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        # Avg comments per article
        avg_comments = [category_stats[c]['replies'].get('avg_comments_per_article', 0) for c in categories]
        ax4 = axes[1, 1]
        bars4 = ax4.bar(category_labels, avg_comments, color=sns.color_palette("husl", len(categories)))
        ax4.set_title('Trung bình bình luận/bài viết', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Danh mục')
        ax4.set_ylabel('Số bình luận TB')
        ax4.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars4, avg_comments):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'category_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: category_comparison.png")
        plt.close()
    
    # Content length distribution (merged)
    if 'articles' in merged_data and not merged_data['articles'].empty:
        df = merged_data['articles'].copy()
        if 'content' in df.columns:
            df['content_length'] = df['content'].fillna('').str.len()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            ax1 = axes[0]
            ax1.hist(df['content_length'], bins=50, edgecolor='black', alpha=0.7)
            ax1.set_title('Phân bố độ dài nội dung bài viết', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Độ dài (ký tự)')
            ax1.set_ylabel('Số lượng')
            ax1.axvline(df['content_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["content_length"].mean():,.0f}')
            ax1.axvline(df['content_length'].median(), color='green', linestyle='--', label=f'Median: {df["content_length"].median():,.0f}')
            ax1.legend()
            
            # Box plot by category
            ax2 = axes[1]
            if 'category_folder' in df.columns:
                df['category_name'] = df['category_folder'].map(CATEGORY_NAMES)
                df.boxplot(column='content_length', by='category_name', ax=ax2)
                ax2.set_title('Độ dài nội dung theo danh mục', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Danh mục')
                ax2.set_ylabel('Độ dài (ký tự)')
                plt.suptitle('')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'content_length_distribution.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: content_length_distribution.png")
            plt.close()
    
    # Reactions distribution
    if 'replies' in merged_data and not merged_data['replies'].empty:
        df = merged_data['replies'].copy()
        if 'parent_reactions' in df.columns:
            df['parent_reactions'] = pd.to_numeric(df['parent_reactions'], errors='coerce')
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram (log scale for better visualization)
            ax1 = axes[0]
            reactions = df['parent_reactions'].dropna()
            reactions_nonzero = reactions[reactions > 0]
            if len(reactions_nonzero) > 0:
                ax1.hist(reactions_nonzero, bins=50, edgecolor='black', alpha=0.7)
                ax1.set_yscale('log')
                ax1.set_title('Phân bố số reactions (log scale)', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Số reactions')
                ax1.set_ylabel('Số lượng (log)')
            
            # Box plot by category
            ax2 = axes[1]
            if 'category_folder' in df.columns:
                df['category_name'] = df['category_folder'].map(CATEGORY_NAMES)
                # Filter outliers for better visualization
                q99 = df['parent_reactions'].quantile(0.99)
                df_filtered = df[df['parent_reactions'] <= q99]
                df_filtered.boxplot(column='parent_reactions', by='category_name', ax=ax2)
                ax2.set_title('Reactions theo danh mục (loại outliers)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Danh mục')
                ax2.set_ylabel('Số reactions')
                plt.suptitle('')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'reactions_distribution.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: reactions_distribution.png")
            plt.close()
    
    # User join date distribution
    if 'user_profiles' in merged_data and not merged_data['user_profiles'].empty:
        df = merged_data['user_profiles'].copy()
        if 'join_date' in df.columns:
            df['join_date_parsed'] = pd.to_datetime(df['join_date'], format='%m/%Y', errors='coerce')
            df['join_year'] = df['join_date_parsed'].dt.year
            
            year_counts = df['join_year'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            bars = ax.bar(year_counts.index.astype(int), year_counts.values, edgecolor='black', alpha=0.7)
            ax.set_title('Phân bố năm tham gia của người dùng', fontsize=12, fontweight='bold')
            ax.set_xlabel('Năm')
            ax.set_ylabel('Số người dùng')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'user_join_distribution.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: user_join_distribution.png")
            plt.close()
    
    # Top tags word cloud / bar chart
    if 'metadata' in merged_data and not merged_data['metadata'].empty:
        df = merged_data['metadata'].copy()
        if 'tags' in df.columns:
            all_tags = []
            for tags_str in df['tags'].dropna():
                try:
                    tags = ast.literal_eval(tags_str)
                    if isinstance(tags, list):
                        all_tags.extend(tags)
                except:
                    pass
            
            if all_tags:
                tag_counts = Counter(all_tags).most_common(20)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                tags = [t[0] for t in tag_counts]
                counts = [t[1] for t in tag_counts]
                
                bars = ax.barh(range(len(tags)), counts, color=sns.color_palette("husl", len(tags)))
                ax.set_yticks(range(len(tags)))
                ax.set_yticklabels(tags)
                ax.invert_yaxis()
                ax.set_title('Top 20 Tags phổ biến nhất', fontsize=12, fontweight='bold')
                ax.set_xlabel('Số lượng')
                
                for i, count in enumerate(counts):
                    ax.text(count + 5, i, f'{count:,}', va='center', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / 'top_tags.png', dpi=150, bbox_inches='tight')
                print(f"  Saved: top_tags.png")
                plt.close()


def generate_summary_report(merged_stats: dict, category_stats: dict):
    """Generate a summary markdown report"""
    report = ["# VNExpress Crawler Data - EDA Report\n"]
    report.append(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Overview
    report.append("\n## Tổng quan\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Tổng số bài viết | {merged_stats['articles'].get('total_articles', 0):,} |")
    report.append(f"| Tổng số bình luận | {merged_stats['replies'].get('total_comments', 0):,} |")
    report.append(f"| Tổng số người dùng | {merged_stats['users'].get('total_users', 0):,} |")
    report.append(f"| Số danh mục | {len(CATEGORIES)} |")
    
    # Category comparison
    report.append("\n## So sánh theo danh mục\n")
    report.append("| Danh mục | Bài viết | Bình luận | Người dùng | TB BL/bài |")
    report.append("|----------|----------|-----------|------------|-----------|")
    for cat in CATEGORIES:
        if cat in category_stats:
            name = CATEGORY_NAMES.get(cat, cat)
            articles = category_stats[cat]['articles'].get('total_articles', 0)
            comments = category_stats[cat]['replies'].get('total_comments', 0)
            users = category_stats[cat]['users'].get('total_users', 0)
            avg = category_stats[cat]['replies'].get('avg_comments_per_article', 0)
            report.append(f"| {name} | {articles:,} | {comments:,} | {users:,} | {avg:.1f} |")
    
    # Top authors
    if 'top_authors' in merged_stats['articles']:
        report.append("\n## Top 10 tác giả (theo số bài)\n")
        report.append("| Tác giả | Số bài viết |")
        report.append("|---------|-------------|")
        for author, count in list(merged_stats['articles']['top_authors'].items())[:10]:
            report.append(f"| {author} | {count:,} |")
    
    # Top commenters
    if 'top_commenters' in merged_stats['replies']:
        report.append("\n## Top 10 người bình luận\n")
        report.append("| Người dùng | Số bình luận |")
        report.append("|------------|--------------|")
        for user, count in list(merged_stats['replies']['top_commenters'].items())[:10]:
            report.append(f"| {user} | {count:,} |")
    
    # Top tags
    if 'top_tags' in merged_stats['metadata']:
        report.append("\n## Top 20 tags phổ biến\n")
        report.append("| Tag | Số lượng |")
        report.append("|-----|----------|")
        for tag, count in list(merged_stats['metadata']['top_tags'].items())[:20]:
            report.append(f"| {tag} | {count:,} |")
    
    # Visualizations
    report.append("\n## Visualizations\n")
    report.append("- `category_comparison.png`: So sánh các chỉ số theo danh mục")
    report.append("- `content_length_distribution.png`: Phân bố độ dài nội dung")
    report.append("- `reactions_distribution.png`: Phân bố reactions")
    report.append("- `user_join_distribution.png`: Phân bố năm tham gia")
    report.append("- `top_tags.png`: Top tags phổ biến")
    
    # Write report
    report_path = OUTPUT_DIR / "eda_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\n  Saved report: {report_path}")


def main():
    print("=" * 70)
    print("VNExpress Crawler Data - Exploratory Data Analysis")
    print("=" * 70)
    
    # Analyze each category
    print("\n" + "=" * 70)
    print("ANALYZING INDIVIDUAL CATEGORIES")
    print("=" * 70)
    
    category_stats = {}
    for category in CATEGORIES:
        print(f"\n{'─'*60}")
        print(f"Category: {CATEGORY_NAMES.get(category, category)} ({category})")
        print(f"{'─'*60}")
        
        data = load_category_data(category)
        
        stats = {
            'articles': analyze_articles(data['articles'], CATEGORY_NAMES.get(category, category)),
            'replies': analyze_replies(data['replies'], CATEGORY_NAMES.get(category, category)),
            'users': analyze_users(data['user_profiles'], CATEGORY_NAMES.get(category, category)),
            'metadata': analyze_metadata(data['metadata'], CATEGORY_NAMES.get(category, category))
        }
        category_stats[category] = stats
    
    # Merge and analyze all
    print("\n" + "=" * 70)
    print("ANALYZING MERGED DATA (ALL CATEGORIES)")
    print("=" * 70)
    
    merged_data = merge_all_categories()
    
    merged_stats = {
        'articles': analyze_articles(merged_data['articles'], "ALL CATEGORIES (Merged)"),
        'replies': analyze_replies(merged_data['replies'], "ALL CATEGORIES (Merged)"),
        'users': analyze_users(merged_data['user_profiles'], "ALL CATEGORIES (Merged)"),
        'metadata': analyze_metadata(merged_data['metadata'], "ALL CATEGORIES (Merged)")
    }
    
    # Create visualizations
    create_visualizations(merged_data, category_stats)
    
    # Generate report
    generate_summary_report(merged_stats, category_stats)
    
    print("\n" + "=" * 70)
    print("EDA COMPLETED!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
