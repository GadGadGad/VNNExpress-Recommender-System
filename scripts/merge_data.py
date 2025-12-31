#!/usr/bin/env python3
"""
Merge data from multiple data_small_* directories into one combined dataset.
"""
import pandas as pd
from pathlib import Path
import argparse


def merge_data(input_pattern="crawlers/data_small_*", output_dir="data/raw"):
    """Merge all data_small_* directories into one."""
    
    input_dirs = sorted(Path(".").glob(input_pattern))
    print(f"Found {len(input_dirs)} directories to merge:")
    for d in input_dirs:
        print(f"  - {d}")
    
    if not input_dirs:
        print("No directories found matching pattern!")
        return
    
    # Initialize empty dataframes
    all_articles = []
    all_metadata = []
    all_replies = []
    all_user_profiles = []
    
    for data_dir in input_dirs:
        category = data_dir.name.replace("data_small_", "")
        print(f"\nProcessing {category}...")
        
        # Load articles
        articles_path = data_dir / "articles.csv"
        if articles_path.exists():
            df = pd.read_csv(articles_path)
            df['source_category'] = category  # Add source tracking
            all_articles.append(df)
            print(f"  Articles: {len(df):,}")
        
        # Load metadata
        metadata_path = data_dir / "metadata.csv"
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            all_metadata.append(df)
            print(f"  Metadata: {len(df):,}")
        
        # Load replies
        replies_path = data_dir / "replies.csv"
        if replies_path.exists():
            df = pd.read_csv(replies_path)
            all_replies.append(df)
            print(f"  Replies: {len(df):,}")
        
        # Load user_profiles
        users_path = data_dir / "user_profiles.csv"
        if users_path.exists():
            df = pd.read_csv(users_path)
            all_user_profiles.append(df)
            print(f"  User Profiles: {len(df):,}")
    
    # Merge all dataframes
    print("\n" + "=" * 50)
    print("MERGING DATA")
    print("=" * 50)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Merge and deduplicate articles
    if all_articles:
        merged_articles = pd.concat(all_articles, ignore_index=True)
        # Deduplicate by URL (keep first occurrence)
        before = len(merged_articles)
        merged_articles = merged_articles.drop_duplicates(subset=['url'], keep='first')
        after = len(merged_articles)
        merged_articles.to_csv(output_path / "articles.csv", index=False)
        print(f"Articles: {before:,} -> {after:,} (removed {before-after:,} duplicates)")
    
    # Merge and deduplicate metadata
    if all_metadata:
        merged_metadata = pd.concat(all_metadata, ignore_index=True)
        before = len(merged_metadata)
        merged_metadata = merged_metadata.drop_duplicates(subset=['article_url'], keep='first')
        after = len(merged_metadata)
        merged_metadata.to_csv(output_path / "metadata.csv", index=False)
        print(f"Metadata: {before:,} -> {after:,} (removed {before-after:,} duplicates)")
    
    # Merge replies (may have duplicates if same comment appears in multiple crawls)
    if all_replies:
        merged_replies = pd.concat(all_replies, ignore_index=True)
        before = len(merged_replies)
        # Deduplicate by article_url + parent_user_id + parent_text (or just keep all)
        merged_replies = merged_replies.drop_duplicates(
            subset=['article_url', 'parent_user_id', 'parent_text'], 
            keep='first'
        )
        after = len(merged_replies)
        merged_replies.to_csv(output_path / "replies.csv", index=False)
        print(f"Replies: {before:,} -> {after:,} (removed {before-after:,} duplicates)")
    
    # Merge user_profiles
    if all_user_profiles:
        merged_users = pd.concat(all_user_profiles, ignore_index=True)
        before = len(merged_users)
        merged_users = merged_users.drop_duplicates(subset=['user_id'], keep='first')
        after = len(merged_users)
        merged_users.to_csv(output_path / "user_profiles.csv", index=False)
        print(f"User Profiles: {before:,} -> {after:,} (removed {before-after:,} duplicates)")
    
    print(f"\n[OK] Merged data saved to: {output_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge data_small_* directories")
    parser.add_argument("--input", "-i", default="crawlers/data_small_*",
                        help="Glob pattern for input directories")
    parser.add_argument("--output", "-o", default="data/raw",
                        help="Output directory for merged data")
    args = parser.parse_args()
    
    merge_data(args.input, args.output)
