#!/usr/bin/env python3
"""
Merge all category data into a single dataset for EDA pipeline
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
CATEGORIES = ['giaoduc', 'khcn', 'kinhdoanh', 'thegioi', 'thethao', 'thoisu']
OUTPUT_DIR = BASE_DIR / 'data_merged'
OUTPUT_DIR.mkdir(exist_ok=True)

def merge_csvs(filename: str):
    """Merge CSV files from all categories"""
    dfs = []
    for cat in CATEGORIES:
        path = BASE_DIR / f'data_small_{cat}' / filename
        if path.exists():
            try:
                df = pd.read_csv(path, on_bad_lines='skip')
                df['source_category'] = cat
                dfs.append(df)
                print(f"  ✓ {cat}: {len(df):,} rows")
            except Exception as e:
                print(f"  ✗ {cat}: {e}")
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        output_path = OUTPUT_DIR / filename
        merged.to_csv(output_path, index=False)
        print(f"  → Saved: {output_path} ({len(merged):,} rows)\n")
        return merged
    return None

print("=" * 60)
print("Merging Category Data")
print("=" * 60)

print("\n📰 Merging articles.csv...")
articles = merge_csvs('articles.csv')

print("💬 Merging replies.csv...")
replies = merge_csvs('replies.csv')

print("👥 Merging user_profiles.csv...")
users = merge_csvs('user_profiles.csv')

print("🏷️ Merging metadata.csv...")
metadata = merge_csvs('metadata.csv')

print("=" * 60)
print(f"✅ All data merged to: {OUTPUT_DIR}")
print("=" * 60)
