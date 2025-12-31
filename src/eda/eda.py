import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(articles_path, replies_path, output_dir='plots'):
    print(f"--- [EDA] Analyzing {articles_path} & {replies_path} ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Data
    articles = pd.read_csv(articles_path)
    replies = pd.read_csv(replies_path)

    # Clean Replies
    replies = replies[replies['parent_user_id'] != 'NO_COMMENT']

    # --- 1. Article Count Summary ---
    print(f"Total Articles: {len(articles)}")

    # --- 2. Comments per Article (Long Tail) ---
    comments_per_article = replies['article_url'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.histplot(comments_per_article, bins=30, kde=False, color='orange')
    plt.title('Distribution of Comments per Article')
    plt.xlabel('Number of Comments')
    plt.yscale('log') # Dùng thang log để dễ nhìn phần đuôi
    plt.ylabel('Count (Log Scale)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_comments_per_article.png')
    print(f"Saved: {output_dir}/2_comments_per_article.png")

    # --- 3. User Activity (Power Law) ---
    # Gộp cả người comment và người reply
    all_users = pd.concat([replies['parent_user_id'], replies['reply_user_id'].dropna().astype(str)])
    user_counts = all_users.value_counts().values

    plt.figure(figsize=(10, 5))
    plt.plot(user_counts)
    plt.title('User Activity (Rank-Frequency Plot)')
    plt.xlabel('User Rank')
    plt.ylabel('Number of Interactions')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_user_activity_powerlaw.png')
    print(f"Saved: {output_dir}/3_user_activity_powerlaw.png")

    print("--- [EDA] Finished ---")

if __name__ == "__main__":
    run_eda('data/raw/articles.csv', 'data/raw/replies.csv')
