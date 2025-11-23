import pandas as pd
import numpy as np

# 1. Load the data
articles = pd.read_csv('data/articles.csv')
replies = pd.read_csv('data/replies.csv')

# 2. Clean User IDs
# Filter out garbage IDs like 'NO_COMMENT'
replies = replies[replies['parent_user_id'] != 'NO_COMMENT']

# Function to standardize IDs to strings (removes .0 from floats)
def clean_id(val):
    if pd.isna(val):
        return None
    try:
        return str(int(float(val)))
    except:
        return str(val)

replies['user_id_source'] = replies['parent_user_id'].apply(clean_id)
replies['user_id_target'] = replies['reply_user_id'].apply(clean_id)

# 3. Create Nodes
# Users: Combine all unique source and target user IDs
unique_users = pd.concat([replies['user_id_source'], replies['user_id_target']]).dropna().unique()
user_nodes = pd.DataFrame(unique_users, columns=['user_id'])
user_nodes['node_type'] = 'user'

# Articles: Just take the ID and URL
article_nodes = articles[['article_id', 'url']].copy()
article_nodes['node_type'] = 'article'

# 4. Create Edge Lists
# Edge Type 1: User COMMENTED_ON Article
# We assume parent_author is commenting on the article
edges_comment = replies[['user_id_source', 'article_url']].dropna().drop_duplicates()
edges_comment.columns = ['source', 'target']
edges_comment['edge_type'] = 'commented_on'

# Edge Type 2: User REPLIED_TO User
# Only take rows where a reply actually exists
edges_reply = replies[['user_id_target', 'user_id_source']].dropna().drop_duplicates()
edges_reply.columns = ['source', 'target']
edges_reply['edge_type'] = 'replied_to'

# 5. Output inspection
print(f"User Nodes: {len(user_nodes)}")
print(f"Article Nodes: {len(article_nodes)}")
print(f"Comment Edges: {len(edges_comment)}")
print(f"Reply Edges: {len(edges_reply)}")

# Example: Checking if we have articles for the comments
# In a graph DB (like Neo4j) or library (like NetworkX/PyG), you map the URLs to IDs.
