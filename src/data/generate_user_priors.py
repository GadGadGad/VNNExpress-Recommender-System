import os
import pickle
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def generate_user_priors(data_path='data/processed', model_name='all-MiniLM-L6-v2'):
    """
    Generate user interest priors by embedding their interaction history content.
    For this 'lite' version, we average the embeddings of clicked articles.
    In a full LLMRec implementation, we would use an LLM to summarize the titles first.
    """
    print(f"Generating User Priors using {model_name}...")
    
    # Load processed data
    with open(os.path.join(data_path, 'lightgcl_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    train_dict = data['train_dict'] # user_idx -> list of item_idx
    idx2item = data['idx2item'] # item_idx -> url
    n_users = data['n_users']
    
    # Load articles to get text
    articles_df = pd.read_csv('data/raw/articles.csv')
    article_map = dict(zip(articles_df['url'], articles_df['title'].fillna("") + " " + articles_df['short_description'].fillna("")))
    
    # Load Sentence Transformer
    model = SentenceTransformer(model_name)
    
    user_priors = torch.zeros((n_users, 384)) # MiniLM dim is 384
    
    for u_idx, items in tqdm(train_dict.items(), desc="Encoding user histories"):
        if not items:
            continue
            
        # Get texts of clicked articles
        item_texts = []
        for i_idx in items:
            url = idx2item.get(i_idx)
            if url in article_map:
                item_texts.append(article_map[url])
        
        if item_texts:
            # Embed all articles and mean-pool
            # In a more advanced version, we'd feed item_texts to GPT to get a summary string first
            with torch.no_grad():
                embs = model.encode(item_texts, convert_to_tensor=True)
                user_priors[u_idx] = embs.mean(dim=0)
                
    output_path = os.path.join(data_path, 'user_priors.pt')
    torch.save(user_priors, output_path)
    print(f"User priors saved to {output_path}")

if __name__ == "__main__":
    generate_user_priors()
