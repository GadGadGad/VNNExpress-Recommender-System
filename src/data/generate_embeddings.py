import torch
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def generate_embeddings(
    articles_path: str,
    output_path: str,
    model_name: str = "vinai/phobert-base",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_content: bool = False
):
    """
    Generate embeddings for articles using a pre-trained language model.
    """
    print(f"Loading articles from {articles_path}...")
    df = pd.read_csv(articles_path)
    
    # Fill NA
    df['title'] = df['title'].fillna("")
    df['short_description'] = df['short_description'].fillna("")
    if 'content' in df.columns:
        df['content'] = df['content'].fillna("")
    
    # Prepare text
    # Strategy: title + " " + short_description. 
    # If use_content is True, append content (truncated by model max_len).
    print("Preparing text data...")
    texts = []
    for _, row in df.iterrows():
        text = f"{row['title']} {row['short_description']}"
        if use_content and 'content' in row and row['content']:
             text += f" {row['content']}"
        texts.append(text)
        
    print(f"Loading model: {model_name} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model. Is 'transformers' installed? {e}")
        return

    # Process in batches
    embeddings = {}
    print(f"Generating embeddings for {len(texts)} articles...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            batch_urls = df.iloc[i : i + batch_size]['url'].values
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass
            outputs = model(**encoded)
            
            # Get [CLS] token embedding (first token)
            # shape: (batch_size, hidden_dim)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Save to dict
            for url, emb in zip(batch_urls, cls_embeddings):
                embeddings[url] = emb.cpu()

    # Save
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(embeddings)} embeddings to {output_path}...")
    torch.save(embeddings, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--articles', default='data/raw/articles.csv')
    parser.add_argument('--output', default='data/processed/phobert_embeddings.pt')
    parser.add_argument('--model', default='vinai/phobert-base')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-content', action='store_true', help='Include full content in embedding')
    args = parser.parse_args()
    
    generate_embeddings(
        articles_path=args.articles,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        use_content=args.use_content
    )
