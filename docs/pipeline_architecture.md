# Smart Hybrid News Recommendation Pipeline

## Kiến trúc tổng quan

```mermaid
flowchart TB
    subgraph DataCollection["📥 Data Collection"]
        A1[VnExpress Articles]
        A2[User Comments/Replies]
        A3[User Profiles]
    end

    subgraph Preprocessing["📊 Preprocessing"]
        B1[K-Core Filtering]
        B2[ID Mapping]
        B3["Time Decay Weighting"]
        B4["Social Edge Extraction"]
        B5[Feature Extraction]
    end

    subgraph Training["🧠 Model Training"]
        subgraph CF["Collaborative Filtering"]
            C1[XSimGCL]
            C2[LightGCL]
            C3[SimGCL]
            C4[NGCF]
        end
        subgraph CB["Content-Based"]
            C5[PhoBERT]
            C6[TF-IDF]
        end
    end

    subgraph Inference["🔮 Inference"]
        D1[CF Scoring]
        D2[CB Scoring]
        D3["Hybrid Blending: α·CF + (1-α)·CB"]
    end

    subgraph PostProcess["📰 Post-Processing"]
        E1["1. Freshness Boost"]
        E2["2. Calibration (KL-Div)"]
        E3["3. MMR Diversity"]
    end

    F[📋 Top-K Recommendations]

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2 --> B3 --> B4 --> B5
    B5 --> C1 & C2 & C3 & C4
    B5 --> C5 & C6
    C1 & C2 & C3 & C4 --> D1
    C5 & C6 --> D2
    D1 --> D3
    D2 --> D3
    D3 --> E1 --> E2 --> E3 --> F
```

---

## Chi tiết từng giai đoạn

### 1. Data Collection
| Module | File | Chức năng |
|:---|:---|:---|
| Article Crawler | `crawlers/main_crawler.py` | Thu thập bài báo (title, content, category) |
| Comment Crawler | `crawlers/comment_crawler.py` | Thu thập comments và replies |
| Profile Crawler | `crawlers/user_profile_crawler.py` | Thu thập thông tin user |

### 2. Preprocessing
```mermaid
flowchart LR
    subgraph Input["Raw Data"]
        R1[articles.csv]
        R2[replies.csv]
    end

    subgraph Process["convert_to_gnn.py"]
        P1["K-Core Filtering<br/>(min_interactions ≥ 3)"]
        P2["ID Mapping<br/>(user_id → idx)"]
        P3["Time Decay<br/>w = e^(-0.01 × days)"]
        P4["Social Edges<br/>(replied_to)"]
        P5["PhoBERT Embeddings"]
    end

    subgraph Output["Processed"]
        O1[graph_with_negatives.pt]
        O2[full_hetero_graph.pt]
        O3[user_map.json]
        O4[article_map.json]
    end

    R1 --> P1
    R2 --> P1
    P1 --> P2 --> P3 --> P4 --> P5
    P5 --> O1 & O2 & O3 & O4
```

| Bước | Công thức/Mô tả |
|:---|:---|
| K-Core Filtering | Loại bỏ user/item có ít hơn N tương tác |
| Time Decay | $w_{edge} = e^{-\lambda \cdot \Delta t}$ với $\lambda = 0.01$ |
| Social Edges | Trích xuất `(replier → parent)` từ replies |

### 3. Model Training

#### Collaborative Filtering Models
```mermaid
flowchart LR
    subgraph GNN["Graph Neural Network"]
        G1["User Embeddings<br/>(N × d)"]
        G2["Item Embeddings<br/>(M × d)"]
        G3["Adjacency Matrix<br/>(Normalized)"]
    end

    subgraph Propagation["Message Passing"]
        P1["Layer 1"]
        P2["Layer 2"]
        P3["Layer 3"]
    end

    subgraph Loss["Training Loss"]
        L1["BPR Loss"]
        L2["SSL Loss (InfoNCE)"]
        L3["Regularization"]
    end

    G1 --> P1
    G2 --> P1
    G3 --> P1
    P1 --> P2 --> P3
    P3 --> L1 & L2 & L3
```

| Model | Đặc điểm chính |
|:---|:---|
| **XSimGCL** | Noise-based contrastive learning, không cần augmentation phức tạp |
| LightGCL | SVD-based augmentation, hiệu quả trên sparse data |
| SimGCL | Uniform noise perturbation |
| NGCF | Bipartite graph convolution cơ bản |

#### Content-Based Models
| Model | Input | Output |
|:---|:---|:---|
| PhoBERT | Vietnamese text | 768-dim embeddings |
| TF-IDF | Bag-of-words | Sparse vectors → Cosine similarity |

### 4. Inference (Scoring)

```mermaid
flowchart TB
    subgraph CF["CF Scoring"]
        CF1["user_emb = model.user_embedding[user_idx]"]
        CF2["item_emb = model.item_embedding"]
        CF3["cf_score = user_emb @ item_emb.T"]
    end

    subgraph CB["CB Scoring"]
        CB1["user_profile = mean(history_embeddings)"]
        CB2["cb_score = cosine(user_profile, article_emb)"]
    end

    subgraph Blend["Hybrid Blending"]
        H1["final = α × cf_score + (1-α) × cb_score"]
    end

    CF1 --> CF2 --> CF3
    CB1 --> CB2
    CF3 --> H1
    CB2 --> H1
```

### 5. Post-Processing

```mermaid
flowchart LR
    subgraph Stage1["Stage 1: Freshness"]
        F1["score_new = (1-w)·score + w·e^(-λ·days)"]
    end

    subgraph Stage2["Stage 2: Calibration"]
        C1["Minimize KL(p_rec || p_history)"]
        C2["Greedy selection"]
    end

    subgraph Stage3["Stage 3: Diversity"]
        D1["MMR = λ·relevance - (1-λ)·max_sim"]
    end

    Stage1 --> Stage2 --> Stage3
```

| Stage | Mục đích | Tham số |
|:---|:---|:---|
| **Freshness Boost** | Ưu tiên tin mới (quan trọng cho News) | `freshness_lambda=0.1`, `boost_weight=0.2` |
| **Calibration** | Đảm bảo tỷ lệ category khớp sở thích user | `alpha=0.5` |
| **MMR Diversity** | Tránh gợi ý nhiều bài giống nhau | `lambda_mmr=0.5` |

---

## Thống kê dữ liệu

| Metric | Giá trị |
|:---|:---|
| **Số Users** | 4,266 |
| **Số Articles** | 2,299 |
| **Số Interactions** | 64,944 |
| **Sparsity** | 99.34% |
| **Social Edges** | 26,359 |

---

## Công thức chính

### Time Decay Weight
$$w_{edge} = e^{-\lambda \cdot (t_{now} - t_{interaction})}$$

### Hybrid Blending
$$score_{final} = \alpha \cdot score_{CF} + (1-\alpha) \cdot score_{CB}$$

### Freshness Boost
$$score_{boosted} = (1-w) \cdot score_{rel} + w \cdot e^{-\lambda \cdot days\_old}$$

### KL-Divergence Calibration
$$\min \sum_c p_{rec}(c) \cdot \log\frac{p_{rec}(c)}{p_{history}(c)}$$

### Maximal Marginal Relevance
$$MMR = \lambda \cdot Relevance(d) - (1-\lambda) \cdot \max_{d' \in S} Similarity(d, d')$$
