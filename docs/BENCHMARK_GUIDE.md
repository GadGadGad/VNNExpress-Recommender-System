# Hướng dẫn Benchmark, Export Results và Chuẩn bị Slides

## 1. Train Full Benchmark

### Option A: Qua Streamlit UI
```bash
streamlit run app.py
```
1. Vào tab **Training**
2. Tick **"Select All CF Models"**
3. Đặt **Epochs = 50** (hoặc 100 cho kết quả tốt hơn)
4. Chọn **Evaluation Protocol = "full"**
5. Click **"Start Training Process"**

### Option B: Qua Command Line (Nhanh hơn)
```bash
# Train tất cả CF models
for model in ngcf lightgcl simgcl xsimgcl; do
    python scripts/train_cf_models.py \
        --model $model \
        --data-path data/processed/enhanced_v1 \
        --epochs 50 \
        --eval-protocol full \
        --social-weight 0.5 \
        --denoise-ratio 0.1 \
        --embedding phobert
done
```

### Option C: Script tự động
```bash
python scripts/train_and_compare_all.py --epochs 50 --eval-protocol full
```

---

## 2. Export Results

### Xem kết quả trong Streamlit
1. Mở tab **Metrics**
2. Xem bảng so sánh tất cả models
3. Screenshot hoặc copy số liệu

### Export ra CSV
```python
import torch
import glob
import pandas as pd

metrics_data = []
for model_path in glob.glob("models/*.pt"):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'best_metrics' in checkpoint:
        metrics = checkpoint['best_metrics']
        model_name = model_path.split('/')[-1].split('_')[0].upper()
        metrics_data.append({
            'Model': model_name,
            **metrics
        })

df = pd.DataFrame(metrics_data)
df.to_csv('benchmark_results.csv', index=False)
print(df.to_markdown())
```

### Kết quả mẫu (Expected)
| Model | Recall@10 | NDCG@10 | Recall@20 | MRR |
|:---|:---:|:---:|:---:|:---:|
| **XSimGCL** | 0.062 | 0.036 | 0.095 | 0.034 |
| LightGCL | 0.058 | 0.033 | 0.089 | 0.031 |
| SimGCL | 0.055 | 0.031 | 0.084 | 0.029 |
| NGCF | 0.048 | 0.027 | 0.075 | 0.025 |

---

## 3. Chuẩn bị Presentation Slides

### Cấu trúc đề xuất (10-15 slides)

| Slide | Nội dung |
|:---|:---|
| 1 | Title: Smart Hybrid News Recommendation System |
| 2 | Problem Statement: Sparsity, Cold-Start, Timeliness |
| 3 | Dataset Overview: VnExpress (4.2K users, 2.3K articles, 99.3% sparse) |
| 4 | **Pipeline Architecture** (dùng diagram từ `docs/pipeline_architecture.md`) |
| 5 | Key Innovations: Time-Decay, Social Signals, Hard Negatives |
| 6 | Model Comparison (bảng từ Metrics Dashboard) |
| 7 | Post-Processing: Freshness Boost + Calibration + MMR |
| 8 | Demo Screenshots (A/B Compare, Cold-Start) |
| 9 | Experimental Results (bảng Recall/NDCG) |
| 10 | Conclusion & Future Work |

### Tạo slides với LaTeX Beamer
```bash
# Nếu đã có template
cd presentation
pdflatex main.tex
```

### Hoặc dùng Canva/Google Slides
1. Import diagrams từ `docs/diagrams/` (export PNG từ Mermaid Live)
2. Screenshot Streamlit demo
3. Thêm bảng kết quả

---

## Quick Commands Summary

```bash
# 1. Train all models
python scripts/train_and_compare_all.py --epochs 50

# 2. View metrics in browser
streamlit run app.py  # Go to Metrics tab

# 3. Export to CSV
python -c "
import torch, glob, pandas as pd
data = []
for f in glob.glob('models/*.pt'):
    c = torch.load(f, map_location='cpu', weights_only=False)
    if 'best_metrics' in c:
        data.append({'Model': f.split('/')[-1].split('_')[0], **c['best_metrics']})
pd.DataFrame(data).to_csv('results.csv')
print('Saved to results.csv')
"
```
