# LightGCL for Recommendation

Implementation of LightGCL (Light Graph Contrastive Learning) for recommendation systems.

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

Đặt data vào thư mục `data/raw/`:
- `user_profiles.csv`: Thông tin users
- `articles.csv`: Thông tin items/articles
- `replies.csv`: User-item interactions

## Configuration

Chỉnh sửa `configs/lightgcl_config.yaml`:

```yaml
data:
  user_col: "user_id"      # Tên cột user trong CSV
  item_col: "article_id"   # Tên cột item trong CSV
```

## Usage

### 1. Explore data
```bash
python -m src.data.dataloader
```

### 2. Train model
```bash
python scripts/run_lightgcl.py --config configs/lightgcl_config.yaml
```

### 3. With custom options
```bash
python scripts/run_lightgcl.py \
    --config configs/lightgcl_config.yaml \
    --device cuda \
    --force_reload
```

## Project Structure

```
DS300-Final-Project/
├── data/
│   ├── raw/           # Raw CSV files
│   └── processed/     # Processed data
├── src/
│   ├── data/          # Data loading
│   ├── models/        # LightGCL model
│   ├── trainers/      # Training logic
│   └── utils/         # Metrics, helpers
├── configs/           # Configuration files
├── scripts/           # Run scripts
├── checkpoints/       # Saved models
└── notebooks/         # Jupyter notebooks
```

## References

- LightGCL Paper: https://arxiv.org/abs/2302.08191
- LightGCN Paper: https://arxiv.org/abs/2002.02126


## Chạy thử
```
# 1. Cài dependencies
cd /media/spidey/Spidey1/SEMESTER_7/DS300/DS300-Final-Project
pip install -r requirements.txt

# 2. Xem structure data trước
python -m src.data.dataloader

# 3. Sau khi biết tên cột, sửa config rồi train
python scripts/run_lightgcl.py --force_reload
```