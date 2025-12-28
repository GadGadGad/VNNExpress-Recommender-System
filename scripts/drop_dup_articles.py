import pandas as pd
import os

def clean_duplicates_smart():
    print("--- [Smart Deduplication] Fix cứng article_id và source_category ---")
    
    input_path = 'data/raw/articles.csv'
    
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file {input_path}")
        return

    print(f"📂 Đang đọc file: {input_path}")
    df = pd.read_csv(input_path)
    
    # Clean tên cột để tránh lỗi khoảng trắng thừa
    df.columns = df.columns.str.strip()
    
    original_count = len(df)

    # Kiểm tra nhanh xem có đủ 2 cột cần thiết không
    if 'article_id' not in df.columns or 'source_category' not in df.columns:
        print("❌ LỖI: File thiếu cột 'article_id' hoặc 'source_category'")
        print(f"   Các cột hiện có: {list(df.columns)}")
        return

    # 1. Tính số lượng bài của mỗi category
    cat_counts = df['source_category'].value_counts().to_dict()
    
    # 2. Map số lượng vào dataframe
    df['cat_popularity'] = df['source_category'].map(cat_counts)
    
    # 3. Sắp xếp: Category nào ít bài (hiếm) lên đầu
    # ascending=True nghĩa là số nhỏ lên trên -> ưu tiên giữ lại
    df = df.sort_values(by=['cat_popularity'], ascending=True)
    
    # 4. Drop duplicates dựa trên article_id
    # keep='first' sẽ giữ lại thằng nằm trên (thằng thuộc category hiếm hơn)
    df_clean = df.drop_duplicates(subset=['article_id'], keep='first')
    
    # Xóa cột tạm
    df_clean = df_clean.drop(columns=['cat_popularity'])
    
    # 5. Lưu file
    output_path = 'data/processed/articles_cleaned.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    deleted = original_count - len(df_clean)
    print("-" * 30)
    print(f"✅ HOÀN TẤT!")
    print(f"   - Ban đầu: {original_count}")
    print(f"   - Còn lại: {len(df_clean)}")
    print(f"   - Đã xóa:  {deleted}")
    print(f"💾 File lưu tại: {output_path}")

if __name__ == "__main__":
    clean_duplicates_smart()