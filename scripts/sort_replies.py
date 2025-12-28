import pandas as pd
import os
from datetime import datetime

def main():
    print("--- [Logic Xịn] Sắp xếp replies.csv theo published_at ---")
    
    # 1. Load Data
    try:
        art_path = 'data/processed/articles.csv' # File đã clean
        rep_path = 'data/raw/replies.csv'        # File replies gốc
        
        # Đọc articles và ép kiểu string cho published_at để tránh lỗi
        articles = pd.read_csv(art_path, dtype={'published_at': str})
        replies = pd.read_csv(rep_path)
        print(f"Loaded: {len(articles)} Articles, {len(replies)} Replies.")
    except Exception as e:
        print(f"❌ Lỗi load file: {e}")
        return

    # 2. Xử lý thời gian cho Articles (Dùng logic Vectorized giống code kiểm tra)
    print("Đang parse cột 'published_at'...")

    # --- ĐOẠN NÀY DÙNG LOGIC GIỐNG HỆT FILE KIỂM TRA ---
    # Pattern: ngày/tháng/năm, giờ:phút (chấp nhận 1 hoặc 2 chữ số)
    # Ví dụ: 1/12/2025 hoặc 21/12/2025 đều bắt được
    articles['clean_time_str'] = articles['published_at'].str.extract(r'(\d{1,2}/\d{1,2}/\d{4}, \d{2}:\d{2})')[0]
    
    # Convert sang datetime objects
    articles['dt_obj'] = pd.to_datetime(articles['clean_time_str'], format='%d/%m/%Y, %H:%M', errors='coerce')
    
    # Check tỉ lệ thành công
    success_count = articles['dt_obj'].notna().sum()
    print(f"   -> Parse thành công: {success_count}/{len(articles)} bài ({success_count/len(articles)*100:.1f}%)")

    # Tạo cột timestamp (float) để map sang replies
    # Nếu bị NaT (lỗi), gán bằng 0 (về năm 1970 - đẩy xuống cũ nhất) hoặc thời gian hiện tại tùy bạn
    # Ở đây mình gán bằng 0 để nó không ảnh hưởng thứ tự các bài mới
    articles['timestamp_score'] = articles['dt_obj'].apply(lambda x: x.timestamp() if pd.notnull(x) else 0.0)

    # 3. Map sang Replies
    print("Mapping thời gian sang replies...")
    # Tạo từ điển: {url: timestamp}
    url_to_time = dict(zip(articles['url'], articles['timestamp_score']))
    
    # Map vào replies dựa trên article_url
    # Những reply không tìm thấy bài gốc (hoặc bài gốc lỗi time) sẽ nhận giá trị 0
    replies['sort_time'] = replies['article_url'].map(url_to_time).fillna(0)
    
    # 4. Sort
    print("Đang sắp xếp...")
    # Sắp xếp tăng dần (Cũ -> Mới)
    replies_sorted = replies.sort_values(by=['sort_time'], ascending=True)

    # 5. Lưu kết quả
    out_dir = 'data/processed'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    output_path = f'{out_dir}/replies.csv'
    
    # Bỏ cột sort_time đi cho file sạch đẹp (hoặc giữ lại để debug thì comment dòng dưới)
    replies_sorted = replies_sorted.drop(columns=['sort_time'])
    
    replies_sorted.to_csv(output_path, index=False)
    
    print(f"✅ Xong! File chuẩn đã lưu tại: {output_path}")
    
    # In kiểm tra đầu cuối
    # Lấy lại time để in ra cho dễ nhìn
    first_ts = replies['article_url'].map(url_to_time).fillna(0).min()
    last_ts = replies['article_url'].map(url_to_time).fillna(0).max()
    
    print(f"   - Timestamp cũ nhất: {datetime.fromtimestamp(first_ts)}")
    print(f"   - Timestamp mới nhất: {datetime.fromtimestamp(last_ts)}")

if __name__ == "__main__":
    main()