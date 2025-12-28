import pandas as pd
import os

def clean_replies_duplicates():
    print("--- 🧹 DỌN DẸP TRÙNG LẶP CHO REPLIES.CSV ---")
    
    input_path = 'data/raw/replies.csv' # File replies gốc
    
    if not os.path.exists(input_path):
        print("❌ Không tìm thấy file replies.csv")
        return

    # Đọc file
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f"📊 Số lượng ban đầu: {original_count} dòng")

    # --- CHIẾN THUẬT LỌC TRÙNG ---
    
    # Cách 1: Ưu tiên lọc theo ID comment (nếu có) - Đây là cách chuẩn nhất
    # Thường các cột ID sẽ có tên kiểu: 'reply_id', 'comment_id', 'id'
    id_col = None
    for col in ['reply_id', 'comment_id', 'id']:
        if col in df.columns:
            id_col = col
            break
            
    if id_col:
        print(f"🎯 Phát hiện cột ID '{id_col}'. Sẽ lọc trùng dựa trên ID này.")
        df_clean = df.drop_duplicates(subset=[id_col], keep='first')
        
    else:
        # Cách 2: Nếu không có ID, lọc dựa trên bộ 3 quyền lực:
        # [Link bài báo + Tên người comment + Nội dung comment]
        # Bỏ qua các cột rác như 'crawl_time' (nếu có)
        print("⚠️ Không thấy cột ID. Sẽ lọc trùng dựa trên Nội dung + Tên user + Link bài.")
        
        # Các cột quan trọng cần check (tùy tên cột thực tế của bạn)
        # Tôi liệt kê các tên phổ biến, code sẽ tự dò
        cols_to_check = []
        
        # Tìm cột url bài báo
        for c in ['article_url', 'url', 'post_url']:
            if c in df.columns: cols_to_check.append(c); break
            
        # Tìm cột nội dung
        for c in ['content', 'reply_content', 'text', 'message']:
            if c in df.columns: cols_to_check.append(c); break
            
        # Tìm cột user
        for c in ['display_name', 'user_name', 'author', 'user']:
            if c in df.columns: cols_to_check.append(c); break
            
        if len(cols_to_check) >= 2:
            print(f"   -> Subset dùng để check: {cols_to_check}")
            df_clean = df.drop_duplicates(subset=cols_to_check, keep='first')
        else:
            # Cách 3: Đường cùng - So khớp toàn bộ các cột (như bạn nói)
            print("⚠️ Không tìm thấy đủ cột định danh. Dùng phương pháp so khớp TOÀN BỘ dòng.")
            df_clean = df.drop_duplicates(keep='first')

    # --- KẾT QUẢ ---
    deleted = original_count - len(df_clean)
    
    if deleted > 0:
        print("-" * 30)
        print(f"✅ Đã dọn dẹp thành công!")
        print(f"   - Giữ lại: {len(df_clean)}")
        print(f"   - Đã xóa:  {deleted} dòng trùng lặp")
        
        # Lưu đè hoặc lưu file mới
        output_path = 'data/r/new_replies.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"💾 File sạch đã lưu tại: {output_path}")
    else:
        print("✅ File replies.csv không có dòng trùng nào.")

if __name__ == "__main__":
    clean_replies_duplicates()