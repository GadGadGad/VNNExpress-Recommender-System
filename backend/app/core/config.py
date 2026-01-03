import os

class Settings:
    PROJECT_NAME: str = "News RecSys API"
    API_V1_STR: str = "/api/v1"
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    DATA_DIR: str = os.path.join(BASE_DIR, "data/processed/strict_g2")
    RAW_DIR: str = os.path.join(BASE_DIR, "data/raw")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    
    # Constants
    CATEGORY_MAP = {
        "giaoduc": "📚 Giáo dục",
        "khcn": "🔬 Khoa học & Công nghệ",
        "kinhdoanh": "💼 Kinh doanh",
        "thegioi": "🌍 Thế giới",
        "thethao": "⚽ Thể thao",
        "thoisu": "📰 Thời sự",
        "giai-tri": "🎬 Giải trí",
        "suc-khoe": "🏥 Sức khỏe",
        "xe": "🚗 Xe",
        "du-lich": "✈️ Du lịch",
        "N/A": "❓ Khác"
    }

settings = Settings()
