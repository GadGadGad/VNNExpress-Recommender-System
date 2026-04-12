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
        "thegioi": "Thế giới",
        "thoisu": "Thời sự",
        "kinhdoanh": "Kinh doanh",
        "giaoduc": "Giáo dục",
        "thethao": "Thể thao",
        "khcn": "Khoa học công nghệ"
    }

settings = Settings()
