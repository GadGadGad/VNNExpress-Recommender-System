#!/usr/bin/env python3
"""
Shared utilities for the VnExpress crawler pipeline.
- Caching
- Hashing
- URL Normalization
- Category Resolving
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

def sha1(s: str) -> str:
    """Returns the SHA1 hash of a string."""
    return hashlib.sha1(s.encode()).hexdigest()

def normalize_url(url: str) -> str:
    """Strips query parameters and fragments from a URL."""
    url = url.split("#")[0].split("?")[0]
    return url[:-1] if url.endswith("/") else url

class Cache:
    """Handles local file caching for HTML and JSON to speed up dev/testing."""
    def __init__(self, cache_dir: Path, enabled=True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(__name__)

    def _path(self, key: str) -> Path:
        """Generates a file path from a cache key."""
        return self.cache_dir / f"{sha1(key)}.json"

    def get(self, key: str) -> Optional[dict]:
        """Reads data from the cache."""
        if not self.enabled:
            return None
        p = self._path(key)
        if p.exists():
            try:
                return json.load(open(p, encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, key: str, val: dict):
        """Writes data to the cache."""
        if not self.enabled:
            return
        try:
            json.dump(val, open(self._path(key), "w", encoding="utf-8"), ensure_ascii=False)
        except Exception as e:
            self.log.warning(f"Cache write failed: {e}")

# 🔽 --- THÊM MỚI: BẢN ĐỒ CATEGORY --- 🔽
VNEXPRESS_CATEGORIES = {
    "thoi-su": "1001005",
    "the-gioi": "1001002",
    "goc-nhin": "1003450",
    "kinh-doanh": "1003159",
    "bat-dong-san": "1005628",
    "giai-tri": "1002691",
    "the-thao": "1002565",
    "phap-luat": "1001007",
    "giao-duc": "1003497",
    "suc-khoe": "1003750",
    "doi-song": "1002966",
    "du-lich": "1003231",
    "khoa-hoc-cong-nghe": "1006219",
    "xe": "1001006",
    "y-kien": "1001012",
    "tam-su": "1001014",
    "cuoi": "1001011",
    "tuyen-dau-chong-dich": "1004565"
}
# Bản đồ ngược để tra cứu ID nhanh
_VNEXPRESS_ID_TO_NAME = {v: k for k, v in VNEXPRESS_CATEGORIES.items()}
# 🔼 --- KẾT THÚC THÊM MỚI --- 🔼


# 🔽 --- THÊM MỚI: HÀM LOOKUP --- 🔽
def resolve_category_id(category_input: str) -> Optional[str]:
    """
    Dịch tên category (ví dụ: 'the-gioi') hoặc ID ('1001002')
    thành một Category ID hợp lệ.
    Trả về ID nếu tìm thấy, ngược lại trả về None.
    """
    if not category_input:
        return None

    # 1. Kiểm tra xem nó có phải là một ID hợp lệ không
    if category_input in _VNEXPRESS_ID_TO_NAME:
        return category_input

    # 2. Kiểm tra xem nó có phải là tên mà chúng ta có thể map không
    if category_input.lower() in VNEXPRESS_CATEGORIES:
        return VNEXPRESS_CATEGORIES[category_input.lower()]

    # 3. Không tìm thấy
    return None
# 🔼 --- KẾT THÚC THÊM MỚI --- 🔼
