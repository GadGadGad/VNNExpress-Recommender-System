
"""
Vietnamese Text Processing Utilities
====================================
"""

import re
import unicodedata

# Common Vietnamese stopwords
# Source: aggregated from standard lists
STOPWORDS = {
    'là', 'của', 'và', 'có', 'những', 'các', 'người', 'trong', 'một', 'cho', 
    'đã', 'sẽ', 'đến', 'về', 'với', 'không', 'được', 'tại', 'hay', 'nhưng', 
    'này', 'thì', 'từ', 'đó', 'cũng', 'để', 'lại', 'làm', 'ra', 'theo', 'sự', 
    'đang', 'việc', 'như', 'vì', 'mình', 'phải', 'còn', 'nó', 'chỉ', 'nhiều', 
    'vẫn', 'nên', 'khi', 'bị', 'rất', 'cả', 'nếu', 'lên', 'cùng', 'gì', 
    'nhà', 'đâu', 'qua', 'bằng', 'do', 'đây', 'thế', 'năm', 'ông', 'bà', 'anh', 'chị'
}

def clean_text(text: str) -> str:
    """
    Clean Vietnamese text:
    - Lowercase
    - Remove special characters/html
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags (basic)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters (keep Vietnamese chars, numbers, basic punctuation)
    # This regex keeps letters, numbers, spaces
    # \w in python regex matches Unicode characters by default in Python 3, so it handles Vietnamese
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text: str) -> str:
    """Remove Vietnamese stopwords"""
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)

def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline"""
    text = clean_text(text)
    text = remove_stopwords(text)
    return text
