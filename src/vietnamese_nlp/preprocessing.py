"""
Vietnamese Text Preprocessing
==============================

Comprehensive text preprocessing for Vietnamese news articles.

Features:
    - Word segmentation using underthesea/pyvi
    - Vietnamese stopwords removal
    - Accent normalization
    - Special character cleaning
    - HTML tag removal
    - Number/date normalization
"""

import re
import unicodedata
from typing import List, Optional, Set, Callable
import os


# Vietnamese Stopwords
# Common Vietnamese stopwords for news articles
VIETNAMESE_STOPWORDS = {
    # Pronouns (Đại từ)
    'tôi', 'tao', 'ta', 'mình', 'chúng_tôi', 'chúng_ta', 'chúng_mình',
    'bạn', 'cậu', 'mày', 'các_bạn', 'các_cậu',
    'anh', 'chị', 'em', 'cô', 'chú', 'bác', 'ông', 'bà',
    'nó', 'họ', 'hắn', 'chúng_nó',
    'ai', 'gì', 'nào', 'đâu', 'sao', 'thế_nào', 'như_thế_nào',
    
    # Articles and determiners (Mạo từ)
    'một', 'những', 'các', 'mỗi', 'mọi', 'từng',
    'này', 'kia', 'đó', 'ấy', 'nọ',
    
    # Prepositions (Giới từ)
    'của', 'cho', 'với', 'về', 'từ', 'đến', 'tại', 'ở', 'trong', 'ngoài',
    'trên', 'dưới', 'trước', 'sau', 'giữa', 'bên', 'cạnh',
    'theo', 'bằng', 'qua', 'để', 'vì', 'do', 'nhờ', 'bởi',
    
    # Conjunctions (Liên từ)
    'và', 'hoặc', 'hay', 'nhưng', 'mà', 'còn', 'song', 'tuy', 'dù', 'nếu',
    'thì', 'vì', 'nên', 'bởi_vì', 'cho_nên', 'vì_vậy', 'do_đó', 'tuy_nhiên',
    'nhưng_mà', 'mặc_dù', 'dù_sao', 'tuy_vậy', 'thế_nhưng',
    
    # Adverbs (Trạng từ)
    'rất', 'lắm', 'quá', 'hơi', 'khá', 'cực_kỳ', 'vô_cùng',
    'đã', 'đang', 'sẽ', 'vừa', 'mới', 'từng', 'chưa', 'không', 'chẳng',
    'cũng', 'vẫn', 'còn', 'luôn', 'thường', 'hay', 'ít', 'hiếm',
    'ngay', 'liền', 'lập_tức', 'tức_thì', 'đột_nhiên', 'bỗng',
    'rồi', 'xong', 'hết', 'xem', 'thử',
    
    # Auxiliary words (Trợ từ)
    'à', 'ạ', 'ư', 'nhỉ', 'nhé', 'nha', 'ha', 'hả', 'chứ', 'đấy', 'đây',
    'kìa', 'thôi', 'nào', 'đi', 'mà', 'cơ', 'kia',
    
    # Numbers (Số từ)
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'trăm', 'nghìn', 'ngàn', 'triệu', 'tỷ',
    'thứ', 'thứ_nhất', 'thứ_hai', 'đầu_tiên', 'cuối_cùng',
    
    # Common verbs (Động từ thường dùng)
    'là', 'có', 'được', 'bị', 'làm', 'cho', 'lấy', 'đưa', 'mang',
    'đi', 'đến', 'về', 'ra', 'vào', 'lên', 'xuống',
    'biết', 'hiểu', 'nghĩ', 'tưởng', 'thấy', 'xem', 'nghe', 'nói', 'bảo',
    'muốn', 'cần', 'phải', 'nên', 'có_thể',
    
    # Question words (Từ nghi vấn)
    'ai', 'gì', 'nào', 'đâu', 'sao', 'bao_nhiêu', 'bao_lâu', 'mấy',
    'tại_sao', 'vì_sao', 'như_thế_nào', 'làm_sao',
    
    # Time words (Từ chỉ thời gian)
    'hôm_nay', 'hôm_qua', 'ngày_mai', 'tuần', 'tháng', 'năm',
    'sáng', 'trưa', 'chiều', 'tối', 'đêm',
    'bây_giờ', 'lúc_này', 'khi', 'lúc', 'đến_khi', 'trước_khi', 'sau_khi',
    
    # News-specific stopwords
    'theo', 'cho_biết', 'chia_sẻ', 'cho_hay', 'nói_rằng', 'cho_rằng',
    'tin', 'thông_tin', 'nguồn_tin', 'phóng_viên', 'báo', 'bài_viết',
    'đọc_thêm', 'xem_thêm', 'liên_quan', 'tiếp_theo',
}


class VietnameseTextPreprocessor:
    """
    Vietnamese text preprocessor with multiple processing options.
    
    Example:
        preprocessor = VietnameseTextPreprocessor(use_word_segmentation=True)
        text = "Hà Nội hôm nay có thời tiết đẹp"
        processed = preprocessor.preprocess(text)
        # Output: "hà_nội hôm_nay thời_tiết đẹp" (with stopwords removed)
    """
    
    def __init__(
        self,
        use_word_segmentation: bool = True,
        segmenter: str = 'underthesea',  # 'underthesea' or 'pyvi'
        remove_stopwords: bool = True,
        custom_stopwords: Optional[Set[str]] = None,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_urls: bool = True,
        remove_html: bool = True,
        normalize_unicode: bool = True,
        min_word_length: int = 1,
        max_word_length: int = 50,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            use_word_segmentation: Whether to use word segmentation
            segmenter: Which segmenter to use ('underthesea' or 'pyvi')
            remove_stopwords: Whether to remove stopwords
            custom_stopwords: Additional stopwords to remove
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove standalone numbers
            remove_urls: Remove URLs
            remove_html: Remove HTML tags
            normalize_unicode: Normalize Unicode (NFC form)
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
        """
        self.use_word_segmentation = use_word_segmentation
        self.segmenter_name = segmenter
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.normalize_unicode = normalize_unicode
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Stopwords
        self.stopwords = VIETNAMESE_STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
            
        # Initialize segmenter
        self.segmenter = None
        if use_word_segmentation:
            self._init_segmenter()
            
        # Compile regex patterns
        self._compile_patterns()
        
    def _init_segmenter(self):
        """Initialize word segmenter"""
        if self.segmenter_name == 'underthesea':
            try:
                from underthesea import word_tokenize
                self.segmenter = word_tokenize
                print("[Preprocessor] Using underthesea for word segmentation")
            except ImportError:
                print("[Warning] underthesea not installed. Install with: pip install underthesea")
                self.use_word_segmentation = False
                
        elif self.segmenter_name == 'pyvi':
            try:
                from pyvi import ViTokenizer
                self.segmenter = ViTokenizer.tokenize
                print("[Preprocessor] Using pyvi for word segmentation")
            except ImportError:
                print("[Warning] pyvi not installed. Install with: pip install pyvi")
                self.use_word_segmentation = False
                
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        # URL pattern
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.(com|vn|net|org|edu|gov)\S*'
        )
        
        # HTML tags
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Multiple spaces
        self.space_pattern = re.compile(r'\s+')
        
        # Punctuation (keeping Vietnamese diacritics)
        self.punct_pattern = re.compile(r'[!"#$%&\'()*+,-./:;<=>?@\[\]\\^_`{|}~""''…–—]')
        
        # Numbers
        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')
        
        # Special characters
        self.special_pattern = re.compile(r'[^\w\s]', re.UNICODE)
        
    def normalize_text(self, text: str) -> str:
        """Normalize Unicode and basic cleaning"""
        if self.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        return text
        
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags"""
        return self.html_pattern.sub(' ', text)
        
    def remove_url_links(self, text: str) -> str:
        """Remove URLs"""
        return self.url_pattern.sub(' ', text)
        
    def remove_punctuation_marks(self, text: str) -> str:
        """Remove punctuation"""
        return self.punct_pattern.sub(' ', text)
        
    def remove_number_tokens(self, text: str) -> str:
        """Remove standalone numbers"""
        return self.number_pattern.sub(' ', text)
        
    def segment_words(self, text: str) -> str:
        """Apply word segmentation"""
        if self.segmenter is None:
            return text
            
        if self.segmenter_name == 'underthesea':
            # underthesea returns list of words
            words = self.segmenter(text)
            return ' '.join(words)
        else:
            # pyvi returns string with underscores
            return self.segmenter(text)
            
    def remove_stopword_tokens(self, text: str) -> str:
        """Remove stopwords"""
        words = text.split()
        words = [w for w in words if w.lower() not in self.stopwords]
        return ' '.join(words)
        
    def filter_by_length(self, text: str) -> str:
        """Filter words by length"""
        words = text.split()
        words = [w for w in words 
                if self.min_word_length <= len(w) <= self.max_word_length]
        return ' '.join(words)
        
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline.
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Normalize Unicode
        text = self.normalize_text(text)
        
        # Remove HTML
        if self.remove_html:
            text = self.remove_html_tags(text)
            
        # Remove URLs
        if self.remove_urls:
            text = self.remove_url_links(text)
            
        # Remove punctuation
        if self.remove_punctuation:
            text = self.remove_punctuation_marks(text)
            
        # Remove numbers
        if self.remove_numbers:
            text = self.remove_number_tokens(text)
            
        # Lowercase
        if self.lowercase:
            text = text.lower()
            
        # Word segmentation
        if self.use_word_segmentation:
            text = self.segment_words(text)
            
        # Remove stopwords
        if self.remove_stopwords:
            text = self.remove_stopword_tokens(text)
            
        # Filter by length
        text = self.filter_by_length(text)
        
        # Clean up whitespace
        text = self.space_pattern.sub(' ', text).strip()
        
        return text
        
    def preprocess_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            show_progress: Show progress bar
            
        Returns:
            List of preprocessed texts
        """
        if show_progress:
            try:
                from tqdm import tqdm
                texts = tqdm(texts, desc="Preprocessing")
            except ImportError:
                pass
                
        return [self.preprocess(text) for text in texts]
        
    def tokenize(self, text: str) -> List[str]:
        """
        Preprocess and return list of tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        processed = self.preprocess(text)
        return processed.split() if processed else []
        
    def tokenize_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[str]]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of input texts
            show_progress: Show progress bar
            
        Returns:
            List of token lists
        """
        if show_progress:
            try:
                from tqdm import tqdm
                texts = tqdm(texts, desc="Tokenizing")
            except ImportError:
                pass
                
        return [self.tokenize(text) for text in texts]
        
    def get_stopwords(self) -> Set[str]:
        """Get the current stopwords set"""
        return self.stopwords.copy()
        
    def add_stopwords(self, words: Set[str]):
        """Add custom stopwords"""
        self.stopwords.update(words)
        
    def remove_from_stopwords(self, words: Set[str]):
        """Remove words from stopwords"""
        self.stopwords -= words


def load_stopwords_from_file(filepath: str) -> Set[str]:
    """
    Load stopwords from a text file (one word per line).
    
    Args:
        filepath: Path to stopwords file
        
    Returns:
        Set of stopwords
    """
    stopwords = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    return stopwords


def save_stopwords_to_file(stopwords: Set[str], filepath: str):
    """
    Save stopwords to a text file.
    
    Args:
        stopwords: Set of stopwords
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for word in sorted(stopwords):
            f.write(word + '\n')


# Convenience functions
def preprocess_vietnamese(
    text: str,
    use_segmentation: bool = True,
    remove_stopwords: bool = True
) -> str:
    """
    Quick preprocessing function for Vietnamese text.
    
    Args:
        text: Input text
        use_segmentation: Use word segmentation
        remove_stopwords: Remove stopwords
        
    Returns:
        Preprocessed text
    """
    preprocessor = VietnameseTextPreprocessor(
        use_word_segmentation=use_segmentation,
        remove_stopwords=remove_stopwords
    )
    return preprocessor.preprocess(text)


def tokenize_vietnamese(text: str, use_segmentation: bool = True) -> List[str]:
    """
    Quick tokenization function for Vietnamese text.
    
    Args:
        text: Input text
        use_segmentation: Use word segmentation
        
    Returns:
        List of tokens
    """
    preprocessor = VietnameseTextPreprocessor(
        use_word_segmentation=use_segmentation,
        remove_stopwords=False  # Keep all words for tokenization
    )
    return preprocessor.tokenize(text)


if __name__ == '__main__':
    # Test preprocessing
    test_texts = [
        "Hà Nội hôm nay có thời tiết rất đẹp, nhiệt độ khoảng 25 độ C.",
        "VTV1 đưa tin về tình hình kinh tế Việt Nam trong năm 2024.",
        "Đội tuyển bóng đá Việt Nam đã giành chiến thắng 2-0 trước Thái Lan.",
    ]
    
    print("=" * 60)
    print("Vietnamese Text Preprocessing Demo")
    print("=" * 60)
    
    # Without segmentation
    print("\n1. Without word segmentation:")
    preprocessor = VietnameseTextPreprocessor(use_word_segmentation=False)
    for text in test_texts:
        print(f"  Input:  {text}")
        print(f"  Output: {preprocessor.preprocess(text)}")
        print()
        
    # With segmentation (if available)
    print("\n2. With word segmentation (underthesea):")
    preprocessor = VietnameseTextPreprocessor(use_word_segmentation=True)
    for text in test_texts:
        print(f"  Input:  {text}")
        print(f"  Output: {preprocessor.preprocess(text)}")
        print()
        
    print("\n3. Tokenization:")
    for text in test_texts:
        tokens = preprocessor.tokenize(text)
        print(f"  Input:  {text}")
        print(f"  Tokens: {tokens}")
        print()
