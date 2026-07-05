"""
Text Preprocessing Utilities for MoodLens

This module provides functions for cleaning and normalizing text
before vectorization and model training.

The preprocessing pipeline is kept light to preserve emotional
context while removing noise.

Example:
    >>> from utils.preprocessor import preprocess_text
    >>> text = "I'm feeling SO happy today!!!"
    >>> clean = preprocess_text(text)
    >>> print(clean)
    "im feeling so happy today"
"""

import re
import string
from typing import Optional, List
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Download stopwords if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# English stopwords
STOPWORDS = set(stopwords.words('english'))


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_special_chars(text: str) -> str:
    """
    Remove special characters and keep only letters, numbers, and spaces.
    
    Preserves:
        - Letters (a-z, A-Z)
        - Numbers (0-9)
        - Spaces
        - Apostrophes (for contractions like "don't")
    
    Removes:
        - Punctuation (!@#$%^&*(), etc.)
        - Emojis
        - Extra whitespace
    """
    # Keep letters, numbers, spaces, and apostrophes
    text = re.sub(r'[^a-zA-Z0-9\s\']', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stopwords(text: str, keep_negations: bool = True) -> str:
    """
    Remove English stopwords from text.
    
    Args:
        text: Input text
        keep_negations: If True, keep words like 'not', 'no', 'never'
                       because they are important for sentiment/emotion
    
    Returns:
        Text with stopwords removed
    """
    words = text.split()
    
    # Negation words to keep (important for emotion detection)
    negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody',
                      'nothing', 'nowhere', 'noone', 'cannot', 'cant', 'wouldnt',
                      'shouldnt', 'couldnt', 'wont', 'doesnt', 'dont', 'isnt'}
    
    if keep_negations:
        stopwords_to_remove = STOPWORDS - negation_words
    else:
        stopwords_to_remove = STOPWORDS
    
    filtered = [w for w in words if w not in stopwords_to_remove]
    return ' '.join(filtered)


def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace and normalize spaces."""
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_special: bool = True,
    remove_stop: bool = True,
    keep_negations: bool = True,
    stem: bool = False
) -> str:
    """
    Complete text preprocessing pipeline.
    
    Args:
        text: Input text string
        lowercase: Convert to lowercase
        remove_special: Remove special characters
        remove_stop: Remove stopwords
        keep_negations: Keep negation words (not, no, never, etc.)
        stem: Apply Porter stemming (currently disabled by default)
    
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # 1. Convert to lowercase
    if lowercase:
        text = to_lowercase(text)
    
    # 2. Remove special characters
    if remove_special:
        text = remove_special_chars(text)
    
    # 3. Remove stopwords
    if remove_stop:
        text = remove_stopwords(text, keep_negations=keep_negations)
    
    # 4. Clean up extra whitespace
    text = remove_extra_whitespace(text)
    
    return text


def preprocess_dataframe(df, text_column: str = 'text', **kwargs) -> pd.Series:
    """
    Apply preprocessing to a DataFrame column.
    
    Args:
        df: DataFrame containing text data
        text_column: Name of column containing text
        **kwargs: Arguments passed to preprocess_text
    
    Returns:
        Series with preprocessed text
    """
    import pandas as pd
    return df[text_column].apply(
        lambda x: preprocess_text(str(x), **kwargs)
    )


# ============================================================================
# Optional: Stemming (disabled by default in v1)
# ============================================================================

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()

def apply_stemming(text: str) -> str:
    """Apply Porter stemming to text."""
    words = text.split()
    stemmed = [_stemmer.stem(w) for w in words]
    return ' '.join(stemmed)


# ============================================================================
# Cleaner class for batch processing
# ============================================================================

class TextPreprocessor:
    """
    Configurable text preprocessor for batch processing.
    
    Example:
        >>> preprocessor = TextPreprocessor(
        ...     lowercase=True,
        ...     remove_stopwords=True,
        ...     keep_negations=True
        ... )
        >>> texts = ["I'm not happy!", "Feeling great!"]
        >>> cleaned = preprocessor.transform(texts)
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_stopwords: bool = True,
        keep_negations: bool = True,
        apply_stemming: bool = False
    ):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.keep_negations = keep_negations
        self.apply_stemming = apply_stemming
    
    def transform(self, text: str) -> str:
        """Preprocess a single text."""
        return preprocess_text(
            text,
            lowercase=self.lowercase,
            remove_special=self.remove_special_chars,
            remove_stop=self.remove_stopwords,
            keep_negations=self.keep_negations,
            stem=self.apply_stemming
        )
    
    def transform_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.transform(t) for t in texts]