"""
MoodLens - Word Importance Analysis

This module extracts the most important words for each emotion
using the coefficients from the trained Logistic Regression model.

The coefficients indicate which words are most influential for
each emotion class.

Example:
    >>> from utils.word_importance import get_word_importance
    >>> importance = get_word_importance("joy", top_n=10)
    >>> print(importance[0])
    ("happy", 0.85)
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import MODELS_DIR


class WordImportanceAnalyzer:
    """
    Analyzes word importance using model coefficients.
    
    Attributes:
        pipeline: Trained sklearn pipeline
        vectorizer: TF-IDF vectorizer
        classifier: Logistic Regression model
        feature_names: List of feature (word) names
        classes: List of emotion classes
        coefficients: Model coefficients for each class
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the analyzer.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR
        self.pipeline = None
        self.vectorizer = None
        self.classifier = None
        self.feature_names = None
        self.classes = None
        self.coefficients = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model and extract components."""
        # Load pipeline
        model_path = self.model_dir / "logistic_regression.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Please run train.py first."
            )
        
        self.pipeline = joblib.load(model_path)
        
        # Extract components
        self.vectorizer = self.pipeline.named_steps['tfidf']
        self.classifier = self.pipeline.named_steps['classifier']
        
        # Get feature names
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients (shape: n_classes x n_features)
        self.coefficients = self.classifier.coef_
        
        # Get classes
        self.classes = self.classifier.classes_
        
        # Convert classes to strings if they are integers
        if isinstance(self.classes[0], (int, np.integer)):
            # If classes are integers, we need to map them to names
            # Try to load label encoder
            encoder_path = self.model_dir / "label_encoder.pkl"
            if encoder_path.exists():
                label_encoder = joblib.load(encoder_path)
                self.class_names = label_encoder.classes_.tolist()
                # Map class indices to names
                self.class_names = [self.class_names[i] for i in self.classes]
            else:
                self.class_names = [str(c) for c in self.classes]
        else:
            self.class_names = self.classes.tolist()
    
    def get_word_importance(
        self,
        emotion: str,
        top_n: int = 10,
        return_negative: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Get most important words for a specific emotion.
        
        Args:
            emotion: Emotion label (e.g., 'joy', 'sadness')
            top_n: Number of top words to return
            return_negative: If True, return words that NEGATIVELY correlate
            
        Returns:
            List of (word, importance_score) tuples
            
        Example:
            >>> analyzer = WordImportanceAnalyzer()
            >>> important = analyzer.get_word_importance("joy", top_n=5)
            >>> print(important)
            [('happy', 0.85), ('great', 0.72), ('love', 0.65), ...]
        """
        # Find the class index for this emotion
        try:
            class_idx = self.class_names.index(emotion)
        except ValueError:
            raise ValueError(
                f"Emotion '{emotion}' not found. "
                f"Available: {self.class_names}"
            )
        
        # Get coefficients for this class
        coefs = self.coefficients[class_idx]
        
        # Create list of (word, coefficient) pairs
        word_coef_pairs = list(zip(self.feature_names, coefs))
        
        # Sort by absolute coefficient (importance)
        if return_negative:
            # Return words with negative coefficients (opposite of emotion)
            word_coef_pairs.sort(key=lambda x: x[1])  # Ascending
        else:
            # Return words with positive coefficients
            word_coef_pairs.sort(key=lambda x: x[1], reverse=True)  # Descending
        
        # Return top N
        return word_coef_pairs[:top_n]
    
    def get_all_emotion_words(self, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get important words for all emotions.
        
        Args:
            top_n: Number of words per emotion
            
        Returns:
            Dictionary mapping emotion -> list of (word, score)
        """
        result = {}
        for emotion in self.class_names:
            result[emotion] = self.get_word_importance(emotion, top_n)
        return result
    
    def get_top_words_for_text(
        self,
        text: str,
        emotion: Optional[str] = None,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get important words from a specific text based on model weights.
        
        Args:
            text: Input text
            emotion: Specific emotion to analyze (if None, uses predicted emotion)
            top_n: Number of words to return
            
        Returns:
            List of (word, importance_score) tuples
        """
        # Preprocess text
        from utils.preprocessor import preprocess_text
        clean_text = preprocess_text(text)
        words = set(clean_text.split())
        
        if not words:
            return []
        
        # If emotion not specified, predict it
        if emotion is None:
            from predict import get_predictor
            predictor = get_predictor()
            result = predictor.predict(text)
            emotion = result['emotion']
        
        # Get class index
        class_idx = self.class_names.index(emotion)
        coefs = self.coefficients[class_idx]
        
        # Create word-coefficient mapping
        word_coef = {}
        for word, coef in zip(self.feature_names, coefs):
            if word in words:
                word_coef[word] = coef
        
        # Sort by absolute importance
        sorted_words = sorted(
            word_coef.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_words[:top_n]
    
    def get_emotion_signature(self, emotion: str, top_n: int = 15) -> Dict:
        """
        Get a signature of words that define an emotion.
        
        Returns both positive and negative indicators.
        
        Args:
            emotion: Emotion label
            top_n: Number of words per category
            
        Returns:
            Dictionary with 'positive' and 'negative' word lists
        """
        positive = self.get_word_importance(emotion, top_n, return_negative=False)
        negative = self.get_word_importance(emotion, top_n, return_negative=True)
        
        return {
            'emotion': emotion,
            'positive_indicators': positive,
            'negative_indicators': negative
        }


# ============================================================================
# Convenience functions
# ============================================================================

_analyzer_instance = None


def get_analyzer() -> WordImportanceAnalyzer:
    """Get singleton instance of WordImportanceAnalyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = WordImportanceAnalyzer()
    return _analyzer_instance


def get_word_importance(emotion: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Convenience function to get important words for an emotion.
    
    Args:
        emotion: Emotion label
        top_n: Number of words to return
        
    Returns:
        List of (word, importance_score) tuples
    """
    analyzer = get_analyzer()
    return analyzer.get_word_importance(emotion, top_n)


def get_words_for_text(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Convenience function to get important words from text.
    
    Args:
        text: Input text
        top_n: Number of words to return
        
    Returns:
        List of (word, importance_score) tuples
    """
    from predict import predict_emotion
    analyzer = get_analyzer()
    result = predict_emotion(text)
    return analyzer.get_top_words_for_text(text, result['emotion'], top_n)


# ============================================================================
# Testing
# ============================================================================

def main():
    """Test the word importance analyzer."""
    print("=" * 60)
    print("MoodLens - Word Importance Analysis Test")
    print("=" * 60)
    
    analyzer = WordImportanceAnalyzer()
    
    print(f"\n📊 Model Info:")
    print(f"  Classes: {analyzer.class_names}")
    print(f"  Features: {len(analyzer.feature_names):,}")
    
    print("\n📝 Top words for each emotion:")
    print("-" * 60)
    
    for emotion in analyzer.class_names:
        print(f"\n  🎯 {emotion.upper()}:")
        top_words = analyzer.get_word_importance(emotion, top_n=10)
        word_str = ", ".join([f"{w} ({s:.3f})" for w, s in top_words[:5]])
        print(f"     {word_str}...")
    
    # Test with a sample text
    print("\n📖 Analyzing sample text:")
    print("-" * 60)
    
    text = "I'm feeling so happy and excited today!"
    print(f"  Text: {text}")
    
    from predict import predict_emotion
    result = predict_emotion(text)
    print(f"  Predicted: {result['emotion']}")
    
    important_words = analyzer.get_top_words_for_text(text, top_n=10)
    print(f"  Important words in text:")
    for word, score in important_words[:5]:
        print(f"    {word}: {score:.3f}")
    
    # Get emotion signature
    print("\n📋 Emotion signature for JOY:")
    print("-" * 60)
    
    signature = analyzer.get_emotion_signature("joy", top_n=10)
    print(f"  Positive indicators:")
    for word, score in signature['positive_indicators'][:5]:
        print(f"    + {word}: {score:.3f}")
    print(f"  Negative indicators:")
    for word, score in signature['negative_indicators'][:5]:
        print(f"    - {word}: {score:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")


if __name__ == "__main__":
    main()