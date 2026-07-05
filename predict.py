"""
MoodLens - Prediction Module

This module provides the main prediction interface for the emotion classifier.
It loads the trained model and provides methods for predicting emotions
from raw text input.

Example:
    >>> from predict import EmotionPredictor
    >>> predictor = EmotionPredictor()
    >>> result = predictor.predict("I'm feeling so happy today!")
    >>> print(result['emotion'])
    'joy'
    >>> print(result['confidence'])
    0.95
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import joblib
import pandas as pd

from config import MODELS_DIR
from utils.preprocessor import TextPreprocessor, preprocess_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmotionPredictor:
    """
    Emotion prediction interface.
    
    Loads the trained model and provides prediction methods.
    
    Attributes:
        pipeline: Trained sklearn pipeline (TF-IDF + Logistic Regression)
        label_encoder: LabelEncoder for converting labels
        classes: List of emotion classes
        preprocessor: TextPreprocessor for cleaning text
    """
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        auto_load: bool = True
    ):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing model files
            auto_load: If True, automatically load the model
        """
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR
        self.pipeline = None
        self.label_encoder = None
        self.classes = None
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_stopwords=True,
            keep_negations=True
        )
        
        if auto_load:
            self.load_model()
    
    def load_model(self) -> None:
        """
        Load the trained model and associated artifacts.
        
        Raises:
            FileNotFoundError: If model files don't exist
        """
        logger.info(f"Loading model from: {self.model_dir}")
        
        # Check if model files exist
        model_path = self.model_dir / "logistic_regression.pkl"
        encoder_path = self.model_dir / "label_encoder.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Please run train.py first to train the model."
            )
        
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at: {encoder_path}\n"
                "Please run train.py first to train the model."
            )
        
        # Load pipeline
        self.pipeline = joblib.load(model_path)
        logger.info(f"✅ Loaded pipeline from: {model_path}")
        
        # Load label encoder
        self.label_encoder = joblib.load(encoder_path)
        self.classes = self.label_encoder.classes_.tolist()
        logger.info(f"✅ Loaded label encoder. Classes: {self.classes}")
        
        # Load metadata if available
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"✅ Loaded metadata")
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text for prediction.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        return self.preprocessor.transform(str(text))
    
    def predict(self, text: str) -> Dict:
        """
        Predict emotion for a single text.
        
        Args:
            text: Raw input text
            
        Returns:
            Dictionary containing:
                - emotion: Predicted emotion label
                - confidence: Confidence score (0-1)
                - probabilities: Dict of class probabilities
                - preprocessed_text: Cleaned text
                
        Example:
            >>> predictor = EmotionPredictor()
            >>> result = predictor.predict("I'm so happy!")
            >>> result['emotion']
            'joy'
            >>> result['confidence']
            0.95
        """
        if not self.pipeline:
            self.load_model()
        
        if not text or not text.strip():
            return {
                'emotion': None,
                'confidence': 0.0,
                'probabilities': {cls: 0.0 for cls in self.classes},
                'preprocessed_text': '',
                'error': 'Empty text provided'
            }
        
        # Preprocess text
        clean_text = self.preprocess(text)
        
        # Get prediction
        pred_encoded = self.pipeline.predict([clean_text])[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get probabilities
        probs = self.pipeline.predict_proba([clean_text])[0]
        confidence = float(max(probs))
        
        # Create probability dictionary
        prob_dict = {
            label: float(prob) 
            for label, prob in zip(self.classes, probs)
        }
        # Sort by probability descending
        prob_dict = dict(sorted(
            prob_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return {
            'emotion': pred_label,
            'confidence': confidence,
            'probabilities': prob_dict,
            'preprocessed_text': clean_text,
            'raw_text': text,
            'error': None
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict emotions for multiple texts.
        
        Args:
            texts: List of raw input texts
            
        Returns:
            List of prediction results
        """
        return [self.predict(text) for text in texts]
    
    def get_top_n(self, text: str, n: int = 3) -> List[Tuple[str, float]]:
        """
        Get top N emotion predictions with probabilities.
        
        Args:
            text: Raw input text
            n: Number of top predictions to return
            
        Returns:
            List of (emotion, probability) tuples
            
        Example:
            >>> predictor = EmotionPredictor()
            >>> top = predictor.get_top_n("I'm feeling okay", n=3)
            >>> print(top)
            [('sadness', 0.45), ('joy', 0.30), ('anger', 0.15)]
        """
        result = self.predict(text)
        if result['error']:
            return []
        
        # Get top N from probabilities
        prob_items = list(result['probabilities'].items())
        return prob_items[:n]
    
    def explain(self, text: str) -> str:
        """
        Get a human-readable explanation of the prediction.
        
        Args:
            text: Raw input text
            
        Returns:
            Formatted string with prediction explanation
        """
        result = self.predict(text)
        
        if result['error']:
            return f"Error: {result['error']}"
        
        lines = [
            f"📝 Text: {result['raw_text'][:100]}...",
            f"🎯 Emotion: {result['emotion']}",
            f"📊 Confidence: {result['confidence']:.2%}",
            "",
            "📈 Top probabilities:"
        ]
        
        for emotion, prob in list(result['probabilities'].items())[:5]:
            bar = "█" * int(prob * 50)
            lines.append(f"  {emotion:12s}: {prob:6.2%} {bar}")
        
        return "\n".join(lines)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.metadata:
            return {
                'classes': self.classes,
                'num_classes': len(self.classes) if self.classes else 0,
                'model_loaded': self.pipeline is not None
            }
        
        return {
            'classes': self.classes,
            'num_classes': len(self.classes) if self.classes else 0,
            'model_type': self.metadata.get('model_type', 'unknown'),
            'training_date': self.metadata.get('training_date', 'unknown'),
            'accuracy': self.metadata.get('metrics', {}).get('accuracy', 0),
            'f1_weighted': self.metadata.get('metrics', {}).get('f1_weighted', 0),
            'model_loaded': self.pipeline is not None
        }


# ============================================================================
# Convenience function (singleton pattern)
# ============================================================================

_predictor_instance = None


def get_predictor() -> EmotionPredictor:
    """
    Get a singleton instance of EmotionPredictor.
    
    Returns:
        EmotionPredictor instance (lazy-loaded)
        
    Example:
        >>> predictor = get_predictor()
        >>> result = predictor.predict("I'm sad")
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EmotionPredictor()
    return _predictor_instance


def predict_emotion(text: str) -> Dict:
    """
    Convenience function to predict emotion from text.
    
    Args:
        text: Raw input text
        
    Returns:
        Prediction result dictionary
        
    Example:
        >>> result = predict_emotion("I'm feeling great!")
        >>> print(result['emotion'])
    """
    predictor = get_predictor()
    return predictor.predict(text)


def get_top_emotions(text: str, n: int = 3) -> List[Tuple[str, float]]:
    """
    Convenience function to get top N emotions.
    
    Args:
        text: Raw input text
        n: Number of top predictions
        
    Returns:
        List of (emotion, probability) tuples
    """
    predictor = get_predictor()
    return predictor.get_top_n(text, n)


# ============================================================================
# Main (for testing)
# ============================================================================

def main():
    """Test the prediction module."""
    print("=" * 60)
    print("MoodLens - Prediction Module Test")
    print("=" * 60)
    
    # Load predictor
    predictor = EmotionPredictor()
    
    # Test texts
    test_texts = [
        "I'm feeling so happy and excited today!",
        "I am so sad and depressed right now.",
        "I'm really angry about what happened.",
        "I love this movie, it's amazing!",
        "I'm scared of what might happen.",
        "Wow, that was unexpected!",
    ]
    
    print("\n📊 Model Info:")
    info = predictor.get_model_info()
    print(f"  Classes: {info['classes']}")
    print(f"  Accuracy: {info.get('accuracy', 'N/A')}")
    
    print("\n📝 Predictions:")
    print("-" * 60)
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n  Text: {text}")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
        # Show top 3 probabilities
        top = list(result['probabilities'].items())[:3]
        prob_str = ", ".join([f"{e}: {p:.1%}" for e, p in top])
        print(f"  Top 3: {prob_str}")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")


if __name__ == "__main__":
    main()