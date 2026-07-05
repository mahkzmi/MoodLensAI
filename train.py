#!/usr/bin/env python3
"""
MoodLens - Model Training Script

Trains a Logistic Regression model for emotion classification using
TF-IDF features with balanced class weights.

Usage:
    python train.py

Output:
    - Models saved to models/
    - Training report saved to experiments/
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.pipeline import Pipeline
import joblib

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import (
    MODELS_DIR, 
    EXPERIMENTS_DIR,
    MODEL_PARAMS,
    VECTORIZER_PARAMS,
    TRAINING_PARAMS
)
from utils.data_loader import load_emotion_dataset
from utils.preprocessor import preprocess_text, TextPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data(train_df, val_df, test_df):
    """
    Preprocess text data and prepare features and labels.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("Preprocessing text data...")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_stopwords=True,
        keep_negations=True
    )
    
    # Preprocess texts
    X_train = train_df['text'].apply(preprocessor.transform)
    X_val = val_df['text'].apply(preprocessor.transform)
    X_test = test_df['text'].apply(preprocessor.transform)
    
    # Extract labels
    y_train = train_df['label']
    y_val = val_df['label']
    y_test = test_df['label']
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression model with TF-IDF features.
    
    Returns:
        Trained pipeline and evaluation metrics
    """
    logger.info("=" * 50)
    logger.info("Training Logistic Regression Model")
    logger.info("=" * 50)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(**VECTORIZER_PARAMS["tfidf"])
    
    # Create Logistic Regression
    lr = LogisticRegression(**MODEL_PARAMS["logistic_regression"])
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', lr)
    ])
    
    # Train the model
    logger.info("Fitting model...")
    pipeline.fit(X_train, y_train_encoded)
    
    # Make predictions on validation set
    y_pred = pipeline.predict(X_val)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val_encoded, y_pred),
        'f1_macro': f1_score(y_val_encoded, y_pred, average='macro'),
        'f1_weighted': f1_score(y_val_encoded, y_pred, average='weighted'),
        'precision_macro': precision_score(y_val_encoded, y_pred, average='macro'),
        'recall_macro': recall_score(y_val_encoded, y_pred, average='macro'),
    }
    
    # Classification report
    report = classification_report(
        y_val_encoded, 
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    logger.info(f"\n{'='*50}")
    logger.info("Validation Results")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1-Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"F1-Weighted: {metrics['f1_weighted']:.4f}")
    logger.info(f"\nClassification Report:")
    for class_name, scores in report.items():
        if isinstance(scores, dict) and 'precision' in scores:
            logger.info(
                f"  {class_name:12s}: "
                f"P={scores['precision']:.3f}, "
                f"R={scores['recall']:.3f}, "
                f"F1={scores['f1-score']:.3f}"
            )
    
    return pipeline, label_encoder, metrics, report


def save_artifacts(pipeline, label_encoder, metrics, report):
    """
    Save trained model, vectorizer, and encoder.
    """
    logger.info("\n" + "=" * 50)
    logger.info("Saving Artifacts")
    logger.info("=" * 50)
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Save pipeline (contains both vectorizer and classifier)
    model_path = MODELS_DIR / "logistic_regression.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"✅ Pipeline saved to: {model_path}")
    
    # Save label encoder separately (for prediction)
    encoder_path = MODELS_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"✅ Label encoder saved to: {encoder_path}")
    
    # Save vectorizer separately (for prediction)
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    joblib.dump(pipeline.named_steps['tfidf'], vectorizer_path)
    logger.info(f"✅ Vectorizer saved to: {vectorizer_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'logistic_regression',
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'classes': label_encoder.classes_.tolist(),
        'num_classes': len(label_encoder.classes_),
        'vectorizer_params': VECTORIZER_PARAMS["tfidf"],
        'model_params': MODEL_PARAMS["logistic_regression"],
        'train_samples': len(pipeline.named_steps['classifier'].classes_)
    }
    
    metadata_path = MODELS_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Metadata saved to: {metadata_path}")
    
    # Save training report
    report_path = EXPERIMENTS_DIR / "model_training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'classification_report': report,
            'model_params': MODEL_PARAMS["logistic_regression"],
            'vectorizer_params': VECTORIZER_PARAMS["tfidf"],
        }, f, indent=2)
    logger.info(f"✅ Training report saved to: {report_path}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("MoodLens - Model Training")
    logger.info("=" * 60)
    
    try:
        # 1. Load dataset
        logger.info("\n📂 Loading dataset...")
        train_df, val_df, test_df = load_emotion_dataset()
        
        # 2. Prepare data
        logger.info("\n🔧 Preparing data...")
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
            train_df, val_df, test_df
        )
        
        # 3. Train model
        pipeline, label_encoder, metrics, report = train_model(
            X_train, y_train, X_val, y_val
        )
        
        # 4. Save artifacts
        save_artifacts(pipeline, label_encoder, metrics, report)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Training Complete!")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Weighted: {metrics['f1_weighted']:.4f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()