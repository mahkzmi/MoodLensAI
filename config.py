"""
MoodLens - Central Configuration

This module contains all project-wide configuration settings including
paths, parameters, and constants.

Usage:
    from config import DATA_DIR, MODEL_PARAMS
"""

from pathlib import Path
from datetime import datetime

# ============================================================================
# Directory Paths
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"

# ============================================================================
# Dataset Files
# ============================================================================

TRAIN_FILE = RAW_DATA_DIR / "train.txt"
VAL_FILE = RAW_DATA_DIR / "val.txt"
TEST_FILE = RAW_DATA_DIR / "test.txt"

# Dataset format
DELIMITER = ";"  # Separator between text and label
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# ============================================================================
# Model Parameters
# ============================================================================

MODEL_PARAMS = {
    "logistic_regression": {
        "C": 1.0,                    # Inverse regularization strength
        "max_iter": 1000,            # Maximum iterations for convergence
        "random_state": 42,          # Reproducibility
        "class_weight": "balanced",  # Handle class imbalance
        "solver": "liblinear",       # Works well for small/medium datasets
        "penalty": "l2",             # L2 regularization
        "verbose": 1,
    }
}

# ============================================================================
# Vectorizer Parameters
# ============================================================================

VECTORIZER_PARAMS = {
    "tfidf": {
        "max_features": 5000,        # Vocabulary size (reduces dimensionality)
        "ngram_range": (1, 2),       # Unigrams and bigrams
        "min_df": 2,                 # Ignore terms appearing in < 2 documents
        "max_df": 0.95,              # Ignore terms appearing in > 95% of docs
        "sublinear_tf": True,        # Apply sublinear scaling to term frequencies
        "stop_words": "english",     # Remove common English stopwords
    }
}

# ============================================================================
# Training Parameters
# ============================================================================

TRAINING_PARAMS = {
    "test_size": 0.2,                # Holdout for validation (if no val set)
    "random_state": 42,
    "cv_folds": 5,                   # For cross-validation
}

# ============================================================================
# Application Settings
# ============================================================================

APP_PARAMS = {
    "title": "MoodLens",
    "description": "Understand the emotions behind your words.",
    "version": "1.0.0",
    "debug": True,
}

# ============================================================================
# Logging
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# ============================================================================
# Ensure Directories Exist
# ============================================================================

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                  EXPERIMENTS_DIR, ASSETS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helper Functions
# ============================================================================

def get_experiment_path(name: str) -> Path:
    """Get path for a new experiment file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return EXPERIMENTS_DIR / f"{timestamp}_{name}.md"


def get_model_path(name: str) -> Path:
    """Get path for a saved model."""
    return MODELS_DIR / f"{name}.pkl"