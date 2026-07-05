"""
Data Loading Utilities for MoodLens

This module provides functionality for loading the emotion dataset
from its three split files (train/val/test).

The dataset format is: "text;label" per line.

Example:
    >>> from utils.data_loader import load_emotion_dataset
    >>> train_df, val_df, test_df = load_emotion_dataset()
    >>> print(train_df.shape)
    (16000, 2)
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

from config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE, 
    DELIMITER, TEXT_COLUMN, LABEL_COLUMN
)

# Configure logger
logger = logging.getLogger(__name__)


class EmotionDataLoader:
    """
    Data loader for the Kaggle Emotion Dataset (train/val/test splits).
    
    The dataset consists of three files with format: "text;label"
    
    Attributes:
        delimiter (str): Separator between text and label
        text_column (str): Name of text column in DataFrame
        label_column (str): Name of label column in DataFrame
    """
    
    def __init__(
        self,
        delimiter: str = DELIMITER,
        text_column: str = TEXT_COLUMN,
        label_column: str = LABEL_COLUMN
    ):
        """
        Initialize the data loader.
        
        Args:
            delimiter: Separator between text and label (default: ";")
            text_column: Name for text column (default: "text")
            label_column: Name for label column (default: "label")
        """
        self.delimiter = delimiter
        self.text_column = text_column
        self.label_column = label_column
    
    def load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single split file.
        
        Args:
            file_path: Path to the .txt file
            
        Returns:
            DataFrame with text and label columns
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or has invalid format
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}\n"
                f"Please ensure all dataset files are in {file_path.parent}"
            )
        
        logger.info(f"Loading: {file_path.name}")
        
        # Read file line by line
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split on the last delimiter (text may contain ';')
                parts = line.rsplit(self.delimiter, 1)
                if len(parts) != 2:
                    logger.warning(
                        f"Skipping malformed line {line_num} in {file_path.name}: {line[:50]}..."
                    )
                    continue
                
                text, label = parts
                data.append({
                    self.text_column: text.strip(),
                    self.label_column: label.strip()
                })
        
        if not data:
            raise ValueError(f"No valid data found in {file_path.name}")
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df):,} rows from {file_path.name}")
        
        return df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three dataset splits.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 50)
        logger.info("Loading Emotion Dataset")
        logger.info("=" * 50)
        
        train_df = self.load_file(TRAIN_FILE)
        val_df = self.load_file(VAL_FILE)
        test_df = self.load_file(TEST_FILE)
        
        total = len(train_df) + len(val_df) + len(test_df)
        logger.info(f"Total samples: {total:,}")
        logger.info(f"  Train: {len(train_df):,}")
        logger.info(f"  Val:   {len(val_df):,}")
        logger.info(f"  Test:  {len(test_df):,}")
        
        return train_df, val_df, test_df
    
    def get_class_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        Get the distribution of classes in a DataFrame.
        
        Args:
            df: DataFrame with label column
            
        Returns:
            Series with class counts sorted by count descending
        """
        return df[self.label_column].value_counts()
    
    def get_sample(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Get random samples from a DataFrame.
        
        Args:
            df: DataFrame
            n: Number of samples
            
        Returns:
            DataFrame with n random samples
        """
        return df.sample(n=n, random_state=42)
    
    def get_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get basic statistics about the dataset.
        
        Args:
            df: DataFrame
            
        Returns:
            Dictionary with statistics
        """
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "classes": sorted(df[self.label_column].unique()),
            "class_counts": self.get_class_distribution(df).to_dict(),
            "missing_text": df[self.text_column].isnull().sum(),
            "missing_label": df[self.label_column].isnull().sum(),
            "duplicates": df.duplicated().sum(),
        }


def load_emotion_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load the entire dataset.
    
    This is the recommended way to load the dataset.
    
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Example:
        >>> train, val, test = load_emotion_dataset()
        >>> train.head()
    """
    loader = EmotionDataLoader()
    return loader.load_all()