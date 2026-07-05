#!/usr/bin/env python3
"""
Test script for data loader.

Run this to verify that the dataset loads correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import load_emotion_dataset
from config import TRAIN_FILE, VAL_FILE, TEST_FILE


def main():
    """Test the data loader."""
    print("=" * 60)
    print("MoodLens - Data Loader Test")
    print("=" * 60)
    
    # Check files exist
    print("\n📂 Checking files:")
    for name, path in [("Train", TRAIN_FILE), ("Val", VAL_FILE), ("Test", TEST_FILE)]:
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {name}: {path.name}")
    
    # Load dataset
    print("\n📊 Loading dataset...")
    try:
        train_df, val_df, test_df = load_emotion_dataset()
        
        print("\n✅ Loading successful!")
        print(f"\n📊 Dataset sizes:")
        print(f"  Train: {len(train_df):,}")
        print(f"  Val:   {len(val_df):,}")
        print(f"  Test:  {len(test_df):,}")
        
        print(f"\n🏷️ Label distribution (Train):")
        for label, count in train_df['label'].value_counts().items():
            pct = count / len(train_df) * 100
            print(f"  {label:12s}: {count:5,} ({pct:5.2f}%)")
        
        print(f"\n📝 Sample from Train:")
        for _, row in train_df.sample(3, random_state=42).iterrows():
            text = row['text'][:80]
            print(f"  [{row['label']}] {text}...")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()