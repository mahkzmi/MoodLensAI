#!/usr/bin/env python3
"""
Test script for text preprocessing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.data_loader import load_emotion_dataset
from utils.preprocessor import preprocess_text, TextPreprocessor


def main():
    print("=" * 60)
    print("MoodLens - Text Preprocessor Test")
    print("=" * 60)
    
    # Sample texts
    test_texts = [
        "I'm feeling SO happy today!!!",
        "I don't feel good at all... :(",
        "This is the best day of my life!",
        "I'm not angry, I'm just disappointed.",
        "Never say never!",
    ]
    
    print("\n📝 Original texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")
    
    print("\n🧹 Preprocessed texts:")
    for i, text in enumerate(test_texts, 1):
        cleaned = preprocess_text(text)
        print(f"  {i}. {cleaned}")
    
    # Test with dataset
    print("\n📊 Testing on actual dataset...")
    train_df, val_df, test_df = load_emotion_dataset()
    
    # Sample rows
    sample = train_df.sample(5, random_state=42)
    
    print("\n📝 Sample preprocessing:")
    for _, row in sample.iterrows():
        original = row['text']
        cleaned = preprocess_text(original)
        print(f"\n  [{row['label']}]")
        print(f"  Original: {original[:80]}...")
        print(f"  Cleaned:  {cleaned[:80]}...")
    
    # Test batch processing
    print("\n⚡ Batch processing test:")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_stopwords=True,
        keep_negations=True
    )
    
    batch = train_df['text'].head(10).tolist()
    cleaned_batch = preprocessor.transform_batch(batch)
    
    print(f"  Processed {len(cleaned_batch)} texts")
    print(f"  First: {cleaned_batch[0][:60]}...")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()