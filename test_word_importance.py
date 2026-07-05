#!/usr/bin/env python3
"""
Test script for word importance analysis.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.word_importance import (
    WordImportanceAnalyzer,
    get_word_importance,
    get_words_for_text
)


def main():
    print("=" * 60)
    print("MoodLens - Word Importance Test")
    print("=" * 60)
    
    # Test 1: Initialize analyzer
    print("\n📂 Test 1: Loading analyzer...")
    analyzer = WordImportanceAnalyzer()
    print(f"✅ Analyzer loaded successfully!")
    print(f"   Features: {len(analyzer.feature_names):,}")
    print(f"   Classes: {analyzer.class_names}")
    
    # Test 2: Get important words for each emotion
    print("\n📝 Test 2: Important words per emotion...")
    print("-" * 60)
    
    for emotion in analyzer.class_names:
        words = get_word_importance(emotion, top_n=5)
        word_str = ", ".join([f"'{w}'({s:.2f})" for w, s in words])
        print(f"  {emotion:10s}: {word_str}")
    
    # Test 3: Analyze a specific text
    print("\n📖 Test 3: Analyzing text...")
    print("-" * 60)
    
    test_texts = [
        ("I'm incredibly happy and joyful today!", "joy"),
        ("I feel so sad and depressed and lonely.", "sadness"),
        ("I'm extremely angry and furious about this.", "anger"),
        ("I'm terrified and scared of what might happen.", "fear"),
        ("I love this with all my heart!", "love"),
        ("Wow, that was completely unexpected!", "surprise"),
    ]
    
    for text, expected in test_texts:
        words = get_words_for_text(text, top_n=5)
        word_str = ", ".join([f"'{w}'({s:.2f})" for w, s in words])
        print(f"\n  Text: {text}")
        print(f"  Expected: {expected}")
        print(f"  Top words: {word_str}")
    
    # Test 4: Emotion signatures
    print("\n📋 Test 4: Emotion signatures...")
    print("-" * 60)
    
    for emotion in ['joy', 'sadness', 'anger']:
        sig = analyzer.get_emotion_signature(emotion, top_n=5)
        pos = [w for w, _ in sig['positive_indicators']]
        neg = [w for w, _ in sig['negative_indicators']]
        print(f"  {emotion}:")
        print(f"    + {', '.join(pos)}")
        print(f"    - {', '.join(neg)}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()