#!/usr/bin/env python3
"""
Test script for recommendation engine.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.recommender import (
    get_recommendation,
    get_all_emotions,
    get_recommendation_for_confidence
)


def main():
    print("=" * 60)
    print("MoodLens - Recommendation Engine Test")
    print("=" * 60)
    
    # Test 1: Get all emotions
    print("\n📊 Test 1: Supported emotions...")
    emotions = get_all_emotions()
    print(f"  {len(emotions)} emotions: {', '.join(emotions)}")
    
    # Test 2: Get recommendation for each emotion
    print("\n📝 Test 2: Recommendations for each emotion...")
    print("-" * 60)
    
    for emotion in emotions:
        rec = get_recommendation(emotion, confidence=0.8)
        print(f"\n  🎯 {emotion.upper()}")
        print(f"     Title: {rec['title']}")
        print(f"     Color: {rec['color']}")
        print(f"     Tip: {rec['tips'][0]}")
        print(f"     Activity: {rec['activities'][0]}")
        print(f"     Quote: {rec['quote'][:50]}...")
    
    # Test 3: Confidence level variations
    print("\n📊 Test 3: Confidence level variations...")
    print("-" * 60)
    
    for confidence in [0.9, 0.7, 0.5, 0.3]:
        rec = get_recommendation_for_confidence("sadness", confidence)
        print(f"\n  Confidence: {confidence:.0%} -> {rec['confidence_level']}")
        print(f"  Message: {rec['message'][:80]}...")
    
    # Test 4: Unknown emotion
    print("\n🚫 Test 4: Unknown emotion...")
    print("-" * 60)
    
    rec = get_recommendation("unknown_emotion")
    print(f"  Title: {rec['title']}")
    print(f"  Message: {rec['message']}")
    
    # Test 5: Visual representation
    print("\n🎨 Test 5: Visual representation...")
    print("-" * 60)
    
    for emotion in ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']:
        rec = get_recommendation(emotion, confidence=0.8)
        color = rec['color']
        # Show color block
        print(f"  {emotion:10s}: {color}  {rec['title']}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()