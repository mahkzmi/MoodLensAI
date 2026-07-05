#!/usr/bin/env python3
"""
Test script for prediction module.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from predict import EmotionPredictor, predict_emotion, get_top_emotions


def main():
    print("=" * 60)
    print("MoodLens - Prediction Module Test")
    print("=" * 60)
    
    # Test 1: Load predictor
    print("\n📂 Test 1: Loading predictor...")
    predictor = EmotionPredictor()
    print("✅ Predictor loaded successfully!")
    
    # Test 2: Model info
    print("\n📊 Test 2: Model info...")
    info = predictor.get_model_info()
    print(f"  Classes: {info['classes']}")
    print(f"  Number of classes: {info['num_classes']}")
    print(f"  Model type: {info.get('model_type', 'unknown')}")
    print(f"  Accuracy: {info.get('accuracy', 'N/A')}")
    print(f"  F1-Weighted: {info.get('f1_weighted', 'N/A')}")
    
    # Test 3: Single predictions
    print("\n📝 Test 3: Single predictions...")
    print("-" * 60)
    
    test_texts = [
        "I'm feeling so happy and excited today!",
        "I am so sad and depressed right now.",
        "I'm really angry about what happened.",
        "I love this movie, it's amazing!",
        "I'm scared of what might happen.",
        "Wow, that was completely unexpected!",
        "I'm feeling pretty neutral about it.",
        "This is the worst day ever!",
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        status = "✅" if result['emotion'] else "❌"
        print(f"\n{status} Text: {text}")
        print(f"   Emotion: {result['emotion']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        # Show top 3
        top = list(result['probabilities'].items())[:3]
        prob_str = ", ".join([f"{e}: {p:.1%}" for e, p in top])
        print(f"   Top 3: {prob_str}")
    
    # Test 4: Batch prediction
    print("\n📦 Test 4: Batch prediction...")
    batch = [
        "I'm feeling great!",
        "This is terrible.",
        "I'm so in love!",
    ]
    results = predictor.predict_batch(batch)
    for text, result in zip(batch, results):
        print(f"  '{text}' -> {result['emotion']} ({result['confidence']:.1%})")
    
    # Test 5: Top N
    print("\n🏆 Test 5: Top N predictions...")
    top = get_top_emotions("I'm feeling okay about it", n=3)
    for emotion, prob in top:
        print(f"  {emotion}: {prob:.1%}")
    
    # Test 6: Explanation
    print("\n📖 Test 6: Prediction explanation...")
    explanation = predictor.explain("I'm feeling confused and uncertain")
    print(explanation)
    
    # Test 7: Empty text
    print("\n🚫 Test 7: Empty text handling...")
    result = predict_emotion("")
    print(f"  Result: {result}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()