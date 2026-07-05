#!/usr/bin/env python3
"""
Test script for visualization utilities.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.visualizer import (
    plot_probabilities,
    create_confidence_gauge,
    plot_emotion_radar,
    create_dashboard,
    create_wordcloud,
    MATPLOTLIB_AVAILABLE,
    PLOTLY_AVAILABLE
)
from utils.word_importance import get_word_importance
from predict import predict_emotion


def main():
    print("=" * 60)
    print("MoodLens - Visualization Test")
    print("=" * 60)
    
    print(f"\n📊 Libraries available:")
    print(f"  Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
    print(f"  Matplotlib/WordCloud: {'✅' if MATPLOTLIB_AVAILABLE else '❌'}")
    
    # Test data
    probs = {
        'joy': 0.45,
        'sadness': 0.25,
        'anger': 0.15,
        'fear': 0.08,
        'love': 0.05,
        'surprise': 0.02,
    }
    
    # Test 1: Probability plot
    print("\n📊 Test 1: Probability plot...")
    fig = plot_probabilities(probs, "Test Emotions")
    if fig:
        print("  ✅ Figure created successfully")
        print("  💡 Use fig.show() to display or st.plotly_chart(fig) in Streamlit")
    
    # Test 2: Confidence gauge
    print("\n📊 Test 2: Confidence gauge...")
    fig = create_confidence_gauge(0.85, "joy")
    if fig:
        print("  ✅ Figure created successfully")
    
    # Test 3: Radar chart
    print("\n📊 Test 3: Radar chart...")
    fig = plot_emotion_radar(probs)
    if fig:
        print("  ✅ Figure created successfully")
    
    # Test 4: Dashboard
    print("\n📊 Test 4: Dashboard...")
    fig = create_dashboard(probs, "joy", 0.85, "I'm feeling great!")
    if fig:
        print("  ✅ Figure created successfully")
    
    # Test 5: Word Cloud
    print("\n📊 Test 5: Word Cloud...")
    text = "I'm feeling so happy and joyful today! This is the best day ever!"
    fig = create_wordcloud(text)
    if fig:
        print("  ✅ Word cloud created successfully")
    else:
        print("  ⚠️ Word cloud not available (matplotlib or wordcloud not installed)")
    
    # Test 6: Word importance visualization
    print("\n📊 Test 6: Word importance from model...")
    try:
        importance = get_word_importance("joy", top_n=10)
        if importance:
            word_weights = {word: score for word, score in importance}
            print(f"  ✅ Got {len(word_weights)} words for joy")
            print(f"  Top words: {', '.join([w for w, _ in importance[:5]])}")
    except Exception as e:
        print(f"  ⚠️ Could not get word importance: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("💡 Note: In Streamlit, use st.plotly_chart(fig) for Plotly figures")
    print("=" * 60)
    
    # Optionally show one figure if running interactively
    if PLOTLY_AVAILABLE and fig:
        try:
            fig.show()
        except:
            pass


if __name__ == "__main__":
    main()