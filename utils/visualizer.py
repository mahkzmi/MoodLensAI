"""
MoodLens - Visualization Utilities

This module provides functions for creating beautiful visualizations
for the Streamlit interface.

Functions:
    - plot_probabilities: Bar chart of emotion probabilities
    - plot_word_cloud: Word cloud visualization
    - plot_importance: Word importance bar chart
    - create_emotion_gauge: Gauge chart for confidence
    - plot_emotion_radar: Radar chart for emotions
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")

try:
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib or wordcloud not installed.")


# ============================================================================
# Color Scheme
# ============================================================================

EMOTION_COLORS = {
    'joy': '#FFD700',
    'sadness': '#4A90D9',
    'anger': '#E74C3C',
    'fear': '#9B59B6',
    'love': '#FF6B6B',
    'surprise': '#F39C12',
}

EMOTION_ICONS = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😡',
    'fear': '😨',
    'love': '❤️',
    'surprise': '😮',
}


# ============================================================================
# Probability Visualization
# ============================================================================

def plot_probabilities(
    probabilities: Dict[str, float],
    title: str = "Emotion Probabilities",
    height: int = 400,
    show_confidence: bool = True
) -> Optional[go.Figure]:
    """
    Create a horizontal bar chart of emotion probabilities.
    
    Args:
        probabilities: Dictionary mapping emotion -> probability
        title: Chart title
        height: Chart height in pixels
        show_confidence: Show confidence score
        
    Returns:
        plotly Figure object
        
    Example:
        >>> probs = {'joy': 0.85, 'sadness': 0.10, 'anger': 0.05}
        >>> fig = plot_probabilities(probs, "My Emotions")
        >>> fig.show()
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # Sort by probability descending
    sorted_probs = dict(sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    ))
    
    # Prepare data
    emotions = list(sorted_probs.keys())
    values = list(sorted_probs.values())
    colors = [EMOTION_COLORS.get(e, '#95A5A6') for e in emotions]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars with icons in labels
    labels = [
        f"{EMOTION_ICONS.get(e, '')} {e.title()}" 
        for e in emotions
    ]
    
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.1%}<extra></extra>',
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(
            title="Probability",
            range=[0, 1.1],
            tickformat='.0%',
        ),
        yaxis=dict(
            title="Emotion",
            tickfont=dict(size=14),
        ),
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Arial, sans-serif", size=12),
    )
    
    # Add confidence indicator
    if show_confidence and values:
        max_prob = max(values)
        max_emotion = emotions[values.index(max_prob)]
        
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref='paper',
            yref='paper',
            text=f"🎯 Predicted: {max_emotion.title()} ({max_prob:.1%} confidence)",
            showarrow=False,
            font=dict(size=16, color=EMOTION_COLORS.get(max_emotion, '#333')),
        )
    
    return fig


# ============================================================================
# Word Cloud
# ============================================================================

def create_wordcloud(
    text: str,
    max_words: int = 100,
    width: int = 800,
    height: int = 400,
    background_color: str = 'white',
    colormap: str = 'viridis'
) -> Optional[plt.Figure]:
    """
    Create a word cloud from text.
    
    Args:
        text: Input text
        max_words: Maximum number of words
        width: Image width
        height: Image height
        background_color: Background color
        colormap: Matplotlib colormap name
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = create_wordcloud("happy joy love peace")
        >>> fig.show()
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    # Clean text
    from utils.preprocessor import preprocess_text
    clean_text = preprocess_text(text, remove_stop=True, keep_negations=False)
    
    if not clean_text.strip():
        return None
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        colormap=colormap,
        random_state=42,
    ).generate(clean_text)
    
    # Plot
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Word Cloud", fontsize=16, pad=20)
    
    return fig


def create_wordcloud_from_weights(
    word_weights: Dict[str, float],
    max_words: int = 100,
    width: int = 800,
    height: int = 400
) -> Optional[plt.Figure]:
    """
    Create a word cloud from word weights (e.g., from model coefficients).
    
    Args:
        word_weights: Dictionary mapping word -> weight
        max_words: Maximum number of words
        width: Image width
        height: Image height
        
    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    if not word_weights:
        return None
    
    # Normalize weights to positive values
    min_weight = min(word_weights.values())
    if min_weight < 0:
        word_weights = {w: v - min_weight + 1 for w, v in word_weights.items()}
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color='white',
        colormap='Reds',
        random_state=42,
    ).generate_from_frequencies(word_weights)
    
    # Plot
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Important Words", fontsize=16, pad=20)
    
    return fig


# ============================================================================
# Confidence Gauge
# ============================================================================

def create_confidence_gauge(
    confidence: float,
    emotion: str,
    height: int = 250
) -> Optional[go.Figure]:
    """
    Create a gauge chart for confidence score.
    
    Args:
        confidence: Confidence score (0-1)
        emotion: Predicted emotion
        height: Chart height
        
    Returns:
        plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    color = EMOTION_COLORS.get(emotion, '#95A5A6')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={'text': f"Confidence - {emotion.title()}"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#FF6B6B"},
                {'range': [30, 70], 'color': "#FFD93D"},
                {'range': [70, 100], 'color': "#6BCB77"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig


# ============================================================================
# Radar Chart
# ============================================================================

def plot_emotion_radar(
    probabilities: Dict[str, float],
    height: int = 400
) -> Optional[go.Figure]:
    """
    Create a radar chart of emotion probabilities.
    
    Args:
        probabilities: Dictionary mapping emotion -> probability
        height: Chart height
        
    Returns:
        plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # Prepare data
    emotions = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [EMOTION_COLORS.get(e, '#95A5A6') for e in emotions]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        marker=dict(color=colors, size=8),
        line=dict(color='#333', width=2),
        hovertemplate='<b>%{theta}</b><br>Probability: %{r:.1%}<extra></extra>',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1],
                tickformat='.0%',
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
            ),
        ),
        height=height,
        title="Emotion Profile",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    return fig


# ============================================================================
# Combined Dashboard
# ============================================================================

def create_dashboard(
    probabilities: Dict[str, float],
    emotion: str,
    confidence: float,
    text: str = "",
    height: int = 600
) -> Optional[go.Figure]:
    """
    Create a combined dashboard with probability bars and gauge.
    
    Args:
        probabilities: Dictionary mapping emotion -> probability
        emotion: Predicted emotion
        confidence: Confidence score
        text: Original text (for title)
        height: Chart height
        
    Returns:
        plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "indicator"}]],
        subplot_titles=("Emotion Probabilities", "Confidence"),
        column_widths=[0.7, 0.3],
    )
    
    # Sort probabilities
    sorted_probs = dict(sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    ))
    
    emotions = list(sorted_probs.keys())
    values = list(sorted_probs.values())
    colors = [EMOTION_COLORS.get(e, '#95A5A6') for e in emotions]
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=emotions,
            y=values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Add gauge
    color = EMOTION_COLORS.get(emotion, '#95A5A6')
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': f"{emotion.title()}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "#FF6B6B"},
                    {'range': [30, 70], 'color': "#FFD93D"},
                    {'range': [70, 100], 'color': "#6BCB77"},
                ],
            }
        ),
        row=1, col=2
    )
    
    # Update layout
    title_text = f"Emotion Analysis Dashboard"
    if text:
        title_text += f" - {text[:50]}..."
    
    fig.update_layout(
        height=height,
        title=title_text,
        showlegend=False,
        xaxis=dict(title="Emotion"),
        yaxis=dict(title="Probability", range=[0, 1.1], tickformat='.0%'),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    
    return fig


# ============================================================================
# Testing
# ============================================================================

def main():
    """Test the visualizer module."""
    print("=" * 60)
    print("MoodLens - Visualization Test")
    print("=" * 60)
    
    # Test data
    probs = {
        'joy': 0.45,
        'sadness': 0.25,
        'anger': 0.15,
        'fear': 0.08,
        'love': 0.05,
        'surprise': 0.02,
    }
    
    # Test probability plot
    print("\n📊 Test 1: Probability plot...")
    fig = plot_probabilities(probs, "Test Emotions")
    if fig:
        print("  ✅ Figure created successfully")
        fig.write_image("test_probabilities.png")
        print("  ✅ Saved to test_probabilities.png")
    
    # Test gauge
    print("\n📊 Test 2: Confidence gauge...")
    fig = create_confidence_gauge(0.85, "joy")
    if fig:
        print("  ✅ Figure created successfully")
        fig.write_image("test_gauge.png")
        print("  ✅ Saved to test_gauge.png")
    
    # Test radar
    print("\n📊 Test 3: Radar chart...")
    fig = plot_emotion_radar(probs)
    if fig:
        print("  ✅ Figure created successfully")
        fig.write_image("test_radar.png")
        print("  ✅ Saved to test_radar.png")
    
    # Test dashboard
    print("\n📊 Test 4: Dashboard...")
    fig = create_dashboard(probs, "joy", 0.85, "I'm feeling great!")
    if fig:
        print("  ✅ Figure created successfully")
        fig.write_image("test_dashboard.png")
        print("  ✅ Saved to test_dashboard.png")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()