"""
MoodLens - Streamlit Application

A modern, minimal web application for emotion detection from text.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Import project modules
from predict import get_predictor
from utils.recommender import get_recommendation
from utils.visualizer import (
    plot_probabilities,
    create_confidence_gauge,
    plot_emotion_radar,
    create_dashboard,
    create_wordcloud,
    EMOTION_COLORS,
    EMOTION_ICONS,
    PLOTLY_AVAILABLE,
    MATPLOTLIB_AVAILABLE
)
from utils.word_importance import get_word_importance, get_words_for_text
from utils.preprocessor import preprocess_text

# Page configuration
st.set_page_config(
    page_title="MoodLens - Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #fafafa;
    }
    
    /* Hero section */
    .hero {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .hero p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #f0f0f0;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.75rem;
    }
    
    /* Emotion result card */
    .emotion-result {
        text-align: center;
        padding: 2rem;
        border-radius: 16px;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .emotion-emoji {
        font-size: 4rem;
    }
    
    .emotion-name {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .emotion-confidence {
        font-size: 1.2rem;
        color: #666;
    }
    
    /* Tips */
    .tip-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .tip-item:last-child {
        border-bottom: none;
    }
    
    /* Word cloud container */
    .wordcloud-container {
        padding: 1rem;
        background: white;
        border-radius: 12px;
        text-align: center;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #999;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Initialize
# ============================================================================

@st.cache_resource
def load_model():
    """Load the prediction model."""
    try:
        predictor = get_predictor()
        return predictor
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


def main():
    """Main application."""
    
    # Load model
    predictor = load_model()
    if predictor is None:
        st.stop()
    
    # ========================================================================
    # HERO SECTION
    # ========================================================================
    
    st.markdown("""
    <div class="hero">
        <h1>🧠 MoodLens</h1>
        <p>Understand the emotions behind your words.</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">
            Powered by Machine Learning • Analyze your text in seconds
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### ✍️ What's on your mind?")
        user_text = st.text_area(
            "Enter your text here",
            placeholder="Write about your feelings, thoughts, or experiences...\n\nExample: I'm feeling really happy and excited about today!",
            height=150,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 🔍 Analyze")
        analyze_button = st.button(
            "🚀 Analyze Emotion",
            type="primary",
            use_container_width=True
        )
    
    # ========================================================================
    # PROCESSING
    # ========================================================================
    
    if analyze_button:
        if not user_text or not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            # Show progress
            with st.spinner("Analyzing your text..."):
                # Get prediction
                result = predictor.predict(user_text)
                emotion = result['emotion']
                confidence = result['confidence']
                probabilities = result['probabilities']
                
                # Get recommendation
                recommendation = get_recommendation(emotion, confidence)
                
                # Get important words
                try:
                    important_words = get_words_for_text(user_text, top_n=10)
                except:
                    important_words = []
                
                # ============================================================
                # RESULTS DISPLAY
                # ============================================================
                
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Row 1: Emotion + Confidence
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    color = EMOTION_COLORS.get(emotion, '#95A5A6')
                    emoji = EMOTION_ICONS.get(emotion, '🧠')
                    
                    st.markdown(f"""
                    <div class="emotion-result" style="border-left: 6px solid {color};">
                        <div class="emotion-emoji">{emoji}</div>
                        <div class="emotion-name" style="color: {color};">
                            {emotion.title()}
                        </div>
                        <div class="emotion-confidence">
                            Confidence: {confidence:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card"><div class="card-title">📈 Confidence Gauge</div>', unsafe_allow_html=True)
                    gauge_fig = create_confidence_gauge(confidence, emotion, height=200)
                    if gauge_fig:
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="card"><div class="card-title">🏷️ Emotion Profile</div>', unsafe_allow_html=True)
                    # Show simple bar of probabilities
                    for e, prob in list(probabilities.items())[:3]:
                        pct = prob * 100
                        bar_width = int(pct)
                        color = EMOTION_COLORS.get(e, '#95A5A6')
                        st.markdown(f"""
                        <div style="margin: 4px 0;">
                            <span style="font-size: 0.9rem;">{EMOTION_ICONS.get(e, '')} {e.title()}</span>
                            <div style="background: #f0f0f0; border-radius: 8px; height: 8px; margin-top: 2px;">
                                <div style="background: {color}; width: {pct}%; height: 8px; border-radius: 8px;"></div>
                            </div>
                            <span style="font-size: 0.8rem; color: #666;">{pct:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Row 2: Probability Chart
                st.markdown('<div class="card"><div class="card-title">📊 Probability Distribution</div>', unsafe_allow_html=True)
                fig = plot_probabilities(probabilities, height=350)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Row 3: Recommendation
                st.markdown(f"""
                <div class="card" style="border-left: 6px solid {recommendation.get('color', '#95A5A6')};">
                    <div class="card-title">💡 {recommendation.get('title', 'Recommendation')}</div>
                    <p style="font-size: 1.05rem; line-height: 1.6;">{recommendation.get('message', '')}</p>
                    <div style="margin-top: 1rem;">
                        <p style="font-weight: 600; color: #333;">✨ Quick Tips:</p>
                        <ul style="list-style: none; padding: 0;">
                """, unsafe_allow_html=True)
                
                for tip in recommendation.get('tips', [])[:4]:
                    st.markdown(f"<li style='padding: 4px 0;'>• {tip}</li>", unsafe_allow_html=True)
                
                st.markdown("""
                        </ul>
                    </div>
                    <div style="margin-top: 0.5rem; padding: 0.75rem; background: #f8f9fa; border-radius: 8px;">
                        <span style="font-style: italic; color: #555;">"{recommendation.get('quote', '')}"</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Row 4: Word Cloud & Important Words
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="card"><div class="card-title">☁️ Word Cloud</div>', unsafe_allow_html=True)
                    if MATPLOTLIB_AVAILABLE:
                        wc_fig = create_wordcloud(user_text, max_words=100)
                        if wc_fig:
                            st.pyplot(wc_fig, use_container_width=True)
                        else:
                            st.info("Word cloud not available. Please install matplotlib and wordcloud.")
                    else:
                        st.info("Install matplotlib and wordcloud for word cloud visualization.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card"><div class="card-title">🔑 Important Words</div>', unsafe_allow_html=True)
                    if important_words:
                        for word, score in important_words[:8]:
                            color = 'green' if score > 0 else 'red'
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f0f0f0;">
                                <span style="font-weight: 500;">{word}</span>
                                <span style="color: {color}; font-weight: 600;">{score:+.3f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No important words extracted.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # ============================================================
                # TEXT STATISTICS
                # ============================================================
                
                st.markdown('<div class="card"><div class="card-title">📝 Text Statistics</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Characters", len(user_text))
                with col2:
                    st.metric("Words", len(user_text.split()))
                with col3:
                    st.metric("Sentences", len(user_text.split('.')) - 1)
                with col4:
                    st.metric("Unique Words", len(set(user_text.split())))
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # ============================================================
                # SUPPORTING ACTIVITIES
                # ============================================================
                
                st.markdown('<div class="card"><div class="card-title">🎯 Recommended Activities</div>', unsafe_allow_html=True)
                activities = recommendation.get('activities', [])[:4]
                cols = st.columns(len(activities) if activities else 1)
                
                for i, activity in enumerate(activities):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 12px;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{activity.split(' ')[0]}</div>
                            <div style="font-weight: 500;">{activity}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("""
    <div class="footer">
        <p>🧠 MoodLens • Understand the emotions behind your words</p>
        <p style="font-size: 0.8rem;">Built with Streamlit, scikit-learn, and ❤️</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()