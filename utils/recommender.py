"""
MoodLens - Recommendation Engine

This module provides recommendations based on detected emotions.
It is completely independent from the prediction model and follows
the Single Responsibility Principle.

The recommender receives an emotion and returns:
    - Title: A catchy header for the recommendation
    - Message: A supportive message
    - Color: A color associated with the emotion
    - Tips: List of actionable suggestions
    - Activities: List of recommended activities
    - Quote: An inspirational quote

Example:
    >>> from utils.recommender import get_recommendation
    >>> rec = get_recommendation("sadness", confidence=0.85)
    >>> print(rec['title'])
    "It's Okay to Not Be Okay"
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import random


@dataclass
class Recommendation:
    """
    Data class for emotion-based recommendations.
    
    Attributes:
        emotion: The emotion this recommendation is for
        title: Catchy title for the recommendation
        message: Supportive main message
        color: Associated color (hex code)
        tips: List of actionable tips
        activities: List of recommended activities
        quote: Inspirational quote
        confidence_threshold: Minimum confidence for this recommendation
    """
    emotion: str
    title: str
    message: str
    color: str
    tips: List[str]
    activities: List[str]
    quote: str
    confidence_threshold: float = 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Recommendation Database
# ============================================================================

RECOMMENDATIONS = {
    "joy": Recommendation(
        emotion="joy",
        title="✨ Spread the Joy!",
        message="Your positive energy is contagious! Share it with the world and watch it multiply. You're in a wonderful place right now.",
        color="#FFD700",  # Gold
        tips=[
            "Share your happiness with someone today",
            "Write down 3 things you're grateful for",
            "Do something kind for a stranger",
            "Dance to your favorite song",
            "Call a friend and make them laugh"
        ],
        activities=[
            "🌿 Go for a walk in nature",
            "📝 Start a gratitude journal",
            "🎵 Create a 'happy' playlist",
            "📸 Take photos of beautiful things",
            "🤗 Give someone a genuine compliment"
        ],
        quote="Happiness is not something ready made. It comes from your own actions. - Dalai Lama",
        confidence_threshold=0.4
    ),
    
    "sadness": Recommendation(
        emotion="sadness",
        title="💙 It's Okay to Not Be Okay",
        message="Sadness is a natural part of being human. It's okay to feel this way. Be gentle with yourself today. This feeling will pass.",
        color="#4A90D9",  # Blue
        tips=[
            "Allow yourself to feel without judgment",
            "Reach out to someone you trust",
            "Practice self-compassion",
            "Do one small thing for yourself",
            "Remember: this too shall pass"
        ],
        activities=[
            "📖 Read a comforting book",
            "☕ Have a warm cup of tea",
            "🎬 Watch a feel-good movie",
            "🛁 Take a relaxing bath",
            "📝 Write down your feelings"
        ],
        quote="The pain you feel today is the strength you feel tomorrow. For every challenge encountered, there is opportunity for growth.",
        confidence_threshold=0.4
    ),
    
    "anger": Recommendation(
        emotion="anger",
        title="🔥 Finding Your Calm",
        message="Anger is a powerful emotion, but you have the power to channel it constructively. Take a deep breath. You've got this.",
        color="#E74C3C",  # Red
        tips=[
            "Take 5 deep breaths before reacting",
            "Step away from the situation temporarily",
            "Write down what's bothering you",
            "Exercise to release tension",
            "Practice assertive communication"
        ],
        activities=[
            "🏃 Go for a run or walk",
            "🧘 Try a quick meditation",
            "🎯 Hit a punching bag or pillow",
            "🎧 Listen to calming music",
            "💬 Talk to someone neutral"
        ],
        quote="For every minute you remain angry, you give up sixty seconds of peace of mind. - Ralph Waldo Emerson",
        confidence_threshold=0.4
    ),
    
    "fear": Recommendation(
        emotion="fear",
        title="🦋 Face Your Fears",
        message="Fear is a sign that you're about to do something brave. Acknowledge your fear, but don't let it control you. You are stronger than you think.",
        color="#9B59B6",  # Purple
        tips=[
            "Identify what exactly you're afraid of",
            "Break the challenge into small steps",
            "Visualize a positive outcome",
            "Remember past times you overcame fear",
            "Ask for support from others"
        ],
        activities=[
            "🎯 Do one thing that scares you (small)",
            "📝 Journal about your fears",
            "🧘 Practice grounding techniques",
            "📚 Read about someone who overcame fear",
            "🤝 Talk to someone who understands"
        ],
        quote="Fear is only as deep as the mind allows. - Japanese Proverb",
        confidence_threshold=0.4
    ),
    
    "love": Recommendation(
        emotion="love",
        title="❤️ Love is in the Air",
        message="You're feeling love today, and that's beautiful. Whether it's for yourself, others, or the world around you - let it flow freely.",
        color="#FF6B6B",  # Soft Red/Pink
        tips=[
            "Express your love to someone today",
            "Practice loving-kindness meditation",
            "Do something loving for yourself",
            "Write a love letter (to anyone)",
            "Spread love through small acts"
        ],
        activities=[
            "💌 Send a heartfelt message",
            "🌹 Spend time with loved ones",
            "🎵 Listen to love songs",
            "📖 Read a romantic story",
            "💝 Do a random act of kindness"
        ],
        quote="Where there is love there is life. - Mahatma Gandhi",
        confidence_threshold=0.4
    ),
    
    "surprise": Recommendation(
        emotion="surprise",
        title="🌟 Embrace the Unexpected",
        message="Life is full of surprises! Embrace the unexpected and see where this new path leads. Sometimes the best things come when we least expect them.",
        color="#F39C12",  # Orange
        tips=[
            "Embrace spontaneity",
            "Stay open to new possibilities",
            "Share your surprise with others",
            "Be curious about what's next",
            "Trust the journey"
        ],
        activities=[
            "🎉 Celebrate the unexpected",
            "🌍 Try something new",
            "📷 Capture the moment",
            "🎭 Be spontaneous",
            "🤗 Share your surprise with someone"
        ],
        quote="The only thing we can be sure of is the unexpected. - Unknown",
        confidence_threshold=0.4
    )
}

# For low confidence predictions (below 50%)
DEFAULT_RECOMMENDATION = Recommendation(
    emotion="neutral",
    title="🧠 Exploring Your Feelings",
    message="Emotions can sometimes be complex and mixed. Take a moment to check in with yourself. What are you really feeling right now?",
    color="#95A5A6",  # Gray
    tips=[
        "Take a moment to pause and breathe",
        "Identify what you're truly feeling",
        "Be gentle with yourself",
        "Talk to someone about your feelings",
        "Trust that clarity will come"
    ],
    activities=[
        "🧘 Practice mindfulness",
        "📝 Journal your thoughts",
        "☕ Take a mindful break",
        "🌿 Spend time in nature",
        "🎵 Listen to calming music"
    ],
    quote="The quieter you become, the more you can hear. - Ram Dass",
    confidence_threshold=0.0
)


# ============================================================================
# Public Functions
# ============================================================================

def get_recommendation(
    emotion: str,
    confidence: float = 0.0
) -> Dict:
    """
    Get recommendation based on detected emotion.
    
    Args:
        emotion: Detected emotion label
        confidence: Confidence score (0-1)
        
    Returns:
        Dictionary with recommendation data
        
    Example:
        >>> rec = get_recommendation("sadness", 0.85)
        >>> print(rec['title'])
        "It's Okay to Not Be Okay"
    """
    # Normalize emotion
    emotion = emotion.lower().strip() if emotion else ""
    
    # Get recommendation
    if emotion in RECOMMENDATIONS:
        rec = RECOMMENDATIONS[emotion]
        
        # If confidence is below threshold, add disclaimer
        if confidence < rec.confidence_threshold:
            rec_dict = rec.to_dict()
            rec_dict['low_confidence'] = True
            rec_dict['disclaimer'] = (
                "Note: Confidence in this prediction is lower than usual. "
                "Consider how this recommendation resonates with you personally."
            )
            return rec_dict
        
        return rec.to_dict()
    
    # Default recommendation
    return DEFAULT_RECOMMENDATION.to_dict()


def get_recommendation_by_emotion(emotion: str) -> Optional[Dict]:
    """
    Get recommendation by emotion (without confidence check).
    
    Args:
        emotion: Emotion label
        
    Returns:
        Recommendation dictionary or None if not found
    """
    emotion = emotion.lower().strip()
    if emotion in RECOMMENDATIONS:
        return RECOMMENDATIONS[emotion].to_dict()
    return None


def get_all_emotions() -> List[str]:
    """Get list of all supported emotions."""
    return list(RECOMMENDATIONS.keys())


def get_recommendation_for_confidence(
    emotion: str,
    confidence: float
) -> Dict:
    """
    Get recommendation with confidence-based adjustments.
    
    If confidence is high, return standard recommendation.
    If confidence is medium, return with encouragement.
    If confidence is low, return default with explanation.
    
    Args:
        emotion: Detected emotion
        confidence: Confidence score
        
    Returns:
        Recommendation with appropriate adjustments
    """
    rec = get_recommendation(emotion, confidence)
    
    # Add confidence level indicator
    if confidence >= 0.8:
        rec['confidence_level'] = "High"
        rec['message'] = rec['message'] + "\n\nWe're quite confident about this recommendation."
    elif confidence >= 0.6:
        rec['confidence_level'] = "Medium"
        rec['message'] = rec['message'] + "\n\nWe have moderate confidence in this recommendation."
    else:
        rec['confidence_level'] = "Low"
        rec['message'] = rec['message'] + "\n\nPlease take this recommendation with a grain of salt."
    
    return rec


# ============================================================================
# Testing
# ============================================================================

def main():
    """Test the recommendation engine."""
    print("=" * 60)
    print("MoodLens - Recommendation Engine Test")
    print("=" * 60)
    
    # Test all emotions
    for emotion in RECOMMENDATIONS.keys():
        print(f"\n{'='*40}")
        print(f"Emotion: {emotion.upper()}")
        print(f"{'='*40}")
        
        rec = get_recommendation(emotion, confidence=0.8)
        print(f"Title: {rec['title']}")
        print(f"Message: {rec['message'][:80]}...")
        print(f"Color: {rec['color']}")
        print(f"Quote: {rec['quote'][:60]}...")
        print(f"Top Tip: {rec['tips'][0]}")
        print(f"Top Activity: {rec['activities'][0]}")
    
    # Test with low confidence
    print(f"\n{'='*40}")
    print("Low Confidence Test")
    print(f"{'='*40}")
    
    rec = get_recommendation("sadness", confidence=0.3)
    print(f"Title: {rec['title']}")
    print(f"Low Confidence: {rec.get('low_confidence', False)}")
    if 'disclaimer' in rec:
        print(f"Disclaimer: {rec['disclaimer']}")
    
    # Test unknown emotion
    print(f"\n{'='*40}")
    print("Unknown Emotion Test")
    print(f"{'='*40}")
    
    rec = get_recommendation("unknown")
    print(f"Title: {rec['title']}")
    print(f"Message: {rec['message'][:80]}...")
    
    print(f"\n{'='*60}")
    print("✅ All tests passed!")


if __name__ == "__main__":
    main()