from transformers import pipeline
from langdetect import detect, LangDetectException

def load_model():
    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return classifier

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def predict_sentiment(text, model):
    if not text.strip():
        return {
            "label": "INVALID",
            "score": 0.0,
            "emoji": "❓",
            "message": "Invalid input: Text is empty",
            "language": "unknown"
        }
    
    result = model(text)[0]
    label = result['label']  # e.g., "1 star", "2 stars", ..., "5 stars"
    score = round(result['score'], 2)
    
    # Map labels to sentiment
    sentiment_map = {
        "1 star": "Very Negative",
        "2 stars": "Negative",
        "3 stars": "Neutral",
        "4 stars": "Positive",
        "5 stars": "Very Positive"
    }
    emoji_map = {
        "1 star": "😣",
        "2 stars": "😔",
        "3 stars": "😐",
        "4 stars": "😊",
        "5 stars": "😄"
    }
    
    sentiment = sentiment_map.get(label, "Unknown")
    emoji = emoji_map.get(label, "❓")
    message = "Safe to Post ✅" if label in ["4 stars", "5 stars"] and score > 0.75 else "Think Before Posting ⚠️"
    
    return {
        "label": sentiment,
        "score": score,
        "emoji": emoji,
        "message": message,
        "language": detect_language(text)
    }