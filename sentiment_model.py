from transformers import pipeline
from langdetect import detect, LangDetectException

# Load the multilingual sentiment analysis model
def load_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    return pipeline("sentiment-analysis", model=model_name)

# Detect language
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

# Predict sentiment - returns a dictionary
def predict_sentiment(text, model):
    result = model(text)[0]  # pipeline returns list with one dict

    label = result['label']
    score = result['score']

    # Map the model's labels to custom output
    if label == "positive":
        custom_label = "Positive"
        emoji = "😊"
        message = "Safe to Post Check Mark"
    elif label == "negative":
        custom_label = "Negative"
        emoji = "😠"
        message = "Not Safe"
    elif label == "neutral":
        custom_label = "Neutral"
        emoji = "😐"
        message = "Think before posting"
    else:
        custom_label = label
        emoji = "❓"
        message = "Unknown"

    return {
        "label": custom_label,
        "score": score,
        "emoji": emoji,
        "message": message
    }
