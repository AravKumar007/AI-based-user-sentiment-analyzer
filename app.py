import streamlit as st
import pandas as pd
import os
import datetime
import tempfile
import re
from sentiment_model import load_model, predict_sentiment, detect_language
from transcribe_audio import transcribe_audio
from filelock import FileLock

# Load model once
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# CSV Logging setup
CSV_LOG_PATH = "sentiment_log.csv"
LOCK_PATH = "sentiment_log.lock"

# Emoji mapping
emoji_map = {
    "Very Negative": "😣 Very Negative",
    "Negative": "😔 Negative",
    "Neutral": "😐 Neutral",
    "Positive": "😊 Positive",
    "Very Positive": "😄 Very Positive",
    "INVALID": "❓ Invalid"
}

# Post suggestion logic based on score
def should_post(score, label):
    if label in ["Positive", "Very Positive"] and score >= 0.75:
        return "✅ Safe to post"
    elif label == "Neutral" or score >= 0.6:
        return "⚠️ Be cautious. May be misunderstood."
    else:
        return "🚫 Not recommended to post. Might hurt sentiments."

# Sanitize input to prevent CSV injection
def sanitize_input(text):
    if not text:
        return ""
    # Remove dangerous characters that could lead to CSV injection
    text = re.sub(r'^[=+@-]', '', text)
    return text.replace(',', ';').replace('\n', ' ')

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🌐 Multi-Lingual Sentiment Analyzer")
st.write("Upload audio or enter text. We'll predict the sentiment and suggest if it's safe to post.")

# Sidebar with help
st.sidebar.header("Help")
st.sidebar.markdown("""
- **Text Input**: Enter text in any supported language (e.g., English, Spanish, French, Hindi).
- **Audio Input**: Upload .mp3 or .wav files (max 10MB). Ensure clear audio for best transcription.
- **Language**: Select a language or use auto-detection for optimal results.
- **Log**: View past analyses in the 'History' tab.
- Built with Hugging Face, Whisper, and Streamlit.
""")

# Tabs for input and history
tab1, tab2 = st.tabs(["Analyze Sentiment", "History"])

with tab1:
    option = st.radio("Choose Input Type:", ("Text", "Audio"))
    language = st.selectbox("Select Language", ["Auto-detect", "English", "Spanish", "French", "German", "Hindi", "Chinese", "Japanese"])

    if option == "Text":
        text_input = st.text_area("Your text", placeholder="Type something like 'I love learning Python!'", max_chars=500)

        if st.button("Analyze Sentiment"):
            if text_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing..."):
                    result = predict_sentiment(text_input, model)

                if result["label"] == "INVALID":
                    st.error(result["message"])
                else:
                    st.success(f"**Sentiment:** {emoji_map.get(result['label'], result['label'])}")
                    st.info(f"**Confidence Score:** {result['score']:.2f}")
                    st.warning(f"**Advice:** {should_post(result['score'], result['label'])}")
                    st.info(f"**Detected Language:** {result['language'].capitalize()}")

                    # Save to CSV with file lock
                    log_data = pd.DataFrame([[
                        datetime.datetime.now(),
                        sanitize_input(text_input),
                        result['label'],
                        result['score'],
                        should_post(result['score'], result['label']),
                        result['language']
                    ]], columns=["Timestamp", "Input", "Sentiment", "Score", "Advice", "Language"])
                    
                    with FileLock(LOCK_PATH):
                        if os.path.exists(CSV_LOG_PATH):
                            log_data.to_csv(CSV_LOG_PATH, mode='a', header=False, index=False)
                        else:
                            log_data.to_csv(CSV_LOG_PATH, index=False)

    elif option == "Audio":
        uploaded_file = st.file_uploader("Upload audio file (.mp3 or .wav)", type=["mp3", "wav"], accept_multiple_files=False)

        if uploaded_file is not None:
            # Validate file size (max 10MB)
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File size exceeds 10MB limit.")
            else:
                with st.spinner("Processing audio..."):
                    # Secure temporary file handling
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_path = temp_file.name

                    try:
                        # Transcribe with specified or detected language
                        lang = None if language == "Auto-detect" else language.lower()
                        transcribed_text = transcribe_audio(temp_path, language=lang)
                        
                        if "Error" in transcribed_text:
                            st.error(transcribed_text)
                        else:
                            st.success("Transcription complete!")
                            st.write("📝 Transcribed Text:")
                            st.markdown(f"> {transcribed_text}")

                            result = predict_sentiment(transcribed_text, model)
                            st.success(f"**Sentiment:** {emoji_map.get(result['label'], result['label'])}")
                            st.info(f"**Confidence Score:** {result['score']:.2f}")
                            st.warning(f"**Advice:** {should_post(result['score'], result['label'])}")
                            st.info(f"**Detected Language:** {result['language'].capitalize()}")

                            # Save to CSV with file lock
                            log_data = pd.DataFrame([[
                                datetime.datetime.now(),
                                sanitize_input(transcribed_text),
                                result['label'],
                                result['score'],
                                should_post(result['score'], result['label']),
                                result['language']
                            ]], columns=["Timestamp", "Input", "Sentiment", "Score", "Advice", "Language"])
                            
                            with FileLock(LOCK_PATH):
                                if os.path.exists(CSV_LOG_PATH):
                                    log_data.to_csv(CSV_LOG_PATH, mode='a', header=False, index=False)
                                else:
                                    log_data.to_csv(CSV_LOG_PATH, index=False)
                    finally:
                        os.remove(temp_path)

with tab2:
    st.subheader("Sentiment Analysis History")
    if os.path.exists(CSV_LOG_PATH):
        log_df = pd.read_csv(CSV_LOG_PATH)
        st.dataframe(log_df)

        # Sentiment trend chart
        if not log_df.empty:
            sentiment_counts = log_df['Sentiment'].value_counts().reindex(
                ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"], fill_value=0
            )
            chart_data = {
                "type": "bar",
                "data": {
                    "labels": ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"],
                    "datasets": [{
                        "label": "Sentiment Distribution",
                        "data": sentiment_counts.tolist(),
                        "backgroundColor": ["#4CAF50", "#66BB6A", "#FFC107", "#EF5350", "#B71C1C"],
                        "borderColor": ["#388E3C", "#4CAF50", "#FFA000", "#D32F2F", "#7F0000"],
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": { "display": True, "text": "Count" }
                        },
                        "x": {
                            "title": { "display": True, "text": "Sentiment" }
                        }
                    }
                }
            }
            st.json(chart_data)  # Render as Chart.js chart in Streamlit
    else:
        st.info("No analysis history available.")
