import streamlit as st
import pandas as pd
import os
import datetime
import tempfile
from filelock import FileLock
from sentiment_model import load_model, predict_sentiment, detect_language
from transcribe_audio import transcribe_audio

# FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Cache model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Initialize session history (private per browser tab)
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🧠 AI-Based User Sentiment Analyzer")

st.markdown("Analyze sentiment from text or voice (English, Hindi, and mixed languages supported).")

tab1, tab2, tab3 = st.tabs(["Analyze", "History", "About"])

with tab1:
    st.header("Input Text or Voice")

    text_input = st.text_area("Enter text here:", height=150)

    audio_file = st.file_uploader("Or upload an audio file (wav/mp3)", type=["wav", "mp3"])

    if st.button("Analyze Sentiment"):
        if text_input or audio_file:
            with st.spinner("Processing..."):
                if audio_file:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(audio_file.read())
                    audio_path = tfile.name

                    try:
                        transcribed_text = transcribe_audio(audio_path)
                        st.success("Transcription successful!")
                        st.write("**Transcribed Text:**")
                        st.write(transcribed_text)
                        input_text = transcribed_text
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        input_text = ""
                    finally:
                        os.unlink(audio_path)
                else:
                    input_text = text_input

                if input_text.strip():
                    lang = detect_language(input_text)
                    st.info(f"Detected Language: {lang.upper()}")

                    result = predict_sentiment(input_text, model)

                    label = result["label"]
                    score = result["score"]
                    emoji = result["emoji"]
                    message = result["message"]

                    if label == "Positive":
                        st.success(f"{emoji} {label} (Confidence: {score:.2f})")
                    elif label == "Negative":
                        st.error(f"{emoji} {label} (Confidence: {score:.2f})")
                    else:
                        st.warning(f"{emoji} {label} (Confidence: {score:.2f})")

                    st.markdown("### Safe to Post?")
                    if message == "Safe to Post Check Mark":
                        st.success("✅ Yes! This content is positive and safe to post.")
                    else:
                        st.warning("⚠️ Think before posting.")

                    # Add to PRIVATE session history
                    log_entry = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "input_type": "Audio" if audio_file else "Text",
                        "language": lang,
                        "sentiment": label,
                        "confidence": round(score, 3),
                        "text": input_text[:500]
                    }
                    st.session_state.history.append(log_entry)

        else:
            st.warning("Please enter text or upload an audio file.")

with tab2:
    st.header("Your Private History (This Session Only)")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        if len(df) > 1:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
            df['score'] = df['sentiment'].map(sentiment_map).fillna(0)

            st.line_chart(df['score'])
            st.caption("Your Personal Sentiment Trend")
        
        # Optional: Clear history button
        if st.button("Clear My History"):
            st.session_state.history = []
            st.success("History cleared!")
    else:
        st.info("No entries yet in this session. Start analyzing!")

with tab3:
    st.header("About")
    st.markdown("Multilingual text/voice sentiment analysis with private session history.")
    st.markdown("[GitHub](https://github.com/AravKumar007/AI-based-user-sentiment-analyzer)")
    st.markdown("[Live Demo](https://arav9696-sentiment-analysis.hf.space)")
