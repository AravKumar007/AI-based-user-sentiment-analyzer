import streamlit as st
import pandas as pd
import os
import datetime
import tempfile
from filelock import FileLock
from sentiment_model import load_model, predict_sentiment, detect_language
from transcribe_audio import transcribe_audio

# FIRST STREAMLIT COMMAND - MUST BE HERE
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Cache the model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

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

                    # Get result as dict
                    result = predict_sentiment(input_text, model)

                    label = result["label"]
                    score = result["score"]
                    emoji = result["emoji"]
                    message = result["message"]

                    # Display sentiment
                    if label == "Positive":
                        st.success(f"{emoji} {label} (Confidence: {score:.2f})")
                    elif label == "Negative":
                        st.error(f"{emoji} {label} (Confidence: {score:.2f})")
                    else:
                        st.warning(f"{emoji} {label} (Confidence: {score:.2f})")

                    # Safe to post
                    st.markdown("### Safe to Post?")
                    if message == "Safe to Post Check Mark":
                        st.success("✅ Yes! This content is positive and safe to post.")
                    else:
                        st.warning("⚠️ Think before posting.")

                    # Log to CSV
                    log_entry = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "input_type": "Audio" if audio_file else "Text",
                        "language": lang,
                        "sentiment": label,
                        "confidence": round(score, 3),
                        "text": input_text[:500]
                    }

                    csv_file = "sentiment_history.csv"
                    lock = FileLock(csv_file + ".lock")
                    with lock:
                        if os.path.exists(csv_file):
                            df = pd.read_csv(csv_file)
                            df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
                        else:
                            df = pd.DataFrame([log_entry])
                        df.to_csv(csv_file, index=False)

        else:
            st.warning("Please enter text or upload an audio file.")

with tab2:
    st.header("Sentiment History")
    if os.path.exists("sentiment_history.csv"):
        df = pd.read_csv("sentiment_history.csv")
        st.dataframe(df)

        if len(df) > 1:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
            df['score'] = df['sentiment'].map(sentiment_map).fillna(0)

            st.line_chart(df['score'])
            st.caption("Sentiment Trend (+1 = Positive, 0 = Neutral, -1 = Negative)")
    else:
        st.info("No history yet. Analyze something first!")

with tab3:
    st.header("About")
    st.markdown("""
    ### Features
    - Multilingual support
    - Voice transcription with Whisper
    - Sentiment analysis
    - History & graphs
    - Safe-to-post advice

    Made with ❤️ using Streamlit, Transformers, and Whisper.
    """)
    st.markdown("[GitHub Repo](https://github.com/AravKumar007/AI-based-user-sentiment-analyzer)")
    st.markdown("[Live Demo](https://arav9696-sentiment-analysis.hf.space)")
