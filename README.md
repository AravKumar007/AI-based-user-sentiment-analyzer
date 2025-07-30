An AI based Multi-Lingual Sentiment Analyzer
------------------------------------------------------------------------------

# AI-based Multi-Lingual Sentiment Analyzer

[![Streamlit App](https://img.shields.io/badge/Live-Demo-00C853?logo=streamlit&logoColor=white)](https://ai-based-user-sentiment-analyzer.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/Source-Code-181717?logo=github)](https://github.com/AravKumar007/AI-based-user-sentiment-analyzer)


A Streamlit-based application for analyzing sentiment in text or audio inputs across multiple languages. It uses a multilingual BERT model for sentiment analysis and Whisper for audio transcription.
Features
🔤 Analyze text OR audio (`.mp3`, `.wav`)
- 🌍 Supports **multiple languages** (auto-detects)
- 🌟 Provides sentiment score (1–5 stars)
- ⚠️ Suggests if content is "Safe to Post" or "Think Before Posting"
- 📁 Logs results to a secure CSV
- 📈 Shows sentiment trend/history graphs


Setup----

Clone the repository:  
`git clone https://github.com/AravKumar007/AI-based-user-sentiment-analyzer.git`


Install dependencies:pip install -r requirements.txt


Run the app:

```bash
python -m venv .venv
.venv\Scripts\activate       # For Windows
pip install -r requirements.txt
streamlit run app.py



Requirements
See requirements.txt for dependencies.
Usage

Text Input: Enter text in the UI and select a language (or auto-detect).
Audio Input: Upload an audio file (max 10MB). Ensure clear audio for accurate transcription.
History: View past analyses in the "History" tab.

Supported Languages

English, Spanish, French, German, Hindi, Chinese, Japanese, and more (via auto-detection).

Limitations

Audio files must be clear for accurate transcription.
Sentiment analysis accuracy depends on the input language and context.

Deployment
Use the provided Dockerfile for containerized deployment:
docker build -t sentiment-analyzer .
docker run -p 8501:8501 sentiment-analyzer

Built With--

Hugging Face Transformers
OpenAI Whisper
Streamlit


