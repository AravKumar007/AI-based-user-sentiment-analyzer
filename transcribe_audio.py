import whisper
from langdetect import detect, LangDetectException

def transcribe_audio(audio_path, language=None):
    try:
        # Load Whisper model (base for performance, can switch to 'large-v3' for better accuracy)
        model = whisper.load_model("base")
        
        # If language is not provided, try to detect it from a sample of the audio
        if language is None:
            # Whisper can auto-detect language, but we can improve accuracy with langdetect
            result = model.transcribe(audio_path, language=None)
            detected_lang = result.get("language", "unknown")
        else:
            detected_lang = language
            result = model.transcribe(audio_path, language=detected_lang)
        
        text = result["text"].strip()
        if not text:
            return "Error: No text transcribed from audio."
        
        return text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"