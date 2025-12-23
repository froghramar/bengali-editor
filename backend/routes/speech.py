"""
API routes for speech-to-text
"""
from fastapi import UploadFile, File
from schemas import SpeechToTextResponse
from services.speech import transcribe_audio


async def speech_to_text(audio: UploadFile = File(...)) -> SpeechToTextResponse:
    """
    Convert speech audio to text using Google Cloud Speech-to-Text API
    
    Accepts audio file (WAV, FLAC, MP3, etc.) and returns transcribed text in Bengali.
    Supports Bengali language (bn-BD or bn-IN).
    
    Args:
        audio: Audio file to transcribe
        
    Returns:
        SpeechToTextResponse with transcribed text and confidence score
    """
    transcribed_text, confidence = await transcribe_audio(audio)
    
    return SpeechToTextResponse(
        text=transcribed_text,
        confidence=confidence
    )
