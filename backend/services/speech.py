"""
Speech-to-Text service using Google Cloud Speech-to-Text API
"""
import logging
import traceback
from fastapi import HTTPException, UploadFile
from google.cloud import speech

logger = logging.getLogger(__name__)

# Global Speech-to-Text client
speech_client = None


def initialize_speech_client():
    """Initialize Google Cloud Speech-to-Text client"""
    global speech_client
    
    if speech_client is not None:
        return speech_client
    
    from config import GOOGLE_APPLICATION_CREDENTIALS, GEMINI_API_KEY
    import os
    
    try:
        # Use the same credentials as Gemini (GOOGLE_APPLICATION_CREDENTIALS)
        if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
            speech_client = speech.SpeechClient()
            logger.info("Google Cloud Speech-to-Text client initialized successfully!")
        elif GEMINI_API_KEY:
            # Try to initialize with API key (though Speech-to-Text typically uses service account)
            logger.warning("Speech-to-Text typically requires service account credentials (GOOGLE_APPLICATION_CREDENTIALS)")
            logger.warning("Attempting to initialize with default credentials...")
            try:
                speech_client = speech.SpeechClient()
                logger.info("Google Cloud Speech-to-Text client initialized with default credentials!")
            except Exception as e:
                logger.warning(f"Could not initialize Speech-to-Text client: {e}")
                logger.warning("Speech-to-Text endpoint will not be available")
                speech_client = None
        else:
            logger.warning("Google Cloud Speech-to-Text not configured. Set GOOGLE_APPLICATION_CREDENTIALS to enable.")
            speech_client = None
    except Exception as e:
        logger.warning(f"Could not initialize Speech-to-Text client: {e}")
        logger.warning("Speech-to-Text endpoint will not be available")
        speech_client = None
    
    return speech_client


async def transcribe_audio(audio: UploadFile) -> tuple[str, float]:
    """
    Transcribe audio file to text using Google Cloud Speech-to-Text API
    
    Returns:
        tuple: (transcribed_text, confidence_score)
    """
    if speech_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Speech-to-Text service not available. Please configure GOOGLE_APPLICATION_CREDENTIALS."
        )
    
    try:
        # Read audio file content
        logger.info(f"Received audio file: {audio.filename}, content_type: {audio.content_type}, size: {audio.size if hasattr(audio, 'size') else 'unknown'}")
        audio_content = await audio.read()
        audio_size = len(audio_content)
        logger.info(f"Audio file size: {audio_size} bytes")
        
        if audio_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty audio file received. Please ensure the recording captured audio."
            )
        
        # Determine audio encoding from file extension or content type
        file_extension = audio.filename.split('.')[-1].lower() if audio.filename else 'webm'
        content_type = audio.content_type or ''
        
        logger.info(f"Detected file extension: {file_extension}, content_type: {content_type}")
        
        # Map file extensions and content types to encoding
        encoding_map = {
            'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            'flac': speech.RecognitionConfig.AudioEncoding.FLAC,
            'mp3': speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,  # Auto-detect
            'ogg': speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            'webm': speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        }
        
        # Also check content type
        if 'webm' in content_type.lower():
            encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
        elif 'wav' in content_type.lower() or 'wave' in content_type.lower():
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        elif 'flac' in content_type.lower():
            encoding = speech.RecognitionConfig.AudioEncoding.FLAC
        elif 'mp3' in content_type.lower() or 'mpeg' in content_type.lower():
            encoding = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED  # Auto-detect MP3
        elif 'ogg' in content_type.lower():
            encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
        else:
            encoding = encoding_map.get(file_extension, speech.RecognitionConfig.AudioEncoding.WEBM_OPUS)
        
        logger.info(f"Selected encoding: {encoding}")
        
        # Configure recognition
        config_params = {
            'language_code': "bn-BD",  # Bengali (Bangladesh), can also use "bn-IN" for Bengali (India)
            'enable_automatic_punctuation': True,
            'model': "latest_long",  # Use latest long-form model for better accuracy
        }
        
        # Set encoding and sample_rate based on format
        if encoding == speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED:
            config_params['encoding'] = encoding
        elif encoding == speech.RecognitionConfig.AudioEncoding.WEBM_OPUS:
            config_params['encoding'] = encoding
        elif encoding == speech.RecognitionConfig.AudioEncoding.OGG_OPUS:
            config_params['encoding'] = encoding
        else:
            # For LINEAR16, FLAC, etc., we need sample_rate_hertz
            config_params['encoding'] = encoding
            config_params['sample_rate_hertz'] = 16000  # Common sample rate, adjust if needed
        
        logger.info(f"Recognition config: {config_params}")
        config = speech.RecognitionConfig(**config_params)
        
        audio_data = speech.RecognitionAudio(content=audio_content)
        
        # Perform the transcription
        logger.info("Sending audio to Google Cloud Speech-to-Text API...")
        try:
            response = speech_client.recognize(config=config, audio=audio_data)
            logger.info(f"Received response from API: {len(response.results)} results")
        except Exception as api_error:
            error_type = type(api_error).__name__
            error_msg = str(api_error)
            logger.error(f"Google Cloud Speech-to-Text API error - Type: {error_type}, Message: {error_msg}")
            logger.error(f"Full error details: {repr(api_error)}")
            
            # Handle specific error types
            if 'PERMISSION_DENIED' in error_msg or 'authentication' in error_msg.lower():
                raise HTTPException(
                    status_code=503,
                    detail="Authentication error with Google Cloud Speech-to-Text. Please check GOOGLE_APPLICATION_CREDENTIALS."
                )
            elif 'INVALID_ARGUMENT' in error_msg or 'invalid' in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid audio format or configuration: {error_msg}"
                )
            elif 'RESOURCE_EXHAUSTED' in error_msg or 'quota' in error_msg.lower():
                raise HTTPException(
                    status_code=503,
                    detail="Google Cloud Speech-to-Text quota exceeded. Please try again later."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Google Cloud Speech-to-Text API error: {error_msg}"
                )
        
        # Extract transcribed text
        transcribed_text = ""
        confidence_sum = 0.0
        result_count = 0
        
        for result in response.results:
            if not result.alternatives:
                logger.warning("Result has no alternatives")
                continue
            alternative = result.alternatives[0]
            transcribed_text += alternative.transcript + " "
            if hasattr(alternative, 'confidence') and alternative.confidence:
                confidence_sum += alternative.confidence
                result_count += 1
        
        # Calculate average confidence
        avg_confidence = confidence_sum / result_count if result_count > 0 else 0.0
        
        # Clean up the text
        transcribed_text = transcribed_text.strip()
        
        if not transcribed_text:
            logger.warning("No transcribed text found in API response")
            raise HTTPException(
                status_code=400,
                detail="No speech detected in the audio file. Please check the audio quality and language."
            )
        
        logger.info(f"Transcribed audio: '{transcribed_text[:50]}...' (confidence: {avg_confidence:.2f})")
        
        return transcribed_text, avg_confidence
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        logger.error(f"Error in speech-to-text endpoint - Type: {error_type}, Message: {error_msg}")
        if error_traceback:
            logger.error(f"Traceback:\n{error_traceback}")
        logger.error(f"Full error details: {repr(e)}")
        
        # Provide more helpful error messages
        if 'google' in error_msg.lower() and 'credentials' in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail="Google Cloud credentials not configured. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable."
            )
        elif 'audio' in error_msg.lower() and ('format' in error_msg.lower() or 'encoding' in error_msg.lower()):
            raise HTTPException(
                status_code=400,
                detail=f"Audio format error: {error_msg}. Please ensure the audio is in a supported format (WebM, WAV, FLAC, MP3, OGG)."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Speech-to-text error: {error_msg}"
            )
