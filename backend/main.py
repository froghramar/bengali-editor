"""
Bengali Text Auto-completion Backend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from typing import List
import logging
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import speech
import io

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bengali Auto-completion API")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration: Switch between implementations using environment variables
# These can be set in a .env file (see .env.example) or as system environment variables
# Set USE_GEMINI_COMPLETE=true to use Gemini for completion
# Set USE_GEMINI_TRANSLITERATE=true to use Gemini for transliteration
# Authentication: Use either GEMINI_API_KEY (API key) or GOOGLE_APPLICATION_CREDENTIALS (service account JSON path)
USE_GEMINI_COMPLETE = os.getenv("USE_GEMINI_COMPLETE", "false").lower() == "true"
USE_GEMINI_TRANSLITERATE = os.getenv("USE_GEMINI_TRANSLITERATE", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Global model variables
model = None
tokenizer = None
device = None

# Transliteration model variables
transliteration_model = None
transliteration_tokenizer = None

# Gemini client
gemini_client = None

# Speech-to-Text client
speech_client = None

class CompletionRequest(BaseModel):
    text: str
    max_suggestions: int = 5
    max_length: int = 20  # Max tokens to generate

class CompletionResponse(BaseModel):
    suggestions: List[str]
    input_text: str

class SpeechToTextResponse(BaseModel):
    text: str
    confidence: float = 0.0

def normalize_bengali_text(text):
    """
    Basic Bengali text normalization
    For full normalization, install: pip install git+https://github.com/csebuetnlp/normalizer
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def complete_with_gemini(input_text: str, max_suggestions: int, max_length: int) -> List[str]:
    """Generate completions using Gemini Flash model"""
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")
    
    prompt = f"""You are a Bengali text completion assistant. Given a partial Bengali text, suggest {max_suggestions} natural completions.

Input text: {input_text}

Provide {max_suggestions} completion suggestions, each should be a natural continuation of the input text. Each suggestion should be around {max_length} words or less. Return only the completion part (not the input text repeated).

Format your response as a numbered list, one suggestion per line."""
    
    try:
        generation_config = {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 50,
            'max_output_tokens': max_length * 10,  # Approximate token count
        }
        response = gemini_client.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        suggestions = []
        # Handle response - in 0.8.6, response.text might raise exception if blocked
        try:
            response_text = response.text.strip()
        except ValueError as e:
            logger.error(f"Gemini response blocked or invalid: {e}")
            # Try to get candidates if available
            if hasattr(response, 'candidates') and response.candidates:
                response_text = response.candidates[0].content.parts[0].text.strip()
            else:
                raise HTTPException(status_code=500, detail=f"Gemini response error: {str(e)}")
        
        # Parse the response - could be numbered list or plain text
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            # Remove numbering if present (e.g., "1. ", "1) ", "- ")
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^-\s*', '', line)
            line = line.strip()
            
            if line and len(line) > 0:
                # Extract just the completion part (remove input if repeated)
                if input_text in line:
                    completion = line.replace(input_text, '').strip()
                else:
                    completion = line
                
                if completion and completion not in suggestions:
                    suggestions.append(completion)
        
        return suggestions[:max_suggestions]
    except Exception as e:
        logger.error(f"Error with Gemini completion: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini completion error: {str(e)}")

async def transliterate_with_gemini(input_text: str, max_suggestions: int) -> List[str]:
    """Transliterate Banglish to Bengali and provide autocompletion suggestions using Gemini Flash model"""
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")
    
    # Check if input is already in Bengali
    is_bengali = any('\u0980' <= char <= '\u09FF' for char in input_text)
    
    if is_bengali:
        # For Bengali input, provide autocompletion suggestions
        prompt = f"""You are a Bengali text autocompletion assistant. Given a partial Bengali word or phrase, suggest {max_suggestions} natural completions.

Input (Bengali): {input_text}

Provide {max_suggestions} Bengali word/phrase completion suggestions that start with or continue from the input. Return only the complete words/phrases in Bengali script.

Format your response as a numbered list, one suggestion per line."""
    else:
        # For Banglish input, transliterate and provide autocompletion suggestions
        prompt = f"""You are a Banglish to Bengali transliteration and autocompletion assistant. 

Input (Banglish): {input_text}

For this input, provide {max_suggestions} suggestions that include:
1. The exact transliteration of the input to Bengali script
2. Autocompletion suggestions - complete Bengali words/phrases that start with the transliterated partial input

For example, if input is "am", return suggestions like: "আম", "আমি", "আমার", "আমাকে", etc.

Return all suggestions in Bengali script. Format your response as a numbered list, one suggestion per line."""
    
    try:
        generation_config = {
            'temperature': 0.5,  # Slightly higher for more diverse autocompletion suggestions
            'top_p': 0.95,
            'top_k': 50,
            'max_output_tokens': 300,  # Increased for more suggestions
        }
        response = gemini_client.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        suggestions = []
        # Handle response - in 0.8.6, response.text might raise exception if blocked
        try:
            response_text = response.text.strip()
        except ValueError as e:
            logger.error(f"Gemini response blocked or invalid: {e}")
            # Try to get candidates if available
            if hasattr(response, 'candidates') and response.candidates:
                response_text = response.candidates[0].content.parts[0].text.strip()
            else:
                raise HTTPException(status_code=500, detail=f"Gemini response error: {str(e)}")
        
        # Parse the response
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            # Remove numbering if present (e.g., "1. ", "1) ", "- ")
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^-\s*', '', line)
            line = line.strip()
            
            if line and len(line) > 0:
                # Check if it contains Bengali characters
                if any('\u0980' <= char <= '\u09FF' for char in line):
                    # Remove any English/Banglish text that might be mixed in
                    # Extract only Bengali parts
                    bengali_parts = re.findall(r'[\u0980-\u09FF]+', line)
                    if bengali_parts:
                        # Join Bengali parts and add to suggestions
                        bengali_text = ' '.join(bengali_parts)
                        if bengali_text and bengali_text not in suggestions:
                            suggestions.append(bengali_text)
                elif not is_bengali:
                    # If input was Banglish and we got non-Bengali text, 
                    # it might be a transliteration attempt - skip it
                    pass
        
        # If no suggestions found, return the input as fallback (if it's already Bengali)
        if not suggestions:
            if is_bengali:
                suggestions = [input_text]
            else:
                # For Banglish input with no suggestions, return empty list
                # (or could try to transliterate manually, but let's keep it simple)
                suggestions = []
        
        return suggestions[:max_suggestions]
    except Exception as e:
        logger.error(f"Error with Gemini transliteration: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini transliteration error: {str(e)}")

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer, device
    global transliteration_model, transliteration_tokenizer
    global gemini_client
    global speech_client
    
    try:
        # Initialize Gemini if needed
        if USE_GEMINI_COMPLETE or USE_GEMINI_TRANSLITERATE:
            # Support both API key and Vertex AI service account authentication
            if GOOGLE_APPLICATION_CREDENTIALS:
                # Use Vertex AI service account (service account JSON file)
                if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
                    logger.error(f"Service account file not found: {GOOGLE_APPLICATION_CREDENTIALS}")
                    raise FileNotFoundError(f"Service account file not found: {GOOGLE_APPLICATION_CREDENTIALS}")
                # Set the environment variable for Google auth libraries
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
                logger.info(f"Using Vertex AI service account: {GOOGLE_APPLICATION_CREDENTIALS}")
                # No need to configure API key when using service account
                gemini_client = genai.GenerativeModel('gemini-2.0-flash')
            elif GEMINI_API_KEY:
                # Use API key authentication
                genai.configure(api_key=GEMINI_API_KEY)
                gemini_client = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Using Gemini API key authentication")
            else:
                logger.warning("Neither GEMINI_API_KEY nor GOOGLE_APPLICATION_CREDENTIALS is set.")
                logger.warning("Set one of these environment variables to use Gemini:")
                logger.warning("  - GEMINI_API_KEY: Your Gemini API key")
                logger.warning("  - GOOGLE_APPLICATION_CREDENTIALS: Path to your service account JSON file")
                raise ValueError("Gemini authentication not configured")
            
            logger.info("Gemini Flash model initialized successfully!")
        
        # Initialize Google Cloud Speech-to-Text client
        try:
            # Use the same credentials as Gemini (GOOGLE_APPLICATION_CREDENTIALS)
            if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
                speech_client = speech.SpeechClient()
                logger.info("Google Cloud Speech-to-Text client initialized successfully!")
            elif GEMINI_API_KEY:
                # Try to initialize with API key (though Speech-to-Text typically uses service account)
                # For API key, we might need to use a different approach
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
            if USE_GEMINI_COMPLETE:
                logger.info("Using Gemini Flash for /complete endpoint")
            if USE_GEMINI_TRANSLITERATE:
                logger.info("Using Gemini Flash for /transliterate endpoint")
        
        # Load transformers models only if not using Gemini
        if not USE_GEMINI_COMPLETE or not USE_GEMINI_TRANSLITERATE:
            logger.info("Loading transformers models...")
            
            # Check for GPU availability
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Load completion model only if not using Gemini
            if not USE_GEMINI_COMPLETE:
                logger.info("Loading completion model...")
                model_name = "bigscience/bloom-560m"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model.to(device)
                model.eval()  # Set to evaluation mode
                logger.info("Completion model loaded successfully!")
                logger.info(f"Model vocabulary size: {len(tokenizer)}")

            # Load transliteration model only if not using Gemini
            if not USE_GEMINI_TRANSLITERATE:
                logger.info("Loading transliteration model...")
                transliteration_model_name = "Mdkaif2782/banglish-to-bangla"
                
                try:
                    transliteration_tokenizer = MBart50TokenizerFast.from_pretrained(transliteration_model_name)
                    transliteration_model = MBartForConditionalGeneration.from_pretrained(transliteration_model_name)
                    transliteration_model.to(device)
                    transliteration_model.eval()
                    logger.info("Transliteration model loaded successfully!")
                    logger.info(f"Transliteration model vocabulary size: {len(transliteration_tokenizer)}")
                except Exception as e:
                    logger.warning(f"Could not load transliteration model: {e}")
                    logger.warning("Transliteration endpoint will not be available")
                    transliteration_model = None
                    transliteration_tokenizer = None
        else:
            logger.info("Skipping transformers model loading (using Gemini for all endpoints)")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "device": str(device) if device else "N/A",
        "config": {
            "completion_backend": "Gemini Flash" if USE_GEMINI_COMPLETE else "Transformers",
            "transliteration_backend": "Gemini Flash" if USE_GEMINI_TRANSLITERATE else "Transformers",
            "gemini_configured": gemini_client is not None
        }
    }

async def complete_with_transformers(input_text: str, max_suggestions: int, max_length: int) -> List[str]:
    """Generate completions using transformers model (original implementation)"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # T5 approach 1: Use task prefix for completion
    task_input = f"Complete: {input_text}"
    
    # Tokenize input
    input_ids = tokenizer.encode(
        task_input, 
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    # Generate completions using beam search
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length + input_ids.shape[1],
            num_beams=max_suggestions * 2,
            num_return_sequences=max_suggestions,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            early_stopping=True,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            length_penalty=1.0
        )
    
    # Decode and extract suggestions
    suggestions = []
    for output in outputs:
        # Decode the generated text
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        
        # Clean up the suggestion
        suggestion = generated_text.strip()
        
        # Remove the input text if it's repeated
        if suggestion.startswith(task_input):
           suggestion = suggestion.removeprefix(task_input).strip()
        
        # Take first few words as suggestion (not the entire generation)
        words = suggestion.split()
        if words:
            # Suggest 1-10 words depending on context
            word_count = min(10, len(words))
            suggestion = ' '.join(words[:word_count])
            
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)
    
    # Remove duplicates and empty suggestions
    suggestions = [s for s in suggestions if s][:max_suggestions]
    return suggestions

@app.post("/complete", response_model=CompletionResponse)
async def get_completions(request: CompletionRequest):
    """
    Generate auto-completion suggestions for Bengali text
    
    Uses either Gemini Flash or transformers model based on USE_GEMINI_COMPLETE environment variable.
    Set USE_GEMINI_COMPLETE=true to use Gemini Flash, false (default) to use transformers.
    
    Args:
        request: CompletionRequest with text and parameters
        
    Returns:
        CompletionResponse with suggestions
    """
    try:
        input_text = normalize_bengali_text(request.text.strip())
        
        if not input_text:
            return CompletionResponse(suggestions=[], input_text=input_text)
        
        # Route to appropriate implementation
        if USE_GEMINI_COMPLETE:
            suggestions = await complete_with_gemini(input_text, request.max_suggestions, request.max_length)
        else:
            suggestions = await complete_with_transformers(input_text, request.max_suggestions, request.max_length)
        
        logger.info(f"Generated {len(suggestions)} suggestions for: '{input_text[:50]}...' (using {'Gemini' if USE_GEMINI_COMPLETE else 'Transformers'})")
        
        return CompletionResponse(
            suggestions=suggestions,
            input_text=input_text
        )
        
    except Exception as e:
        logger.error(f"Error generating completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
async def transliterate_with_transformers(input_text: str, max_suggestions: int) -> List[str]:
    """Transliterate Banglish to Bengali using transformers model (original implementation)"""
    if transliteration_model is None or transliteration_tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Transliteration model not loaded. Please check backend logs."
        )
    
    # Check if input is already in Bengali
    if any('\u0980' <= char <= '\u09FF' for char in input_text):
        # Input contains Bengali characters, return as-is
        return [input_text]
    
    # Prepare input for mBART model
    inputs = transliteration_tokenizer(
        input_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    ).to(device)
    
    # Generate multiple suggestions
    with torch.no_grad():
        outputs = transliteration_model.generate(
            **inputs,
            decoder_start_token_id=transliteration_tokenizer.lang_code_to_id["bn_IN"],
            max_length=128,
            num_beams=max_suggestions * 2,
            num_return_sequences=min(max_suggestions, 5),
            temperature=0.7,
            do_sample=True,
            top_k=50,
            early_stopping=True
        )
    
    # Decode suggestions
    suggestions = []
    for output in outputs:
        translated = transliteration_tokenizer.decode(output, skip_special_tokens=True)
        if translated and translated not in suggestions:
            suggestions.append(translated)
    
    # Remove duplicates while preserving order
    suggestions = list(dict.fromkeys(suggestions))[:max_suggestions]
    return suggestions if suggestions else [input_text]

@app.post("/transliterate", response_model=CompletionResponse)
async def transliterate_text(request: CompletionRequest):
    """
    Transliterate Banglish to Bengali and provide autocompletion suggestions
    
    Uses either Gemini Flash or transformers model based on USE_GEMINI_TRANSLITERATE environment variable.
    Set USE_GEMINI_TRANSLITERATE=true to use Gemini Flash, false (default) to use transformers.
    
    This endpoint provides both transliteration and autocompletion:
    - For partial Banglish input: Returns transliterated partial + autocompletion suggestions
    - For complete Banglish input: Returns transliteration + related suggestions
    - For partial Bengali input: Returns autocompletion suggestions
    - For complete Bengali input: Returns the input + related suggestions
    
    Examples:
    - "ami" -> ["আমি", "আমি তোমাকে", "আমি ভাত"]
    - "am" -> ["আম", "আমি", "আমার", "আমাকে"]
    - "তুমি" -> ["তুমি", "তুমি কেমন", "তুমি কেমন আছো"]
    - "তু" -> ["তুমি", "তুমার", "তুই"]
    """
    try:
        input_text = request.text.strip()
        
        if not input_text:
            return CompletionResponse(suggestions=[], input_text=input_text)
        
        # Route to appropriate implementation
        if USE_GEMINI_TRANSLITERATE:
            suggestions = await transliterate_with_gemini(input_text, request.max_suggestions)
        else:
            suggestions = await transliterate_with_transformers(input_text, request.max_suggestions)
        
        logger.info(f"Transliterated '{input_text}' -> {suggestions} (using {'Gemini' if USE_GEMINI_TRANSLITERATE else 'Transformers'})")
        
        return CompletionResponse(
            suggestions=suggestions,
            input_text=input_text
        )
        
    except Exception as e:
        logger.error(f"Error in transliteration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Convert speech audio to text using Google Cloud Speech-to-Text API
    
    Accepts audio file (WAV, FLAC, MP3, etc.) and returns transcribed text in Bengali.
    Supports Bengali language (bn-BD or bn-IN).
    
    Args:
        audio: Audio file to transcribe
        
    Returns:
        SpeechToTextResponse with transcribed text and confidence score
    """
    if speech_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Speech-to-Text service not available. Please configure GOOGLE_APPLICATION_CREDENTIALS."
        )
    
    try:
        # Read audio file content
        audio_content = await audio.read()
        
        # Determine audio encoding from file extension or content type
        file_extension = audio.filename.split('.')[-1].lower() if audio.filename else 'webm'
        content_type = audio.content_type or ''
        
        # Map file extensions and content types to encoding
        encoding_map = {
            'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            'flac': speech.RecognitionConfig.AudioEncoding.FLAC,
            'mp3': speech.RecognitionConfig.AudioEncoding.MP3,
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
            encoding = speech.RecognitionConfig.AudioEncoding.MP3
        elif 'ogg' in content_type.lower():
            encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
        else:
            encoding = encoding_map.get(file_extension, speech.RecognitionConfig.AudioEncoding.WEBM_OPUS)
        
        # Configure recognition
        # For WebM/Opus, we don't need to specify sample_rate_hertz
        # For other formats, we'll use a common rate or let the API auto-detect
        config_params = {
            'language_code': "bn-BD",  # Bengali (Bangladesh), can also use "bn-IN" for Bengali (India)
            'enable_automatic_punctuation': True,
            'model': "latest_long",  # Use latest long-form model for better accuracy
        }
        
        # Only set encoding and sample_rate for formats that require it
        if encoding != speech.RecognitionConfig.AudioEncoding.WEBM_OPUS:
            config_params['encoding'] = encoding
            config_params['sample_rate_hertz'] = 16000  # Common sample rate
        else:
            config_params['encoding'] = encoding
        
        config = speech.RecognitionConfig(**config_params)
        
        audio_data = speech.RecognitionAudio(content=audio_content)
        
        # Perform the transcription
        response = speech_client.recognize(config=config, audio=audio_data)
        
        # Extract transcribed text
        transcribed_text = ""
        confidence_sum = 0.0
        result_count = 0
        
        for result in response.results:
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
            raise HTTPException(
                status_code=400,
                detail="No speech detected in the audio file. Please check the audio quality and language."
            )
        
        logger.info(f"Transcribed audio: '{transcribed_text[:50]}...' (confidence: {avg_confidence:.2f})")
        
        return SpeechToTextResponse(
            text=transcribed_text,
            confidence=avg_confidence
        )
        
    except Exception as e:
        logger.error(f"Error in speech-to-text: {e}")
        raise HTTPException(status_code=500, detail=f"Speech-to-text error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)