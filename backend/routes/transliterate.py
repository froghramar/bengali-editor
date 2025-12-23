"""
API routes for transliteration
"""
import logging
from fastapi import HTTPException
from schemas import CompletionRequest, CompletionResponse
from config import USE_GEMINI_TRANSLITERATE
from services.gemini import transliterate_with_gemini
from services.transformers import transliterate_with_transformers
from models.loader import get_models

logger = logging.getLogger(__name__)


async def transliterate_text(request: CompletionRequest) -> CompletionResponse:
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
            models = get_models()
            suggestions = await transliterate_with_transformers(
                input_text, 
                request.max_suggestions,
                models['transliteration_model'],
                models['transliteration_tokenizer'],
                models['device']
            )
        
        logger.info(f"Transliterated '{input_text}' -> {suggestions} (using {'Gemini' if USE_GEMINI_TRANSLITERATE else 'Transformers'})")
        
        return CompletionResponse(
            suggestions=suggestions,
            input_text=input_text
        )
        
    except Exception as e:
        logger.error(f"Error in transliteration: {e}")
        raise HTTPException(status_code=500, detail=str(e))
