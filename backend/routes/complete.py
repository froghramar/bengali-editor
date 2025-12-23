"""
API routes for text completion
"""
import logging
from fastapi import HTTPException
from schemas import CompletionRequest, CompletionResponse
from utils import normalize_bengali_text
from config import USE_GEMINI_COMPLETE
from services.gemini import complete_with_gemini
from services.transformers import complete_with_transformers
from models.loader import get_models

logger = logging.getLogger(__name__)


async def get_completions(request: CompletionRequest) -> CompletionResponse:
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
            models = get_models()
            suggestions = await complete_with_transformers(
                input_text, 
                request.max_suggestions, 
                request.max_length,
                models['model'],
                models['tokenizer'],
                models['device']
            )
        
        logger.info(f"Generated {len(suggestions)} suggestions for: '{input_text[:50]}...' (using {'Gemini' if USE_GEMINI_COMPLETE else 'Transformers'})")
        
        return CompletionResponse(
            suggestions=suggestions,
            input_text=input_text
        )
        
    except Exception as e:
        logger.error(f"Error generating completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
