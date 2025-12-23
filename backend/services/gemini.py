"""
Gemini service for text completion and transliteration
"""
import re
import logging
from fastapi import HTTPException
from config import GEMINI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Global Gemini client
gemini_client = None


def initialize_gemini_client():
    """Initialize Gemini client with appropriate authentication"""
    global gemini_client
    
    if gemini_client is not None:
        return gemini_client
    
    # Support both API key and Vertex AI service account authentication
    if GOOGLE_APPLICATION_CREDENTIALS:
        import os
        if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
            logger.error(f"Service account file not found: {GOOGLE_APPLICATION_CREDENTIALS}")
            raise FileNotFoundError(f"Service account file not found: {GOOGLE_APPLICATION_CREDENTIALS}")
        # Set the environment variable for Google auth libraries
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
        logger.info(f"Using Vertex AI service account: {GOOGLE_APPLICATION_CREDENTIALS}")
        gemini_client = genai.GenerativeModel('gemini-2.0-flash')
    elif GEMINI_API_KEY:
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
    return gemini_client


async def complete_with_gemini(input_text: str, max_suggestions: int, max_length: int) -> list[str]:
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


async def transliterate_with_gemini(input_text: str, max_suggestions: int) -> list[str]:
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
                suggestions = []
        
        return suggestions[:max_suggestions]
    except Exception as e:
        logger.error(f"Error with Gemini transliteration: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini transliteration error: {str(e)}")
