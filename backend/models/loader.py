"""
Model loading and management for transformers models
"""
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MBartForConditionalGeneration, MBart50TokenizerFast
from config import USE_GEMINI_COMPLETE, USE_GEMINI_TRANSLITERATE, GOOGLE_APPLICATION_CREDENTIALS, GEMINI_API_KEY
from services.gemini import initialize_gemini_client
from services.speech import initialize_speech_client

logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None
device = None

# Transliteration model variables
transliteration_model = None
transliteration_tokenizer = None


async def load_models():
    """Load all models on startup"""
    global model, tokenizer, device
    global transliteration_model, transliteration_tokenizer
    
    try:
        # Initialize Gemini if needed
        if USE_GEMINI_COMPLETE or USE_GEMINI_TRANSLITERATE:
            initialize_gemini_client()
            if USE_GEMINI_COMPLETE:
                logger.info("Using Gemini Flash for /complete endpoint")
            if USE_GEMINI_TRANSLITERATE:
                logger.info("Using Gemini Flash for /transliterate endpoint")
        
        # Initialize Speech-to-Text client
        initialize_speech_client()
        
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


def get_models():
    """Get loaded models"""
    return {
        'model': model,
        'tokenizer': tokenizer,
        'device': device,
        'transliteration_model': transliteration_model,
        'transliteration_tokenizer': transliteration_tokenizer
    }
